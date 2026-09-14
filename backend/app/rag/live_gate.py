"""Opaque execution authority shared by the live runner and quality evaluator.

This first Task25-B slice performs no dispatch and writes no release rows.  It
only turns the already issued Task24 approval/source pair into a one-use,
process-local capability while the existing release/provider barrier is held.
"""

from __future__ import annotations

import hmac
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from uuid import UUID
from weakref import WeakKeyDictionary

from sqlalchemy import select

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_safety_identity import require_lower_hmac
from backend.app.rag.release_review import (
    AuthorizedRagLiveGate,
    FrozenCorpusSnapshot,
    FrozenLiveManifestSnapshot,
    _exact_frozen_equal,
    _manifest_payload,
    require_approved_case_source,
    require_issued_live_authorization_identity,
)
from backend.app.rag.release_schema import (
    build_rag_release_metadata,
    release_tables,
)


class RagLiveGateCapabilityError(ValueError):
    """Bounded refusal for execution-capability issuance or consumption."""

    def __init__(self, code: str = 'execution_capability_invalid') -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class _ApprovedQualityAuthority:
    authorization: AuthorizedRagLiveGate
    manifest: FrozenLiveManifestSnapshot
    corpus: FrozenCorpusSnapshot
    release_generation: int


def _capability_boundary():
    capabilities = WeakKeyDictionary()
    issued_sources = WeakKeyDictionary()
    lock = RLock()

    class ApprovedRagExecutionCapability:
        """One-use process identity; it deliberately exposes no payload."""

        __slots__ = ('__weakref__',)

        def __init__(self):
            raise TypeError('execution capabilities require approved issuance')

        def __copy__(self):
            raise TypeError('execution capabilities cannot be copied')

        def __deepcopy__(self, memo):
            raise TypeError('execution capabilities cannot be copied')

        def __reduce__(self):
            raise TypeError('execution capabilities cannot be serialized')

        def __reduce_ex__(self, protocol):
            raise TypeError('execution capabilities cannot be serialized')

        def __getstate__(self):
            raise TypeError('execution capabilities cannot be serialized')

        def __repr__(self):
            return '<ApprovedRagExecutionCapability opaque>'

    def refuse(code='execution_capability_invalid'):
        raise RagLiveGateCapabilityError(code)

    def issue(
        connection,
        *,
        authorization,
        authorization_row,
        source_binding,
        identity_secret,
        barrier_guard,
    ):
        with lock:
            try:
                if (
                    type(authorization) is not AuthorizedRagLiveGate
                    or type(authorization.ledger_uuid) is not UUID
                    or authorization.ledger_uuid.int == 0
                    or type(authorization.ledger_epoch) is not int
                    or authorization.ledger_epoch < 1
                    or type(identity_secret) is not bytes
                    or len(identity_secret) < 32
                    or type(authorization_row) is not dict
                    or any(type(key) is not str for key in authorization_row)
                    or barrier_guard is None
                    or connection.engine is None
                ):
                    refuse()
                require_issued_live_authorization_identity(
                    source_binding, authorization
                )
                if source_binding in issued_sources:
                    refuse('execution_capability_already_issued')

                authorization_snapshot = deepcopy(authorization)
                executable_bytes = canonical_json_bytes(
                    _manifest_payload(authorization.manifest.executable)
                )
                row_snapshot = deepcopy(authorization_row)
                if (
                    type(row_snapshot.get('ledger_uuid')) is not str
                    or row_snapshot['ledger_uuid'] != str(authorization.ledger_uuid)
                    or type(row_snapshot.get('ledger_epoch')) is not int
                    or row_snapshot['ledger_epoch'] != authorization.ledger_epoch
                    or row_snapshot.get('approval_id_hmac')
                    != authorization.approval_id_hmac
                    or row_snapshot.get('approval_hmac') != authorization.approval_hmac
                ):
                    refuse()

                # This is the existing Task24 source/corpus/provider/release
                # verifier.  It also authenticates the supplied barrier guard.
                require_approved_case_source(
                    connection,
                    manifest=authorization.manifest.executable,
                    source_binding=source_binding,
                    authorization=row_snapshot,
                    identity_secret=identity_secret,
                    barrier_guard=barrier_guard,
                )

                tables = release_tables(build_rag_release_metadata())
                rows = (
                    connection.execute(
                        select(tables.ledgers)
                        .where(
                            tables.ledgers.c.ledger_uuid
                            == str(authorization.ledger_uuid),
                            tables.ledgers.c.ledger_epoch == authorization.ledger_epoch,
                        )
                        .with_for_update()
                    )
                    .mappings()
                    .all()
                )
                if len(rows) != 1:
                    refuse()
                ledger = dict(rows[0])
                generation = ledger.get('generation')
                transition_digest = ledger.get('last_transition_digest')
                if (
                    type(generation) is not int
                    or generation <= authorization.approval_base_generation
                    or type(transition_digest) is not str
                ):
                    refuse()
                require_lower_hmac(transition_digest)
                if (
                    not _exact_frozen_equal(authorization, authorization_snapshot)
                    or canonical_json_bytes(
                        _manifest_payload(authorization.manifest.executable)
                    )
                    != executable_bytes
                    or row_snapshot != authorization_row
                ):
                    refuse()

                capability = object.__new__(ApprovedRagExecutionCapability)
                capabilities[capability] = {
                    'used': False,
                    'identity_secret': bytes(identity_secret),
                    'authorization_object': authorization,
                    'authorization': authorization_snapshot,
                    'source_binding': source_binding,
                    'ledger_uuid': authorization.ledger_uuid,
                    'ledger_epoch': authorization.ledger_epoch,
                    'release_generation': generation,
                    'last_transition_digest': transition_digest,
                    'executable_bytes': executable_bytes,
                    'fixture_binding': (
                        authorization.manifest.fixture_manifest_path,
                        authorization.manifest.fixture_manifest_sha256,
                        authorization.manifest.fixture_manifest_hmac,
                    ),
                    'manifest': deepcopy(authorization.manifest),
                    'corpus': deepcopy(authorization.corpus),
                    'provider_binding': (
                        authorization.approved_provider_safety_snapshot_hmac,
                        row_snapshot.get('provider_safety_envelope_digest'),
                    ),
                }
                issued_sources[source_binding] = capability
                return capability
            except RagLiveGateCapabilityError:
                raise
            except Exception:
                refuse()

    def consume(capability, *, identity_secret):
        with lock:
            if (
                type(capability) is not ApprovedRagExecutionCapability
                or capability not in capabilities
            ):
                refuse()
            state = capabilities[capability]
            if state['used']:
                refuse()
            # The attempt itself consumes the authority.  Malformed quality
            # inputs cannot be fixed and replayed after the approved run.
            state['used'] = True
            try:
                authorization = state['authorization_object']
                snapshot = state['authorization']
                if (
                    type(identity_secret) is not bytes
                    or not hmac.compare_digest(
                        identity_secret, state['identity_secret']
                    )
                    or type(authorization) is not AuthorizedRagLiveGate
                    or not _exact_frozen_equal(authorization, snapshot)
                    or type(authorization.ledger_uuid) is not UUID
                    or authorization.ledger_uuid != state['ledger_uuid']
                    or authorization.ledger_epoch != state['ledger_epoch']
                    or canonical_json_bytes(
                        _manifest_payload(authorization.manifest.executable)
                    )
                    != state['executable_bytes']
                    or not _exact_frozen_equal(
                        authorization.manifest, state['manifest']
                    )
                    or not _exact_frozen_equal(authorization.corpus, state['corpus'])
                    or (
                        authorization.manifest.fixture_manifest_path,
                        authorization.manifest.fixture_manifest_sha256,
                        authorization.manifest.fixture_manifest_hmac,
                    )
                    != state['fixture_binding']
                    or authorization.approved_provider_safety_snapshot_hmac
                    != state['provider_binding'][0]
                    or type(state['release_generation']) is not int
                    or state['release_generation']
                    <= authorization.approval_base_generation
                ):
                    refuse()
                return _ApprovedQualityAuthority(
                    authorization=deepcopy(snapshot),
                    manifest=deepcopy(state['manifest']),
                    corpus=deepcopy(state['corpus']),
                    release_generation=state['release_generation'],
                )
            except RagLiveGateCapabilityError:
                raise
            except Exception:
                refuse()

    return ApprovedRagExecutionCapability, issue, consume


(
    ApprovedRagExecutionCapability,
    issue_approved_execution_capability,
    _consume_approved_execution_capability,
) = _capability_boundary()


__all__ = [
    'ApprovedRagExecutionCapability',
    'RagLiveGateCapabilityError',
    'issue_approved_execution_capability',
]
