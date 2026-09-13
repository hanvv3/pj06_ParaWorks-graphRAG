from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import UUID

from sqlalchemy import Connection, create_engine, func, select

from backend.app.agent_runtime.durable_file_authority import (
    DurableFileAuthority,
    DurableFileAuthorityError,
)
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyReviewAuthority,
    RagProviderSafetyReviewContext,
    RagProviderSafetyService,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
)
from backend.app.agent_runtime.rag_safety_identity import require_lower_hmac
from backend.app.core.config import Settings
from backend.app.models.rag_runtime import (
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)

MAX_REVIEW_ENVELOPE_BYTES = 32_768
_REVIEW_DOMAIN = b'paraworks:provider-safety-admin-review:v1\x00'
_REVIEW_SCHEMA = 'rag-provider-safety-admin-review:v1'
_OPERATIONS = frozenset(
    {
        'provider-safety-init',
        'provider-safety-bootstrap-recovery',
        'provider-safety-mark-rebind-required',
        'provider-safety-rebind',
        'provider-safety-reset',
        'provider-safety-supersede',
    }
)
_SIGNED_KEYS = frozenset(
    {
        'actor_subject_hmac',
        'expected_context',
        'historical_block_acknowledged',
        'implementation_plan_reference_hmac',
        'nonce',
        'operation',
        'review_authority_key_id',
        'schema_version',
        'successor',
        'target',
    }
)
_TARGET_KEYS = frozenset(
    {
        'database_identity_hmac',
        'designated_environment_id',
        'kind',
        'latch_identity_hmac',
    }
)
_CONTEXT_KEYS = frozenset(
    {
        'authority_uuid',
        'component',
        'current_family_identity',
        'current_state',
        'current_state_version',
        'designated_environment_id',
        'global_safety_generation',
        'has_historical_blocker',
    }
)
_RECOVERY_CONTEXT_KEYS = frozenset(
    {
        'authority_uuid',
        'designated_environment_id',
        'envelope_digest',
        'global_safety_generation',
        'implementation_plan_reference_hmac',
        'review_authority_key_id',
        'review_key_material_verifier',
        'reviewed_transition_reference_hmac',
    }
)
_SUCCESSOR_KEYS = frozenset(
    {
        'authorized_cost_policy_version',
        'authorized_model_config_snapshot_hmac',
        'authorized_model_config_version',
        'authorized_policy_snapshot_hmac',
        'authorized_token_estimator_version',
        'component',
        'fingerprint_key_material_verifier',
        'fingerprint_key_version',
        'model',
        'provider',
        'reasoning_or_config_identity',
    }
)

# Adding a successor is a reviewed code change. Empty means supersession is
# intentionally unavailable until a concrete provider snapshot is committed.
COMMITTED_PROVIDER_SAFETY_SUCCESSORS: tuple[
    AuthorizedProviderPolicySnapshot, ...
] = ()

# A production key is enabled only by committing its opaque verifier. The key
# bytes remain in the owner-controlled external file and are never committed.
COMMITTED_PROVIDER_SAFETY_REVIEW_KEYS: Mapping[str, str] = {}

_LEDGER_SCHEMA = 'rag-provider-safety-admin-review-ledger:v1'
_LEDGER_DOMAIN = b'paraworks:provider-safety-admin-review-ledger:v1\x00'
_REVIEW_KEY_VERIFIER_DOMAIN = (
    b'paraworks:provider-safety-admin-review-key-verifier:v1\x00'
)
_LEDGER_SIGNED_KEYS = frozenset(
    {
        'events',
        'implementation_plan_reference_hmac',
        'review_authority_key_id',
        'review_key_material_verifier',
        'schema_version',
        'target',
    }
)
_LEDGER_EVENT_KEYS = frozenset(
    {'envelope_digest', 'event', 'nonce', 'sequence'}
)


class ProviderSafetyReviewError(RagProviderSafetyError):
    pass


def review_key_material_verifier(review_secret: bytes) -> str:
    if type(review_secret) is not bytes or len(review_secret) < 32:
        raise ProviderSafetyReviewError('review authority is unavailable')
    return hmac.new(
        review_secret, _REVIEW_KEY_VERIFIER_DOMAIN, hashlib.sha256
    ).hexdigest()


def _identity_hmac(value: object, *, secret: bytes, domain: bytes) -> str:
    return hmac.new(secret, domain + canonical_json_bytes(value), hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderSafetyAdminTarget:
    kind: Literal['production', 'live_validation']
    database_url: str
    latch_path: Path
    designated_environment_id: str
    review_identity: dict[str, str]

    @classmethod
    def build(
        cls,
        *,
        kind: str,
        database_url: str,
        latch_path: str | Path,
        designated_environment_id: str,
        review_secret: bytes,
    ) -> ProviderSafetyAdminTarget:
        if kind not in {'production', 'live_validation'}:
            raise ProviderSafetyReviewError('provider safety target is invalid')
        if type(review_secret) is not bytes or len(review_secret) < 32:
            raise ProviderSafetyReviewError('provider safety review key is invalid')
        if type(database_url) is not str or not database_url.strip():
            raise ProviderSafetyReviewError('provider safety database target is invalid')
        path = DurableFileAuthority.validate_configured_path(latch_path)
        if (
            type(designated_environment_id) is not str
            or designated_environment_id != designated_environment_id.strip()
            or not designated_environment_id
        ):
            raise ProviderSafetyReviewError('provider safety environment is invalid')
        target = {
            'database_identity_hmac': _identity_hmac(
                {'database_url': database_url},
                secret=review_secret,
                domain=b'paraworks:provider-safety-admin-db-target:v1\x00',
            ),
            'designated_environment_id': designated_environment_id,
            'kind': kind,
            'latch_identity_hmac': _identity_hmac(
                {'latch_path': str(path)},
                secret=review_secret,
                domain=b'paraworks:provider-safety-admin-latch-target:v1\x00',
            ),
        }
        return cls(
            kind=kind,  # type: ignore[arg-type]
            database_url=database_url,
            latch_path=path,
            designated_environment_id=designated_environment_id,
            review_identity=target,
        )


@dataclass(frozen=True, slots=True)
class VerifiedProviderSafetyReview:
    operation: str
    actor_subject_hmac: str
    reviewed_gate_reference_hmac: str
    expected_context: dict[str, object] | None
    successor: dict[str, object] | None
    historical_block_acknowledged: bool
    nonce: str
    envelope_digest: str


class _ProviderSafetyReviewLedger:
    """Runtime-key-authenticated, logically append-only review consumption log."""

    def __init__(
        self,
        *,
        path: Path,
        runtime_secret: bytes,
        target: Mapping[str, object],
        review_key_id: str,
        review_key_verifier: str,
        plan_hmac: str,
    ) -> None:
        self.path = DurableFileAuthority.validate_configured_path(path)
        self._runtime_secret = runtime_secret
        self._target = dict(target)
        self._review_key_id = review_key_id
        self._review_key_verifier = review_key_verifier
        self._plan_hmac = plan_hmac

    def _signature(self, payload: Mapping[str, object]) -> str:
        return hmac.new(
            self._runtime_secret,
            _LEDGER_DOMAIN + canonical_json_bytes(dict(payload)),
            hashlib.sha256,
        ).hexdigest()

    def _wrap(self, payload: dict[str, object]) -> dict[str, object]:
        return {'hmac_sha256': self._signature(payload), 'signed_payload': payload}

    def _new_payload(self) -> dict[str, object]:
        return {
            'events': [],
            'implementation_plan_reference_hmac': self._plan_hmac,
            'review_authority_key_id': self._review_key_id,
            'review_key_material_verifier': self._review_key_verifier,
            'schema_version': _LEDGER_SCHEMA,
            'target': self._target,
        }

    def _validate(self, value: dict[str, object]) -> dict[str, object]:
        if set(value) != {'hmac_sha256', 'signed_payload'}:
            raise ProviderSafetyReviewError('review ledger is inconsistent')
        payload = value.get('signed_payload')
        signature = value.get('hmac_sha256')
        if (
            type(payload) is not dict
            or set(payload) != _LEDGER_SIGNED_KEYS
            or type(signature) is not str
            or not hmac.compare_digest(signature, self._signature(payload))
            or payload['schema_version'] != _LEDGER_SCHEMA
            or payload['target'] != self._target
            or payload['review_authority_key_id'] != self._review_key_id
            or payload['review_key_material_verifier'] != self._review_key_verifier
            or payload['implementation_plan_reference_hmac'] != self._plan_hmac
        ):
            raise ProviderSafetyReviewError('review authority pin differs')
        events = payload['events']
        if type(events) is not list:
            raise ProviderSafetyReviewError('review ledger is inconsistent')
        reservations: dict[str, str] = {}
        consumed: set[str] = set()
        for sequence, event in enumerate(events):
            if (
                type(event) is not dict
                or set(event) != _LEDGER_EVENT_KEYS
                or event['sequence'] != sequence
                or event['event'] not in {'reserved', 'consumed'}
            ):
                raise ProviderSafetyReviewError('review ledger is inconsistent')
            try:
                nonce = UUID(event['nonce'])  # type: ignore[arg-type]
                require_lower_hmac(event['envelope_digest'])
            except (TypeError, ValueError, AttributeError) as exc:
                raise ProviderSafetyReviewError(
                    'review ledger is inconsistent'
                ) from exc
            nonce_text = str(nonce)
            if nonce.int == 0 or event['nonce'] != nonce_text:
                raise ProviderSafetyReviewError('review ledger is inconsistent')
            digest = event['envelope_digest']
            if event['event'] == 'reserved':
                if nonce_text in reservations:
                    raise ProviderSafetyReviewError('review ledger is inconsistent')
                reservations[nonce_text] = digest
            elif (
                reservations.get(nonce_text) != digest or nonce_text in consumed
            ):
                raise ProviderSafetyReviewError('review ledger is inconsistent')
            else:
                consumed.add(nonce_text)
        return payload

    @staticmethod
    def _event(
        *, sequence: int, event: str, reviewed: VerifiedProviderSafetyReview
    ) -> dict[str, object]:
        return {
            'envelope_digest': reviewed.envelope_digest,
            'event': event,
            'nonce': reviewed.nonce,
            'sequence': sequence,
        }

    def assert_pin(self) -> None:
        if not self.path.exists():
            raise ProviderSafetyReviewError('review authority pin is unavailable')
        try:
            value = DurableFileAuthority.open_runtime(self.path).read()
            self._validate(value)
        except ProviderSafetyReviewError:
            raise
        except DurableFileAuthorityError as exc:
            raise ProviderSafetyReviewError(
                'review authority pin is unavailable'
            ) from exc

    def reserve(self, reviewed: VerifiedProviderSafetyReview) -> None:
        def reserve_in(value: dict[str, object]) -> dict[str, object]:
            payload = self._validate(value)
            events = list(payload['events'])
            matching = [event for event in events if event['nonce'] == reviewed.nonce]
            if matching:
                if (
                    matching[0]['envelope_digest'] != reviewed.envelope_digest
                    or any(event['event'] == 'consumed' for event in matching)
                ):
                    raise ProviderSafetyReviewError('review nonce was already used')
                return value
            events.append(
                self._event(
                    sequence=len(events), event='reserved', reviewed=reviewed
                )
            )
            return self._wrap({**payload, 'events': events})

        try:
            if self.path.exists():
                DurableFileAuthority.open_runtime(self.path).update(reserve_in)
                return
            initializer = DurableFileAuthority(self.path)
            payload = self._new_payload()
            payload['events'] = [
                self._event(sequence=0, event='reserved', reviewed=reviewed)
            ]
            initializer.write(self._wrap(payload))
        except ProviderSafetyReviewError:
            raise
        except DurableFileAuthorityError as exc:
            raise ProviderSafetyReviewError('review ledger is unavailable') from exc

    def consume(self, reviewed: VerifiedProviderSafetyReview) -> None:
        def consume_in(value: dict[str, object]) -> dict[str, object]:
            payload = self._validate(value)
            events = list(payload['events'])
            matching = [event for event in events if event['nonce'] == reviewed.nonce]
            if (
                len(matching) != 1
                or matching[0]['event'] != 'reserved'
                or matching[0]['envelope_digest'] != reviewed.envelope_digest
            ):
                raise ProviderSafetyReviewError('review nonce was already used')
            events.append(
                self._event(
                    sequence=len(events), event='consumed', reviewed=reviewed
                )
            )
            return self._wrap({**payload, 'events': events})

        try:
            DurableFileAuthority.open_runtime(self.path).update(consume_in)
        except ProviderSafetyReviewError:
            raise
        except DurableFileAuthorityError as exc:
            raise ProviderSafetyReviewError('review ledger is unavailable') from exc


def _read_review(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_REVIEW_ENVELOPE_BYTES:
        raise ProviderSafetyReviewError('review envelope is invalid')
    try:
        value = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderSafetyReviewError('review envelope is invalid') from exc
    if (
        type(value) is not dict
        or set(value) != {'hmac_sha256', 'signed_payload'}
        or canonical_json_bytes(value) != raw
        or type(value.get('signed_payload')) is not dict
        or set(value['signed_payload']) != _SIGNED_KEYS
        or type(value.get('hmac_sha256')) is not str
    ):
        raise ProviderSafetyReviewError('review envelope is not exact canonical JSON')
    return value


def _successor_registry_key(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        b'paraworks:provider-safety-successor-registry:v1\x00'
        + canonical_json_bytes(dict(value))
    ).hexdigest()


def successor_registry_from_snapshots(
    snapshots: tuple[AuthorizedProviderPolicySnapshot, ...],
) -> dict[str, AuthorizedProviderPolicySnapshot]:
    """Build the exact committed-code allowlist; it never accepts caller strings."""
    registry: dict[str, AuthorizedProviderPolicySnapshot] = {}
    for snapshot in snapshots:
        payload = _snapshot_payload(snapshot)
        key = _successor_registry_key(payload)
        if key in registry:
            raise ValueError('provider successor registry is duplicated')
        registry[key] = snapshot
    return registry


def verify_review_envelope(
    raw: bytes,
    *,
    expected_operation: str,
    expected_target: ProviderSafetyAdminTarget,
    expected_context: Mapping[str, object] | None,
    expected_successor: Mapping[str, object] | None,
    review_secret: bytes,
    review_key_id: str,
    implementation_plan_reference_hmac: str,
    successor_registry: Mapping[str, AuthorizedProviderPolicySnapshot],
) -> VerifiedProviderSafetyReview:
    envelope = _read_review(raw)
    payload = envelope['signed_payload']
    signature = envelope['hmac_sha256']
    if type(review_secret) is not bytes or len(review_secret) < 32:
        raise ProviderSafetyReviewError('review authority is unavailable')
    expected_signature = hmac.new(
        review_secret,
        _REVIEW_DOMAIN + canonical_json_bytes(payload),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ProviderSafetyReviewError('review envelope signature is invalid')
    if (
        expected_operation not in _OPERATIONS
        or payload['schema_version'] != _REVIEW_SCHEMA
        or payload['operation'] != expected_operation
        or payload['review_authority_key_id'] != review_key_id
    ):
        raise ProviderSafetyReviewError('review envelope operation is invalid')
    try:
        nonce = UUID(payload['nonce'])  # type: ignore[arg-type]
        require_lower_hmac(payload['actor_subject_hmac'])
        require_lower_hmac(payload['implementation_plan_reference_hmac'])
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProviderSafetyReviewError('review envelope identity is invalid') from exc
    if nonce.int == 0 or payload['nonce'] != str(nonce):
        raise ProviderSafetyReviewError('review envelope nonce is invalid')
    if not hmac.compare_digest(
        payload['implementation_plan_reference_hmac'],
        implementation_plan_reference_hmac,
    ):
        raise ProviderSafetyReviewError('implementation plan reference differs')
    target = payload['target']
    if (
        type(target) is not dict
        or set(target) != _TARGET_KEYS
        or target != expected_target.review_identity
    ):
        raise ProviderSafetyReviewError('review envelope target differs')
    successor = payload['successor']
    expected_successor_value = (
        None if expected_successor is None else dict(expected_successor)
    )
    if successor != expected_successor_value:
        raise ProviderSafetyReviewError('review envelope successor differs')
    if successor is not None and (
        type(successor) is not dict or set(successor) != _SUCCESSOR_KEYS
    ):
        raise ProviderSafetyReviewError('review envelope successor is invalid')
    successor_operations = {'provider-safety-rebind', 'provider-safety-supersede'}
    if (expected_operation in successor_operations) != (successor is not None):
        raise ProviderSafetyReviewError('review envelope successor is invalid')
    context = payload['expected_context']
    expected_context_value = None if expected_context is None else dict(expected_context)
    if context != expected_context_value:
        raise ProviderSafetyReviewError('review envelope context differs')
    if expected_operation == 'provider-safety-init':
        valid_context = context is None
    elif expected_operation == 'provider-safety-bootstrap-recovery':
        valid_context = type(context) is dict and set(context) == _RECOVERY_CONTEXT_KEYS
    else:
        valid_context = type(context) is dict and set(context) == _CONTEXT_KEYS
    if not valid_context:
        raise ProviderSafetyReviewError('review envelope context is invalid')
    if type(payload['historical_block_acknowledged']) is not bool:
        raise ProviderSafetyReviewError('review acknowledgement is invalid')
    if (
        expected_operation
        in {
            'provider-safety-init',
            'provider-safety-bootstrap-recovery',
            'provider-safety-mark-rebind-required',
        }
        and payload['historical_block_acknowledged']
    ):
        raise ProviderSafetyReviewError('review acknowledgement is inapplicable')
    if expected_operation == 'provider-safety-supersede':
        if successor is None:
            raise ProviderSafetyReviewError('provider successor is required')
        registry_snapshot = successor_registry.get(_successor_registry_key(successor))
        if registry_snapshot is None or _snapshot_payload(registry_snapshot) != successor:
            raise ProviderSafetyReviewError('provider successor is absent from registry')
    return VerifiedProviderSafetyReview(
        operation=expected_operation,
        actor_subject_hmac=payload['actor_subject_hmac'],  # type: ignore[arg-type]
        reviewed_gate_reference_hmac=payload[
            'implementation_plan_reference_hmac'
        ],  # type: ignore[arg-type]
        expected_context=context,
        successor=successor,
        historical_block_acknowledged=payload['historical_block_acknowledged'],
        nonce=str(nonce),
        envelope_digest=hashlib.sha256(raw).hexdigest(),
    )


def _snapshot_payload(snapshot: AuthorizedProviderPolicySnapshot) -> dict[str, object]:
    return RagProviderSafetyReviewAuthority._successor(snapshot)


def _context_payload(context: RagProviderSafetyReviewContext) -> dict[str, object]:
    return RagProviderSafetyReviewAuthority._context(context)


@dataclass(frozen=True, slots=True)
class ProviderSafetyAdminStatus:
    authority_present: bool
    global_safety_generation: int | None
    ready_family_count: int
    blocked_family_count: int
    rebind_required_family_count: int


@dataclass(frozen=True, slots=True)
class ProviderSafetyAdminResult:
    operation: str
    global_safety_generation: int
    ready_family_count: int


class RagProviderSafetyAdminService:
    def __init__(
        self,
        *,
        target: ProviderSafetyAdminTarget,
        connection_factory: Callable[[], Connection],
        provider_safety: RagProviderSafetyService,
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
        runtime_identity_secret: bytes,
        review_secret: bytes,
        review_key_id: str,
        implementation_plan_reference_hmac: str,
        successor_registry: Mapping[str, AuthorizedProviderPolicySnapshot],
        review_key_registry: Mapping[str, str] | None = None,
    ) -> None:
        require_lower_hmac(implementation_plan_reference_hmac)
        if (
            type(runtime_identity_secret) is not bytes
            or type(review_secret) is not bytes
            or not runtime_identity_secret
            or not review_secret
            or hmac.compare_digest(runtime_identity_secret, review_secret)
        ):
            raise ProviderSafetyReviewError(
                'review authority key must be distinct from runtime authority key'
            )
        if (
            type(review_key_id) is not str
            or not review_key_id
            or review_key_id != review_key_id.strip()
        ):
            raise ProviderSafetyReviewError('review authority key id is invalid')
        key_verifier = review_key_material_verifier(review_secret)
        if review_key_registry is not None:
            committed = review_key_registry.get(review_key_id)
            if committed is None or not hmac.compare_digest(committed, key_verifier):
                raise ProviderSafetyReviewError(
                    'review authority key is absent from committed registry'
                )
        self._committed_review_key = review_key_registry is not None
        self.target = target
        self.connection_factory = connection_factory
        self.provider_safety = provider_safety
        self.snapshots = snapshots
        self._runtime_secret = runtime_identity_secret
        self._review_secret = review_secret
        self._review_key_id = review_key_id
        self._review_key_verifier = key_verifier
        self._plan_hmac = implementation_plan_reference_hmac
        self._successor_registry = successor_registry
        self._ledger = _ProviderSafetyReviewLedger(
            path=Path(str(target.latch_path) + '.admin-review-ledger.json'),
            runtime_secret=runtime_identity_secret,
            target=target.review_identity,
            review_key_id=review_key_id,
            review_key_verifier=key_verifier,
            plan_hmac=implementation_plan_reference_hmac,
        )

    def _verify(
        self,
        raw: bytes,
        *,
        operation: str,
        context: Mapping[str, object] | None,
        successor: Mapping[str, object] | None,
    ) -> VerifiedProviderSafetyReview:
        return verify_review_envelope(
            raw,
            expected_operation=operation,
            expected_target=self.target,
            expected_context=context,
            expected_successor=successor,
            review_secret=self._review_secret,
            review_key_id=self._review_key_id,
            implementation_plan_reference_hmac=self._plan_hmac,
            successor_registry=self._successor_registry,
        )

    def _reserve(self, reviewed: VerifiedProviderSafetyReview) -> None:
        if (
            not self._ledger.path.exists()
            and self.target.latch_path.exists()
            and not self._committed_review_key
        ):
            raise ProviderSafetyReviewError('review authority pin is unavailable')
        self._ledger.reserve(reviewed)

    def status(self) -> ProviderSafetyAdminStatus:
        if not self.target.latch_path.exists():
            with self.connection_factory() as connection:
                counts = (
                    connection.scalar(
                        select(func.count()).select_from(RagProviderSafetyAuthority)
                    ),
                    connection.scalar(
                        select(func.count()).select_from(RagProviderReadiness)
                    ),
                    connection.scalar(
                        select(func.count()).select_from(RagProviderSafetyTransition)
                    ),
                )
            if counts != (0, 0, 0) or self._ledger.path.exists():
                raise RagProviderSafetyError(
                    'provider safety DB/latch authority is inconsistent'
                )
            return ProviderSafetyAdminStatus(False, None, 0, 0, 0)
        with self.connection_factory() as connection:
            contexts = tuple(
                self.provider_safety.review_context(connection, component)
                for component in ('query_embedding', 'answer_generation')
            )
        states = [item.state for item in contexts]
        return ProviderSafetyAdminStatus(
            authority_present=True,
            global_safety_generation=contexts[0].global_safety_generation,
            ready_family_count=states.count('ready'),
            blocked_family_count=sum(state.startswith('blocked_') for state in states),
            rebind_required_family_count=states.count('rebind_required'),
        )

    def _result(self, operation: str) -> ProviderSafetyAdminResult:
        status = self.status()
        if status.global_safety_generation is None:
            raise RagProviderSafetyError('provider safety authority is absent')
        return ProviderSafetyAdminResult(
            operation=operation,
            global_safety_generation=status.global_safety_generation,
            ready_family_count=status.ready_family_count,
        )

    def initialize(self, raw: bytes) -> ProviderSafetyAdminResult:
        reviewed = self._verify(
            raw, operation='provider-safety-init', context=None, successor=None
        )
        self._reserve(reviewed)
        with self.connection_factory() as connection:
            self.provider_safety.bootstrap(
                connection,
                self.snapshots,
                reviewed_transition_reference_hmac=reviewed.reviewed_gate_reference_hmac,
            )
        self._ledger.consume(reviewed)
        return self._result(reviewed.operation)

    def bootstrap_recovery_review_context(self) -> dict[str, object]:
        self._ledger.assert_pin()
        with self.connection_factory() as connection:
            context = self.provider_safety.bootstrap_recovery_context(
                connection, self.snapshots
            )
        if not hmac.compare_digest(
            context['reviewed_transition_reference_hmac'], self._plan_hmac
        ):
            raise ProviderSafetyReviewError(
                'partial bootstrap implementation plan reference differs'
            )
        return {
            **context,
            'implementation_plan_reference_hmac': self._plan_hmac,
            'review_authority_key_id': self._review_key_id,
            'review_key_material_verifier': self._review_key_verifier,
        }

    def recover_bootstrap(self, raw: bytes) -> ProviderSafetyAdminResult:
        envelope = _read_review(raw)
        context_value = envelope['signed_payload']['expected_context']
        if type(context_value) is not dict:
            raise ProviderSafetyReviewError('review envelope context is invalid')
        reviewed = self._verify(
            raw,
            operation='provider-safety-bootstrap-recovery',
            context=context_value,
            successor=None,
        )
        self._reserve(reviewed)
        if context_value != self.bootstrap_recovery_review_context():
            raise ProviderSafetyReviewError('review envelope context differs')
        with self.connection_factory() as connection:
            self.provider_safety.recover_partial_bootstrap(connection, self.snapshots)
        self._ledger.consume(reviewed)
        return self._result(reviewed.operation)

    def _current_review(
        self, raw: bytes, operation: str
    ) -> tuple[VerifiedProviderSafetyReview, RagProviderSafetyReviewContext]:
        envelope = _read_review(raw)
        context_value = envelope['signed_payload']['expected_context']
        if type(context_value) is not dict or set(context_value) != _CONTEXT_KEYS:
            raise ProviderSafetyReviewError('review envelope context is invalid')
        component = context_value.get('component')
        if component not in {'query_embedding', 'answer_generation'}:
            raise ProviderSafetyReviewError('review envelope component is invalid')
        successor = envelope['signed_payload']['successor']
        expected_successor = (
            successor
            if operation in {'provider-safety-rebind', 'provider-safety-supersede'}
            and isinstance(successor, dict)
            else None
        )
        reviewed = self._verify(
            raw,
            operation=operation,
            context=context_value,
            successor=expected_successor,
        )
        self._reserve(reviewed)
        with self.connection_factory() as connection:
            context = self.provider_safety.review_context(connection, component)
        if context_value != _context_payload(context):
            raise ProviderSafetyReviewError('review envelope context differs')
        return reviewed, context

    def _internal_command(
        self,
        reviewed: VerifiedProviderSafetyReview,
        context: RagProviderSafetyReviewContext,
        *,
        operation: Literal['mark_rebind_required', 'rebind', 'reset', 'supersession'],
        successor: AuthorizedProviderPolicySnapshot | None,
    ):
        return RagProviderSafetyReviewAuthority(
            identity_secret=self._runtime_secret
        ).issue(
            context,
            operation=operation,
            successor=successor,
            actor_subject_hmac=reviewed.actor_subject_hmac,
            reviewed_gate_reference_hmac=reviewed.reviewed_gate_reference_hmac,
            historical_block_acknowledged=reviewed.historical_block_acknowledged,
        )

    def mark_rebind_required(self, raw: bytes) -> ProviderSafetyAdminResult:
        reviewed, context = self._current_review(
            raw, 'provider-safety-mark-rebind-required'
        )
        command = self._internal_command(
            reviewed, context, operation='mark_rebind_required', successor=None
        )
        with self.connection_factory() as connection:
            self.provider_safety.mark_rebind_required(connection, command)
        self._ledger.consume(reviewed)
        return self._result(reviewed.operation)

    def rebind(self, raw: bytes) -> ProviderSafetyAdminResult:
        reviewed, context = self._current_review(raw, 'provider-safety-rebind')
        successor = _payload_snapshot(reviewed.successor)
        command = self._internal_command(
            reviewed, context, operation='rebind', successor=successor
        )
        with self.connection_factory() as connection:
            self.provider_safety.reviewed_rebind(connection, command, successor)
        self._ledger.consume(reviewed)
        return self._result(reviewed.operation)

    def reset(self, raw: bytes) -> ProviderSafetyAdminResult:
        reviewed, context = self._current_review(raw, 'provider-safety-reset')
        command = self._internal_command(
            reviewed, context, operation='reset', successor=None
        )
        with self.connection_factory() as connection:
            self.provider_safety.reviewed_reset(connection, command)
        self._ledger.consume(reviewed)
        return self._result(reviewed.operation)

    def supersede(self, raw: bytes) -> ProviderSafetyAdminResult:
        reviewed, context = self._current_review(raw, 'provider-safety-supersede')
        successor = _payload_snapshot(reviewed.successor)
        command = self._internal_command(
            reviewed, context, operation='supersession', successor=successor
        )
        with self.connection_factory() as connection:
            self.provider_safety.reviewed_supersession(connection, command, successor)
        self._ledger.consume(reviewed)
        return self._result(reviewed.operation)


def _configured_target(
    settings: Settings, *, review_secret: bytes
) -> ProviderSafetyAdminTarget:
    from sqlalchemy.engine import make_url

    def database_identity(value: str) -> tuple[str, str | None, int | None, str | None]:
        parsed = make_url(value)
        host = parsed.host.rstrip('.').casefold() if parsed.host else None
        if host is not None:
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                try:
                    host = host.encode('idna').decode('ascii')
                except UnicodeError as exc:
                    raise ProviderSafetyReviewError(
                        'provider safety database target is ambiguous'
                    ) from exc
                if host == 'localhost':
                    host = 'loopback'
            else:
                host = 'loopback' if address.is_loopback else address.compressed
        port = parsed.port
        if parsed.get_backend_name() == 'postgresql' and port is None:
            port = 5432
        return parsed.get_backend_name(), host, port, parsed.database

    live_database = settings.paraworks_rag_live_validation_database_url
    live_latch = settings.paraworks_rag_live_validation_provider_safety_latch_path
    if live_database and live_latch:
        production_latch = Path(settings.paraworks_provider_safety_latch_path)
        production_database_identity = database_identity(
            settings.resolved_database_url()
        )
        live_database_identity = database_identity(live_database)
        same_database_on_ambiguous_hosts = (
            production_database_identity[0] == live_database_identity[0]
            and production_database_identity[2:] == live_database_identity[2:]
            and production_database_identity[1] != live_database_identity[1]
        )
        if (
            live_database_identity == production_database_identity
            or same_database_on_ambiguous_hosts
            or os.path.normcase(str(Path(live_latch).absolute()))
            == os.path.normcase(str(production_latch.absolute()))
            or settings.paraworks_rag_live_validation_environment_id
            == settings.paraworks_env
        ):
            raise ProviderSafetyReviewError(
                'production and live-validation targets must be separate'
            )
    if settings.paraworks_provider_safety_admin_target == 'production':
        database_url = settings.resolved_database_url()
        latch_path = settings.paraworks_provider_safety_latch_path
        environment = settings.paraworks_env
        kind = 'production'
    else:
        database_url = live_database
        latch_path = live_latch
        environment = settings.paraworks_rag_live_validation_environment_id
        kind = 'live_validation'
    if not database_url or not latch_path:
        raise ProviderSafetyReviewError('provider safety target is incomplete')
    return ProviderSafetyAdminTarget.build(
        kind=kind,
        database_url=database_url,
        latch_path=latch_path,
        designated_environment_id=environment,
        review_secret=review_secret,
    )


def _load_review_secret(path_value: str | None) -> bytes:
    if not path_value:
        raise ProviderSafetyReviewError('review authority key file is unavailable')
    path = Path(path_value)
    if not path.is_absolute():
        raise ProviderSafetyReviewError('review authority key path is invalid')
    try:
        DurableFileAuthority.open_runtime(path)._validate_existing_regular(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_nlink != 1:
            raise ProviderSafetyReviewError('review authority key file is untrusted')
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ProviderSafetyReviewError('review authority key file changed')
            value = os.read(descriptor, 4097)
        finally:
            os.close(descriptor)
    except ProviderSafetyReviewError:
        raise
    except OSError as exc:
        raise ProviderSafetyReviewError(
            'review authority key file is unavailable'
        ) from exc
    if len(value) < 32 or len(value) > 4096:
        raise ProviderSafetyReviewError('review authority key file is invalid')
    return value


@dataclass(slots=True)
class _DefaultAdminResources:
    service: RagProviderSafetyAdminService
    engine: object

    def close(self) -> None:
        self.engine.dispose()


def _build_default_admin_resources(settings: Settings) -> _DefaultAdminResources:
    from sqlalchemy.engine import make_url

    from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
    from backend.app.agent_runtime.rag_advisory_locks import (
        RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
        load_registered_advisory_capability,
    )
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agent_runtime.rag_v2_composition import (
        build_rag_provider_policy_snapshots,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
        build_answer_output_schema_hmac,
        build_answer_prompt_renderer_hmac,
    )

    plan_hmac = settings.paraworks_provider_safety_implementation_plan_reference_hmac
    if plan_hmac is None:
        raise ProviderSafetyReviewError('implementation plan reference is unavailable')
    require_lower_hmac(plan_hmac)
    review_secret = _load_review_secret(
        settings.paraworks_provider_safety_review_key_path
    )
    target = _configured_target(settings, review_secret=review_secret)
    if make_url(target.database_url).get_backend_name() != 'postgresql':
        raise ProviderSafetyReviewError('provider safety admin requires PostgreSQL')
    engine = create_engine(target.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            advisory = load_registered_advisory_capability(
                connection,
                RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                identity_namespace='static',
            )
        runtime_secret, _ = fingerprint_secret_bytes(settings)
        provider_safety = RagProviderSafetyService(
            latch_path=target.latch_path,
            identity_secret=runtime_secret,
            designated_environment_id=target.designated_environment_id,
            advisory_capability=advisory,
        )
        policy = RagCostPolicy(
            settings=settings,
            answer_output_schema_hmac=build_answer_output_schema_hmac(settings),
            answer_prompt_renderer_hmac=build_answer_prompt_renderer_hmac(settings),
        )
        snapshots = build_rag_provider_policy_snapshots(policy, settings)
        return _DefaultAdminResources(
            service=RagProviderSafetyAdminService(
                target=target,
                connection_factory=engine.connect,
                provider_safety=provider_safety,
                snapshots=snapshots,
                runtime_identity_secret=runtime_secret,
                review_secret=review_secret,
                review_key_id=settings.paraworks_provider_safety_review_key_id,
                implementation_plan_reference_hmac=plan_hmac,
                successor_registry=successor_registry_from_snapshots(
                    COMMITTED_PROVIDER_SAFETY_SUCCESSORS
                ),
                review_key_registry=COMMITTED_PROVIDER_SAFETY_REVIEW_KEYS,
            ),
            engine=engine,
        )
    except Exception:
        engine.dispose()
        raise


def _payload_snapshot(
    value: Mapping[str, object] | None,
) -> AuthorizedProviderPolicySnapshot:
    if value is None or set(value) != _SUCCESSOR_KEYS:
        raise ProviderSafetyReviewError('provider successor is invalid')
    try:
        return AuthorizedProviderPolicySnapshot(**dict(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProviderSafetyReviewError('provider successor is invalid') from exc


class _BoundedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ProviderSafetyReviewError('command arguments were refused')


def build_cli_parser() -> argparse.ArgumentParser:
    parser = _BoundedArgumentParser(prog='python -m backend.app.admin.rag_provider_safety')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('provider-safety-status')
    for operation in sorted(_OPERATIONS):
        commands.add_parser(operation)
    return parser


def _read_stdin_review() -> bytes:
    stream = getattr(sys.stdin, 'buffer', sys.stdin)
    if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
        raise ProviderSafetyReviewError('review envelope is required on stdin')
    raw = stream.read(MAX_REVIEW_ENVELOPE_BYTES + 1)
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    if not raw or len(raw) > MAX_REVIEW_ENVELOPE_BYTES:
        raise ProviderSafetyReviewError('review envelope is invalid')
    return raw


@dataclass(frozen=True, slots=True)
class _CliOutcome:
    payload: dict[str, object]
    exit_code: int


def _stdin_bytes(stdin: BinaryIO) -> bytes:
    raw = stdin.read(MAX_REVIEW_ENVELOPE_BYTES + 1)
    if type(raw) is not bytes or not raw or len(raw) > MAX_REVIEW_ENVELOPE_BYTES:
        raise ProviderSafetyReviewError('review envelope is invalid')
    return raw


def _run_cli(
    argv: list[str] | None,
    *,
    stdin: BinaryIO,
    service_factory: Callable[[], object],
) -> _CliOutcome:
    try:
        args = build_cli_parser().parse_args(argv)
    except Exception:
        return _CliOutcome({'code': 'command_refused', 'ok': False}, 2)
    try:
        service = service_factory()
        if args.command == 'provider-safety-status':
            status = service.status()
            return _CliOutcome(
                {
                    'authority_present': status.authority_present,
                    'blocked_family_count': status.blocked_family_count,
                    'global_safety_generation': status.global_safety_generation,
                    'ok': True,
                    'ready_family_count': status.ready_family_count,
                    'rebind_required_family_count': (
                        status.rebind_required_family_count
                    ),
                },
                0,
            )
        raw = _stdin_bytes(stdin)
        method_name = {
            'provider-safety-init': 'initialize',
            'provider-safety-bootstrap-recovery': 'recover_bootstrap',
            'provider-safety-mark-rebind-required': 'mark_rebind_required',
            'provider-safety-rebind': 'rebind',
            'provider-safety-reset': 'reset',
            'provider-safety-supersede': 'supersede',
        }[args.command]
        result = getattr(service, method_name)(raw)
        return _CliOutcome(
            {
                'global_safety_generation': result.global_safety_generation,
                'ok': True,
                'operation': result.operation,
                'ready_family_count': result.ready_family_count,
            },
            0,
        )
    except ProviderSafetyReviewError:
        return _CliOutcome({'code': 'review_refused', 'ok': False}, 2)
    except RagProviderSafetyError:
        return _CliOutcome({'code': 'authority_refused', 'ok': False}, 3)
    except Exception:
        return _CliOutcome({'code': 'operation_failed', 'ok': False}, 3)


def main(argv: list[str] | None = None) -> int:
    stream = getattr(sys.stdin, 'buffer', sys.stdin)
    resources: _DefaultAdminResources | None = None

    def service_factory() -> RagProviderSafetyAdminService:
        nonlocal resources
        resources = _build_default_admin_resources(Settings())
        return resources.service

    outcome = _run_cli(argv, stdin=stream, service_factory=service_factory)
    if resources is not None:
        try:
            resources.close()
        except Exception:
            outcome = _CliOutcome({'code': 'operation_failed', 'ok': False}, 3)
    print(json.dumps(outcome.payload, separators=(',', ':'), sort_keys=True))
    return outcome.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
