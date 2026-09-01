from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import Connection, func, insert, select, update

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    RagLockOrderCapability,
    RagLockOrderCoordinator,
    RegisteredAdvisoryLock,
    acquire_advisory_lock,
    release_advisory_lock,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    RagProviderSafetyBinding,
)
from backend.app.agent_runtime.rag_safety_identity import (
    provider_safety_snapshot_identity,
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import (
    AgentRunCostComponent,
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)
from backend.app.rag.retrieval import RagPaidComponent

ProviderSafetyState = Literal[
    'ready', 'rebind_required', 'blocked_overrun', 'blocked_remediation'
]

_COMPONENTS = ('query_embedding', 'answer_generation')
_STATES = {'ready', 'rebind_required', 'blocked_overrun', 'blocked_remediation'}
_BLOCKER_CATEGORIES = {
    'provider_usage_overrun',
    'provider_response_identity_invalid',
    'provider_embedding_payload_invalid',
    'provider_safety_unavailable',
}
_BODY_KEYS = {
    'active_family_by_component',
    'authority_uuid',
    'designated_environment_id',
    'family_records',
    'global_safety_generation',
    'latch_schema_version',
}
_FAMILY_IDENTITY_KEYS = {
    'component',
    'model',
    'provider',
    'reasoning_or_config_identity',
}
_FAMILY_RECORD_KEYS = _FAMILY_IDENTITY_KEYS | {
    'authorized_policy_snapshot_hmac',
    'family_safety_generation',
    'first_blocker_agent_run_hmac',
    'first_blocker_category',
    'first_blocker_observed_at',
    'reviewed_transition_reference_hmac',
    'state',
    'state_version',
}
_BOOTSTRAP_FAMILIES = {
    'query_embedding': {
        'provider': 'openai',
        'model': 'text-embedding-3-small',
        'reasoning_or_config_identity': 'dimensions:1536',
        'authorized_model_config_version': 'rag-query-embedding-config:v1',
        'authorized_cost_policy_version': 'rag-query-embedding-cost:v1',
        'authorized_token_estimator_version': (
            'openai-cl100k-text-embedding-3-small:v1'
        ),
    },
    'answer_generation': {
        'provider': 'openai',
        'model': 'gpt-5.4-mini-2026-03-17',
        'reasoning_or_config_identity': 'reasoning:none',
        'authorized_model_config_version': 'rag-answer-model-config:v1',
        'authorized_cost_policy_version': 'rag-answer-cost:v1',
        'authorized_token_estimator_version': 'openai-o200k-rag-answer:v1',
    },
}


class RagProviderSafetyError(RuntimeError):
    pass


_REVIEW_CAPABILITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class RagProviderSafetyReviewContext:
    authority_uuid: str
    designated_environment_id: str
    component: RagPaidComponent
    family_identity: tuple[str, str, str, str]
    state: str
    state_version: int
    global_safety_generation: int
    has_historical_blocker: bool
    _seal: object = field(repr=False, compare=False)


class ReviewedProviderSafetyCommand:
    __slots__ = ('_payload', '_hmac', '_seal', '_consumed')

    def __init__(self, *, payload: dict[str, object], command_hmac: str) -> None:
        self._payload = payload
        self._hmac = command_hmac
        self._seal = _REVIEW_CAPABILITY_SEAL
        self._consumed = False

    def _consume(
        self,
        *,
        secret: bytes,
        expected_context: RagProviderSafetyReviewContext,
        expected_operation: str,
        expected_successor: AuthorizedProviderPolicySnapshot | None,
    ) -> dict[str, object]:
        if self._seal is not _REVIEW_CAPABILITY_SEAL:
            raise RagProviderSafetyError('reviewed command capability is invalid')
        if self._consumed:
            raise RagProviderSafetyError('reviewed command was already consumed')
        self._consumed = True
        expected_hmac = rag_identity_hmac(
            self._payload,
            secret=secret,
            schema_version='rag-provider-safety-reviewed-command:v1',
            policy_version='rag-provider-safety:v1',
        )
        successor = (
            None
            if expected_successor is None
            else RagProviderSafetyReviewAuthority._successor(expected_successor)
        )
        context_payload = RagProviderSafetyReviewAuthority._context(expected_context)
        if (
            not hmac.compare_digest(self._hmac, expected_hmac)
            or self._payload['context'] != context_payload
            or self._payload['operation'] != expected_operation
            or self._payload['successor'] != successor
        ):
            raise RagProviderSafetyError('reviewed command binding does not match')
        return dict(self._payload)

    def _bound_component(self) -> RagPaidComponent:
        if self._seal is not _REVIEW_CAPABILITY_SEAL:
            raise RagProviderSafetyError('reviewed command capability is invalid')
        if self._consumed:
            raise RagProviderSafetyError('reviewed command was already consumed')
        context = self._payload.get('context')
        if not isinstance(context, dict) or context.get('component') not in _COMPONENTS:
            raise RagProviderSafetyError('reviewed command capability is invalid')
        return context['component']  # type: ignore[return-value]


class RagProviderSafetyReviewAuthority:
    """Separate provider-free review authority that issues one-use commands."""

    def __init__(self, *, identity_secret: bytes) -> None:
        if type(identity_secret) is not bytes or not identity_secret:
            raise ValueError('provider safety review signer is required')
        self._secret = identity_secret

    @staticmethod
    def _context(context: RagProviderSafetyReviewContext) -> dict[str, object]:
        if (
            type(context) is not RagProviderSafetyReviewContext
            or context._seal is not _REVIEW_CAPABILITY_SEAL
        ):
            raise RagProviderSafetyError('provider safety review context is invalid')
        return {
            'authority_uuid': context.authority_uuid,
            'component': context.component,
            'current_family_identity': list(context.family_identity),
            'current_state': context.state,
            'current_state_version': context.state_version,
            'designated_environment_id': context.designated_environment_id,
            'global_safety_generation': context.global_safety_generation,
            'has_historical_blocker': context.has_historical_blocker,
        }

    @staticmethod
    def _successor(
        snapshot: AuthorizedProviderPolicySnapshot,
    ) -> dict[str, object]:
        return {
            'authorized_cost_policy_version': snapshot.authorized_cost_policy_version,
            'authorized_model_config_snapshot_hmac': (
                snapshot.authorized_model_config_snapshot_hmac
            ),
            'authorized_model_config_version': (
                snapshot.authorized_model_config_version
            ),
            'authorized_policy_snapshot_hmac': (
                snapshot.authorized_policy_snapshot_hmac
            ),
            'authorized_token_estimator_version': (
                snapshot.authorized_token_estimator_version
            ),
            'component': snapshot.component,
            'fingerprint_key_material_verifier': (
                snapshot.fingerprint_key_material_verifier
            ),
            'fingerprint_key_version': snapshot.fingerprint_key_version,
            'model': snapshot.model,
            'provider': snapshot.provider,
            'reasoning_or_config_identity': snapshot.reasoning_or_config_identity,
        }

    def issue(
        self,
        context: RagProviderSafetyReviewContext,
        *,
        operation: Literal[
            'mark_rebind_required', 'rebind', 'reset', 'supersession'
        ],
        successor: AuthorizedProviderPolicySnapshot | None,
        actor_subject_hmac: str,
        reviewed_gate_reference_hmac: str,
        historical_block_acknowledged: bool,
    ) -> ReviewedProviderSafetyCommand:
        if operation not in {
            'mark_rebind_required', 'rebind', 'reset', 'supersession'
        }:
            raise RagProviderSafetyError('reviewed command operation is invalid')
        require_lower_hmac(actor_subject_hmac)
        require_lower_hmac(reviewed_gate_reference_hmac)
        if type(historical_block_acknowledged) is not bool:
            raise RagProviderSafetyError('historical blocker acknowledgement is invalid')
        if (
            context.has_historical_blocker
            and operation in {'reset', 'supersession'}
            and not historical_block_acknowledged
        ):
            raise RagProviderSafetyError('historical blocker acknowledgement is required')
        if (operation in {'rebind', 'supersession'}) != (successor is not None):
            raise RagProviderSafetyError('reviewed command successor is invalid')
        payload = {
            'actor_subject_hmac': actor_subject_hmac,
            'context': self._context(context),
            'historical_block_acknowledged': historical_block_acknowledged,
            'operation': operation,
            'reviewed_gate_reference_hmac': reviewed_gate_reference_hmac,
            'successor': (
                None if successor is None else self._successor(successor)
            ),
        }
        return ReviewedProviderSafetyCommand(
            payload=payload,
            command_hmac=rag_identity_hmac(
                payload,
                secret=self._secret,
                schema_version='rag-provider-safety-reviewed-command:v1',
                policy_version='rag-provider-safety:v1',
            ),
        )


def _identity_tuple(value: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(value['component']),
        str(value['model']),
        str(value['provider']),
        str(value['reasoning_or_config_identity']),
    )


class RagProviderSafetyService:
    """External-first, exact-whole-set authority for D-managed paid families."""

    def __init__(
        self,
        *,
        latch_path: str | Path,
        identity_secret: bytes,
        designated_environment_id: str,
        advisory_capability: RegisteredAdvisoryLock | None = None,
    ) -> None:
        if type(identity_secret) is not bytes or not identity_secret:
            raise ValueError('provider safety signer is required')
        if (
            type(designated_environment_id) is not str
            or not designated_environment_id.strip()
            or designated_environment_id != designated_environment_id.strip()
        ):
            raise ValueError('provider safety environment is required')
        self._latch_path = DurableFileAuthority.validate_configured_path(latch_path)
        self._authority = DurableFileAuthority.open_runtime(self._latch_path)
        self._secret = identity_secret
        self._environment = designated_environment_id
        if advisory_capability is not None and not advisory_capability.matches(
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        ):
            raise ValueError('provider safety advisory capability identity is invalid')
        self._advisory_capability = advisory_capability

    @staticmethod
    def validate_disabled_path(path: str | Path) -> None:
        DurableFileAuthority.validate_configured_path(path)

    def _signature(self, signed_payload: object) -> str:
        return rag_identity_hmac(
            signed_payload,
            secret=self._secret,
            schema_version='rag-provider-safety-latch-envelope:v1',
            policy_version='rag-provider-safety-latch-body:v1',
        )

    @staticmethod
    def _file_digest_bytes(raw: bytes) -> str:
        return hashlib.sha256(
            b'paraworks:provider-safety-envelope-file:v1\x00' + raw
        ).hexdigest()

    @staticmethod
    def _file_digest(envelope: Mapping[str, object]) -> str:
        return RagProviderSafetyService._file_digest_bytes(
            canonical_json_bytes(dict(envelope))
        )

    def _wrap(
        self,
        body: dict[str, object],
        *,
        key_version: str,
        key_verifier: str,
    ) -> dict[str, object]:
        signed_payload = {
            'body': body,
            'fingerprint_key_material_verifier': key_verifier,
            'fingerprint_key_version': key_version,
        }
        return {
            'hmac_sha256': self._signature(signed_payload),
            'signed_payload': signed_payload,
        }

    @staticmethod
    def _identity(snapshot: AuthorizedProviderPolicySnapshot) -> dict[str, str]:
        return {
            'component': snapshot.component,
            'model': snapshot.model,
            'provider': snapshot.provider,
            'reasoning_or_config_identity': snapshot.reasoning_or_config_identity,
        }

    @classmethod
    def _record(
        cls,
        snapshot: AuthorizedProviderPolicySnapshot,
        *,
        family_generation: int,
        reviewed_reference: str | None,
    ) -> dict[str, object]:
        return {
            **cls._identity(snapshot),
            'authorized_policy_snapshot_hmac': (
                snapshot.authorized_policy_snapshot_hmac
            ),
            'family_safety_generation': family_generation,
            'first_blocker_agent_run_hmac': None,
            'first_blocker_category': None,
            'first_blocker_observed_at': None,
            'reviewed_transition_reference_hmac': reviewed_reference,
            'state': 'ready',
            'state_version': 1,
        }

    @staticmethod
    def _snapshot_values(
        snapshot: AuthorizedProviderPolicySnapshot,
    ) -> dict[str, object]:
        return {
            'component': snapshot.component,
            'provider': snapshot.provider,
            'model': snapshot.model,
            'reasoning_or_config_identity': snapshot.reasoning_or_config_identity,
            'authorized_model_config_version': (
                snapshot.authorized_model_config_version
            ),
            'authorized_model_config_snapshot_hmac': (
                snapshot.authorized_model_config_snapshot_hmac
            ),
            'authorized_cost_policy_version': (snapshot.authorized_cost_policy_version),
            'authorized_token_estimator_version': (
                snapshot.authorized_token_estimator_version
            ),
            'authorized_fingerprint_key_version': snapshot.fingerprint_key_version,
            'authorized_fingerprint_key_material_verifier': (
                snapshot.fingerprint_key_material_verifier
            ),
            'authorized_policy_snapshot_hmac': (
                snapshot.authorized_policy_snapshot_hmac
            ),
        }

    @staticmethod
    def _validate_snapshots(
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
    ) -> None:
        if (
            type(snapshots) is not tuple
            or len(snapshots) != 2
            or tuple(item.component for item in snapshots) != _COMPONENTS
        ):
            raise RagProviderSafetyError(
                'exactly two ordered provider families are required'
            )
        if (
            snapshots[0].fingerprint_key_version != snapshots[1].fingerprint_key_version
            or snapshots[0].fingerprint_key_material_verifier
            != snapshots[1].fingerprint_key_material_verifier
        ):
            raise RagProviderSafetyError('provider family key authority differs')
        for snapshot in snapshots:
            expected = _BOOTSTRAP_FAMILIES[snapshot.component]
            if any(getattr(snapshot, key) != value for key, value in expected.items()):
                raise RagProviderSafetyError('frozen provider family is invalid')

    @staticmethod
    def _connection_dialect(connection: object) -> str | None:
        return getattr(getattr(connection, 'dialect', None), 'name', None)

    @contextmanager
    def _registered_advisory(self, connection: object) -> Iterator[None]:
        if self._connection_dialect(connection) != 'postgresql':
            yield
            return
        if self._advisory_capability is None:
            raise RagProviderSafetyError(
                'registered provider safety advisory capability is required'
            )
        acquire_advisory_lock(connection, self._advisory_capability, shared=False)
        try:
            yield
        finally:
            release_advisory_lock(connection, self._advisory_capability, shared=False)

    def _read_unlocked(self) -> dict[str, object]:
        try:
            raw = self._authority._read_bytes_unlocked()
            parsed = json.loads(raw.decode('utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RagProviderSafetyError(
                'provider safety authority is inconsistent'
            ) from exc
        if type(parsed) is not dict or canonical_json_bytes(parsed) != raw:
            raise RagProviderSafetyError(
                'provider safety authority is not exact canonical JSON'
            )
        return self._validate_envelope(parsed, raw=raw)

    def _validate_envelope(
        self, envelope: dict[str, object], *, raw: bytes | None = None
    ) -> dict[str, object]:
        if set(envelope) != {'signed_payload', 'hmac_sha256'}:
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        signed = envelope.get('signed_payload')
        signature = envelope.get('hmac_sha256')
        if (
            type(signed) is not dict
            or set(signed)
            != {
                'body',
                'fingerprint_key_material_verifier',
                'fingerprint_key_version',
            }
            or type(signature) is not str
            or not hmac.compare_digest(signature, self._signature(signed))
        ):
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        key_version = signed['fingerprint_key_version']
        key_verifier = signed['fingerprint_key_material_verifier']
        if type(key_version) is not str or not key_version.strip():
            raise RagProviderSafetyError('provider safety key version is invalid')
        try:
            require_lower_hmac(key_verifier)
        except ValueError as exc:
            raise RagProviderSafetyError(
                'provider safety key verifier is invalid'
            ) from exc
        body = signed.get('body')
        if type(body) is not dict or set(body) != _BODY_KEYS:
            raise RagProviderSafetyError('provider safety body is invalid')
        if (
            body['latch_schema_version'] != 'rag-provider-safety-latch-body:v1'
            or body['designated_environment_id'] != self._environment
        ):
            raise RagProviderSafetyError('provider safety authority is inconsistent')
        try:
            UUID(body['authority_uuid'])  # type: ignore[arg-type]
        except (TypeError, ValueError, AttributeError) as exc:
            raise RagProviderSafetyError(
                'provider safety authority UUID is invalid'
            ) from exc
        generation = body['global_safety_generation']
        if type(generation) is not int or generation < 0:
            raise RagProviderSafetyError('provider safety generation is invalid')
        active = body['active_family_by_component']
        records = body['family_records']
        if (
            type(active) is not dict
            or set(active) != set(_COMPONENTS)
            or type(records) is not list
            or len(records) < 2
        ):
            raise RagProviderSafetyError('provider safety families are inconsistent')
        identities: list[tuple[str, str, str, str]] = []
        by_identity: dict[tuple[str, str, str, str], dict[str, object]] = {}
        for item in records:
            if type(item) is not dict or set(item) != _FAMILY_RECORD_KEYS:
                raise RagProviderSafetyError('provider safety family is invalid')
            if item['component'] not in _COMPONENTS or any(
                type(item[key]) is not str or not str(item[key]).strip()
                for key in ('model', 'provider', 'reasoning_or_config_identity')
            ):
                raise RagProviderSafetyError(
                    'provider safety family identity is invalid'
                )
            identity = _identity_tuple(item)
            if identity in by_identity:
                raise RagProviderSafetyError('provider safety family is duplicated')
            identities.append(identity)
            by_identity[identity] = item
            if item['state'] not in _STATES:
                raise RagProviderSafetyError('provider safety state is invalid')
            if type(item['state_version']) is not int or item['state_version'] < 1:
                raise RagProviderSafetyError('provider safety state version is invalid')
            if (
                type(item['family_safety_generation']) is not int
                or item['family_safety_generation'] < 0
                or item['family_safety_generation'] > generation
            ):
                raise RagProviderSafetyError('provider family generation is invalid')
            try:
                require_lower_hmac(item['authorized_policy_snapshot_hmac'])
                if item['reviewed_transition_reference_hmac'] is not None:
                    require_lower_hmac(item['reviewed_transition_reference_hmac'])
            except ValueError as exc:
                raise RagProviderSafetyError('provider family HMAC is invalid') from exc
            blocker = (
                item['first_blocker_agent_run_hmac'],
                item['first_blocker_category'],
                item['first_blocker_observed_at'],
            )
            if any(value is None for value in blocker) != all(
                value is None for value in blocker
            ):
                raise RagProviderSafetyError('provider blocker attribution is partial')
            if blocker[0] is not None:
                try:
                    require_lower_hmac(blocker[0])
                    parsed_time = datetime.fromisoformat(
                        str(blocker[2]).replace('Z', '+00:00')
                    )
                except (ValueError, TypeError) as exc:
                    raise RagProviderSafetyError(
                        'provider blocker attribution is invalid'
                    ) from exc
                if blocker[1] not in _BLOCKER_CATEGORIES or parsed_time.tzinfo is None:
                    raise RagProviderSafetyError(
                        'provider blocker attribution is invalid'
                    )
        if identities != sorted(identities):
            raise RagProviderSafetyError('provider safety families are not ordered')
        for component in _COMPONENTS:
            identity = active[component]
            if type(identity) is not dict or set(identity) != _FAMILY_IDENTITY_KEYS:
                raise RagProviderSafetyError('active provider family is invalid')
            if identity.get('component') != component:
                raise RagProviderSafetyError('active provider family component differs')
            if _identity_tuple(identity) not in by_identity:
                raise RagProviderSafetyError('active provider family is absent')
        result = dict(body)
        result['_records_by_identity'] = by_identity
        result['_signed_payload'] = signed
        result['_envelope'] = envelope
        result['envelope_digest'] = self._file_digest_bytes(
            raw if raw is not None else canonical_json_bytes(envelope)
        )
        return result

    def _db_is_empty_and_unattempted(self, connection: Connection) -> bool:
        counts = (
            connection.scalar(
                select(func.count()).select_from(RagProviderSafetyAuthority)
            ),
            connection.scalar(select(func.count()).select_from(RagProviderReadiness)),
            connection.scalar(
                select(func.count()).select_from(RagProviderSafetyTransition)
            ),
        )
        attempted = connection.scalar(
            select(func.count())
            .select_from(AgentRunCostComponent)
            .join(AgentRun, AgentRun.id == AgentRunCostComponent.agent_run_id)
            .where(
                AgentRun.run_contract_version == 'rag-run:v2',
                AgentRunCostComponent.component.in_(_COMPONENTS),
                (
                    AgentRunCostComponent.attempted.is_(True)
                    | (AgentRunCostComponent.dispatch_count > 0)
                ),
            )
        )
        return counts == (0, 0, 0) and attempted == 0

    def _insert_bootstrap_rows(
        self,
        connection: Connection,
        *,
        body: Mapping[str, object],
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
        envelope_digest: str,
        reviewed_reference: str,
    ) -> None:
        signed = body['_signed_payload']
        connection.execute(
            insert(RagProviderSafetyAuthority).values(
                id=1,
                authority_uuid=body['authority_uuid'],
                designated_environment_id=self._environment,
                global_safety_generation=0,
                envelope_digest=envelope_digest,
                fingerprint_key_version=signed['fingerprint_key_version'],
                fingerprint_key_material_verifier=(
                    signed['fingerprint_key_material_verifier']
                ),
            )
        )
        for snapshot in snapshots:
            connection.execute(
                insert(RagProviderReadiness).values(
                    authority_id=1,
                    **self._snapshot_values(snapshot),
                    active=True,
                    state='ready',
                    state_version=1,
                    family_safety_generation=0,
                    reviewed_gate_reference_hmac=reviewed_reference,
                )
            )
        connection.execute(
            insert(RagProviderSafetyTransition).values(
                authority_id=1,
                readiness_id=None,
                global_safety_generation=0,
                transition_kind='bootstrap',
                prior_state=None,
                new_state=None,
                prior_state_version=None,
                new_state_version=None,
                prior_family_safety_generation=None,
                new_family_safety_generation=None,
                envelope_digest=envelope_digest,
                reviewed_transition_reference_hmac=reviewed_reference,
                actor_subject_hmac=None,
                agent_run_id=None,
            )
        )

    def bootstrap(
        self,
        connection: Connection,
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
        *,
        reviewed_transition_reference_hmac: str,
    ) -> None:
        self._validate_snapshots(snapshots)
        require_lower_hmac(reviewed_transition_reference_hmac)
        if self._latch_path.exists():
            raise RagProviderSafetyError('provider safety authority already exists')
        initializer = DurableFileAuthority(self._latch_path)
        try:
            with initializer.locked(), self._registered_advisory(connection):
                if self._latch_path.exists() or not self._db_is_empty_and_unattempted(
                    connection
                ):
                    raise RagProviderSafetyError(
                        'provider safety authority already exists'
                    )
                records = [
                    self._record(
                        snapshot,
                        family_generation=0,
                        reviewed_reference=reviewed_transition_reference_hmac,
                    )
                    for snapshot in snapshots
                ]
                records.sort(key=_identity_tuple)
                body = {
                    'active_family_by_component': {
                        snapshot.component: self._identity(snapshot)
                        for snapshot in snapshots
                    },
                    'authority_uuid': str(uuid4()),
                    'designated_environment_id': self._environment,
                    'family_records': records,
                    'global_safety_generation': 0,
                    'latch_schema_version': ('rag-provider-safety-latch-body:v1'),
                }
                envelope = self._wrap(
                    body,
                    key_version=snapshots[0].fingerprint_key_version,
                    key_verifier=(snapshots[0].fingerprint_key_material_verifier),
                )
                initializer._replace_unlocked(envelope)
                self._authority = DurableFileAuthority.open_runtime(self._latch_path)
                parsed = self._read_unlocked()
                self._insert_bootstrap_rows(
                    connection,
                    body=parsed,
                    snapshots=snapshots,
                    envelope_digest=parsed['envelope_digest'],
                    reviewed_reference=reviewed_transition_reference_hmac,
                )
                connection.commit()
                self._match_db_whole_set(connection, parsed)
        except RagProviderSafetyError:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise RagProviderSafetyError(
                'provider safety database bootstrap failed'
            ) from None

    def recover_partial_bootstrap(
        self,
        connection: Connection,
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
    ) -> None:
        self._validate_snapshots(snapshots)
        try:
            with self._authority.locked(), self._registered_advisory(connection):
                body = self._read_unlocked()
                if body[
                    'global_safety_generation'
                ] != 0 or not self._db_is_empty_and_unattempted(connection):
                    raise RagProviderSafetyError(
                        'provider safety bootstrap recovery is not eligible'
                    )
                self._match_external_snapshots(body, snapshots)
                records = list(body['_records_by_identity'].values())
                references = {
                    record['reviewed_transition_reference_hmac'] for record in records
                }
                if (
                    len(records) != 2
                    or len(references) != 1
                    or None in references
                    or any(
                        record['state'] != 'ready'
                        or record['state_version'] != 1
                        or record['family_safety_generation'] != 0
                        or record['first_blocker_agent_run_hmac'] is not None
                        or record['first_blocker_category'] is not None
                        or record['first_blocker_observed_at'] is not None
                        for record in records
                    )
                ):
                    raise RagProviderSafetyError(
                        'provider safety bootstrap recovery shape is invalid'
                    )
                reviewed_reference = references.pop()
                require_lower_hmac(reviewed_reference)
                self._insert_bootstrap_rows(
                    connection,
                    body=body,
                    snapshots=snapshots,
                    envelope_digest=body['envelope_digest'],
                    reviewed_reference=reviewed_reference,
                )
                connection.commit()
                fresh = self._read_unlocked()
                self._match_db_whole_set(connection, fresh)
        except RagProviderSafetyError:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise RagProviderSafetyError(
                'provider safety bootstrap recovery failed'
            ) from None

    def _match_external_snapshots(
        self,
        body: Mapping[str, object],
        snapshots: tuple[
            AuthorizedProviderPolicySnapshot, AuthorizedProviderPolicySnapshot
        ],
    ) -> None:
        active = body['active_family_by_component']
        records = body['_records_by_identity']
        signed = body['_signed_payload']
        if (
            signed['fingerprint_key_version'] != snapshots[0].fingerprint_key_version
            or signed['fingerprint_key_material_verifier']
            != snapshots[0].fingerprint_key_material_verifier
        ):
            raise RagProviderSafetyError('provider family key authority differs')
        for snapshot in snapshots:
            identity = self._identity(snapshot)
            if active[snapshot.component] != identity:
                raise RagProviderSafetyError('provider family rebind is required')
            record = records.get(_identity_tuple(identity))
            if (
                record is None
                or record['authorized_policy_snapshot_hmac']
                != snapshot.authorized_policy_snapshot_hmac
            ):
                raise RagProviderSafetyError('provider family rebind is required')

    def _match_db_whole_set(
        self,
        connection: Connection,
        body: Mapping[str, object],
        *,
        for_update: bool = False,
    ) -> tuple[Mapping[str, object], list[Mapping[str, object]]]:
        authority_query = select(RagProviderSafetyAuthority.__table__).where(
            RagProviderSafetyAuthority.id == 1
        )
        readiness_query = select(RagProviderReadiness.__table__).order_by(
            RagProviderReadiness.component,
            RagProviderReadiness.model,
            RagProviderReadiness.provider,
            RagProviderReadiness.reasoning_or_config_identity,
        )
        if for_update:
            authority_query = authority_query.with_for_update()
            readiness_query = readiness_query.with_for_update()
        authority = connection.execute(authority_query).mappings().one_or_none()
        rows = connection.execute(readiness_query).mappings().all()
        signed = body['_signed_payload']
        if (
            authority is None
            or authority['authority_uuid'] != body['authority_uuid']
            or authority['designated_environment_id'] != self._environment
            or authority['global_safety_generation'] != body['global_safety_generation']
            or authority['envelope_digest'] != body['envelope_digest']
            or authority['fingerprint_key_version'] != signed['fingerprint_key_version']
            or authority['fingerprint_key_material_verifier']
            != signed['fingerprint_key_material_verifier']
        ):
            raise RagProviderSafetyError('provider safety DB/external authority drift')
        external_records = body['_records_by_identity']
        if len(rows) != len(external_records):
            raise RagProviderSafetyError('provider safety DB/external authority drift')
        active = body['active_family_by_component']
        seen_active: set[str] = set()
        for row in rows:
            identity = {key: row[key] for key in _FAMILY_IDENTITY_KEYS}
            record = external_records.get(_identity_tuple(identity))
            should_be_active = active[row['component']] == identity
            if (
                record is None
                or row['active'] is not should_be_active
                or row['authorized_policy_snapshot_hmac']
                != record['authorized_policy_snapshot_hmac']
                or row['state'] != record['state']
                or row['state_version'] != record['state_version']
                or row['family_safety_generation'] != record['family_safety_generation']
                or row['reviewed_gate_reference_hmac']
                != record['reviewed_transition_reference_hmac']
            ):
                raise RagProviderSafetyError(
                    'provider safety DB/external authority drift'
                )
            if should_be_active:
                if row['component'] in seen_active:
                    raise RagProviderSafetyError(
                        'provider safety DB/external authority drift'
                    )
                seen_active.add(row['component'])
        if seen_active != set(_COMPONENTS):
            raise RagProviderSafetyError('provider safety DB/external authority drift')
        return authority, rows

    def _active_record(
        self, body: Mapping[str, object], component: RagPaidComponent
    ) -> dict[str, object]:
        active = body['active_family_by_component'][component]
        return body['_records_by_identity'][_identity_tuple(active)]

    def _binding(
        self,
        connection: Connection,
        body: Mapping[str, object],
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
    ) -> RagProviderSafetyBinding:
        if component not in _COMPONENTS or policy_snapshot.component != component:
            raise RagProviderSafetyError('provider family component differs')
        family = self._active_record(body, component)
        if self._identity(policy_snapshot) != {
            key: family[key] for key in _FAMILY_IDENTITY_KEYS
        }:
            raise RagProviderSafetyError('provider family rebind is required')
        if (
            family['authorized_policy_snapshot_hmac']
            != policy_snapshot.authorized_policy_snapshot_hmac
        ):
            raise RagProviderSafetyError('provider family rebind is required')
        if family['state'] != 'ready':
            raise RagProviderSafetyError('provider family is blocked')
        _, rows = self._match_db_whole_set(connection, body)
        db_family = next(
            row for row in rows if row['active'] and row['component'] == component
        )
        expected_snapshot = self._snapshot_values(policy_snapshot)
        if any(db_family[key] != value for key, value in expected_snapshot.items()):
            raise RagProviderSafetyError('provider family rebind is required')
        snapshot_hmac = provider_safety_snapshot_identity(
            {
                'authority_uuid': body['authority_uuid'],
                'global_safety_generation': body['global_safety_generation'],
                'active_family': {key: family[key] for key in _FAMILY_RECORD_KEYS},
                'envelope_digest': body['envelope_digest'],
            },
            secret=self._secret,
        )
        return RagProviderSafetyBinding(
            policy_snapshot=policy_snapshot,
            authority_uuid=UUID(body['authority_uuid']),  # type: ignore[arg-type]
            designated_environment_id=self._environment,
            envelope_digest=body['envelope_digest'],  # type: ignore[arg-type]
            fingerprint_key_version=policy_snapshot.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                policy_snapshot.fingerprint_key_material_verifier
            ),
            global_safety_generation=body['global_safety_generation'],  # type: ignore[arg-type]
            family_state='ready',
            family_state_version=family['state_version'],  # type: ignore[arg-type]
            family_safety_generation=family['family_safety_generation'],  # type: ignore[arg-type]
            provider_safety_snapshot_hmac=snapshot_hmac,
        )

    def require_ready(
        self,
        connection: Connection,
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
    ) -> RagProviderSafetyBinding:
        try:
            with self._authority.locked(), self._registered_advisory(connection):
                body = self._read_unlocked()
                return self._binding(connection, body, component, policy_snapshot)
        except RagProviderSafetyError:
            raise
        except Exception as exc:
            raise RagProviderSafetyError(
                'provider safety authority is unavailable'
            ) from exc

    def review_context(
        self,
        connection: Connection,
        component: RagPaidComponent,
    ) -> RagProviderSafetyReviewContext:
        if component not in _COMPONENTS:
            raise RagProviderSafetyError('provider family component is invalid')
        try:
            with self._authority.locked(), self._registered_advisory(connection):
                body = self._read_unlocked()
                self._match_db_whole_set(connection, body)
                record = self._active_record(body, component)
                return self._review_context_from(body, component, record)
        except RagProviderSafetyError:
            raise
        except Exception as exc:
            raise RagProviderSafetyError(
                'provider safety review context is unavailable'
            ) from exc

    def _review_context_from(
        self,
        body: Mapping[str, object],
        component: RagPaidComponent,
        record: Mapping[str, object],
    ) -> RagProviderSafetyReviewContext:
        return RagProviderSafetyReviewContext(
            authority_uuid=body['authority_uuid'],  # type: ignore[arg-type]
            designated_environment_id=self._environment,
            component=component,
            family_identity=_identity_tuple(record),
            state=record['state'],  # type: ignore[arg-type]
            state_version=record['state_version'],  # type: ignore[arg-type]
            global_safety_generation=body['global_safety_generation'],  # type: ignore[arg-type]
            has_historical_blocker=any(
                item['component'] == component
                and item['first_blocker_category'] is not None
                for item in body['_records_by_identity'].values()
            ),
            _seal=_REVIEW_CAPABILITY_SEAL,
        )

    @staticmethod
    def _command_component(
        command: ReviewedProviderSafetyCommand,
    ) -> RagPaidComponent:
        if type(command) is not ReviewedProviderSafetyCommand:
            raise RagProviderSafetyError('reviewed command capability is required')
        return command._bound_component()

    def _revalidate(
        self,
        connection: Connection,
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
        *,
        expected_binding: RagProviderSafetyBinding | None,
    ) -> RagProviderSafetyBinding:
        binding = self.require_ready(connection, component, policy_snapshot)
        if expected_binding is not None and binding != expected_binding:
            raise RagProviderSafetyError('provider safety binding changed')
        return binding

    def revalidate_before_send(
        self,
        connection: Connection,
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
        *,
        expected_binding: RagProviderSafetyBinding | None = None,
    ) -> RagProviderSafetyBinding:
        return self._revalidate(
            connection,
            component,
            policy_snapshot,
            expected_binding=expected_binding,
        )

    def revalidate_after_response(
        self,
        connection: Connection,
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
        *,
        expected_binding: RagProviderSafetyBinding | None = None,
    ) -> RagProviderSafetyBinding:
        return self._revalidate(
            connection,
            component,
            policy_snapshot,
            expected_binding=expected_binding,
        )

    @contextmanager
    def dispatch_barrier(
        self,
        connection: Connection,
        component: RagPaidComponent,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
        *,
        expected_binding: RagProviderSafetyBinding,
        order: RagLockOrderCoordinator,
        sidecar_capability: RagLockOrderCapability,
        safety_capability: RagLockOrderCapability,
    ) -> Iterator[None]:
        """Hold stable sidecar and registered DB authority through one send."""
        order.require(sidecar_capability, stage='provider_stable_sidecar')
        order.require(safety_capability, stage='provider_safety_rows')
        with self._authority.locked(), self._registered_advisory(connection):
            before = self._binding(
                connection,
                self._read_unlocked(),
                component,
                policy_snapshot,
            )
            if before != expected_binding:
                raise RagProviderSafetyError('provider safety binding changed')
            yield
            after = self._binding(
                connection,
                self._read_unlocked(),
                component,
                policy_snapshot,
            )
            if after != expected_binding:
                raise RagProviderSafetyError('provider safety binding changed')

    @contextmanager
    def finalization_barrier(
        self,
        connection: Connection,
        requirements: tuple[
            tuple[AuthorizedProviderPolicySnapshot, RagProviderSafetyBinding], ...
        ],
        *,
        order: RagLockOrderCoordinator,
        sidecar_capability: RagLockOrderCapability,
        safety_capability: RagLockOrderCapability,
    ) -> Iterator[None]:
        """Fresh-lock phase-2 safety rows while retaining the stable sidecar."""
        order.require(sidecar_capability, stage='provider_stable_sidecar')
        order.require(safety_capability, stage='provider_safety_rows')
        if (
            type(requirements) is not tuple
            or not requirements
            or any(
                type(snapshot) is not AuthorizedProviderPolicySnapshot
                or type(binding) is not RagProviderSafetyBinding
                or binding.policy_snapshot != snapshot
                for snapshot, binding in requirements
            )
        ):
            raise RagProviderSafetyError(
                'phase-2 provider safety requirements are invalid'
            )
        with self._authority.locked(), self._registered_advisory(connection):
            body = self._read_unlocked()
            self._match_db_whole_set(connection, body, for_update=True)
            fresh = tuple(
                self._binding(connection, body, snapshot.component, snapshot)
                for snapshot, _ in requirements
            )
            if fresh != tuple(binding for _, binding in requirements):
                raise RagProviderSafetyError('provider safety binding changed')
            yield
            after_body = self._read_unlocked()
            after = tuple(
                self._binding(connection, after_body, snapshot.component, snapshot)
                for snapshot, _ in requirements
            )
            if after != fresh:
                raise RagProviderSafetyError('provider safety binding changed')

    def _blocker_agent_run_hmac(self, agent_run_id: int) -> str:
        return rag_identity_hmac(
            {'agent_run_id': agent_run_id},
            secret=self._secret,
            schema_version='rag-provider-blocker-agent-run:v1',
            policy_version='rag-provider-safety:v1',
        )

    def _incident_reference(
        self,
        *,
        component: RagPaidComponent,
        state: ProviderSafetyState,
        agent_run_id: int,
        category: str,
    ) -> str:
        return rag_identity_hmac(
            {
                'agent_run_id': agent_run_id,
                'component': component,
                'category': category,
                'state': state,
            },
            secret=self._secret,
            schema_version='rag-provider-safety-incident-reference:v1',
            policy_version='rag-provider-safety:v1',
        )

    def _commit_transition(
        self,
        connection: Connection,
        *,
        old_body: Mapping[str, object],
        new_body: Mapping[str, object],
        new_digest: str,
        component: RagPaidComponent,
        transition_kind: str,
        reviewed_reference: str,
        actor_subject_hmac: str | None,
        agent_run_id: int | None,
        snapshot: AuthorizedProviderPolicySnapshot | None,
        supersession: bool,
        blocker_values: tuple[int, int, Decimal, datetime] | None,
    ) -> None:
        _, rows = self._match_db_whole_set(connection, old_body, for_update=True)
        old_record = self._active_record(old_body, component)
        old_identity = {key: old_record[key] for key in _FAMILY_IDENTITY_KEYS}
        old_row = next(
            row for row in rows if row['active'] and row['component'] == component
        )
        new_record = self._active_record(new_body, component)
        new_generation = new_body['global_safety_generation']
        authority_result = connection.execute(
            update(RagProviderSafetyAuthority)
            .where(
                RagProviderSafetyAuthority.id == 1,
                RagProviderSafetyAuthority.global_safety_generation
                == old_body['global_safety_generation'],
                RagProviderSafetyAuthority.envelope_digest
                == old_body['envelope_digest'],
            )
            .values(
                global_safety_generation=new_generation,
                envelope_digest=new_digest,
                updated_at=datetime.now(UTC),
            )
        )
        if authority_result.rowcount != 1:
            raise RagProviderSafetyError('provider safety authority CAS failed')
        if supersession:
            connection.execute(
                update(RagProviderReadiness)
                .where(
                    RagProviderReadiness.id == old_row['id'],
                    RagProviderReadiness.active.is_(True),
                )
                .values(active=False, updated_at=datetime.now(UTC))
            )
            if snapshot is None:
                raise RagProviderSafetyError('provider successor snapshot is absent')
            readiness_id = connection.execute(
                insert(RagProviderReadiness)
                .values(
                    authority_id=1,
                    **self._snapshot_values(snapshot),
                    active=True,
                    state='ready',
                    state_version=1,
                    family_safety_generation=new_generation,
                    reviewed_gate_reference_hmac=reviewed_reference,
                )
                .returning(RagProviderReadiness.id)
            ).scalar_one()
            prior_state = None
            prior_state_version = None
            prior_family_generation = None
        else:
            values: dict[str, object] = {
                'state': new_record['state'],
                'state_version': new_record['state_version'],
                'family_safety_generation': new_generation,
                'reviewed_gate_reference_hmac': reviewed_reference,
                'updated_at': datetime.now(UTC),
            }
            if transition_kind == 'reset':
                values.update(
                    reset_by=actor_subject_hmac,
                    reset_at=datetime.now(UTC),
                )
            if snapshot is not None:
                values.update(self._snapshot_values(snapshot))
            if blocker_values is not None and old_row['overrun_agent_run_id'] is None:
                input_tokens, output_tokens, cost_usd, observed_at = blocker_values
                values.update(
                    overrun_agent_run_id=agent_run_id,
                    overrun_input_tokens=input_tokens,
                    overrun_output_tokens=output_tokens,
                    overrun_cost_usd=cost_usd,
                    overrun_observed_at=observed_at,
                )
            row_result = connection.execute(
                update(RagProviderReadiness)
                .where(
                    RagProviderReadiness.id == old_row['id'],
                    RagProviderReadiness.active.is_(True),
                    RagProviderReadiness.state_version == old_record['state_version'],
                    RagProviderReadiness.family_safety_generation
                    == old_record['family_safety_generation'],
                )
                .values(**values)
            )
            if row_result.rowcount != 1:
                raise RagProviderSafetyError('provider family CAS failed')
            readiness_id = old_row['id']
            prior_state = old_record['state']
            prior_state_version = old_record['state_version']
            prior_family_generation = old_record['family_safety_generation']
        connection.execute(
            insert(RagProviderSafetyTransition).values(
                authority_id=1,
                readiness_id=readiness_id,
                global_safety_generation=new_generation,
                transition_kind=transition_kind,
                prior_state=prior_state,
                new_state=new_record['state'],
                prior_state_version=prior_state_version,
                new_state_version=new_record['state_version'],
                prior_family_safety_generation=prior_family_generation,
                new_family_safety_generation=new_generation,
                envelope_digest=new_digest,
                reviewed_transition_reference_hmac=reviewed_reference,
                actor_subject_hmac=actor_subject_hmac,
                agent_run_id=agent_run_id,
            )
        )
        del old_identity
        connection.commit()

    def _mutate(
        self,
        connection: Connection,
        component: RagPaidComponent,
        *,
        transition_kind: str,
        new_state: ProviderSafetyState,
        reviewed_reference: str | None,
        expected_global_generation: int | None,
        expected_state_version: int | None,
        allowed_prior_states: set[str],
        actor_subject_hmac: str | None = None,
        agent_run_id: int | None = None,
        blocker_category: str | None = None,
        blocker_values: tuple[int, int, Decimal] | None = None,
        snapshot: AuthorizedProviderPolicySnapshot | None = None,
        supersession: bool = False,
        reviewed_command: ReviewedProviderSafetyCommand | None = None,
        reviewed_operation: str | None = None,
        preserve_reviewed_reference: bool = False,
    ) -> None:
        if component not in _COMPONENTS:
            raise RagProviderSafetyError('provider family component is invalid')
        if reviewed_reference is not None:
            require_lower_hmac(reviewed_reference)
        if actor_subject_hmac is not None:
            require_lower_hmac(actor_subject_hmac)
        try:
            with self._authority.locked(), self._registered_advisory(connection):
                old = self._read_unlocked()
                self._match_db_whole_set(connection, old, for_update=True)
                record = self._active_record(old, component)
                if preserve_reviewed_reference:
                    reviewed_reference = record[
                        'reviewed_transition_reference_hmac'
                    ]  # type: ignore[assignment]
                if reviewed_command is not None:
                    if reviewed_operation is None:
                        raise RagProviderSafetyError('reviewed operation is absent')
                    context = self._review_context_from(old, component, record)
                    payload = reviewed_command._consume(
                        secret=self._secret,
                        expected_context=context,
                        expected_operation=reviewed_operation,
                        expected_successor=snapshot,
                    )
                    reviewed_reference = payload[
                        'reviewed_gate_reference_hmac'
                    ]  # type: ignore[assignment]
                    actor_subject_hmac = payload['actor_subject_hmac']  # type: ignore[assignment]
                    expected_global_generation = context.global_safety_generation
                    expected_state_version = context.state_version
                if reviewed_reference is None:
                    raise RagProviderSafetyError('reviewed transition reference is absent')
                if (
                    record['state'] not in allowed_prior_states
                    or expected_global_generation is not None
                    and old['global_safety_generation'] != expected_global_generation
                    or expected_state_version is not None
                    and record['state_version'] != expected_state_version
                ):
                    raise RagProviderSafetyError(
                        'provider safety transition CAS failed'
                    )
                parsed = json.loads(
                    canonical_json_bytes(old['_envelope']).decode('utf-8')
                )
                signed = parsed['signed_payload']
                body = signed['body']
                records = body['family_records']
                active_identity = body['active_family_by_component'][component]
                target = next(
                    item
                    for item in records
                    if all(
                        item[key] == active_identity[key]
                        for key in _FAMILY_IDENTITY_KEYS
                    )
                )
                new_generation = body['global_safety_generation'] + 1
                now = datetime.now(UTC)
                if supersession:
                    if snapshot is None or snapshot.component != component:
                        raise RagProviderSafetyError(
                            'provider successor snapshot is invalid'
                        )
                    successor_identity = self._identity(snapshot)
                    if successor_identity == active_identity or any(
                        _identity_tuple(item) == _identity_tuple(successor_identity)
                        for item in records
                    ):
                        raise RagProviderSafetyError(
                            'provider successor family is not new'
                        )
                    successor = self._record(
                        snapshot,
                        family_generation=new_generation,
                        reviewed_reference=reviewed_reference,
                    )
                    records.append(successor)
                    records.sort(key=_identity_tuple)
                    body['active_family_by_component'][component] = successor_identity
                else:
                    target['state'] = new_state
                    target['state_version'] += 1
                    target['family_safety_generation'] = new_generation
                    target['reviewed_transition_reference_hmac'] = reviewed_reference
                    if snapshot is not None:
                        if self._identity(snapshot) != active_identity:
                            raise RagProviderSafetyError(
                                'provider rebind cannot change family identity'
                            )
                        target['authorized_policy_snapshot_hmac'] = (
                            snapshot.authorized_policy_snapshot_hmac
                        )
                    if (
                        blocker_category is not None
                        and target['first_blocker_category'] is None
                    ):
                        target['first_blocker_agent_run_hmac'] = (
                            self._blocker_agent_run_hmac(agent_run_id)  # type: ignore[arg-type]
                        )
                        target['first_blocker_category'] = blocker_category
                        target['first_blocker_observed_at'] = now.isoformat().replace(
                            '+00:00', 'Z'
                        )
                body['global_safety_generation'] = new_generation
                envelope = self._wrap(
                    body,
                    key_version=signed['fingerprint_key_version'],
                    key_verifier=signed['fingerprint_key_material_verifier'],
                )
                try:
                    self._authority._replace_unlocked(envelope)
                except Exception:
                    if blocker_category is not None:
                        self._commit_minimal_remediation(
                            connection,
                            old=old,
                            component=component,
                            reviewed_reference=reviewed_reference,
                            actor_subject_hmac=actor_subject_hmac,
                            agent_run_id=agent_run_id,
                            blocker_values=blocker_values,
                        )
                    raise RagProviderSafetyError(
                        'provider safety external transition failed'
                    ) from None
                new = self._read_unlocked()
                observed_values = None
                if blocker_values is not None:
                    observed_values = (*blocker_values, now)
                try:
                    self._commit_transition(
                        connection,
                        old_body=old,
                        new_body=new,
                        new_digest=new['envelope_digest'],
                        component=component,
                        transition_kind=transition_kind,
                        reviewed_reference=reviewed_reference,
                        actor_subject_hmac=actor_subject_hmac,
                        agent_run_id=agent_run_id,
                        snapshot=snapshot,
                        supersession=supersession,
                        blocker_values=observed_values,
                    )
                    fresh = self._read_unlocked()
                    self._match_db_whole_set(connection, fresh)
                except Exception:
                    connection.rollback()
                    raise RagProviderSafetyError(
                        'external provider transition persisted but DB transition failed'
                    ) from None
        except RagProviderSafetyError:
            connection.rollback()
            raise
        except Exception as exc:
            connection.rollback()
            raise RagProviderSafetyError('provider safety transition failed') from exc

    def _commit_minimal_remediation(
        self,
        connection: Connection,
        *,
        old: Mapping[str, object],
        component: RagPaidComponent,
        reviewed_reference: str,
        actor_subject_hmac: str | None,
        agent_run_id: int | None,
        blocker_values: tuple[int, int, Decimal] | None,
    ) -> None:
        record = self._active_record(old, component)
        if record['state'] != 'ready':
            return
        body = json.loads(canonical_json_bytes(old['_envelope']).decode('utf-8'))[
            'signed_payload'
        ]['body']
        active_identity = body['active_family_by_component'][component]
        target = next(
            item
            for item in body['family_records']
            if _identity_tuple(item) == _identity_tuple(active_identity)
        )
        generation = body['global_safety_generation'] + 1
        target['state'] = 'blocked_remediation'
        target['state_version'] += 1
        target['family_safety_generation'] = generation
        target['reviewed_transition_reference_hmac'] = reviewed_reference
        body['global_safety_generation'] = generation
        remediation_event = {
            'authority_uuid': old['authority_uuid'],
            'component': component,
            'external_envelope_digest': old['envelope_digest'],
            'global_safety_generation': generation,
            'incident_reference_hmac': actor_subject_hmac,
            'state': 'blocked_remediation',
        }
        remediation_digest = hashlib.sha256(
            b'paraworks:provider-safety-db-only-remediation:v1\x00'
            + canonical_json_bytes(remediation_event)
        ).hexdigest()
        synthetic = dict(old)
        synthetic['global_safety_generation'] = generation
        synthetic['_records_by_identity'] = {
            _identity_tuple(item): item for item in body['family_records']
        }
        synthetic['active_family_by_component'] = body['active_family_by_component']
        observed = None
        if blocker_values is not None:
            observed = (*blocker_values, datetime.now(UTC))
        self._commit_transition(
            connection,
            old_body=old,
            new_body=synthetic,
            new_digest=remediation_digest,
            component=component,
            transition_kind='block_remediation',
            reviewed_reference=reviewed_reference,
            actor_subject_hmac=actor_subject_hmac,
            agent_run_id=agent_run_id,
            snapshot=None,
            supersession=False,
            blocker_values=observed,
        )

    def mark_rebind_required(
        self,
        connection: Connection,
        command: ReviewedProviderSafetyCommand,
    ) -> None:
        component = self._command_component(command)
        self._mutate(
            connection,
            component,
            transition_kind='rebind_required',
            new_state='rebind_required',
            reviewed_reference=None,
            expected_global_generation=None,
            expected_state_version=None,
            allowed_prior_states={'ready'},
            reviewed_command=command,
            reviewed_operation='mark_rebind_required',
        )

    def reviewed_rebind(
        self,
        connection: Connection,
        command: ReviewedProviderSafetyCommand,
        policy_snapshot: AuthorizedProviderPolicySnapshot,
    ) -> None:
        component = self._command_component(command)
        self._mutate(
            connection,
            component,
            transition_kind='rebind',
            new_state='ready',
            reviewed_reference=None,
            expected_global_generation=None,
            expected_state_version=None,
            allowed_prior_states={'rebind_required'},
            snapshot=policy_snapshot,
            reviewed_command=command,
            reviewed_operation='rebind',
        )

    def reviewed_reset(
        self,
        connection: Connection,
        command: ReviewedProviderSafetyCommand,
    ) -> None:
        component = self._command_component(command)
        self._mutate(
            connection,
            component,
            transition_kind='reset',
            new_state='ready',
            reviewed_reference=None,
            expected_global_generation=None,
            expected_state_version=None,
            allowed_prior_states={'blocked_overrun', 'blocked_remediation'},
            reviewed_command=command,
            reviewed_operation='reset',
        )

    def reviewed_supersession(
        self,
        connection: Connection,
        command: ReviewedProviderSafetyCommand,
        successor: AuthorizedProviderPolicySnapshot,
    ) -> None:
        component = self._command_component(command)
        self._mutate(
            connection,
            component,
            transition_kind='supersession',
            new_state='ready',
            reviewed_reference=None,
            expected_global_generation=None,
            expected_state_version=None,
            allowed_prior_states=set(_STATES),
            snapshot=successor,
            supersession=True,
            reviewed_command=command,
            reviewed_operation='supersession',
        )

    def _block(
        self,
        connection: Connection,
        component: RagPaidComponent,
        state: Literal['blocked_overrun', 'blocked_remediation'],
        *,
        agent_run_id: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
        category: str,
    ) -> None:
        if (
            type(agent_run_id) is not int
            or agent_run_id <= 0
            or type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens < 0
            or type(cost_usd) is not Decimal
            or not cost_usd.is_finite()
            or cost_usd < 0
        ):
            raise RagProviderSafetyError('provider blocker values are invalid')
        reference = self._incident_reference(
            component=component,
            state=state,
            agent_run_id=agent_run_id,
            category=category,
        )
        self._mutate(
            connection,
            component,
            transition_kind=(
                'block_overrun' if state == 'blocked_overrun' else 'block_remediation'
            ),
            new_state=state,
            reviewed_reference=None,
            expected_global_generation=None,
            expected_state_version=None,
            allowed_prior_states={'ready'},
            agent_run_id=agent_run_id,
            blocker_category=category,
            blocker_values=(input_tokens, output_tokens, cost_usd),
            actor_subject_hmac=reference,
            preserve_reviewed_reference=True,
        )

    def block_overrun(
        self,
        connection: Connection,
        component: RagPaidComponent,
        *,
        agent_run_id: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
    ) -> None:
        self._block(
            connection,
            component,
            'blocked_overrun',
            agent_run_id=agent_run_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            category='provider_usage_overrun',
        )

    def block_remediation(
        self,
        connection: Connection,
        component: RagPaidComponent,
        *,
        category: Literal[
            'provider_response_identity_invalid',
            'provider_embedding_payload_invalid',
            'provider_safety_unavailable',
        ],
        agent_run_id: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
    ) -> None:
        if category not in {
            'provider_response_identity_invalid',
            'provider_embedding_payload_invalid',
            'provider_safety_unavailable',
        }:
            raise RagProviderSafetyError('provider remediation category is invalid')
        self._block(
            connection,
            component,
            'blocked_remediation',
            agent_run_id=agent_run_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            category=category,
        )
