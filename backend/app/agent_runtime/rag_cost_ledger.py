from __future__ import annotations

import hmac
import secrets
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, cast

from sqlalchemy import select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_advisory_locks import (
    RagLockOrderCapability,
    RagLockOrderCoordinator,
    RegisteredAdvisoryLock,
    acquire_advisory_lock,
    begin_rag_lock_order,
    rag_projection_owner_lock_id,
    release_advisory_lock,
)
from backend.app.agent_runtime.rag_cost_policy import (
    RagAnswerCeilingReservation,
    RagBudgetExceededError,
    RagCostPolicy,
    RagUnusedComponentReservation,
)
from backend.app.agent_runtime.rag_postgres_binding import (
    RagPostgresAdvisoryTransport,
)
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyInspectionError,
    RagProviderSafetyService,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    ADMISSION_SOURCE_WINDOWS,
    AuthorizedProviderPolicySnapshot,
    CommittedRagDispatchGrant,
    RagComponentFinal,
    RagProviderSafetyBinding,
    RagRunAdmission,
    RagRunTerminal,
    _ClassifiedProviderObservation,
    admission_source_window,
    terminal_source_window,
)
from backend.app.agent_runtime.rag_safety_identity import (
    admission_identity,
    dead_process_attestation_identity,
    dispatch_fence_identity,
    final_error_identity,
    process_instance_identity,
    projection_owner_identity,
    require_lower_hmac,
    runtime_cost_identity,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    PreparedAnswerInvocation,
    _prepared_invocation_authority_payload,
)
from backend.app.db.initialization import TrustedPostgresRuntimeHealth
from backend.app.models.agent_runs import AgentRun
from backend.app.models.auto_review import AutoReviewRuntimeKeyState
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.models.rag_serving import (
    RagLexicalServingProjection,
    RagServingCorpusGeneration,
)
from backend.app.rag.evidence_projection import PreparedModelInfluenceSet
from backend.app.rag.retrieval import (
    PreparedPaidCallBudget,
    PreparedQueryEmbedding,
    RagPaidComponent,
    validate_rag_serving_index_readiness,
)

_ZERO = Decimal('0.000000')
_COMPONENT_ORDER = ('query_embedding', 'answer_generation')
_LEDGER_ASSEMBLY_SEAL = object()

if TYPE_CHECKING:
    from backend.app.agent_runtime.rag_finalization import RagProjectionPending

RagAdmissionBudget = PreparedPaidCallBudget | RagAnswerCeilingReservation | RagUnusedComponentReservation


class RagCostLedgerError(RuntimeError):
    """A fail-closed durable cost authority refusal."""


class RagCostPersistenceError(RagCostLedgerError):
    """Ledger persistence could not be acknowledged; no authority may be retried."""


class RagPreclaimSafetyRefusalError(RagCostLedgerError):
    """Authenticated safety refusal before this request creates a dispatch claim."""


class RagServingEvidenceChangedError(RagCostLedgerError):
    """Authenticated answer evidence no longer matches current serving rows."""


@dataclass(frozen=True, slots=True)
class PendingProjectionPaidComponentAuthority:
    component: str
    provider: str
    model: str
    authorized_model_config_version: str
    authorized_model_config_snapshot_hmac: str
    authorized_cost_policy_version: str
    authorized_token_estimator_version: str
    authorized_policy_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class PendingProjectionRecoverySnapshot:
    """Fresh, read-only phase-1 identity used before recovery lock acquisition."""

    agent_run_id: int
    projection_owner_fence_hmac: str
    runtime_cost_snapshot_hmac: str
    paid_work_performed: bool
    paid_component_authorities: tuple[
        PendingProjectionPaidComponentAuthority, ...
    ] = ()


@dataclass(frozen=True, slots=True)
class _RagCostLedgerAuthority:
    session: Session
    identity_secret: bytes = field(repr=False)
    after_commit: Callable[[], None] | None = field(repr=False)
    cost_policy: RagCostPolicy = field(repr=False)
    provider_safety: RagProviderSafetyService = field(repr=False)
    provider_connection_factory: Callable[[], Connection] = field(repr=False)
    designated_environment_id: str
    designated_host_id: str
    projection_lock_capability_factory: (
        Callable[[int], RegisteredAdvisoryLock] | None
    ) = field(repr=False)
    runtime_health: TrustedPostgresRuntimeHealth = field(repr=False)
    _seal: object = field(repr=False, compare=False)


def _assemble_rag_cost_ledger(
    session: Session,
    *,
    identity_secret: bytes,
    cost_policy: RagCostPolicy,
    provider_safety: RagProviderSafetyService,
    provider_connection_factory: Callable[[], Connection],
    designated_environment_id: str,
    designated_host_id: str,
    projection_lock_capability_factory: (
        Callable[[int], RegisteredAdvisoryLock] | None
    ) = None,
    runtime_health: TrustedPostgresRuntimeHealth,
    after_commit: Callable[[], None] | None = None,
) -> RagCostLedger:
    """Private composition seam used by the assembly root and deterministic tests."""
    return RagCostLedger(_RagCostLedgerAuthority(
        session=session,
        identity_secret=identity_secret,
        after_commit=after_commit,
        cost_policy=cost_policy,
        provider_safety=provider_safety,
        provider_connection_factory=provider_connection_factory,
        designated_environment_id=designated_environment_id,
        designated_host_id=designated_host_id,
        projection_lock_capability_factory=projection_lock_capability_factory,
        runtime_health=runtime_health,
        _seal=_LEDGER_ASSEMBLY_SEAL,
    ))


def _runtime_health_effect(method):
    """Guard one complete paid ledger effect under the shared runtime gate."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._runtime_health._effect(f'rag_cost_ledger:{method.__name__}'):
            return method(self, *args, **kwargs)

    return guarded


def _make_grant(
    *,
    agent_run_id: int,
    component: RagPaidComponent,
    reserved_cost_usd: Decimal,
    dispatch_fence_hmac: str,
    provider_safety_snapshot_hmac: str,
    on_consume: Callable[[], None],
) -> CommittedRagDispatchGrant:
    lock = threading.Lock()
    consumed = False

    class _Grant:
        __slots__ = ()

        @property
        def agent_run_id(self) -> int:
            return agent_run_id

        @property
        def component(self) -> RagPaidComponent:
            return component

        @property
        def reserved_cost_usd(self) -> Decimal:
            return reserved_cost_usd

        @property
        def dispatch_fence_hmac(self) -> str:
            return dispatch_fence_hmac

        @property
        def provider_safety_snapshot_hmac(self) -> str:
            return provider_safety_snapshot_hmac

        @property
        def consumed(self) -> bool:
            with lock:
                return consumed

        def consume_at_dispatch(self) -> None:
            nonlocal consumed
            with lock:
                if consumed:
                    raise RagCostLedgerError('dispatch grant is already consumed')
                on_consume()
                consumed = True

        def __copy__(self):
            raise TypeError('dispatch grants cannot be copied')

        def __deepcopy__(self, _memo: object):
            raise TypeError('dispatch grants cannot be copied')

        def __reduce_ex__(self, _protocol: int):
            raise TypeError('dispatch grants cannot be serialized')

        def __repr__(self) -> str:
            return '<CommittedRagDispatchGrant opaque>'

    return cast(CommittedRagDispatchGrant, _Grant())


class RagCostLedger:
    """Transactional exact-two RAG cost ledger.

    The exact Task 9 cost policy and provider-safety service are concrete
    authorities. Callers may report typed observations but cannot supply
    pricing or safety side effects.
    """

    def __init__(self, authority: object) -> None:
        if (
            type(authority) is not _RagCostLedgerAuthority
            or authority._seal is not _LEDGER_ASSEMBLY_SEAL
        ):
            raise TypeError('RAG cost ledger requires assembly-owned authority')
        session = authority.session
        identity_secret = authority.identity_secret
        after_commit = authority.after_commit
        cost_policy = authority.cost_policy
        provider_safety = authority.provider_safety
        provider_connection_factory = authority.provider_connection_factory
        designated_environment_id = authority.designated_environment_id
        designated_host_id = authority.designated_host_id
        projection_lock_capability_factory = (
            authority.projection_lock_capability_factory
        )
        runtime_health = authority.runtime_health
        if type(identity_secret) is not bytes or not identity_secret:
            raise ValueError('cost-ledger identity secret is required')
        if (
            type(cost_policy) is not RagCostPolicy
            or type(provider_safety) is not RagProviderSafetyService
            or not callable(provider_connection_factory)
            or type(runtime_health) is not TrustedPostgresRuntimeHealth
            or type(designated_environment_id) is not str
            or not designated_environment_id.strip()
            or designated_environment_id != designated_environment_id.strip()
            or type(designated_host_id) is not str
            or not designated_host_id.strip()
            or designated_host_id != designated_host_id.strip()
        ):
            raise ValueError('cost-ledger authorities are unavailable')
        bind = session.get_bind()
        application_engine = bind.engine if isinstance(bind, Connection) else bind
        if bind.dialect.name == 'postgresql' and (
            type(provider_connection_factory) is not RagPostgresAdvisoryTransport
            or provider_connection_factory.application_engine_authority
            is not application_engine
            or provider_connection_factory.runtime_health_authority
            is not runtime_health
            or provider_safety.advisory_transport_authority
            is not provider_connection_factory
        ):
            raise ValueError(
                'cost-ledger PostgreSQL advisory transport is unavailable'
            )
        self._session = session
        self._secret = identity_secret
        self._after_commit = after_commit
        self._cost_policy = cost_policy
        self._provider_safety = provider_safety
        self._provider_connection_factory = provider_connection_factory
        self._projection_lock_capability_factory = (
            projection_lock_capability_factory
        )
        self._runtime_health = runtime_health
        self._projection_mutex = threading.RLock()
        self._active_grants: dict[tuple[int, str], object] = {}
        self._evidence_refused_grants: dict[int, CommittedRagDispatchGrant] = {}
        self._admission_snapshots: dict[
            tuple[int, str], AuthorizedProviderPolicySnapshot
        ] = {}
        self._admission_budgets: dict[
            tuple[int, str], PreparedPaidCallBudget
        ] = {}
        self._answer_reservations: dict[int, RagAnswerCeilingReservation] = {}
        self._answer_binding_lock = threading.RLock()
        self._grant_bindings: dict[
            tuple[int, str], RagProviderSafetyBinding
        ] = {}
        self._terminal_bindings: dict[tuple[int, str], RagProviderSafetyBinding] = {}
        self._terminal_cost_rows: dict[tuple[int, str], tuple[tuple[str, object], ...]] = {}
        self._pending_projection_identities: dict[int, tuple[str, str, str]] = {}
        self._process_hmac = process_instance_identity(
            {
                'designated_environment_id_bytes': exact_utf8_bytes(
                    designated_environment_id
                ),
                'designated_host_id_bytes': exact_utf8_bytes(designated_host_id),
                'process_boot_nonce_hex': secrets.token_hex(32),
                'process_role': 'application_runtime',
            },
            secret=self._secret,
        )

    def _commit(self) -> None:
        try:
            self._session.commit()
        except Exception:
            # Rollback cannot establish whether the server committed. Never retry,
            # and do not let a broken connection's rollback mask this boundary.
            with suppress(Exception):
                self._session.rollback()
            raise RagCostPersistenceError('ledger commit acknowledgement unavailable') from None
        if self._after_commit is not None:
            try:
                self._after_commit()
            except Exception:
                raise RagCostPersistenceError('ledger commit acknowledgement unavailable') from None

    @property
    def provider_safety_authority(self) -> RagProviderSafetyService:
        return self._provider_safety

    @property
    def cost_policy_authority(self) -> RagCostPolicy:
        return self._cost_policy

    @property
    def provider_connection_factory(self) -> Callable[[], Connection]:
        return self._provider_connection_factory

    @property
    def runtime_health_authority(self) -> TrustedPostgresRuntimeHealth:
        return self._runtime_health

    @property
    def process_instance_hmac(self) -> str:
        """Return the non-secret process identity used by reviewed recovery."""
        return self._process_hmac

    def _expected_dead_process_attestation_hmac(
        self,
        *,
        run_id: int,
        dead_process_instance_hmac: str,
    ) -> str:
        """Build the reviewed operator attestation identity for a drained process."""
        require_lower_hmac(dead_process_instance_hmac, 'dead_process_instance_hmac')
        return dead_process_attestation_identity(
            {
                'agent_run_id': run_id,
                'dead_process_instance_hmac': dead_process_instance_hmac,
                'recovery_outcome': 'abandoned_unknown',
            },
            secret=self._secret,
        )

    def _transport_context(
        self,
        grant: object,
    ) -> tuple[
        AuthorizedProviderPolicySnapshot,
        RagProviderSafetyBinding,
        PreparedPaidCallBudget,
        str,
        str,
    ]:
        try:
            key = (grant.agent_run_id, grant.component)
            consumed = grant.consumed
        except Exception:
            raise RagCostLedgerError('dispatch grant is invalid') from None
        snapshot = self._admission_snapshots.get(key)
        budget = self._admission_budgets.get(key)
        binding = self._grant_bindings.get(key)
        parent = self._session.get(AgentRun, key[0])
        if (
            self._active_grants.get(key) is not grant
            or consumed is not False
            or type(snapshot) is not AuthorizedProviderPolicySnapshot
            or type(binding) is not RagProviderSafetyBinding
            or type(budget) is not PreparedPaidCallBudget
            or parent is None
        ):
            raise RagCostLedgerError('dispatch grant is not transport-authorized')
        retrieval_query_hmac = parent.metadata_.get('retrieval_query_hmac')
        require_lower_hmac(retrieval_query_hmac, 'retrieval_query_hmac')
        context_version = parent.metadata_.get('query_context_version')
        if context_version not in {'direct-query:v1', 'assistant-context:v1'}:
            raise RagCostLedgerError('request query context is invalid')
        return snapshot, binding, budget, retrieval_query_hmac, context_version

    @staticmethod
    def _admission_runtime_component(
        snapshot: AuthorizedProviderPolicySnapshot,
        budget: RagAdmissionBudget,
    ) -> dict[str, object]:
        return {
            'actual_input_tokens': None,
            'actual_output_tokens': None,
            'attempted': False,
            'authorized_cost_policy_version_bytes': exact_utf8_bytes(
                snapshot.authorized_cost_policy_version
            ),
            'authorized_model_config_snapshot_hmac': (
                snapshot.authorized_model_config_snapshot_hmac
            ),
            'authorized_model_config_version_bytes': exact_utf8_bytes(
                snapshot.authorized_model_config_version
            ),
            'authorized_policy_snapshot_hmac': (
                snapshot.authorized_policy_snapshot_hmac
            ),
            'authorized_token_estimator_version_bytes': exact_utf8_bytes(
                snapshot.authorized_token_estimator_version
            ),
            'charge_basis': 'zero',
            'charged_cost_usd': '0.000000',
            'component': budget.component,
            'dispatch_count': 0,
            'dispatch_fence_hmac': None,
            'dispatch_state': 'not_attempted',
            'model_bytes': exact_utf8_bytes(snapshot.model),
            'overrun': False,
            'process_instance_hmac': None,
            'provider_bytes': exact_utf8_bytes(snapshot.provider),
            'reserved_cost_usd': format(budget.reserved_cost_usd, '.6f'),
            'reserved_input_tokens': budget.estimated_input_tokens,
            'reserved_output_tokens': budget.maximum_output_tokens,
        }

    @staticmethod
    def _row_runtime_component(row: AgentRunCostComponent) -> dict[str, object]:
        return {
            'actual_input_tokens': row.actual_input_tokens,
            'actual_output_tokens': row.actual_output_tokens,
            'attempted': row.attempted,
            'authorized_cost_policy_version_bytes': exact_utf8_bytes(
                row.authorized_cost_policy_version
            ),
            'authorized_model_config_snapshot_hmac': (
                row.authorized_model_config_snapshot_hmac
            ),
            'authorized_model_config_version_bytes': exact_utf8_bytes(
                row.authorized_model_config_version
            ),
            'authorized_policy_snapshot_hmac': row.authorized_policy_snapshot_hmac,
            'authorized_token_estimator_version_bytes': exact_utf8_bytes(
                row.authorized_token_estimator_version
            ),
            'charge_basis': row.charge_basis,
            'charged_cost_usd': format(Decimal(row.charged_cost_usd), '.6f'),
            'component': row.component,
            'dispatch_count': row.dispatch_count,
            'dispatch_fence_hmac': row.dispatch_fence_hmac,
            'dispatch_state': row.dispatch_state,
            'model_bytes': exact_utf8_bytes(row.model),
            'overrun': row.overrun,
            'process_instance_hmac': row.process_instance_hmac,
            'provider_bytes': exact_utf8_bytes(row.provider),
            'reserved_cost_usd': format(Decimal(row.reserved_cost_usd), '.6f'),
            'reserved_input_tokens': row.reserved_input_tokens,
            'reserved_output_tokens': row.reserved_output_tokens,
        }

    def _runtime_cost_identity(
        self,
        *,
        agent_run_id: int,
        components: list[dict[str, object]],
        parent_outcome: str | None,
        parent_run_record_phase: str,
        parent_status: str,
        snapshot_stage: str,
    ) -> str:
        total_reserved = sum(
            (Decimal(item['reserved_cost_usd']) for item in components), _ZERO
        )
        total_charged = sum(
            (Decimal(item['charged_cost_usd']) for item in components), _ZERO
        )
        return runtime_cost_identity(
            {
                'agent_run_id': agent_run_id,
                'components': components,
                'parent_outcome': parent_outcome,
                'parent_run_record_phase': parent_run_record_phase,
                'parent_status': parent_status,
                'run_contract_version': 'rag-run:v2',
                'snapshot_stage': snapshot_stage,
                'total_charged_cost_usd': format(total_charged, '.6f'),
                'total_reserved_cost_usd': format(total_reserved, '.6f'),
            },
            secret=self._secret,
        )

    @_runtime_health_effect
    def create_admission(
        self,
        *,
        agent_run_id: int,
        surface: str,
        mode: str,
        cutover_stage: str,
        configured_backend: str,
        query_context_version: str,
        current_text_hmac: str,
        retrieval_query_hmac: str,
        security_scope_fingerprint: str,
        admission_cache_identity_hmac: str | None,
        source_window: str,
        components: tuple[
            tuple[AuthorizedProviderPolicySnapshot, RagAdmissionBudget],
            tuple[AuthorizedProviderPolicySnapshot, RagAdmissionBudget],
        ],
    ) -> RagRunAdmission:
        expected_source_window = admission_source_window(
            mode=mode,
            surface=surface,
            backend=configured_backend,
        )
        stage_rank = {'none': 0, 'ask': 1, 'search': 2, 'assistant': 3}
        surface_rank = {'ask': 1, 'search': 2, 'assistant': 3}
        route_is_active = (
            surface in surface_rank
            and cutover_stage in stage_rank
            and stage_rank[cutover_stage] >= surface_rank[surface]
            and (mode != 'shadow' or configured_backend == 'pgvector')
        )
        context_matches_surface = (
            query_context_version == (
                'assistant-context:v1'
                if surface == 'assistant'
                else 'direct-query:v1'
            )
        )
        if (
            type(agent_run_id) is not int
            or agent_run_id <= 0
            or source_window not in ADMISSION_SOURCE_WINDOWS
            or source_window != expected_source_window
            or surface not in {'ask', 'search', 'assistant'}
            or mode not in {'shadow', 'enforce'}
            or cutover_stage not in {'none', 'ask', 'search', 'assistant'}
            or configured_backend not in {'keyword', 'pgvector'}
            or query_context_version not in {'direct-query:v1', 'assistant-context:v1'}
            or not route_is_active
            or not context_matches_surface
            or type(components) is not tuple
            or len(components) != 2
        ):
            raise ValueError('RAG admission is invalid')
        for value in (
            current_text_hmac, retrieval_query_hmac, security_scope_fingerprint,
        ):
            require_lower_hmac(value)
        normalized = []
        for ordinal, expected in enumerate(_COMPONENT_ORDER):
            pair = components[ordinal]
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('RAG admission components are invalid')
            snapshot, budget = pair
            if (
                type(snapshot) is not AuthorizedProviderPolicySnapshot
                or type(budget) not in {PreparedPaidCallBudget, RagAnswerCeilingReservation, RagUnusedComponentReservation}
                or snapshot.component != expected
                or budget.component != expected
                or not hmac.compare_digest(
                    snapshot.authorized_policy_snapshot_hmac,
                    budget.cost_policy_snapshot_hmac,
                )
            ):
                raise ValueError('RAG admission components are misaligned')
            if type(budget) is RagAnswerCeilingReservation:
                if expected != 'answer_generation' or mode != 'enforce' or surface == 'search':
                    raise ValueError('answer ceiling is unused by this route')
                if budget != self._cost_policy.reserve_answer_generation():
                    raise ValueError('answer ceiling authority is invalid')
            if type(budget) is RagUnusedComponentReservation:
                unused = (
                    configured_backend == 'keyword' if expected == 'query_embedding'
                    else mode != 'enforce' or surface == 'search'
                )
                if not unused or budget != self._cost_policy.reserve_unused_component(expected):
                    raise ValueError('unused component reservation is invalid')
            normalized.append((snapshot, budget))
        total_reserved = sum(
            (budget.reserved_cost_usd for _, budget in normalized), _ZERO
        )
        if any(type(budget) is RagAnswerCeilingReservation for _, budget in normalized):
            if type(normalized[0][1]) is PreparedPaidCallBudget:
                self._cost_policy.validate_prepared_budget(normalized[0][1])
            if total_reserved > Decimal('0.012000'):
                raise RagBudgetExceededError
        computed_admission_hmac = admission_identity(
            {
                'answer_provider_policy_snapshot_hmac': (
                    normalized[1][0].authorized_policy_snapshot_hmac
                    if mode == 'enforce' and surface in {'ask', 'assistant'}
                    else None
                ),
                'configured_backend': configured_backend,
                'current_text_hmac': current_text_hmac,
                'cutover_stage': cutover_stage,
                'graph_version': 'company-memory-rag-answer-v2.0',
                'mode': mode,
                'query_context_version_bytes': exact_utf8_bytes(query_context_version),
                'query_embedding_provider_policy_snapshot_hmac': (
                    normalized[0][0].authorized_policy_snapshot_hmac
                    if configured_backend == 'pgvector' else None
                ),
                'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
                'retrieval_query_hmac': retrieval_query_hmac,
                'security_scope_fingerprint': security_scope_fingerprint,
                'surface': surface,
            }, secret=self._secret,
        )
        if (
            admission_cache_identity_hmac is not None
            and admission_cache_identity_hmac != computed_admission_hmac
        ):
            raise ValueError('caller admission identity does not match server authority')
        runtime_hmac = self._runtime_cost_identity(
            agent_run_id=agent_run_id,
            components=[
                self._admission_runtime_component(snapshot, budget)
                for snapshot, budget in normalized
            ],
            parent_outcome=None,
            parent_run_record_phase='admission',
            parent_status='running',
            snapshot_stage='pre_projection',
        )
        parent = AgentRun(
            id=agent_run_id,
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v2',
            status='running',
            source_window=source_window,
            cache_key='rag-v2-admission:' + computed_admission_hmac,
            model_name='rag-v2-admission',
            generation_provider=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=float(total_reserved),
            permission_level='restricted',
            metadata_={
                'configured_backend': configured_backend,
                'current_text_hmac': current_text_hmac,
                'cutover_stage': cutover_stage,
                'mode': mode,
                'query_context_version': query_context_version,
                'retrieval_query_hmac': retrieval_query_hmac,
                'runtime_cost_snapshot_hmac': runtime_hmac,
                'security_scope_fingerprint': security_scope_fingerprint,
                'surface': surface,
            },
            run_contract_version='rag-run:v2',
            run_record_phase='admission',
            total_charged_cost_usd=_ZERO,
            projection_owner_fence_hmac=None,
            completed_at=None,
        )
        self._session.add(parent)
        for ordinal, (snapshot, budget) in enumerate(normalized):
            self._session.add(AgentRunCostComponent(
                agent_run_id=agent_run_id,
                component=budget.component,
                component_ordinal=ordinal,
                dispatch_state='not_attempted',
                dispatch_fence_hmac=None,
                process_instance_hmac=None,
                attempted=False,
                dispatch_count=0,
                reserved_input_tokens=budget.estimated_input_tokens,
                reserved_output_tokens=budget.maximum_output_tokens,
                actual_input_tokens=None,
                actual_output_tokens=None,
                reserved_cost_usd=budget.reserved_cost_usd,
                charged_cost_usd=_ZERO,
                charge_basis='zero',
                overrun=False,
                provider=snapshot.provider,
                model=snapshot.model,
                authorized_model_config_version=snapshot.authorized_model_config_version,
                authorized_model_config_snapshot_hmac=snapshot.authorized_model_config_snapshot_hmac,
                authorized_cost_policy_version=snapshot.authorized_cost_policy_version,
                authorized_token_estimator_version=snapshot.authorized_token_estimator_version,
                authorized_policy_snapshot_hmac=snapshot.authorized_policy_snapshot_hmac,
                terminal_outcome=None,
            ))
        self._commit()
        for snapshot, saved_budget in normalized:
            self._admission_snapshots[(agent_run_id, snapshot.component)] = snapshot
            if type(saved_budget) is RagAnswerCeilingReservation:
                self._answer_reservations[agent_run_id] = saved_budget
            elif type(saved_budget) is PreparedPaidCallBudget:
                self._admission_budgets[(agent_run_id, snapshot.component)] = saved_budget
        return RagRunAdmission(
            agent_run_id=agent_run_id,
            surface=surface,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            cutover_stage=cutover_stage,  # type: ignore[arg-type]
            configured_backend=configured_backend,  # type: ignore[arg-type]
            query_context_version=query_context_version,  # type: ignore[arg-type]
            current_text_hmac=current_text_hmac,
            retrieval_query_hmac=retrieval_query_hmac,
            security_scope_fingerprint=security_scope_fingerprint,
            query_embedding_provider_policy_snapshot_hmac=(
                normalized[0][0].authorized_policy_snapshot_hmac
                if configured_backend == 'pgvector'
                else None
            ),
            answer_provider_policy_snapshot_hmac=(
                normalized[1][0].authorized_policy_snapshot_hmac
                if mode == 'enforce' and surface in {'ask', 'assistant'}
                else None
            ),
            admission_cache_identity_hmac=computed_admission_hmac,
            source_window=source_window,  # type: ignore[arg-type]
            status='running',
            run_record_phase='admission',
            component_order=_COMPONENT_ORDER,
            total_reserved_cost_usd=total_reserved,
            total_charged_cost_usd=_ZERO,
            runtime_cost_snapshot_hmac=runtime_hmac,
        )

    @_runtime_health_effect
    def bind_answer_budget(
        self, *, run_id: int, prepared: PreparedPaidCallBudget,
        invocation: PreparedAnswerInvocation | None = None,
        model_influence: PreparedModelInfluenceSet | None = None,
    ) -> None:
        """Narrow an undispatched ceiling once; failed commit never grants dispatch."""
        self._cost_policy.validate_prepared_budget(prepared)
        projection_identity = {}
        if invocation is not None or model_influence is not None:
            if (type(invocation) is not PreparedAnswerInvocation
                or type(model_influence) is not PreparedModelInfluenceSet
                or invocation.budget != prepared
                or invocation.model_config_snapshot_hmac != self._cost_policy.answer_model_config_snapshot_hmac
                or invocation.rendered_input_hmac != model_influence.rendered_input_hmac
                or invocation.prepared_invocation_hmac != self._cost_policy.sign_answer_artifact(
                    'prepared_invocation', _prepared_invocation_authority_payload(invocation))):
                raise RagCostLedgerError('answer projection identity is invalid')
            self._cost_policy.verify_answer_model_influence(
                invocation.evidence_slots, invocation.model_influence, prepared_set=model_influence)
            projection_identity = {
                'rendered_input_hmac': invocation.rendered_input_hmac,
                'answer_model_config_snapshot_hmac': invocation.model_config_snapshot_hmac,
                'prepared_model_influence_observation_hmac': model_influence.aggregate_observation_hmac,
            }
        with self._answer_binding_lock:
            reservation = self._answer_reservations.get(run_id)
            if (
                reservation is None
                or prepared.component != 'answer_generation'
                or prepared.estimated_input_tokens > reservation.estimated_input_tokens
                or prepared.maximum_output_tokens > reservation.maximum_output_tokens
                or prepared.reserved_cost_usd > reservation.reserved_cost_usd
                or prepared.cost_policy_snapshot_hmac != reservation.cost_policy_snapshot_hmac
            ):
                raise RagCostLedgerError('answer ceiling binding is unavailable')
            parent, rows = self._locked_run(run_id)
            if invocation is not None and (
                invocation.retrieval_query_hmac != (parent.metadata_ or {}).get('retrieval_query_hmac')
                or any((parent.metadata_ or {}).get(key) is not None for key in projection_identity)
            ):
                raise RagCostLedgerError('answer projection identity changed')
            query, answer = rows
            if (
                parent.status != 'running'
                or parent.run_record_phase != 'admission'
                or self._component_is_unused(parent, 'answer_generation')
                or answer.dispatch_state != 'not_attempted'
                or answer.attempted is not False
                or answer.dispatch_count != 0
                or answer.charged_cost_usd != _ZERO
                or answer.reserved_input_tokens != reservation.estimated_input_tokens
                or answer.reserved_output_tokens != reservation.maximum_output_tokens
                or answer.reserved_cost_usd != reservation.reserved_cost_usd
                or answer.authorized_policy_snapshot_hmac != reservation.cost_policy_snapshot_hmac
                or parent.metadata_.get('configured_backend') == 'pgvector'
                and (query.dispatch_state != 'terminal' or query.terminal_outcome != 'component_succeeded')
            ):
                raise RagCostLedgerError('durable answer reservation changed')
            if parent.metadata_.get('configured_backend') == 'pgvector':
                self._require_terminal_cost_row(run_id, query)
            # Retire before fallible commit/ACK. Recovery may close the run, but
            # neither an uncertain commit nor a second caller may rebind it.
            del self._answer_reservations[run_id]
            answer.reserved_input_tokens = prepared.estimated_input_tokens
            answer.reserved_output_tokens = prepared.maximum_output_tokens
            answer.reserved_cost_usd = prepared.reserved_cost_usd
            parent.estimated_cost_usd = float(sum((Decimal(row.reserved_cost_usd) for row in rows), _ZERO))
            parent.metadata_ = {
                **parent.metadata_,
                **projection_identity,
                'runtime_cost_snapshot_hmac': self._runtime_cost_identity(
                    agent_run_id=run_id,
                    components=[self._row_runtime_component(row) for row in rows],
                    parent_outcome=None, parent_run_record_phase='admission',
                    parent_status='running', snapshot_stage='pre_projection',
                ),
            }
            self._commit()
            self._admission_budgets[(run_id, 'answer_generation')] = prepared

    @_runtime_health_effect
    def claim_component(
        self,
        *,
        run_id: int,
        component: RagPaidComponent,
        prepared: PreparedPaidCallBudget,
    ) -> CommittedRagDispatchGrant:
        if component not in _COMPONENT_ORDER or type(prepared) is not PreparedPaidCallBudget:
            raise ValueError('component claim is invalid')
        self._cost_policy.validate_prepared_budget(prepared)
        if type(self._admission_budgets.get((run_id, component))) is not PreparedPaidCallBudget:
            raise RagCostLedgerError('concrete component budget is unavailable')
        parent = self._session.get(AgentRun, run_id, with_for_update=True)
        rows = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
            .with_for_update()
            .execution_options(populate_existing=True)
        ))
        if (
            parent is None
            or parent.status != 'running'
            or parent.run_record_phase != 'admission'
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
        ):
            raise RagCostLedgerError('committed exact-two admission is unavailable')
        if self._component_is_unused(parent, component):
            raise RagCostLedgerError('component is unused by the admitted route')
        if (
            component == 'answer_generation'
            and parent.metadata_.get('configured_backend') == 'pgvector'
            and (
                rows[0].dispatch_state != 'terminal'
                or rows[0].terminal_outcome != 'component_succeeded'
            )
        ):
            raise RagCostLedgerError('component claim order is invalid')
        row = rows[_COMPONENT_ORDER.index(component)]
        snapshot = self._admission_snapshots.get((run_id, component))
        if (
            row.dispatch_state != 'not_attempted'
            or prepared.component != component
            or row.reserved_input_tokens != prepared.estimated_input_tokens
            or row.reserved_output_tokens != prepared.maximum_output_tokens
            or Decimal(row.reserved_cost_usd) != prepared.reserved_cost_usd
            or not hmac.compare_digest(
                row.authorized_policy_snapshot_hmac,
                prepared.cost_policy_snapshot_hmac,
            )
            or type(snapshot) is not AuthorizedProviderPolicySnapshot
        ):
            raise RagCostLedgerError('component claim authority is stale')
        self._cost_policy.require_authorized_policy_snapshot(
            component,
            row.authorized_policy_snapshot_hmac,
        )
        try:
            with self._provider_connection_factory() as safety_connection:
                binding = self._provider_safety.require_ready(
                    safety_connection,
                    component,
                    snapshot,
                )
        except RagProviderSafetyInspectionError:
            raise RagCostLedgerError('provider safety inspection failed') from None
        except RagProviderSafetyError:
            # This branch precedes every claim/charge mutation. Release its
            # SELECT FOR UPDATE locks before failure closure starts lock order.
            if self._session.new or self._session.dirty or self._session.deleted:
                raise RagCostPersistenceError('preclaim transaction has pending writes') from None
            try:
                self._session.rollback()
            except Exception:
                raise RagCostPersistenceError('preclaim read transaction release failed') from None
            if self._session.in_transaction():
                raise RagCostPersistenceError('preclaim read transaction remains active') from None
            raise RagPreclaimSafetyRefusalError('provider safety refused claim') from None
        except Exception as exc:
            raise RagCostLedgerError(
                'provider safety authority is unavailable'
            ) from exc
        fence = dispatch_fence_identity(
            {
                'agent_run_id': run_id,
                'approval_hmac': None,
                'approval_id_hmac': None,
                'case_id_hmac': None,
                'component': component,
                'dispatch_count': 1,
                'dispatch_nonce_hex': secrets.token_hex(32),
                'prepared_input_hmac': (
                    parent.metadata_['retrieval_query_hmac']
                    if component == 'query_embedding'
                    else prepared.estimator_input_hmac
                ),
                'process_instance_hmac': self._process_hmac,
                'provider_safety_snapshot_hmac': (
                    binding.provider_safety_snapshot_hmac
                ),
                'release_to_generation': None,
                'retrieval_query_hmac': parent.metadata_['retrieval_query_hmac'],
            }, secret=self._secret,
        )
        safety = binding.provider_safety_snapshot_hmac
        row.dispatch_state = 'dispatching'
        row.dispatch_fence_hmac = fence
        row.process_instance_hmac = self._process_hmac
        row.attempted = True
        row.dispatch_count = 1
        row.charged_cost_usd = row.reserved_cost_usd
        row.charge_basis = 'reserved'
        parent.total_charged_cost_usd = self.total_charged_cost(run_id)
        row.updated_at = row.updated_at
        self._commit()

        key = (run_id, component)
        holder: list[CommittedRagDispatchGrant] = []

        def consume() -> None:
            if not holder or self._active_grants.get(key) is not holder[0]:
                raise RagCostLedgerError('dispatch grant authority is unavailable')

        grant = _make_grant(
            agent_run_id=run_id,
            component=component,
            reserved_cost_usd=Decimal(row.reserved_cost_usd),
            dispatch_fence_hmac=fence,
            provider_safety_snapshot_hmac=safety,
            on_consume=consume,
        )
        holder.append(grant)
        self._active_grants[key] = grant
        self._grant_bindings[key] = binding
        return grant

    @_runtime_health_effect
    def consume_committed_grant(self, grant: object) -> None:
        """Authenticate and consume the exact process-local store capability."""
        try:
            key = (grant.agent_run_id, grant.component)
        except Exception:
            raise RagCostLedgerError('dispatch grant is invalid') from None
        if self._active_grants.get(key) is not grant:
            raise RagCostLedgerError('dispatch grant is not store-owned')
        try:
            grant.consume_at_dispatch()
        except RagCostLedgerError:
            raise
        except Exception:
            raise RagCostLedgerError('dispatch grant consumption failed') from None

    @_runtime_health_effect
    def consume_transport_grant(
        self,
        grant: object,
        *,
        order: RagLockOrderCoordinator,
        order_capability: RagLockOrderCapability,
    ) -> None:
        """Consume a grant only under the executable AgentRun-cost order stage."""
        order.require(order_capability, stage='agent_run_cost')
        try:
            run_id = grant.agent_run_id
            component = grant.component
            dispatch_fence_hmac = grant.dispatch_fence_hmac
        except Exception:
            raise RagCostLedgerError('dispatch grant is invalid') from None
        self._session.expire_all()
        parent = self._session.get(
            AgentRun,
            run_id,
            with_for_update=True,
            populate_existing=True,
        )
        rows = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
            .with_for_update()
            .execution_options(populate_existing=True)
        ))
        selected = next(
            (row for row in rows if row.component == component), None
        )
        if (
            parent is None
            or parent.run_contract_version != 'rag-run:v2'
            or parent.status != 'running'
            or parent.run_record_phase != 'admission'
            or parent.projection_owner_fence_hmac is not None
            or len(rows) != 2
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
            or selected is None
            or selected.dispatch_state != 'dispatching'
            or selected.dispatch_count != 1
            or selected.attempted is not True
            or not hmac.compare_digest(
                selected.dispatch_fence_hmac or '', dispatch_fence_hmac
            )
        ):
            raise RagCostLedgerError('durable dispatch row changed before send')
        self.consume_committed_grant(grant)

    @_runtime_health_effect
    def revalidate_c5_before_send(
        self,
        prepared: object,
        *,
        order: RagLockOrderCoordinator,
        order_capability: RagLockOrderCapability,
        load_current_readiness: Callable[[], object],
    ) -> None:
        """Lock and exact-check C.5 key/corpus/readiness rows before a send."""
        order.require(order_capability, stage='c5_key_corpus')
        if type(prepared) not in {PreparedQueryEmbedding, PreparedAnswerInvocation}:
            raise RagCostLedgerError('C.5 prepared carrier is unavailable')
        key_state = self._session.scalar(
            select(AutoReviewRuntimeKeyState)
            .where(
                AutoReviewRuntimeKeyState.component
                == 'auto_review_trust_promotion'
            )
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        corpus = self._session.scalar(
            select(RagServingCorpusGeneration)
            .where(RagServingCorpusGeneration.id == 1)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        if (
            key_state is None
            or corpus is None
            or key_state.ready is not True
            or key_state.generation < 1
            or key_state.fingerprint_key_version
            != corpus.fingerprint_key_version
            or not hmac.compare_digest(
                key_state.fingerprint_key_material_verifier,
                corpus.fingerprint_key_material_verifier,
            )
            or corpus.embedding_model != 'text-embedding-3-small'
            or corpus.embedding_dimensions != 1536
            or corpus.index_policy_version != 'rag-v2-serving-index:v1'
            or corpus.pgvector_cosine_policy_version
            != 'pgvector-cosine-indexable:v1'
        ):
            raise RagCostLedgerError('C.5 serving authority changed before send')
        if type(prepared) is PreparedQueryEmbedding:
            try:
                readiness = validate_rag_serving_index_readiness(
                    load_current_readiness()
                )
            except Exception:
                raise RagCostLedgerError(
                    'C.5 serving readiness changed before send'
                ) from None
            if (
                readiness[0] is not True
                or (
                    prepared.corpus_generation,
                    prepared.vector_index_generation,
                    prepared.readiness_snapshot_hmac,
                )
                != (readiness[1], readiness[2], readiness[10])
                or prepared.corpus_generation != corpus.corpus_generation
                or prepared.vector_index_generation
                != corpus.vector_index_generation
            ):
                raise RagCostLedgerError(
                    'C.5 serving readiness changed before send'
                )
            return
        observations = prepared.model_influence
        if (
            type(observations) is not tuple
            or not 1 <= len(observations) <= 8
            or len(observations) != len(prepared.evidence_slots)
        ):
            raise RagCostLedgerError('C.5 prepared evidence is unavailable')
        bound = tuple(enumerate(zip(
            prepared.evidence_slots, observations, strict=True
        )))
        for ordinal, (slot, observation) in sorted(
            bound,
            key=lambda value: (
                value[1][1].lookup_identity.serving_kind,
                value[1][1].lookup_identity.serving_document_id,
                value[1][1].serving_identity_hmac,
            ),
        ):
            if (
                observation.ordinal != ordinal
                or observation.slot_id != slot.slot_id
                or observation.support_mode != slot.support_mode
            ):
                raise RagCostLedgerError('C.5 prepared evidence is unavailable')
            row = self._session.scalar(
                select(RagLexicalServingProjection)
                .where(
                    RagLexicalServingProjection.serving_identity_hmac
                    == observation.serving_identity_hmac
                )
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
            if row is not None and (
                row.fingerprint_key_version != key_state.fingerprint_key_version
                or not hmac.compare_digest(
                    row.fingerprint_key_material_verifier,
                    key_state.fingerprint_key_material_verifier,
                )
            ):
                raise RagCostLedgerError('C.5 serving authority changed before send')
            if (
                row is None
                or row.corpus_generation_id != 1
                or row.corpus_generation != corpus.corpus_generation
                or row.serving_document_id
                != observation.lookup_identity.serving_document_id
                or row.serving_kind
                != observation.lookup_identity.serving_kind
                or row.support_mode != observation.support_mode
                or row.effective_permission != observation.effective_permission
                or row.serving_version_fingerprint
                != observation.serving_version_fingerprint
                or row.model_content_hmac != observation.model_content_hmac
                or row.canonical_citation_projection_hmac
                != observation.canonical_citation_projection_hmac
            ):
                raise RagServingEvidenceChangedError(
                    'C.5 serving evidence changed before send'
                )

    @_runtime_health_effect
    def finalize_component(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        observation: object,
    ) -> RagComponentFinal:
        if (
            type(observation) is not _ClassifiedProviderObservation
            or observation.component != grant.component
        ):
            raise ValueError('provider outcome is invalid')
        observation._consume()
        outcome = observation
        key = (grant.agent_run_id, grant.component)
        if self._active_grants.get(key) is not grant:
            raise RagCostLedgerError('dispatch grant is not active')
        if getattr(grant, 'consumed', None) is not True:
            raise RagCostLedgerError('dispatch grant was not consumed by transport')
        self._validate_outcome(outcome)
        preview_row = self._session.scalar(
            select(AgentRunCostComponent).where(
                AgentRunCostComponent.agent_run_id == grant.agent_run_id,
                AgentRunCostComponent.component == grant.component,
            )
        )
        if (
            preview_row is None
            or preview_row.dispatch_state != 'dispatching'
            or not hmac.compare_digest(
                preview_row.dispatch_fence_hmac or '', grant.dispatch_fence_hmac
            )
        ):
            raise RagCostLedgerError('dispatch row is not finalizable')
        if not outcome.provider_dispatch_started:
            raise ValueError('claimed provider component must be attempted')
        preview_reserved = Decimal(preview_row.reserved_cost_usd)
        actual = outcome.actual_cost_usd
        usage = outcome.strict_usage
        if (actual is None) != (usage is None):
            raise ValueError('actual cost and strict usage must appear together')
        expected_actual: Decimal | None = None
        if usage is not None:
            expected_actual = self._cost_policy.charge_actual(
                outcome.component,
                usage,
            )
            if actual != expected_actual:
                raise ValueError('actual provider cost does not match authority')
        prepared_budget = self._admission_budgets.get(key)
        if type(prepared_budget) is not PreparedPaidCallBudget:
            raise RagCostLedgerError('prepared provider budget is unavailable')
        overrun = (
            usage is not None
            and expected_actual is not None
            and (
                usage.input_tokens > prepared_budget.estimated_input_tokens
                or usage.output_tokens > prepared_budget.maximum_output_tokens
                or expected_actual > prepared_budget.reserved_cost_usd
                or self._cost_policy.usage_exceeds_authorized_cap(
                    outcome.component,
                    usage,
                )
            )
        )
        if overrun != (outcome.classification == 'known_overrun'):
            raise ValueError('provider overrun classification is inconsistent')
        binding = self._grant_bindings.get(key)
        snapshot = self._admission_snapshots.get(key)
        if (
            type(binding) is not RagProviderSafetyBinding
            or type(snapshot) is not AuthorizedProviderPolicySnapshot
        ):
            raise RagCostLedgerError('provider safety binding is unavailable')
        try:
            with self._provider_connection_factory() as safety_connection:
                if outcome.safety_action == 'block_overrun':
                    assert usage is not None and expected_actual is not None
                    self._provider_safety.block_overrun(
                        safety_connection,
                        outcome.component,
                        agent_run_id=grant.agent_run_id,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=expected_actual,
                    )
                elif outcome.safety_action == 'block_remediation':
                    category = {
                        'response_identity_invalid': (
                            'provider_response_identity_invalid'
                        ),
                        'embedding_payload_invalid': (
                            'provider_embedding_payload_invalid'
                        ),
                        'usage_contract_invalid': 'provider_safety_unavailable',
                        'usage_storage_invalid': 'provider_safety_unavailable',
                    }[outcome.classification]
                    self._provider_safety.block_remediation(
                        safety_connection,
                        outcome.component,
                        category=category,
                        agent_run_id=grant.agent_run_id,
                        input_tokens=0 if usage is None else usage.input_tokens,
                        output_tokens=0 if usage is None else usage.output_tokens,
                        cost_usd=(
                            preview_reserved
                            if expected_actual is None
                            else expected_actual
                        ),
                    )
                else:
                    self._provider_safety.revalidate_after_response(
                        safety_connection,
                        outcome.component,
                        snapshot,
                        expected_binding=binding,
                    )
        except Exception as exc:
            raise RagCostLedgerError(
                'provider safety finalization did not complete'
            ) from exc
        self._session.expire_all()
        parent = self._session.get(
            AgentRun, grant.agent_run_id, with_for_update=True
        )
        row = self._session.scalar(
            select(AgentRunCostComponent).where(
                AgentRunCostComponent.agent_run_id == grant.agent_run_id,
                AgentRunCostComponent.component == grant.component,
            ).with_for_update().execution_options(populate_existing=True)
        )
        if (
            parent is None
            or row is None
            or row.dispatch_state != 'dispatching'
            or not hmac.compare_digest(
                row.dispatch_fence_hmac or '', grant.dispatch_fence_hmac
            )
            or Decimal(row.reserved_cost_usd) != preview_reserved
        ):
            raise RagCostLedgerError('dispatch row changed before finalization')
        row.dispatch_state = (
            'abandoned_unknown'
            if outcome.terminal_outcome == 'abandoned_unknown'
            else 'terminal'
        )
        row.terminal_outcome = outcome.terminal_outcome
        if actual is None:
            row.actual_input_tokens = None
            row.actual_output_tokens = None
            row.charged_cost_usd = row.reserved_cost_usd
            row.charge_basis = 'reserved'
            row.overrun = False
        else:
            if type(actual) is not Decimal or actual < 0:
                raise ValueError('actual provider cost is invalid')
            row.actual_input_tokens = usage.input_tokens
            row.actual_output_tokens = usage.output_tokens
            row.charged_cost_usd = actual
            row.charge_basis = 'actual'
            row.overrun = overrun
        self._session.flush()
        parent.total_charged_cost_usd = self.total_charged_cost(grant.agent_run_id)
        siblings = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == grant.agent_run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
        ))
        if outcome.classification != 'validated_success':
            for sibling in siblings:
                if sibling.dispatch_state == 'not_attempted':
                    self._make_terminal_zero(sibling)
        else:
            for sibling in siblings:
                if (
                    sibling.dispatch_state == 'not_attempted'
                    and self._component_is_unused(parent, sibling.component)
                ):
                    self._make_terminal_zero(sibling)
        if all(item.dispatch_state in {'terminal', 'abandoned_unknown'} for item in siblings):
            if outcome.classification != 'validated_success':
                self._active_grants.pop(key, None)
                self._grant_bindings.pop(key, None)
                terminal = self._finalize_failed_run(
                    parent,
                    cast(
                        tuple[AgentRunCostComponent, AgentRunCostComponent],
                        siblings,
                    ),
                    outcome=outcome.terminal_outcome,
                    completed_at=None,
                    admission_only=False,
                )
                return terminal.component_finals[
                    _COMPONENT_ORDER.index(outcome.component)
                ]
            owner_process = next(
                item.process_instance_hmac for item in siblings
                if item.process_instance_hmac is not None
            )
            parent.run_record_phase = 'cost_finalized_pending_projection'
            parent.projection_owner_fence_hmac = self._projection_owner_fence(
                agent_run_id=grant.agent_run_id,
                process_instance_hmac=owner_process,
            )
            parent.metadata_ = {
                **parent.metadata_,
                'runtime_cost_snapshot_hmac': self._runtime_cost_identity(
                    agent_run_id=parent.id,
                    components=[
                        self._row_runtime_component(item) for item in siblings
                    ],
                    parent_outcome=None,
                    parent_run_record_phase='cost_finalized_pending_projection',
                    parent_status='running',
                    snapshot_stage='pre_projection',
                ),
            }
        self._commit()
        if outcome.classification == 'validated_success':
            self._terminal_bindings[key] = binding
            self._terminal_cost_rows[key] = tuple(sorted(self._row_runtime_component(row).items()))
            if parent.run_record_phase == 'cost_finalized_pending_projection':
                self._pending_projection_identities[parent.id] = (
                    parent.projection_owner_fence_hmac,
                    parent.metadata_['security_scope_fingerprint'],
                    parent.metadata_['runtime_cost_snapshot_hmac'],
                )
        self._active_grants.pop(key, None)
        self._grant_bindings.pop(key, None)
        return self._component_final(parent, row)

    @_runtime_health_effect
    def load_pending_projection(
        self, *, run_id: int, corpus_generation: int,
        vector_index_generation: int | None,
    ) -> RagProjectionPending:
        """Read this request's committed phase-1 receipt; never renew ownership."""
        from backend.app.agent_runtime.rag_finalization import RagProjectionPending

        expected = self._pending_projection_identities.get(run_id)
        if expected is None:
            raise RagCostLedgerError('request-owned pending projection is unavailable')
        parent, rows = self._locked_run(run_id)
        actual = (
            parent.projection_owner_fence_hmac,
            parent.metadata_.get('security_scope_fingerprint'),
            parent.metadata_.get('runtime_cost_snapshot_hmac'),
        )
        snapshot = self._runtime_cost_identity(
            agent_run_id=run_id,
            components=[self._row_runtime_component(row) for row in rows],
            parent_outcome=None, parent_run_record_phase='cost_finalized_pending_projection',
            parent_status='running', snapshot_stage='pre_projection',
        )
        if (
            actual != expected or snapshot != expected[2]
            or parent.status != 'running'
            or parent.run_record_phase != 'cost_finalized_pending_projection'
            or parent.completed_at is not None
            or any(row.dispatch_state != 'terminal' for row in rows)
            or parent.total_charged_cost_usd != sum((Decimal(row.charged_cost_usd) for row in rows), _ZERO)
        ):
            raise RagCostLedgerError('committed pending projection changed')
        return RagProjectionPending(
            parent_agent_run_id=run_id, projection_owner_fence_hmac=expected[0],
            security_scope_fingerprint=expected[1],
            prepared_corpus_generation=corpus_generation,
            prepared_vector_index_generation=vector_index_generation,
            terminal_cost_snapshot_hmac=expected[2],
        )

    @_runtime_health_effect
    def commit_provider_free_pending(
        self,
        *,
        run_id: int,
        corpus_generation: int,
        vector_index_generation: int | None,
    ) -> RagProjectionPending:
        return self._commit_safe_pending(
            run_id=run_id, corpus_generation=corpus_generation,
            vector_index_generation=vector_index_generation, embedding_only=False,
        )

    @_runtime_health_effect
    def commit_embedding_only_pending(
        self,
        *,
        run_id: int,
        corpus_generation: int,
        vector_index_generation: int,
    ) -> RagProjectionPending:
        return self._commit_safe_pending(
            run_id=run_id, corpus_generation=corpus_generation,
            vector_index_generation=vector_index_generation, embedding_only=True,
        )

    @contextmanager
    def _safe_pending_owner(self, run_id: int, *, embedding_only: bool, paid_components: tuple[str, ...] | None = None):
        order = begin_rag_lock_order('ordinary')
        sidecar = order.acquire('provider_stable_sidecar')
        safety = order.acquire('provider_safety_rows')
        with ExitStack() as stack:
            components = paid_components if paid_components is not None else (('query_embedding',) if embedding_only else ())
            if components:
                bindings = tuple(self._terminal_bindings.get((run_id, component)) for component in components)
                if any(binding is None for binding in bindings):
                    raise RagCostLedgerError('terminal paid authority is unavailable')
                connection = stack.enter_context(self._provider_connection_factory())
                stack.enter_context(self._provider_safety.finalization_barrier(
                    connection, tuple((binding.policy_snapshot, binding) for binding in bindings),
                    order=order, sidecar_capability=sidecar, safety_capability=safety,
                ))
            # Zero dispatch skips provider singleton/family/latch entirely.
            stack.enter_context(self.projection_owner_barrier(
                run_id, order=order,
                order_capability=order.acquire('projection_owner'),
            ))
            for stage in ('evidence_shared_barrier', 'c5_key_corpus', 'agent_run_cost', 'optional_assistant'):
                order.acquire(stage)
            order.finish()
            yield

    @contextmanager
    def _terminal_failure_owner(self, run_id: int, *, require_safety_authority: bool):
        """Cost-only failure closure; never usable by pending/success projections."""
        order = begin_rag_lock_order('ordinary')
        sidecar = order.acquire('provider_stable_sidecar')
        safety = order.acquire('provider_safety_rows')
        try:
            with ExitStack() as stack:
                if require_safety_authority:
                    connection = stack.enter_context(self._provider_connection_factory())
                    stack.enter_context(self._provider_safety.accounting_failure_barrier(
                        connection, order=order,
                        sidecar_capability=sidecar, safety_capability=safety,
                    ))
                stack.enter_context(self.projection_owner_barrier(
                    run_id, order=order,
                    order_capability=order.acquire('projection_owner'),
                ))
                for stage in ('evidence_shared_barrier', 'c5_key_corpus', 'agent_run_cost', 'optional_assistant'):
                    order.acquire(stage)
                order.finish()
                yield
        except RagCostPersistenceError:
            raise
        except Exception:
            raise RagCostPersistenceError('terminal failure persistence unavailable') from None

    @_runtime_health_effect
    def finalize_inter_component_failure(self, *, run_id: int, outcome: str, assistant_finalizer=None) -> RagRunTerminal:
        """End this request between dispatches without changing any paid child."""
        if outcome not in {
            'budget_exceeded', 'retriever_not_configured', 'retriever_unavailable',
            'runtime_version_unavailable', 'model_unavailable', 'model_provider_failed',
            'provider_safety_unavailable', 'structured_output_invalid', 'citation_validation_failed',
            'persistence_failed', 'unexpected_internal_error',
        }:
            raise ValueError('inter-component failure outcome is invalid')
        if any((run_id, component) not in self._admission_snapshots for component in _COMPONENT_ORDER):
            raise RagCostLedgerError('request-owned admission is unavailable')
        paid = tuple(component for component in _COMPONENT_ORDER if (run_id, component) in self._terminal_bindings)
        with self._answer_binding_lock, self._terminal_failure_owner(
            run_id, require_safety_authority=bool(paid) or outcome == 'provider_safety_unavailable',
        ):
            parent, rows = self._locked_run(run_id)
            if (parent.status != 'running' or parent.completed_at is not None
                or parent.run_record_phase not in {'admission', 'cost_finalized_pending_projection'}
                or any((run_id, component) in self._active_grants for component in _COMPONENT_ORDER)):
                raise RagCostLedgerError('inter-component terminalization is unavailable')
            if parent.run_record_phase == 'cost_finalized_pending_projection':
                expected = self._pending_projection_identities.get(run_id)
                actual = (parent.projection_owner_fence_hmac,
                          parent.metadata_.get('security_scope_fingerprint'),
                          parent.metadata_.get('runtime_cost_snapshot_hmac'))
                current_hmac = self._runtime_cost_identity(
                    agent_run_id=run_id, components=[self._row_runtime_component(row) for row in rows],
                    parent_outcome=None, parent_run_record_phase='cost_finalized_pending_projection',
                    parent_status='running', snapshot_stage='pre_projection',
                )
                if expected is None or expected != actual or current_hmac != expected[2]:
                    raise RagCostLedgerError('committed pending projection changed')
            for row in rows:
                if row.component in paid:
                    binding = self._terminal_bindings.get((run_id, row.component))
                    if (
                        type(binding) is not RagProviderSafetyBinding
                        or binding.policy_snapshot != self._admission_snapshots.get((run_id, row.component))
                    ):
                        raise RagCostLedgerError('terminal paid receipt is unavailable')
                    self._require_terminal_cost_row(run_id, row)
                elif (row.attempted is not False or row.dispatch_count != 0
                      or row.dispatch_state not in {'not_attempted', 'terminal'}
                      or row.charged_cost_usd != _ZERO or row.dispatch_fence_hmac is not None):
                    raise RagCostLedgerError('undispatched component changed')
            for row in rows:
                if row.component not in paid:
                    self._make_terminal_zero(row)
            parent.projection_owner_fence_hmac = None
            self._answer_reservations.pop(run_id, None)
            self._admission_budgets.pop((run_id, 'answer_generation'), None)
            self._pending_projection_identities.pop(run_id, None)
            return self._finalize_failed_run(parent, rows, outcome=outcome, completed_at=None, admission_only=False,
                                             assistant_finalizer=assistant_finalizer)

    def _defer_answer_evidence_refusal(self, grant: CommittedRagDispatchGrant) -> None:
        """Retire transport's exact unconsumed answer grant; never allow resend."""
        key = (grant.agent_run_id, grant.component)
        if (grant.component != 'answer_generation' or grant.consumed is not False
            or self._active_grants.get(key) is not grant):
            raise RagCostLedgerError('evidence refusal grant is unavailable')
        self._evidence_refused_grants[grant.agent_run_id] = grant
        self._active_grants.pop(key)

    @_runtime_health_effect
    def commit_answer_evidence_changed_pending(
        self, *, grant: CommittedRagDispatchGrant, corpus_generation: int,
        vector_index_generation: int | None,
    ) -> RagProjectionPending:
        with self._answer_binding_lock:
            if self._evidence_refused_grants.get(grant.agent_run_id) is not grant:
                raise RagCostLedgerError('evidence refusal proof is unavailable')
            # Release pre-send C.5 read locks before starting ordinary lock order.
            if self._session.new or self._session.dirty or self._session.deleted:
                raise RagCostLedgerError('evidence refusal has uncommitted writes')
            self._session.rollback()
            return self._commit_safe_pending(
                run_id=grant.agent_run_id, corpus_generation=corpus_generation,
                vector_index_generation=vector_index_generation,
                embedding_only=(grant.agent_run_id, 'query_embedding') in self._terminal_bindings,
                refused_grant=grant,
            )

    def _commit_safe_pending(
        self, *, run_id: int, corpus_generation: int,
        vector_index_generation: int | None, embedding_only: bool,
        refused_grant: CommittedRagDispatchGrant | None = None,
    ) -> RagProjectionPending:
        from backend.app.agent_runtime.rag_finalization import RagProjectionPending

        if (
            type(run_id) is not int or run_id <= 0
            or type(corpus_generation) is not int or corpus_generation < 0
            or vector_index_generation is not None
            and (type(vector_index_generation) is not int or vector_index_generation < 0)
            or embedding_only and vector_index_generation is None
        ):
            raise ValueError('pending projection generation is invalid')
        if any((run_id, component) not in self._admission_snapshots for component in _COMPONENT_ORDER):
            raise RagCostLedgerError('request-owned admission is unavailable')
        with self._answer_binding_lock, self._safe_pending_owner(run_id, embedding_only=embedding_only):
            parent, rows = self._locked_run(run_id)
            query, answer = rows
            if refused_grant is not None and (
                self._evidence_refused_grants.get(run_id) is not refused_grant
                or refused_grant.consumed is not False
                or answer.dispatch_state != 'dispatching'
                or answer.attempted is not True or answer.dispatch_count != 1
                or answer.charged_cost_usd != refused_grant.reserved_cost_usd
                or answer.reserved_cost_usd != refused_grant.reserved_cost_usd
                or answer.charge_basis != 'reserved'
                or answer.process_instance_hmac != self._process_hmac
                or answer.dispatch_fence_hmac != refused_grant.dispatch_fence_hmac
                or (run_id, 'answer_generation') in self._active_grants
            ):
                raise RagCostLedgerError('evidence refusal proof is unavailable')
            if (
                parent.status != 'running' or parent.run_record_phase != 'admission'
                or parent.projection_owner_fence_hmac is not None
                or parent.completed_at is not None
                or any(
                    row.dispatch_state != 'not_attempted'
                    or row.attempted is not False or row.dispatch_count != 0
                    or row.charged_cost_usd != _ZERO
                    or row.dispatch_fence_hmac is not None
                    for row in ((answer,) if embedding_only else rows)
                    if refused_grant is None or row.component != 'answer_generation'
                )
            ):
                raise RagCostLedgerError('safe pending transition is unavailable')
            if embedding_only and (
                parent.metadata_.get('configured_backend') != 'pgvector'
                or query.dispatch_state != 'terminal'
                or query.terminal_outcome != 'component_succeeded'
                or query.attempted is not True or query.dispatch_count != 1
                or query.charge_basis != 'actual' or query.overrun is not False
                or query.process_instance_hmac != self._process_hmac
            ):
                raise RagCostLedgerError('terminal embedding cost is unavailable')
            if embedding_only:
                self._require_terminal_cost_row(run_id, query)
            for row in ((answer,) if embedding_only else rows):
                self._make_terminal_zero(row)
            parent.run_record_phase = 'cost_finalized_pending_projection'
            parent.total_charged_cost_usd = sum((Decimal(row.charged_cost_usd) for row in rows), _ZERO)
            parent.estimated_cost_usd = float(sum((Decimal(row.reserved_cost_usd) for row in rows), _ZERO))
            parent.projection_owner_fence_hmac = self._projection_owner_fence(
                agent_run_id=run_id, process_instance_hmac=self._process_hmac,
            )
            snapshot_hmac = self._runtime_cost_identity(
                agent_run_id=run_id,
                components=[self._row_runtime_component(row) for row in rows],
                parent_outcome=None, parent_run_record_phase='cost_finalized_pending_projection',
                parent_status='running', snapshot_stage='pre_projection',
            )
            parent.metadata_ = {**parent.metadata_, 'runtime_cost_snapshot_hmac': snapshot_hmac}
            pending = RagProjectionPending(
                parent_agent_run_id=run_id,
                projection_owner_fence_hmac=parent.projection_owner_fence_hmac,
                security_scope_fingerprint=parent.metadata_['security_scope_fingerprint'],
                prepared_corpus_generation=corpus_generation,
                prepared_vector_index_generation=vector_index_generation,
                terminal_cost_snapshot_hmac=snapshot_hmac,
            )
            self._answer_reservations.pop(run_id, None)
            self._admission_budgets.pop((run_id, 'answer_generation'), None)
            self._commit()
            if refused_grant is not None:
                self._evidence_refused_grants.pop(run_id)
                self._grant_bindings.pop((run_id, 'answer_generation'), None)
            self._pending_projection_identities[run_id] = (
                pending.projection_owner_fence_hmac,
                pending.security_scope_fingerprint,
                pending.terminal_cost_snapshot_hmac,
            )
        return pending

    def _require_terminal_cost_row(self, run_id: int, row: AgentRunCostComponent) -> None:
        expected = self._terminal_cost_rows.get((run_id, row.component))
        if expected is None or tuple(sorted(self._row_runtime_component(row).items())) != expected:
            raise RagCostLedgerError('terminal component cost changed')

    @staticmethod
    def _component_is_unused(parent: AgentRun, component: str) -> bool:
        metadata = parent.metadata_
        if component == 'query_embedding':
            return metadata.get('configured_backend') == 'keyword'
        if component == 'answer_generation':
            return not (
                metadata.get('mode') == 'enforce'
                and metadata.get('surface') in {'ask', 'assistant'}
            )
        raise RagCostLedgerError('unknown RAG paid component')

    def _projection_owner_fence(
        self,
        *,
        agent_run_id: int,
        process_instance_hmac: str,
    ) -> str:
        dialect = self._session.get_bind().dialect.name
        if dialect == 'sqlite':
            advisory_digest = None
            lock_backend = 'sqlite_process_mutex'
            lock_namespace = 'sqlite_rag_smoke_projection_owner'
        elif dialect == 'postgresql':
            if self._projection_lock_capability_factory is None:
                raise RagCostLedgerError(
                    'PostgreSQL projection owner capability is unavailable'
                )
            capability = self._projection_lock_capability_factory(agent_run_id)
            if (
                type(capability) is not RegisteredAdvisoryLock
                or not capability.matches(
                    rag_projection_owner_lock_id(agent_run_id),
                    identity_namespace='dynamic',
                )
            ):
                raise RagCostLedgerError(
                    'PostgreSQL projection owner capability is invalid'
                )
            advisory_digest = capability.identity_digest
            lock_backend = 'postgres_advisory'
            lock_namespace = 'rag_projection_owner'
        else:
            raise RagCostLedgerError('projection owner backend is unsupported')
        return projection_owner_identity(
            {
                'advisory_lock_identity_digest': advisory_digest,
                'agent_run_id': agent_run_id,
                'lock_backend': lock_backend,
                'lock_namespace': lock_namespace,
                'owner_nonce_hex': secrets.token_hex(32),
                'owner_phase': 'cost_finalized_pending_projection',
                'process_instance_hmac': process_instance_hmac,
            },
            secret=self._secret,
        )

    @contextmanager
    def projection_owner_barrier(
        self,
        agent_run_id: int,
        *,
        order: RagLockOrderCoordinator,
        order_capability: RagLockOrderCapability,
    ) -> Iterator[None]:
        """Hold the backend-correct projection-owner lock through provider send."""
        order.require(order_capability, stage='projection_owner')
        identity = rag_projection_owner_lock_id(agent_run_id)
        dialect = self._session.get_bind().dialect.name
        if dialect == 'sqlite':
            with self._projection_mutex:
                yield
            return
        if dialect != 'postgresql' or self._projection_lock_capability_factory is None:
            raise RagCostLedgerError('projection owner authority is unavailable')
        capability = self._projection_lock_capability_factory(agent_run_id)
        if (
            type(capability) is not RegisteredAdvisoryLock
            or not capability.matches(identity, identity_namespace='dynamic')
        ):
            raise RagCostLedgerError('projection owner capability is invalid')
        connection_context = self._provider_connection_factory()
        if type(self._provider_connection_factory) is RagPostgresAdvisoryTransport:
            with (
                connection_context as connection,
                self._provider_connection_factory.advisory_connection(
                    connection,
                    capability,
                    shared=False,
                ),
            ):
                yield
            return
        connection = connection_context
        locked = False
        try:
            acquire_advisory_lock(connection, capability, shared=False)
            locked = True
            yield
        finally:
            try:
                if locked:
                    release_advisory_lock(connection, capability, shared=False)
            finally:
                connection.close()

    @staticmethod
    def _validate_outcome(outcome: _ClassifiedProviderObservation) -> None:
        response_less_terminal = (
            'retriever_unavailable'
            if outcome.component == 'query_embedding'
            else 'model_provider_failed'
        )
        matrix = {
            'validated_success': ('component_succeeded', True, True, 'actual', 'unchanged'),
            'response_less_failure': (
                response_less_terminal, True, False, 'reserved', 'unchanged'
            ),
            'response_identity_invalid': (
                'provider_response_identity_invalid', True, True, 'actual_or_reserved',
                'block_remediation',
            ),
            'embedding_payload_invalid': (
                'provider_embedding_payload_invalid', True, True,
                'actual_or_reserved',
                'block_remediation',
            ),
            'usage_contract_invalid': (
                'provider_safety_unavailable', True, True, 'reserved',
                'block_remediation',
            ),
            'usage_storage_invalid': (
                'provider_safety_unavailable', True, True, 'reserved',
                'block_remediation',
            ),
            'known_overrun': (
                'provider_usage_overrun', True, True, 'actual', 'block_overrun'
            ),
            'structured_output_invalid': (
                'structured_output_invalid', True, True, 'actual', 'unchanged'
            ),
            'citation_validation_failed': (
                'citation_validation_failed', True, True, 'actual', 'unchanged'
            ),
            'evidence_validation_failed': (
                'evidence_unavailable', True, True, 'actual', 'unchanged'
            ),
        }
        expected = matrix.get(outcome.classification)
        if expected is None or outcome.classification == 'pre_send_refusal':
            raise ValueError('provider outcome classification is not finalizable')
        terminal, started, received, basis, action = expected
        has_actual = outcome.strict_usage is not None and outcome.actual_cost_usd is not None
        if (
            outcome.terminal_outcome != terminal
            or outcome.provider_dispatch_started is not started
            or outcome.provider_response_received is not received
            or outcome.safety_action != action
            or (basis == 'actual' and not has_actual)
            or (basis == 'reserved' and has_actual)
            or (outcome.strict_usage is None) != (outcome.actual_cost_usd is None)
        ):
            raise ValueError('provider outcome classification is inconsistent')

    def total_charged_cost(self, run_id: int) -> Decimal:
        rows = self._session.scalars(select(AgentRunCostComponent).where(
            AgentRunCostComponent.agent_run_id == run_id
        ))
        return sum((Decimal(row.charged_cost_usd) for row in rows), _ZERO)

    @_runtime_health_effect
    def finalize_projectionless_failure(
        self,
        *,
        run_id: int,
        outcome: str,
        completed_at: datetime | None = None,
    ) -> RagRunTerminal:
        """Close an admission before any paid dispatch using exact terminal-zero rows."""
        if outcome != 'abandoned_unknown':
            raise ValueError('admission-only outcome must be abandoned_unknown')
        parent = self._session.get(AgentRun, run_id, with_for_update=True)
        rows = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
            .with_for_update()
            .execution_options(populate_existing=True)
        ))
        if (
            parent is None
            or parent.status != 'running'
            or parent.run_record_phase != 'admission'
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
            or any(row.dispatch_state != 'not_attempted' for row in rows)
        ):
            raise RagCostLedgerError('projectionless terminal-zero closure is unavailable')
        timestamp = completed_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError('completion timestamp must be timezone-aware')
        for row in rows:
            row.dispatch_state = 'terminal'
            row.attempted = False
            row.dispatch_count = 0
            row.reserved_input_tokens = 0
            row.reserved_output_tokens = 0
            row.actual_input_tokens = None
            row.actual_output_tokens = None
            row.reserved_cost_usd = _ZERO
            row.charged_cost_usd = _ZERO
            row.charge_basis = 'zero'
            row.overrun = False
            row.dispatch_fence_hmac = None
            row.process_instance_hmac = None
            row.terminal_outcome = None
        parent.status = 'failed'
        parent.run_record_phase = 'admission_only'
        parent.total_charged_cost_usd = _ZERO
        parent.completed_at = timestamp
        terminal_runtime_hmac = self._runtime_cost_identity(
            agent_run_id=run_id,
            components=[self._row_runtime_component(row) for row in rows],
            parent_outcome=outcome,
            parent_run_record_phase='admission_only',
            parent_status='failed',
            snapshot_stage='terminal',
        )
        parent.metadata_ = {
            **parent.metadata_,
            'runtime_cost_snapshot_hmac': terminal_runtime_hmac,
            'outcome': outcome,
        }
        self._commit()
        finals = tuple(self._component_final(parent, row) for row in rows)
        return RagRunTerminal(
            agent_run_id=run_id,
            status='failed',
            run_record_phase='admission_only',
            outcome=outcome,  # type: ignore[arg-type]
            admission_cache_identity_hmac=parent.cache_key.removeprefix(
                'rag-v2-admission:'
            ),
            source_window=parent.source_window,  # type: ignore[arg-type]
            cache_key=parent.cache_key,
            total_reserved_cost_usd=_ZERO,
            total_charged_cost_usd=_ZERO,
            component_finals=cast(tuple[RagComponentFinal, RagComponentFinal], finals),
            projection_owner_fence_hmac=None,
            runtime_cost_snapshot_hmac=terminal_runtime_hmac,
            terminal_identity_hmac=None,
            completed_at=timestamp,
        )

    @_runtime_health_effect
    def finalize_pre_send_refusal(
        self,
        *,
        run_id: int,
        component: RagPaidComponent,
        outcome: str,
        completed_at: datetime | None = None,
    ) -> RagRunTerminal:
        """Close an exact admission before dispatch; both children are terminal-zero."""
        if (
            component not in _COMPONENT_ORDER
            or outcome not in {
                'provider_safety_unavailable',
                'retriever_unavailable',
                'model_provider_failed',
                'serving_index_not_ready',
            }
        ):
            raise ValueError('pre-send refusal is invalid')
        parent, rows = self._locked_run(run_id)
        if (
            parent.status != 'running'
            or parent.run_record_phase != 'admission'
            or any(row.dispatch_state != 'not_attempted' for row in rows)
        ):
            raise RagCostLedgerError('pre-send terminal-zero closure is unavailable')
        for row in rows:
            self._make_terminal_zero(row)
        return self._finalize_failed_run(
            parent,
            rows,
            outcome=outcome,
            completed_at=completed_at,
            admission_only=False,
        )

    def cancel_claim_before_dispatch(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        outcome: str,
    ) -> RagRunTerminal:
        """Close a committed claim that was authoritatively refused before send."""
        try:
            key = (grant.agent_run_id, grant.component)
            consumed = grant.consumed
        except Exception:
            raise RagCostLedgerError('dispatch grant is invalid') from None
        if (
            self._active_grants.get(key) is not grant
            or consumed is not False
            or outcome not in {
                'provider_safety_unavailable',
                'evidence_unavailable',
            }
        ):
            raise RagCostLedgerError('pre-send claimed refusal is unavailable')
        parent, rows = self._locked_run(grant.agent_run_id)
        if parent.status != 'running' or parent.run_record_phase != 'admission':
            raise RagCostLedgerError('pre-send claimed refusal parent is unavailable')
        selected = rows[_COMPONENT_ORDER.index(grant.component)]
        if (
            selected.dispatch_state != 'dispatching'
            or not hmac.compare_digest(
                selected.dispatch_fence_hmac or '', grant.dispatch_fence_hmac
            )
        ):
            raise RagCostLedgerError('pre-send claimed refusal row is unavailable')
        for row in rows:
            if row.dispatch_state in {'dispatching', 'not_attempted'}:
                self._make_terminal_zero(row)
        self._active_grants.pop(key, None)
        self._grant_bindings.pop(key, None)
        return self._finalize_failed_run(
            parent,
            rows,
            outcome=outcome,
            completed_at=None,
            admission_only=False,
        )

    def recover_incomplete_run(
        self,
        *,
        run_id: int,
        projection_owner_fence_hmac: str | None = None,
        expected_runtime_cost_snapshot_hmac: str | None = None,
        dead_process_attestation_hmac: str | None = None,
        completed_at: datetime | None = None,
    ) -> RagRunTerminal:
        """Fail closed without redispatch after a dead dispatch or projection owner."""
        parent, rows = self._locked_run(run_id)
        if parent.status != 'running':
            raise RagCostLedgerError('incomplete RAG run is unavailable')
        pending_projection = parent.run_record_phase == 'cost_finalized_pending_projection'
        if pending_projection:
            current_runtime_hmac = self._runtime_cost_identity(
                agent_run_id=parent.id,
                components=[self._row_runtime_component(row) for row in rows],
                parent_outcome=None,
                parent_run_record_phase='cost_finalized_pending_projection',
                parent_status='running',
                snapshot_stage='pre_projection',
            )
            if (
                type(projection_owner_fence_hmac) is not str
                or not hmac.compare_digest(
                    parent.projection_owner_fence_hmac or '',
                    projection_owner_fence_hmac,
                )
            ):
                raise RagCostLedgerError('projection owner fence is stale')
            if (
                type(expected_runtime_cost_snapshot_hmac) is not str
                or not hmac.compare_digest(
                    parent.metadata_.get('runtime_cost_snapshot_hmac', ''),
                    expected_runtime_cost_snapshot_hmac,
                )
                or not hmac.compare_digest(
                    current_runtime_hmac,
                    expected_runtime_cost_snapshot_hmac,
                )
            ):
                raise RagCostLedgerError('pending projection cost snapshot changed')
            if any(row.dispatch_state != 'terminal' for row in rows):
                raise RagCostLedgerError('pending projection children are incomplete')
            outcome = 'persistence_failed'
        else:
            dead_processes = {
                row.process_instance_hmac
                for row in rows
                if row.process_instance_hmac is not None
                and (
                    row.dispatch_state == 'dispatching'
                    or (
                        row.dispatch_state == 'terminal'
                        and row.attempted
                        and row.dispatch_count == 1
                    )
                )
            }
            dead_process = next(iter(dead_processes), None)
            expected_attestation = (
                None
                if dead_process is None or len(dead_processes) != 1
                else self._expected_dead_process_attestation_hmac(
                    run_id=run_id,
                    dead_process_instance_hmac=dead_process,
                )
            )
            if (
                parent.run_record_phase != 'admission'
                or not (
                    any(row.dispatch_state == 'dispatching' for row in rows)
                    or (
                        any(
                            row.dispatch_state == 'terminal' and row.attempted
                            for row in rows
                        )
                        and any(
                            row.dispatch_state == 'not_attempted' for row in rows
                        )
                    )
                )
                or projection_owner_fence_hmac is not None
                or expected_attestation is None
                or type(dead_process_attestation_hmac) is not str
                or not hmac.compare_digest(
                    expected_attestation,
                    dead_process_attestation_hmac,
                )
            ):
                raise RagCostLedgerError(
                    'reviewed dead-process recovery authority is unavailable'
                )
            for row in rows:
                if row.dispatch_state == 'dispatching':
                    row.dispatch_state = 'abandoned_unknown'
                    row.terminal_outcome = 'abandoned_unknown'
                elif row.dispatch_state == 'not_attempted':
                    self._make_terminal_zero(row)
            outcome = 'abandoned_unknown'
        for component in _COMPONENT_ORDER:
            self._active_grants.pop((run_id, component), None)
            self._grant_bindings.pop((run_id, component), None)
        return self._finalize_failed_run(
            parent,
            rows,
            outcome=outcome,
            completed_at=completed_at,
            admission_only=not pending_projection,
        )

    def pending_projection_recovery_snapshot(
        self, run_id: int
    ) -> PendingProjectionRecoverySnapshot:
        """Read exact phase-1 identities; recovery later CASes them under row locks."""
        self._session.expire_all()
        parent = self._session.get(AgentRun, run_id)
        rows = tuple(
            self._session.scalars(
                select(AgentRunCostComponent)
                .where(AgentRunCostComponent.agent_run_id == run_id)
                .order_by(AgentRunCostComponent.component_ordinal)
                .execution_options(populate_existing=True)
            )
        )
        fence = None if parent is None else parent.projection_owner_fence_hmac
        runtime_hmac = (
            None
            if parent is None
            else parent.metadata_.get('runtime_cost_snapshot_hmac')
        )
        if (
            parent is None
            or parent.status != 'running'
            or parent.run_record_phase != 'cost_finalized_pending_projection'
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
            or any(row.dispatch_state != 'terminal' for row in rows)
            or type(fence) is not str
            or type(runtime_hmac) is not str
        ):
            raise RagCostLedgerError('pending projection recovery is unavailable')
        current_runtime_hmac = self._runtime_cost_identity(
            agent_run_id=parent.id,
            components=[self._row_runtime_component(row) for row in rows],
            parent_outcome=None,
            parent_run_record_phase='cost_finalized_pending_projection',
            parent_status='running',
            snapshot_stage='pre_projection',
        )
        if not hmac.compare_digest(current_runtime_hmac, runtime_hmac):
            raise RagCostLedgerError('pending projection cost snapshot changed')
        return PendingProjectionRecoverySnapshot(
            agent_run_id=run_id,
            projection_owner_fence_hmac=fence,
            runtime_cost_snapshot_hmac=runtime_hmac,
            paid_work_performed=any(row.attempted for row in rows),
            paid_component_authorities=tuple(
                PendingProjectionPaidComponentAuthority(
                    component=row.component,
                    provider=row.provider,
                    model=row.model,
                    authorized_model_config_version=(
                        row.authorized_model_config_version
                    ),
                    authorized_model_config_snapshot_hmac=(
                        row.authorized_model_config_snapshot_hmac
                    ),
                    authorized_cost_policy_version=(
                        row.authorized_cost_policy_version
                    ),
                    authorized_token_estimator_version=(
                        row.authorized_token_estimator_version
                    ),
                    authorized_policy_snapshot_hmac=(
                        row.authorized_policy_snapshot_hmac
                    ),
                )
                for row in rows
                if row.attempted
            ),
        )

    def _locked_run(
        self,
        run_id: int,
    ) -> tuple[AgentRun, tuple[AgentRunCostComponent, AgentRunCostComponent]]:
        parent = self._session.get(AgentRun, run_id, with_for_update=True)
        rows = tuple(self._session.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == run_id)
            .order_by(AgentRunCostComponent.component_ordinal)
            .with_for_update()
            .execution_options(populate_existing=True)
        ))
        if (
            parent is None
            or tuple(row.component for row in rows) != _COMPONENT_ORDER
        ):
            raise RagCostLedgerError('exact-two RAG run is unavailable')
        return parent, cast(
            tuple[AgentRunCostComponent, AgentRunCostComponent], rows
        )

    @staticmethod
    def _make_terminal_zero(row: AgentRunCostComponent) -> None:
        row.dispatch_state = 'terminal'
        row.attempted = False
        row.dispatch_count = 0
        row.reserved_input_tokens = 0
        row.reserved_output_tokens = 0
        row.actual_input_tokens = None
        row.actual_output_tokens = None
        row.reserved_cost_usd = _ZERO
        row.charged_cost_usd = _ZERO
        row.charge_basis = 'zero'
        row.overrun = False
        row.dispatch_fence_hmac = None
        row.process_instance_hmac = None
        row.terminal_outcome = None

    def _finalize_failed_run(
        self,
        parent: AgentRun,
        rows: tuple[AgentRunCostComponent, AgentRunCostComponent],
        *,
        outcome: str,
        completed_at: datetime | None,
        admission_only: bool,
        assistant_finalizer=None,
    ) -> RagRunTerminal:
        timestamp = completed_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError('completion timestamp must be timezone-aware')
        total_reserved = sum(
            (Decimal(row.reserved_cost_usd) for row in rows), _ZERO
        )
        total_charged = sum(
            (Decimal(row.charged_cost_usd) for row in rows), _ZERO
        )
        admission_hmac = parent.cache_key.removeprefix('rag-v2-admission:')
        phase = 'admission_only' if admission_only else 'final'
        terminal_runtime_hmac = self._runtime_cost_identity(
            agent_run_id=parent.id,
            components=[self._row_runtime_component(row) for row in rows],
            parent_outcome=outcome,
            parent_run_record_phase=phase,
            parent_status='failed',
            snapshot_stage='terminal',
        )
        terminal_hmac = None
        if admission_only:
            source_window = parent.source_window
            cache_key = parent.cache_key
            parent.projection_owner_fence_hmac = None
        else:
            parts = parent.source_window.split(':')
            if len(parts) != 5 or parts[:2] != ['rag-v2', 'admission']:
                raise RagCostLedgerError('admission source window is invalid')
            source_window = terminal_source_window(
                stage='final_error', surface=parts[3], backend=parts[4]
            )
            terminal_hmac = final_error_identity(
                {
                    'admission_cache_identity_hmac': admission_hmac,
                    'answer_question_hmac': None,
                    'configured_backend': parent.metadata_['configured_backend'],
                    'current_text_hmac': parent.metadata_['current_text_hmac'],
                    'fallback_category': None,
                    'graph_version': 'company-memory-rag-answer-v2.0',
                    'outcome': outcome,
                    'prepared_model_influence_observation_hmac': None,
                    'retrieval_query_hmac': parent.metadata_[
                        'retrieval_query_hmac'
                    ],
                    'runtime_cost_snapshot_hmac': terminal_runtime_hmac,
                    'security_scope_fingerprint': parent.metadata_[
                        'security_scope_fingerprint'
                    ],
                    'surface': parent.metadata_['surface'],
                },
                secret=self._secret,
            )
            cache_key = 'rag-v2-final-error:' + terminal_hmac
        parent.status = 'failed'
        parent.run_record_phase = phase
        parent.source_window = source_window
        parent.cache_key = cache_key
        parent.total_charged_cost_usd = total_charged
        parent.completed_at = timestamp
        parent.metadata_ = {
            **parent.metadata_,
            'outcome': outcome,
            'runtime_cost_snapshot_hmac': terminal_runtime_hmac,
            'terminal_identity_hmac': terminal_hmac,
        }
        if assistant_finalizer is not None:
            from backend.app.agent_runtime.rag_finalization import (
                AssistantInterComponentFailureFinalizer,
            )
            if type(assistant_finalizer) is not AssistantInterComponentFailureFinalizer or admission_only:
                raise RagCostLedgerError('assistant finalization authority is invalid')
            assistant_finalizer.append(self._session, parent)
        self._commit()
        finals = cast(
            tuple[RagComponentFinal, RagComponentFinal],
            tuple(self._component_final(parent, row) for row in rows),
        )
        return RagRunTerminal(
            agent_run_id=parent.id,
            status='failed',
            run_record_phase=phase,  # type: ignore[arg-type]
            outcome=outcome,  # type: ignore[arg-type]
            admission_cache_identity_hmac=admission_hmac,
            source_window=source_window,  # type: ignore[arg-type]
            cache_key=cache_key,
            total_reserved_cost_usd=total_reserved,
            total_charged_cost_usd=total_charged,
            component_finals=finals,
            projection_owner_fence_hmac=parent.projection_owner_fence_hmac,
            runtime_cost_snapshot_hmac=terminal_runtime_hmac,
            terminal_identity_hmac=terminal_hmac,
            completed_at=timestamp,
        )

    def _component_final(
        self, parent: AgentRun, row: AgentRunCostComponent
    ) -> RagComponentFinal:
        return RagComponentFinal(
            agent_run_id=parent.id,
            component=row.component,  # type: ignore[arg-type]
            terminal_outcome=row.terminal_outcome,  # type: ignore[arg-type]
            dispatch_state=row.dispatch_state,  # type: ignore[arg-type]
            attempted=row.attempted,
            dispatch_count=row.dispatch_count,  # type: ignore[arg-type]
            reserved_input_tokens=row.reserved_input_tokens,
            reserved_output_tokens=row.reserved_output_tokens,
            actual_input_tokens=row.actual_input_tokens,
            actual_output_tokens=row.actual_output_tokens,
            reserved_cost_usd=Decimal(row.reserved_cost_usd),
            charged_cost_usd=Decimal(row.charged_cost_usd),
            charge_basis=row.charge_basis,  # type: ignore[arg-type]
            overrun=row.overrun,
            provider=row.provider,
            model=row.model,
            authorized_model_config_version=row.authorized_model_config_version,
            authorized_model_config_snapshot_hmac=row.authorized_model_config_snapshot_hmac,
            authorized_cost_policy_version=row.authorized_cost_policy_version,
            authorized_token_estimator_version=row.authorized_token_estimator_version,
            authorized_policy_snapshot_hmac=row.authorized_policy_snapshot_hmac,
            dispatch_fence_hmac=row.dispatch_fence_hmac,
            process_instance_hmac=row.process_instance_hmac,
            parent_status=parent.status,  # type: ignore[arg-type]
            parent_run_record_phase=parent.run_record_phase,  # type: ignore[arg-type]
            parent_total_charged_cost_usd=Decimal(parent.total_charged_cost_usd),
            projection_owner_fence_hmac=parent.projection_owner_fence_hmac,
            runtime_cost_snapshot_hmac=parent.metadata_['runtime_cost_snapshot_hmac'],
        )
