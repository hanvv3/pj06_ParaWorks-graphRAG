from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Protocol, TypeAlias, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.provider_send_fence import RagEvidenceSendBarrier
from backend.app.agent_runtime.provider_usage import (
    AssistantPersistedOutcome,
    RagCannedMessageIdentity,
    RagResultOutcome,
)
from backend.app.agent_runtime.rag_advisory_locks import (
    RegisteredAdvisoryLock,
    begin_rag_lock_order,
    rag_projection_owner_lock_id,
    release_advisory_lock,
    try_acquire_advisory_lock,
)
from backend.app.agent_runtime.rag_cost_ledger import (
    PendingProjectionPaidComponentAuthority,
    PendingProjectionRecoverySnapshot,
    RagCostLedger,
)
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyService,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    RagProviderSafetyBinding,
    terminal_source_window,
)
from backend.app.agent_runtime.rag_safety_identity import (
    FinalProductIdentityInput,
    final_product_identity,
    runtime_cost_identity,
)
from backend.app.agent_runtime.rag_v2_contracts import RagEffectiveBackend
from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    exact_utf8_bytes,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ValidatedAnswerBlocks,
    build_answer_output_schema_hmac,
    build_answer_prompt_renderer_hmac,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import PreparedRagRequestText
from backend.app.assistant.evidence_persistence import (
    AssistantEvidenceWriter,
    AssistantMessageProjection,
    _mint_assistant_exact_write_authority,
)
from backend.app.core.config import Settings
from backend.app.models import (
    AgentRun,
    AgentRunCostComponent,
    AssistantConversation,
    AssistantMessage,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    ModelInfluenceDependencySnapshot,
    PreparedModelInfluenceObservation,
    PreparedModelInfluenceSet,
    ProjectionFence,
    V1EvidenceProjection,
    build_model_influence_set_hmac,
    build_v1_search_result_set_projection_hmac,
    build_v1_selected_evidence_projection_hmac,
)
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import (
    EvidenceSlot,
    EvidenceSlotId,
    QueryEmbeddingCallResult,
    RetrievalRequest,
    RetrievalResult,
    rank_evidence_slots,
    validate_query_embedding_call_result,
)
from backend.app.rag.serving_locks import (
    ServingProjectionReadCoordinator,
    ServingProjectionTailLockedContext,
)


class RagFinalizationError(RuntimeError):
    """No immutable product was committed; callers must not synthesize one."""


_PHASE2_AUTHORITY_SEAL = object()
_RECOVERY_AUTHORITY_SEAL = object()
_ANSWER_SAFE_NO_EVIDENCE_OUTCOMES = frozenset({
    'no_match',
    'hidden_only',
    'safety_filter_empty',
    'insufficient_evidence',
})
_RAG_PRODUCT_OUTCOMES = _ANSWER_SAFE_NO_EVIDENCE_OUTCOMES | {
    'supported',
    'search_projected',
    'evidence_unavailable',
}


@dataclass(frozen=True, slots=True)
class _ProviderFreePhase2Assembly:
    owner_connection_factory: Callable[[], object] = field(repr=False)
    owner_capability_factory: Callable[[int], RegisteredAdvisoryLock] = field(
        repr=False
    )
    load_current_owner_fence: Callable[[int], str] = field(repr=False)
    evidence_barrier: RagEvidenceSendBarrier = field(repr=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PaidPhase2Assembly:
    provider_safety: RagProviderSafetyService = field(repr=False)
    safety_connection_factory: Callable[[], object] = field(repr=False)
    safety_requirements: tuple[
        tuple[AuthorizedProviderPolicySnapshot, RagProviderSafetyBinding], ...
    ] = field(repr=False)
    owner_connection_factory: Callable[[], object] = field(repr=False)
    owner_capability_factory: Callable[[int], RegisteredAdvisoryLock] = field(
        repr=False
    )
    load_current_owner_fence: Callable[[int], str] = field(repr=False)
    evidence_barrier: RagEvidenceSendBarrier = field(repr=False)
    load_current_readiness: Callable[[], RagServingIndexReadiness] = field(
        repr=False
    )
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _ProjectionOwnerRecoveryAssembly:
    ledger: RagCostLedger = field(repr=False)
    provider_free: ProviderFreeRagPhase2Authority = field(repr=False)
    paid: PaidRagPhase2Authority = field(repr=False)
    projection_read: ServingProjectionReadCoordinator = field(repr=False)
    _seal: object = field(repr=False, compare=False)


class ProviderFreeRagPhase2Authority:
    """Concrete zero-provider owner/evidence authority; never reads safety."""

    __slots__ = ('_assembly',)

    def __init__(self, assembly: object) -> None:
        if (
            type(assembly) is not _ProviderFreePhase2Assembly
            or assembly._seal is not _PHASE2_AUTHORITY_SEAL
            or not callable(assembly.owner_connection_factory)
            or not callable(assembly.owner_capability_factory)
            or not callable(assembly.load_current_owner_fence)
            or type(assembly.evidence_barrier) is not RagEvidenceSendBarrier
        ):
            raise TypeError('provider-free phase-2 authority is unavailable')
        self._assembly = assembly

    @property
    def evidence_barrier_is_postgresql(self) -> bool:
        return self._assembly.evidence_barrier.is_postgresql_backed

    @contextmanager
    def acquire(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        branch: str,
    ):
        if (
            branch != 'provider_free'
            or prepared.query_embedding_result is not None
            or prepared.model_influence_observations
        ):
            raise RagFinalizationError('provider-free phase-2 branch is invalid')
        assembly = self._assembly
        order = begin_rag_lock_order('ordinary')
        # Provider-free work dispatches no paid component. Advance the shared
        # order without touching provider sidecar or safety storage.
        order.acquire('provider_stable_sidecar')
        order.acquire('provider_safety_rows')
        owner_order = order.acquire('projection_owner')
        owner_capability = assembly.owner_capability_factory(
            pending.parent_agent_run_id
        )
        if (
            type(owner_capability) is not RegisteredAdvisoryLock
            or not owner_capability.matches(
                rag_projection_owner_lock_id(pending.parent_agent_run_id),
                identity_namespace='dynamic',
            )
        ):
            raise RagFinalizationError('projection-owner capability is invalid')
        owner_connection = assembly.owner_connection_factory()
        locked = False
        try:
            order.require(owner_order, stage='projection_owner')
            locked = try_acquire_advisory_lock(
                owner_connection, owner_capability, shared=False
            )
            if not locked:
                raise RagFinalizationError('projection owner is still live')
            if (
                assembly.load_current_owner_fence(pending.parent_agent_run_id)
                != pending.projection_owner_fence_hmac
            ):
                raise RagFinalizationError('projection-owner fence changed')
            evidence_order = order.acquire('evidence_shared_barrier')
            c5_order = order.acquire('c5_key_corpus')
            order.acquire('agent_run_cost')
            order.acquire('optional_assistant')
            order.finish()
            with assembly.evidence_barrier.finalization_barrier(
                order=order,
                evidence_capability=evidence_order,
                c5_capability=c5_order,
            ):
                yield
        finally:
            try:
                if locked:
                    release_advisory_lock(
                        owner_connection, owner_capability, shared=False
                    )
            finally:
                owner_connection.close()  # type: ignore[attr-defined]

    @contextmanager
    def acquire_recovery(self, snapshot: PendingProjectionRecoverySnapshot):
        if (
            type(snapshot) is not PendingProjectionRecoverySnapshot
            or snapshot.paid_work_performed
        ):
            raise RagFinalizationError(
                'provider-free projection recovery authority is invalid'
            )
        assembly = self._assembly
        order = begin_rag_lock_order('ordinary')
        order.acquire('provider_stable_sidecar')
        order.acquire('provider_safety_rows')
        owner_order = order.acquire('projection_owner')
        owner_capability = assembly.owner_capability_factory(
            snapshot.agent_run_id
        )
        if (
            type(owner_capability) is not RegisteredAdvisoryLock
            or not owner_capability.matches(
                rag_projection_owner_lock_id(snapshot.agent_run_id),
                identity_namespace='dynamic',
            )
        ):
            raise RagFinalizationError('projection-owner capability is invalid')
        owner_connection = assembly.owner_connection_factory()
        locked = False
        try:
            order.require(owner_order, stage='projection_owner')
            locked = try_acquire_advisory_lock(
                owner_connection, owner_capability, shared=False
            )
            if not locked:
                raise RagFinalizationError('projection owner is still live')
            if (
                assembly.load_current_owner_fence(snapshot.agent_run_id)
                != snapshot.projection_owner_fence_hmac
            ):
                raise RagFinalizationError('projection-owner fence changed')
            evidence_order = order.acquire('evidence_shared_barrier')
            c5_order = order.acquire('c5_key_corpus')
            order.acquire('agent_run_cost')
            order.acquire('optional_assistant')
            order.finish()
            with assembly.evidence_barrier.finalization_barrier(
                order=order,
                evidence_capability=evidence_order,
                c5_capability=c5_order,
            ):
                yield
        finally:
            try:
                if locked:
                    release_advisory_lock(
                        owner_connection, owner_capability, shared=False
                    )
            finally:
                owner_connection.close()  # type: ignore[attr-defined]


class PaidRagPhase2Authority:
    """Concrete retained-sidecar safety/readiness/owner phase-2 authority."""

    __slots__ = ('_assembly', '_current_readiness')

    def __init__(self, assembly: object) -> None:
        if (
            type(assembly) is not _PaidPhase2Assembly
            or assembly._seal is not _PHASE2_AUTHORITY_SEAL
            or type(assembly.provider_safety) is not RagProviderSafetyService
            or not callable(assembly.safety_connection_factory)
            or not callable(assembly.owner_connection_factory)
            or not callable(assembly.owner_capability_factory)
            or not callable(assembly.load_current_owner_fence)
            or type(assembly.evidence_barrier) is not RagEvidenceSendBarrier
            or not callable(assembly.load_current_readiness)
        ):
            raise TypeError('paid phase-2 authority is unavailable')
        self._assembly = assembly
        self._current_readiness: RagServingIndexReadiness | None = None

    @property
    def current_readiness_hmac(self) -> str | None:
        value = self._current_readiness
        return value.readiness_snapshot_hmac if value is not None else None

    @property
    def evidence_barrier_is_postgresql(self) -> bool:
        return self._assembly.evidence_barrier.is_postgresql_backed

    def _require_exact_requirement_components(
        self,
        expected_components: tuple[str, ...],
    ) -> None:
        requirements = self._assembly.safety_requirements
        actual = tuple(snapshot.component for snapshot, _ in requirements)
        if (
            len(actual) != len(set(actual))
            or actual != expected_components
        ):
            raise RagFinalizationError(
                'paid phase-2 safety requirements do not match request'
            )

    def validate_locked_cost_authority(
        self,
        children: tuple[AgentRunCostComponent, ...],
        *,
        expected_components: tuple[str, ...],
    ) -> None:
        self._require_exact_requirement_components(expected_components)
        rows = {row.component: row for row in children}
        for snapshot, _ in self._assembly.safety_requirements:
            row = rows.get(snapshot.component)
            if row is None or (
                row.provider != snapshot.provider
                or row.model != snapshot.model
                or row.authorized_model_config_version
                != snapshot.authorized_model_config_version
                or row.authorized_model_config_snapshot_hmac
                != snapshot.authorized_model_config_snapshot_hmac
                or row.authorized_cost_policy_version
                != snapshot.authorized_cost_policy_version
                or row.authorized_token_estimator_version
                != snapshot.authorized_token_estimator_version
                or row.authorized_policy_snapshot_hmac
                != snapshot.authorized_policy_snapshot_hmac
            ):
                raise RagFinalizationError(
                    'paid phase-2 safety authority changed from committed cost'
                )

    @contextmanager
    def acquire(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        branch: str,
    ):
        if branch not in {'paid_embedding_only', 'paid_prepared'}:
            raise RagFinalizationError('paid phase-2 branch is invalid')
        self._require_exact_requirement_components(
            _expected_paid_components(prepared, branch=branch)
        )
        assembly = self._assembly
        order = begin_rag_lock_order('ordinary')
        sidecar_order = order.acquire('provider_stable_sidecar')
        safety_order = order.acquire('provider_safety_rows')
        safety_connection = assembly.safety_connection_factory()
        try:
            with assembly.provider_safety.finalization_barrier(
                safety_connection,  # type: ignore[arg-type]
                assembly.safety_requirements,
                order=order,
                sidecar_capability=sidecar_order,
                safety_capability=safety_order,
            ):
                owner_order = order.acquire('projection_owner')
                owner_capability = assembly.owner_capability_factory(
                    pending.parent_agent_run_id
                )
                if (
                    type(owner_capability) is not RegisteredAdvisoryLock
                    or not owner_capability.matches(
                        rag_projection_owner_lock_id(
                            pending.parent_agent_run_id
                        ),
                        identity_namespace='dynamic',
                    )
                ):
                    raise RagFinalizationError(
                        'projection-owner capability is invalid'
                    )
                owner_connection = assembly.owner_connection_factory()
                locked = False
                try:
                    order.require(owner_order, stage='projection_owner')
                    locked = try_acquire_advisory_lock(
                        owner_connection, owner_capability, shared=False
                    )
                    if not locked:
                        raise RagFinalizationError(
                            'projection owner is still live'
                        )
                    if (
                        assembly.load_current_owner_fence(
                            pending.parent_agent_run_id
                        )
                        != pending.projection_owner_fence_hmac
                    ):
                        raise RagFinalizationError(
                            'projection-owner fence changed'
                        )
                    readiness = assembly.load_current_readiness()
                    if type(readiness) is not RagServingIndexReadiness:
                        raise RagFinalizationError(
                            'phase-2 index readiness is unavailable'
                        )
                    self._current_readiness = readiness
                    prepared_embedding = (
                        prepared.query_embedding_result.prepared
                        if prepared.query_embedding_result is not None
                        else None
                    )
                    if prepared_embedding is not None and (
                        readiness.ready is not True
                        or readiness.corpus_generation
                        != prepared_embedding.corpus_generation
                        or readiness.vector_index_generation
                        != prepared_embedding.vector_index_generation
                        or readiness.readiness_snapshot_hmac
                        != prepared_embedding.readiness_snapshot_hmac
                    ):
                        raise RagFinalizationError(
                            'phase-2 index readiness changed'
                        )
                    evidence_order = order.acquire('evidence_shared_barrier')
                    c5_order = order.acquire('c5_key_corpus')
                    order.acquire('agent_run_cost')
                    order.acquire('optional_assistant')
                    order.finish()
                    with assembly.evidence_barrier.finalization_barrier(
                        order=order,
                        evidence_capability=evidence_order,
                        c5_capability=c5_order,
                    ):
                        yield
                finally:
                    try:
                        if locked:
                            release_advisory_lock(
                                owner_connection,
                                owner_capability,
                                shared=False,
                            )
                    finally:
                        owner_connection.close()  # type: ignore[attr-defined]
        except RagProviderSafetyError as exc:
            raise RagFinalizationError('phase-2 provider safety changed') from exc
        finally:
            safety_connection.close()  # type: ignore[attr-defined]

    @contextmanager
    def acquire_recovery(self, snapshot: PendingProjectionRecoverySnapshot):
        if (
            type(snapshot) is not PendingProjectionRecoverySnapshot
            or snapshot.paid_work_performed is not True
        ):
            raise RagFinalizationError('paid projection recovery authority is invalid')
        expected_components = tuple(
            value.component for value in snapshot.paid_component_authorities
        )
        self._require_exact_requirement_components(expected_components)
        committed = {
            value.component: value
            for value in snapshot.paid_component_authorities
            if type(value) is PendingProjectionPaidComponentAuthority
        }
        if len(committed) != len(snapshot.paid_component_authorities):
            raise RagFinalizationError('paid projection recovery authority is invalid')
        for requirement, _ in self._assembly.safety_requirements:
            value = committed.get(requirement.component)
            if value is None or (
                value.provider != requirement.provider
                or value.model != requirement.model
                or value.authorized_model_config_version
                != requirement.authorized_model_config_version
                or value.authorized_model_config_snapshot_hmac
                != requirement.authorized_model_config_snapshot_hmac
                or value.authorized_cost_policy_version
                != requirement.authorized_cost_policy_version
                or value.authorized_token_estimator_version
                != requirement.authorized_token_estimator_version
                or value.authorized_policy_snapshot_hmac
                != requirement.authorized_policy_snapshot_hmac
            ):
                raise RagFinalizationError(
                    'paid recovery safety authority changed from committed cost'
                )
        assembly = self._assembly
        order = begin_rag_lock_order('ordinary')
        sidecar_order = order.acquire('provider_stable_sidecar')
        safety_order = order.acquire('provider_safety_rows')
        safety_connection = assembly.safety_connection_factory()
        try:
            with assembly.provider_safety.finalization_barrier(
                safety_connection,  # type: ignore[arg-type]
                assembly.safety_requirements,
                order=order,
                sidecar_capability=sidecar_order,
                safety_capability=safety_order,
            ):
                owner_order = order.acquire('projection_owner')
                owner_capability = assembly.owner_capability_factory(
                    snapshot.agent_run_id
                )
                if (
                    type(owner_capability) is not RegisteredAdvisoryLock
                    or not owner_capability.matches(
                        rag_projection_owner_lock_id(snapshot.agent_run_id),
                        identity_namespace='dynamic',
                    )
                ):
                    raise RagFinalizationError(
                        'projection-owner capability is invalid'
                    )
                owner_connection = assembly.owner_connection_factory()
                locked = False
                try:
                    order.require(owner_order, stage='projection_owner')
                    locked = try_acquire_advisory_lock(
                        owner_connection, owner_capability, shared=False
                    )
                    if not locked:
                        raise RagFinalizationError(
                            'projection owner is still live'
                        )
                    if (
                        assembly.load_current_owner_fence(snapshot.agent_run_id)
                        != snapshot.projection_owner_fence_hmac
                    ):
                        raise RagFinalizationError(
                            'projection-owner fence changed'
                        )
                    evidence_order = order.acquire('evidence_shared_barrier')
                    c5_order = order.acquire('c5_key_corpus')
                    order.acquire('agent_run_cost')
                    order.acquire('optional_assistant')
                    order.finish()
                    with assembly.evidence_barrier.finalization_barrier(
                        order=order,
                        evidence_capability=evidence_order,
                        c5_capability=c5_order,
                    ):
                        yield
                finally:
                    try:
                        if locked:
                            release_advisory_lock(
                                owner_connection,
                                owner_capability,
                                shared=False,
                            )
                    finally:
                        owner_connection.close()  # type: ignore[attr-defined]
        except RagProviderSafetyError as exc:
            raise RagFinalizationError(
                'phase-2 provider safety changed'
            ) from exc
        finally:
            safety_connection.close()  # type: ignore[attr-defined]


def _assemble_provider_free_rag_phase2_authority(
    *,
    owner_connection_factory: Callable[[], object],
    owner_capability_factory: Callable[[int], RegisteredAdvisoryLock],
    load_current_owner_fence: Callable[[int], str],
    evidence_barrier: RagEvidenceSendBarrier,
) -> ProviderFreeRagPhase2Authority:
    return ProviderFreeRagPhase2Authority(_ProviderFreePhase2Assembly(
        owner_connection_factory=owner_connection_factory,
        owner_capability_factory=owner_capability_factory,
        load_current_owner_fence=load_current_owner_fence,
        evidence_barrier=evidence_barrier,
        _seal=_PHASE2_AUTHORITY_SEAL,
    ))


def _assemble_paid_rag_phase2_authority(
    *,
    provider_safety: RagProviderSafetyService,
    safety_connection_factory: Callable[[], object],
    safety_requirements: tuple[
        tuple[AuthorizedProviderPolicySnapshot, RagProviderSafetyBinding], ...
    ],
    owner_connection_factory: Callable[[], object],
    owner_capability_factory: Callable[[int], RegisteredAdvisoryLock],
    load_current_owner_fence: Callable[[int], str],
    evidence_barrier: RagEvidenceSendBarrier,
    load_current_readiness: Callable[[], RagServingIndexReadiness],
) -> PaidRagPhase2Authority:
    return PaidRagPhase2Authority(_PaidPhase2Assembly(
        provider_safety=provider_safety,
        safety_connection_factory=safety_connection_factory,
        safety_requirements=safety_requirements,
        owner_connection_factory=owner_connection_factory,
        owner_capability_factory=owner_capability_factory,
        load_current_owner_fence=load_current_owner_fence,
        evidence_barrier=evidence_barrier,
        load_current_readiness=load_current_readiness,
        _seal=_PHASE2_AUTHORITY_SEAL,
    ))


class RagProjectionOwnerRecoveryAuthority:
    """Concrete no-redispatch recovery across safety, owner, C.5 and cost CAS."""

    __slots__ = ('_assembly',)

    def __init__(self, assembly: object) -> None:
        if (
            type(assembly) is not _ProjectionOwnerRecoveryAssembly
            or assembly._seal is not _RECOVERY_AUTHORITY_SEAL
            or type(assembly.ledger) is not RagCostLedger
            or type(assembly.provider_free) is not ProviderFreeRagPhase2Authority
            or type(assembly.paid) is not PaidRagPhase2Authority
            or type(assembly.projection_read)
            is not ServingProjectionReadCoordinator
            or assembly.projection_read._db is not assembly.ledger._session
        ):
            raise TypeError('projection-owner recovery authority is unavailable')
        self._assembly = assembly

    def recover(self, run_id: int) -> object:
        if type(run_id) is not int or run_id <= 0:
            raise RagFinalizationError('projection-owner recovery run is invalid')
        assembly = self._assembly
        snapshot = assembly.ledger.pending_projection_recovery_snapshot(run_id)
        phase2 = assembly.paid if snapshot.paid_work_performed else assembly.provider_free
        with phase2.acquire_recovery(
            snapshot
        ), assembly.projection_read.acquire():
            tail = assembly.projection_read.lock_canonical_tail(())
            assembly.projection_read.validate_tail_context(tail)
            return assembly.ledger.recover_incomplete_run(
                run_id=run_id,
                projection_owner_fence_hmac=(
                    snapshot.projection_owner_fence_hmac
                ),
                expected_runtime_cost_snapshot_hmac=(
                    snapshot.runtime_cost_snapshot_hmac
                ),
            )


def _assemble_projection_owner_recovery_authority(
    *,
    ledger: RagCostLedger,
    provider_free: ProviderFreeRagPhase2Authority,
    paid: PaidRagPhase2Authority,
    projection_read: ServingProjectionReadCoordinator,
) -> RagProjectionOwnerRecoveryAuthority:
    return RagProjectionOwnerRecoveryAuthority(
        _ProjectionOwnerRecoveryAssembly(
            ledger=ledger,
            provider_free=provider_free,
            paid=paid,
            projection_read=projection_read,
            _seal=_RECOVERY_AUTHORITY_SEAL,
        )
    )


@dataclass(frozen=True, slots=True)
class RagProjectionPending:
    parent_agent_run_id: int
    projection_owner_fence_hmac: str
    security_scope_fingerprint: str
    prepared_corpus_generation: int
    prepared_vector_index_generation: int | None
    terminal_cost_snapshot_hmac: str

    def __post_init__(self) -> None:
        if type(self.parent_agent_run_id) is not int or self.parent_agent_run_id <= 0:
            raise ValueError('pending projection parent is invalid')
        for value in (
            self.projection_owner_fence_hmac,
            self.security_scope_fingerprint,
            self.terminal_cost_snapshot_hmac,
        ):
            if not _lower_hmac(value):
                raise ValueError('pending projection identity is invalid')
        if (
            type(self.prepared_corpus_generation) is not int
            or self.prepared_corpus_generation < 0
            or (
                self.prepared_vector_index_generation is not None
                and (
                    type(self.prepared_vector_index_generation) is not int
                    or self.prepared_vector_index_generation < 0
                )
            )
        ):
            raise ValueError('pending projection generation is invalid')


@dataclass(frozen=True, slots=True)
class CanonicalRagProjection:
    outcome: RagResultOutcome
    answer_text: str | None
    evidence: V1EvidenceProjection
    model_influence: tuple[ModelInfluenceDependencySnapshot, ...]
    hidden_match_count: int
    effective_backend: RagEffectiveBackend
    result_hmac: str


@dataclass(frozen=True, slots=True)
class AssistantProjectionTarget:
    conversation_id: int
    user_message_id: int
    owner_user_id: str

    def __post_init__(self) -> None:
        if (
            type(self.conversation_id) is not int
            or self.conversation_id <= 0
            or type(self.user_message_id) is not int
            or self.user_message_id <= 0
            or type(self.owner_user_id) is not str
            or not self.owner_user_id
        ):
            raise ValueError('assistant projection target is invalid')


@dataclass(frozen=True, slots=True)
class AssistantFinalizationRecord:
    assistant_message_id: int
    parent_agent_run_id: int
    application_outcome: AssistantPersistedOutcome
    finalization_kind: Literal['substantive', 'canned_safe', 'terminal_failure']


RagFinalProjection: TypeAlias = CanonicalRagProjection | AssistantFinalizationRecord


@dataclass(frozen=True, slots=True)
class PreparedRagFinalization:
    product_kind: Literal['answer', 'search']
    tentative_outcome: RagResultOutcome
    prepared_text: PreparedRagRequestText
    security_scope: SecurityScope
    query_embedding_result: QueryEmbeddingCallResult | None
    retrieval_result: RetrievalResult
    evidence_slots: tuple[EvidenceSlot, ...]
    model_influence_observations: tuple[PreparedModelInfluenceObservation, ...]
    selected_slot_ids: tuple[EvidenceSlotId, ...]
    validated_answer: ValidatedAnswerBlocks | None
    canned_message_identity: RagCannedMessageIdentity | None
    prepared_model_influence: PreparedModelInfluenceSet | None = None
    rendered_input_hmac: str | None = None
    answer_model_config_snapshot_hmac: str | None = None

    def __post_init__(self) -> None:
        if self.product_kind not in {'answer', 'search'}:
            raise ValueError('RAG finalization product kind is invalid')
        if type(self.prepared_text) is not PreparedRagRequestText:
            raise ValueError('prepared RAG text carrier is invalid')
        if type(self.security_scope) is not SecurityScope:
            raise ValueError('prepared RAG security scope is invalid')
        if type(self.retrieval_result) is not RetrievalResult:
            raise ValueError('prepared RAG retrieval carrier is invalid')
        for value in (
            self.evidence_slots,
            self.model_influence_observations,
            self.selected_slot_ids,
        ):
            if type(value) is not tuple:
                raise ValueError('prepared RAG carrier collections must be immutable')
        selected = set(self.selected_slot_ids)
        if len(selected) != len(self.selected_slot_ids) or not selected.issubset(
            {slot.slot_id for slot in self.evidence_slots}
        ):
            raise ValueError('prepared RAG selected slots are invalid')
        if self.model_influence_observations and (
            not _lower_hmac(self.rendered_input_hmac)
            or not _lower_hmac(self.answer_model_config_snapshot_hmac)
            or type(self.prepared_model_influence) is not PreparedModelInfluenceSet
            or self.prepared_model_influence.observations
            != self.model_influence_observations
            or self.prepared_model_influence.rendered_input_hmac
            != self.rendered_input_hmac
            or not _lower_hmac(
                self.prepared_model_influence.aggregate_observation_hmac
            )
        ):
            raise ValueError('prepared model influence authority is incomplete')
        if (
            not self.model_influence_observations
            and self.prepared_model_influence is not None
        ):
            raise ValueError('prepared model influence authority is unexpected')
        if self.validated_answer is not None and (
            self.validated_answer.selected_slot_ids != self.selected_slot_ids
            or not self.model_influence_observations
        ):
            raise ValueError('validated answer projection carrier is incomplete')
        _validate_prepared_product_contract(self)


class RagFinalizationTransactionPort(Protocol):
    def acquire_phase2(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        branch: str,
    ): ...
    def begin(self): ...
    def validate_pending(self, pending: RagProjectionPending) -> None: ...
    def current_generations(self) -> tuple[int, int | None]: ...
    def retrieve_fresh(self, request: RetrievalRequest) -> RetrievalResult: ...
    def project(
        self,
        prepared: PreparedRagFinalization,
        fresh: RetrievalResult,
        *,
        drifted: bool,
    ) -> object: ...
    def finalize_parent(
        self, pending: RagProjectionPending, projection: CanonicalRagProjection
    ) -> None: ...
    def commit(self) -> None: ...


class RagFinalizationService:
    """Own the sole product commit and return only post-commit immutable output."""

    def __init__(
        self,
        *,
        transaction_boundary: RagFinalizationTransactionPort,
        settings: Settings,
    ) -> None:
        self._boundary = transaction_boundary
        self._settings = settings

    def finalize_direct(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
    ) -> CanonicalRagProjection:
        return cast(
            CanonicalRagProjection,
            self._finalize(pending, prepared, None, branch='paid_prepared'),
        )

    def finalize_assistant(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        target: AssistantProjectionTarget,
    ) -> AssistantFinalizationRecord:
        return cast(
            AssistantFinalizationRecord,
            self._finalize(pending, prepared, target, branch='paid_prepared'),
        )

    def finalize_provider_free_safe(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection:
        self._require_safe_surface_outcome(prepared, evidence_changed=False)
        if (
            prepared.query_embedding_result is not None
            or prepared.model_influence_observations
            or prepared.validated_answer is not None
        ):
            raise RagFinalizationError('provider-free finalization has provider state')
        return self._finalize(
            pending, prepared, assistant_target, branch='provider_free'
        )

    def finalize_paid_embedding_only_safe(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection:
        self._require_safe_surface_outcome(prepared, evidence_changed=False)
        if (
            prepared.query_embedding_result is None
            or prepared.model_influence_observations
            or prepared.validated_answer is not None
        ):
            raise RagFinalizationError('paid embedding finalization lacks its receipt')
        return self._finalize(
            pending, prepared, assistant_target, branch='paid_embedding_only'
        )

    def finalize_pre_generation_evidence_changed(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection:
        self._require_safe_surface_outcome(prepared, evidence_changed=True)
        if (
            prepared.canned_message_identity
            != 'rag-canned-evidence-unavailable:v1'
            or prepared.validated_answer is not None
            or prepared.selected_slot_ids
        ):
            raise RagFinalizationError('evidence-changed finalization identity is invalid')
        return self._finalize(
            pending,
            prepared,
            assistant_target,
            branch=(
                'paid_embedding_only'
                if prepared.query_embedding_result is not None
                else 'provider_free'
            ),
            force_drift=True,
        )

    @staticmethod
    def _require_safe_surface_outcome(
        prepared: PreparedRagFinalization,
        *,
        evidence_changed: bool,
    ) -> None:
        try:
            _validate_prepared_product_contract(prepared)
        except ValueError as exc:
            raise RagFinalizationError('safe product outcome is invalid') from exc
        if evidence_changed:
            if prepared.tentative_outcome != 'evidence_unavailable':
                raise RagFinalizationError(
                    'evidence-changed finalization outcome is invalid'
                )
            return
        if (
            prepared.product_kind == 'search'
            and prepared.tentative_outcome == 'search_projected'
        ):
            return
        if (
            prepared.product_kind == 'answer'
            and prepared.tentative_outcome in _ANSWER_SAFE_NO_EVIDENCE_OUTCOMES
        ):
            return
        raise RagFinalizationError('safe product outcome is invalid')

    def finalize_inter_component_failure(self, *args, **kwargs):
        finalizer = getattr(self._boundary, 'finalize_inter_component_failure', None)
        if not callable(finalizer):
            raise RagFinalizationError('inter-component finalizer is unavailable')
        return finalizer(*args, **kwargs)

    def recover_dead_projection_owner(self, run_id: int):
        recover = getattr(self._boundary, 'recover_dead_projection_owner', None)
        if not callable(recover):
            raise RagFinalizationError('projection-owner recovery is unavailable')
        return recover(run_id)

    def _finalize(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        assistant_target: AssistantProjectionTarget | None,
        *,
        branch: Literal['provider_free', 'paid_embedding_only', 'paid_prepared'],
        force_drift: bool = False,
    ) -> RagFinalProjection:
        if type(pending) is not RagProjectionPending:
            raise TypeError('typed pending projection is required')
        if type(prepared) is not PreparedRagFinalization:
            raise TypeError('typed prepared finalization is required')
        try:
            _validate_prepared_product_contract(prepared)
        except ValueError as exc:
            raise RagFinalizationError('RAG product outcome is invalid') from exc
        if assistant_target is not None and prepared.product_kind != 'answer':
            raise RagFinalizationError(
                'assistant finalization requires an answer product'
            )
        if security_scope_fingerprint(
            prepared.security_scope, settings=self._settings
        ) != pending.security_scope_fingerprint:
            raise RagFinalizationError('prepared security scope changed')
        request = RetrievalRequest(
            retrieval_query_text=prepared.prepared_text.retrieval_query_text,
            security_scope=prepared.security_scope,
            security_scope_fingerprint=pending.security_scope_fingerprint,
            query_embedding_result=prepared.query_embedding_result,
            candidate_scan_limit=50,
            visible_limit=5 if prepared.product_kind == 'search' else 8,
            relevance_policy_version='rag-retrieval-policy:v2.0',
        )
        if prepared.query_embedding_result is not None:
            try:
                validated_embedding = validate_query_embedding_call_result(
                    request,
                    settings=self._settings,
                )
            except (TypeError, ValueError) as exc:
                raise RagFinalizationError(
                    'paid query embedding carrier is invalid'
                ) from exc
            if validated_embedding is not prepared.query_embedding_result:
                raise RagFinalizationError(
                    'paid query embedding carrier is invalid'
                )
        committed: RagFinalProjection | None = None
        try:
            with self._boundary.acquire_phase2(
                pending, prepared, branch=branch
            ), self._boundary.begin():
                acquire_prefix = getattr(
                    self._boundary, 'acquire_projection_prefix', None
                )
                if callable(acquire_prefix):
                    acquire_prefix()
                current_corpus, current_index = (
                    self._boundary.current_generations()
                )
                current_index_authority = (
                    current_index
                    if prepared.retrieval_result.configured_backend == 'pgvector'
                    else None
                )
                fresh = self._boundary.retrieve_fresh(request)
                drifted = force_drift or (
                    current_corpus != pending.prepared_corpus_generation
                    or current_index_authority
                    != pending.prepared_vector_index_generation
                    or not _same_retrieval_authority(
                        fresh, prepared.retrieval_result
                    )
                )
                candidate = self._boundary.project(
                    prepared, fresh, drifted=drifted
                )
                projection = _canonical(candidate)
                self._boundary.validate_pending(pending)
                validate_prepared = getattr(
                    self._boundary, 'validate_prepared', None
                )
                if callable(validate_prepared):
                    validate_prepared(pending, prepared)
                validate_branch_costs = getattr(
                    self._boundary, 'validate_branch_costs', None
                )
                if not callable(validate_branch_costs):
                    raise RagFinalizationError(
                        'branch-specific cost authority is unavailable'
                    )
                validate_branch_costs(prepared, branch=branch)
                if assistant_target is None:
                    self._boundary.finalize_parent(pending, projection)
                    committed = projection
                else:
                    finalize_assistant = getattr(
                        self._boundary, 'finalize_assistant_product', None
                    )
                    if not callable(finalize_assistant):
                        raise RagFinalizationError(
                            'assistant product finalizer is unavailable'
                        )
                    committed = finalize_assistant(
                        pending, projection, assistant_target
                    )
                self._boundary.commit()
        except RagFinalizationError:
            raise
        except Exception as exc:
            raise RagFinalizationError(
                'RAG final projection commit failed; no product is available'
            ) from exc
        if committed is None:
            raise RagFinalizationError('RAG final projection was not committed')
        return committed


class SqlAlchemyRagFinalizationBoundary:
    """PostgreSQL product transaction; no route is allowed to reconstruct it."""

    def __init__(
        self,
        *,
        db: Session,
        settings: Settings,
        retriever: object,
        assistant_writer: AssistantEvidenceWriter | None = None,
        recovery_authority: RagProjectionOwnerRecoveryAuthority | None = None,
        phase2_authority: object | None = None,
    ) -> None:
        if db.get_bind().dialect.name != 'postgresql':
            raise TypeError('production RAG finalization requires PostgreSQL')
        if not callable(getattr(retriever, 'invoke', None)):
            raise TypeError('RAG finalization retriever is unavailable')
        if type(phase2_authority) not in {
            ProviderFreeRagPhase2Authority,
            PaidRagPhase2Authority,
        }:
            raise TypeError('concrete RAG phase-2 authority is required')
        if phase2_authority.evidence_barrier_is_postgresql is not True:
            raise TypeError(
                'PostgreSQL evidence barrier requires registered DB authority'
            )
        self._db = db
        self._settings = settings
        self._retriever = retriever
        self._writer = assistant_writer
        if (
            recovery_authority is not None
            and type(recovery_authority) is not RagProjectionOwnerRecoveryAuthority
        ):
            raise TypeError('concrete projection-owner recovery is required')
        self._recovery_authority = recovery_authority
        self._phase2_authority = phase2_authority
        self._prefix: AbstractContextManager | None = None
        self._projection_coordinator: ServingProjectionReadCoordinator | None = None
        self._projection_tail: ServingProjectionTailLockedContext | None = None
        self._prefix_generations: tuple[int, int | None] | None = None
        self._pending_parent: AgentRun | None = None
        self._pending_children: tuple[AgentRunCostComponent, ...] = ()
        self._pending: RagProjectionPending | None = None
        self._assembled_answer_hmac: str | None = None
        self._committed = False
        self._secret, _ = fingerprint_secret_bytes(settings)

    def begin(self) -> AbstractContextManager:
        return _SessionFinalizationTransaction(self)

    def acquire_phase2(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
        *,
        branch: str,
    ) -> AbstractContextManager:
        acquire = getattr(self._phase2_authority, 'acquire', None)
        if not callable(acquire):
            raise RagFinalizationError('phase-2 finalization authority is unavailable')
        self._pending = pending
        return acquire(pending, prepared, branch=branch)

    def acquire_projection_prefix(self) -> None:
        if self._prefix is not None:
            raise RagFinalizationError('projection prefix was already acquired')
        coordinator = ServingProjectionReadCoordinator(
            db=self._db, settings=self._settings
        )
        prefix = coordinator.acquire()
        generations = prefix.__enter__()
        self._projection_coordinator = coordinator
        self._prefix = prefix
        self._prefix_generations = generations

    def validate_pending(self, pending: RagProjectionPending) -> None:
        if self._prefix_generations is None:
            raise RagFinalizationError('projection prefix is unavailable')
        parent = self._db.scalar(
            select(AgentRun)
            .where(AgentRun.id == pending.parent_agent_run_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        children = tuple(
            self._db.scalars(
                select(AgentRunCostComponent)
                .where(
                    AgentRunCostComponent.agent_run_id
                    == pending.parent_agent_run_id
                )
                .order_by(AgentRunCostComponent.component_ordinal)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        if (
            parent is None
            or parent.run_contract_version != 'rag-run:v2'
            or parent.status != 'running'
            or parent.run_record_phase != 'cost_finalized_pending_projection'
            or parent.completed_at is not None
            or parent.projection_owner_fence_hmac
            != pending.projection_owner_fence_hmac
            or parent.metadata_.get('security_scope_fingerprint')
            != pending.security_scope_fingerprint
            or parent.metadata_.get('runtime_cost_snapshot_hmac')
            != pending.terminal_cost_snapshot_hmac
            or _pre_projection_cost_snapshot_hmac(
                agent_run_id=pending.parent_agent_run_id,
                children=children,
                secret=self._secret,
            )
            != pending.terminal_cost_snapshot_hmac
            or tuple(value.component for value in children)
            != ('query_embedding', 'answer_generation')
            or any(value.dispatch_state != 'terminal' for value in children)
            or sum(
                (Decimal(value.charged_cost_usd) for value in children),
                Decimal('0.000000'),
            )
            != Decimal(parent.total_charged_cost_usd)
        ):
            raise RagFinalizationError('pending RAG projection changed')
        self._pending_parent = parent
        self._pending_children = children
        self._pending = pending

    def validate_branch_costs(
        self,
        prepared: PreparedRagFinalization,
        *,
        branch: str,
    ) -> None:
        children = self._pending_children
        if tuple(value.component for value in children) != (
            'query_embedding',
            'answer_generation',
        ):
            raise RagFinalizationError('exact-two RAG cost authority is unavailable')
        query, answer = children
        if branch != 'provider_free':
            if type(self._phase2_authority) is not PaidRagPhase2Authority:
                raise RagFinalizationError(
                    'paid phase-2 safety authority is unavailable'
                )
            self._phase2_authority.validate_locked_cost_authority(
                children,
                expected_components=_expected_paid_components(
                    prepared,
                    branch=branch,
                ),
            )
        if branch == 'provider_free':
            if not all(_exact_terminal_zero(value) for value in children):
                raise RagFinalizationError('provider-free cost shape is invalid')
            return
        embedding = prepared.query_embedding_result
        if embedding is None:
            if not _exact_terminal_zero(query):
                raise RagFinalizationError('query embedding cost shape is invalid')
        elif not _exact_paid_embedding_cost(query, embedding):
            raise RagFinalizationError('paid embedding cost shape is invalid')
        if branch == 'paid_embedding_only':
            if embedding is None or not _exact_terminal_zero(answer):
                raise RagFinalizationError(
                    'paid embedding-only cost shape is invalid'
                )
            return
        if branch != 'paid_prepared':
            raise RagFinalizationError('RAG finalization cost branch is invalid')
        generated = bool(prepared.model_influence_observations)
        if generated:
            if (
                answer.dispatch_state != 'terminal'
                or answer.attempted is not True
                or answer.dispatch_count != 1
                or answer.charge_basis != 'actual'
                or answer.actual_input_tokens is None
                or answer.actual_output_tokens is None
                or not _lower_hmac(answer.dispatch_fence_hmac)
                or not _lower_hmac(answer.process_instance_hmac)
                or answer.terminal_outcome != 'component_succeeded'
                or answer.authorized_model_config_snapshot_hmac
                != prepared.answer_model_config_snapshot_hmac
            ):
                raise RagFinalizationError('answer generation cost shape is invalid')
        elif not _exact_terminal_zero(answer):
            raise RagFinalizationError('answer generation cost shape is invalid')

    def current_generations(self) -> tuple[int, int | None]:
        if self._prefix_generations is None:
            raise RagFinalizationError('projection generations are unavailable')
        return self._prefix_generations

    def validate_prepared(
        self,
        pending: RagProjectionPending,
        prepared: PreparedRagFinalization,
    ) -> None:
        parent = self._require_parent(pending)
        metadata = parent.metadata_ or {}
        expected_surface = (
            'search'
            if prepared.product_kind == 'search'
            else prepared.prepared_text.query_context_version
            == 'assistant-context:v1'
            and 'assistant'
            or 'ask'
        )
        if (
            metadata.get('current_text_hmac')
            != prepared.prepared_text.current_text_hmac
            or metadata.get('retrieval_query_hmac')
            != prepared.prepared_text.retrieval_query_hmac
            or metadata.get('configured_backend')
            != prepared.retrieval_result.configured_backend
            or metadata.get('surface') != expected_surface
            or metadata.get('rendered_input_hmac') != prepared.rendered_input_hmac
            or metadata.get('answer_model_config_snapshot_hmac')
            != prepared.answer_model_config_snapshot_hmac
            or metadata.get('prepared_model_influence_observation_hmac')
            != (
                prepared.prepared_model_influence.aggregate_observation_hmac
                if prepared.prepared_model_influence is not None
                else None
            )
        ):
            raise RagFinalizationError('prepared RAG carrier changed')

    def retrieve_fresh(self, request: RetrievalRequest) -> RetrievalResult:
        result = self._retriever.invoke(request)
        if type(result) is not RetrievalResult:
            raise RagFinalizationError('fresh retriever result is invalid')
        return result

    def project(
        self,
        prepared: PreparedRagFinalization,
        fresh: RetrievalResult,
        *,
        drifted: bool,
    ) -> CanonicalRagProjection:
        if self._prefix_generations is None:
            raise RagFinalizationError('projection prefix is unavailable')
        if self._projection_coordinator is None:
            raise RagFinalizationError('projection canonical lock authority is unavailable')
        document_ids = tuple(
            candidate.evidence.serving_document_id
            for candidate in fresh.visible
        )
        self._projection_tail = self._projection_coordinator.lock_canonical_tail(
            document_ids
        )
        self._projection_coordinator.validate_tail_context(self._projection_tail)
        current_corpus, current_index = self._prefix_generations
        slots = rank_evidence_slots(fresh.visible)
        hidden_hmac = _hidden_membership_hmac(fresh, secret=self._secret)
        prepared_hidden_hmac = _hidden_membership_hmac(
            prepared.retrieval_result, secret=self._secret
        )
        if self._pending is None:
            raise RagFinalizationError('pending projection is unavailable')
        fence = ProjectionFence(
            prepared_corpus_generation=self._pending.prepared_corpus_generation,
            prepared_index_generation=(
                prepared.query_embedding_result.prepared.vector_index_generation
                if prepared.query_embedding_result is not None
                else None
            ),
            current_corpus_generation=current_corpus,
            current_index_generation=(
                current_index if prepared.query_embedding_result is not None else None
            ),
            prepared_readiness_hmac=(
                prepared.query_embedding_result.prepared.readiness_snapshot_hmac
                if prepared.query_embedding_result is not None
                else None
            ),
            current_readiness_hmac=(
                getattr(
                    self._phase2_authority,
                    'current_readiness_hmac',
                    None,
                )
                if prepared.query_embedding_result is not None
                else None
            ),
            prepared_hidden_membership_hmac=prepared_hidden_hmac,
            current_hidden_membership_hmac=hidden_hmac,
        )
        projector = CanonicalEvidenceProjector(
            db=self._db, settings=self._settings
        )
        substantive = bool(
            not drifted
            and prepared.product_kind == 'answer'
            and prepared.validated_answer is not None
            and prepared.validated_answer.blocks
        )
        if prepared.product_kind == 'search':
            if drifted:
                evidence = _empty_search_projection(self._settings)
                outcome: RagResultOutcome = 'evidence_unavailable'
            else:
                evidence = projector.project_search(
                    slots[:5], scope=prepared.security_scope, fence=fence
                )
                outcome = 'search_projected'
            answer_text = None
            dependencies: tuple[ModelInfluenceDependencySnapshot, ...] = ()
        elif substantive:
            selected = prepared.validated_answer.selected_slot_ids
            evidence = projector.project_selected(
                slots,
                selected_slot_ids=selected,
                scope=prepared.security_scope,
                fence=fence,
            )
            dependencies = projector.finalize_model_influence_dependencies(
                prepared.prepared_model_influence,
                selected,
                scope=prepared.security_scope,
                fence=fence,
            )
            if (
                not dependencies
                or evidence.projection_hmac
                == build_v1_selected_evidence_projection_hmac(
                    citation_hmacs=(),
                    source_ids=(),
                    source_links=(),
                    source_snippets=(),
                    settings=self._settings,
                )
            ):
                substantive = False
            if substantive:
                outcome = 'supported'
                answer_text = prepared.validated_answer.assembled_answer
                self._assembled_answer_hmac = (
                    prepared.validated_answer.assembled_answer_hmac
                )
            else:
                outcome = 'evidence_unavailable'
                answer_text = _canned_text('rag-canned-evidence-unavailable:v1')
                evidence = _empty_answer_projection(self._settings)
                dependencies = ()
        else:
            outcome = (
                'evidence_unavailable' if drifted else prepared.tentative_outcome
            )
            answer_text = _canned_text(
                'rag-canned-evidence-unavailable:v1'
                if drifted
                else prepared.canned_message_identity
            )
            evidence = _empty_answer_projection(self._settings)
            dependencies = ()
        result_hmac = _build_result_hmac(
            pending=self._pending,
            prepared=prepared,
            outcome=outcome,
            evidence=evidence,
            dependencies=dependencies,
            hidden_hmac=hidden_hmac,
            effective_backend=fresh.effective_backend,
            secret=self._secret,
            settings=self._settings,
        )
        return CanonicalRagProjection(
            outcome=outcome,
            answer_text=answer_text,
            evidence=evidence,
            model_influence=dependencies,
            hidden_match_count=(
                0 if outcome == 'evidence_unavailable' else fresh.hidden_match_count
            ),
            effective_backend=fresh.effective_backend,
            result_hmac=result_hmac,
        )

    def finalize_parent(
        self, pending: RagProjectionPending, projection: CanonicalRagProjection
    ) -> None:
        parent = self._require_parent(pending)
        _apply_parent_final(parent, projection, secret=self._secret)
        self._db.flush([parent])

    def finalize_assistant_product(
        self,
        pending: RagProjectionPending,
        projection: CanonicalRagProjection,
        target: AssistantProjectionTarget,
    ) -> AssistantFinalizationRecord:
        if self._writer is None:
            raise RagFinalizationError('assistant evidence writer is unavailable')
        conversation = self._db.scalar(
            select(AssistantConversation)
            .where(AssistantConversation.id == target.conversation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        user_message = self._db.scalar(
            select(AssistantMessage)
            .where(AssistantMessage.id == target.user_message_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if (
            conversation is None
            or user_message is None
            or conversation.user_id != target.owner_user_id
            or user_message.conversation_id != conversation.id
            or user_message.role != 'user'
        ):
            raise RagFinalizationError('assistant projection target changed')
        canned = (
            None
            if projection.model_influence
            else (
                'rag-canned-evidence-unavailable:v1'
                if projection.outcome == 'evidence_unavailable'
                else 'rag-canned-no-evidence:v1'
            )
        )
        message_projection = assistant_message_projection(
            projection,
            metadata={'status': projection.outcome},
            canned_message_identity=canned,
            assembled_answer_hmac=(
                self._assembled_answer_hmac
                if projection.model_influence
                else None
            ),
            permission_level=_output_permission(projection),
            permission_notice=(
                'evidence_unavailable'
                if projection.outcome == 'evidence_unavailable'
                else None
            ),
        )
        self.finalize_parent(pending, projection)
        parent = self._require_parent(pending)
        authority = _mint_assistant_exact_write_authority(
            parent_agent_run_id=parent.id,
            conversation_id=conversation.id,
            projection=message_projection,
            parent_result_hmac=parent.metadata_.get('rag_result_hmac'),
        )
        message = self._writer.append_final(
            db=self._db,
            conversation=conversation,
            projection=message_projection,
            authority=authority,
        )
        return AssistantFinalizationRecord(
            assistant_message_id=message.id,
            parent_agent_run_id=pending.parent_agent_run_id,
            application_outcome=cast(AssistantPersistedOutcome, projection.outcome),
            finalization_kind=(
                'substantive' if projection.model_influence else 'canned_safe'
            ),
        )

    def commit(self) -> None:
        if self._pending_parent is None or self._committed:
            raise RagFinalizationError('final projection commit is unavailable')
        self._db.commit()
        self._committed = True

    def recover_dead_projection_owner(self, run_id: int):
        if type(run_id) is not int or run_id <= 0:
            raise ValueError('projection-owner recovery run id is invalid')
        if (
            type(self._recovery_authority)
            is not RagProjectionOwnerRecoveryAuthority
        ):
            raise RagFinalizationError('projection-owner recovery is unavailable')
        # No time/process probe or provider retry is accepted. The concrete
        # authority owns nonblocking owner reacquisition and the cost CAS.
        return self._recovery_authority.recover(run_id)

    def _require_parent(self, pending: RagProjectionPending) -> AgentRun:
        if (
            self._pending_parent is None
            or self._pending_parent.id != pending.parent_agent_run_id
        ):
            raise RagFinalizationError('pending projection parent is unavailable')
        return self._pending_parent


class _SessionFinalizationTransaction:
    def __init__(self, boundary: SqlAlchemyRagFinalizationBoundary) -> None:
        self._boundary = boundary

    def __enter__(self) -> _SessionFinalizationTransaction:
        db = self._boundary._db
        if db.in_transaction():
            raise RagFinalizationError('final projection requires a fresh transaction')
        db.begin()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        boundary = self._boundary
        try:
            if boundary._prefix is not None:
                boundary._prefix.__exit__(exc_type, exc, traceback)
        finally:
            if boundary._db.in_transaction():
                boundary._db.rollback()
            boundary._prefix = None
            boundary._prefix_generations = None
            boundary._projection_coordinator = None
            boundary._projection_tail = None
        return False


def _hidden_membership_hmac(result: RetrievalResult, *, secret: bytes) -> str:
    return keyed_fingerprint(
        {
            'hidden_count_capped': result.hidden_count_capped,
            'public_hidden_match_count': result.hidden_match_count,
            'top_candidate_window_hmac': result.top_candidate_window_hmac,
        },
        secret=secret,
        schema_version='rag-hidden-membership:v1',
        policy_version='rag-retrieval-bounds:v1',
    )


def _same_retrieval_authority(left: RetrievalResult, right: RetrievalResult) -> bool:
    return bool(
        type(left) is RetrievalResult
        and type(right) is RetrievalResult
        and left.configured_backend == right.configured_backend
        and left.effective_backend == right.effective_backend
        and left.visible == right.visible
        and left.hidden_match_count == right.hidden_match_count
        and left.hidden_count_capped == right.hidden_count_capped
        and left.top_candidate_window_hmac == right.top_candidate_window_hmac
        and left.query_embedding_receipt == right.query_embedding_receipt
    )


def _empty_answer_projection(settings: Settings) -> V1EvidenceProjection:
    return V1EvidenceProjection(
        citations=(),
        source_ids=(),
        source_links=(),
        source_snippets=(),
        search_results=(),
        projection_hmac=build_v1_selected_evidence_projection_hmac(
            citation_hmacs=(),
            source_ids=(),
            source_links=(),
            source_snippets=(),
            settings=settings,
        ),
    )


def _empty_search_projection(settings: Settings) -> V1EvidenceProjection:
    return V1EvidenceProjection(
        citations=(),
        source_ids=(),
        source_links=(),
        source_snippets=(),
        search_results=(),
        projection_hmac=build_v1_search_result_set_projection_hmac(
            (), settings=settings
        ),
    )


def _canned_text(identity: RagCannedMessageIdentity | None) -> str:
    values = {
        'rag-canned-no-evidence:v1': '질문에 답할 수 있는 근거를 찾지 못했습니다.',
        'rag-canned-evidence-unavailable:v1': (
            '근거가 변경되어 안전하게 답변할 수 없습니다. 다시 시도해 주세요.'
        ),
        'rag-canned-budget-failure:v1': '현재 비용 한도 내에서 답변할 수 없습니다.',
        'rag-canned-generation-failure:v1': '답변을 생성하지 못했습니다. 다시 시도해 주세요.',
    }
    try:
        return values[identity]
    except KeyError:
        raise RagFinalizationError('canned RAG message identity is unavailable') from None


def _validate_prepared_product_contract(
    prepared: PreparedRagFinalization,
) -> None:
    outcome = prepared.tentative_outcome
    if outcome not in _RAG_PRODUCT_OUTCOMES:
        raise ValueError('prepared RAG product outcome is invalid')
    if prepared.product_kind == 'search':
        if outcome != 'search_projected' or prepared.canned_message_identity is not None:
            raise ValueError('prepared search product canned identity is invalid')
        return
    if outcome == 'supported':
        if (
            prepared.validated_answer is None
            or prepared.canned_message_identity is not None
        ):
            raise ValueError('prepared supported product canned identity is invalid')
        return
    if outcome == 'evidence_unavailable':
        expected = 'rag-canned-evidence-unavailable:v1'
    else:
        expected = 'rag-canned-no-evidence:v1'
    if (
        prepared.validated_answer is not None
        or prepared.canned_message_identity != expected
    ):
        raise ValueError('prepared answer product canned identity is invalid')


def _build_result_hmac(
    *,
    pending: RagProjectionPending,
    prepared: PreparedRagFinalization,
    outcome: RagResultOutcome,
    evidence: V1EvidenceProjection,
    dependencies: tuple[ModelInfluenceDependencySnapshot, ...],
    hidden_hmac: str,
    effective_backend: RagEffectiveBackend,
    secret: bytes,
    settings: Settings,
) -> str:
    search = prepared.product_kind == 'search'
    substantive = bool(dependencies)
    answer_prepared = (
        None if search else bool(prepared.model_influence_observations)
    )
    canned = None
    if not search and not substantive:
        canned = (
            'rag-canned-evidence-unavailable:v1'
            if outcome == 'evidence_unavailable'
            else prepared.canned_message_identity
        )
    selected_hmac = evidence.projection_hmac if substantive else None
    model_set_hmac = (
        build_model_influence_set_hmac(
            dependencies=dependencies, settings=settings
        )
        if dependencies
        else None
    )
    validated = prepared.validated_answer
    payload = {
        'answer_block_audit_set_hmac': (
            validated.answer_block_audit_set_hmac
            if substantive and validated is not None
            else None
        ),
        'answer_invocation_prepared': answer_prepared,
        'answer_output_schema_hmac': (
            None if search else build_answer_output_schema_hmac(settings)
        ),
        'answer_prompt_renderer_hmac': (
            None if search else build_answer_prompt_renderer_hmac(settings)
        ),
        'answer_question_hmac': (
            None if search else prepared.prepared_text.answer_question_hmac
        ),
        'assembled_answer_hmac': (
            validated.assembled_answer_hmac
            if substantive and validated is not None
            else None
        ),
        'canned_message_identity': canned,
        'citation_projection_version': 'rag-citation-projection:v1',
        'configured_backend': prepared.retrieval_result.configured_backend,
        'current_text_hmac': prepared.prepared_text.current_text_hmac,
        'effective_backend': effective_backend,
        'fallback_category': prepared.retrieval_result.trace.fallback_category,
        'graph_version': 'company-memory-rag-answer-v2.0',
        'hidden_membership_hmac': hidden_hmac,
        'joiner_version': 'rag-answer-block-joiner:v1' if substantive else None,
        'model_config_snapshot_hmac': prepared.answer_model_config_snapshot_hmac,
        'model_influence_set_hmac': model_set_hmac,
        'outcome': outcome,
        'output_permission': _dependency_permission(dependencies),
        'permission_fingerprint': keyed_fingerprint(
            {
                'allowed_permission_levels': list(
                    prepared.security_scope.allowed_permission_levels
                ),
                'permission_policy_version': (
                    prepared.security_scope.permission_policy_version
                ),
                'resource_scope_mode': prepared.security_scope.resource_scope_mode,
                'workspace_scope_id': prepared.security_scope.workspace_scope_id,
            },
            secret=secret,
            schema_version='rag-permission-fingerprint:v1',
            policy_version='rag-permission-policy:v1',
        ),
        'prepared_model_influence_observation_hmac': (
            prepared.prepared_model_influence.aggregate_observation_hmac
            if prepared.prepared_model_influence is not None
            else None
        ),
        'prompt_version': None if search else 'rag-answer:v2',
        'rendered_input_hmac': prepared.rendered_input_hmac,
        'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
        'retrieval_query_context_version': prepared.prepared_text.query_context_version,
        'retrieval_query_hmac': prepared.prepared_text.retrieval_query_hmac,
        'security_scope_fingerprint': pending.security_scope_fingerprint,
        'selected_evidence_projection_hmac': selected_hmac,
        'search_result_set_projection_hmac': (
            evidence.projection_hmac if search else None
        ),
        'surface': (
            'search'
            if search
            else (
                'assistant'
                if prepared.prepared_text.query_context_version
                == 'assistant-context:v1'
                else 'ask'
            )
        ),
    }
    return keyed_fingerprint(
        payload,
        secret=secret,
        schema_version='rag-result:v1',
        policy_version='company-memory-rag-answer-v2.0',
    )


def _apply_parent_final(
    parent: AgentRun,
    projection: CanonicalRagProjection,
    *,
    secret: bytes,
) -> None:
    if projection.outcome not in _RAG_PRODUCT_OUTCOMES:
        raise RagFinalizationError(
            'terminal failure requires the explicit failure finalizer'
        )
    parts = parent.source_window.split(':')
    if len(parts) != 5 or parts[:2] != ['rag-v2', 'admission']:
        raise RagFinalizationError('RAG parent source window is invalid')
    surface, backend = parts[3], parts[4]
    admission_hmac = parent.cache_key.removeprefix('rag-v2-admission:')
    if not _lower_hmac(admission_hmac):
        raise RagFinalizationError('RAG admission identity is invalid')
    runtime_hmac = keyed_fingerprint(
        {
            'prior_runtime_cost_snapshot_hmac': parent.metadata_.get(
                'runtime_cost_snapshot_hmac'
            ),
            'rag_result_hmac': projection.result_hmac,
            'snapshot_stage': 'terminal_projection',
            'total_charged_cost_usd': format(
                Decimal(parent.total_charged_cost_usd), 'f'
            ),
        },
        secret=secret,
        schema_version='rag-runtime-cost-terminal-projection:v1',
        policy_version='rag-run:v2',
    )
    final_identity = final_product_identity(
        FinalProductIdentityInput(
            admission_cache_identity_hmac=admission_hmac,
            rag_result_hmac=projection.result_hmac,
            surface=cast(Literal['ask', 'search', 'assistant'], surface),
        ),
        secret=secret,
    )
    parent.status = 'complete'
    parent.run_record_phase = 'final'
    parent.source_window = terminal_source_window(
        stage='product', surface=surface, backend=backend
    )
    parent.cache_key = 'rag-v2-final:' + final_identity
    parent.projection_owner_fence_hmac = None
    parent.completed_at = datetime.now(UTC)
    parent.metadata_ = {
        **parent.metadata_,
        'effective_backend': projection.effective_backend,
        'hidden_match_count': projection.hidden_match_count,
        'outcome': projection.outcome,
        'rag_result_hmac': projection.result_hmac,
        'runtime_cost_snapshot_hmac': runtime_hmac,
        'terminal_identity_hmac': final_identity,
    }


def _dependency_permission(
    dependencies: tuple[ModelInfluenceDependencySnapshot, ...]
) -> str | None:
    rank = {'public': 0, 'internal': 1, 'restricted': 2}
    if not dependencies:
        return None
    values = [value.observation.effective_permission for value in dependencies]
    if any(value not in rank for value in values):
        raise RagFinalizationError('RAG dependency permission is invalid')
    return max(values, key=rank.__getitem__)


def _output_permission(projection: CanonicalRagProjection) -> str | None:
    return _dependency_permission(projection.model_influence)


def _canonical(value: object) -> CanonicalRagProjection:
    try:
        projection = CanonicalRagProjection(
            outcome=value.outcome,
            answer_text=value.answer_text,
            evidence=value.evidence,
            model_influence=value.model_influence,
            hidden_match_count=value.hidden_match_count,
            effective_backend=value.effective_backend,
            result_hmac=value.result_hmac,
        )
    except Exception:
        raise RagFinalizationError('canonical RAG projection is invalid') from None
    if (
        type(projection.evidence) is not V1EvidenceProjection
        or type(projection.model_influence) is not tuple
        or type(projection.hidden_match_count) is not int
        or projection.hidden_match_count < 0
        or not _lower_hmac(projection.result_hmac)
    ):
        raise RagFinalizationError('canonical RAG projection is invalid')
    return projection


def assistant_message_projection(
    projection: CanonicalRagProjection,
    *,
    metadata: dict[str, object],
    canned_message_identity: RagCannedMessageIdentity | None,
    assembled_answer_hmac: str | None,
    permission_level: str | None,
    permission_notice: str | None,
) -> AssistantMessageProjection:
    if projection.answer_text is None:
        raise RagFinalizationError('assistant projection content is unavailable')
    return AssistantMessageProjection(
        content=projection.answer_text,
        metadata=metadata,
        evidence=projection.evidence,
        permission_level=permission_level,
        permission_notice=permission_notice,
        hidden_match_count=projection.hidden_match_count,
        assembled_answer_hmac=assembled_answer_hmac,
        canned_message_identity=canned_message_identity,
        result_hmac=projection.result_hmac,
        model_influence=projection.model_influence,
    )


def _lower_hmac(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _pre_projection_cost_snapshot_hmac(
    *,
    agent_run_id: int,
    children: tuple[AgentRunCostComponent, ...],
    secret: bytes,
) -> str:
    components = [
        {
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
        for row in children
    ]
    total_reserved = sum(
        (Decimal(item['reserved_cost_usd']) for item in components),
        Decimal('0.000000'),
    )
    total_charged = sum(
        (Decimal(item['charged_cost_usd']) for item in components),
        Decimal('0.000000'),
    )
    return runtime_cost_identity(
        {
            'agent_run_id': agent_run_id,
            'components': components,
            'parent_outcome': None,
            'parent_run_record_phase': 'cost_finalized_pending_projection',
            'parent_status': 'running',
            'run_contract_version': 'rag-run:v2',
            'snapshot_stage': 'pre_projection',
            'total_charged_cost_usd': format(total_charged, '.6f'),
            'total_reserved_cost_usd': format(total_reserved, '.6f'),
        },
        secret=secret,
    )


def _exact_terminal_zero(row: AgentRunCostComponent) -> bool:
    return bool(
        row.dispatch_state == 'terminal'
        and row.attempted is False
        and row.dispatch_count == 0
        and row.reserved_input_tokens == 0
        and row.reserved_output_tokens == 0
        and row.actual_input_tokens is None
        and row.actual_output_tokens is None
        and Decimal(row.reserved_cost_usd) == Decimal('0.000000')
        and Decimal(row.charged_cost_usd) == Decimal('0.000000')
        and row.charge_basis == 'zero'
        and row.overrun is False
        and row.dispatch_fence_hmac is None
        and row.process_instance_hmac is None
        and row.terminal_outcome is None
    )


def _exact_paid_embedding_cost(
    row: AgentRunCostComponent,
    result: QueryEmbeddingCallResult,
) -> bool:
    prepared = result.prepared
    return bool(
        row.dispatch_state == 'terminal'
        and row.attempted is True
        and row.dispatch_count == 1
        and row.actual_input_tokens == result.validated_input_tokens
        and row.actual_output_tokens == 0
        and Decimal(row.charged_cost_usd) == result.actual_cost_usd
        and row.charge_basis == 'actual'
        and row.overrun is False
        and row.dispatch_fence_hmac == prepared.attempt_fence_hmac
        and _lower_hmac(row.process_instance_hmac)
        and row.terminal_outcome == 'component_succeeded'
        and row.authorized_model_config_snapshot_hmac
        == prepared.model_config_snapshot_hmac
        and row.authorized_policy_snapshot_hmac
        == prepared.provider_policy_snapshot_hmac
    )


def _expected_paid_components(
    prepared: PreparedRagFinalization,
    *,
    branch: str,
) -> tuple[str, ...]:
    if branch == 'paid_embedding_only':
        return ('query_embedding',)
    if branch != 'paid_prepared':
        raise RagFinalizationError('paid phase-2 branch is invalid')
    return (
        ('query_embedding', 'answer_generation')
        if prepared.query_embedding_result is not None
        else ('answer_generation',)
    )
