from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Literal, NotRequired, Protocol, TypedDict

from sqlalchemy.orm import Session

from backend.app.agent_runtime.provider_usage import RagResultOutcome
from backend.app.agent_runtime.rag_cost_ledger import RagCostLedger
from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
from backend.app.agent_runtime.rag_finalization import (
    AssistantFinalizationRecord,
    AssistantProjectionTarget,
    CanonicalRagProjection,
    PreparedRagFinalization,
    RagFinalizationService,
    RagProjectionPending,
)
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderDispatchAuthority,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    RagComponentFinal,
    RagPaidComponent,
)
from backend.app.agent_runtime.rag_sqlite_smoke import SQLiteRagGraphScope
from backend.app.agent_runtime.rag_v2_contracts import RagEffectiveBackend, RagSurface
from backend.app.agent_runtime.rag_v2_identity import (
    RagSecurityScopeResolver,
    SecurityScope,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    PreparedAnswerInvocation,
    StructuredRagAnswerModel,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ValidatedAnswerBlocks,
)
from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
    StrictQueryEmbeddingAdapter,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import PreparedRagRequestText
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.rag.evidence_projection import (
    ModelInfluenceDependencySnapshot,
    PreparedModelInfluenceSet,
    V1EvidenceProjection,
)
from backend.app.rag.index_readiness import RagV2ServingIndexReadinessService
from backend.app.rag.retrieval import (
    EvidenceSlot,
    EvidenceSlotId,
    PreparedQueryEmbedding,
    QueryEmbeddingCallResult,
    RagRetrieverRegistry,
    RetrievalCandidate,
    RetrievalResult,
)
from backend.app.rag.trusted_evidence import ServingEvidenceResolver

RagFallbackCategory = Literal[
    'graph_unavailable',
    'graph_no_benefit',
    'graph_invalid',
    'pgvector_storage_runtime_failure',
    'serving_corpus_changed_during_pgvector_query',
    'serving_corpus_changed_during_answer_revalidation',
    'pgvector_hidden_recompute_runtime_failure',
]


@dataclass(frozen=True, slots=True)
class RagRunTrace:
    node_counts: Mapping[str, int]
    node_latency_ms: Mapping[str, int]
    provider_attempt_counts: Mapping[RagPaidComponent, int]
    bounded_candidate_counts: Mapping[str, int]
    domain_hmacs: Mapping[str, str]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class SanitizedRagToolEvent:
    event_kind: str
    outcome: str
    latency_ms: int
    domain_hmac: str


class SafeRagRunRecorder(Protocol):
    def record(self, trace: RagRunTrace) -> None: ...


class SafeRagToolRecorder(Protocol):
    def record(self, event: SanitizedRagToolEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class RagRequestServices:
    """Created and closed by one request factory; never attached to a graph."""

    db: Session
    security_scope_resolver: RagSecurityScopeResolver
    retrievers: RagRetrieverRegistry
    evidence_resolver: ServingEvidenceResolver
    cost_policy: RagCostPolicy
    cost_ledger: RagCostLedger | None
    policy_snapshots: tuple[AuthorizedProviderPolicySnapshot, ...]
    allocate_run_id: Callable[[], int] | None
    load_generations: Callable[[], tuple[int, int | None]] | None
    finalizer_factory: (
        Callable[
            [RagProjectionPending, PreparedRagFinalization], RagFinalizationService
        ]
        | None
    )
    query_embedding_adapter: StrictQueryEmbeddingAdapter | None = None
    index_readiness: RagV2ServingIndexReadinessService | None = None
    provider_transport: RagProviderDispatchAuthority | None = None
    answer_model: StructuredRagAnswerModel | None = None
    sqlite_scope: SQLiteRagGraphScope | None = None
    provider_transport_factory: Callable[[], RagProviderDispatchAuthority] | None = None


@dataclass(slots=True)
class AssistantExecutionDisposition:
    """Invocation-owned proof; exception type/absence of run_id is not authority."""

    phase: str = 'unproven'
    preclaim_failure: Exception | None = field(default=None, repr=False)

    def begin_admission(self) -> None:
        self.phase = 'admission_started'
        self.preclaim_failure = None

    def record_preclaim_failure(self, failure: Exception) -> None:
        if self.phase in {'preflight', 'factory_preclaim', 'graph_preclaim'}:
            self.preclaim_failure = failure


@dataclass(frozen=True, slots=True)
class RagRuntimeContext:
    actor: DemoUser
    surface: RagSurface
    settings: Settings
    services: RagRequestServices = field(repr=False)
    assistant_target: AssistantProjectionTarget | None = None
    assistant_execution: AssistantExecutionDisposition | None = None


class RagGraphInput(TypedDict):
    prepared_text: PreparedRagRequestText


class RagGraphOutput(TypedDict):
    assistant_finalization: NotRequired[AssistantFinalizationRecord]
    committed_projection: NotRequired[CanonicalRagProjection]
    run_id: NotRequired[int]
    generation_component: NotRequired[RagComponentFinal]
    deterministic_answer_generated: NotRequired[bool]
    error_component: NotRequired[RagPaidComponent]
    outcome: RagResultOutcome
    answer_blocks: ValidatedAnswerBlocks | None
    selected_slot_ids: tuple[EvidenceSlotId, ...]
    evidence_projection: V1EvidenceProjection
    model_influence: tuple[ModelInfluenceDependencySnapshot, ...]
    hidden_match_count: int
    effective_backend: RagEffectiveBackend
    fallback_category: RagFallbackCategory | None
    charged_cost_usd: Decimal
    sanitized_trace: RagRunTrace


class RagGraphState(RagGraphInput, RagGraphOutput, total=False):
    security_scope: SecurityScope
    scope_fingerprint: str
    configured_backend: str
    visible_limit: int
    prepared_embedding: PreparedQueryEmbedding
    query_embedding_result: QueryEmbeddingCallResult
    query_embedding_attempted: bool
    answer_generation_attempted: bool
    prepared_answer: PreparedAnswerInvocation
    prepared_model_influence: PreparedModelInfluenceSet
    validated_answer: ValidatedAnswerBlocks
    influence_validated: bool
    selected_membership: tuple[EvidenceSlotId, ...]
    safe_outcome: str
    run_id: int
    generations: tuple[int, int | None]
    retrieval: RetrievalResult
    candidates: tuple[RetrievalCandidate, ...]
    evidence_slots: tuple[EvidenceSlot, ...]
    route: str
    pending: RagProjectionPending
    prepared_finalization: PreparedRagFinalization
    node_counts: dict[str, int]
    node_latency_ms: dict[str, int]
