from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from langchain_core.runnables import Runnable

from backend.app.agent_runtime.rag_v2_contracts import (
    RagEffectiveBackend,
    RagRetrievalBackend,
)
from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    ServingEvidence,
)
from backend.app.rag.vector_validation import CanonicalFloat32Vector

RagPaidComponent = Literal['query_embedding', 'answer_generation']


@dataclass(frozen=True, slots=True)
class StrictProviderUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.input_tokens) is not int
            or type(self.output_tokens) is not int
            or type(self.total_tokens) is not int
            or self.input_tokens < 0
            or self.output_tokens < 0
            or self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError('provider usage must contain exact nonnegative totals')


@dataclass(frozen=True, slots=True)
class PreparedPaidCallBudget:
    component: RagPaidComponent
    estimated_input_tokens: int
    maximum_output_tokens: int
    reserved_cost_usd: Decimal
    cost_policy_snapshot_hmac: str
    estimator_input_hmac: str


@dataclass(frozen=True, slots=True)
class QueryEmbeddingCostInput:
    retrieval_query_utf8: bytes
    model_config_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class AnswerGenerationCostInput:
    exact_messages_json: bytes
    exact_response_schema_json: bytes
    model_config_snapshot_hmac: str


class EmbeddingUsageParserPort(Protocol):
    def parse_usage(self, usage: object) -> StrictProviderUsage: ...


class RagCostPolicyPort(Protocol):
    def prepare_query_embedding(
        self,
        value: QueryEmbeddingCostInput,
    ) -> PreparedPaidCallBudget: ...

    def charge_actual(
        self,
        component: RagPaidComponent,
        usage: StrictProviderUsage,
    ) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class PreparedQueryEmbedding:
    retrieval_query_hmac: str
    transient_query_utf8: bytes
    corpus_generation: int
    vector_index_generation: int
    readiness_snapshot_hmac: str
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str
    estimated_input_tokens: int
    reserved_cost_usd: Decimal
    attempt_fence_hmac: str
    budget: PreparedPaidCallBudget


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    evidence: ServingEvidence
    relevance_score: float
    matched_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.relevance_score) is not float
            or not math.isfinite(self.relevance_score)
            or self.relevance_score <= 0
            or type(self.matched_terms) is not tuple
        ):
            raise ValueError('retrieval candidate is invalid')


@dataclass(frozen=True, slots=True)
class QueryEmbeddingReceipt:
    attempted: bool
    input_tokens: int
    actual_cost_usd: Decimal
    latency_ms: int
    outcome: str
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class QueryEmbeddingCallResult:
    prepared: PreparedQueryEmbedding
    vector: CanonicalFloat32Vector
    attempted: Literal[True]
    validated_input_tokens: int
    actual_cost_usd: Decimal
    receipt: QueryEmbeddingReceipt


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    retrieval_query_text: str
    security_scope: SecurityScope
    security_scope_fingerprint: str
    query_embedding_result: QueryEmbeddingCallResult | None
    candidate_scan_limit: Literal[50]
    visible_limit: Literal[5, 8]
    relevance_policy_version: Literal['rag-retrieval-policy:v2.0']

    def __post_init__(self) -> None:
        if type(self.retrieval_query_text) is not str:
            raise ValueError('retrieval query must be text')
        if type(self.security_scope_fingerprint) is not str:
            raise ValueError('security scope fingerprint must be text')
        if self.candidate_scan_limit != 50:
            raise ValueError('candidate scan limit must be exactly 50')
        if self.visible_limit not in {5, 8}:
            raise ValueError('visible limit must be exactly 5 or 8')
        if self.relevance_policy_version != 'rag-retrieval-policy:v2.0':
            raise ValueError('relevance policy version is invalid')


@dataclass(frozen=True, slots=True)
class SanitizedRetrievalTrace:
    candidate_window_count: int
    visible_count: int
    hidden_match_count: int
    provider_attempt_count: int
    latency_ms: int
    fallback_category: str | None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    configured_backend: RagRetrievalBackend
    effective_backend: RagEffectiveBackend
    visible: tuple[RetrievalCandidate, ...]
    hidden_match_count: int
    hidden_count_capped: bool
    top_candidate_window_hmac: str
    query_embedding_receipt: QueryEmbeddingReceipt | None
    trace: SanitizedRetrievalTrace


@dataclass(frozen=True, slots=True)
class ClassifiedRetrievalCandidate:
    """Request-local internal candidate; denied identities never escape the retriever."""

    evidence: ServingEvidence
    relevance_score: float
    matched_terms: tuple[str, ...]
    access: EvidenceAccessClassification

    def __post_init__(self) -> None:
        if (
            type(self.relevance_score) is not float
            or not math.isfinite(self.relevance_score)
            or self.relevance_score <= 0
            or type(self.matched_terms) is not tuple
            or not self.matched_terms
        ):
            raise ValueError('classified retrieval candidate is invalid')


class KeywordSearchStorePort(Protocol):
    def search(
        self,
        request: RetrievalRequest,
    ) -> tuple[ClassifiedRetrievalCandidate, ...]: ...


class RagRetrieverRegistry:
    def __init__(self) -> None:
        self._retrievers: dict[
            RagRetrievalBackend,
            Runnable[RetrievalRequest, RetrievalResult],
        ] = {}

    def register(
        self,
        backend: RagRetrievalBackend,
        retriever: Runnable[RetrievalRequest, RetrievalResult],
    ) -> None:
        _require_known_backend(backend)
        if not isinstance(retriever, Runnable):
            raise TypeError('retriever must be a LangChain Runnable')
        if backend in self._retrievers:
            raise ValueError(f'retrieval backend already registered: {backend}')
        self._retrievers[backend] = retriever

    def resolve(
        self,
        backend: RagRetrievalBackend,
    ) -> Runnable[RetrievalRequest, RetrievalResult]:
        _require_known_backend(backend)
        try:
            return self._retrievers[backend]
        except KeyError:
            raise KeyError(f'retrieval backend is not registered: {backend}') from None


def _require_known_backend(value: object) -> None:
    if value not in {'keyword', 'pgvector'}:
        raise ValueError(f'unknown retrieval backend: {value}')
