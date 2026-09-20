from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

from langchain_core.runnables import Runnable

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_contracts import (
    RagEffectiveBackend,
    RagRetrievalBackend,
)
from backend.app.agent_runtime.rag_v2_identity import SecurityScope, exact_utf8_bytes
from backend.app.rag.embeddings import (
    ValidatedQueryEmbeddingVector,
    validate_query_embedding_vector_carrier,
)
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    ServingEvidence,
    SupportMode,
)

RagPaidComponent = Literal['query_embedding', 'answer_generation']

if TYPE_CHECKING:
    from backend.app.core.config import Settings
    from backend.app.rag.graph_projection import GraphPathDependency


QUERY_EMBEDDING_MODEL = 'text-embedding-3-small'
QUERY_EMBEDDING_DIMENSIONS = 1536
QUERY_EMBEDDING_MODEL_CONFIG_VERSION = 'rag-query-embedding-config:v1'
QUERY_EMBEDDING_PROVIDER_POLICY_VERSION = 'openai-embeddings-api:v1'
QUERY_EMBEDDING_PAYLOAD_VALIDATOR_VERSION = 'rag-query-embedding-payload:v1'
QUERY_EMBEDDING_INDEX_POLICY_VERSION = 'rag-v2-serving-index:v1'
SIGNED_BIGINT_MAX = 2**63 - 1
RagServingReadinessSnapshot: TypeAlias = tuple[
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    str,
    int,
    str,
    str,
]


@dataclass(frozen=True, slots=True)
class StrictProviderUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        validate_strict_provider_usage(self)


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


EvidenceSlotId: TypeAlias = Literal['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
_EVIDENCE_SLOT_IDS: tuple[EvidenceSlotId, ...] = (
    'E1',
    'E2',
    'E3',
    'E4',
    'E5',
    'E6',
    'E7',
    'E8',
)


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    slot_id: EvidenceSlotId
    support_mode: SupportMode
    evidence: ServingEvidence
    relevance_score: float
    matched_terms: tuple[str, ...]


def rank_evidence_slots(
    candidates: tuple[RetrievalCandidate, ...],
    *,
    max_serialized_content_chars: int | None = None,
) -> tuple[EvidenceSlot, ...]:
    """Stable-partition approved candidates and remove only whole tail rows."""
    if type(candidates) is not tuple:
        raise ValueError('retrieval candidates must be an immutable tuple')
    if max_serialized_content_chars is not None and (
        type(max_serialized_content_chars) is not int
        or max_serialized_content_chars < 0
    ):
        raise ValueError('evidence content budget is invalid')
    validated: list[RetrievalCandidate] = []
    for candidate in candidates:
        if type(candidate) is not RetrievalCandidate:
            raise ValueError('retrieval candidate is invalid')
        validated.append(candidate)
    ordered = [
        *(
            candidate
            for candidate in validated
            if candidate.evidence.serving_kind == 'trusted_knowledge'
        ),
        *(
            candidate
            for candidate in validated
            if candidate.evidence.serving_kind == 'raw_chunk'
        ),
    ][:8]
    if max_serialized_content_chars is not None:
        while (
            ordered
            and sum(len(value.evidence.model_content) for value in ordered)
            > max_serialized_content_chars
        ):
            ordered.pop()
    return tuple(
        EvidenceSlot(
            slot_id=_EVIDENCE_SLOT_IDS[index],
            support_mode=cast(SupportMode, candidate.evidence.support_mode),
            evidence=candidate.evidence,
            relevance_score=candidate.relevance_score,
            matched_terms=candidate.matched_terms,
        )
        for index, candidate in enumerate(ordered)
    )


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
    vector: ValidatedQueryEmbeddingVector
    attempted: Literal[True]
    validated_input_tokens: int
    actual_cost_usd: Decimal
    receipt: QueryEmbeddingReceipt
    committed_dispatch_hmac: str | None = field(default=None, repr=False)


def validate_query_embedding_call_result(
    request: RetrievalRequest,
    *,
    settings: Settings,
) -> QueryEmbeddingCallResult:
    try:
        result = request.query_embedding_result
        if type(result) is not QueryEmbeddingCallResult:
            raise ValueError
        query_utf8 = request.retrieval_query_text.encode('utf-8', errors='strict')
        prepared = result.prepared
        if (
            type(prepared) is PreparedQueryEmbedding
            and type(prepared.transient_query_utf8) is bytes
            and prepared.transient_query_utf8 != query_utf8
        ):
            raise ValueError('query bytes do not match the prepared embedding')
        prepared = validate_prepared_query_embedding(
            prepared,
            settings=settings,
            expected_query_utf8=query_utf8,
        )
        receipt = result.receipt
        if type(receipt) is not QueryEmbeddingReceipt:
            raise ValueError
        vector = validate_query_embedding_vector_carrier(
            result.vector,
            expected_dimensions=QUERY_EMBEDDING_DIMENSIONS,
        )
        if (
            result.vector is not vector
            or result.attempted is not True
            or not is_signed_bigint(result.validated_input_tokens)
            or result.validated_input_tokens > prepared.estimated_input_tokens
            or not is_storage_decimal(result.actual_cost_usd)
            or result.actual_cost_usd > prepared.reserved_cost_usd
            or receipt.attempted is not True
            or not is_signed_bigint(receipt.input_tokens)
            or receipt.input_tokens != result.validated_input_tokens
            or not same_decimal_storage_value(
                receipt.actual_cost_usd,
                result.actual_cost_usd,
            )
            or not is_signed_bigint(receipt.latency_ms)
            or type(receipt.outcome) is not str
            or receipt.outcome != 'component_succeeded'
            or not is_lower_hex_64(receipt.model_config_snapshot_hmac)
            or receipt.model_config_snapshot_hmac != prepared.model_config_snapshot_hmac
            or not is_lower_hex_64(receipt.provider_policy_snapshot_hmac)
            or receipt.provider_policy_snapshot_hmac
            != prepared.provider_policy_snapshot_hmac
        ):
            raise ValueError
        return result
    except ValueError as exc:
        if str(exc) == 'query bytes do not match the prepared embedding':
            raise
        raise ValueError('query embedding carrier is invalid') from None
    except (AttributeError, TypeError, UnicodeError):
        raise ValueError('query embedding carrier is invalid') from None


def validate_prepared_query_embedding(
    value: object,
    *,
    settings: Settings,
    expected_query_utf8: bytes | None = None,
    expected_readiness: RagServingReadinessSnapshot | None = None,
) -> PreparedQueryEmbedding:
    try:
        if type(value) is not PreparedQueryEmbedding:
            raise ValueError
        if (
            type(value.transient_query_utf8) is not bytes
            or (
                expected_query_utf8 is not None
                and (
                    type(expected_query_utf8) is not bytes
                    or value.transient_query_utf8 != expected_query_utf8
                )
            )
            or not is_signed_bigint(value.corpus_generation)
            or not is_signed_bigint(value.vector_index_generation)
            or not is_lower_hex_64(value.readiness_snapshot_hmac)
            or not is_signed_bigint(value.estimated_input_tokens)
            or not is_lower_hex_64(value.retrieval_query_hmac)
            or value.retrieval_query_hmac
            != build_query_embedding_retrieval_query_hmac_from_utf8(
                value.transient_query_utf8,
                settings=settings,
            )
            or (
                expected_readiness is not None
                and (
                    type(expected_readiness) is not tuple
                    or len(expected_readiness) != 11
                    or (
                        value.corpus_generation,
                        value.vector_index_generation,
                        value.readiness_snapshot_hmac,
                    )
                    != (
                        expected_readiness[1],
                        expected_readiness[2],
                        expected_readiness[10],
                    )
                )
            )
        ):
            raise ValueError
        budget = value.budget
        validate_query_embedding_budget(budget)
        expected_model_hmac = build_query_embedding_model_config_snapshot_hmac(settings)
        expected_provider_hmac = build_query_embedding_provider_policy_snapshot_hmac(
            settings
        )
        expected_fence = build_query_embedding_attempt_fence_hmac(
            retrieval_query_hmac=value.retrieval_query_hmac,
            corpus_generation=value.corpus_generation,
            vector_index_generation=value.vector_index_generation,
            readiness_snapshot_hmac=value.readiness_snapshot_hmac,
            model_config_snapshot_hmac=expected_model_hmac,
            provider_policy_snapshot_hmac=expected_provider_hmac,
            budget=budget,
            settings=settings,
        )
        if (
            not is_lower_hex_64(value.model_config_snapshot_hmac)
            or value.model_config_snapshot_hmac != expected_model_hmac
            or not is_lower_hex_64(value.provider_policy_snapshot_hmac)
            or value.provider_policy_snapshot_hmac != expected_provider_hmac
            or value.estimated_input_tokens != budget.estimated_input_tokens
            or not same_decimal_storage_value(
                value.reserved_cost_usd,
                budget.reserved_cost_usd,
            )
            or not is_lower_hex_64(value.attempt_fence_hmac)
            or value.attempt_fence_hmac != expected_fence
        ):
            raise ValueError
        return value
    except (AttributeError, TypeError, UnicodeError, ValueError):
        raise ValueError('prepared query embedding is invalid') from None


def validate_rag_serving_index_readiness(
    value: object,
) -> RagServingReadinessSnapshot:
    from backend.app.rag.index_readiness import RagServingIndexReadiness

    if (
        type(value) is not RagServingIndexReadiness
        or type(value.ready) is not bool
        or not is_signed_bigint(value.corpus_generation)
        or not is_signed_bigint(value.vector_index_generation)
        or not is_signed_bigint(value.expected_document_count)
        or not is_signed_bigint(value.live_vector_count)
        or not is_signed_bigint(value.tombstone_count)
        or not is_signed_bigint(value.mismatch_count_capped_at_20)
        or value.mismatch_count_capped_at_20 > 20
        or type(value.embedding_model) is not str
        or value.embedding_model != QUERY_EMBEDDING_MODEL
        or type(value.embedding_dimensions) is not int
        or value.embedding_dimensions != QUERY_EMBEDDING_DIMENSIONS
        or type(value.index_policy_version) is not str
        or value.index_policy_version != QUERY_EMBEDDING_INDEX_POLICY_VERSION
        or not is_lower_hex_64(value.readiness_snapshot_hmac)
    ):
        raise ValueError('serving index readiness snapshot is invalid')
    return (
        value.ready,
        value.corpus_generation,
        value.vector_index_generation,
        value.expected_document_count,
        value.live_vector_count,
        value.tombstone_count,
        value.mismatch_count_capped_at_20,
        value.embedding_model,
        value.embedding_dimensions,
        value.index_policy_version,
        value.readiness_snapshot_hmac,
    )


def validate_query_embedding_budget(value: PreparedPaidCallBudget) -> None:
    if (
        type(value) is not PreparedPaidCallBudget
        or type(value.component) is not str
        or value.component != 'query_embedding'
        or not is_signed_bigint(value.estimated_input_tokens)
        or not is_signed_bigint(value.maximum_output_tokens)
        or value.maximum_output_tokens != 0
        or not is_storage_decimal(value.reserved_cost_usd)
        or not is_lower_hex_64(value.cost_policy_snapshot_hmac)
        or not is_lower_hex_64(value.estimator_input_hmac)
    ):
        raise ValueError('query embedding budget is invalid')


def build_query_embedding_retrieval_query_hmac(
    query: str,
    *,
    settings: Settings,
) -> str:
    if type(query) is not str:
        raise ValueError('query embedding query is invalid')
    query_identity = exact_utf8_bytes(query)
    return _build_query_embedding_retrieval_query_hmac(
        query_identity,
        settings=settings,
    )


def build_query_embedding_retrieval_query_hmac_from_utf8(
    query_utf8: bytes,
    *,
    settings: Settings,
) -> str:
    if type(query_utf8) is not bytes:
        raise ValueError('query embedding query bytes are invalid')
    try:
        query_identity = exact_utf8_bytes(query_utf8.decode('utf-8', errors='strict'))
    except (UnicodeDecodeError, ValueError):
        raise ValueError('query embedding query bytes are invalid') from None
    return _build_query_embedding_retrieval_query_hmac(
        query_identity,
        settings=settings,
    )


def _build_query_embedding_retrieval_query_hmac(
    query_identity: dict[str, int | str],
    *,
    settings: Settings,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        query_identity,
        secret=secret,
        schema_version='rag-query-embedding-query:v1',
        policy_version=QUERY_EMBEDDING_MODEL_CONFIG_VERSION,
    )


def build_query_embedding_model_config_snapshot_hmac(settings: Settings) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'api_base_url': 'https://api.openai.com/v1',
            'dimensions': QUERY_EMBEDDING_DIMENSIONS,
            'encoding_format': 'float',
            'endpoint_identity': 'openai-direct-standard-global:v1',
            'input_count_per_query_call': 1,
            'max_provider_attempts': 1,
            'model': QUERY_EMBEDDING_MODEL,
            'provider': 'openai',
            'provider_send_start_window_seconds': 5,
            'regional_processing': False,
            'sdk_retry': 0,
            'timeout_seconds': 30,
        },
        secret=secret,
        schema_version='rag-query-embedding-model-config-snapshot:v1',
        policy_version=QUERY_EMBEDDING_MODEL_CONFIG_VERSION,
    )


def build_query_embedding_provider_policy_snapshot_hmac(settings: Settings) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'provider': 'openai',
            'protocol_family': QUERY_EMBEDDING_PROVIDER_POLICY_VERSION,
            'payload_validator': QUERY_EMBEDDING_PAYLOAD_VALIDATOR_VERSION,
            'max_provider_attempts': 1,
        },
        secret=secret,
        schema_version='rag-query-embedding-provider-policy:v1',
        policy_version=QUERY_EMBEDDING_PROVIDER_POLICY_VERSION,
    )


def build_query_embedding_attempt_fence_hmac(
    *,
    retrieval_query_hmac: str,
    corpus_generation: int,
    vector_index_generation: int,
    readiness_snapshot_hmac: str,
    model_config_snapshot_hmac: str,
    provider_policy_snapshot_hmac: str,
    budget: PreparedPaidCallBudget,
    settings: Settings,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'query_hmac': retrieval_query_hmac,
            'corpus_generation': corpus_generation,
            'vector_index_generation': vector_index_generation,
            'readiness_snapshot_hmac': readiness_snapshot_hmac,
            'model_config_snapshot_hmac': model_config_snapshot_hmac,
            'provider_policy_snapshot_hmac': provider_policy_snapshot_hmac,
            'budget_component': budget.component,
            'estimated_input_tokens': budget.estimated_input_tokens,
            'maximum_output_tokens': budget.maximum_output_tokens,
            'reserved_cost_usd': decimal_storage_representation(
                budget.reserved_cost_usd
            ),
            'cost_policy_snapshot_hmac': budget.cost_policy_snapshot_hmac,
            'estimator_input_hmac': budget.estimator_input_hmac,
        },
        secret=secret,
        schema_version='rag-query-embedding-attempt-fence:v1',
        policy_version=QUERY_EMBEDDING_MODEL_CONFIG_VERSION,
    )


def is_lower_hex_64(value: object) -> bool:
    return bool(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value))


def is_numeric_24_6_representable(value: Decimal) -> bool:
    if type(value) is not Decimal or not value.is_finite():
        return False
    if value.as_tuple().exponent < -6:
        return False
    normalized = value.normalize()
    if normalized.is_zero():
        return True
    return normalized.adjusted() <= 17


def is_signed_bigint(value: object) -> bool:
    return bool(type(value) is int and 0 <= value <= SIGNED_BIGINT_MAX)


def is_storage_decimal(value: object) -> bool:
    return bool(
        type(value) is Decimal
        and value.is_finite()
        and value >= Decimal('0')
        and not (value.is_zero() and value.is_signed())
        and is_numeric_24_6_representable(value)
    )


def decimal_storage_representation(value: object) -> str:
    if not is_storage_decimal(value):
        raise ValueError('query embedding cost is invalid')
    assert type(value) is Decimal
    if value.is_zero():
        return '0'
    return format(value, 'f')


def same_decimal_storage_value(left: object, right: object) -> bool:
    return bool(
        is_storage_decimal(left)
        and is_storage_decimal(right)
        and decimal_storage_representation(left)
        == decimal_storage_representation(right)
    )


def validate_strict_provider_usage(value: object) -> StrictProviderUsage:
    if (
        type(value) is not StrictProviderUsage
        or not is_signed_bigint(value.input_tokens)
        or not is_signed_bigint(value.output_tokens)
        or not is_signed_bigint(value.total_tokens)
        or value.total_tokens != value.input_tokens + value.output_tokens
    ):
        raise ValueError('provider usage must contain exact bounded totals')
    return value


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
    graph_paths: tuple[GraphPathDependency, ...] = ()
    graph_policy_version: str | None = None


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
