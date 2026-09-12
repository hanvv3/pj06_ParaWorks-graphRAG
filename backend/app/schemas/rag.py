"""Exact V1 wire contracts. Tuples keep committed arrays immutable in memory."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExactV1Projection(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class RagCitationResponse(ExactV1Projection):
    source_id: str
    source_url: str
    source_type: str | None
    permission_level: str
    source_snippet: str
    relevance_score: float
    matched_terms: tuple[str, ...]


PublicRagErrorCode = Literal[
    'input_safety_blocked',
    'input_scanner_unavailable',
    'permission_denied',
    'budget_exceeded',
    'runtime_version_unavailable',
    'retriever_not_configured',
    'retriever_unavailable',
    'model_unavailable',
    'provider_safety_unavailable',
    'provider_response_identity_invalid',
    'provider_usage_overrun',
    'provider_embedding_payload_invalid',
    'model_provider_failed',
    'structured_output_invalid',
    'citation_validation_failed',
    'persistence_failed',
    'unexpected_internal_error',
]


class RagPublicErrorDetail(ExactV1Projection):
    code: PublicRagErrorCode


class RagPublicErrorResponse(ExactV1Projection):
    detail: RagPublicErrorDetail


class RagValidationDetail(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class RagValidationErrorResponse(BaseModel):
    detail: list[RagValidationDetail]


class RagTokenUsageResponse(ExactV1Projection):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AskV1Projection(ExactV1Projection):
    agent_name: str
    prompt_version: str
    question: str
    answer: str
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    citations: tuple[RagCitationResponse, ...]
    permission_level: str | None
    hidden_match_count: int
    permission_notice: str | None
    agent_run_id: int | None
    cache_key: str
    model_name: str
    estimated_cost_usd: float
    token_usage: RagTokenUsageResponse


class SearchCostPolicyResponse(ExactV1Projection):
    embedding_query_call: bool
    paid_llm_call: Literal[False]
    requires_pgvector_flag: Literal[True]


class SearchResultResponse(ExactV1Projection):
    id: int
    source_id: str
    text: str
    source_snippet: str
    source_url: str
    source_type: str | None
    permission_level: str
    relevance_score: float
    matched_terms: tuple[str, ...]
    citation: RagCitationResponse
    parser_status: str | None
    parser_status_reason: str | None
    revision_id: str | None


class SearchV1Projection(ExactV1Projection):
    retrieval_backend: Literal['deterministic_lexical', 'pgvector']
    cost_policy: SearchCostPolicyResponse
    hidden_match_count: int
    results: tuple[SearchResultResponse, ...]
    permission_notice: Literal['Some sources may be hidden by permissions.'] | None = (
        Field(
            default=None,
            exclude_if=lambda value: value is None,
        )
    )
