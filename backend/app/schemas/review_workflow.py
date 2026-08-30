from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

COMPANY_MEMORY_REVIEW_WORKFLOW = 'company-memory-review'
COMPANY_MEMORY_REVIEW_GRAPH_VERSION = 'company-memory-review-v2.0'
COMPANY_MEMORY_SELECTION_POLICY_VERSION = 'company-memory-review-selection:v1'
COMPANY_MEMORY_INPUT_SCHEMA_VERSION = 'review-source-versions:v1'

DEFAULT_REVIEW_AGENT_NAMES = (
    'mail_document_agent',
    'timeline_agent',
    'history_agent',
    'decision_record_agent',
    'todo_agent',
)

BudgetStatus = Literal['within_budget', 'cached', 'no_input', 'over_budget']
CheckpointMode = Literal['disabled', 'memory', 'postgres']
ReviewItemResolutionStatus = Literal[
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
]
ReviewWorkflowErrorCode = Literal[
    'invalid_input',
    'not_found',
    'idempotency_key_reused',
    'evidence_changed',
    'permission_denied',
    'checkpoint_unavailable',
    'checkpoint_failed',
    'review_unresolved',
    'runtime_version_unavailable',
    'model_unavailable',
    'budget_exceeded',
    'concurrent_resume',
    'invalid_state_transition',
    'cost_preview_changed',
]


def normalize_agent_names(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(value.strip() for value in values)
    if len(requested) != len(set(requested)):
        raise ValueError('duplicate agent name')
    unknown = set(requested) - set(DEFAULT_REVIEW_AGENT_NAMES)
    if unknown:
        raise ValueError('unsupported agent name')
    return tuple(name for name in DEFAULT_REVIEW_AGENT_NAMES if name in requested)


class ReviewWorkflowSourceRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_type: Literal['gmail', 'gmail_attachment', 'drive', 'calendar']
    source_id: str = Field(min_length=1, max_length=255)
    version_or_signature: str = Field(min_length=1, max_length=128)


class ReviewWorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_refs: list[ReviewWorkflowSourceRef] = Field(min_length=1, max_length=500)
    agent_names: list[str] = Field(min_length=1, max_length=5)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator('agent_names')
    @classmethod
    def _normalize_agent_names(cls, values: list[str]) -> list[str]:
        return list(normalize_agent_names(values))


class ReviewWorkflowDiagnosticResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool
    available: bool
    checkpoint_mode: CheckpointMode
    durable: bool
    graph_version: str
    default_agent_names: list[str]
    error_code: ReviewWorkflowErrorCode | None


class ReviewWorkflowDryRunResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    workflow_name: Literal['company-memory-review']
    graph_version: Literal['company-memory-review-v2.0']
    source_count: int
    agent_names: list[str]
    selection_policy_version: Literal['company-memory-review-selection:v1']
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    budget_limit_usd: float | None
    budget_status: BudgetStatus
    cache_hit: bool
    requires_explicit_run: Literal[True]


class ReviewWorkflowStatusResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    thread_id: str
    status: Literal[
        'created',
        'drafting',
        'checkpoint_pending',
        'awaiting_human_review',
        'resuming',
        'completed',
        'needs_more_evidence',
        'checkpoint_failed',
        'failed',
        'cancelled',
    ]
    review_item_count: int
    review_status_counts: dict[ReviewItemResolutionStatus, int]
    durable: bool
    graph_version: str
    review_resolution_ready: bool
    checkpoint_resumable: bool
    resume_allowed: bool
    retry_allowed: bool
    created_at: datetime
    updated_at: datetime
    error_code: ReviewWorkflowErrorCode | None
    resume_error_code: ReviewWorkflowErrorCode | None


class ReviewWorkflowRunRequestV21(ReviewWorkflowRunRequest):
    launch_confirmation_token: str = Field(min_length=32, max_length=2048)


class ReviewWorkflowDryRunResponseV21(BaseModel):
    model_config = ConfigDict(extra='forbid')

    workflow_name: Literal['company-memory-review']
    graph_version: Literal['company-memory-review-v2.1-auto-review']
    source_count: int
    agent_names: list[str]
    selection_policy_version: Literal['company-memory-review-selection:v1']
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    budget_limit_usd: float | None
    budget_status: BudgetStatus
    cache_hit: bool
    requires_explicit_run: Literal[True]
    auto_review_mode: Literal['shadow', 'enforce']
    auto_review_policy_version: Literal['auto-review-policy:v1']
    auto_review_validator_provider: Literal['openai']
    auto_review_validator_model: Literal['gpt-5.6-terra']
    auto_review_reasoning_effort: Literal['medium']
    auto_review_validator_prompt_version: Literal['auto-review-validation:v2']
    auto_review_validator_output_contract_version: Literal[
        'candidate-validation-batch:v1'
    ]
    auto_review_cost_policy_version: Literal['auto-review-cost:v1']
    auto_review_enforce_percentage: Literal[0, 10, 100]
    auto_review_estimated_input_tokens: int
    auto_review_estimated_output_tokens: int
    auto_review_estimated_cost_usd: float
    total_estimated_input_tokens: int
    total_estimated_output_tokens: int
    total_estimated_cost_usd: float
    launch_confirmation_token: str = Field(min_length=32, max_length=2048)


class ReviewStatusCountsV21(BaseModel):
    model_config = ConfigDict(extra='forbid')

    pending_review: int = Field(ge=0)
    approved: int = Field(ge=0)
    rejected: int = Field(ge=0)
    needs_more_evidence: int = Field(ge=0)
    revoked: int = Field(ge=0)


class ReviewWorkflowStatusResponseV21(BaseModel):
    model_config = ConfigDict(extra='forbid')

    thread_id: str
    status: Literal[
        'created',
        'drafting',
        'checkpoint_pending',
        'awaiting_human_review',
        'resuming',
        'completed',
        'needs_more_evidence',
        'checkpoint_failed',
        'failed',
        'cancelled',
    ]
    review_item_count: int
    review_status_counts: ReviewStatusCountsV21
    durable: bool
    graph_version: Literal['company-memory-review-v2.1-auto-review']
    review_resolution_ready: bool
    checkpoint_resumable: bool
    resume_allowed: bool
    retry_allowed: bool
    created_at: datetime
    updated_at: datetime
    error_code: ReviewWorkflowErrorCode | None
    resume_error_code: ReviewWorkflowErrorCode | None
    auto_review_mode: Literal['shadow', 'enforce']
    auto_review_policy_version: Literal['auto-review-policy:v1']
    auto_review_enforce_percentage: Literal[0, 10, 100]
    auto_approved_count: int
    human_review_required_count: int
    auto_review_fallback_count: int


class ReviewWorkflowStatusResponseV20(ReviewWorkflowStatusResponse):
    """V2.0-shaped discriminator alias; the original class stays unchanged."""

    graph_version: Literal['company-memory-review-v2.0']


ReviewWorkflowDryRunUnion = Annotated[
    ReviewWorkflowDryRunResponse | ReviewWorkflowDryRunResponseV21,
    Field(discriminator='graph_version'),
]
ReviewWorkflowStatusUnion = Annotated[
    ReviewWorkflowStatusResponseV20 | ReviewWorkflowStatusResponseV21,
    Field(discriminator='graph_version'),
]
