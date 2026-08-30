from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.schemas.auto_review import (
    AutoReviewAuditResponse,
    RevokeAutoApprovalResponse,
)


class ReviewItemUpdate(BaseModel):
    payload: dict[str, Any] | None = None
    source_links: list[str] | None = None
    source_snippets: list[str] | None = None
    confidence_score: float | None = None
    permission_level: str | None = None


class ReviewEvidenceRequest(BaseModel):
    note: str | None = None


class ReviewBulkActionRequest(BaseModel):
    action: Literal['approve', 'reject']
    item_ids: list[int] | None = None


class RevokeAutoApprovalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    reason_code: Literal[
        'business_withdrawal',
        'incorrect_content',
        'permission_violation',
        'wrong_source_version',
        'policy_violation',
    ]


class AutoReviewAuditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    outcome: Literal[
        'confirmed',
        'incorrect',
        'permission_violation',
        'source_version_violation',
        'policy_violation',
    ]
    reason: str = Field(min_length=1, max_length=500)

    @field_validator('reason')
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = ' '.join(value.split())
        if not 1 <= len(normalized) <= 500:
            raise ValueError('reason must contain 1 to 500 characters')
        return normalized


__all__ = [
    'AutoReviewAuditRequest',
    'AutoReviewAuditResponse',
    'ReviewBulkActionRequest',
    'ReviewEvidenceRequest',
    'ReviewItemUpdate',
    'RevokeAutoApprovalRequest',
    'RevokeAutoApprovalResponse',
]
