from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.app.agent_runtime.provider_usage import AssistantApplicationOutcome

AssistantBodyKind = Literal[
    'assistant_message',
    'budget_error',
    'generation_error',
    'permission_error',
    'owner_not_found',
    'validation_error',
    'persistence_error',
    'reconciliation_required',
]
AssistantDeliveryState = Literal[
    'committed_success',
    'committed_safe_failure',
    'committed_run_failure',
    'not_persisted',
    'commit_unknown',
]
AssistantPublicStatus = Literal[200, 403, 404, 409, 422, 500, 502]

_SUCCESS_OUTCOMES = frozenset(
    {
        'supported',
        'no_match',
        'hidden_only',
        'safety_filter_empty',
        'insufficient_evidence',
        'evidence_unavailable',
    }
)
_SAFE_GENERATION_FAILURE_OUTCOMES = frozenset(
    {
        'retriever_not_configured',
        'retriever_unavailable',
        'runtime_version_unavailable',
        'model_unavailable',
        'provider_safety_unavailable',
        'provider_response_identity_invalid',
        'provider_usage_overrun',
        'provider_embedding_payload_invalid',
        'model_provider_failed',
        'structured_output_invalid',
        'citation_validation_failed',
        'unexpected_internal_error',
    }
)


@dataclass(frozen=True, slots=True)
class AssistantDeliveryResult:
    application_outcome: AssistantApplicationOutcome
    assistant_message_id: int | None
    body_kind: AssistantBodyKind
    delivery_state: AssistantDeliveryState
    parent_agent_run_id: int | None
    public_status: AssistantPublicStatus

    def __post_init__(self) -> None:
        _require_optional_positive_id(
            self.assistant_message_id, 'assistant_message_id'
        )
        _require_optional_positive_id(
            self.parent_agent_run_id, 'parent_agent_run_id'
        )
        if not self._is_legal():
            raise ValueError('assistant delivery result combination is invalid')

    def _is_legal(self) -> bool:
        identifiers = (self.assistant_message_id, self.parent_agent_run_id)
        if self.delivery_state == 'committed_success':
            return bool(
                self.application_outcome in _SUCCESS_OUTCOMES
                and self.public_status == 200
                and self.body_kind == 'assistant_message'
                and all(identifier is not None for identifier in identifiers)
            )
        if self.delivery_state == 'committed_safe_failure':
            return bool(
                all(identifier is not None for identifier in identifiers)
                and (
                    (
                        self.application_outcome == 'budget_exceeded'
                        and self.public_status == 409
                        and self.body_kind == 'budget_error'
                    )
                    or (
                        self.application_outcome
                        in _SAFE_GENERATION_FAILURE_OUTCOMES
                        and self.public_status == 502
                        and self.body_kind == 'generation_error'
                    )
                )
            )
        if self.delivery_state == 'committed_run_failure':
            return (
                self.application_outcome == 'persistence_failed'
                and self.public_status == 500
                and self.body_kind == 'persistence_error'
                and self.assistant_message_id is None
                and self.parent_agent_run_id is not None
            )
        if self.delivery_state == 'not_persisted':
            return identifiers == (None, None) and (
                (
                    self.application_outcome == 'permission_denied'
                    and self.public_status == 403
                    and self.body_kind == 'permission_error'
                )
                or (
                    self.application_outcome == 'owner_not_found'
                    and self.public_status == 404
                    and self.body_kind == 'owner_not_found'
                )
                or (
                    self.application_outcome
                    in {'invalid_input', 'input_safety_blocked'}
                    and self.public_status == 422
                    and self.body_kind == 'validation_error'
                )
                or (
                    self.application_outcome == 'runtime_version_unavailable'
                    and self.public_status == 502
                    and self.body_kind == 'generation_error'
                )
                or (
                    self.application_outcome
                    in {'input_scanner_unavailable', 'persistence_failed'}
                    and self.public_status == 500
                    and self.body_kind == 'persistence_error'
                )
            )
        return (
            self.delivery_state == 'commit_unknown'
            and self.application_outcome == 'commit_unknown'
            and self.public_status == 500
            and self.body_kind == 'reconciliation_required'
            and identifiers == (None, None)
        )


def _require_optional_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f'{field_name} must be a positive integer or null')
