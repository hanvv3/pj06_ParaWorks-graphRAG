from itertools import product

import pytest

from backend.app.assistant.delivery import AssistantDeliveryResult

SAFE_200_OUTCOMES = {
    'supported',
    'insufficient_evidence',
    'no_match',
    'hidden_only',
    'safety_filter_empty',
    'evidence_unavailable',
}
SAFE_GENERATION_FAILURES = {
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
ALL_OUTCOMES = (
    SAFE_200_OUTCOMES
    | SAFE_GENERATION_FAILURES
    | {
        'budget_exceeded',
        'permission_denied',
        'owner_not_found',
        'invalid_input',
        'input_safety_blocked',
        'input_scanner_unavailable',
        'persistence_failed',
        'commit_unknown',
    }
)
STATES = {
    'committed_success',
    'committed_safe_failure',
    'committed_run_failure',
    'not_persisted',
    'commit_unknown',
}
BODY_KINDS = {
    'assistant_message',
    'budget_error',
    'generation_error',
    'permission_error',
    'owner_not_found',
    'validation_error',
    'persistence_error',
    'reconciliation_required',
}
STATUSES = {200, 403, 404, 409, 422, 500, 502}


def _legal_cases() -> set[tuple[str, int | None, str, str, int | None, int]]:
    cases = {
        (outcome, 11, 'assistant_message', 'committed_success', 17, 200)
        for outcome in SAFE_200_OUTCOMES
    }
    cases.add(
        ('budget_exceeded', 11, 'budget_error', 'committed_safe_failure', 17, 409)
    )
    cases.update(
        (outcome, 11, 'generation_error', 'committed_safe_failure', 17, 502)
        for outcome in SAFE_GENERATION_FAILURES
    )
    cases.add(
        ('persistence_failed', None, 'persistence_error', 'committed_run_failure', 17, 500)
    )
    cases.update(
        {
            ('permission_denied', None, 'permission_error', 'not_persisted', None, 403),
            ('owner_not_found', None, 'owner_not_found', 'not_persisted', None, 404),
            ('invalid_input', None, 'validation_error', 'not_persisted', None, 422),
            ('input_safety_blocked', None, 'validation_error', 'not_persisted', None, 422),
            (
                'runtime_version_unavailable',
                None,
                'generation_error',
                'not_persisted',
                None,
                502,
            ),
            (
                'input_scanner_unavailable',
                None,
                'persistence_error',
                'not_persisted',
                None,
                500,
            ),
            (
                'persistence_failed',
                None,
                'persistence_error',
                'not_persisted',
                None,
                500,
            ),
            (
                'commit_unknown',
                None,
                'reconciliation_required',
                'commit_unknown',
                None,
                500,
            ),
        }
    )
    return cases


def test_assistant_delivery_result_accepts_every_approved_legal_case() -> None:
    for values in _legal_cases():
        result = AssistantDeliveryResult(*values)
        assert (
            result.application_outcome,
            result.assistant_message_id,
            result.body_kind,
            result.delivery_state,
            result.parent_agent_run_id,
            result.public_status,
        ) == values


def test_assistant_delivery_result_rejects_every_cross_combination() -> None:
    legal = _legal_cases()
    for values in product(
        ALL_OUTCOMES,
        (None, 11),
        BODY_KINDS,
        STATES,
        (None, 17),
        STATUSES,
    ):
        if values in legal:
            continue
        with pytest.raises(ValueError):
            AssistantDeliveryResult(*values)


@pytest.mark.parametrize('invalid_id', [True, False, 0, -1, 1.0, '1'])
@pytest.mark.parametrize('field', ['assistant_message_id', 'parent_agent_run_id'])
def test_assistant_delivery_result_rejects_non_positive_or_non_integer_ids(
    invalid_id: object,
    field: str,
) -> None:
    values = {
        'application_outcome': 'supported',
        'assistant_message_id': 11,
        'body_kind': 'assistant_message',
        'delivery_state': 'committed_success',
        'parent_agent_run_id': 17,
        'public_status': 200,
    }
    values[field] = invalid_id
    with pytest.raises(ValueError):
        AssistantDeliveryResult(**values)
