from datetime import UTC, datetime

from backend.app.agent_runtime.review_v21_service import ReviewWorkflowStatusV21
from backend.app.api.v1.orchestration_v2 import (
    _status_response,
    _validated_run_request,
)
from backend.app.schemas.review_workflow import (
    ReviewWorkflowRunRequestV21,
    ReviewWorkflowStatusResponseV21,
)


def test_v21_start_body_is_selected_only_by_confirmation_token() -> None:
    body = {
        'source_refs': [{
            'source_type': 'gmail',
            'source_id': 'message-1',
            'version_or_signature': 'version-1',
        }],
        'agent_names': ['history_agent'],
        'launch_confirmation_token': 'x' * 32,
    }

    parsed = _validated_run_request(body, allow_v21_token=True)

    assert isinstance(parsed, ReviewWorkflowRunRequestV21)


def test_v21_status_mapper_emits_the_discriminated_five_count_shape() -> None:
    now = datetime(2026, 8, 30, tzinfo=UTC)
    status = ReviewWorkflowStatusV21(
        thread_id='thread-v21',
        status='completed',
        review_item_count=1,
        review_status_counts={
            'pending_review': 0,
            'approved': 0,
            'rejected': 0,
            'needs_more_evidence': 0,
            'revoked': 1,
        },
        durable=True,
        graph_version='company-memory-review-v2.1-auto-review',
        review_resolution_ready=True,
        checkpoint_resumable=False,
        resume_allowed=False,
        retry_allowed=False,
        created_at=now,
        updated_at=now,
        error_code=None,
        resume_error_code=None,
        auto_review_mode='enforce',
        auto_review_policy_version='auto-review-policy:v1',
        auto_review_enforce_percentage=10,
        auto_approved_count=0,
        human_review_required_count=0,
        auto_review_fallback_count=0,
    )

    response = _status_response(status)  # type: ignore[arg-type]

    assert isinstance(response, ReviewWorkflowStatusResponseV21)
    payload = response.model_dump(mode='json')
    assert payload['graph_version'] == 'company-memory-review-v2.1-auto-review'
    assert set(payload['review_status_counts']) == {
        'pending_review',
        'approved',
        'rejected',
        'needs_more_evidence',
        'revoked',
    }
    assert not {
        'review_item_ids', 'validation_output', 'source_snippets', 'model_output'
    } & set(payload)
