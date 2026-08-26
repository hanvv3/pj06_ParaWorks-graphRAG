import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


def _workflow_request(payload: dict):
    from backend.app.schemas.review_workflow import ReviewWorkflowRunRequest

    return ReviewWorkflowRunRequest.model_validate(payload)


def test_review_workflow_request_rejects_raw_prompt_and_group_alias() -> None:
    payload = {
        'source_refs': [
            {
                'source_type': 'gmail',
                'source_id': 'gmail-message-123',
                'version_or_signature': 'signature-123',
            }
        ],
        'agent_names': ['mail_document_agent'],
    }

    with pytest.raises(ValidationError):
        _workflow_request({**payload, 'prompt': 'summarize this email'})
    with pytest.raises(ValidationError):
        _workflow_request({**payload, 'agent_names': ['memory_extraction_agent']})


def test_review_workflow_request_accepts_all_four_canonical_source_types() -> None:
    request = _workflow_request(
        {
            'source_refs': [
                {'source_type': 'gmail', 'source_id': 'gmail-1', 'version_or_signature': 'v1'},
                {
                    'source_type': 'gmail_attachment',
                    'source_id': 'gmail-attachment-1',
                    'version_or_signature': 'v2',
                },
                {'source_type': 'drive', 'source_id': 'drive-1', 'version_or_signature': 'v3'},
                {'source_type': 'calendar', 'source_id': 'calendar-1', 'version_or_signature': 'v4'},
            ],
            'agent_names': [
                ' todo_agent ',
                'mail_document_agent',
                'history_agent',
                'decision_record_agent',
                'timeline_agent',
            ],
            'client_request_id': 'retry-key-123',
        }
    )

    assert [item.source_type for item in request.source_refs] == [
        'gmail',
        'gmail_attachment',
        'drive',
        'calendar',
    ]
    assert request.agent_names == [
        'mail_document_agent',
        'timeline_agent',
        'history_agent',
        'decision_record_agent',
        'todo_agent',
    ]


def test_review_workflow_response_schema_excludes_internal_ids_and_evidence() -> None:
    from backend.app.schemas.review_workflow import (
        ReviewWorkflowDiagnosticResponse,
        ReviewWorkflowDryRunResponse,
        ReviewWorkflowStatusResponse,
    )

    serialized = json.dumps(
        [
            ReviewWorkflowDiagnosticResponse(
                enabled=True,
                available=True,
                checkpoint_mode='postgres',
                durable=True,
                graph_version='company-memory-review-v2.0',
                default_agent_names=['mail_document_agent'],
                error_code=None,
            ).model_dump(mode='json'),
            ReviewWorkflowDryRunResponse(
                workflow_name='company-memory-review',
                graph_version='company-memory-review-v2.0',
                source_count=1,
                agent_names=['mail_document_agent'],
                selection_policy_version='company-memory-review-selection:v1',
                estimated_input_tokens=12,
                estimated_output_tokens=8,
                estimated_cost_usd=0.00001,
                budget_limit_usd=0.001,
                budget_status='within_budget',
                cache_hit=False,
                requires_explicit_run=True,
            ).model_dump(mode='json'),
            ReviewWorkflowStatusResponse(
                thread_id='opaque-thread-id',
                status='awaiting_human_review',
                review_item_count=1,
                review_status_counts={'pending_review': 1},
                durable=True,
                graph_version='company-memory-review-v2.0',
                review_resolution_ready=False,
                checkpoint_resumable=True,
                resume_allowed=False,
                retry_allowed=False,
                created_at=datetime(2026, 8, 27, tzinfo=UTC),
                updated_at=datetime(2026, 8, 27, 1, tzinfo=UTC),
                error_code=None,
                resume_error_code=None,
            ).model_dump(mode='json'),
        ]
    )

    for internal_field in (
        'review_item_ids',
        'checkpoint_thread_id',
        'source_refs',
        'source_snippets',
        'exception',
    ):
        assert internal_field not in serialized
