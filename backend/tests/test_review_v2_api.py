from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from backend.app.agent_runtime.review_v2_service import (
    ReviewWorkflowServiceError,
    ReviewWorkflowStatus,
)
from backend.app.main import create_app
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    DEFAULT_REVIEW_AGENT_NAMES,
    ReviewWorkflowDiagnosticResponse,
    ReviewWorkflowDryRunResponse,
)


def _status(status: str = 'awaiting_human_review') -> ReviewWorkflowStatus:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    return ReviewWorkflowStatus(
        thread_id='opaque-thread-id',
        status=status,
        review_item_count=1 if status != 'completed' else 0,
        review_status_counts=(
            {'pending_review': 1}
            if status != 'completed'
            else {}
        ),
        durable=False,
        graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        review_resolution_ready=False,
        checkpoint_resumable=status == 'awaiting_human_review',
        resume_allowed=False,
        retry_allowed=False,
        created_at=now,
        updated_at=now,
        error_code=None,
        resume_error_code=None,
    )


def _diagnostic(
    *,
    enabled: bool,
    available: bool,
    checkpoint_mode: str,
    durable: bool,
    error_code: str | None = None,
) -> ReviewWorkflowDiagnosticResponse:
    return ReviewWorkflowDiagnosticResponse(
        enabled=enabled,
        available=available,
        checkpoint_mode=checkpoint_mode,  # type: ignore[arg-type]
        durable=durable,
        graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        default_agent_names=list(DEFAULT_REVIEW_AGENT_NAMES),
        error_code=error_code,  # type: ignore[arg-type]
    )


def _dry_run() -> ReviewWorkflowDryRunResponse:
    return ReviewWorkflowDryRunResponse(
        workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
        graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        source_count=1,
        agent_names=['mail_document_agent'],
        selection_policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        estimated_input_tokens=100,
        estimated_output_tokens=50,
        estimated_cost_usd=0.001,
        budget_limit_usd=1.0,
        budget_status='within_budget',
        cache_hit=False,
        requires_explicit_run=True,
    )


@dataclass
class _FakeService:
    diagnostic_response: ReviewWorkflowDiagnosticResponse = field(
        default_factory=lambda: _diagnostic(
            enabled=True,
            available=True,
            checkpoint_mode='memory',
            durable=False,
        )
    )
    start_status: ReviewWorkflowStatus = field(default_factory=_status)
    failure_by_method: dict[str, str] = field(default_factory=dict)
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def _fail(self, method: str) -> None:
        if code := self.failure_by_method.get(method):
            raise ReviewWorkflowServiceError(code)

    def diagnostic(self):
        self.calls.append(('diagnostic', None))
        self._fail('diagnostic')
        return self.diagnostic_response

    def dry_run(self, *, actor, request):
        self.calls.append(('dry_run', actor.id))
        assert request.agent_names == ['mail_document_agent']
        self._fail('dry_run')
        return _dry_run()

    def start(self, *, actor, request):
        self.calls.append(('start', actor.id))
        assert request.client_request_id == 'client-1'
        self._fail('start')
        return self.start_status

    def status(self, *, actor, thread_id: str):
        self.calls.append(('status', thread_id))
        self._fail('status')
        return self.start_status

    def resume(self, *, actor, thread_id: str):
        self.calls.append(('resume', thread_id))
        self._fail('resume')
        return _status('completed')

    def cancel(self, *, actor, thread_id: str):
        self.calls.append(('cancel', thread_id))
        self._fail('cancel')
        return _status('cancelled')


@pytest.fixture
def api_client() -> tuple[TestClient, _FakeService]:
    app = create_app()
    fake = _FakeService()
    with TestClient(app) as client:
        app.state.review_workflow_service = fake
        client.cookies.set(
            'paraworks_csrf',
            'test-csrf-token',
            domain='testserver.local',
            path='/',
        )
        client.headers.update({'X-CSRF-Token': 'test-csrf-token'})
        yield client, fake


def _request_body() -> dict[str, object]:
    return {
        'source_refs': [
            {
                'source_type': 'gmail',
                'source_id': 'gmail:message-1',
                'version_or_signature': 'signature-1',
            }
        ],
        'agent_names': ['mail_document_agent'],
        'client_request_id': 'client-1',
    }


@pytest.mark.parametrize(
    ('diagnostic', 'expected'),
    [
        (
            _diagnostic(
                enabled=False,
                available=False,
                checkpoint_mode='disabled',
                durable=False,
            ),
            (False, False, 'disabled', False, None),
        ),
        (
            _diagnostic(
                enabled=True,
                available=True,
                checkpoint_mode='memory',
                durable=False,
            ),
            (True, True, 'memory', False, None),
        ),
        (
            _diagnostic(
                enabled=True,
                available=True,
                checkpoint_mode='postgres',
                durable=True,
            ),
            (True, True, 'postgres', True, None),
        ),
        (
            _diagnostic(
                enabled=True,
                available=False,
                checkpoint_mode='postgres',
                durable=False,
                error_code='checkpoint_unavailable',
            ),
            (True, False, 'postgres', False, 'checkpoint_unavailable'),
        ),
    ],
)
def test_diagnostic_feature_flag_and_readiness_matrix(
    api_client,
    diagnostic,
    expected,
) -> None:
    client, fake = api_client
    fake.diagnostic_response = diagnostic

    response = client.get('/api/v1/orchestration/v2/company-memory')

    assert response.status_code == 200
    body = response.json()
    assert (
        body['enabled'],
        body['available'],
        body['checkpoint_mode'],
        body['durable'],
        body['error_code'],
    ) == expected


def test_six_exact_v2_endpoints_are_registered(api_client) -> None:
    client, _ = api_client
    app = client.app
    prefix = '/api/v1/orchestration/v2/company-memory'
    registered = {
        (method, route.path)
        for route in app.routes
        if route.path.startswith(prefix)
        for method in route.methods
        if method not in {'HEAD', 'OPTIONS'}
    }

    assert registered == {
        ('GET', prefix),
        ('POST', f'{prefix}/dry-run'),
        ('POST', f'{prefix}/runs'),
        ('GET', f'{prefix}/runs/{{thread_id}}'),
        ('POST', f'{prefix}/runs/{{thread_id}}/resume'),
        ('POST', f'{prefix}/runs/{{thread_id}}/cancel'),
    }


def test_candidate_start_is_202_and_no_candidate_start_is_200(api_client) -> None:
    client, fake = api_client

    paused = client.post(
        '/api/v1/orchestration/v2/company-memory/runs',
        json=_request_body(),
    )
    fake.start_status = _status('completed')
    completed = client.post(
        '/api/v1/orchestration/v2/company-memory/runs',
        json=_request_body(),
    )

    assert paused.status_code == 202
    assert paused.json()['status'] == 'awaiting_human_review'
    assert completed.status_code == 200
    assert completed.json()['status'] == 'completed'


def test_disabled_new_run_is_hidden_but_existing_thread_routes_remain_available(
    api_client,
) -> None:
    client, fake = api_client
    fake.diagnostic_response = _diagnostic(
        enabled=False,
        available=False,
        checkpoint_mode='disabled',
        durable=False,
    )
    fake.failure_by_method = {'dry_run': 'not_found', 'start': 'not_found'}

    dry_run = client.post(
        '/api/v1/orchestration/v2/company-memory/dry-run',
        json=_request_body(),
    )
    start = client.post(
        '/api/v1/orchestration/v2/company-memory/runs',
        json=_request_body(),
    )
    status = client.get(
        '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id'
    )
    cancel = client.post(
        '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id/cancel'
    )

    assert dry_run.status_code == 404
    assert start.status_code == 404
    assert dry_run.json() == start.json() == {'detail': {'code': 'not_found'}}
    assert status.status_code == 200
    assert cancel.status_code == 200


def test_unready_checkpoint_blocks_new_work_and_resume_but_not_status_or_cancel(
    api_client,
) -> None:
    client, fake = api_client
    fake.diagnostic_response = _diagnostic(
        enabled=True,
        available=False,
        checkpoint_mode='postgres',
        durable=False,
        error_code='checkpoint_unavailable',
    )
    fake.failure_by_method = {
        'dry_run': 'checkpoint_unavailable',
        'start': 'checkpoint_unavailable',
        'resume': 'checkpoint_unavailable',
    }

    diagnostic = client.get('/api/v1/orchestration/v2/company-memory')
    dry_run = client.post(
        '/api/v1/orchestration/v2/company-memory/dry-run',
        json=_request_body(),
    )
    start = client.post(
        '/api/v1/orchestration/v2/company-memory/runs',
        json=_request_body(),
    )
    status = client.get(
        '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id'
    )
    resume = client.post(
        '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id/resume'
    )
    cancel = client.post(
        '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id/cancel'
    )

    assert diagnostic.status_code == status.status_code == cancel.status_code == 200
    assert dry_run.status_code == start.status_code == resume.status_code == 503
    assert dry_run.json() == start.json() == resume.json() == {
        'detail': {'code': 'checkpoint_unavailable'}
    }


@pytest.mark.parametrize(
    ('method', 'code', 'expected_status'),
    [
        ('start', 'invalid_input', 400),
        ('status', 'not_found', 404),
        ('resume', 'review_unresolved', 409),
        ('resume', 'runtime_version_unavailable', 409),
        ('cancel', 'invalid_state_transition', 409),
        ('dry_run', 'checkpoint_unavailable', 503),
        ('start', 'checkpoint_failed', 503),
        ('start', 'model_unavailable', 503),
    ],
)
def test_bounded_http_error_mapping(api_client, method, code, expected_status) -> None:
    client, fake = api_client
    fake.failure_by_method = {method: code}
    paths = {
        'dry_run': ('POST', '/api/v1/orchestration/v2/company-memory/dry-run'),
        'start': ('POST', '/api/v1/orchestration/v2/company-memory/runs'),
        'status': ('GET', '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id'),
        'resume': ('POST', '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id/resume'),
        'cancel': ('POST', '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id/cancel'),
    }
    http_method, path = paths[method]
    kwargs = {'json': _request_body()} if method in {'dry_run', 'start'} else {}

    response = client.request(http_method, path, **kwargs)

    assert response.status_code == expected_status
    assert response.json() == {'detail': {'code': code}}
    assert 'sensitive provider' not in response.text


def test_invalid_request_shape_is_400_without_service_call(api_client) -> None:
    client, fake = api_client
    before = list(fake.calls)

    response = client.post(
        '/api/v1/orchestration/v2/company-memory/runs',
        json={**_request_body(), 'prompt': 'do not accept raw prompt'},
    )

    assert response.status_code == 400
    assert response.json() == {'detail': {'code': 'invalid_input'}}
    assert fake.calls == before


@pytest.mark.parametrize(
    'request_kwargs',
    [
        {},
        {
            'content': '{',
            'headers': {'Content-Type': 'application/json'},
        },
    ],
)
def test_missing_or_malformed_json_is_bounded_400(api_client, request_kwargs) -> None:
    client, fake = api_client
    before = list(fake.calls)

    response = client.post(
        '/api/v1/orchestration/v2/company-memory/runs',
        **request_kwargs,
    )

    assert response.status_code == 400
    assert response.json() == {'detail': {'code': 'invalid_input'}}
    assert fake.calls == before


def test_status_body_exposes_only_the_approved_summary_fields(api_client) -> None:
    client, _ = api_client

    response = client.get(
        '/api/v1/orchestration/v2/company-memory/runs/opaque-thread-id'
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        'thread_id',
        'status',
        'review_item_count',
        'review_status_counts',
        'durable',
        'graph_version',
        'review_resolution_ready',
        'checkpoint_resumable',
        'resume_allowed',
        'retry_allowed',
        'created_at',
        'updated_at',
        'error_code',
        'resume_error_code',
    }
    serialized = response.text.lower()
    for forbidden in (
        'source_ref',
        'source_snippet',
        'review_item_id',
        'checkpoint_thread_id',
        'prompt',
        'model_output',
        'provider_error',
        'api_key',
    ):
        assert forbidden not in serialized
