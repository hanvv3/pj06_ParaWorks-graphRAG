from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from backend.app.agent_runtime import AgentRegistry
from backend.app.agent_runtime.canonical_sources import ReviewWorkflowPreflightError
from backend.app.agent_runtime.review_v2_preflight import (
    create_or_reuse_review_thread,
    prepare_review_request,
)
from backend.app.agents.mail_document_agent import MAIL_DOCUMENT_AGENT_MANIFEST
from backend.app.agents.memory_extraction_agent import (
    DECISION_RECORD_AGENT_MANIFEST,
    HISTORY_AGENT_MANIFEST,
    TIMELINE_AGENT_MANIFEST,
    TODO_AGENT_MANIFEST,
)
from backend.app.api.v1 import integrations
from backend.app.connectors.base import ConnectorManifest, SourceEvent
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import USERS
from backend.app.models import AgentWorkflowThread, AuditLog, ReviewItem, Source
from backend.app.schemas.review_workflow import (
    DEFAULT_REVIEW_AGENT_NAMES,
    ReviewWorkflowRunRequest,
)


@dataclass
class FakeConnector:
    source_type: str
    events: list[SourceEvent]

    @property
    def manifest(self) -> ConnectorManifest:
        return ConnectorManifest(
            connector_type=self.source_type,
            display_name=self.source_type,
            mode='mock',
            auth_type='oauth',
            required_scopes=(),
            sync_strategy='incremental',
            cost_policy='test only',
        )

    def fetch_events(self) -> list[SourceEvent]:
        return self.events


def _event(
    *,
    source_type: str = 'gmail',
    source_id: str = 'gmail:waterline-1',
    signature: str = 'gmail:waterline-1:v1',
) -> SourceEvent:
    return SourceEvent(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://example.test/{source_id}',
        title='Waterline evidence',
        body='Project review evidence with an explicit canonical version.',
        author='owner@example.test',
        participants=['owner@example.test'],
        timestamp=datetime(2026, 8, 27, 9, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'content_signature': signature,
            'document_version': 'v1',
            'source_snippet': 'Project review evidence.',
        },
    )


def _settings(*, v2_enabled: bool) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=True,
        langgraph_review_v2_enabled=v2_enabled,
        agent_runtime_security_scope_id='waterline-scope',
        agent_runtime_fingerprint_secret='waterline-test-secret',
        agent_runtime_fingerprint_key_version='waterline-test-v1',
    )


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    for manifest in (
        MAIL_DOCUMENT_AGENT_MANIFEST,
        TIMELINE_AGENT_MANIFEST,
        HISTORY_AGENT_MANIFEST,
        DECISION_RECORD_AGENT_MANIFEST,
        TODO_AGENT_MANIFEST,
    ):
        registry.register(manifest)
    return registry


def _request(refs: list[dict[str, str]], client_request_id: str) -> ReviewWorkflowRunRequest:
    return ReviewWorkflowRunRequest(
        source_refs=refs,
        agent_names=list(DEFAULT_REVIEW_AGENT_NAMES),
        client_request_id=client_request_id,
    )


def test_v2_mode_marks_explicit_waterline_and_suppresses_legacy_candidates(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(v2_enabled=True)
    client.app.dependency_overrides[get_settings] = lambda: settings
    connector = FakeConnector('gmail', [_event()])
    monkeypatch.setattr(integrations, 'get_sync_connector', lambda *args, **kwargs: connector)
    monkeypatch.setattr(
        integrations,
        '_run_connector_agent_review',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('legacy agent bridge ran')),
    )
    monkeypatch.setattr(
        integrations,
        'create_project_assignment_review_items',
        lambda db: (_ for _ in ()).throw(AssertionError('legacy project generation ran')),
    )

    response = client.post('/api/v1/integrations/gmail/sync')

    assert response.status_code == 200
    payload = response.json()
    assert payload['created_review_items'] == 0
    assert payload['changed_source_refs'] == [
        {
            'source_type': 'gmail',
            'source_id': 'gmail:waterline-1',
            'version_or_signature': 'gmail:waterline-1:v1',
        }
    ]
    source = db_session.scalar(select(Source).where(Source.source_id == 'gmail:waterline-1'))
    assert source is not None
    assert source.raw_metadata['last_changed_sync_job_id'] == payload['job_id']
    assert source.raw_metadata['review_batch_mode'] == 'v2_explicit'
    assert source.raw_metadata['review_batch_signature'] == 'gmail:waterline-1:v1'
    audit = db_session.scalar(select(AuditLog).order_by(AuditLog.id.desc()))
    assert audit is not None
    assert audit.metadata_['changed_source_count'] == 1
    assert len(audit.metadata_['review_batch_hmac']) == 64
    assert 'changed_source_ids' not in audit.metadata_


def test_v2_mode_no_change_recovery_never_runs_legacy_candidates(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(v2_enabled=True)
    client.app.dependency_overrides[get_settings] = lambda: settings
    connector = FakeConnector('gmail', [_event()])
    monkeypatch.setattr(integrations, 'get_sync_connector', lambda *args, **kwargs: connector)
    first = client.post('/api/v1/integrations/gmail/sync')
    assert first.status_code == 200
    monkeypatch.setattr(
        integrations,
        '_run_connector_agent_review',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('legacy agent recovery ran')),
    )
    monkeypatch.setattr(
        integrations,
        'create_project_assignment_review_items',
        lambda db: (_ for _ in ()).throw(AssertionError('legacy project recovery ran')),
    )

    second = client.post('/api/v1/integrations/gmail/sync')

    assert second.status_code == 200
    assert second.json()['changed_source_refs'] == []
    assert second.json()['created_review_items'] == 0


def test_legacy_success_marks_current_signature_legacy_inline(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(v2_enabled=False)
    client.app.dependency_overrides[get_settings] = lambda: settings
    connector = FakeConnector('gmail', [_event()])
    monkeypatch.setattr(integrations, 'get_sync_connector', lambda *args, **kwargs: connector)

    response = client.post('/api/v1/integrations/gmail/sync')

    assert response.status_code == 200
    source = db_session.scalar(select(Source).where(Source.source_id == 'gmail:waterline-1'))
    assert source is not None
    assert source.raw_metadata['review_batch_mode'] == 'legacy_inline'
    assert source.raw_metadata['review_batch_signature'] == 'gmail:waterline-1:v1'
    assert db_session.query(ReviewItem).count() == response.json()['created_review_items']
    audit = db_session.scalar(select(AuditLog).order_by(AuditLog.id.desc()))
    assert audit is not None
    assert audit.metadata_['changed_source_count'] == 1
    assert len(audit.metadata_['review_batch_hmac']) == 64
    assert 'changed_source_ids' not in audit.metadata_


@pytest.mark.parametrize('failure_stage', ['review', 'project'])
def test_failed_legacy_generation_does_not_advance_waterline(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    settings = _settings(v2_enabled=False)
    client.app.dependency_overrides[get_settings] = lambda: settings
    connector = FakeConnector('gmail', [_event()])
    monkeypatch.setattr(integrations, 'get_sync_connector', lambda *args, **kwargs: connector)
    if failure_stage == 'review':
        monkeypatch.setattr(
            integrations,
            '_run_connector_agent_review',
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError('review failed')),
        )
    else:
        monkeypatch.setattr(integrations, '_run_connector_agent_review', lambda **kwargs: 0)
        monkeypatch.setattr(
            integrations,
            'create_project_assignment_review_items',
            lambda db: (_ for _ in ()).throw(RuntimeError('project failed')),
        )

    with pytest.raises(RuntimeError, match=f'{failure_stage} failed'):
        client.post('/api/v1/integrations/gmail/sync')

    db_session.expire_all()
    source = db_session.scalar(select(Source).where(Source.source_id == 'gmail:waterline-1'))
    assert source is not None
    assert source.raw_metadata.get('review_batch_mode') is None
    assert source.raw_metadata.get('review_batch_signature') is None


@pytest.mark.parametrize('marker_mode', [None, 'legacy_inline'])
def test_pre_waterline_or_legacy_marker_is_rejected_by_v2_preflight(
    db_session,
    marker_mode: str | None,
) -> None:
    metadata = {'content_signature': 'gmail:waterline-1:v1'}
    if marker_mode is not None:
        metadata.update(
            review_batch_mode=marker_mode,
            review_batch_signature='gmail:waterline-1:v1',
        )
    source = Source(
        source_type='gmail',
        source_id='gmail:waterline-1',
        source_url='https://example.test/gmail:waterline-1',
        title='Waterline evidence',
        permission_level='internal',
        raw_metadata=metadata,
    )
    db_session.add(source)
    db_session.commit()
    settings = _settings(v2_enabled=True)
    request = _request(
        [
            {
                'source_type': 'gmail',
                'source_id': 'gmail:waterline-1',
                'version_or_signature': 'gmail:waterline-1:v1',
            }
        ],
        'pre-waterline',
    )
    prepared = prepare_review_request(
        db_session,
        request=request,
        actor=USERS['admin'],
        registry=_registry(),
        settings=settings,
    )

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        create_or_reuse_review_thread(
            db_session,
            prepared=prepared,
            request=request,
            actor=USERS['admin'],
            settings=settings,
        )

    assert exc_info.value.code == 'evidence_changed'
    assert db_session.query(AgentWorkflowThread).count() == 0


def test_rollback_v1_does_not_process_batch_with_existing_v2_thread(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_settings = _settings(v2_enabled=True)
    client.app.dependency_overrides[get_settings] = lambda: v2_settings
    connector = FakeConnector('gmail', [_event()])
    monkeypatch.setattr(integrations, 'get_sync_connector', lambda *args, **kwargs: connector)
    sync_response = client.post('/api/v1/integrations/gmail/sync')
    refs = sync_response.json()['changed_source_refs']
    request = _request(refs, 'owned-v2-batch')
    prepared = prepare_review_request(
        db_session,
        request=request,
        actor=USERS['admin'],
        registry=_registry(),
        settings=v2_settings,
    )
    created = create_or_reuse_review_thread(
        db_session,
        prepared=prepared,
        request=request,
        actor=USERS['admin'],
        settings=v2_settings,
    )
    assert created.created is True
    client.app.dependency_overrides[get_settings] = lambda: _settings(v2_enabled=False)
    monkeypatch.setattr(
        integrations,
        '_run_connector_agent_review',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('owned batch ran in V1')),
    )
    monkeypatch.setattr(
        integrations,
        'create_project_assignment_review_items',
        lambda db: (_ for _ in ()).throw(AssertionError('owned project batch ran in V1')),
    )

    rollback_response = client.post('/api/v1/integrations/gmail/sync')

    assert rollback_response.status_code == 200
    assert rollback_response.json()['created_review_items'] == 0
    source = db_session.scalar(select(Source).where(Source.source_id == 'gmail:waterline-1'))
    assert source is not None
    assert source.raw_metadata['review_batch_mode'] == 'v2_explicit'


def test_rollback_v1_does_not_suppress_mixed_canonical_and_legacy_ids(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_settings = _settings(v2_enabled=True)
    client.app.dependency_overrides[get_settings] = lambda: v2_settings
    connector = FakeConnector(
        'gmail',
        [
            _event(
                source_id='gmail-legacy-message',
                signature='gmail-legacy-message:v1',
            ),
            _event(
                source_type='gmail_attachment',
                source_id='gmail_attachment:legacy-message:file-1',
                signature='gmail_attachment:legacy-message:file-1:v1',
            ),
        ],
    )
    monkeypatch.setattr(integrations, 'get_sync_connector', lambda *args, **kwargs: connector)
    sync_response = client.post('/api/v1/integrations/gmail/sync')
    refs = sync_response.json()['changed_source_refs']
    assert [ref['source_id'] for ref in refs] == ['gmail_attachment:legacy-message:file-1']
    request = _request(refs, 'mixed-owned-v2-batch')
    prepared = prepare_review_request(
        db_session,
        request=request,
        actor=USERS['admin'],
        registry=_registry(),
        settings=v2_settings,
    )
    create_or_reuse_review_thread(
        db_session,
        prepared=prepared,
        request=request,
        actor=USERS['admin'],
        settings=v2_settings,
    )
    calls: list[list[str]] = []
    client.app.dependency_overrides[get_settings] = lambda: _settings(v2_enabled=False)
    monkeypatch.setattr(
        integrations,
        '_run_connector_agent_review',
        lambda **kwargs: calls.append(kwargs['source_ids']) or 0,
    )
    monkeypatch.setattr(integrations, 'create_project_assignment_review_items', lambda db: [])

    rollback_response = client.post('/api/v1/integrations/gmail/sync')

    assert rollback_response.status_code == 200
    assert calls == [
        [
            'gmail-legacy-message',
            'gmail_attachment:legacy-message:file-1',
        ]
    ]


def test_slack_sync_behavior_is_unchanged_when_v2_flag_changes(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        _event(
            source_type='slack',
            source_id='slack-waterline-off',
            signature='slack-waterline-off:v1',
        ),
        _event(
            source_type='slack',
            source_id='slack-waterline-on',
            signature='slack-waterline-on:v1',
        ),
    ]
    calls: list[list[str]] = []
    connector_calls = 0

    def connector_factory(*args, **kwargs):
        nonlocal connector_calls
        connector = FakeConnector('slack', [events[connector_calls]])
        connector_calls += 1
        return connector

    monkeypatch.setattr(integrations, 'get_sync_connector', connector_factory)
    monkeypatch.setattr(
        integrations,
        '_run_connector_agent_review',
        lambda **kwargs: calls.append(kwargs['source_ids']) or 0,
    )
    monkeypatch.setattr(integrations, 'create_project_assignment_review_items', lambda db: [])

    client.app.dependency_overrides[get_settings] = lambda: _settings(v2_enabled=False)
    disabled = client.post('/api/v1/integrations/slack/sync')
    client.app.dependency_overrides[get_settings] = lambda: _settings(v2_enabled=True)
    enabled = client.post('/api/v1/integrations/slack/sync')

    assert disabled.status_code == enabled.status_code == 200
    assert calls == [['slack-waterline-off'], ['slack-waterline-on']]
    assert disabled.json()['changed_source_refs'] == []
    assert enabled.json()['changed_source_refs'] == []
