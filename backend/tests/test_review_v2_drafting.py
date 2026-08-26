from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.contracts import (
    AgentCostBudgetDecision,
    AgentManifest,
    AgentRunCost,
    AgentRunResult,
    EvidencePacket,
    ReviewCandidate,
    TokenUsage,
)
from backend.app.agent_runtime.review_v2_agents import ReviewAgentCatalog
from backend.app.agent_runtime.review_v2_drafting import (
    ReviewDraftError,
    ReviewDraftService,
    build_permission_fingerprint,
    build_review_effect_key,
)
from backend.app.agent_runtime.review_v2_preflight import (
    PreparedReviewRequest,
    create_or_reuse_review_thread,
    prepare_review_request,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models.agent_runs import AgentRun
from backend.app.models.agent_workflows import AgentWorkflowThread
from backend.app.models.review import ReviewItem
from backend.app.models.source import Document, DocumentChunk, DocumentVersion, Source
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    DEFAULT_REVIEW_AGENT_NAMES,
    ReviewWorkflowRunRequest,
)

PROMPT_VERSIONS = {
    'mail_document_agent': 'mail-document-history:v1',
    'timeline_agent': 'timeline-extraction:v1',
    'history_agent': 'history-extraction:v1',
    'decision_record_agent': 'decision-record-extraction:v1',
    'todo_agent': 'todo-extraction:v1',
}


def _settings(**overrides) -> Settings:
    values = {
        '_env_file': None,
        'paraworks_demo_mode': True,
        'agent_runtime_security_scope_id': 'scope-review-drafting',
        'agent_runtime_fingerprint_secret': 'test-review-drafting-secret',
        'agent_runtime_fingerprint_key_version': 'test-v1',
        'agent_llm_max_estimated_cost_usd': 1.0,
        'agent_llm_max_evidence_messages': 12,
        'agent_llm_max_input_chars': 12_000,
    }
    values.update(overrides)
    return Settings(**values)


def _actor(subject_id: str = 'actor-1') -> DemoUser:
    return DemoUser(
        id=subject_id,
        email=f'{subject_id}@example.test',
        role='employee',
        permission_levels={'public', 'internal'},
        name=subject_id,
        title='Tester',
        department='Quality',
    )


def _seed_source(
    db: Session,
    *,
    sequence: int = 1,
    source_type: str = 'gmail',
    permission_level: str = 'internal',
) -> Source:
    prefix = 'message' if source_type == 'gmail' else 'file'
    source = Source(
        source_type=source_type,
        source_id=f'{source_type}:{prefix}-{sequence}',
        source_url=f'https://evidence.example.test/{source_type}/{sequence}',
        title=f'업무 근거 {sequence}',
        author='owner@example.test',
        permission_level=permission_level,
        raw_metadata={
            'content_signature': f'{source_type}-signature-{sequence}',
            'revision_id': f'revision-{sequence}',
        },
    )
    db.add(source)
    db.flush()
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v1',
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body='고객 데모 일정 때문에 QA를 완료하고 배포하기로 결정했습니다.',
    )
    db.add(version)
    db.flush()
    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text='고객 데모 일정 때문에 QA를 완료하고 배포하기로 결정했습니다.',
            source_snippet='고객 데모 일정 때문에 QA를 완료했습니다.',
            permission_level=permission_level,
            metadata_={
                'content_signature': source.raw_metadata['content_signature'],
                'parser_status': 'parsed',
                'section_path': 'body',
            },
        )
    )
    db.flush()
    return source


@dataclass
class _FakeAdapter:
    manifest: AgentManifest
    estimated_output_tokens: int = 64
    model_name: str = 'fake-review-model'
    model_route_version: str = 'fake-review-route:v1'
    budget_status: str = 'within_budget'
    candidate_factory: Callable[[EvidencePacket], list[ReviewCandidate]] | None = None
    on_run: Callable[[EvidencePacket], None] | None = None
    run_calls: int = 0
    preflight_calls: int = 0

    def preflight(self, packet: EvidencePacket) -> AgentCostBudgetDecision:
        self.preflight_calls += 1
        input_tokens = sum(max(1, len(message.text) // 4) for message in packet.messages)
        output_tokens = self.estimated_output_tokens if packet.messages else 0
        if not packet.messages:
            return AgentCostBudgetDecision(
                action='skip',
                reason='no_input',
                budget_status='no_input',
                model_name=self.model_name,
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
                estimated_cost_usd=0.0,
                budget_limit_usd=1.0,
                cache_hit=False,
            )
        if self.budget_status == 'over_budget':
            return AgentCostBudgetDecision(
                action='skip',
                reason='budget_exceeded',
                budget_status='over_budget',
                model_name=self.model_name,
                token_usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                estimated_cost_usd=2.0,
                budget_limit_usd=1.0,
                cache_hit=False,
            )
        return AgentCostBudgetDecision(
            action='run',
            reason='within_budget',
            budget_status='within_budget',
            model_name=self.model_name,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            estimated_cost_usd=0.000123,
            budget_limit_usd=1.0,
            cache_hit=False,
        )

    def run(self, packet: EvidencePacket) -> AgentRunResult:
        self.run_calls += 1
        if self.on_run is not None:
            self.on_run(packet)
        candidates = (
            self.candidate_factory(packet)
            if self.candidate_factory is not None
            else [_candidate_from_packet(packet)]
        )
        return AgentRunResult(
            agent_name=self.manifest.name,
            prompt_version=self.manifest.prompt_versions[0],
            candidates=candidates,
            cost=AgentRunCost(
                model_name=self.model_name,
                token_usage=TokenUsage(input_tokens=17, output_tokens=9),
                estimated_cost_usd=0.000321,
                cache_hit=False,
            ),
            cache_key=f'cache-{self.manifest.name}',
        )


def _candidate_from_packet(packet: EvidencePacket) -> ReviewCandidate:
    return ReviewCandidate(
        item_type='history_event',
        title='회사 히스토리 후보',
        summary='고객 데모 일정으로 QA 및 배포 계획이 변경되었습니다.',
        source_links=list(packet.source_links),
        source_snippets=list(packet.source_snippets),
        confidence_score=0.91,
        permission_level=packet.strictest_permission,
        uncertainty_reason=None,
        payload_fields={'reason': '고객 데모 일정이 근거에 명시되어 있습니다.'},
    )


def _manifest(name: str, *, prompt_version: str | None = None) -> AgentManifest:
    return AgentManifest(
        name=name,
        owner='Test Owner',
        input_contract='EvidencePacket',
        output_contract='AgentRunResult',
        prompt_versions=(prompt_version or PROMPT_VERSIONS[name],),
        supported_permissions=('public', 'internal', 'restricted'),
        capabilities=('review_draft',),
    )


def _catalog(
    *,
    selected_name: str = 'history_agent',
    selected_adapter: _FakeAdapter | None = None,
) -> tuple[ReviewAgentCatalog, _FakeAdapter]:
    chosen = selected_adapter or _FakeAdapter(_manifest(selected_name))
    adapters = [
        chosen if name == selected_name else _FakeAdapter(_manifest(name))
        for name in DEFAULT_REVIEW_AGENT_NAMES
    ]
    return ReviewAgentCatalog(adapters), chosen


def _session_factory(db_session: Session, *, tracked: list[Session] | None = None):
    local = sessionmaker(
        bind=db_session.get_bind(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    def create_session() -> Session:
        session = local()
        if tracked is not None:
            tracked.append(session)
        return session

    return create_session


def _request(source: Source, agent_name: str = 'history_agent') -> ReviewWorkflowRunRequest:
    return ReviewWorkflowRunRequest(
        source_refs=[
            {
                'source_type': source.source_type,
                'source_id': source.source_id,
                'version_or_signature': source.raw_metadata['content_signature'],
            }
        ],
        agent_names=[agent_name],
        client_request_id=f'test-{uuid4().hex}',
    )


def _prepare(
    db: Session,
    *,
    source: Source,
    catalog: ReviewAgentCatalog,
    settings: Settings,
    actor: DemoUser | None = None,
) -> PreparedReviewRequest:
    prepared = prepare_review_request(
        db,
        request=_request(source),
        actor=actor or _actor(),
        registry=catalog.registry,
        settings=settings,
    )
    db.rollback()
    return prepared


def _create_workflow(
    db: Session,
    *,
    source: Source,
    catalog: ReviewAgentCatalog,
    settings: Settings,
    actor: DemoUser | None = None,
) -> tuple[PreparedReviewRequest, str]:
    workflow_actor = actor or _actor()
    request = _request(source)
    prepared = prepare_review_request(
        db,
        request=request,
        actor=workflow_actor,
        registry=catalog.registry,
        settings=settings,
    )
    result = create_or_reuse_review_thread(
        db,
        prepared=prepared,
        request=request,
        actor=workflow_actor,
        settings=settings,
    )
    return prepared, result.thread.thread_id


def _row_counts(factory) -> tuple[int, int, int]:
    with factory() as db:
        return (
            db.scalar(select(func.count()).select_from(AgentWorkflowThread)) or 0,
            db.scalar(select(func.count()).select_from(AgentRun)) or 0,
            db.scalar(select(func.count()).select_from(ReviewItem)) or 0,
        )


def test_preview_never_invokes_provider_or_writes_rows(db_session) -> None:
    settings = _settings(agent_llm_max_evidence_messages=1)
    catalog, adapter = _catalog()
    source = _seed_source(db_session, sequence=1)
    second_source = _seed_source(db_session, sequence=2)
    same_created_at = datetime(2026, 8, 27, 9, 0, tzinfo=UTC)
    source.created_at = same_created_at
    second_source.created_at = same_created_at
    db_session.commit()
    prepared = prepare_review_request(
        db_session,
        request=ReviewWorkflowRunRequest(
            source_refs=[
                {
                    'source_type': current.source_type,
                    'source_id': current.source_id,
                    'version_or_signature': current.raw_metadata['content_signature'],
                }
                for current in (source, second_source)
            ],
            agent_names=['history_agent'],
        ),
        actor=_actor(),
        registry=catalog.registry,
        settings=settings,
    )
    db_session.rollback()
    factory = _session_factory(db_session)
    before = _row_counts(factory)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    preview = service.preview_prepared(
        prepared=prepared,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    assert preview.source_count == 2
    assert preview.agent_names == ['history_agent']
    assert preview.budget_status == 'within_budget'
    assert adapter.preflight_calls == 1
    assert adapter.run_calls == 0
    assert _row_counts(factory) == before == (0, 0, 0)


def test_over_budget_preflight_never_invokes_provider_or_writes_rows(
    db_session,
) -> None:
    settings = _settings()
    expensive = _FakeAdapter(_manifest('history_agent'), budget_status='over_budget')
    catalog, adapter = _catalog(selected_adapter=expensive)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    before = _row_counts(factory)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'budget_exceeded'
    assert adapter.run_calls == 0
    assert _row_counts(factory) == before == (1, 0, 0)
    with factory() as db:
        thread = db.get(AgentWorkflowThread, thread_id)
        assert thread is not None
        assert (thread.status, thread.state_version, thread.lease_token) == (
            'created',
            0,
            None,
        )


def test_different_security_scope_cannot_draft_known_thread_id(db_session) -> None:
    owner_settings = _settings()
    catalog, adapter = _catalog()
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=owner_settings,
    )
    factory = _session_factory(db_session)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=_settings(agent_runtime_security_scope_id='other-scope'),
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'not_found'
    assert adapter.run_calls == 0
    assert _row_counts(factory) == (1, 0, 0)


def test_draft_runs_provider_outside_transaction_and_lock(db_session) -> None:
    settings = _settings()
    tracked_sessions: list[Session] = []
    factory = _session_factory(db_session, tracked=tracked_sessions)

    def assert_no_transaction(_packet: EvidencePacket) -> None:
        assert tracked_sessions
        assert all(not session.in_transaction() for session in tracked_sessions)

    adapter = _FakeAdapter(_manifest('history_agent'), on_run=assert_no_transaction)
    catalog, _ = _catalog(selected_adapter=adapter)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    result = service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    assert len(result.review_item_ids) == 1
    assert adapter.run_calls == 1


def test_effect_replay_skips_model_and_reuses_agent_run_and_candidates(
    db_session,
) -> None:
    settings = _settings()
    catalog, adapter = _catalog()
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    first = service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )
    with factory() as db:
        review_item = db.get(ReviewItem, first.review_item_ids[0])
        assert review_item is not None
        review_item.status = 'approved'
        db.commit()
    second = service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    assert second.review_item_ids == first.review_item_ids
    assert second.review_status_counts == {
        'pending_review': 0,
        'approved': 1,
        'rejected': 0,
        'needs_more_evidence': 0,
    }
    assert adapter.run_calls == 1
    assert _row_counts(factory) == (1, 1, 1)


def test_prompt_or_evidence_change_invalidates_effect_key(db_session) -> None:
    del db_session
    settings = _settings()
    permission_fingerprint = build_permission_fingerprint(
        settings=settings,
        allowed_permission_levels=('public', 'internal'),
    )
    values = {
        'settings': settings,
        'workflow_thread_id': 'thread-1',
        'agent_name': 'history_agent',
        'prompt_version': 'history-extraction:v1',
        'model_route_version': 'route:v1:fake',
        'evidence_hash': '1' * 64,
        'permission_fingerprint': permission_fingerprint,
        'selection_policy_version': COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    }

    baseline = build_review_effect_key(**values)
    prompt_changed = build_review_effect_key(
        **{**values, 'prompt_version': 'history-extraction:v2'}
    )
    evidence_changed = build_review_effect_key(
        **{**values, 'evidence_hash': '2' * 64}
    )
    permission_changed = build_review_effect_key(
        **{
            **values,
            'permission_fingerprint': build_permission_fingerprint(
                settings=settings,
                allowed_permission_levels=('internal',),
            ),
        }
    )
    model_route_changed = build_review_effect_key(
        **{**values, 'model_route_version': 'route:v2:fake'}
    )
    selection_policy_changed = build_review_effect_key(
        **{**values, 'selection_policy_version': 'selection:v2'}
    )

    assert len({
        baseline,
        prompt_changed,
        evidence_changed,
        permission_changed,
        model_route_changed,
        selection_policy_changed,
    }) == 6


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 27, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def test_lease_expiry_discards_model_result_before_candidate_write(
    db_session,
) -> None:
    settings = _settings()
    clock = _Clock()

    def expire_lease(_packet: EvidencePacket) -> None:
        clock.value += timedelta(minutes=2)

    adapter = _FakeAdapter(_manifest('history_agent'), on_run=expire_lease)
    catalog, _ = _catalog(selected_adapter=adapter)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
        lease_ttl_seconds=30,
        now=clock,
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'concurrent_resume'
    assert adapter.run_calls == 1
    assert _row_counts(factory) == (1, 0, 0)


def test_changed_lease_expiry_discards_model_result_before_candidate_write(
    db_session,
) -> None:
    settings = _settings()
    clock = _Clock()
    factory = _session_factory(db_session)
    thread_ids: list[str] = []

    def rewrite_lease_expiry(_packet: EvidencePacket) -> None:
        with factory() as db:
            thread = db.get(AgentWorkflowThread, thread_ids[0])
            assert thread is not None
            assert thread.lease_expires_at is not None
            thread.lease_expires_at += timedelta(minutes=1)
            db.commit()

    adapter = _FakeAdapter(_manifest('history_agent'), on_run=rewrite_lease_expiry)
    catalog, _ = _catalog(selected_adapter=adapter)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    thread_ids.append(thread_id)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
        now=clock,
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'concurrent_resume'
    assert _row_counts(factory) == (1, 0, 0)


def test_changed_thread_evidence_hash_discards_model_result_before_candidate_write(
    db_session,
) -> None:
    settings = _settings()
    factory = _session_factory(db_session)
    thread_ids: list[str] = []

    def rewrite_evidence_hash(_packet: EvidencePacket) -> None:
        with factory() as db:
            thread = db.get(AgentWorkflowThread, thread_ids[0])
            assert thread is not None
            thread.evidence_version_hash = 'f' * 64
            db.commit()

    adapter = _FakeAdapter(_manifest('history_agent'), on_run=rewrite_evidence_hash)
    catalog, _ = _catalog(selected_adapter=adapter)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    thread_ids.append(thread_id)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'evidence_changed'
    assert _row_counts(factory) == (1, 0, 0)


@pytest.mark.parametrize(
    ('mutation', 'expected_code'),
    [('permission', 'permission_denied'), ('signature', 'evidence_changed')],
)
def test_permission_or_signature_change_discards_model_result(
    db_session,
    mutation: str,
    expected_code: str,
) -> None:
    settings = _settings()
    factory = _session_factory(db_session)
    source = _seed_source(db_session)
    source_id = source.id
    db_session.commit()

    def mutate_current_evidence(_packet: EvidencePacket) -> None:
        with factory() as db:
            current = db.get(Source, source_id)
            assert current is not None
            if mutation == 'permission':
                current.permission_level = 'restricted'
            else:
                current.raw_metadata['content_signature'] = 'changed-signature'
            db.commit()

    adapter = _FakeAdapter(
        _manifest('history_agent'),
        on_run=mutate_current_evidence,
    )
    catalog, _ = _catalog(selected_adapter=adapter)
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == expected_code
    assert _row_counts(factory) == (1, 0, 0)
    with factory() as db:
        thread = db.get(AgentWorkflowThread, thread_id)
        assert thread is not None
        assert thread.status == 'created'
        assert thread.lease_token is None


def test_candidate_without_evidence_is_rejected(db_session) -> None:
    settings = _settings()

    def source_less(_packet: EvidencePacket) -> list[ReviewCandidate]:
        return [
            ReviewCandidate(
                item_type='history_event',
                title='근거 없는 후보',
                summary='이 후보는 저장되면 안 됩니다.',
                source_links=[],
                source_snippets=[],
                confidence_score=0.9,
                permission_level='internal',
                payload_fields={'reason': 'none'},
            )
        ]

    adapter = _FakeAdapter(
        _manifest('history_agent'),
        candidate_factory=source_less,
    )
    catalog, _ = _catalog(selected_adapter=adapter)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'invalid_input'
    assert _row_counts(factory) == (1, 0, 0)


def test_draft_persists_token_cost_and_cache_metadata(db_session) -> None:
    settings = _settings()
    catalog, _ = _catalog()
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    )

    result = service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    with factory() as db:
        agent_run = db.scalar(select(AgentRun))
        review_item = db.get(ReviewItem, result.review_item_ids[0])
        thread = db.get(AgentWorkflowThread, thread_id)
        assert agent_run is not None
        assert review_item is not None
        assert thread is not None
        assert (
            agent_run.input_tokens,
            agent_run.output_tokens,
            agent_run.total_tokens,
            agent_run.estimated_cost_usd,
        ) == (17, 9, 26, 0.000321)
        assert agent_run.metadata_['cache_hit'] is False
        assert agent_run.cache_key == agent_run.effect_key
        assert len(agent_run.cache_key) == 64
        assert agent_run.metadata_['model_route_version'] == 'fake-review-route:v1'
        assert agent_run.metadata_['selection_policy_version'] == (
            COMPANY_MEMORY_SELECTION_POLICY_VERSION
        )
        assert len(agent_run.metadata_['permission_fingerprint']) == 64
        assert review_item.payload['agent_run_id'] == agent_run.id
        assert review_item.payload['cache_key'] == agent_run.effect_key
        assert review_item.payload['token_usage']['total_tokens'] == 26
        assert review_item.payload['source_ids'] == ['gmail:message-1']
        assert review_item.permission_level == 'internal'
        assert thread.status == 'checkpoint_pending'
        assert thread.lease_token is None


def _seed_predecessor(
    db: Session,
    *,
    source_ids: list[str],
    created_at: datetime,
    scope: str = 'scope-review-drafting',
    permission_level: str = 'internal',
) -> ReviewItem:
    thread_id = uuid4().hex
    db.add(
        AgentWorkflowThread(
            thread_id=thread_id,
            workflow_name='company-memory-review',
            graph_version='company-memory-review-v2.0',
            checkpoint_thread_id=f'review-v2:{uuid4().hex}',
            checkpoint_store='memory',
            owner_subject_id='prior-owner',
            security_scope_id=scope,
            input_hash='a' * 64,
            evidence_version_hash='b' * 64,
            status='needs_more_evidence',
        )
    )
    item = ReviewItem(
        item_type='history_event',
        payload={
            'title': '이전 히스토리 후보',
            'summary': '추가 근거가 필요합니다.',
            'agent_name': 'history_agent',
            'source_ids': source_ids,
        },
        source_links=['https://evidence.example.test/prior'],
        source_snippets=['prior evidence'],
        confidence_score=0.7,
        permission_level=permission_level,
        status='needs_more_evidence',
        workflow_thread_id=thread_id,
        candidate_key=uuid4().hex,
        created_at=created_at,
    )
    db.add(item)
    db.flush()
    return item


def test_new_exact_source_set_links_latest_needs_more_predecessor(
    db_session,
) -> None:
    settings = _settings()
    older = _seed_predecessor(
        db_session,
        source_ids=['gmail:message-1'],
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )
    newest = _seed_predecessor(
        db_session,
        source_ids=[' gmail:message-1 ', 'gmail:message-1'],
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
        permission_level='restricted',
    )
    catalog, _ = _catalog()
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    result = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    ).draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    with factory() as db:
        created = db.get(ReviewItem, result.review_item_ids[0])
        assert created is not None
        assert created.predecessor_review_item_id == newest.id
        assert created.predecessor_review_item_id != older.id


def test_predecessor_link_never_uses_fuzzy_or_partial_source_match(
    db_session,
) -> None:
    settings = _settings()
    _seed_predecessor(
        db_session,
        source_ids=['gmail:message-1', 'drive:file-extra'],
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
    )
    _seed_predecessor(
        db_session,
        source_ids=['gmail:message'],
        created_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    catalog, _ = _catalog()
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    factory = _session_factory(db_session)
    result = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
    ).draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    with factory() as db:
        created = db.get(ReviewItem, result.review_item_ids[0])
        assert created is not None
        assert created.predecessor_review_item_id is None
