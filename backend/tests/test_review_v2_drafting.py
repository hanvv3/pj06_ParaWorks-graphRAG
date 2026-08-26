from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
from backend.app.agent_runtime.review_v2_agents import (
    ReviewAgentCatalog,
    build_review_agent_catalog,
)
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
from backend.app.agents.mail_document_agent import (
    MAIL_DOCUMENT_AGENT_MANIFEST,
)
from backend.app.agents.memory_extraction_agent import (
    DECISION_RECORD_AGENT_MANIFEST,
    HISTORY_AGENT_MANIFEST,
    TIMELINE_AGENT_MANIFEST,
    TODO_AGENT_MANIFEST,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models.agent_runs import AgentRun
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.models.source import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    DEFAULT_REVIEW_AGENT_NAMES,
    ReviewWorkflowRunRequest,
)

APPROVED_MANIFESTS = {
    manifest.name: manifest
    for manifest in (
        MAIL_DOCUMENT_AGENT_MANIFEST,
        TIMELINE_AGENT_MANIFEST,
        HISTORY_AGENT_MANIFEST,
        DECISION_RECORD_AGENT_MANIFEST,
        TODO_AGENT_MANIFEST,
    )
}
ALLOWED_ITEM_TYPES = {
    'mail_document_agent': frozenset(
        {'timeline_event', 'history_event', 'decision_record', 'todo'}
    ),
    'timeline_agent': frozenset({'timeline_event'}),
    'history_agent': frozenset({'history_event'}),
    'decision_record_agent': frozenset({'decision_record'}),
    'todo_agent': frozenset({'todo'}),
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

    @property
    def allowed_item_types(self) -> frozenset[str]:
        return ALLOWED_ITEM_TYPES[self.manifest.name]

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
            else [
                _candidate_from_packet(
                    packet,
                    item_type=next(iter(self.allowed_item_types)),
                )
            ]
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


def _candidate_from_packet(
    packet: EvidencePacket,
    *,
    item_type: str = 'history_event',
) -> ReviewCandidate:
    return ReviewCandidate(
        item_type=item_type,
        title='회사 히스토리 후보',
        summary='고객 데모 일정으로 QA 및 배포 계획이 변경되었습니다.',
        source_links=list(packet.source_links),
        source_snippets=list(packet.source_snippets),
        confidence_score=0.91,
        permission_level=packet.strictest_permission,
        uncertainty_reason=None,
        payload_fields={'reason': '고객 데모 일정이 근거에 명시되어 있습니다.'},
    )


def _manifest(name: str) -> AgentManifest:
    return APPROVED_MANIFESTS[name]


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


def _request(
    source: Source,
    agent_names: str | Sequence[str] = 'history_agent',
) -> ReviewWorkflowRunRequest:
    requested_agents = [agent_names] if isinstance(agent_names, str) else list(agent_names)
    return ReviewWorkflowRunRequest(
        source_refs=[
            {
                'source_type': source.source_type,
                'source_id': source.source_id,
                'version_or_signature': source.raw_metadata['content_signature'],
            }
        ],
        agent_names=requested_agents,
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
    agent_name: str = 'history_agent',
    agent_names: Sequence[str] | None = None,
) -> tuple[PreparedReviewRequest, str]:
    workflow_actor = actor or _actor()
    request = _request(source, agent_names or agent_name)
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


def test_failed_exact_version_parser_metadata_and_body_fallback_reach_mail_adapter(
    db_session,
) -> None:
    settings = _settings()
    source = _seed_source(db_session, source_type='drive')
    version = db_session.scalar(
        select(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.source_id == source.id, DocumentVersion.version == 'v1')
    )
    chunk = db_session.scalar(
        select(DocumentChunk).where(DocumentChunk.version_id == version.id)
    )
    assert version is not None
    assert chunk is not None
    chunk.text = '   '
    chunk.source_snippet = ''
    version.body = '문서 본문에서 고객 데모 QA 완료와 배포 결정을 확인했습니다.'
    later_version = DocumentVersion(
        document_id=version.document_id,
        version='v2',
        body='현재 선택되지 않은 다른 버전입니다.',
    )
    db_session.add(later_version)
    db_session.flush()
    db_session.add_all(
        [
            DocumentParserRun(
                document_id=version.document_id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='older-parser',
                parser_status='parsed',
                parser_status_reason=None,
                mime_type='application/pdf',
                document_version_label='v1',
                revision_id='older-revision',
                content_signature='older-signature',
                chunk_count=1,
                finished_at=datetime(2026, 8, 26, tzinfo=UTC),
            ),
            DocumentParserRun(
                document_id=version.document_id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='authoritative-parser',
                parser_status='failed',
                parser_status_reason='encrypted_body',
                mime_type='application/pdf',
                document_version_label='v1',
                revision_id='selected-revision',
                content_signature='selected-signature',
                chunk_count=0,
                finished_at=datetime(2026, 8, 27, tzinfo=UTC),
            ),
            DocumentParserRun(
                document_id=version.document_id,
                document_version_id=later_version.id,
                source_id=source.id,
                parser_name='foreign-version-parser',
                parser_status='unsupported',
                parser_status_reason='wrong_version',
                mime_type='application/octet-stream',
                document_version_label='v2',
                revision_id='foreign-revision',
                content_signature='foreign-signature',
                chunk_count=0,
                finished_at=datetime(2026, 8, 28, tzinfo=UTC),
            ),
        ]
    )
    db_session.commit()

    captured: list[EvidencePacket] = []
    base_catalog = build_review_agent_catalog(settings)
    mail_adapter = base_catalog.get('mail_document_agent')
    delegate_model = mail_adapter.agent.model

    class CapturingMailModel:
        def extract(self, packet: EvidencePacket):
            captured.append(packet)
            return delegate_model.extract(packet)

    adapters = [base_catalog.get(name) for name in DEFAULT_REVIEW_AGENT_NAMES]
    adapters[0] = replace(
        mail_adapter,
        agent=replace(
            mail_adapter.agent,
            model=CapturingMailModel(),
        ),
    )
    catalog = ReviewAgentCatalog(adapters)
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
        agent_name='mail_document_agent',
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

    assert len(captured) == 1
    message = captured[0].messages[0]
    assert message.text == version.body
    assert message.metadata['fallback_body'] is True
    assert message.metadata['parser_name'] == 'authoritative-parser'
    assert message.metadata['parser_status'] == 'failed'
    assert message.metadata['parser_status_reason'] == 'encrypted_body'
    assert message.metadata['document_version_label'] == 'v1'
    assert message.metadata['revision_id'] == 'selected-revision'
    assert message.metadata['content_signature'] == 'selected-signature'
    with factory() as db:
        review_item = db.get(ReviewItem, result.review_item_ids[0])
        assert review_item is not None
        assert 'failed(encrypted_body)' in review_item.payload['uncertainty_reason']


def test_failed_agent_run_is_not_replayed_as_cached_effect(db_session) -> None:
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
    service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )
    with factory() as db:
        run = db.scalar(select(AgentRun))
        assert run is not None
        run.status = 'failed'
        db.commit()

    with pytest.raises(ReviewDraftError) as exc_info:
        service.draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'concurrent_resume'
    assert adapter.run_calls == 2
    assert _row_counts(factory) == (1, 1, 1)


def test_cache_replay_excludes_candidate_associated_with_foreign_thread(
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
        run = db.scalar(select(AgentRun))
        assert run is not None
        foreign = _seed_predecessor(
            db,
            source_ids=['gmail:message-1'],
            created_at=datetime(2026, 8, 27, tzinfo=UTC),
        )
        foreign.payload['agent_run_id'] = run.id
        db.commit()

    replay = service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    assert replay.review_item_ids == first.review_item_ids
    assert adapter.run_calls == 1


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


def test_five_adapter_draft_renews_owned_lease_between_slow_provider_calls(
    db_session,
) -> None:
    settings = _settings()
    clock = _Clock()
    tracked_sessions: list[Session] = []
    factory = _session_factory(db_session, tracked=tracked_sessions)

    def advance_provider_clock(_packet: EvidencePacket) -> None:
        assert all(not session.in_transaction() for session in tracked_sessions)
        clock.value += timedelta(seconds=30)

    adapters = [
        _FakeAdapter(_manifest(name), on_run=advance_provider_clock)
        for name in DEFAULT_REVIEW_AGENT_NAMES
    ]
    catalog = ReviewAgentCatalog(adapters)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
        agent_names=DEFAULT_REVIEW_AGENT_NAMES,
    )
    service = ReviewDraftService(
        session_factory=factory,
        catalog=catalog,
        settings=settings,
        lease_ttl_seconds=45,
        now=clock,
    )

    result = service.draft(
        workflow_thread_id=thread_id,
        actor_subject_id='actor-1',
        allowed_permission_levels=('public', 'internal'),
    )

    assert len(result.review_item_ids) == 5
    assert [adapter.run_calls for adapter in adapters] == [1, 1, 1, 1, 1]
    assert clock.value == datetime(2026, 8, 27, 9, 2, 30, tzinfo=UTC)
    with factory() as db:
        thread = db.get(AgentWorkflowThread, thread_id)
        assert thread is not None
        assert thread.state_version == 6
        assert thread.status == 'checkpoint_pending'
        assert thread.lease_token is None


def test_lease_renewal_requires_exact_owned_token_and_state_cas(db_session) -> None:
    settings = _settings()
    clock = _Clock()
    factory = _session_factory(db_session)
    thread_ids: list[str] = []

    def steal_lease(_packet: EvidencePacket) -> None:
        with factory() as db:
            thread = db.get(AgentWorkflowThread, thread_ids[0])
            assert thread is not None
            thread.lease_token = 'foreign-owner-token'
            thread.state_version += 1
            db.commit()
        clock.value += timedelta(seconds=30)

    first = _FakeAdapter(_manifest('mail_document_agent'), on_run=steal_lease)
    second = _FakeAdapter(_manifest('history_agent'))
    adapters = [
        first
        if name == 'mail_document_agent'
        else second
        if name == 'history_agent'
        else _FakeAdapter(_manifest(name))
        for name in DEFAULT_REVIEW_AGENT_NAMES
    ]
    catalog = ReviewAgentCatalog(adapters)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
        agent_names=('mail_document_agent', 'history_agent'),
    )
    thread_ids.append(thread_id)

    with pytest.raises(ReviewDraftError) as exc_info:
        ReviewDraftService(
            session_factory=factory,
            catalog=catalog,
            settings=settings,
            lease_ttl_seconds=45,
            now=clock,
        ).draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'concurrent_resume'
    assert first.run_calls == 1
    assert second.run_calls == 0
    assert _row_counts(factory) == (1, 0, 0)
    with factory() as db:
        thread = db.get(AgentWorkflowThread, thread_id)
        assert thread is not None
        assert thread.status == 'drafting'
        assert thread.lease_token == 'foreign-owner-token'
        assert thread.state_version == 2


@pytest.mark.parametrize('lease_stolen_after_commit', [False, True])
def test_ambiguous_renewal_commit_cleanup_checks_old_and_prospective_claims(
    db_session,
    lease_stolen_after_commit: bool,
) -> None:
    class InjectedRenewalCommitError(RuntimeError):
        pass

    settings = _settings()
    source = _seed_source(db_session)
    db_session.commit()
    first = _FakeAdapter(_manifest('mail_document_agent'))
    second = _FakeAdapter(_manifest('history_agent'))
    adapters = [
        first
        if name == 'mail_document_agent'
        else second
        if name == 'history_agent'
        else _FakeAdapter(_manifest(name))
        for name in DEFAULT_REVIEW_AGENT_NAMES
    ]
    catalog = ReviewAgentCatalog(adapters)
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
        agent_names=('mail_document_agent', 'history_agent'),
    )

    normal_factory = _session_factory(db_session)
    injected = False

    class AmbiguousRenewalSession(Session):
        def commit(self) -> None:
            nonlocal injected
            renewal = next(
                (
                    value
                    for value in self.dirty
                    if isinstance(value, AgentWorkflowThread)
                    and value.lease_token is not None
                    and value.state_version == 2
                ),
                None,
            )
            super().commit()
            if renewal is None or injected:
                return
            injected = True
            if lease_stolen_after_commit:
                with normal_factory() as thief_db:
                    stolen = thief_db.get(AgentWorkflowThread, thread_id)
                    assert stolen is not None
                    stolen.lease_token = 'foreign-owner-token'
                    stolen.lease_expires_at = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
                    stolen.state_version += 1
                    thief_db.commit()
            raise InjectedRenewalCommitError('renewal commit outcome is ambiguous')

    ambiguous_local = sessionmaker(
        bind=db_session.get_bind(),
        class_=AmbiguousRenewalSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    with pytest.raises(
        InjectedRenewalCommitError,
        match='renewal commit outcome is ambiguous',
    ):
        ReviewDraftService(
            session_factory=ambiguous_local,
            catalog=catalog,
            settings=settings,
        ).draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert injected is True
    assert first.run_calls == 1
    assert second.run_calls == 0
    assert _row_counts(normal_factory) == (1, 0, 0)
    with normal_factory() as db:
        thread = db.get(AgentWorkflowThread, thread_id)
        assert thread is not None
        if lease_stolen_after_commit:
            assert thread.status == 'drafting'
            assert thread.lease_token == 'foreign-owner-token'
            assert thread.lease_expires_at is not None
            foreign_expiry = thread.lease_expires_at
            if foreign_expiry.tzinfo is None:
                foreign_expiry = foreign_expiry.replace(tzinfo=UTC)
            assert foreign_expiry == datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
            assert thread.state_version == 3
        else:
            assert thread.status == 'created'
            assert thread.lease_token is None
            assert thread.lease_expires_at is None
            assert thread.state_version == 3


def test_persistence_failure_rolls_back_and_releases_only_owned_lease(
    db_session,
    monkeypatch,
) -> None:
    class InjectedPersistenceError(RuntimeError):
        pass

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

    def fail_persistence(*_args, **_kwargs):
        raise InjectedPersistenceError('injected serialization failure')

    monkeypatch.setattr(
        'backend.app.agent_runtime.review_v2_drafting._insert_or_get_review_item',
        fail_persistence,
    )

    with pytest.raises(InjectedPersistenceError, match='serialization failure'):
        ReviewDraftService(
            session_factory=factory,
            catalog=catalog,
            settings=settings,
        ).draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert adapter.run_calls == 1
    assert _row_counts(factory) == (1, 0, 0)
    with factory() as db:
        thread = db.get(AgentWorkflowThread, thread_id)
        assert thread is not None
        assert thread.status == 'created'
        assert thread.lease_token is None


@pytest.mark.parametrize('prior_status', ['checkpoint_pending', 'checkpoint_failed'])
def test_provider_failure_restores_captured_prior_allowed_status(
    db_session,
    prior_status: str,
) -> None:
    settings = _settings()

    def fail_provider(_packet: EvidencePacket) -> None:
        raise TimeoutError('provider timed out')

    adapter = _FakeAdapter(_manifest('history_agent'), on_run=fail_provider)
    catalog, _ = _catalog(selected_adapter=adapter)
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
    )
    thread = db_session.get(AgentWorkflowThread, thread_id)
    assert thread is not None
    thread.status = prior_status
    db_session.commit()
    factory = _session_factory(db_session)

    with pytest.raises(ReviewDraftError) as exc_info:
        ReviewDraftService(
            session_factory=factory,
            catalog=catalog,
            settings=settings,
        ).draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'model_unavailable'
    with factory() as db:
        restored = db.get(AgentWorkflowThread, thread_id)
        assert restored is not None
        assert restored.status == prior_status
        assert restored.lease_token is None


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


@pytest.mark.parametrize(
    'mutation',
    [
        'workflow_name',
        'graph_version',
        'request_schema',
        'request_kind',
        'request_hash',
        'fingerprint_key_version',
        'agent_names',
        'selection_policy',
        'evidence_ordinal',
    ],
)
def test_draft_rejects_mutated_frozen_workflow_identity_before_model(
    db_session,
    mutation: str,
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
    thread = db_session.get(AgentWorkflowThread, thread_id)
    request = db_session.get(AgentWorkflowRequest, thread_id)
    evidence = db_session.scalar(
        select(AgentWorkflowEvidenceRef).where(
            AgentWorkflowEvidenceRef.workflow_thread_id == thread_id
        )
    )
    assert thread is not None
    assert request is not None
    assert evidence is not None
    if mutation == 'workflow_name':
        thread.workflow_name = 'mutated-workflow'
    elif mutation == 'graph_version':
        thread.graph_version = 'mutated-graph'
    elif mutation == 'request_schema':
        request.input_schema_version = 'mutated-schema'
    elif mutation == 'request_kind':
        request.request_kind = 'mutated-kind'
    elif mutation == 'request_hash':
        request.input_hash = 'f' * 64
    elif mutation == 'fingerprint_key_version':
        request.fingerprint_key_version = 'mutated-key'
    elif mutation == 'agent_names':
        request.agent_names = ['history_agent', 'history_agent']
    elif mutation == 'selection_policy':
        request.selection_policy_version = 'mutated-policy'
    elif mutation == 'evidence_ordinal':
        evidence.ordinal = 4
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mutation)
    db_session.commit()
    factory = _session_factory(db_session)

    with pytest.raises(ReviewDraftError) as exc_info:
        ReviewDraftService(
            session_factory=factory,
            catalog=catalog,
            settings=settings,
        ).draft(
            workflow_thread_id=thread_id,
            actor_subject_id='actor-1',
            allowed_permission_levels=('public', 'internal'),
        )

    assert exc_info.value.code == 'invalid_state_transition'
    assert adapter.run_calls == 0
    assert _row_counts(factory) == (1, 0, 0)


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


@pytest.mark.parametrize(
    ('agent_name', 'wrong_item_type'),
    [
        ('timeline_agent', 'history_event'),
        ('history_agent', 'timeline_event'),
        ('decision_record_agent', 'todo'),
        ('todo_agent', 'decision_record'),
    ],
)
def test_memory_adapter_rejects_wrong_item_type_without_writes(
    db_session,
    agent_name: str,
    wrong_item_type: str,
) -> None:
    settings = _settings()

    def wrong_output(packet: EvidencePacket) -> list[ReviewCandidate]:
        return [_candidate_from_packet(packet, item_type=wrong_item_type)]

    adapter = _FakeAdapter(
        _manifest(agent_name),
        candidate_factory=wrong_output,
    )
    catalog, _ = _catalog(
        selected_name=agent_name,
        selected_adapter=adapter,
    )
    source = _seed_source(db_session)
    db_session.commit()
    _, thread_id = _create_workflow(
        db_session,
        source=source,
        catalog=catalog,
        settings=settings,
        agent_name=agent_name,
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
