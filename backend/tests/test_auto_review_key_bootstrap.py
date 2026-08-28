import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.admin.auto_review_keys import (
    AutoReviewKeyBootstrapError,
    AutoReviewKeyBootstrapService,
)
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models import (
    AgentRun,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AssistantConversation,
    AssistantMessage,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    ReviewItem,
    Source,
    TrustedKnowledgeFingerprintProjectionState,
)


def _service(engine, *, settings: Settings | None = None):
    return AutoReviewKeyBootstrapService(
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
        settings=settings or Settings(_env_file=None, paraworks_demo_mode=True),
    )


def test_bootstrap_is_a_noop_before_c5_tables_exist() -> None:
    engine = create_engine('sqlite:///:memory:')

    result = _service(engine).ensure_initialized()

    assert result.schema_available is False
    assert result.initialized is False


def test_disabled_sqlite_bootstrap_initializes_not_ready_state_idempotently() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    service = _service(engine)

    first = service.ensure_initialized()
    second = service.ensure_initialized()

    assert first.schema_available is True
    assert first.initialized is True
    assert first.ready is False
    assert second.initialized is False
    assert second.ready is False
    with sessionmaker(bind=engine)() as db:
        runtime = db.query(AutoReviewRuntimeKeyState).one()
        projection = db.query(TrustedKnowledgeFingerprintProjectionState).one()
        assert runtime.component == 'auto_review_trust_promotion'
        assert runtime.generation == 1
        assert runtime.ready is False
        assert projection.ready is False
        assert projection.rebuild_required is True


def test_bootstrap_refuses_missing_runtime_state_with_retained_keyed_artifact() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        db.add(
            AgentWorkflowRequest(
                workflow_thread_id='retained-c5-request',
                input_schema_version='review-workflow:v2.1',
                request_kind='review_source_versions',
                agent_names=['timeline_agent'],
                selection_policy_version='company-memory-review-selection:v1',
                input_hash='a' * 64,
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='b' * 64,
                auto_review_mode='shadow',
            )
        )
        db.commit()

    try:
        _service(engine).ensure_initialized()
    except AutoReviewKeyBootstrapError as exc:
        assert exc.code == 'retained_keyed_state_without_runtime_identity'
    else:
        raise AssertionError('retained keyed state must refuse bootstrap repair')


def test_bootstrap_refuses_runtime_key_identity_mismatch() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=True,
        agent_runtime_fingerprint_secret='x' * 32,
        agent_runtime_fingerprint_key_version='v1',
    )
    service = _service(engine, settings=settings)
    service.ensure_initialized()
    with sessionmaker(bind=engine)() as db:
        row = db.query(AutoReviewRuntimeKeyState).one()
        row.fingerprint_key_version = 'different'
        db.commit()

    try:
        service.ensure_initialized()
    except AutoReviewKeyBootstrapError as exc:
        assert exc.code == 'runtime_key_identity_mismatch'
    else:
        raise AssertionError('runtime key identity mismatch must fail closed')


def test_bootstrap_refuses_partial_c5_schema_instead_of_initializing() -> None:
    engine = create_engine('sqlite:///:memory:')
    AutoReviewRuntimeKeyState.__table__.create(engine)
    TrustedKnowledgeFingerprintProjectionState.__table__.create(engine)

    try:
        _service(engine).ensure_initialized()
    except AutoReviewKeyBootstrapError as exc:
        assert exc.code == 'incomplete_c5_schema'
    else:
        raise AssertionError('partial C.5 schema must fail closed')


def test_retained_state_probe_never_ignores_request_only_snapshot() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        db.add(
            AgentWorkflowRequest(
                workflow_thread_id='validator-model-only',
                input_schema_version='review-workflow:v2.1',
                request_kind='review_source_versions',
                agent_names=['timeline_agent'],
                selection_policy_version='selection:v1',
                input_hash='a' * 64,
                fingerprint_key_version='v1',
                auto_review_validator_model='gpt-5.6-terra',
            )
        )
        db.commit()

    try:
        _service(engine).ensure_initialized()
    except AutoReviewKeyBootstrapError as exc:
        assert exc.code == 'retained_keyed_state_without_runtime_identity'
    else:
        raise AssertionError('every request snapshot must count as retained state')


@pytest.mark.parametrize(
    'category',
    [
        'new_table',
        'graph',
        'request',
        'review_item',
        'agent_run',
        'assistant_message',
        'source',
        'document_pointer',
        'parser_identity',
        'chunk_lineage',
    ],
)
def test_bootstrap_refuses_every_retained_c5_state_category(category: str) -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        if category == 'new_table':
            db.add(
                AutoReviewRolloutState(
                    security_scope_id='retained-scope', policy_version='policy:v1'
                )
            )
        elif category == 'graph':
            db.add(
                AgentWorkflowThread(
                    thread_id='retained-graph',
                    workflow_name='company-memory-review',
                    graph_version='company-memory-review-v2.1-auto-review',
                    checkpoint_thread_id='retained-graph-checkpoint',
                    checkpoint_store='sqlite',
                    owner_subject_id='owner',
                    security_scope_id='scope',
                    input_hash='a' * 64,
                    evidence_version_hash='b' * 64,
                )
            )
        elif category == 'request':
            db.add(
                AgentWorkflowRequest(
                    workflow_thread_id='retained-request',
                    input_schema_version='review-workflow:v2.1',
                    request_kind='review_source_versions',
                    agent_names=['timeline_agent'],
                    selection_policy_version='selection:v1',
                    input_hash='a' * 64,
                    fingerprint_key_version='v1',
                    auto_review_extraction_route_version='route:v1',
                )
            )
        elif category == 'review_item':
            db.add(
                ReviewItem(
                    item_type='timeline_event',
                    payload={},
                    source_links=[],
                    source_snippets=[],
                    confidence_score=0.99,
                    permission_level='internal',
                    resolution_source='human',
                )
            )
        elif category == 'agent_run':
            db.add(
                AgentRun(
                    agent_name='timeline_agent',
                    prompt_version='timeline:v1',
                    status='complete',
                    source_window='window',
                    cache_key='retained-run',
                    model_name='model',
                    permission_level='internal',
                    generation_provider='openai',
                )
            )
        elif category == 'assistant_message':
            conversation = AssistantConversation(user_id='retained-user')
            db.add(
                AssistantMessage(
                    conversation=conversation,
                    role='user',
                    content='retained',
                    citations=[],
                    source_ids=[],
                    source_links=[],
                    source_snippets=[],
                    permission_level='internal',
                    evidence_contract_version='none-v1',
                    serving_dependency_count=0,
                    metadata_={},
                )
            )
        else:
            source = Source(
                source_type='drive',
                source_id=f'retained-{category}',
                source_url='https://example.test/retained',
                title='Retained',
                permission_level='internal',
                raw_metadata={},
            )
            db.add(source)
            db.flush()
            if category == 'source':
                source.server_content_signature_schema = 'server-source-content:v1'
                source.server_content_signature = 'a' * 64
            else:
                document = Document(source=source, title='Retained document')
                version = DocumentVersion(document=document, version='v1', body='body')
                db.add_all([document, version])
                db.flush()
                if category == 'document_pointer':
                    document.current_document_version_id = version.id
                else:
                    parser = DocumentParserRun(
                        document_id=document.id,
                        document_version_id=version.id,
                        source_id=source.id,
                        parser_name='parser',
                        parser_status='complete',
                        server_content_signature_schema='server-source-content:v1',
                        server_content_signature='a' * 64,
                        parser_policy_version='policy:v1',
                        parser_version='parser:v1',
                        chunk_policy_version='chunk:v1',
                    )
                    db.add(parser)
                    db.flush()
                    if category == 'chunk_lineage':
                        db.add(
                            DocumentChunk(
                                version_id=version.id,
                                source_id=source.id,
                                parser_run_id=parser.id,
                                chunk_index=0,
                                text='body',
                                source_snippet='body',
                                permission_level='internal',
                                metadata_={},
                            )
                        )
        db.commit()

    with pytest.raises(AutoReviewKeyBootstrapError) as exc_info:
        _service(engine).ensure_initialized()

    assert exc_info.value.code == 'retained_keyed_state_without_runtime_identity'
