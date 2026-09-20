from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.rag_finalization import AssistantProjectionTarget
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    prepare_assistant_request_text,
    prepare_direct_request_text,
)
from backend.app.core.demo_auth import DemoUser
from backend.app.db.base import Base
from backend.app.models import AgentRun, AssistantConversation, AssistantMessage
from backend.tests.test_rag_v2_keyword_retriever import (
    _seed_sqlite_raw_projection,
    _settings,
)
from backend.tests.test_rag_v2_sqlite_smoke import _engine


@pytest.mark.parametrize(
    'surface,query,outcome,result_count',
    (
        ('search', 'observation', 'search_projected', 1),
        ('ask', 'no-matching-term', 'no_match', 0),
        ('ask', 'observation', 'supported', 0),
        ('assistant', 'observation', 'supported', 0),
        ('assistant', 'no-matching-term', 'no_match', 0),
    ),
)
def test_default_runtime_sqlite_owns_single_atomic_graph_scope(
    tmp_path, surface, query, outcome, result_count
):
    from backend.app.agent_runtime.rag_v2_composition import build_rag_v2_runtime

    path = (tmp_path / 'graph-default.db').absolute()
    engine = _engine(path)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
        target = None
        if surface == 'assistant':
            conversation = AssistantConversation(user_id='user-1', title='RAG')
            db.add(conversation)
            db.flush()
            message = AssistantMessage(
                conversation_id=conversation.id,
                role='user',
                content=query,
                citations=[],
                source_ids=[],
                source_links=[],
                source_snippets=[],
                metadata_={},
            )
            db.add(message)
            db.commit()
            target = AssistantProjectionTarget(conversation.id, message.id, 'user-1')
    settings = _settings().model_copy(
        update={
            'database_url': f'sqlite:///{path.as_posix()}',
            'openai_api_key': None,
            'rag_retrieval_backend': 'keyword',
            'rag_use_pgvector_search': False,
            'langgraph_rag_v2_mode': 'enforce',
            'langgraph_rag_v2_stage': 'assistant',
        }
    )
    statements = []
    event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, parameters, context, executemany: (
            statements.append(statement)
        ),
    )
    bundle = build_rag_v2_runtime(
        settings=settings, session_factory=sessionmaker(bind=engine)
    )
    actor = DemoUser(
        id='user-1',
        email='test@example.test',
        name='Tester',
        role='member',
        title='Test',
        department='Test',
        permission_levels={'public', 'internal'},
    )
    text = (
        prepare_assistant_request_text(
            query, (), key=settings.agent_runtime_fingerprint_secret.encode()
        )
        if surface == 'assistant'
        else prepare_direct_request_text(
            query, key=settings.agent_runtime_fingerprint_secret.encode()
        )
    )
    result = bundle.facade.invoke_graph(
        actor=actor, surface=surface, prepared_text=text, assistant_target=target
    )
    assert result['outcome'] == outcome
    assert len(result['evidence_projection'].search_results) == result_count
    assert result['charged_cost_usd'] == 0
    assert result['sanitized_trace'].node_counts['keyword_retrieval'] == 1
    finalizer = (
        'finalize_run_and_search_projection'
        if surface == 'search'
        else 'finalize_run_and_answer_projection_or_assistant_message'
        if outcome == 'supported'
        else 'provider_free_safe_outcome_finalizer'
    )
    assert result['sanitized_trace'].node_counts[finalizer] == 1
    if outcome == 'supported':
        assert result['answer_blocks'].assembled_answer
        assert len(result['evidence_projection'].citations) == 1
        assert len(result['model_influence']) == 1
    assert result['sanitized_trace'].provider_attempt_counts == {
        'query_embedding': 0,
        'answer_generation': 0,
    }
    assert (
        sum(statement.upper().startswith('BEGIN IMMEDIATE') for statement in statements)
        == 1
    )
    with Session(engine) as db:
        parent = db.scalar(select(AgentRun))
        assert parent.status == 'complete' and parent.run_record_phase == 'final'
        if surface == 'assistant':
            saved = db.scalar(
                select(AssistantMessage).where(AssistantMessage.role == 'assistant')
            )
            assert saved.linked_agent_run_id == parent.id
            assert saved.content_write_mode == 'rag_v2_exact'
            if outcome == 'supported':
                assert saved.citations[0]['source_id'] == 'gmail:keyword-task-6'
                assert saved.citations[0]['permission_level'] == 'internal'
                assert (
                    saved.citations[0]['source_snippet']
                    == 'Exact raw observation with %_Case and Unicode 한글'
                )
                assert saved.content == result['answer_blocks'].assembled_answer
            else:
                assert saved.citations == []
    engine.dispose()


def test_default_sqlite_filtered_slot_persists_only_selected_safe_source(tmp_path):
    from backend.app.agent_runtime.rag_v2_composition import build_rag_v2_runtime
    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections
    from backend.tests.test_rag_source_observations import _seed_source_chunk

    path = (tmp_path / 'mixed.db').absolute()
    engine = _engine(path)
    Base.metadata.create_all(engine)
    settings = _settings().model_copy(
        update={
            'database_url': f'sqlite:///{path.as_posix()}',
            'openai_api_key': None,
            'rag_retrieval_backend': 'keyword',
            'rag_use_pgvector_search': False,
            'langgraph_rag_v2_mode': 'enforce',
            'langgraph_rag_v2_stage': 'assistant',
        }
    )
    query = 'mixedneedle'
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
        _seed_source_chunk(
            db, text='mixedneedle Authorization: Bearer test-secret-token-1234567890'
        )
        _seed_source_chunk(
            db, source_type='drive', text='mixedneedle Safe observation 한글'
        )
        refresh_rag_lexical_projections(db, settings=settings, corpus_generation=1)
        conversation = AssistantConversation(user_id='user-1', title='RAG')
        db.add(conversation)
        db.flush()
        message = AssistantMessage(
            conversation_id=conversation.id,
            role='user',
            content=query,
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            metadata_={},
        )
        db.add(message)
        db.commit()
        target = AssistantProjectionTarget(conversation.id, message.id, 'user-1')
    bundle = build_rag_v2_runtime(
        settings=settings, session_factory=sessionmaker(bind=engine)
    )
    actor = DemoUser(
        id='user-1',
        email='test@example.test',
        name='Tester',
        role='member',
        title='Test',
        department='Test',
        permission_levels={'public', 'internal'},
    )
    result = bundle.facade.invoke_graph(
        actor=actor,
        surface='assistant',
        assistant_target=target,
        prepared_text=prepare_assistant_request_text(
            query, (), key=settings.agent_runtime_fingerprint_secret.encode()
        ),
    )
    assert result['outcome'] == 'supported'
    assert result['evidence_projection'].source_ids == ('drive:source-1',)
    assert (
        result['model_influence'][0].fresh_lookup_identity.public_source_id
        == 'drive:source-1'
    )
    with Session(engine) as db:
        saved = db.scalar(
            select(AssistantMessage).where(AssistantMessage.role == 'assistant')
        )
        assert saved.source_ids == ['drive:source-1']
        assert saved.citations[0]['source_id'] == 'drive:source-1'
        assert (
            saved.citations[0]['source_snippet'] == 'mixedneedle Safe observation 한글'
        )
        assert saved.citations[0]['permission_level'] == 'internal'
    engine.dispose()


def test_default_pgvector_missing_key_refuses_before_cost_authority_assembly(
    monkeypatch,
):
    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime.rag_application import RagApplicationError
    from backend.app.agent_runtime.rag_v2_composition import _postgres_request_services

    def forbidden(**kwargs):
        pytest.fail('missing retrieval configuration must precede paid DB assembly')

    monkeypatch.setattr(transport, '_assemble_rag_request_cost_authority', forbidden)
    settings = _settings().model_copy(
        update={
            'rag_retrieval_backend': 'pgvector',
            'rag_use_pgvector_search': True,
            'openai_api_key': None,
        }
    )
    with (
        pytest.raises(RagApplicationError) as error,
        _postgres_request_services(
            db=object(), settings=settings, session_factory=forbidden
        ),
    ):
        pytest.fail('missing key must not expose request services')
    assert error.value.code == 'retriever_not_configured'


@pytest.mark.parametrize('failure', ('node', 'commit', 'retriever'))
def test_default_sqlite_graph_failure_rolls_back_without_dto_or_pending_parent(
    tmp_path, monkeypatch, failure
):
    from backend.app.agent_runtime import rag_graph
    from backend.app.agent_runtime.rag_v2_composition import build_rag_v2_runtime

    path = (tmp_path / 'rollback.db').absolute()
    engine = _engine(path)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        _seed_sqlite_raw_projection(db)
    settings = _settings().model_copy(
        update={
            'database_url': f'sqlite:///{path.as_posix()}',
            'openai_api_key': None,
            'rag_retrieval_backend': 'keyword',
            'rag_use_pgvector_search': False,
            'langgraph_rag_v2_mode': 'enforce',
            'langgraph_rag_v2_stage': 'search',
        }
    )
    actor = DemoUser(
        id='user-1',
        email='test@example.test',
        name='Tester',
        role='member',
        title='Test',
        department='Test',
        permission_levels={'public', 'internal'},
    )
    bundle = build_rag_v2_runtime(
        settings=settings, session_factory=sessionmaker(bind=engine)
    )
    text = prepare_direct_request_text(
        'observation', key=settings.agent_runtime_fingerprint_secret.encode()
    )

    def fail(*args, **kwargs):
        raise RuntimeError('deliberate request failure')

    if failure == 'node':
        monkeypatch.setattr(rag_graph, 'rank_evidence_slots', fail)
    elif failure == 'retriever':
        monkeypatch.setattr(
            'backend.app.rag.keyword_retriever.KeywordEvidenceRetriever.invoke', fail
        )
    else:
        event.listen(engine, 'commit', fail)
    with pytest.raises(
        RuntimeError,
        match='retriever_unavailable'
        if failure == 'retriever'
        else 'deliberate request failure',
    ):
        bundle.facade.invoke_graph(actor=actor, surface='search', prepared_text=text)
    if failure == 'commit':
        event.remove(engine, 'commit', fail)
    with Session(engine) as db:
        assert db.scalar(select(AgentRun)) is None
    engine.dispose()


def test_provider_free_runtime_can_hold_concrete_safety_without_opening_paid_latch(
    tmp_path,
):
    from backend.app.agent_runtime.durable_file_authority import (
        DurableFileAuthorityError,
    )
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService

    path = (tmp_path / 'uninitialized-paid-latch.json').absolute()
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'task14-not-a-provider-key',
        designated_environment_id='test',
    )
    assert service.advisory_transport_authority is None
    assert not path.exists()
    with pytest.raises(DurableFileAuthorityError), service._authority.locked():
        pass
    assert not path.exists()


def test_postgres_request_service_composition_uses_owned_cost_assembly_without_clients(
    tmp_path, monkeypatch
):
    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.tests.test_rag_v2_graph import _answer_context

    context, client = _answer_context(tmp_path)
    services = context.services
    authority = services.provider_transport
    bundle = transport.RagRequestCostAuthority(
        services.db,
        services.cost_ledger,
        services.cost_policy,
        services.cost_ledger.provider_safety_authority,
        services.cost_ledger.provider_connection_factory,
        authority._barrier,
        authority._load_current_readiness,
        services.cost_ledger.runtime_health_authority,
        object(),
    )

    def assemble(*, settings, session):
        assert session is services.db
        return bundle

    def no_client(*args, **kwargs):
        raise AssertionError(
            'keyless cost/retrieval composition must not initialize a provider'
        )

    monkeypatch.setattr(transport, '_assemble_rag_request_cost_authority', assemble)
    monkeypatch.setattr(transport, '_DirectOpenAIProviderClient', no_client)
    with composition._postgres_request_services(
        db=services.db, settings=context.settings, session_factory=lambda: None
    ) as composed:
        assert composed.db is services.db
        assert composed.cost_ledger is services.cost_ledger
        assert composed.answer_model is None
        assert composed.provider_transport is None
        assert composed.load_generations() == (1, None)
    assert client.seen == []


@pytest.mark.parametrize('cleanup_failure', (False, True))
def test_postgres_factory_passes_exact_pending_to_fresh_session_and_closes_it(
    tmp_path, monkeypatch, cleanup_failure
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.tests.test_rag_v2_graph import _answer_context

    context, client = _answer_context(tmp_path)
    source = context.services
    authority = source.provider_transport
    assembly = transport.RagRequestCostAuthority(
        source.db,
        source.cost_ledger,
        source.cost_policy,
        source.cost_ledger.provider_safety_authority,
        source.cost_ledger.provider_connection_factory,
        authority._barrier,
        authority._load_current_readiness,
        source.cost_ledger.runtime_health_authority,
        object(),
    )
    monkeypatch.setattr(
        transport, '_assemble_rag_request_cost_authority', lambda **kwargs: assembly
    )
    closed = []

    def close():
        closed.append(True)
        if cleanup_failure:
            raise RuntimeError('secondary cleanup failure')

    fresh = SimpleNamespace(close=close)
    pending, prepared = object(), object()

    def finalizer(**kwargs):
        assert kwargs['db'] is fresh and kwargs['db'] is not source.db
        assert kwargs['pending'] is pending and kwargs['prepared'] is prepared
        assert kwargs['assembly'] is assembly
        return 'exact-fresh-finalizer'

    monkeypatch.setattr(composition, '_postgres_finalizer', finalizer)
    guard = (
        pytest.raises(RuntimeError, match='primary request failure')
        if cleanup_failure
        else nullcontext()
    )
    with (
        guard,
        composition._postgres_request_services(
            db=source.db, settings=context.settings, session_factory=lambda: fresh
        ) as services,
    ):
        assert services.finalizer_factory(pending, prepared) == 'exact-fresh-finalizer'
        assert closed == []
        if cleanup_failure:
            raise RuntimeError('primary request failure')
    assert closed == [True] and client.seen == []


def test_lazy_postgres_provider_factory_consumed_once_by_actual_answer_graph(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.tests import test_rag_v2_costs as costs
    from backend.tests.test_rag_v2_graph import _answer_context
    from backend.tests.test_rag_v2_provider_transport import _TEST_SETTINGS

    original_snapshot = costs._snapshot

    def authenticated_snapshot(component, policy=None):
        return replace(
            original_snapshot(component, policy),
            fingerprint_key_version=_TEST_SETTINGS.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                _TEST_SETTINGS.agent_runtime_fingerprint_secret
            ),
        )

    monkeypatch.setattr(costs, '_snapshot', authenticated_snapshot)
    context, client = _answer_context(tmp_path)
    source = context.services
    authority = source.provider_transport
    settings = context.settings.model_copy(
        update={'openai_api_key': 'fake-not-a-real-key'}
    )
    assembly = transport.RagRequestCostAuthority(
        source.db,
        source.cost_ledger,
        source.cost_policy,
        source.cost_ledger.provider_safety_authority,
        source.cost_ledger.provider_connection_factory,
        authority._barrier,
        authority._load_current_readiness,
        source.cost_ledger.runtime_health_authority,
        object(),
    )
    monkeypatch.setattr(
        transport, '_assemble_rag_request_cost_authority', lambda **kwargs: assembly
    )
    events = []
    client._embedding_client = SimpleNamespace(close=lambda: events.append('close'))
    client.close = client._embedding_client.close

    def client_factory(actual_settings):
        assert actual_settings is settings
        events.append('construct')
        return client

    monkeypatch.setattr(transport, '_DirectOpenAIProviderClient', client_factory)
    with composition._postgres_request_services(
        db=source.db, settings=settings, session_factory=lambda: None
    ) as composed:
        assert events == []
        # Only PostgreSQL synchronization/sequence and corpus lookup are portable
        # test ports. Concrete cost authority, lazy transport, model and graph run.
        services = replace(
            composed,
            allocate_run_id=source.allocate_run_id,
            retrievers=source.retrievers,
            finalizer_factory=source.finalizer_factory,
        )
        text = prepare_direct_request_text(
            'observation', key=settings.agent_runtime_fingerprint_secret.encode()
        )
        result = build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text},
            context=replace(context, settings=settings, services=services),
        )
        assert result['outcome'] == 'supported'
        assert result['answer_blocks'].assembled_answer == '관찰한 근거입니다.'
        assert len(client.seen) == 1
        assert events == ['construct']
        assert (
            composed.provider_transport_factory()
            is composed.provider_transport_factory()
        )
    assert events == ['construct', 'close']


@pytest.mark.parametrize('paid', (False, True))
def test_postgres_finalizer_owns_fresh_phase2_safety_and_exact_run_requirements(
    tmp_path, monkeypatch, paid
):
    from types import SimpleNamespace

    from backend.app.agent_runtime import rag_finalization as finalization
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.app.agent_runtime.rag_advisory_locks import (
        RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
        RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
        register_advisory_identity_db,
    )
    from backend.tests.test_rag_v2_graph import _answer_context

    context, _ = _answer_context(tmp_path)
    source = context.services
    engine = source.db.get_bind()
    with engine.begin() as connection:
        for identity in (
            RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
        ):
            register_advisory_identity_db(
                connection, identity, identity_namespace='static'
            )
    bound_database = SimpleNamespace(connect=engine.connect, close=lambda: None)
    monkeypatch.setattr(
        'backend.app.agent_runtime.rag_postgres_binding._bind_rag_postgres_database',
        lambda *args, **kwargs: bound_database,
    )
    monkeypatch.setattr(
        'backend.app.agent_runtime.provider_send_fence._assemble_rag_evidence_barrier',
        lambda **kwargs: object(),
    )
    advisory_transport = object()
    captured_transport: list[object] = []
    if paid:
        class ComposedProviderSafety:
            def __init__(
                self,
                *,
                latch_path,
                identity_secret,
                designated_environment_id,
                advisory_capability,
                advisory_transport,
            ):
                captured_transport.append(advisory_transport)
                self.advisory_transport_authority = advisory_transport
                self._latch_path = Path(latch_path)
                self._secret = identity_secret

        monkeypatch.setattr(
            'backend.app.agent_runtime.rag_postgres_binding.'
            '_bind_rag_postgres_advisory_transport',
            lambda *args, **kwargs: advisory_transport,
        )
        monkeypatch.setattr(
            'backend.app.agent_runtime.rag_provider_safety.RagProviderSafetyService',
            ComposedProviderSafety,
        )
    phase2 = object()
    binding = SimpleNamespace(policy_snapshot=source.policy_snapshots[1])
    unrelated = SimpleNamespace(policy_snapshot=source.policy_snapshots[0])
    bindings = {(99, 'query_embedding'): unrelated}
    if paid:
        bindings[(161, 'answer_generation')] = binding
    assembly = SimpleNamespace(
        session=source.db,
        bootstrap_capability=object(),
        provider_safety=source.cost_ledger.provider_safety_authority,
        store=SimpleNamespace(_terminal_bindings=bindings),
    )
    settings = context.settings.model_copy(
        update={
            'paraworks_provider_safety_latch_path': str(
                (tmp_path / 'phase2-latch.json').absolute()
            )
        }
    )

    def paid_phase2(**kwargs):
        safety = kwargs['provider_safety']
        assert safety is not assembly.provider_safety
        assert safety.advisory_transport_authority is advisory_transport
        assert safety._latch_path == (tmp_path / 'phase2-latch.json').absolute()
        assert safety._secret == settings.agent_runtime_fingerprint_secret.encode()
        assert kwargs['postgres_database'] is bound_database
        assert kwargs['safety_requirements'] == ((binding.policy_snapshot, binding),)
        return phase2

    def free_phase2(**kwargs):
        assert not paid and kwargs['postgres_database'] is bound_database
        return phase2

    monkeypatch.setattr(
        finalization, '_assemble_paid_rag_phase2_authority', paid_phase2
    )
    monkeypatch.setattr(
        finalization, '_assemble_provider_free_rag_phase2_authority', free_phase2
    )
    with Session(engine) as fresh:

        def boundary(**kwargs):
            assert kwargs['db'] is fresh and kwargs['phase2_authority'] is phase2
            return object()

        monkeypatch.setattr(finalization, 'SqlAlchemyRagFinalizationBoundary', boundary)
        result = composition._postgres_finalizer(
            db=fresh,
            settings=settings,
            assembly=assembly,
            pending=SimpleNamespace(parent_agent_run_id=161),
            prepared=SimpleNamespace(
                retrieval_result=SimpleNamespace(configured_backend='keyword')
            ),
        )
        assert type(result) is finalization.RagFinalizationService
    if paid:
        assert captured_transport == [advisory_transport]
