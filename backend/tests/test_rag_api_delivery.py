from contextlib import contextmanager
from dataclasses import replace

import pytest

from backend.app.models import AgentRun


@contextmanager
def _graph_http(context):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.app.api.v1.ask import router as ask_router
    from backend.app.api.v1.search import router as search_router
    from backend.app.core.demo_auth import get_demo_user

    @contextmanager
    def factory(**kwargs):
        assert kwargs['actor'].id == context.actor.id
        assert kwargs['surface'] == context.surface
        yield context.services

    facade = build_rag_v2_runtime(
        settings=context.settings,
        session_factory=lambda: context.services.db,
        dependencies=RagRuntimeDependencies(request_factory=factory),
    ).facade
    app = FastAPI()
    app.include_router(ask_router)
    app.include_router(search_router)
    app.dependency_overrides[get_demo_user] = lambda: context.actor
    app.state.rag_application_facade = facade
    with TestClient(app) as http:
        yield http


@pytest.mark.parametrize(
    'kind',
    [
        'supported',
        'insufficient',
        'safety_filter_empty',
        'before_generation',
        'after_generation',
    ],
)
def test_real_answer_graph_http_result_table_preserves_paid_usage(
    tmp_path, monkeypatch, kind
):
    from sqlalchemy import select

    from backend.app.models import Source
    from backend.tests.test_rag_v2_graph import _answer_context

    context, provider = _answer_context(
        tmp_path,
        source_text='Authorization: Bearer test-secret-token-1234567890'
        if kind == 'safety_filter_empty'
        else 'Exact current source observation',
    )
    if kind == 'insufficient':
        provider.response['parsed'] = {
            'answer_blocks': [],
            'insufficient_evidence_reason': 'discarded private reason',
        }

    def revoke():
        context.services.db.scalar(select(Source)).permission_level = 'restricted'
        context.services.db.commit()

    if kind == 'before_generation':
        original = context.services.answer_model.prepare_input

        def prepare(**kwargs):
            result = original(**kwargs)
            revoke()
            context.services.db.begin()
            return result

        monkeypatch.setattr(context.services.answer_model, 'prepare_input', prepare)
    elif kind == 'after_generation':
        original = provider.send

        def send(*args, **kwargs):
            result = original(*args, **kwargs)
            revoke()
            return result

        monkeypatch.setattr(provider, 'send', send)
    caller = '  observation\n한글  '
    with _graph_http(context) as http:
        response = http.post('/ask', json={'question': caller})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['question'] == caller
    generated = kind in {'supported', 'insufficient', 'after_generation'}
    assert len(provider.seen) == int(generated)
    assert body['token_usage'] == (
        {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}
        if generated
        else {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    )
    assert body['estimated_cost_usd'] == (0.000030 if generated else 0.0)
    assert body['model_name'] == 'gpt-5.4-mini-2026-03-17'
    assert body['agent_run_id'] == 161
    assert len(body['cache_key']) == 64
    if kind == 'supported':
        assert body['answer'] == '관찰한 근거입니다.'
        assert body['source_ids'] and body['permission_level'] == 'internal'
    else:
        for name in ('source_ids', 'source_links', 'source_snippets', 'citations'):
            assert body[name] == []
        assert body['permission_level'] is None
        assert body['hidden_match_count'] == 0
        changed = kind in {'before_generation', 'after_generation'}
        assert body['permission_notice'] == (
            'evidence_unavailable' if changed else None
        )
        assert body['answer'] == (
            '이 답변의 근거를 더 이상 확인할 수 없습니다. 다시 생성해 주세요.'
            if changed
            else '권한 내에서 확인 가능한 근거를 찾지 못했습니다.'
        )
    assert 'discarded private reason' not in response.text


@pytest.mark.parametrize('surface', ['ask', 'search'])
@pytest.mark.parametrize('kind', ['embedding', 'fallback', 'malformed'])
def test_real_embedding_graph_http_cost_backend_and_failure(tmp_path, surface, kind):
    from types import SimpleNamespace

    from langchain_core.runnables import RunnableLambda

    from backend.app.rag.pgvector_retriever import (
        PgVectorEvidenceRetriever,
        PgVectorSearchRuntimeError,
    )
    from backend.app.rag.retrieval import RagRetrieverRegistry
    from backend.tests.test_rag_v2_graph import _empty_retrieval, _paid_context

    context, provider = _paid_context(tmp_path, malformed=kind == 'malformed')
    context = replace(context, surface=surface)
    if kind == 'fallback':

        def fail_storage(request, vector):
            raise PgVectorSearchRuntimeError('unavailable storage')

        keyword = RunnableLambda(_empty_retrieval)
        registry = RagRetrieverRegistry()
        registry.register('keyword', keyword)
        registry.register(
            'pgvector',
            PgVectorEvidenceRetriever(
                store=SimpleNamespace(search=fail_storage),
                settings=context.settings,
                readiness=SimpleNamespace(
                    inspect=context.services.provider_transport._load_current_readiness
                ),
                keyword_retriever=keyword,
            ),
        )
        context = replace(
            context, services=replace(context.services, retrievers=registry)
        )
    with _graph_http(context) as http:
        response = http.post(
            f'/{surface}',
            json={('question' if surface == 'ask' else 'query'): '민감한 근거'},
        )
    assert len(provider.seen) == 1
    if kind == 'malformed':
        assert response.status_code == 503
        assert response.json() == {
            'detail': {'code': 'provider_embedding_payload_invalid'}
        }
    else:
        assert response.status_code == 200, response.text
        body = response.json()
        if surface == 'search':
            assert body == {
                'retrieval_backend': 'deterministic_lexical'
                if kind == 'fallback'
                else 'pgvector',
                'hidden_match_count': 0,
                'results': [],
                'cost_policy': {
                    'embedding_query_call': True,
                    'paid_llm_call': False,
                    'requires_pgvector_flag': True,
                },
            }
        else:
            assert body['estimated_cost_usd'] == 0.000001
            assert body['token_usage'] == {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
            }
            assert body['source_ids'] == []


@pytest.mark.parametrize(
    'kind,code,status',
    [
        ('schema', 'structured_output_invalid', 502),
        ('citation', 'citation_validation_failed', 502),
        ('identity', 'provider_response_identity_invalid', 502),
        ('missing_model', 'model_unavailable', 503),
        ('budget', 'budget_exceeded', 409),
        ('persistence', 'persistence_failed', 500),
    ],
)
def test_real_answer_graph_http_error_mapping(
    tmp_path, monkeypatch, kind, code, status
):
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
    from backend.tests.test_rag_v2_graph import _answer_context, _AnswerFinalizationPort

    context, provider = _answer_context(tmp_path)
    if kind == 'schema':
        provider.response['parsed'] = []
    elif kind == 'citation':
        provider.response['parsed']['answer_blocks'][0]['evidence_slot_ids'] = ['E8']
    elif kind == 'identity':
        provider.response['raw'].response_metadata['model'] = 'unknown'
    elif kind == 'missing_model':
        context = replace(
            context, services=replace(context.services, answer_model=None)
        )
    elif kind == 'budget':

        def refuse(**kwargs):
            raise RagBudgetExceededError()

        monkeypatch.setattr(context.services.answer_model, 'prepare_input', refuse)
    else:

        def fail_commit(_self):
            raise RuntimeError('private commit failure')

        monkeypatch.setattr(_AnswerFinalizationPort, 'commit', fail_commit)
    with _graph_http(context) as http:
        response = http.post('/ask', json={'question': 'observation'})
    assert response.status_code == status
    assert response.json() == {'detail': {'code': code}}


@pytest.mark.parametrize(
    'surface,query,outcome',
    [
        ('ask', 'observation', 'supported'),
        ('ask', 'absent', 'no_match'),
        ('search', 'observation', 'search_projected'),
    ],
)
def test_real_default_graph_facade_returns_committed_v1_product(
    tmp_path, surface, query, outcome
):
    from sqlalchemy import event, select
    from sqlalchemy.orm import Session, sessionmaker

    from backend.app.agent_runtime.rag_v2_composition import build_rag_v2_runtime
    from backend.app.core.demo_auth import USERS
    from backend.app.db.base import Base
    from backend.tests.test_rag_v2_keyword_retriever import (
        _seed_sqlite_raw_projection,
        _settings,
    )
    from backend.tests.test_rag_v2_sqlite_smoke import _engine

    path = (tmp_path / 'delivery.db').absolute()
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
    facade = build_rag_v2_runtime(
        settings=settings, session_factory=sessionmaker(bind=engine)
    ).facade
    statements = []
    event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, parameters, context, executemany: (
            statements.append(statement)
        ),
    )
    result = getattr(facade, f'invoke_{surface}')(
        actor=USERS['viewer'], caller_text=query
    )
    assert result.public_status == 200
    assert result.application_outcome == outcome
    body = result.projection.model_dump(mode='json')
    # The facade returns only after the SQLite product transaction commits.
    assert result.projection is not None
    with Session(engine) as db:
        run = db.scalar(select(AgentRun))
        assert run.status == 'complete' and run.run_record_phase == 'final'
        if surface == 'ask':
            assert body['agent_run_id'] == run.id
            assert body['question'] == query
            assert len(body['cache_key']) == 64
            assert body['token_usage'] == {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
            }
            assert body['estimated_cost_usd'] == 0.0
            assert body['model_name'] == (
                'fake-rag-v2-model'
                if outcome == 'supported'
                else 'gpt-5.4-mini-2026-03-17'
            )
        else:
            assert body['cost_policy']['embedding_query_call'] is False
            assert body['results'][0]['source_id'] == 'gmail:keyword-task-6'
    engine.dispose()


@pytest.mark.parametrize('mode', ['disabled', 'enforce'])
@pytest.mark.parametrize('endpoint,field', [('ask', 'question'), ('search', 'query')])
def test_scanner_outage_returns_typed_503_without_retrieval_or_mutation(
    client, db_session, monkeypatch, endpoint, field, mode
):
    from sqlalchemy import event

    from backend.app.agents.rag_orchestrator_agent import v2_input

    opened = []

    def forbidden_factory(**kwargs):
        opened.append(True)
        raise AssertionError('request services cannot open before scanner readiness')

    facade = client.app.state.rag_application_facade
    client.app.state.rag_application_facade = replace(
        facade,
        _settings=facade._settings.model_copy(
            update={
                'langgraph_rag_v2_mode': mode,
                'langgraph_rag_v2_stage': 'search',
            }
        ),
        _session_factory=forbidden_factory,
        _request_factory=forbidden_factory,
    )
    mutations = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().split()[0].upper() in {'INSERT', 'UPDATE', 'DELETE'}:
            mutations.append(statement)

    engine = db_session.get_bind()
    event.listen(engine, 'before_cursor_execute', record)

    def unavailable(_text):
        raise RuntimeError('scanner private failure')

    monkeypatch.setattr(v2_input, 'scan_auto_review_plaintext', unavailable)
    try:
        response = client.post(f'/api/v1/{endpoint}', json={field: 'observation'})
    finally:
        event.remove(engine, 'before_cursor_execute', record)
    assert response.status_code == 503
    assert response.json() == {'detail': {'code': 'input_scanner_unavailable'}}
    assert db_session.query(AgentRun).count() == 0
    assert opened == [] and mutations == []


@pytest.mark.parametrize('endpoint,field', [('ask', 'question'), ('search', 'query')])
def test_unsafe_input_is_distinct_from_scanner_outage(
    client, db_session, endpoint, field
):
    response = client.post(
        f'/api/v1/{endpoint}',
        json={field: 'PASSWORD=' + 'realistic-secret-value-12345'},
    )
    assert response.status_code == 422
    assert response.json() == {'detail': {'code': 'input_safety_blocked'}}
    assert db_session.query(AgentRun).count() == 0


@pytest.mark.parametrize('endpoint,field', [('ask', 'question'), ('search', 'query')])
def test_routes_deliver_only_facade_projection(
    client, db_session, monkeypatch, endpoint, field
):
    from sqlalchemy import event

    from backend.app.agent_runtime import rag_application

    assert hasattr(rag_application.RagApplicationFacade, f'invoke_{endpoint}')
    original = getattr(rag_application.RagApplicationFacade, f'invoke_{endpoint}')
    calls = []
    engine = db_session.get_bind()

    def forbidden_read(*args):
        raise AssertionError(
            'HTTP delivery reread the database after committed projection'
        )

    def invoke(self, *, actor, caller_text):
        result = original(self, actor=actor, caller_text=caller_text)
        assert result.projection is not None
        calls.append(result.projection.model_dump(mode='json'))
        event.listen(engine, 'before_cursor_execute', forbidden_read)
        return result

    monkeypatch.setattr(
        rag_application.RagApplicationFacade, f'invoke_{endpoint}', invoke
    )
    try:
        response = client.post(f'/api/v1/{endpoint}', json={field: 'no matches'})
    finally:
        if calls:
            event.remove(engine, 'before_cursor_execute', forbidden_read)
    assert response.status_code == 200
    assert calls == [response.json()]


@pytest.mark.parametrize(
    'code,component,status',
    [
        ('input_safety_blocked', None, 422),
        ('input_scanner_unavailable', None, 503),
        ('permission_denied', None, 403),
        ('budget_exceeded', None, 409),
        ('runtime_version_unavailable', None, 503),
        ('retriever_not_configured', None, 503),
        ('retriever_unavailable', None, 503),
        ('model_unavailable', None, 503),
        ('provider_safety_unavailable', None, 503),
        ('provider_response_identity_invalid', 'query_embedding', 503),
        ('provider_response_identity_invalid', 'answer_generation', 502),
        ('provider_usage_overrun', 'query_embedding', 503),
        ('provider_usage_overrun', 'answer_generation', 502),
        ('provider_embedding_payload_invalid', 'query_embedding', 503),
        ('model_provider_failed', 'answer_generation', 502),
        ('structured_output_invalid', 'answer_generation', 502),
        ('citation_validation_failed', 'answer_generation', 502),
        ('persistence_failed', None, 500),
        ('unexpected_internal_error', None, 500),
    ],
)
def test_exhaustive_public_error_status_body(code, component, status):
    import json

    from backend.app.agent_runtime.rag_application import direct_rag_error
    from backend.app.api.v1.rag_delivery import deliver_direct_rag

    response = deliver_direct_rag(direct_rag_error(code, component=component))
    assert response.status_code == status
    assert json.loads(response.body) == {'detail': {'code': code}}


def test_shared_provider_error_without_component_is_not_guessed():
    from backend.app.agent_runtime.rag_application import direct_rag_error

    result = direct_rag_error('provider_usage_overrun')
    assert (
        result.public_status == 500 and result.error.code == 'unexpected_internal_error'
    )


@pytest.mark.parametrize('surface', ['ask', 'search'])
def test_non_security_permissionerror_is_internal_not_actor_denial(tmp_path, surface):
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.tests.test_rag_v2_graph import _context

    context = _context(tmp_path)

    def unavailable(**kwargs):
        raise PermissionError('filesystem private path')

    facade = build_rag_v2_runtime(
        settings=context.settings,
        session_factory=lambda: context.services.db,
        dependencies=RagRuntimeDependencies(request_factory=unavailable),
    ).facade
    result = getattr(facade, f'invoke_{surface}')(
        actor=context.actor, caller_text='safe query'
    )
    assert result.public_status == 500
    assert result.error.code == 'unexpected_internal_error'


@pytest.mark.parametrize('surface', ['ask', 'search'])
@pytest.mark.parametrize('hidden', [0, 3])
def test_real_empty_graph_http_hidden_notice_and_no_generation(
    tmp_path, surface, hidden
):
    from langchain_core.runnables import RunnableLambda

    from backend.app.agent_runtime.rag_finalization import RagFinalizationService
    from backend.app.rag.retrieval import RagRetrieverRegistry, SanitizedRetrievalTrace
    from backend.tests.test_rag_v2_graph import (
        _context,
        _empty_retrieval,
        _KeywordFinalizationPort,
    )

    context = replace(_context(tmp_path), surface=surface)
    retriever = RunnableLambda(
        lambda request: replace(
            _empty_retrieval(request),
            hidden_match_count=hidden,
            trace=SanitizedRetrievalTrace(hidden, 0, hidden, 0, 0, None),
        )
    )
    registry = RagRetrieverRegistry()
    registry.register('keyword', retriever)
    services = replace(
        context.services,
        retrievers=registry,
        finalizer_factory=lambda pending, prepared: RagFinalizationService(
            transaction_boundary=_KeywordFinalizationPort(
                context.services.cost_ledger, pending, retriever, context.settings
            ),
            settings=context.settings,
        ),
    )
    context = replace(context, services=services)
    with _graph_http(context) as http:
        body = http.post(
            f'/{surface}', json={('question' if surface == 'ask' else 'query'): '   '}
        ).json()
    assert body['hidden_match_count'] == hidden
    if hidden:
        assert body['permission_notice'] == 'Some sources may be hidden by permissions.'
    elif surface == 'search':
        assert 'permission_notice' not in body
    else:
        assert body['permission_notice'] is None
    if surface == 'ask':
        assert body['question'] == '   '
        assert body['answer'] == '권한 내에서 확인 가능한 근거를 찾지 못했습니다.'
        assert body['permission_level'] is None
        assert body['token_usage'] == {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
        }
        assert body['estimated_cost_usd'] == 0.0
        assert all(
            body[name] == []
            for name in ('source_ids', 'source_links', 'source_snippets', 'citations')
        )
    else:
        assert body['results'] == []


@pytest.mark.parametrize('surface', ['ask', 'search'])
def test_real_graph_actor_scope_denial_is_typed_and_no_run(tmp_path, surface):
    from backend.tests.test_rag_v2_graph import _context

    context = _context(tmp_path)
    context = replace(
        context, surface=surface, actor=replace(context.actor, permission_levels=set())
    )
    with _graph_http(context) as http:
        response = http.post(
            f'/{surface}', json={('question' if surface == 'ask' else 'query'): 'query'}
        )
    assert response.status_code == 403
    assert response.json() == {'detail': {'code': 'permission_denied'}}
    assert context.services.db.query(AgentRun).count() == 0


@pytest.mark.parametrize(
    'mode,stage,surface',
    [
        ('disabled', 'assistant', 'ask'),
        ('disabled', 'assistant', 'search'),
        ('shadow', 'search', 'ask'),
        ('shadow', 'search', 'search'),
        ('enforce', 'none', 'ask'),
        ('enforce', 'ask', 'search'),
    ],
)
def test_noncutover_preserves_exact_legacy_error_body(
    client, monkeypatch, mode, stage, surface
):
    from fastapi import HTTPException

    from backend.app.agent_runtime import rag_application

    facade = client.app.state.rag_application_facade
    client.app.state.rag_application_facade = replace(
        facade,
        _settings=facade._settings.model_copy(
            update={
                'langgraph_rag_v2_mode': mode,
                'langgraph_rag_v2_stage': stage,
            }
        ),
    )
    detail = {
        'code': 'legacy-specific',
        'retrieval_backend': 'pgvector',
        'requires_pgvector_flag': True,
    }

    def refuse(**kwargs):
        raise HTTPException(status_code=503, detail=detail)

    monkeypatch.setattr(rag_application, '_legacy_pgvector_search_store', refuse)
    response = client.post(
        f'/api/v1/{surface}',
        json={('question' if surface == 'ask' else 'query'): 'safe query'},
    )
    assert response.status_code == 503
    assert response.json() == {'detail': detail}


def test_ask_permission_uses_all_committed_model_influence_not_only_citations(tmp_path):
    from backend.app.agent_runtime.rag_application import _project_graph_result
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_input import (
        prepare_direct_request_text,
    )
    from backend.tests.test_rag_v2_graph import _answer_context

    context, _ = _answer_context(tmp_path)
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {
            'prepared_text': prepare_direct_request_text(
                'observation',
                key=context.settings.agent_runtime_fingerprint_secret.encode(),
            ),
        },
        context=context,
    )
    committed = result['committed_projection']
    selected = committed.model_influence[0]
    # Unit boundary fixture: a previously authenticated, unselected restricted
    # influence is still part of the committed answer's permission envelope.
    unselected = replace(
        selected,
        dependency_role='unselected_model_influence',
        observation=replace(
            selected.observation,
            effective_permission='restricted',
        ),
    )
    result['committed_projection'] = replace(
        committed, model_influence=(selected, unselected)
    )
    body = _project_graph_result(result, surface='ask', caller_text='observation')
    assert body.citations[0].permission_level == 'internal'
    assert body.permission_level == 'restricted'


def test_legacy_facade_hidden_only_retains_truthful_application_outcome(
    client, db_session
):
    from backend.app.core.demo_auth import USERS
    from backend.tests.test_rag_orchestrator_service import seed_chunk

    seed_chunk(db_session, 'gmail', 'gmail:hidden', 'Hidden observation', 'restricted')
    result = client.app.state.rag_application_facade.invoke_ask(
        actor=USERS['viewer'], caller_text='Hidden observation'
    )
    assert result.projection.hidden_match_count == 1
    assert result.projection.source_ids == ()
    assert result.application_outcome == 'hidden_only'
