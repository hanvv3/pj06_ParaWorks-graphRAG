from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent_runtime.rag_finalization import (
    RagFinalizationService,
    SqlAlchemyRagFinalizationBoundary,
)
from backend.app.agent_runtime.rag_v2_identity import ServerRagSecurityScopeResolver
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    _prepared,
    prepare_direct_request_text,
)
from backend.app.core.demo_auth import DemoUser
from backend.app.models.agent_runs import AgentRun
from backend.app.rag.retrieval import (
    RagRetrieverRegistry,
    RetrievalResult,
    SanitizedRetrievalTrace,
)
from backend.app.rag.trusted_evidence import ServingEvidenceResolver
from backend.tests.test_rag_v2_costs import _ledger, _snapshot
from backend.tests.test_rag_v2_provider_transport import _TEST_SETTINGS


def _empty_retrieval(request):
    assert request.query_embedding_result is None
    return RetrievalResult(
        configured_backend='keyword',
        effective_backend='deterministic_lexical',
        visible=(),
        hidden_match_count=0,
        hidden_count_capped=False,
        top_candidate_window_hmac='b' * 64,
        query_embedding_receipt=None,
        trace=SanitizedRetrievalTrace(0, 0, 0, 0, 0, None),
    )


class _KeywordFinalizationPort(SqlAlchemyRagFinalizationBoundary):
    """Only synchronization is fake; finalizer reads and commits actual cost rows."""

    def __init__(self, ledger, pending, retriever, settings):
        self._db = ledger._session
        self._secret = ledger._secret
        self._settings = settings
        self._pending = pending
        self._pending_parent = None
        self._pending_children = ()
        self._prefix_generations = (1, None)
        self._phase2_authority = None
        self._retriever = retriever
        self._projection_coordinator = SimpleNamespace(
            lock_canonical_tail=lambda _ids: object(),
            validate_tail_context=lambda _tail: None,
        )

    def acquire_request_database_authority(self):
        assert not self._db.in_transaction(), (
            'graph must end the pending-read transaction'
        )
        return nullcontext()

    def close_request_database_authority(self):
        pass

    def acquire_phase2(self, pending, prepared, *, branch):
        assert branch == 'provider_free'
        return nullcontext()

    @contextmanager
    def begin(self):
        with self._db.begin():
            yield

    def acquire_projection_prefix(self):
        pass

    def commit(self):
        self._db.commit()


def _context(tmp_path, *, actor_id='graph-user'):
    from backend.app.agent_runtime.rag_v2_state import (
        RagRequestServices,
        RagRuntimeContext,
    )

    settings = _TEST_SETTINGS.model_copy(
        update={
            'langgraph_rag_v2_mode': 'enforce',
            'langgraph_rag_v2_stage': 'search',
            'rag_retrieval_backend': 'keyword',
        }
    )
    ledger = _ledger(tmp_path)
    retrievers = RagRetrieverRegistry()
    retriever = RunnableLambda(_empty_retrieval)
    retrievers.register('keyword', retriever)
    services = RagRequestServices(
        db=ledger._session,
        security_scope_resolver=ServerRagSecurityScopeResolver(settings),
        retrievers=retrievers,
        evidence_resolver=ServingEvidenceResolver(settings=settings),
        cost_policy=ledger.cost_policy_authority,
        cost_ledger=ledger,
        policy_snapshots=tuple(
            _snapshot(component, ledger.cost_policy_authority)
            for component in ('query_embedding', 'answer_generation')
        ),
        allocate_run_id=lambda: 161,
        load_generations=lambda: (1, None),
        finalizer_factory=lambda pending, prepared: RagFinalizationService(
            transaction_boundary=_KeywordFinalizationPort(
                ledger, pending, retriever, settings
            ),
            settings=settings,
        ),
    )
    return RagRuntimeContext(
        actor=DemoUser(
            id=actor_id,
            email='graph@example.test',
            name='Graph user',
            role='member',
            title='Test',
            department='Test',
            permission_levels={'public', 'internal'},
        ),
        surface='search',
        settings=settings,
        services=services,
    )


def test_actual_stategraph_keyword_search_commits_fresh_product(tmp_path):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    graph = build_company_memory_rag_answer_v2_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert graph.checkpointer is None
    context = _context(tmp_path)
    text = prepare_direct_request_text(
        '  민감한 근거  ', key=_TEST_SETTINGS.agent_runtime_fingerprint_secret.encode()
    )
    result = graph.invoke({'prepared_text': text}, context=context)
    assert result['outcome'] == 'search_projected'
    assert result['charged_cost_usd'] == Decimal('0.000000')
    assert result['evidence_projection'].citations == ()
    assert result['sanitized_trace'].provider_attempt_counts == {
        'query_embedding': 0,
        'answer_generation': 0,
    }
    parent = context.services.db.get(AgentRun, 161)
    assert parent.status == 'complete' and parent.run_record_phase == 'final'
    assert '민감한 근거' not in str(parent.metadata_)
    assert result['sanitized_trace'].node_counts['keyword_retrieval'] == 1
    assert (
        result['sanitized_trace'].node_counts['finalize_run_and_search_projection'] == 1
    )


def test_compiled_graph_exact_product_topology_and_conditional_edges():
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    graph = build_company_memory_rag_answer_v2_graph()
    expected = {
        'validate_input',
        'resolve_current_permission_context',
        'select_configured_backend',
        'preflight_retrieval_paid_cost_ceiling',
        'keyword_retrieval',
        'prepare_and_call_query_embedding',
        'pgvector_retrieval',
        'keyword_retrieval_on_pgvector_runtime_failure',
        'canonical_permission_guard',
        'rank_and_bound_evidence',
        'assemble_server_evidence_slots',
        'route_surface',
        'commit_zero_provider_search_costs_and_mark_projection_pending',
        'commit_paid_or_prepared_cost_components_and_mark_projection_pending',
        'project_visible_search_results',
        'finalize_run_and_search_projection',
        'provider_free_safe_outcome_finalizer',
        'paid_embedding_only_safe_outcome_finalizer',
        'prepare_and_preflight_answer_invocation',
        'generate_structured_answer_blocks',
        'validate_claim_evidence_refs',
        'commit_cost_components_and_mark_projection_pending',
        'revalidate_model_influence_and_selected_evidence',
        'recompute_bounded_hidden_count_and_selected_membership',
        'project_selected_server_citations',
        'finalize_run_and_answer_projection_or_assistant_message',
    }
    topology = graph.get_graph()
    assert set(topology.nodes) == expected | {'__start__', '__end__'}
    branches = {
        (edge.source, edge.target) for edge in topology.edges if edge.conditional
    }
    assert {
        ('preflight_retrieval_paid_cost_ceiling', 'keyword_retrieval'),
        ('preflight_retrieval_paid_cost_ceiling', 'prepare_and_call_query_embedding'),
        ('prepare_and_call_query_embedding', '__end__'),
        ('pgvector_retrieval', 'keyword_retrieval_on_pgvector_runtime_failure'),
        ('route_surface', 'provider_free_safe_outcome_finalizer'),
        ('route_surface', 'paid_embedding_only_safe_outcome_finalizer'),
        ('route_surface', 'prepare_and_preflight_answer_invocation'),
        ('generate_structured_answer_blocks', '__end__'),
    } <= branches
    reachable = {'__start__'}
    for _ in range(len(expected) + 2):
        reachable.update(
            edge.target for edge in topology.edges if edge.source in reachable
        )
    assert reachable == set(topology.nodes)
    assert graph.checkpointer is None


@pytest.mark.parametrize(
    'surface,count', (('assistant', 1000), ('ask', 1001), ('search', 1001))
)
def test_context_term_limit_does_not_refuse_boundary_or_direct_requests(
    tmp_path, surface, count
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context = replace(_context(tmp_path), surface=surface)
    context = replace(
        context,
        settings=context.settings.model_copy(
            update={'langgraph_rag_v2_stage': 'assistant'}
        ),
    )
    query = ' '.join(f't{i}' for i in range(count))
    key = context.settings.agent_runtime_fingerprint_secret.encode()
    text = (
        _prepared(
            caller_text='current',
            normalized_current_user_text='current',
            retrieval_query_text=query,
            answer_question_text='current',
            query_context_version='assistant-context:v1',
            key=key,
        )
        if surface == 'assistant'
        else prepare_direct_request_text(query, key=key)
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == (
        'search_projected' if surface == 'search' else 'no_match'
    )
    assert result['sanitized_trace'].node_counts['keyword_retrieval'] == 1


@pytest.mark.parametrize('hidden,outcome', ((0, 'no_match'), (3, 'hidden_only')))
def test_keyword_no_generation_safe_path_still_commits_final_product(
    tmp_path, hidden, outcome
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context = _context(tmp_path)
    context = replace(context, surface='ask')
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
    text = prepare_direct_request_text(
        '근거', key=_TEST_SETTINGS.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == outcome
    assert result['charged_cost_usd'] == Decimal('0.000000')
    assert (
        result['sanitized_trace'].node_counts['provider_free_safe_outcome_finalizer']
        == 1
    )
    assert context.services.db.get(AgentRun, 161).run_record_phase == 'final'


def test_direct_graph_assistant_1001_terms_ends_before_permission_or_db_work(tmp_path):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context = replace(_context(tmp_path), surface='assistant')
    query = ' '.join(f'term{i}' for i in range(1001))
    text = _prepared(
        caller_text='현재',
        normalized_current_user_text='현재',
        retrieval_query_text=query,
        answer_question_text='현재',
        query_context_version='assistant-context:v1',
        key=context.settings.agent_runtime_fingerprint_secret.encode(),
    )

    def fail(**_kwargs):
        raise AssertionError('permission/DB work must not begin')

    context = replace(
        context,
        services=replace(
            context.services, security_scope_resolver=SimpleNamespace(resolve=fail)
        ),
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'budget_exceeded'
    assert result['sanitized_trace'].node_counts == {'validate_input': 1}
    assert context.services.db.get(AgentRun, 161) is None


def _paid_context(tmp_path, *, malformed=False):
    from backend.app.agent_runtime.provider_usage import StrictEmbeddingUsageParser
    from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
        StrictQueryEmbeddingAdapter,
    )
    from backend.tests.test_rag_task14_finalization_bridge import (
        _SQLiteFinalizationPort,
    )
    from backend.tests.test_rag_v2_provider_transport import (
        _authority,
        _Client,
        _seed_transport_corpus,
        _transport_ledger,
    )

    base_dir = tmp_path / 'base'
    paid_dir = tmp_path / 'paid'
    base_dir.mkdir()
    paid_dir.mkdir()
    context = _context(base_dir)
    settings = context.settings.model_copy(update={'rag_retrieval_backend': 'pgvector'})
    client = _Client([])
    client.response = {
        'object': 'list',
        'model': 'text-embedding-3-small',
        'data': [
            {
                'object': 'embedding',
                'index': 0,
                'embedding': [0.0 if malformed else 0.25, *([0.0] * 1535)],
            }
        ],
        'usage': {'prompt_tokens': 1, 'total_tokens': 1},
    }
    ledger = _transport_ledger(paid_dir, client)
    _seed_transport_corpus(ledger)
    transport = _authority(ledger)
    registry = RagRetrieverRegistry()

    def vector_search(request):
        assert request.query_embedding_result is not None
        return replace(
            _empty_retrieval(replace(request, query_embedding_result=None)),
            configured_backend='pgvector',
            effective_backend='pgvector',
            query_embedding_receipt=request.query_embedding_result.receipt,
            trace=SanitizedRetrievalTrace(0, 0, 0, 1, 0, None),
        )

    registry.register('pgvector', RunnableLambda(vector_search))
    services = replace(
        context.services,
        db=ledger._session,
        cost_ledger=ledger,
        cost_policy=ledger.cost_policy_authority,
        allocate_run_id=lambda: 151,
        retrievers=registry,
        load_generations=lambda: (1, 1),
        query_embedding_adapter=StrictQueryEmbeddingAdapter(
            usage_parser=StrictEmbeddingUsageParser(),
            cost_policy=ledger.cost_policy_authority,
            transport=object(),
            settings=settings,
        ),
        index_readiness=SimpleNamespace(
            inspect=lambda **_kwargs: transport._load_current_readiness()
        ),
        provider_transport=transport,
        finalizer_factory=lambda pending, prepared: RagFinalizationService(
            transaction_boundary=_SQLiteFinalizationPort(ledger, prepared),
            settings=settings,
        ),
    )
    return replace(context, settings=settings, services=services), client


@pytest.mark.parametrize('malformed', (False, True))
def test_pgvector_graph_search_uses_one_paid_embedding_and_never_generation(
    tmp_path, malformed
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _paid_context(tmp_path, malformed=malformed)
    text = prepare_direct_request_text(
        '민감한 근거', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == (
        'provider_embedding_payload_invalid' if malformed else 'search_projected'
    )
    assert len(client.seen) == 1
    assert result['charged_cost_usd'] == Decimal('0.000001')
    assert result['sanitized_trace'].provider_attempt_counts == {
        'query_embedding': 1,
        'answer_generation': 0,
    }
    parent = context.services.db.get(AgentRun, 151)
    assert parent.run_record_phase == 'final'
    assert parent.status == ('failed' if malformed else 'complete')


def test_pgvector_runtime_failure_takes_real_conditional_keyword_node_once(tmp_path):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.rag.pgvector_retriever import (
        PgVectorEvidenceRetriever,
        PgVectorSearchRuntimeError,
    )

    context, client = _paid_context(tmp_path)
    calls = []

    def vector_search(request, vector):
        calls.append('vector')
        raise PgVectorSearchRuntimeError('sanitized storage failure')

    def lexical(request):
        calls.append('keyword')
        return _empty_retrieval(request)

    keyword = RunnableLambda(lexical)
    registry = RagRetrieverRegistry()
    registry.register('keyword', keyword)
    registry.register(
        'pgvector',
        PgVectorEvidenceRetriever(
            store=SimpleNamespace(search=vector_search),
            settings=context.settings,
            readiness=SimpleNamespace(
                inspect=context.services.provider_transport._load_current_readiness
            ),
            keyword_retriever=keyword,
        ),
    )
    context = replace(context, services=replace(context.services, retrievers=registry))
    text = prepare_direct_request_text(
        '민감한 근거', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'search_projected'
    assert result['fallback_category'] == 'pgvector_storage_runtime_failure'
    assert result['effective_backend'] == 'deterministic_lexical'
    assert calls == ['vector', 'keyword']
    assert len(client.seen) == 1
    assert (
        result['sanitized_trace'].node_counts[
            'keyword_retrieval_on_pgvector_runtime_failure'
        ]
        == 1
    )


@pytest.mark.parametrize('paid', (False, True))
def test_graph_retriever_failure_closes_parent_and_preserves_paid_cost(tmp_path, paid):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _paid_context(tmp_path) if paid else (_context(tmp_path), None)
    registry = RagRetrieverRegistry()

    def fail(_request):
        raise RuntimeError('sensitive SQL/provider detail must not escape')

    registry.register('pgvector' if paid else 'keyword', RunnableLambda(fail))
    context = replace(context, services=replace(context.services, retrievers=registry))
    text = prepare_direct_request_text(
        '민감한 근거', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'retriever_unavailable'
    assert result['charged_cost_usd'] == (
        Decimal('0.000001') if paid else Decimal('0.000000')
    )
    parent = context.services.db.get(AgentRun, 151 if paid else 161)
    assert parent.status == 'failed' and parent.run_record_phase == 'final'
    assert parent.projection_owner_fence_hmac is None
    assert 'sensitive SQL' not in str(parent.metadata_)
    assert len(client.seen) == 1 if paid else True


def test_graph_fallback_retriever_failure_terminalizes_existing_embedding_charge(
    tmp_path,
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.rag.pgvector_retriever import PgVectorFallbackRequiredError

    context, client = _paid_context(tmp_path)

    def vector(_request):
        raise PgVectorFallbackRequiredError('pgvector_storage_runtime_failure')

    def lexical(_request):
        raise RuntimeError('private storage error')

    registry = RagRetrieverRegistry()
    registry.register('pgvector', RunnableLambda(vector))
    registry.register('keyword', RunnableLambda(lexical))
    context = replace(context, services=replace(context.services, retrievers=registry))
    text = prepare_direct_request_text(
        '근거', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'retriever_unavailable'
    assert result['charged_cost_usd'] > 0 and len(client.seen) == 1
    assert context.services.db.get(AgentRun, 151).run_record_phase == 'final'


@pytest.mark.parametrize('unknown', (False, True))
def test_graph_final_product_commit_failure_returns_no_dto(
    tmp_path, monkeypatch, unknown
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(tmp_path)

    def fail_commit(boundary):
        if unknown:
            boundary._db.commit()
        raise RuntimeError('product commit acknowledgement failed')

    monkeypatch.setattr(_AnswerFinalizationPort, 'commit', fail_commit)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    with pytest.raises(RuntimeError, match='no product is available'):
        build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text}, context=context
        )
    assert len(client.seen) == 1


def test_paid_ask_no_evidence_finalizes_embedding_only_without_answer_dispatch(
    tmp_path,
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _paid_context(tmp_path)
    context = replace(context, surface='ask')
    text = prepare_direct_request_text(
        '민감한 근거', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'no_match'
    assert result['charged_cost_usd'] == Decimal('0.000001')
    assert len(client.seen) == 1
    assert (
        result['sanitized_trace'].node_counts[
            'paid_embedding_only_safe_outcome_finalizer'
        ]
        == 1
    )
    assert context.services.db.get(AgentRun, 151).run_record_phase == 'final'


def _answer_context(
    tmp_path, *, source_text='Exact current source observation', mixed_sources=False
):
    from langchain_core.messages import AIMessage

    from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agents.rag_orchestrator_agent.v2_answer import (
        StructuredRagAnswerModel,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
        build_answer_output_schema_hmac,
        build_answer_prompt_renderer_hmac,
    )
    from backend.app.models.rag_serving import RagLexicalServingProjection
    from backend.app.rag.retrieval import RetrievalCandidate
    from backend.app.rag.source_observations import CanonicalSourceObservationResolver
    from backend.tests.test_rag_source_observations import _seed_source_chunk
    from backend.tests.test_rag_v2_provider_transport import (
        _authority,
        _Client,
        _seed_transport_corpus,
        _transport_ledger,
    )

    base = tmp_path / 'base'
    paid = tmp_path / 'answer'
    base.mkdir()
    paid.mkdir()
    context = replace(_context(base), surface='ask')
    client = _Client([])
    client.response = {
        'raw': AIMessage(
            content='',
            usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
            response_metadata={
                'model': 'gpt-5.4-mini-2026-03-17',
                'object': 'response',
                'service_tier': 'default',
            },
        ),
        'parsed': {
            'answer_blocks': [
                {
                    'text': '관찰한 근거입니다.',
                    'evidence_slot_ids': ['E1'],
                    'support_mode': 'source_observation',
                }
            ],
            'insufficient_evidence_reason': None,
        },
        'parsing_error': None,
    }
    policy = RagCostPolicy(
        settings=context.settings,
        answer_output_schema_hmac=build_answer_output_schema_hmac(context.settings),
        answer_prompt_renderer_hmac=build_answer_prompt_renderer_hmac(context.settings),
    )
    ledger = _transport_ledger(paid, client, cost_policy=policy)
    _seed_transport_corpus(ledger)
    *_, chunk = _seed_source_chunk(ledger._session, text=source_text)
    observation = CanonicalSourceObservationResolver(
        db=ledger._session, settings=context.settings
    ).resolve_for_index(chunk.id)
    evidence = observation.evidence
    ledger._session.add(
        RagLexicalServingProjection(
            corpus_generation_id=1,
            corpus_generation=1,
            serving_document_id=evidence.serving_document_id,
            serving_kind=evidence.serving_kind,
            support_mode=evidence.support_mode,
            effective_permission=evidence.effective_permission,
            serving_identity_hmac=evidence.serving_identity_hmac,
            serving_version_fingerprint=evidence.serving_version_fingerprint,
            model_content_hmac=evidence.model_content_hmac,
            canonical_citation_projection_hmac=evidence.canonical_citation_projection_hmac,
            title_lower='canonical source',
            searchable_lower='observation',
            lexical_contract_version='rag-keyword-lexical-compat:v1',
            fingerprint_key_version='task-nine-v1',
            fingerprint_key_material_verifier='b' * 64,
        )
    )
    ledger._session.commit()
    candidates = [
        RetrievalCandidate(
            evidence=evidence, relevance_score=0.9, matched_terms=('observation',)
        )
    ]
    if mixed_sources:
        *_, safe_chunk = _seed_source_chunk(
            ledger._session, source_type='drive', text='Safe observation 한글'
        )
        safe = (
            CanonicalSourceObservationResolver(
                db=ledger._session, settings=context.settings
            )
            .resolve_for_index(safe_chunk.id)
            .evidence
        )
        row = ledger._session.query(RagLexicalServingProjection).one()
        values = {
            column.name: getattr(row, column.name)
            for column in RagLexicalServingProjection.__table__.columns
            if column.name not in {'id', 'created_at', 'updated_at'}
        }
        for field in (
            'serving_document_id',
            'serving_kind',
            'support_mode',
            'effective_permission',
            'serving_identity_hmac',
            'serving_version_fingerprint',
            'model_content_hmac',
            'canonical_citation_projection_hmac',
        ):
            values[field] = getattr(safe, field)
        ledger._session.add(RagLexicalServingProjection(**values))
        ledger._session.commit()
        candidates.append(
            RetrievalCandidate(
                evidence=safe, relevance_score=0.8, matched_terms=('observation',)
            )
        )
    retriever = RunnableLambda(
        lambda request: replace(
            _empty_retrieval(request),
            visible=tuple(candidates),
            trace=SanitizedRetrievalTrace(
                len(candidates), len(candidates), 0, 0, 0, None
            ),
        )
    )
    registry = RagRetrieverRegistry()
    registry.register('keyword', retriever)
    policy = ledger.cost_policy_authority
    model = StructuredRagAnswerModel(
        routed_model=RoutedRagAnswerModel(
            model=object(),
            provider='openai',
            model_name='gpt-5.4-mini-2026-03-17',
            model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
        ),
        cost_policy=policy,
    )
    services = replace(
        context.services,
        db=ledger._session,
        cost_ledger=ledger,
        cost_policy=policy,
        policy_snapshots=tuple(
            _snapshot(component, policy)
            for component in ('query_embedding', 'answer_generation')
        ),
        retrievers=registry,
        answer_model=model,
        provider_transport=_authority(ledger, answer_model=model),
        finalizer_factory=lambda pending, prepared: RagFinalizationService(
            transaction_boundary=(
                _AnswerFinalizationPort
                if prepared.validated_answer is not None
                else _KeywordFinalizationPort
            )(ledger, pending, retriever, context.settings),
            settings=context.settings,
        ),
    )
    return replace(context, services=services), client


class _AnswerFinalizationPort(_KeywordFinalizationPort):
    def __init__(
        self, ledger, pending, retriever, settings, *, component='answer_generation'
    ):
        super().__init__(ledger, pending, retriever, settings)
        from backend.app.agent_runtime.provider_send_fence import (
            _assemble_rag_evidence_barrier,
        )
        from backend.app.agent_runtime.rag_finalization import (
            _assemble_paid_rag_phase2_authority,
        )
        from backend.tests.test_rag_v2_finalization import (
            _ClosableConnection,
            _owner_capability,
        )

        binding = ledger._terminal_bindings[(161, component)]
        self._expected_branch = (
            'paid_prepared'
            if component == 'answer_generation'
            else 'paid_embedding_only'
        )
        self._phase2_authority = _assemble_paid_rag_phase2_authority(
            provider_safety=ledger._provider_safety,
            safety_connection_factory=ledger._provider_connection_factory,
            safety_requirements=((binding.policy_snapshot, binding),),
            owner_connection_factory=_ClosableConnection,
            owner_capability_factory=_owner_capability,
            load_current_owner_fence=lambda _run_id: None,
            evidence_barrier=_assemble_rag_evidence_barrier(
                load_current_identity=lambda: 'a' * 64
            ),
            load_current_readiness=lambda: None,
        )
        if component == 'query_embedding':
            from backend.tests.test_rag_v2_provider_transport import _authority

            readiness = _authority(ledger)._load_current_readiness()
            self._phase2_authority._current_readiness = readiness
            self._prefix_generations = (1, 1)

    def acquire_phase2(self, pending, prepared, *, branch):
        assert branch == self._expected_branch
        return nullcontext()


def test_keyword_answer_graph_commits_actual_model_answer_and_fresh_citation(tmp_path):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(tmp_path)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'supported'
    assert result['answer_blocks'].assembled_answer == '관찰한 근거입니다.'
    assert result['selected_slot_ids'] == ('E1',)
    assert len(result['evidence_projection'].citations) == 1
    assert len(result['model_influence']) == 1
    assert len(client.seen) == 1
    assert result['sanitized_trace'].provider_attempt_counts == {
        'query_embedding': 0,
        'answer_generation': 1,
    }
    assert context.services.db.get(AgentRun, 161).run_record_phase == 'final'


@pytest.mark.parametrize('phase', ('before_generation', 'after_generation'))
def test_answer_graph_permission_change_discards_product_and_never_retries(
    tmp_path, monkeypatch, phase
):
    from sqlalchemy import select

    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.models import Source

    context, client = _answer_context(tmp_path)

    def revoke():
        source = context.services.db.scalar(select(Source))
        source.permission_level = 'restricted'
        context.services.db.commit()

    if phase == 'before_generation':
        original = context.services.answer_model.prepare_input

        def prepare(**kwargs):
            draft = original(**kwargs)
            revoke()
            context.services.db.begin()
            return draft

        monkeypatch.setattr(context.services.answer_model, 'prepare_input', prepare)
    else:
        original = client.send

        def send(*args, **kwargs):
            response = original(*args, **kwargs)
            revoke()
            return response

        monkeypatch.setattr(client, 'send', send)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'evidence_unavailable'
    assert result['answer_blocks'] is None and result['selected_slot_ids'] == ()
    assert result['evidence_projection'].citations == ()
    assert len(client.seen) == (0 if phase == 'before_generation' else 1)
    assert (result['charged_cost_usd'] == 0) == (phase == 'before_generation')
    assert context.services.db.get(AgentRun, 161).run_record_phase == 'final'


def test_lazy_provider_construction_failure_terminalizes_undispatched_parent(tmp_path):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(tmp_path)

    def unavailable():
        raise RuntimeError('sensitive constructor detail')

    context = replace(
        context,
        services=replace(
            context.services,
            provider_transport=None,
            provider_transport_factory=unavailable,
        ),
    )
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'model_unavailable' and result['charged_cost_usd'] == 0
    assert context.services.db.get(AgentRun, 161).run_record_phase == 'final'
    assert client.seen == []


@pytest.mark.parametrize(
    'kind,outcome',
    (
        ('schema', 'structured_output_invalid'),
        ('citation', 'citation_validation_failed'),
        ('identity', 'provider_response_identity_invalid'),
    ),
)
def test_answer_graph_failed_provider_payload_commits_cost_without_product(
    tmp_path, kind, outcome
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(tmp_path)
    if kind == 'schema':
        client.response['parsed'] = []
    elif kind == 'citation':
        client.response['parsed']['answer_blocks'][0]['evidence_slot_ids'] = ['E8']
    else:
        client.response['raw'].response_metadata['model'] = 'unapproved-model'
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == outcome
    assert result['charged_cost_usd'] > 0 and len(client.seen) == 1
    assert (
        result['answer_blocks'] is None
        and result['evidence_projection'].citations == ()
    )
    parent = context.services.db.get(AgentRun, 161)
    assert parent.status == 'failed' and parent.run_record_phase == 'final'


def test_compiled_graph_concurrent_requests_keep_actor_session_model_and_cost_isolated(
    tmp_path,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    graph = build_company_memory_rag_answer_v2_graph()
    barrier = Barrier(2)

    def invoke(index):
        path = tmp_path / str(index)
        path.mkdir()
        context, client = _answer_context(path, source_text=f'private-source-{index}')
        context = replace(context, actor=replace(context.actor, id=f'actor-{index}'))
        client.response['parsed']['answer_blocks'][0]['text'] = f'answer-{index}'
        client.response['raw'].usage_metadata = {
            'input_tokens': 10 * (index + 1),
            'output_tokens': 5,
            'total_tokens': 10 * (index + 1) + 5,
        }
        original = client.send

        def send(*args, **kwargs):
            barrier.wait(timeout=20)
            return original(*args, **kwargs)

        client.send = send
        text = prepare_direct_request_text(
            f'observation-{index}',
            key=context.settings.agent_runtime_fingerprint_secret.encode(),
        )
        result = graph.invoke({'prepared_text': text}, context=context)
        parent = context.services.db.get(AgentRun, 161)
        assert parent.run_record_phase == 'final'
        assert f'answer-{index}' not in str(parent.metadata_)
        assert f'private-source-{index}' not in str(parent.metadata_)
        return (
            result,
            context.services.db,
            context.services.answer_model,
            context.services.cost_ledger,
            client,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, (0, 1)))
    for index, (result, *_) in enumerate(results):
        assert result['answer_blocks'].assembled_answer == f'answer-{index}'
        assert result['sanitized_trace'].provider_attempt_counts == {
            'query_embedding': 0,
            'answer_generation': 1,
        }
        assert f'private-source-{index}' not in str(result['sanitized_trace'])
    assert results[0][0]['charged_cost_usd'] != results[1][0]['charged_cost_usd']
    assert (
        results[0][0]['sanitized_trace'].domain_hmacs['scope']
        != results[1][0]['sanitized_trace'].domain_hmacs['scope']
    )
    for index in range(1, 5):
        assert results[0][index] is not results[1][index]
    assert graph.checkpointer is None


@pytest.mark.parametrize('kind', ('missing_model', 'budget', 'insufficient'))
def test_answer_graph_safe_and_preflight_failures_close_costs(
    tmp_path, monkeypatch, kind
):
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(tmp_path)
    if kind == 'missing_model':
        context = replace(
            context, services=replace(context.services, answer_model=None)
        )
    elif kind == 'budget':

        def refuse(**kwargs):
            raise RagBudgetExceededError

        monkeypatch.setattr(context.services.answer_model, 'prepare_input', refuse)
    else:
        client.response['parsed'] = {
            'answer_blocks': [],
            'insufficient_evidence_reason': '근거 부족',
        }
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert (
        result['outcome']
        == {
            'missing_model': 'model_unavailable',
            'budget': 'budget_exceeded',
            'insufficient': 'insufficient_evidence',
        }[kind]
    )
    assert len(client.seen) == (1 if kind == 'insufficient' else 0)
    assert context.services.db.get(AgentRun, 161).run_record_phase == 'final'
    assert result['answer_blocks'] is None
    assert result['evidence_projection'].citations == ()


@pytest.mark.parametrize('unknown', (False, True))
def test_answer_identity_binding_commit_failure_never_dispatches_or_returns_dto(
    tmp_path, monkeypatch, unknown
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(tmp_path)
    ledger = context.services.cost_ledger
    commit = ledger._commit
    commits = 0

    def fail_bind():
        nonlocal commits
        commits += 1
        if commits != 2:
            return commit()
        if unknown:
            commit()
        else:
            context.services.db.rollback()
        raise RuntimeError('binding commit acknowledgement unavailable')

    monkeypatch.setattr(ledger, '_commit', fail_bind)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    with pytest.raises(RuntimeError, match='acknowledgement unavailable'):
        build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text}, context=context
        )
    assert client.seen == []
    assert 161 not in ledger._answer_reservations
    assert (161, 'answer_generation') not in ledger._admission_budgets


def test_graph_rejects_tampered_full_model_influence_before_binding_or_dispatch(
    tmp_path, monkeypatch
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.rag.evidence_projection import CanonicalEvidenceProjector

    context, client = _answer_context(tmp_path)
    prepare = CanonicalEvidenceProjector.prepare_model_influence

    def tamper(self, *args, **kwargs):
        return replace(
            prepare(self, *args, **kwargs), aggregate_observation_hmac='f' * 64
        )

    monkeypatch.setattr(CanonicalEvidenceProjector, 'prepare_model_influence', tamper)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    with pytest.raises(ValueError, match='influence set'):
        build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text}, context=context
        )
    assert client.seen == []
    assert (
        context.services.db.get(AgentRun, 161).metadata_.get('rendered_input_hmac')
        is None
    )


def test_graph_filters_credential_only_evidence_to_zero_provider_safe_product(tmp_path):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(
        tmp_path, source_text='Authorization: Bearer test-secret-token-1234567890'
    )
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'safety_filter_empty'
    assert result['charged_cost_usd'] == Decimal('0.000000')
    assert result['evidence_projection'].citations == ()
    assert client.seen == []
    assert context.services.db.get(AgentRun, 161).run_record_phase == 'final'


def test_filtered_model_slot_preserves_canonical_source_through_paid_finalizer(
    tmp_path,
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _answer_context(
        tmp_path,
        source_text='Authorization: Bearer test-secret-token-1234567890',
        mixed_sources=True,
    )
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert result['outcome'] == 'supported'
    assert len(client.seen) == 1
    assert result['evidence_projection'].source_ids == ('drive:source-1',)
    assert result['evidence_projection'].source_snippets == ('Safe observation 한글',)
    assert len(result['model_influence']) == 1
    assert (
        result['model_influence'][0].fresh_lookup_identity.public_source_id
        == 'drive:source-1'
    )


@pytest.mark.parametrize('paid_embedding', (False, True))
@pytest.mark.parametrize('drift_kind', ('c5', 'fence'))
def test_actual_answer_transport_pre_send_evidence_drift_commits_canned_product(
    tmp_path, monkeypatch, paid_embedding, drift_kind
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderDispatchAuthority,
    )
    from backend.app.models import AgentRunCostComponent, RagLexicalServingProjection

    context, client = _answer_context(tmp_path)
    if paid_embedding:
        context = _with_paid_embedding_answer(context, client)
    original = RagProviderDispatchAuthority.prepare
    prepared_calls = []

    def drift_after_real_prepare(authority, **kwargs):
        dispatch = original(authority, **kwargs)
        if kwargs['grant'].component != 'answer_generation':
            return dispatch
        prepared_calls.append((kwargs['grant'], dispatch))
        if drift_kind == 'c5':
            row = context.services.db.query(RagLexicalServingProjection).one()
            row.model_content_hmac = 'f' * 64
            context.services.db.commit()
        else:
            monkeypatch.setattr(
                authority._barrier._freshness,
                '_load_current_identity',
                lambda: 'f' * 64,
            )
        owner = context.services.cost_ledger._safe_pending_owner

        @contextmanager
        def pending_owner(*args, **kwargs):
            assert not context.services.db.in_transaction(), (
                'C.5 read locks must end before reacquiring safety/owner'
            )
            with owner(*args, **kwargs):
                yield

        monkeypatch.setattr(
            context.services.cost_ledger, '_safe_pending_owner', pending_owner
        )
        return dispatch

    monkeypatch.setattr(
        RagProviderDispatchAuthority, 'prepare', drift_after_real_prepare
    )
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = build_company_memory_rag_answer_v2_graph().invoke(
        {'prepared_text': text}, context=context
    )
    assert len(prepared_calls) == 1 and len(client.seen) == int(paid_embedding)
    assert result['outcome'] == 'evidence_unavailable'
    assert result['evidence_projection'].citations == ()
    assert result['answer_blocks'] is None and result['model_influence'] == ()
    assert result['charged_cost_usd'] == (Decimal('0.000001') if paid_embedding else 0)
    parent = context.services.db.get(AgentRun, 161)
    assert parent.status == 'complete' and parent.run_record_phase == 'final'
    answer = (
        context.services.db.query(AgentRunCostComponent)
        .filter_by(agent_run_id=161, component='answer_generation')
        .one()
    )
    assert answer.attempted is False and answer.dispatch_count == 0
    assert answer.charged_cost_usd == 0
    from backend.app.agent_runtime.rag_cost_ledger import RagCostLedgerError
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderTransportError,
    )

    grant, dispatch = prepared_calls[0]
    with pytest.raises(RagCostLedgerError):
        context.services.cost_ledger.consume_committed_grant(grant)
    with pytest.raises(RagProviderTransportError):
        context.services.provider_transport.dispatch_and_finalize(
            grant=grant, prepared=dispatch
        )
    with pytest.raises(RagCostLedgerError):
        context.services.cost_ledger.commit_answer_evidence_changed_pending(
            grant=grant,
            corpus_generation=1,
            vector_index_generation=1 if paid_embedding else None,
        )
    assert len(client.seen) == int(paid_embedding)


@pytest.mark.parametrize('failure', ('key', 'provider_safety', 'invalid_identity'))
def test_pre_send_authority_failure_is_not_canned_evidence_drift(
    tmp_path, monkeypatch, failure
):
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderDispatchAuthority,
        RagProviderTransportError,
    )
    from backend.app.models import AutoReviewRuntimeKeyState

    context, client = _answer_context(tmp_path)
    prepare = RagProviderDispatchAuthority.prepare

    def after_prepare(authority, **kwargs):
        dispatch = prepare(authority, **kwargs)
        if failure == 'key':
            context.services.db.query(AutoReviewRuntimeKeyState).one().ready = False
            context.services.db.commit()
        elif failure == 'invalid_identity':
            monkeypatch.setattr(
                authority._barrier._freshness,
                '_load_current_identity',
                lambda: 'invalid',
            )
        else:

            def refuse(*args, **kwargs):
                raise RagProviderSafetyError('provider authority unavailable')

            monkeypatch.setattr(authority._safety, 'dispatch_barrier', refuse)
        return dispatch

    monkeypatch.setattr(RagProviderDispatchAuthority, 'prepare', after_prepare)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    with pytest.raises(RagProviderTransportError):
        build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text}, context=context
        )
    parent = context.services.db.get(AgentRun, 161)
    assert parent.status == 'failed' and parent.run_record_phase == 'final'
    assert parent.metadata_['outcome'] == 'provider_safety_unavailable'
    assert client.seen == []


@pytest.mark.parametrize('unknown', (False, True))
def test_evidence_refusal_pending_commit_failure_has_no_product_or_resend(
    tmp_path, monkeypatch, unknown
):
    from backend.app.agent_runtime.rag_cost_ledger import RagCostLedgerError
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.agent_runtime.rag_provider_transport import (
        RagProviderDispatchAuthority,
    )
    from backend.app.models import RagLexicalServingProjection

    context, client = _answer_context(tmp_path)
    prepare = RagProviderDispatchAuthority.prepare
    seen = []

    def after_prepare(authority, **kwargs):
        dispatch = prepare(authority, **kwargs)
        seen.append(kwargs['grant'])
        context.services.db.query(
            RagLexicalServingProjection
        ).one().model_content_hmac = 'f' * 64
        context.services.db.commit()
        original_commit = context.services.db.commit

        def fail_commit():
            with pytest.raises(RagCostLedgerError):
                context.services.cost_ledger.consume_committed_grant(seen[0])
            if unknown:
                original_commit()
            raise RuntimeError('pending commit acknowledgement failed')

        monkeypatch.setattr(context.services.db, 'commit', fail_commit)
        return dispatch

    monkeypatch.setattr(RagProviderDispatchAuthority, 'prepare', after_prepare)
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    with pytest.raises(RuntimeError, match='pending commit acknowledgement failed'):
        build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text}, context=context
        )
    assert client.seen == []
    with pytest.raises(RagCostLedgerError):
        context.services.cost_ledger.consume_committed_grant(seen[0])


def _with_paid_embedding_answer(context, client):
    from backend.app.agent_runtime.provider_usage import StrictEmbeddingUsageParser
    from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
        StrictQueryEmbeddingAdapter,
    )

    services = context.services
    settings = context.settings.model_copy(
        update={'rag_retrieval_backend': 'pgvector', 'rag_use_pgvector_search': True}
    )
    keyword = services.retrievers.resolve('keyword')

    def vector(request):
        return replace(
            keyword.invoke(replace(request, query_embedding_result=None)),
            configured_backend='pgvector',
            effective_backend='pgvector',
            query_embedding_receipt=request.query_embedding_result.receipt,
        )

    retriever = RunnableLambda(vector)
    registry = RagRetrieverRegistry()
    registry.register('pgvector', retriever)
    original_send = client.send

    def send(*args, **kwargs):
        response = client.response
        if not client.seen:
            client.response = {
                'object': 'list',
                'model': 'text-embedding-3-small',
                'data': [
                    {
                        'object': 'embedding',
                        'index': 0,
                        'embedding': [0.25, *([0.0] * 1535)],
                    }
                ],
                'usage': {'prompt_tokens': 1, 'total_tokens': 1},
            }
        try:
            return original_send(*args, **kwargs)
        finally:
            client.response = response

    client.send = send
    return replace(
        context,
        settings=settings,
        services=replace(
            services,
            retrievers=registry,
            load_generations=lambda: (1, 1),
            query_embedding_adapter=StrictQueryEmbeddingAdapter(
                usage_parser=StrictEmbeddingUsageParser(),
                cost_policy=services.cost_policy,
                transport=object(),
                settings=settings,
            ),
            index_readiness=SimpleNamespace(
                inspect=lambda **kwargs: (
                    services.provider_transport._load_current_readiness()
                )
            ),
            finalizer_factory=lambda pending, prepared: RagFinalizationService(
                transaction_boundary=_AnswerFinalizationPort(
                    services.cost_ledger,
                    pending,
                    retriever,
                    settings,
                    component='query_embedding',
                ),
                settings=settings,
            ),
        ),
    )


@pytest.mark.parametrize('failure', ('missing_adapter', 'constructor'))
def test_embedding_setup_failure_is_typed_and_never_generation_failure(
    tmp_path, failure
):
    from backend.app.agent_runtime.rag_application import RagApplicationError
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )

    context, client = _paid_context(tmp_path)

    def fail_factory():
        raise RuntimeError('embedding SDK construction failed')

    services = replace(
        context.services,
        provider_transport=None,
        provider_transport_factory=fail_factory,
    )
    if failure == 'missing_adapter':
        services = replace(services, query_embedding_adapter=None)
    context = replace(context, services=services)
    graph = build_company_memory_rag_answer_v2_graph()
    text = prepare_direct_request_text(
        'observation', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    if failure == 'missing_adapter':
        with pytest.raises(RagApplicationError) as error:
            graph.invoke({'prepared_text': text}, context=context)
        assert error.value.code == 'retriever_not_configured'
        assert services.db.get(AgentRun, 151) is None
    else:
        result = graph.invoke({'prepared_text': text}, context=context)
        assert result['outcome'] == 'retriever_unavailable'
        parent = services.db.get(AgentRun, 151)
        assert parent.status == 'failed' and parent.run_record_phase == 'final'
        assert result['charged_cost_usd'] == 0
    assert client.seen == []
