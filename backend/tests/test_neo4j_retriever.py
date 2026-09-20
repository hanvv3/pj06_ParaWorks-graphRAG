# ruff: noqa: F811
import importlib
from dataclasses import replace

import pytest
from langchain_core.runnables import RunnableLambda

from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.rag.graph_projection import GraphPathDependency, read_projection_page
from backend.app.rag.graph_store import GraphUnavailable
from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
from backend.app.rag.retrieval import RetrievalRequest
from backend.app.rag.search_store import SqlAlchemyKeywordSearchStore
from backend.tests.graph_projection_fixtures import SCOPE, SETTINGS, seed_corpus
from backend.tests.test_graph_projection_neo4j import canonical_pg  # noqa: F401


@pytest.fixture
def lexical_pg(canonical_pg):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = importlib.import_module(
        'backend.migrations.versions.d1a2b3c4e5f6_add_rag_serving_projection'
    )
    with Operations.context(MigrationContext.configure(canonical_pg.connection())):
        migration._install_postgresql_scorers()
    canonical_pg.commit()
    return canonical_pg


def module():
    assert importlib.util.find_spec('backend.app.rag.neo4j_retriever'), (
        'graph retriever missing'
    )
    return importlib.import_module('backend.app.rag.neo4j_retriever')


def request(scope=SCOPE):
    return RetrievalRequest(
        'Canonical approved knowledge',
        scope,
        security_scope_fingerprint(scope, settings=SETTINGS),
        None,
        50,
        5,
        'rag-retrieval-policy:v2.0',
    )


def paths(db):
    page = read_projection_page(db, settings=SETTINGS, scope=SCOPE)
    nodes = {node.document_id: node for node in page.nodes}
    return tuple(
        GraphPathDependency((nodes[e.from_id], nodes[e.to_id]), (e,))
        for e in page.edges
    )


def adapter(db, store, seed=None):
    m = module()
    return m.Neo4jEvidenceRetriever(
        db=db,
        settings=SETTINGS,
        graph_store=store,
        seed_retriever=seed
        or KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=SETTINGS),
            settings=SETTINGS,
        ),
    )


def test_relation_adds_only_canonical_evidence_and_preserves_ordered_paths(db_session):
    seed_corpus(db_session)
    expected = paths(db_session)

    class Store:
        def traverse(self, **kwargs):
            return expected

    result = adapter(db_session, Store()).invoke(request())
    assert result.effective_backend == 'neo4j'
    assert {c.evidence.serving_document_id for c in result.visible} == {
        'history_event:1',
        'chunk:1',
        'chunk:2',
    }
    assert result.graph_paths == expected
    assert result.trace.candidate_window_count <= 50


@pytest.mark.parametrize('graph_state', ['unavailable', 'stale', 'no_benefit'])
def test_eight_seed_ask_preserves_exact_evidence_and_receipt(db_session, graph_state):
    from decimal import Decimal

    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections
    from backend.app.rag.retrieval import QueryEmbeddingReceipt
    from backend.tests.test_rag_trusted_evidence import _seed_source

    seed_corpus(db_session)
    for ordinal in range(4, 9):
        _seed_source(db_session, ordinal=ordinal)
    refresh_rag_lexical_projections(db_session, settings=SETTINGS, corpus_generation=1)
    db_session.commit()
    ask = replace(request(), retrieval_query_text='Exact evidence', visible_limit=8)
    original = KeywordEvidenceRetriever(
        store=SqlAlchemyKeywordSearchStore(db=db_session, settings=SETTINGS),
        settings=SETTINGS,
    ).invoke(ask)
    assert len(original.visible) == 8
    receipt = QueryEmbeddingReceipt(
        True, 10, Decimal('0.000001'), 1, 'component_succeeded', 'a' * 64, 'b' * 64
    )
    original = replace(original, query_embedding_receipt=receipt)
    calls = []

    class Store:
        def traverse(self, **kwargs):
            calls.append(kwargs)
            if graph_state == 'unavailable':
                raise GraphUnavailable('offline')
            return ()  # incomplete/stale projection or no matching relation

    result = adapter(db_session, Store(), RunnableLambda(lambda _: original)).invoke(
        ask
    )
    assert result.visible == original.visible
    assert result.query_embedding_receipt is receipt
    assert result.top_candidate_window_hmac == original.top_candidate_window_hmac
    assert result.trace.visible_count == 8
    assert not result.graph_paths
    assert not calls  # Decline enrichment before graph I/O when seed fills its cap.


def test_driver_orders_oversubscribed_subset_before_limit(db_session):
    from dataclasses import asdict

    from backend.app.rag.graph_projection import GraphTraversalPolicy
    from backend.app.rag.graph_store import Neo4jGraphStore

    seed_corpus(db_session)
    expected = paths(db_session)
    physical_orders = iter((expected[::-1], expected))

    class Transaction:
        def run(self, query, **parameters):
            rows = next(physical_orders)
            if 'ORDER BY r.edge_id' in query:
                rows = sorted(rows, key=lambda p: p.edges[0].edge_id)
            return [
                {
                    'left': asdict(p.nodes[0]),
                    'right': asdict(p.nodes[1]),
                    'edge': asdict(p.edges[0]),
                }
                for p in rows[: parameters['limit']]
            ]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute_read(self, callback):
            return callback(Transaction())

    class Driver:
        def session(self, **kwargs):
            return Session()

    store = Neo4jGraphStore(Driver())
    results = [
        store.traverse(
            scope_id=expected[0].nodes[0].scope_id,
            generation=1,
            seeds=('history_event:1',),
            permissions=('public', 'internal'),
            policy=GraphTraversalPolicy(candidate_limit=1),
        )
        for _ in range(2)
    ]
    assert results == [(expected[0],), (expected[0],)]


@pytest.mark.parametrize(
    'change', ['restricted', 'revoke', 'edge_version', 'delete', 'scope']
)
def test_stale_path_is_not_authority(db_session, change):
    sources, chunks, _, approval = seed_corpus(db_session)
    expected = paths(db_session)
    if change == 'restricted':
        sources[1].permission_level = chunks[1].permission_level = 'restricted'
    elif change == 'revoke':
        approval.active = False
    elif change == 'edge_version':
        from backend.app.models import TrustedKnowledgeEvidenceLink

        db_session.get(TrustedKnowledgeEvidenceLink, 2).evidence_hash = 'e' * 64
    elif change == 'delete':
        from backend.app.models import TrustedKnowledgeEvidenceLink

        db_session.delete(db_session.get(TrustedKnowledgeEvidenceLink, 2))
    db_session.commit()
    scope = replace(SCOPE, workspace_scope_id='other') if change == 'scope' else SCOPE
    assert not module().validate_graph_paths(
        db_session, expected, settings=SETTINGS, scope=scope
    )


def test_graph_outage_reuses_seed_once_without_provider_or_embedding(db_session):
    seed_corpus(db_session)
    keyword = KeywordEvidenceRetriever(
        store=SqlAlchemyKeywordSearchStore(db=db_session, settings=SETTINGS),
        settings=SETTINGS,
    )
    original = keyword.invoke(request())
    calls = []

    def seed(value):
        calls.append(value)
        return original

    class Store:
        def traverse(self, **kwargs):
            raise GraphUnavailable('graph unavailable')

    result = adapter(db_session, Store(), RunnableLambda(seed)).invoke(request())
    assert result.visible == original.visible
    assert result.query_embedding_receipt is original.query_embedding_receipt
    assert len(calls) == 1 and not result.graph_paths
    assert result.trace.fallback_category == 'graph_unavailable'


def test_prepared_influence_preserves_paths_and_rejects_edge_only_drift(db_session):
    from backend.app.models import TrustedKnowledgeEvidenceLink
    from backend.app.rag.evidence_projection import (
        CanonicalEvidenceProjector,
        ProjectionFence,
    )
    from backend.app.rag.retrieval import EvidenceSlot

    seed_corpus(db_session)
    expected = paths(db_session)

    class Store:
        def traverse(self, **kwargs):
            return expected

    result = adapter(db_session, Store()).invoke(request())
    slots = tuple(
        EvidenceSlot(
            slot_id=f'E{i + 1}',
            support_mode=c.evidence.support_mode,
            evidence=c.evidence,
            relevance_score=c.relevance_score,
            matched_terms=c.matched_terms,
        )
        for i, c in enumerate(result.visible)
    )
    projector = CanonicalEvidenceProjector(db=db_session, settings=SETTINGS)
    assert (
        'graph_paths'
        in __import__('inspect').signature(projector.prepare_model_influence).parameters
    )
    prepared = projector.prepare_model_influence(
        slots,
        scope=SCOPE,
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
        rendered_input_hmac='a' * 64,
        graph_paths=result.graph_paths,
    )
    assert prepared.graph_paths == expected
    assert prepared.observations
    fence = ProjectionFence(
        1,
        None,
        1,
        None,
        prepared_hidden_membership_hmac='b' * 64,
        current_hidden_membership_hmac='b' * 64,
    )
    assert projector.finalize_model_influence_dependencies(
        prepared, ('E1',), scope=SCOPE, fence=fence
    )
    db_session.get(TrustedKnowledgeEvidenceLink, 2).evidence_hash = 'e' * 64
    db_session.flush()
    assert not projector.finalize_model_influence_dependencies(
        prepared, ('E1',), scope=SCOPE, fence=fence
    )


def test_real_pg_keyword_fallback_binding(lexical_pg):
    canonical_pg = lexical_pg
    seed_corpus(canonical_pg)
    from backend.app.rag.lexical_projection import tokenize_rag_lexical_query

    SqlAlchemyKeywordSearchStore(db=canonical_pg, settings=SETTINGS)._postgresql_search(
        request(), terms=tokenize_rag_lexical_query(request().retrieval_query_text)
    )

    class Store:
        def traverse(self, **kwargs):
            raise GraphUnavailable('offline')

    result = adapter(canonical_pg, Store()).invoke(request())
    assert result.visible and result.effective_backend == 'deterministic_lexical'
    assert result.trace.fallback_category == 'graph_unavailable'


def test_real_pg_neo4j_parameterized_traversal(lexical_pg):
    canonical_pg = lexical_pg
    import os
    from uuid import uuid4

    from neo4j import GraphDatabase

    from backend.app.rag.graph_projection import (
        GraphTraversalPolicy,
        reconcile_graph_step,
    )
    from backend.app.rag.graph_store import Neo4jGraphStore

    uri = os.getenv('PARAWORKS_TEST_NEO4J_URI')
    if not uri:
        pytest.skip('disposable Neo4j required')
    seed_corpus(canonical_pg)
    scope = replace(SCOPE, principal_subject='e2-' + uuid4().hex)
    scope_id = security_scope_fingerprint(scope, settings=SETTINGS)
    driver = GraphDatabase.driver(
        uri,
        auth=(
            os.environ['PARAWORKS_TEST_NEO4J_USER'],
            os.environ['PARAWORKS_TEST_NEO4J_PASSWORD'],
        ),
        max_transaction_retry_time=0,
    )
    store = Neo4jGraphStore(driver)
    try:
        for _ in range(5):
            status = reconcile_graph_step(
                canonical_pg, settings=SETTINGS, scope=scope, store=store
            )
            canonical_pg.commit()
            if status.complete:
                break
        assert status.complete
        assert hasattr(store, 'traverse'), 'bounded official driver traversal missing'
        result = adapter(canonical_pg, store).invoke(request(scope))
        assert result.effective_backend == 'neo4j'
        assert len(result.graph_paths) == 2
        assert len(result.visible) == 3
        # Two eligible edges exceed the remaining one-candidate budget.
        # Seed/physical visitation order must not select a different subset.
        expected_edge = min(p.edges[0].edge_id for p in result.graph_paths)
        for seeds in (('history_event:1', 'chunk:1'), ('chunk:1', 'history_event:1')):
            bounded = store.traverse(
                scope_id=scope_id,
                generation=1,
                seeds=seeds,
                permissions=('public', 'internal'),
                policy=GraphTraversalPolicy(candidate_limit=1),
            )
            assert tuple(p.edges[0].edge_id for p in bounded) == (expected_edge,)
        assert not store.traverse(
            scope_id=scope_id,
            generation=2,
            seeds=('history_event:1',),
            permissions=('public', 'internal'),
            policy=GraphTraversalPolicy(),
        )
        assert not store.traverse(
            scope_id='foreign',
            generation=1,
            seeds=('history_event:1',),
            permissions=('public', 'internal'),
            policy=GraphTraversalPolicy(),
        )
    finally:
        with driver.session() as session:
            session.run(
                'MATCH (n {scope_id:$scope}) DETACH DELETE n', scope=scope_id
            ).consume()
        driver.close()


def test_configured_composition_wraps_seed_and_default_is_off(db_session):
    from backend.app.agent_runtime import rag_v2_composition

    seed = RunnableLambda(lambda value: value)
    assert hasattr(rag_v2_composition, '_graph_enrichment'), (
        'production graph assembly missing'
    )
    assert rag_v2_composition._graph_enrichment(db_session, SETTINGS, seed) is seed
    enabled = SETTINGS.model_copy(
        update={
            'rag_graph_enrichment_enabled': True,
            'rag_neo4j_uri': 'bolt://127.0.0.1:1',
            'rag_neo4j_username': 'unused',
            'rag_neo4j_password': 'unused',
        }
    )
    assert isinstance(
        rag_v2_composition._graph_enrichment(db_session, enabled, seed),
        module().Neo4jEvidenceRetriever,
    )


@pytest.mark.parametrize(
    'drift', ['none', 'before_send', 'after_send', 'missing_binding', 'changed_binding']
)
def test_actual_answer_graph_rechecks_unselected_edge_before_send_and_exposure(
    tmp_path, monkeypatch, drift
):
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.agent_runtime.rag_finalization import RagFinalizationService
    from backend.app.agent_runtime.rag_graph import (
        build_company_memory_rag_answer_v2_graph,
    )
    from backend.app.models import (
        AutoReviewRuntimeKeyState,
        RagServingCorpusGeneration,
        TrustedKnowledgeEvidenceLink,
    )
    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections
    from backend.app.rag.retrieval import RagRetrieverRegistry
    from backend.tests import test_rag_trusted_evidence as trusted
    from backend.tests import test_rag_v2_costs as costs
    from backend.tests import test_rag_v2_graph as graph_tests
    from backend.tests.test_rag_v2_provider_transport import _TEST_SETTINGS

    original_snapshot = costs._snapshot
    verifier = fingerprint_key_material_verifier(
        _TEST_SETTINGS.agent_runtime_fingerprint_secret
    )
    monkeypatch.setattr(
        costs,
        '_snapshot',
        lambda component, policy=None: replace(
            original_snapshot(component, policy),
            fingerprint_key_version=_TEST_SETTINGS.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
        ),
    )
    monkeypatch.setattr(graph_tests, '_snapshot', costs._snapshot)
    context, client = graph_tests._answer_context(tmp_path)
    db, settings, ledger = (
        context.services.db,
        context.settings,
        context.services.cost_ledger,
    )
    scope = context.services.security_scope_resolver.resolve(db=db, actor=context.actor)
    source1, chunk1 = trusted._seed_source(db, ordinal=11)
    source2, chunk2 = trusted._seed_source(db, ordinal=12)
    item = trusted._seed_item(
        db, source=source1, chunk=chunk1, resolution_source='human'
    )
    item.source_links = [source1.source_url, source2.source_url]
    item.source_snippets = [chunk1.source_snippet, chunk2.source_snippet]
    target = trusted._seed_target(db, source_review_item_id=item.id)
    approval = trusted._seed_link(
        db, target=target, item=item, source=source1, resolution_source='human'
    )
    approval.security_scope_id = scope.workspace_scope_id
    approval.fingerprint_key_version = settings.agent_runtime_fingerprint_key_version
    approval.fingerprint_key_material_verifier = verifier
    first = (
        db.query(TrustedKnowledgeEvidenceLink)
        .filter_by(approval_link_id=approval.id)
        .one()
    )
    first.fingerprint_key_version = settings.agent_runtime_fingerprint_key_version
    first.fingerprint_key_material_verifier = verifier
    second = TrustedKnowledgeEvidenceLink(
        approval_link_id=approval.id,
        canonical_source_kind='gmail',
        canonical_source_id=str(source2.id),
        canonical_version_or_signature=source2.server_content_signature,
        evidence_hash='d' * 64,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=verifier,
    )
    db.add(second)
    db.get(RagServingCorpusGeneration, 1).fingerprint_key_material_verifier = verifier
    db.query(AutoReviewRuntimeKeyState).filter_by(
        component='auto_review_trust_promotion'
    ).one().fingerprint_key_material_verifier = verifier
    db.flush()
    refresh_rag_lexical_projections(db, settings=settings, corpus_generation=1)
    db.commit()
    page = read_projection_page(db, settings=settings, scope=scope)
    nodes = {n.document_id: n for n in page.nodes}
    expected = tuple(
        GraphPathDependency((nodes[e.from_id], nodes[e.to_id]), (e,))
        for e in page.edges
    )
    assert len(expected) == 2

    class Store:
        def traverse(self, **kwargs):
            return expected

    retriever = module().Neo4jEvidenceRetriever(
        db=db,
        settings=settings,
        graph_store=Store(),
        seed_retriever=KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=settings),
            settings=settings,
        ),
    )
    registry = RagRetrieverRegistry()
    registry.register('keyword', retriever)

    def change_edge():
        db.get(TrustedKnowledgeEvidenceLink, second.id).evidence_hash = 'e' * 64
        db.commit()

    if drift in {'before_send', 'missing_binding', 'changed_binding'}:
        bind = ledger.bind_answer_budget

        def mutate_after_bind(**kwargs):
            bind(**kwargs)
            if drift == 'before_send':
                change_edge()
            elif drift == 'missing_binding':
                ledger._graph_answer_bindings.clear()
            else:
                key, binding = next(iter(ledger._graph_answer_bindings.items()))
                ledger._graph_answer_bindings[key] = (
                    replace(binding[0], graph_paths=()),
                    *binding[1:],
                )

        monkeypatch.setattr(ledger, 'bind_answer_budget', mutate_after_bind)
    if drift == 'after_send':
        send = client.send

        def mutate_after_send(*args, **kwargs):
            result = send(*args, **kwargs)
            change_edge()
            return result

        monkeypatch.setattr(client, 'send', mutate_after_send)
    client.response['parsed']['answer_blocks'][0]['support_mode'] = 'trusted_fact'

    def finalizer(pending, prepared):
        boundary = (
            graph_tests._AnswerFinalizationPort
            if prepared.validated_answer is not None
            else graph_tests._KeywordFinalizationPort
        )
        return RagFinalizationService(
            transaction_boundary=boundary(ledger, pending, retriever, settings),
            settings=settings,
        )

    context = replace(
        context,
        services=replace(
            context.services, retrievers=registry, finalizer_factory=finalizer
        ),
    )
    text = graph_tests.prepare_direct_request_text(
        'Canonical approved knowledge',
        key=settings.agent_runtime_fingerprint_secret.encode(),
    )
    # Assemble the production keyword + graph registry and lazy model transport;
    # only external graph/provider I/O and PG synchronization ports are fake.
    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.app.rag.graph_store import Neo4jGraphStore

    settings = settings.model_copy(
        update={
            'rag_graph_enrichment_enabled': True,
            'rag_neo4j_uri': 'bolt://127.0.0.1:1',
            'rag_neo4j_username': 'unused',
            'rag_neo4j_password': 'unused',
            'openai_api_key': 'fake-not-a-real-key',
        }
    )
    authority = context.services.provider_transport
    assembly = transport.RagRequestCostAuthority(
        db,
        ledger,
        context.services.cost_policy,
        ledger.provider_safety_authority,
        ledger.provider_connection_factory,
        authority._barrier,
        authority._load_current_readiness,
        ledger.runtime_health_authority,
        object(),
    )
    monkeypatch.setattr(
        transport, '_assemble_rag_request_cost_authority', lambda **kwargs: assembly
    )
    monkeypatch.setattr(
        transport, '_DirectOpenAIProviderClient', lambda actual_settings: client
    )
    monkeypatch.setattr(client, 'close', lambda: None, raising=False)
    monkeypatch.setattr(Neo4jGraphStore, 'traverse', lambda self, **kwargs: expected)
    with composition._postgres_request_services(
        db=db, settings=settings, session_factory=lambda: None
    ) as composed:
        context = replace(
            context,
            settings=settings,
            services=replace(
                composed,
                allocate_run_id=context.services.allocate_run_id,
                finalizer_factory=finalizer,
            ),
        )
        result = build_company_memory_rag_answer_v2_graph().invoke(
            {'prepared_text': text}, context=context
        )
    assert len(client.seen) == (
        0 if drift in {'before_send', 'missing_binding', 'changed_binding'} else 1
    ), (result.get('outcome'), result.get('sanitized_trace'))
    assert result['outcome'] == (
        'supported' if drift == 'none' else 'evidence_unavailable'
    )
    if drift == 'after_send':
        assert result['charged_cost_usd'] > 0
        assert not result['evidence_projection'].citations


def test_actual_pg_prepared_paths_revalidate_unselected_edge(lexical_pg):
    test_prepared_influence_preserves_paths_and_rejects_edge_only_drift(lexical_pg)


def test_candidate_budget_is_shared_with_seeds_and_receipt_is_reused(db_session):
    from decimal import Decimal

    from backend.app.rag.retrieval import QueryEmbeddingReceipt

    seed_corpus(db_session)
    original = KeywordEvidenceRetriever(
        store=SqlAlchemyKeywordSearchStore(db=db_session, settings=SETTINGS),
        settings=SETTINGS,
    ).invoke(request())
    receipt = QueryEmbeddingReceipt(
        True, 10, Decimal('0.000001'), 1, 'component_succeeded', 'a' * 64, 'b' * 64
    )
    original = replace(
        original,
        configured_backend='pgvector',
        effective_backend='pgvector',
        query_embedding_receipt=receipt,
        trace=replace(original.trace, candidate_window_count=50),
    )

    class Store:
        def traverse(self, **kwargs):
            pytest.fail('candidate budget exhausted by seed search')

    calls = []

    def seed(value):
        calls.append(value)
        return original

    result = adapter(db_session, Store(), RunnableLambda(seed)).invoke(request())
    assert len(calls) == 1
    assert result.query_embedding_receipt is receipt
    assert result.trace.candidate_window_count == 50 and not result.graph_paths


def test_authority_read_failure_never_falls_back_to_graph_or_seed(
    db_session, monkeypatch
):
    seed_corpus(db_session)
    expected = paths(db_session)

    class Store:
        def traverse(self, **kwargs):
            return expected

    def fail(*args, **kwargs):
        raise RuntimeError('canonical read failed')

    monkeypatch.setattr(module(), 'validate_graph_paths', fail)
    with pytest.raises(RuntimeError, match='canonical read failed'):
        adapter(db_session, Store()).invoke(request())


def test_graph_trace_includes_traversal_elapsed_time(db_session, monkeypatch):
    seed_corpus(db_session)
    expected = paths(db_session)

    class Store:
        def traverse(self, **kwargs):
            return expected

    ticks = iter((10_000_000, 27_000_000))
    monkeypatch.setattr(module(), 'perf_counter_ns', lambda: next(ticks), raising=False)
    result = adapter(db_session, Store()).invoke(request())
    assert result.trace.latency_ms == 17
