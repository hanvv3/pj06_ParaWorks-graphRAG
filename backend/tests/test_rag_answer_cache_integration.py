"""Real graph/canonical rows with fake provider and synchronization; PG cache I/O."""

# ruff: noqa: F811 -- imported pytest fixture is intentionally named by tests
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal
from itertools import count

import pytest
from sqlalchemy import select

from backend.app.agent_runtime.rag_finalization import RagFinalizationService
from backend.app.agent_runtime.rag_graph import build_company_memory_rag_answer_v2_graph
from backend.app.models import AgentRun, AgentRunCostComponent, AuditLog
from backend.app.models.answer_cache import RagAnswerCacheEntry
from backend.app.rag.answer_cache_store import create_answer_cache
from backend.tests.test_rag_answer_cache import pg_engine  # noqa: F401
from backend.tests.test_rag_v2_graph import (
    _answer_context,
    _KeywordFinalizationPort,
    prepare_direct_request_text,
)


@pytest.mark.parametrize('failure', [None, 'delete', 'connection'])
def test_operator_cleanup_distinguishes_empty_from_database_failure(
    pg_engine, monkeypatch, capsys, failure
):
    import sys

    import sqlalchemy
    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError

    from backend.scripts import cleanup_answer_cache
    from backend.tests.graph_projection_fixtures import SETTINGS
    from backend.tests.test_rag_answer_cache import migrate

    migrate(pg_engine)

    def fail(*args, **kwargs):
        raise OperationalError(
            'DELETE synthetic_private_sql',
            {'secret': 'synthetic_private_credential'},
            Exception('synthetic_private_connection'),
        )

    def fail_delete(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('DELETE'):
            fail()

    monkeypatch.setattr(sys, 'argv', ['cleanup_answer_cache', '--limit', '1'])
    monkeypatch.setattr(cleanup_answer_cache, 'get_settings', lambda: SETTINGS)
    monkeypatch.setattr(sqlalchemy, 'create_engine', lambda _url: pg_engine)
    hook = 'engine_connect' if failure == 'connection' else 'before_cursor_execute'
    handler = fail if failure == 'connection' else fail_delete
    if failure:
        event.listen(pg_engine, hook, handler)
    try:
        if failure:
            cache = create_answer_cache(
                engine=pg_engine, settings=SETTINGS, validator=None, enabled=True
            )
            assert cache.cleanup(limit=1) == 0  # Request path stays best effort.
        status = cleanup_answer_cache.main()
    finally:
        if failure:
            event.remove(pg_engine, hook, handler)
    captured = capsys.readouterr()
    if failure:
        assert captured.out == ''
        assert status != 0 and status is not None
        assert (
            captured.err
            == 'answer cache cleanup failed; database operation unavailable\n'
        )
        assert 'synthetic_private' not in captured.out + captured.err
    else:
        assert status == 0
        assert captured.out == 'deleted_count=0\n'
        assert captured.err == ''


def test_operator_cleanup_sqlite_null_does_not_connect(monkeypatch, capsys):
    import sys

    import sqlalchemy
    from sqlalchemy import event

    from backend.scripts import cleanup_answer_cache
    from backend.tests.graph_projection_fixtures import SETTINGS

    engine = sqlalchemy.create_engine('sqlite://')

    def forbid(*args, **kwargs):
        pytest.fail('SQLite operator cleanup must not open a connection')

    event.listen(engine, 'do_connect', forbid)
    monkeypatch.setattr(sys, 'argv', ['cleanup_answer_cache'])
    monkeypatch.setattr(cleanup_answer_cache, 'get_settings', lambda: SETTINGS)
    monkeypatch.setattr(sqlalchemy, 'create_engine', lambda _url: engine)
    assert cleanup_answer_cache.main() == 0
    captured = capsys.readouterr()
    assert captured.out == 'deleted_count=0\n'
    assert captured.err == ''


class CacheBoundary(_KeywordFinalizationPort):
    def __init__(self, ledger, pending, retriever, settings):
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
        from backend.tests.test_rag_v2_provider_transport import _authority

        bindings = tuple(
            ledger._terminal_bindings[(pending.parent_agent_run_id, name)]
            for name in ('query_embedding', 'answer_generation')
            if (pending.parent_agent_run_id, name) in ledger._terminal_bindings
        )
        if bindings:
            self._phase2_authority = _assemble_paid_rag_phase2_authority(
                provider_safety=ledger._provider_safety,
                safety_connection_factory=ledger._provider_connection_factory,
                safety_requirements=tuple((b.policy_snapshot, b) for b in bindings),
                owner_connection_factory=_ClosableConnection,
                owner_capability_factory=_owner_capability,
                load_current_owner_fence=lambda _id: None,
                evidence_barrier=_assemble_rag_evidence_barrier(
                    load_current_identity=lambda: 'a' * 64
                ),
                load_current_readiness=lambda: None,
            )
            if settings.rag_retrieval_backend == 'pgvector':
                self._phase2_authority._current_readiness = _authority(
                    ledger
                )._load_current_readiness()
                self._prefix_generations = (1, 1)

    def acquire_phase2(self, pending, prepared, *, branch):
        assert branch == (
            'answer-cache-hit:v1' if prepared.answer_cache_hit else 'paid_prepared'
        )
        return nullcontext()


def cache_context(tmp_path, pg_engine, *, paid=False):
    context, client = _answer_context(tmp_path, mixed_sources=True)
    if paid:
        import json

        from backend.tests.test_rag_v2_graph import _with_paid_embedding_answer

        original_send = client.send
        answer_response = client.response
        context = _with_paid_embedding_answer(context, client)

        def send(body, **kwargs):
            client.response = (
                {
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
                if json.loads(body)['model'] == 'text-embedding-3-small'
                else answer_response
            )
            return original_send(body, **kwargs)

        client.send = send
    RagAnswerCacheEntry.__table__.create(pg_engine)
    cache = create_answer_cache(
        engine=pg_engine,
        settings=context.settings,
        validator=context.services.answer_model._validator,
        enabled=True,
    )
    sequence = count(161)

    def factory(pending, prepared):
        return RagFinalizationService(
            transaction_boundary=CacheBoundary(
                context.services.cost_ledger,
                pending,
                context.services.retrievers.resolve(
                    context.settings.rag_retrieval_backend
                ),
                context.settings,
            ),
            settings=context.settings,
        )

    context = replace(
        context,
        services=replace(
            context.services,
            answer_cache=cache,
            allocate_run_id=lambda: next(sequence),
            finalizer_factory=factory,
        ),
    )
    return context, client, cache


def request(context):
    return {
        'prepared_text': prepare_direct_request_text(
            'observation',
            key=context.settings.agent_runtime_fingerprint_secret.encode(),
        )
    }


def test_warm_cache_retains_fresh_paid_embedding_actual_cost(tmp_path, pg_engine):
    context, client, cache = cache_context(tmp_path, pg_engine, paid=True)
    graph = build_company_memory_rag_answer_v2_graph()
    cold = graph.invoke(request(context), context=context)
    warm = graph.invoke(request(context), context=context)
    assert cold['outcome'] == warm['outcome'] == 'supported'
    assert len(client.seen) == 3
    assert warm['sanitized_trace'].provider_attempt_counts == {
        'query_embedding': 1,
        'answer_generation': 0,
    }
    assert warm['charged_cost_usd'] == Decimal('0.000001')
    assert cold['charged_cost_usd'] > warm['charged_cost_usd']
    rows = list(
        context.services.db.scalars(
            select(AgentRunCostComponent)
            .where(AgentRunCostComponent.agent_run_id == warm['run_id'])
            .order_by(AgentRunCostComponent.component_ordinal)
        )
    )
    assert rows[0].attempted and rows[0].charge_basis == 'actual'
    assert rows[0].actual_input_tokens == 1
    assert rows[1].attempted is False and rows[1].charged_cost_usd == 0


def test_composition_cleans_expired_entries_with_bounded_separate_transaction(
    tmp_path, pg_engine
):
    from backend.app.agent_runtime.rag_v2_composition import _answer_cache

    context, client, cache = cache_context(tmp_path, pg_engine)
    build_company_memory_rag_answer_v2_graph().invoke(request(context), context=context)
    from sqlalchemy import func, update

    with pg_engine.begin() as conn:
        conn.execute(update(RagAnswerCacheEntry).values(created_at=1, expires_at=2))
    from sqlalchemy.orm import Session

    with Session(pg_engine) as db:
        _answer_cache(
            db,
            context.settings.model_copy(update={'rag_answer_cache_enabled': True}),
            context.services.cost_policy,
        )
        assert not db.in_transaction()
    with pg_engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(RagAnswerCacheEntry)) == 0


def test_cold_generation_warm_substantive_hit_with_new_zero_run_and_audit(
    tmp_path, pg_engine
):
    context, client, cache = cache_context(tmp_path, pg_engine)
    request = {
        'prepared_text': prepare_direct_request_text(
            'observation',
            key=context.settings.agent_runtime_fingerprint_secret.encode(),
        )
    }
    graph = build_company_memory_rag_answer_v2_graph()
    cold = graph.invoke(request, context=context)
    warm = graph.invoke(request, context=context)
    assert cold['outcome'] == warm['outcome'] == 'supported'
    assert (
        cold['committed_projection'].answer_text
        == warm['committed_projection'].answer_text
        == '관찰한 근거입니다.'
    )
    assert len(client.seen) == 1
    assert cold['run_id'] != warm['run_id']
    assert (
        cold['committed_projection'].result_hmac
        != warm['committed_projection'].result_hmac
    )
    assert warm['sanitized_trace'].provider_attempt_counts == {
        'query_embedding': 0,
        'answer_generation': 0,
    }
    assert warm['sanitized_trace'].node_counts['keyword_retrieval'] == 1
    assert len(warm['model_influence']) == 2
    assert len(warm['evidence_projection'].citations) == 1
    assert cold['charged_cost_usd'] > 0
    assert warm['charged_cost_usd'] == Decimal('0.000000')
    db = context.services.db
    parent = db.get(AgentRun, warm['run_id'])
    assert parent.run_record_phase == 'final'
    assert parent.metadata_['answer_finalization_mode'] == 'answer-cache-hit:v1'
    costs = list(
        db.scalars(
            select(AgentRunCostComponent).where(
                AgentRunCostComponent.agent_run_id == warm['run_id']
            )
        )
    )
    assert all(
        not c.attempted
        and c.dispatch_count == 0
        and c.charged_cost_usd == 0
        and c.reserved_cost_usd == 0
        for c in costs
    )
    audit = db.scalar(
        select(AuditLog).where(
            AuditLog.action == 'rag_answer_cache_hit',
            AuditLog.target_id == str(warm['run_id']),
        )
    )
    assert audit is not None
    assert '관찰한' not in str(audit.metadata_)


@pytest.mark.parametrize(
    'change',
    [
        'same_role_principal',
        'assistant_context',
        'input_order',
        'cache_off',
        'store_failure',
    ],
)
def test_fresh_miss_never_reuses_stale_or_other_principal_answer(
    tmp_path, pg_engine, change
):
    context, client, cache = cache_context(tmp_path, pg_engine)
    graph = build_company_memory_rag_answer_v2_graph()
    first_request = request(context)
    if change == 'assistant_context':
        from backend.app.agents.rag_orchestrator_agent.v2_input import _prepared

        context = replace(
            context,
            surface='assistant',
            settings=context.settings.model_copy(
                update={'langgraph_rag_v2_stage': 'assistant'}
            ),
        )
        first_request = {
            'prepared_text': _prepared(
                caller_text='observation',
                normalized_current_user_text='observation',
                answer_question_text='observation',
                retrieval_query_text='observation',
                query_context_version='assistant-context:v1',
                key=context.settings.agent_runtime_fingerprint_secret.encode(),
            )
        }
    graph.invoke(first_request, context=context)
    next_request = request(context)
    if change == 'same_role_principal':
        context = replace(context, actor=replace(context.actor, id='another-member'))
    elif change == 'assistant_context':
        from backend.app.agents.rag_orchestrator_agent.v2_input import _prepared

        next_request = {
            'prepared_text': _prepared(
                caller_text='observation',
                normalized_current_user_text='observation',
                answer_question_text='observation',
                retrieval_query_text='earlier history observation',
                query_context_version='assistant-context:v1',
                key=context.settings.agent_runtime_fingerprint_secret.encode(),
            )
        }
        context = replace(context, surface='assistant')
        context = replace(
            context,
            settings=context.settings.model_copy(
                update={'langgraph_rag_v2_stage': 'assistant'}
            ),
        )
    elif change == 'input_order':
        # Fresh retrieval changes even though the current question is identical.
        from langchain_core.runnables import RunnableLambda

        from backend.app.rag.retrieval import RagRetrieverRegistry

        original = context.services.retrievers.resolve('keyword')
        registry = RagRetrieverRegistry()
        registry.register(
            'keyword',
            RunnableLambda(
                lambda req: replace(
                    original.invoke(req),
                    visible=tuple(reversed(original.invoke(req).visible)),
                )
            ),
        )
        context = replace(
            context, services=replace(context.services, retrievers=registry)
        )
    elif change == 'cache_off':
        context = replace(
            context, services=replace(context.services, answer_cache=None)
        )
    else:

        def fail(*a, **k):
            raise OSError('cache storage unavailable')

        cache.get = fail
    warm = graph.invoke(next_request, context=context)
    assert len(client.seen) == 2
    assert warm['sanitized_trace'].provider_attempt_counts['answer_generation'] == 1


@pytest.mark.parametrize('outcome', ['insufficient', 'generation_failure', 'redacted'])
def test_only_committed_substantive_generation_is_stored(tmp_path, pg_engine, outcome):
    from sqlalchemy import func

    from backend.app.models import Source

    context, client, cache = cache_context(tmp_path, pg_engine)
    if outcome == 'insufficient':
        client.response['parsed'] = {
            'answer_blocks': [],
            'insufficient_evidence_reason': '근거가 부족합니다.',
        }
    elif outcome == 'generation_failure':
        client.response['parsed'] = {'unrecognized': 'invalid'}
    else:
        send = client.send

        def revoked(*args, **kwargs):
            result = send(*args, **kwargs)
            source = context.services.db.scalar(select(Source).order_by(Source.id))
            source.permission_level = 'restricted'
            context.services.db.commit()
            return result

        client.send = revoked
    result = build_company_memory_rag_answer_v2_graph().invoke(
        request(context), context=context
    )
    assert result['outcome'] != 'supported'
    with pg_engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(RagAnswerCacheEntry)) == 0


@pytest.mark.parametrize(
    'drift', ['selected', 'nonselected', 'signature', 'expiry', 'authority']
)
def test_post_lookup_drift_cannot_publish_cached_answer(
    tmp_path, pg_engine, monkeypatch, drift
):
    from backend.app.agent_runtime.rag_finalization import RagFinalizationError
    from backend.app.models import Source

    context, client, cache = cache_context(tmp_path, pg_engine)
    graph = build_company_memory_rag_answer_v2_graph()
    graph.invoke(request(context), context=context)
    factory = context.services.finalizer_factory

    def changed(pending, prepared):
        assert prepared.answer_cache_hit is not None
        if drift in {'selected', 'nonselected'}:
            source = list(
                context.services.db.scalars(select(Source).order_by(Source.id))
            )[0 if drift == 'selected' else 1]
            source.permission_level = 'restricted'
            context.services.db.commit()
        elif drift == 'signature':
            row = context.services.db.get(AgentRun, pending.parent_agent_run_id)
            row.metadata_ = {**row.metadata_, 'answer_cache_value_hmac': 'a' * 64}
            context.services.db.commit()
        elif drift == 'expiry':
            monkeypatch.setattr(
                'backend.app.rag.answer_cache.time',
                lambda: prepared.answer_cache_hit.expires_at,
            )
        else:
            monkeypatch.setattr(
                'backend.app.rag.evidence_projection.CanonicalEvidenceProjector.finalize_model_influence_dependencies',
                lambda *a, **k: (_ for _ in ()).throw(
                    RuntimeError('authority unavailable')
                ),
            )
        return factory(pending, prepared)

    context = replace(
        context, services=replace(context.services, finalizer_factory=changed)
    )
    if drift in {'selected', 'nonselected'}:
        result = graph.invoke(request(context), context=context)
        assert result['outcome'] == 'evidence_unavailable'
        assert not result['evidence_projection'].citations
        assert not result['model_influence']
    else:
        with pytest.raises(RagFinalizationError):
            graph.invoke(request(context), context=context)
    assert len(client.seen) == 1
    assert (
        context.services.db.scalar(
            select(AuditLog).where(
                AuditLog.action == 'rag_answer_cache_hit', AuditLog.target_id == '162'
            )
        )
        is not None
    )


def seed_graph_relations(db, settings, scope):
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.models import TrustedKnowledgeEvidenceLink
    from backend.tests import test_rag_trusted_evidence as trusted

    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
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
    db.commit()
    return second.id, source2.id


@pytest.fixture
def real_pg_composition(tmp_path, pg_engine, monkeypatch):
    # E's isolated canonical-PG fixture shape, with real lexical SQL functions.
    import importlib

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from langchain_core.messages import AIMessage

    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
    from backend.app.agent_runtime import rag_advisory_locks as locks
    from backend.app.agent_runtime import rag_provider_transport as transport
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.app.core.demo_auth import DemoUser
    from backend.app.db import session as session_module
    from backend.app.db.base import Base
    from backend.app.db.initialization import initialize_database_runtime
    from backend.app.models import AutoReviewRuntimeKeyState, RagServingCorpusGeneration
    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections
    from backend.tests.test_rag_source_observations import _seed_source_chunk
    from backend.tests.test_rag_v2_provider_transport import (
        _TEST_SETTINGS,
        _Client,
        _seed_transport_corpus,
    )

    Base.metadata.create_all(pg_engine, checkfirst=False)
    with (
        pg_engine.begin() as conn,
        Operations.context(MigrationContext.configure(conn)),
    ):
        importlib.import_module(
            'backend.migrations.versions.d1a2b3c4e5f6_add_rag_serving_projection'
        )._install_postgresql_scorers()
    runtime = initialize_database_runtime(
        pg_engine.url.render_as_string(hide_password=False)
    )
    monkeypatch.setattr(session_module, 'engine', runtime.engine)
    monkeypatch.setattr(session_module, 'SessionLocal', runtime.session_factory)
    monkeypatch.setattr(
        session_module, 'RagPostgresDatabaseBootstrap', runtime.rag_postgres_bootstrap
    )
    settings = _TEST_SETTINGS.model_copy(
        update={
            'rag_answer_cache_enabled': True,
            'rag_retrieval_backend': 'keyword',
            'langgraph_rag_v2_mode': 'enforce',
            'langgraph_rag_v2_stage': 'assistant',
            'openai_api_key': 'fake-test-key',
            'rag_graph_enrichment_enabled': True,
            'rag_neo4j_uri': 'bolt://127.0.0.1:1',
            'rag_neo4j_username': 'unused',
            'rag_neo4j_password': 'unused',
            'paraworks_provider_safety_latch_path': str(tmp_path / 'pg-safety.json'),
        }
    )
    client = _Client([])
    client.close = lambda: None
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
                    'support_mode': 'trusted_fact',
                }
            ],
            'insufficient_evidence_reason': None,
        },
        'parsing_error': None,
    }
    monkeypatch.setattr(
        transport, '_DirectOpenAIProviderClient', lambda settings: client
    )
    actor = DemoUser(
        id='pg-cache-user',
        email='cache@example.test',
        name='Cache',
        role='member',
        title='Test',
        department='Test',
        permission_levels={'public', 'internal'},
    )
    try:
        with runtime.engine.begin() as conn:
            for identity in (
                locks.RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
                locks.RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                locks.RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
                locks.RAG_C5_KEY_CORPUS_AUTHORITY_LOCK_ID,
                locks.RAG_AGENT_RUN_COST_AUTHORITY_LOCK_ID,
            ):
                locks.register_advisory_identity_db(
                    conn, identity, identity_namespace='static'
                )
        with runtime.session_factory() as db:
            assembly = transport._assemble_rag_request_cost_authority(
                settings=settings, session=db
            )
            with assembly.provider_connection_factory() as conn:
                assembly.provider_safety.bootstrap(
                    conn,
                    composition.build_rag_provider_policy_snapshots(
                        assembly.cost_policy, settings
                    ),
                    reviewed_transition_reference_hmac='9' * 64,
                )
            _seed_transport_corpus(assembly.store)
            verifier = fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            )
            db.get(
                RagServingCorpusGeneration, 1
            ).fingerprint_key_material_verifier = verifier
            db.query(
                AutoReviewRuntimeKeyState
            ).one().fingerprint_key_material_verifier = verifier
            _seed_source_chunk(db, text='Exact current observation')
            from backend.app.agent_runtime.rag_v2_identity import (
                ServerRagSecurityScopeResolver,
            )

            scope = ServerRagSecurityScopeResolver(settings).resolve(db=db, actor=actor)
            edge_id, source_id = seed_graph_relations(db, settings, scope)
            refresh_rag_lexical_projections(db, settings=settings, corpus_generation=1)
            from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore

            PgVectorStore(
                session=db, config=PgVectorConfig(embedding_dimensions=1536)
            ).ensure_schema()
            db.commit()
            from backend.app.rag.graph_projection import (
                GraphPathDependency,
                read_projection_page,
            )

            page = read_projection_page(db, settings=settings, scope=scope)
            nodes = {n.document_id: n for n in page.nodes}
            paths = tuple(
                GraphPathDependency((nodes[e.from_id], nodes[e.to_id]), (e,))
                for e in page.edges
            )
            assert len(paths) == 2
        from backend.app.rag.graph_store import Neo4jGraphStore

        monkeypatch.setattr(Neo4jGraphStore, 'traverse', lambda self, **kwargs: paths)
        from types import SimpleNamespace

        yield SimpleNamespace(
            runtime=runtime,
            settings=settings,
            actor=actor,
            client=client,
            edge_id=edge_id,
            source_id=source_id,
        )
    finally:
        runtime.dispose()


@pytest.mark.parametrize(
    'surface,drift',
    [
        ('ask', 'none'),
        ('assistant', 'none'),
        ('ask', 'edge'),
        ('ask', 'node'),
        ('ask', 'new_before'),
    ],
)
def test_real_postgres_production_composition_cold_warm(
    real_pg_composition, surface, drift
):
    from backend.app.agent_runtime import rag_v2_composition as composition
    from backend.app.agent_runtime.rag_v2_state import RagRuntimeContext
    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections
    from backend.tests.test_rag_source_observations import _seed_source_chunk

    fixture = real_pg_composition
    runtime, settings, actor, client = (
        fixture.runtime,
        fixture.settings,
        fixture.actor,
        fixture.client,
    )
    edge_id, source_id = fixture.edge_id, fixture.source_id
    results = []
    for iteration in range(2):
        if iteration == 1 and drift == 'new_before':
            with runtime.session_factory() as added:
                _seed_source_chunk(
                    added,
                    source_type='calendar',
                    text='Canonical approved knowledge newly related evidence',
                )
                refresh_rag_lexical_projections(
                    added, settings=settings, corpus_generation=1
                )
                added.commit()
        with (
            runtime.session_factory() as db,
            composition._postgres_request_services(
                db=db, settings=settings, session_factory=runtime.session_factory
            ) as services,
        ):
            target = None
            if surface == 'assistant':
                from backend.app.agent_runtime.rag_finalization import (
                    AssistantProjectionTarget,
                )
                from backend.app.models import (
                    AssistantConversation,
                    AssistantMessage,
                )

                conversation = AssistantConversation(user_id=actor.id)
                db.add(conversation)
                db.flush()
                message = AssistantMessage(
                    conversation_id=conversation.id,
                    role='user',
                    content='Canonical approved knowledge',
                )
                db.add(message)
                db.flush()
                target = AssistantProjectionTarget(
                    conversation.id, message.id, actor.id
                )
                db.commit()
            context = RagRuntimeContext(
                actor=actor,
                surface=surface,
                settings=settings,
                services=services,
                assistant_target=target,
            )
            if iteration == 1 and drift in {'edge', 'node'}:
                get = services.answer_cache.get

                def revoke_after_lookup(*args, get=get, **kwargs):
                    hit = get(*args, **kwargs)
                    assert hit is not None
                    from backend.app.models import (
                        Source,
                        TrustedKnowledgeEvidenceLink,
                    )

                    with runtime.session_factory() as mutation:
                        if drift == 'edge':
                            mutation.get(
                                TrustedKnowledgeEvidenceLink, edge_id
                            ).evidence_hash = 'e' * 64
                        else:
                            mutation.get(
                                Source, source_id
                            ).permission_level = 'restricted'
                        refresh_rag_lexical_projections(
                            mutation, settings=settings, corpus_generation=1
                        )
                        mutation.commit()
                    return hit

                services.answer_cache.get = revoke_after_lookup
            text_value = prepare_direct_request_text(
                'Canonical approved knowledge',
                key=settings.agent_runtime_fingerprint_secret.encode(),
            )
            if surface == 'assistant':
                from backend.app.agents.rag_orchestrator_agent.v2_input import (
                    _prepared,
                )

                text_value = _prepared(
                    caller_text='Canonical approved knowledge',
                    normalized_current_user_text='Canonical approved knowledge',
                    answer_question_text='Canonical approved knowledge',
                    retrieval_query_text='Canonical approved knowledge',
                    query_context_version='assistant-context:v1',
                    key=settings.agent_runtime_fingerprint_secret.encode(),
                )
            results.append(
                build_company_memory_rag_answer_v2_graph().invoke(
                    {'prepared_text': text_value}, context=context
                )
            )
    assert [r['outcome'] for r in results] == [
        'supported',
        'evidence_unavailable' if drift in {'edge', 'node'} else 'supported',
    ]
    assert len(client.seen) == (2 if drift == 'new_before' else 1)
    assert (results[1]['charged_cost_usd'] > 0) == (drift == 'new_before')
    assert results[0]['run_id'] != results[1]['run_id']
    assert len(results[0]['model_influence']) == 4
    if surface == 'assistant':
        assert results[1]['assistant_finalization'].finalization_kind == 'substantive'
    with runtime.session_factory() as db:
        assert len(list(db.scalars(select(AgentRun)))) == 2
        assert (
            db.scalar(select(AuditLog).where(AuditLog.action == 'rag_answer_cache_hit'))
            is not None
        ) == (drift != 'new_before')
        if surface == 'assistant':
            messages = list(
                db.scalars(
                    select(AssistantMessage).where(AssistantMessage.role == 'assistant')
                )
            )
            assert len(messages) == 2
            assert all(m.content_origin == 'rag_assembled' for m in messages)
