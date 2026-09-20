"""S3: actual leased PG/pgvector + Neo4j, synthetic ingestion and fake models.

Metadata/scorer schema setup follows C2, not the full migration release gate.
All target sources and approvals enter through S2; no trust rows are seeded.
"""

import importlib
import json
import os
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from langchain_core.messages import AIMessage
from neo4j import GraphDatabase
from sqlalchemy import create_engine, select

from backend.app.admin.auto_review_keys import AutoReviewKeyBootstrapService
from backend.app.agent_runtime import rag_advisory_locks as locks
from backend.app.agent_runtime import rag_provider_transport as transport
from backend.app.agent_runtime import rag_v2_composition as composition
from backend.app.agent_runtime.rag_graph import build_company_memory_rag_answer_v2_graph
from backend.app.agent_runtime.rag_v2_identity import (
    ServerRagSecurityScopeResolver,
    security_scope_fingerprint,
)
from backend.app.agent_runtime.rag_v2_state import RagRuntimeContext
from backend.app.core.demo_auth import USERS
from backend.app.db import session as session_module
from backend.app.db.base import Base
from backend.app.db.initialization import initialize_database_runtime
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import AgentRun, AgentRunCostComponent, AuditLog
from backend.app.rag.embeddings import DeterministicHashEmbeddingModel
from backend.app.rag.graph_projection import reconcile_graph_step
from backend.app.rag.graph_store import Neo4jGraphStore
from backend.app.rag.indexing import (
    build_rag_index_documents,
    build_rag_v2_index_documents,
    index_changed_vector_documents,
)
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.tests.postgres_isolation import lease_postgres_schema
from backend.tests.slack_synthetic_fixture import SyntheticSlackClient
from backend.tests.test_rag_v2_graph import prepare_direct_request_text
from backend.tests.test_rag_v2_provider_transport import _TEST_SETTINGS, _Client
from backend.tests.test_slack_synthetic_authority import (
    RecordingSlackModel,
    approve,
    draft,
    local_connector,
)


@pytest.fixture
def demo(tmp_path, monkeypatch):
    url = os.environ.get('PARAWORKS_TEST_POSTGRES_URL')
    uri = os.environ.get('PARAWORKS_TEST_NEO4J_URI')
    if not url or not uri:
        pytest.skip('explicit disposable PG and Neo4j required')
    with lease_postgres_schema(
        url, run_id=uuid4().hex[:12], scope_name='slack_s3'
    ) as lease:
        engine = create_engine(lease.database_url)
        Base.metadata.create_all(engine, checkfirst=False)
        with (
            engine.begin() as conn,
            Operations.context(MigrationContext.configure(conn)),
        ):
            importlib.import_module(
                'backend.migrations.versions.d1a2b3c4e5f6_add_rag_serving_projection'
            )._install_postgresql_scorers()
        runtime = initialize_database_runtime(lease.database_url)
        monkeypatch.setattr(session_module, 'engine', runtime.engine)
        monkeypatch.setattr(session_module, 'SessionLocal', runtime.session_factory)
        monkeypatch.setattr(
            session_module,
            'RagPostgresDatabaseBootstrap',
            runtime.rag_postgres_bootstrap,
        )
        settings = _TEST_SETTINGS.model_copy(
            update={
                'database_url': lease.database_url,
                'agent_runtime_fingerprint_secret': uuid4().hex + uuid4().hex,
                'rag_answer_cache_enabled': True,
                'rag_graph_enrichment_enabled': True,
                'rag_retrieval_backend': 'keyword',
                'langgraph_rag_v2_mode': 'enforce',
                'langgraph_rag_v2_stage': 'assistant',
                'openai_api_key': 'fake-test-key',
                'rag_neo4j_uri': uri,
                'rag_neo4j_username': os.environ['PARAWORKS_TEST_NEO4J_USER'],
                'rag_neo4j_password': os.environ['PARAWORKS_TEST_NEO4J_PASSWORD'],
                'paraworks_provider_safety_latch_path': str(tmp_path / 'safety.json'),
            }
        )
        actor = replace(USERS['viewer'], id='slack-s3-' + uuid4().hex)
        scope = ServerRagSecurityScopeResolver(settings).resolve(db=None, actor=actor)
        scope_id = security_scope_fingerprint(scope, settings=settings)
        driver = GraphDatabase.driver(
            uri, auth=(settings.rag_neo4j_username, settings.rag_neo4j_password)
        )
        store = Neo4jGraphStore(driver)
        client = _Client([])
        client.close = lambda: None
        client.response = {
            'raw': AIMessage(
                content='',
                usage_metadata={
                    'input_tokens': 10,
                    'output_tokens': 5,
                    'total_tokens': 15,
                },
                response_metadata={
                    'model': 'gpt-5.4-mini-2026-03-17',
                    'object': 'response',
                    'service_tier': 'default',
                },
            ),
            'parsed': {
                'answer_blocks': [
                    {
                        'text': '합성 검색 색인을 검증합니다.',
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
            AutoReviewKeyBootstrapService(
                session_factory=runtime.session_factory, settings=settings
            ).ensure_initialized()
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
                PgVectorStore(
                    session=db,
                    config=PgVectorConfig(embedding_dimensions=1536),
                    settings=settings,
                ).ensure_schema()
                db.commit()
            store.ensure_schema()
            yield SimpleNamespace(
                runtime=runtime,
                settings=settings,
                actor=actor,
                scope=scope,
                scope_id=scope_id,
                driver=driver,
                store=store,
                client=client,
            )
        finally:
            driver.execute_query(
                'MATCH (n) WHERE (n:PwEvidence OR n:PwProjection) AND n.scope_id=$scope DETACH DELETE n',
                scope=scope_id,
            )
            driver.close()
            runtime.dispose()
            engine.dispose()


def project(f):
    with f.runtime.session_factory() as db:
        for _ in range(12):
            result = reconcile_graph_step(
                db, settings=f.settings, scope=f.scope, store=f.store, batch_size=2
            )
            db.commit()
            if result.complete:
                return result
    pytest.fail('bounded graph projection did not converge')


def ask(f, *, configure_services=None, **flags):
    settings = f.settings.model_copy(update=flags)
    before = len(f.client.seen)
    with (
        f.runtime.session_factory() as db,
        composition._postgres_request_services(
            db=db, settings=settings, session_factory=f.runtime.session_factory
        ) as services,
    ):
        if configure_services is not None:
            configure_services(services)
        result = build_company_memory_rag_answer_v2_graph().invoke(
            {
                'prepared_text': prepare_direct_request_text(
                    '합성 pgvector 결정',
                    key=settings.agent_runtime_fingerprint_secret.encode(),
                )
            },
            context=RagRuntimeContext(
                actor=f.actor, surface='ask', settings=settings, services=services
            ),
        )
    with f.runtime.session_factory() as db:
        run = db.get(AgentRun, result['run_id'])
        assert run.run_record_phase == 'final'
        assert run.total_charged_cost_usd == result['charged_cost_usd']
        generation = db.scalar(
            select(AgentRunCostComponent).where(
                AgentRunCostComponent.agent_run_id == run.id,
                AgentRunCostComponent.component == 'answer_generation',
            )
        )
        assert generation.dispatch_count == len(f.client.seen) - before
        hit = run.metadata_.get('answer_finalization_mode') == 'answer-cache-hit:v1'
        if hit:
            assert generation.charged_cost_usd == 0
            actions = set(
                db.scalars(
                    select(AuditLog.action).where(
                        AuditLog.target_id == str(run.id),
                        AuditLog.target_type == 'agent_run',
                    )
                )
            )
            assert {'rag_answer_cache_hit', 'rag_answer_cache_finalized'} <= actions
    return result, hit


@pytest.mark.parametrize(
    'mutation',
    [
        'edit',
        'delete',
        'restrict',
        'revoke_access',
        'missing_path',
        'mismatched_path',
        'after_lookup',
    ],
)
def test_synthetic_review_graph_cache_demo(demo, mutation):
    f = demo
    fake = SyntheticSlackClient()
    model = RecordingSlackModel()
    with f.runtime.session_factory() as db:
        writer = PgVectorStore(
            session=db,
            config=PgVectorConfig(embedding_dimensions=1536),
            settings=f.settings,
        )
        first = sync_connector_events(
            db,
            local_connector(fake),
            settings=f.settings,
            vector_writer=writer,
            incremental_reindex_enqueuer=lambda job_id: None,
        )
        fake.add_late_reply()
        late = sync_connector_events(
            db,
            local_connector(fake),
            settings=f.settings,
            vector_writer=writer,
            incremental_reindex_enqueuer=lambda job_id: None,
        )
        replay = sync_connector_events(
            db,
            local_connector(fake),
            settings=f.settings,
            vector_writer=writer,
            incremental_reindex_enqueuer=lambda job_id: None,
        )
        assert (first.fetched_events, late.fetched_events, replay.skipped_events) == (3, 1, 1)
        assert first.created_review_items == late.created_review_items == 0
        item = draft(
            db, model, source_ids=[f'CPUBLIC:{fake.reply_ts}'], settings=f.settings
        )[0]
        assert item.status == 'pending_review'
        assert item.payload['token_usage']['total_tokens'] == 155
        assert (
            draft(
                db, model, source_ids=[f'CPUBLIC:{fake.reply_ts}'], settings=f.settings
            )
            == []
        )
        assert len(model.packets) == 1
        assert not any(
            d.document_id.startswith('history_event:')
            for d in build_rag_v2_index_documents(db, settings=f.settings)
        )
        item_id = item.id
    before, hit = ask(f)
    assert before['outcome'] == 'no_match' and not hit
    assert len(f.client.seen) == 0
    with f.runtime.session_factory() as db:
        from backend.app.models import ReviewItem

        approve(db, db.get(ReviewItem, item_id), settings=f.settings)
        assert approve(db, db.get(ReviewItem, item_id), settings=f.settings).replayed
        writer = PgVectorStore(
            session=db,
            config=PgVectorConfig(embedding_dimensions=1536),
            settings=f.settings,
        )
        embedding = DeterministicHashEmbeddingModel(dimensions=1536)
        docs = build_rag_index_documents(db)
        indexed = index_changed_vector_documents(
            db=db,
            documents=docs,
            writer=writer,
            embedding_model=embedding,
            embedding_model_name='s3-local:1536',
            settings=f.settings,
        )
        again = index_changed_vector_documents(
            db=db,
            documents=docs,
            writer=writer,
            embedding_model=embedding,
            embedding_model_name='s3-local:1536',
            settings=f.settings,
        )
        assert indexed.indexed_count == 2  # approved history/timeline; raw Slack stays closed
        assert (
            again.indexed_count == 0
            and again.saved_embedding_calls == indexed.indexed_count
        )
    projected = project(f)
    assert projected.edge_count == 4  # history + timeline, each parent + reply
    cold, cold_hit = ask(f)
    warm, warm_hit = ask(f)
    assert cold['outcome'] == warm['outcome'] == 'supported', (
        cold['sanitized_trace'],
        len(f.client.seen),
    )
    assert not cold_hit and warm_hit
    assert cold['run_id'] != warm['run_id']
    assert cold['effective_backend'] == warm['effective_backend'] == 'neo4j'
    assert len(cold['model_influence']) == 4
    assert len(f.client.seen) == 1
    if mutation in {'missing_path', 'mismatched_path'}:

        def break_binding(services):
            ledger = services.cost_ledger
            original = ledger.bind_answer_budget

            def bind(**kwargs):
                original(**kwargs)
                if mutation == 'missing_path':
                    ledger._graph_answer_bindings.clear()
                else:
                    key, binding = next(iter(ledger._graph_answer_bindings.items()))
                    ledger._graph_answer_bindings[key] = (
                        replace(binding[0], graph_paths=()),
                        *binding[1:],
                    )

            ledger.bind_answer_budget = bind

        denied, hit = ask(
            f, configure_services=break_binding, rag_answer_cache_enabled=False
        )
        assert denied['outcome'] == 'evidence_unavailable' and not hit
        assert not denied['evidence_projection'].citations and len(f.client.seen) == 1
        print('S3_DENIAL=' + mutation + ': zero additional provider calls')
        return
    if mutation == 'after_lookup':

        def revoke_after_lookup(services):
            get = services.answer_cache.get

            def changed(*args, **kwargs):
                hit = get(*args, **kwargs)
                assert hit is not None
                from backend.app.connectors.slack_synthetic import (
                    LocalSyntheticSlackConnector,
                )

                connector = LocalSyntheticSlackConnector(
                    channels=fake.conversations_list(),
                    messages_by_channel={},
                    deleted_messages=[('CPUBLIC', fake.parent_ts)],
                )
                with f.runtime.session_factory() as db:
                    writer = PgVectorStore(
                        session=db,
                        config=PgVectorConfig(embedding_dimensions=1536),
                        settings=f.settings,
                    )
                    sync_connector_events(
                        db,
                        connector,
                        settings=f.settings,
                        vector_writer=writer,
                        incremental_reindex_enqueuer=lambda job_id: None,
                    )
                return hit

            services.answer_cache.get = changed

        denied, _ = ask(f, configure_services=revoke_after_lookup)
        assert denied['outcome'] == 'evidence_unavailable'
        assert (
            not denied['evidence_projection'].citations
            and not denied['model_influence']
        )
        assert len(f.client.seen) == 1
        assert project(f).edge_count == 0
        print('S3_DENIAL=after_lookup: canonical delete redacts cached answer')
        return
    if mutation == 'edit':
        off, hit = ask(f, rag_answer_cache_enabled=False)
        assert off['effective_backend'] == 'neo4j' and not hit
        graph_off, hit = ask(
            f, rag_graph_enrichment_enabled=False, rag_answer_cache_enabled=False
        )
        assert graph_off['outcome'] == 'supported'
        assert graph_off['effective_backend'] == 'deterministic_lexical' and not hit
        assert len(f.client.seen) == 3
    old_scope_id = f.scope_id
    if mutation == 'revoke_access':
        f.actor = replace(f.actor, permission_levels={'public'})
        f.scope = ServerRagSecurityScopeResolver(f.settings).resolve(
            db=None, actor=f.actor
        )
    else:
        from backend.app.connectors.slack_synthetic import LocalSyntheticSlackConnector

        channels = fake.conversations_list()
        if mutation == 'restrict':
            channels[0]['is_private'] = True
        connector = LocalSyntheticSlackConnector(
            channels=channels,
            messages_by_channel={}
            if mutation == 'delete'
            else {
                'CPUBLIC': [
                    fake.message(
                        fake.parent_ts,
                        '결정: 현재 변경된 검색 정책'
                        if mutation == 'edit'
                        else '결정: pgvector를 사용합니다.',
                        'UPARENT',
                    )
                ]
            },
            deleted_messages=[('CPUBLIC', fake.parent_ts)]
            if mutation == 'delete'
            else [],
            users=fake.users_list(),
        )
        with f.runtime.session_factory() as db:
            writer = PgVectorStore(
                session=db,
                config=PgVectorConfig(embedding_dimensions=1536),
                settings=f.settings,
            )
            sync_connector_events(
                db,
                connector,
                settings=f.settings,
                vector_writer=writer,
                incremental_reindex_enqueuer=lambda job_id: None,
            )
    sends = len(f.client.seen)
    # The old Neo4j scope and cache entries still exist here: fresh PG authority
    # must block exposure before asynchronous projection catches up.
    assert f.store.status(old_scope_id).edge_count == 4
    stale, hit = ask(f)
    assert stale['outcome'] != 'supported' and not hit
    assert not stale['evidence_projection'].citations and len(f.client.seen) == sends
    if mutation != 'revoke_access':
        assert project(f).edge_count == 0
    f.scope_id = old_scope_id
    print(
        'S3_DEMO='
        + json.dumps(
            {
                'fetched': [first.fetched_events, late.fetched_events],
                'created_review_items': first.created_review_items
                + late.created_review_items,
                'pending_created': 1,
                'replay_skipped': replay.skipped_events,
                'saved_extraction_calls': 1,
                'indexed': indexed.indexed_count,
                'saved_embedding_calls': again.saved_embedding_calls,
                'edges': projected.edge_count,
                'mutation': mutation,
                'cold_run': cold['run_id'],
                'warm_run': warm['run_id'],
                'warm_cost': str(warm['charged_cost_usd']),
                'generation_calls': len(f.client.seen),
            }
        )
    )
