import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from backend.app.admin.auto_review_keys import (
    fingerprint_key_material_verifier,
)
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.db.base import Base
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AutoReviewRuntimeKeyState,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    VectorIndexState,
)
from backend.app.rag.embeddings import DeterministicHashEmbeddingModel
from backend.app.rag.indexing import (
    build_rag_index_documents,
    compute_vector_document_hash,
    index_changed_vector_documents,
)
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    build_serving_lock_plan,
)
from backend.app.rag.vector_store import VectorDocument
from backend.app.review.actors import human_review_actor
from backend.app.review.auto_review_revoke import AutoReviewRevokeService
from backend.app.review.auto_review_source_reconciliation import (
    AutoReviewSourceReconciliationService,
    CommittedSourceStateChange,
)
from backend.tests.test_auto_review_source_reconciliation import (
    _seed_explicit_history,
)


class RefusingEmbeddingModel:
    dimensions = 8

    def embed(self, text: str) -> list[float]:
        raise AssertionError('stale evidence reached the embedding provider')

    def embed_many(self, texts: list[str]):
        raise AssertionError('stale evidence reached the embedding provider')


def _task6_recovery_postgres_url() -> str:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.fail(
            'PARAWORKS_TEST_POSTGRES_URL is required for Task 6 recovery tests'
        )
    parsed = make_url(database_url)
    if (
        parsed.get_backend_name() != 'postgresql'
        or not (parsed.database or '').endswith('_test')
        or not (parsed.username or '').endswith('_test')
    ):
        pytest.fail(
            'Task 6 recovery requires disposable PostgreSQL database and role '
            'names ending in _test'
        )
    return database_url


def _seed_pg_schedule(
    db,
    *,
    table_name: str,
    settings: Settings,
):
    history, item, source, _ = _seed_explicit_history(
        db, resolution_source='auto_policy'
    )
    store = PgVectorStore(
        session=db,
        config=PgVectorConfig(
            table_name=table_name, embedding_dimensions=8
        ),
        settings=settings,
    )
    store.ensure_schema()
    db.commit()
    document_id = f'history_event:{history.id}'
    document = next(
        candidate
        for candidate in build_rag_index_documents(db)
        if candidate.document_id == document_id
    )
    return history, item, source, store, document


def test_startup_recovery_removes_stale_physical_pgvector_before_ready() -> None:
    from backend.app.main import _recover_source_reconciliation_batch

    database_url = _task6_recovery_postgres_url()
    schema_name = f'task6_recovery_{uuid4().hex}'
    admin_engine = create_engine(database_url)
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema_name}'))
    engine = create_engine(
        database_url,
        connect_args={'options': f'-csearch_path={schema_name},public'},
    )
    session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    settings = Settings(
        database_url=database_url,
        openai_embedding_dimensions=8,
    )
    try:
        Base.metadata.create_all(engine)
        with session_local() as db:
            history, _, source, store, _ = _seed_pg_schedule(
                db,
                table_name='rag_vector_documents',
                settings=settings,
            )
            document_id = f'history_event:{history.id}'
            db.execute(
                text(
                    'INSERT INTO rag_vector_documents ('
                    'document_id, text, source_url, source_snippet, '
                    'permission_level, metadata_json, embedding'
                    ') VALUES ('
                    ':document_id, :text, :source_url, :source_snippet, '
                    ":permission_level, '{}'::jsonb, CAST(:embedding AS vector)"
                    ')'
                ),
                {
                    'document_id': document_id,
                    'text': 'stale trusted bytes',
                    'source_url': source.source_url,
                    'source_snippet': 'stale trusted bytes',
                    'permission_level': 'public',
                    'embedding': '[0,0,0,0,0,0,0,1]',
                },
            )
            source.permission_level = 'restricted'
            db.commit()
            assert db.scalar(
                text(
                    'SELECT count(*) FROM rag_vector_documents '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': document_id},
            ) == 1

        result = _recover_source_reconciliation_batch(
            session_local,
            settings=settings,
            limit=100,
        )

        with session_local() as db:
            physical_count = db.scalar(
                text(
                    'SELECT count(*) FROM rag_vector_documents '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': document_id},
            )
        assert result.reconciled_count == 1
        assert result.revoked_count == 1
        assert result.remaining_count == 0
        assert result.readiness is True
        assert physical_count == 0
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema_name} CASCADE'))
        admin_engine.dispose()

@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_reindex_path_with_fake_embedding() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    test_id = uuid4().hex[:8]
    table_name = f'rag_vector_documents_test_{test_id}'
    settings = Settings(database_url=database_url)

    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            signature = 'a' * 64
            verifier = fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            )
            db.add(
                AutoReviewRuntimeKeyState(
                    component='auto_review_trust_promotion',
                    fingerprint_key_version=(
                        settings.agent_runtime_fingerprint_key_version
                    ),
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=True,
                )
            )
            source = Source(
                source_type='gmail',
                source_id=f'gmail:pgvector:{test_id}',
                source_url='https://pgvector.mock/company-memory',
                title='Current pgvector evidence',
                permission_level='internal',
                raw_metadata={},
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
            )
            db.add(source)
            db.flush([source])
            db.add(
                ReviewItem(
                    item_type='document_summary',
                    payload={'source_ids': [source.source_id]},
                    source_links=[source.source_url],
                    source_snippets=['pgvector stores durable company memory'],
                    confidence_score=1.0,
                    permission_level='internal',
                    status='approved',
                    resolution_source='human',
                )
            )
            document = Document(
                source_id=source.id,
                title=source.title,
                current_version='v1',
            )
            db.add(document)
            db.flush([document])
            version = DocumentVersion(
                document_id=document.id,
                version='v1',
                body=(
                    'PostgreSQL pgvector stores durable company memory '
                    'embeddings.'
                ),
            )
            db.add(version)
            db.flush([version])
            parser_run = DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='plain_text',
                parser_status='parsed',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
                parser_policy_version='parser-policy:v1',
                parser_version='plain-text:v1',
                chunk_policy_version='chunk-policy:v1',
            )
            db.add(parser_run)
            db.flush([parser_run])
            chunk = DocumentChunk(
                version_id=version.id,
                source_id=source.id,
                parser_run_id=parser_run.id,
                chunk_index=0,
                text=version.body,
                source_snippet='pgvector stores durable company memory',
                permission_level='internal',
                metadata_={},
            )
            db.add(chunk)
            db.flush([chunk])
            document.current_document_version_id = version.id
            parser_run.chunk_count = 1
            db.commit()

            store = PgVectorStore(
                session=db,
                config=PgVectorConfig(
                    table_name=table_name, embedding_dimensions=8
                ),
                settings=settings,
            )
            store.ensure_schema()
            db.commit()
            document_id = f'chunk:{chunk.id}'
            serving_documents = build_rag_index_documents(db)
            assert [row.document_id for row in serving_documents] == [
                document_id
            ]
            result = index_changed_vector_documents(
                db=db,
                documents=serving_documents,
                writer=store,
                embedding_model=DeterministicHashEmbeddingModel(
                    dimensions=8
                ),
                embedding_model_name='deterministic-hash:integration',
                settings=settings,
            )

            search_result = store.search_with_embedding(
                query_embedding=DeterministicHashEmbeddingModel(
                    dimensions=8
                ).embed('durable company memory'),
                user=USERS['viewer'],
                limit=5,
            )

            assert result.indexed_count == 1
            assert result.embedding_request_count == 1
            assert search_result.matches[0].document.document_id == document_id

            source.permission_level = 'unknown'
            db.commit()
            plan = build_serving_lock_plan(db, [document_id])
            with KeyedMutationGuard.generation_barrier(db):
                key_context = lock_runtime_state(db)
                locked = ServingMutationLockCoordinator(
                    db=db, settings=settings
                ).acquire(key_context=key_context, plan=plan)
                store.upsert_with_embedding(
                    replace(
                        serving_documents[0],
                        text='Stale bytes must not cross relational guard.',
                    ),
                    [0.0] * 8,
                    locked_context=locked,
                )
                db.commit()
            persisted_text = db.scalar(
                text(
                    f'SELECT text FROM {table_name} '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': document_id},
            )
            assert persisted_text == serving_documents[0].text

            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_reindex_rechecks_live_evidence_before_provider_input() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    test_id = uuid4().hex[:8]
    table_name = f'rag_vector_documents_test_{test_id}'
    settings = Settings(database_url=database_url)

    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            signature = 'b' * 64
            verifier = fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            )
            db.add(
                AutoReviewRuntimeKeyState(
                    component='auto_review_trust_promotion',
                    fingerprint_key_version=(
                        settings.agent_runtime_fingerprint_key_version
                    ),
                    fingerprint_key_material_verifier=verifier,
                    generation=1,
                    ready=True,
                )
            )
            source = Source(
                source_type='gmail',
                source_id=f'gmail:pgvector-preflight:{test_id}',
                source_url='https://pgvector.mock/preflight',
                title='Provider preflight evidence',
                permission_level='internal',
                raw_metadata={},
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
            )
            db.add(source)
            db.flush([source])
            db.add(
                ReviewItem(
                    item_type='document_summary',
                    payload={'source_ids': [source.source_id]},
                    source_links=[source.source_url],
                    source_snippets=['Provider preflight evidence'],
                    confidence_score=1.0,
                    permission_level='internal',
                    status='approved',
                    resolution_source='human',
                )
            )
            document = Document(
                source_id=source.id,
                title=source.title,
                current_version='v1',
            )
            db.add(document)
            db.flush([document])
            version = DocumentVersion(
                document_id=document.id,
                version='v1',
                body='Exact bytes must not reach a provider after drift.',
            )
            db.add(version)
            db.flush([version])
            parser_run = DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='plain_text',
                parser_status='parsed',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
                parser_policy_version='parser-policy:v1',
                parser_version='plain-text:v1',
                chunk_policy_version='chunk-policy:v1',
            )
            db.add(parser_run)
            db.flush([parser_run])
            chunk = DocumentChunk(
                version_id=version.id,
                source_id=source.id,
                parser_run_id=parser_run.id,
                chunk_index=0,
                text=version.body,
                source_snippet='Provider preflight evidence',
                permission_level='internal',
                metadata_={},
            )
            db.add(chunk)
            db.flush([chunk])
            document.current_document_version_id = version.id
            parser_run.chunk_count = 1
            db.commit()

            detached = VectorDocument(
                document_id=f'chunk:{chunk.id}',
                text=chunk.text,
                source_url=source.source_url,
                source_snippet=chunk.source_snippet,
                permission_level='internal',
                metadata={'chunk_id': chunk.id, 'source_pk': source.id},
            )
            source.permission_level = 'unknown'
            db.commit()
            store = PgVectorStore(
                session=db,
                config=PgVectorConfig(
                    table_name=table_name, embedding_dimensions=8
                ),
                settings=settings,
            )
            store.ensure_schema()
            db.commit()

            result = index_changed_vector_documents(
                db=db,
                documents=[detached],
                writer=store,
                embedding_model=RefusingEmbeddingModel(),
                embedding_model_name='deterministic-hash:preflight',
                settings=settings,
            )

            assert result.indexed_count == 0
            assert result.embedding_request_count == 0
            assert result.skipped_document_ids == [detached.document_id]
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_knowledge_permission_is_strict_before_ranking_and_hidden_count() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    test_id = uuid4().hex[:8]
    table_name = f'rag_vector_documents_test_{test_id}'
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            source = Source(
                source_type='gmail',
                source_id=f'gmail:strict-permission:{test_id}',
                source_url='https://pgvector.mock/strict-permission',
                title='Strict permission source',
                permission_level='public',
                raw_metadata={},
                server_content_signature_schema='server-source-content:v1',
                server_content_signature='c' * 64,
            )
            item = ReviewItem(
                item_type='history_event',
                payload={'title': 'Restricted exact history'},
                source_links=[source.source_url],
                source_snippets=['Restricted exact history'],
                confidence_score=1.0,
                permission_level='restricted',
                status='approved',
                candidate_contract_version='c5-v1',
                resolution_source='human',
            )
            db.add_all([source, item])
            db.flush()
            history = HistoryEvent(
                project_key='strict-permission',
                title='Restricted exact history',
                reason='Only a restricted actor may see this history.',
                source_links=[source.source_url],
                source_snippets=['Restricted exact history'],
                confidence_score=1.0,
                permission_level='restricted',
                review_status='approved',
                source_review_item_id=item.id,
            )
            db.add(history)
            db.flush()
            link = TrustedKnowledgeApprovalLink(
                knowledge_type='history_event',
                knowledge_id=history.id,
                review_item_id=item.id,
                security_scope_id='workspace-a',
                promotion_effect_kind='primary',
                resolution_source='human',
                claim_fingerprint='c' * 64,
                permission_level='restricted',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='d' * 64,
                active=True,
            )
            db.add(link)
            db.flush()
            db.add(
                TrustedKnowledgeEvidenceLink(
                    approval_link_id=link.id,
                    canonical_source_kind='gmail',
                    canonical_source_id=str(source.id),
                    canonical_version_or_signature=(
                        source.server_content_signature
                    ),
                    evidence_hash='e' * 64,
                    fingerprint_key_version='v1',
                    fingerprint_key_material_verifier='d' * 64,
                )
            )
            store = PgVectorStore(
                session=db,
                config=PgVectorConfig(
                    table_name=table_name, embedding_dimensions=8
                ),
            )
            store.ensure_schema()
            document_id = f'history_event:{history.id}'
            db.execute(
                text(
                    f'INSERT INTO {table_name} '
                    '(document_id, text, source_url, source_snippet, '
                    'permission_level, metadata_json, embedding) VALUES '
                    '(:document_id, :body, :url, :snippet, '
                    ":permission, '{}'::jsonb, CAST(:embedding AS vector))"
                ),
                {
                    'document_id': document_id,
                    'body': history.reason,
                    'url': source.source_url,
                    'snippet': 'Restricted exact history',
                    'permission': 'public',
                    'embedding': '[1,0,0,0,0,0,0,0]',
                },
            )
            db.commit()

            result = store.search_with_embedding(
                query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                user=USERS['viewer'],
            )

            assert result.matches == []
            assert result.hidden_match_count == 0
            db.execute(
                text(
                    f'UPDATE {table_name} SET permission_level = '
                    "'restricted' WHERE document_id = :document_id"
                ),
                {'document_id': document_id},
            )
            db.commit()
            hidden = store.search_with_embedding(
                query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                user=USERS['viewer'],
            )
            assert hidden.matches == []
            assert hidden.hidden_match_count == 1
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_workflow_evidence_snapshot_is_strict_before_rank_and_hidden_count() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    test_id = uuid4().hex[:8]
    table_name = f'rag_vector_documents_test_{test_id}'
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            history, item, source, link = _seed_explicit_history(
                db,
                resolution_source='auto_policy',
                current_signature='f' * 64,
                evidence_signature='f' * 64,
            )
            history.permission_level = 'public'
            item.permission_level = 'public'
            source.permission_level = 'public'
            link.permission_level = 'public'
            workflow_evidence = AgentWorkflowEvidenceRef(
                workflow_thread_id=item.workflow_thread_id,
                ordinal=1,
                canonical_source_type='gmail',
                canonical_table='sources',
                canonical_row_id=source.id,
                document_version_id=None,
                external_revision=None,
                content_signature=source.server_content_signature,
                permission_level_snapshot='restricted',
                content_fingerprint='1' * 64,
            )
            db.add(workflow_evidence)
            db.flush([workflow_evidence])
            db.add(
                ReviewItemEvidenceRef(
                    review_item_id=item.id,
                    workflow_thread_id=item.workflow_thread_id,
                    workflow_evidence_ref_id=workflow_evidence.id,
                    candidate_slot_ordinal=1,
                    message_content_fingerprint='2' * 64,
                    fingerprint_key_version='v1',
                    fingerprint_key_material_verifier='3' * 64,
                )
            )
            store = PgVectorStore(
                session=db,
                config=PgVectorConfig(
                    table_name=table_name, embedding_dimensions=8
                ),
            )
            store.ensure_schema()
            document_id = f'history_event:{history.id}'
            db.execute(
                text(
                    f'INSERT INTO {table_name} '
                    '(document_id, text, source_url, source_snippet, '
                    'permission_level, metadata_json, embedding) VALUES '
                    '(:document_id, :body, :url, :snippet, '
                    ":permission, '{}'::jsonb, CAST(:embedding AS vector))"
                ),
                {
                    'document_id': document_id,
                    'body': history.reason,
                    'url': source.source_url,
                    'snippet': 'Immutable restricted workflow evidence',
                    'permission': 'public',
                    'embedding': '[1,0,0,0,0,0,0,0]',
                },
            )
            db.commit()

            stale_broad = store.search_with_embedding(
                query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                user=USERS['viewer'],
            )
            assert stale_broad.matches == []
            assert stale_broad.hidden_match_count == 0

            db.execute(
                text(
                    f'UPDATE {table_name} SET permission_level = '
                    "'restricted' WHERE document_id = :document_id"
                ),
                {'document_id': document_id},
            )
            db.commit()
            hidden = store.search_with_embedding(
                query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                user=USERS['viewer'],
            )
            visible = store.search_with_embedding(
                query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                user=USERS['admin'],
            )
            assert hidden.matches == []
            assert hidden.hidden_match_count == 1
            assert [
                match.document.document_id for match in visible.matches
            ] == [document_id]
            assert visible.hidden_match_count == 0
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_preserves_narrow_proven_pre_c5_human_knowledge() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    test_id = uuid4().hex[:8]
    table_name = f'rag_vector_documents_test_{test_id}'
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            item = ReviewItem(
                item_type='history_event',
                payload={'title': 'Legacy human history'},
                source_links=['https://legacy.mock/history'],
                source_snippets=['Legacy human history'],
                confidence_score=1.0,
                permission_level='internal',
                status='approved',
                resolution_source='human',
            )
            db.add(item)
            db.flush()
            history = HistoryEvent(
                project_key='legacy-human',
                title='Legacy human history',
                reason='A proven pre-C.5 human approval remains readable.',
                source_links=item.source_links,
                source_snippets=item.source_snippets,
                confidence_score=1.0,
                permission_level='internal',
                review_status='approved',
                source_review_item_id=item.id,
            )
            db.add(history)
            db.flush()
            store = PgVectorStore(
                session=db,
                config=PgVectorConfig(
                    table_name=table_name, embedding_dimensions=8
                ),
            )
            store.ensure_schema()
            document_id = f'history_event:{history.id}'
            db.execute(
                text(
                    f'INSERT INTO {table_name} '
                    '(document_id, text, source_url, source_snippet, '
                    'permission_level, metadata_json, embedding) VALUES '
                    '(:document_id, :body, :url, :snippet, '
                    ":permission, '{}'::jsonb, CAST(:embedding AS vector))"
                ),
                {
                    'document_id': document_id,
                    'body': history.reason,
                    'url': item.source_links[0],
                    'snippet': item.source_snippets[0],
                    'permission': 'internal',
                    'embedding': '[1,0,0,0,0,0,0,0]',
                },
            )
            db.commit()

            result = store.search_with_embedding(
                query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                user=USERS['viewer'],
            )

            assert [match.document.document_id for match in result.matches] == [
                document_id
            ]
            assert result.hidden_match_count == 0
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_schedule_reindex_commit_then_revoke_deletes_vector() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            _, item, _, store, document = _seed_pg_schedule(
                db, table_name=table_name, settings=settings
            )
            indexed = index_changed_vector_documents(
                db=db,
                documents=[document],
                writer=store,
                embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
                embedding_model_name='deterministic-hash:schedule-1',
                settings=settings,
            )
            assert indexed.indexed_count == 1
            assert db.scalar(
                text(f'SELECT count(*) FROM {table_name}')
            ) == 1

            AutoReviewRevokeService(
                db, settings=settings, vector_writer=store
            ).revoke(
                review_item_id=item.id,
                actor=human_review_actor(USERS['admin']),
                reason_code='business_withdrawal',
            )

            assert db.scalar(
                text(f'SELECT count(*) FROM {table_name}')
            ) == 0
            assert db.scalar(
                text(
                    'SELECT count(*) FROM vector_serving_tombstones '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': document.document_id},
            ) == 1
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_schedule_revoke_commit_then_guarded_upsert_affects_zero() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            _, item, _, store, detached = _seed_pg_schedule(
                db, table_name=table_name, settings=settings
            )
            AutoReviewRevokeService(
                db, settings=settings, vector_writer=store
            ).revoke(
                review_item_id=item.id,
                actor=human_review_actor(USERS['admin']),
                reason_code='business_withdrawal',
            )
            plan = build_serving_lock_plan(db, [detached.document_id])
            db.rollback()
            with KeyedMutationGuard.generation_barrier(db):
                key_context = lock_runtime_state(db)
                locked = ServingMutationLockCoordinator(
                    db=db, settings=settings
                ).acquire(key_context=key_context, plan=plan)
                store.upsert_with_embedding(
                    detached,
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    locked_context=locked,
                )
                db.commit()

            assert db.scalar(
                text(f'SELECT count(*) FROM {table_name}')
            ) == 0
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_schedule_initial_reindex_read_then_revoke_blocks_stale_write() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            _, item, _, store, detached = _seed_pg_schedule(
                db, table_name=table_name, settings=settings
            )

            class RevokeDuringProvider:
                dimensions = 8
                calls = 0

                def embed_many(self, texts):
                    self.calls += 1
                    with session_local() as revoke_db:
                        revoke_store = PgVectorStore(
                            session=revoke_db,
                            config=PgVectorConfig(
                                table_name=table_name,
                                embedding_dimensions=8,
                            ),
                            settings=settings,
                        )
                        AutoReviewRevokeService(
                            revoke_db,
                            settings=settings,
                            vector_writer=revoke_store,
                        ).revoke(
                            review_item_id=item.id,
                            actor=human_review_actor(USERS['admin']),
                            reason_code='business_withdrawal',
                        )
                    return DeterministicHashEmbeddingModel(
                        dimensions=8
                    ).embed_many(texts)

            embedding = RevokeDuringProvider()
            result = index_changed_vector_documents(
                db=db,
                documents=[detached],
                writer=store,
                embedding_model=embedding,
                embedding_model_name='deterministic-hash:schedule-3',
                settings=settings,
            )

            assert embedding.calls == 1
            assert result.indexed_count == 0
            assert result.skipped_document_ids == [detached.document_id]
            assert db.scalar(
                text(f'SELECT count(*) FROM {table_name}')
            ) == 0
            assert db.scalar(
                text(
                    'SELECT count(*) FROM vector_serving_tombstones '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': detached.document_id},
            ) == 1
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_reindex_skips_still_eligible_canonical_drift_after_provider() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            history, _, source, store, detached = _seed_pg_schedule(
                db, table_name=table_name, settings=settings
            )

            class SourceDriftDuringProvider:
                dimensions = 8
                calls = 0

                def embed_many(self, texts):
                    self.calls += 1
                    with session_local() as source_db:
                        current = source_db.get(Source, source.id)
                        current.source_url = 'https://gmail.mock/current-evidence-v2'
                        current_history = source_db.get(HistoryEvent, history.id)
                        current_history.reason = 'Exact current evidence v2'
                        current_history.source_links = [current.source_url]
                        current_history.source_snippets = [
                            'Exact current evidence v2'
                        ]
                        source_db.commit()
                    return DeterministicHashEmbeddingModel(
                        dimensions=8
                    ).embed_many(texts)

            embedding = SourceDriftDuringProvider()
            result = index_changed_vector_documents(
                db=db,
                documents=[detached],
                writer=store,
                embedding_model=embedding,
                embedding_model_name='deterministic-hash:source-race',
                settings=settings,
            )

            assert embedding.calls == 1
            assert result.indexed_count == 0
            assert result.skipped_document_ids == [detached.document_id]
            assert result.saved_embedding_calls == 0
            assert result.saved_serving_writes == 1
            assert result.embedding_request_count == 1
            assert db.scalar(
                text(f'SELECT count(*) FROM {table_name}')
            ) == 0
            assert db.scalar(
                select(VectorIndexState.id).where(
                    VectorIndexState.document_id == detached.document_id,
                    VectorIndexState.embedding_model
                    == 'deterministic-hash:source-race',
                )
            ) is None
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_reindex_applies_permission_only_narrowing_after_provider() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            history, item, source, store, _ = _seed_pg_schedule(
                db, table_name=table_name, settings=settings
            )
            link = db.scalar(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.review_item_id == item.id
                )
            )
            history.permission_level = 'public'
            item.permission_level = 'public'
            link.permission_level = 'public'
            source.permission_level = 'public'
            db.commit()
            detached = next(
                candidate
                for candidate in build_rag_index_documents(db)
                if candidate.document_id == f'history_event:{history.id}'
            )
            assert detached.permission_level == 'public'

            class NarrowDuringProvider:
                dimensions = 8
                calls = 0

                def embed_many(self, texts):
                    self.calls += 1
                    with session_local() as source_db:
                        current = source_db.get(Source, source.id)
                        current.permission_level = 'internal'
                        source_db.commit()
                    return DeterministicHashEmbeddingModel(
                        dimensions=8
                    ).embed_many(texts)

            embedding = NarrowDuringProvider()
            result = index_changed_vector_documents(
                db=db,
                documents=[detached],
                writer=store,
                embedding_model=embedding,
                embedding_model_name='deterministic-hash:permission-race',
                settings=settings,
            )

            current = next(
                candidate
                for candidate in build_rag_index_documents(db)
                if candidate.document_id == detached.document_id
            )
            stored = db.execute(
                text(
                    f'SELECT permission_level, text FROM {table_name} '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': detached.document_id},
            ).mappings().one()
            state = db.scalar(
                select(VectorIndexState).where(
                    VectorIndexState.document_id == detached.document_id,
                    VectorIndexState.embedding_model
                    == 'deterministic-hash:permission-race',
                )
            )
            assert embedding.calls == 1
            assert result.indexed_count == 1
            assert result.saved_embedding_calls == 0
            assert result.saved_serving_writes == 0
            assert stored['permission_level'] == 'internal'
            assert stored['text'] == current.text
            assert state.content_hash == compute_vector_document_hash(current)
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_reconciliation_narrows_under_coordinator_context() -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            history, item, source, store, _ = _seed_pg_schedule(
                db, table_name=table_name, settings=settings
            )
            link = db.scalar(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.review_item_id == item.id
                )
            )
            history.permission_level = 'public'
            item.permission_level = 'public'
            link.permission_level = 'public'
            source.permission_level = 'public'
            db.commit()
            detached = next(
                candidate
                for candidate in build_rag_index_documents(db)
                if candidate.document_id == f'history_event:{history.id}'
            )
            indexed = index_changed_vector_documents(
                db=db,
                documents=[detached],
                writer=store,
                embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
                embedding_model_name='deterministic-hash:reconcile-narrow',
                settings=settings,
            )
            assert indexed.indexed_count == 1
            source.permission_level = 'internal'
            db.commit()

            result = AutoReviewSourceReconciliationService(
                db, settings=settings, vector_writer=store
            ).reconcile(
                [
                    CommittedSourceStateChange(
                        source_id=source.id,
                        content_changed=False,
                        permission_changed=True,
                        parser_policy_changed=False,
                        primary_code='permission_changed',
                    )
                ]
            )

            assert result.narrowed_count == 1
            assert db.scalar(
                text(
                    f'SELECT permission_level FROM {table_name} '
                    'WHERE document_id = :document_id'
                ),
                {'document_id': detached.document_id},
            ) == 'internal'
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
