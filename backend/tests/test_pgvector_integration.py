import os
from contextlib import contextmanager
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
from backend.app.agents.rag_orchestrator_agent.service import (
    build_serving_dependency_snapshot,
    candidates_from_vector_matches,
    filter_live_serving_candidates,
)
from backend.app.assistant.service import (
    append_assistant_message,
    assistant_message_evidence_is_live,
    create_conversation,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.db.base import Base
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
    AutoReviewRuntimeKeyState,
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TimelineEvent,
    Todo,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    TrustedKnowledgeFingerprint,
    VectorIndexState,
    VectorServingTombstone,
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


def _convert_pg_schedule_to_decision(
    db,
    *,
    history: HistoryEvent,
    item: ReviewItem,
    knowledge_type: str,
    permission_level: str,
) -> tuple[DecisionRecord, TrustedKnowledgeApprovalLink, VectorDocument]:
    decision = DecisionRecord(
        project_key='project-a',
        title='Canonical decision',
        decision_summary='Legacy decision mutation evidence',
        source_links=item.source_links,
        source_snippets=item.source_snippets,
        confidence_score=0.99,
        permission_level=permission_level,
        review_status='approved',
        source_review_item_id=item.id,
    )
    db.add(decision)
    db.flush()
    link = (
        db.query(TrustedKnowledgeApprovalLink)
        .filter_by(
            knowledge_type='history_event',
            knowledge_id=history.id,
        )
        .one()
    )
    link.knowledge_type = knowledge_type
    link.knowledge_id = decision.id
    link.permission_level = permission_level
    item.item_type = 'decision_record'
    item.permission_level = permission_level
    db.commit()
    document_id = f'decision_record:{decision.id}'
    document = next(
        candidate
        for candidate in build_rag_index_documents(db)
        if candidate.document_id == document_id
    )
    return decision, link, document


def _add_canonical_decision_fingerprint(
    db,
    *,
    decision: DecisionRecord,
    link: TrustedKnowledgeApprovalLink,
    permission_level: str,
) -> TrustedKnowledgeFingerprint:
    fingerprint = TrustedKnowledgeFingerprint(
        knowledge_type='decision_record',
        knowledge_id=decision.id,
        security_scope_id='workspace-a',
        scope_resolution='exact',
        project_scope_hmac='a' * 64,
        normalized_title_bucket_hmac='b' * 64,
        normalized_claim_fingerprint='c' * 64,
        fingerprint_key_version=link.fingerprint_key_version,
        fingerprint_key_material_verifier=link.fingerprint_key_material_verifier,
        permission_level=permission_level,
        review_status='approved',
    )
    db.add(fingerprint)
    db.commit()
    return fingerprint


@contextmanager
def _pgvector_test_db(database_url: str):
    engine = create_engine(database_url)
    session_local = sessionmaker(bind=engine)
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(database_url=database_url, openai_embedding_dimensions=8)
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            yield db, table_name, settings
    finally:
        with engine.begin() as connection:
            connection.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
        Base.metadata.drop_all(engine)
        engine.dispose()


def _seed_pg_decision_schedule(
    db,
    *,
    table_name: str,
    settings: Settings,
    knowledge_type: str,
    permission_level: str,
):
    history, item, source, store, _ = _seed_pg_schedule(
        db,
        table_name=table_name,
        settings=settings,
    )
    source.permission_level = permission_level
    db.commit()
    decision, link, document = _convert_pg_schedule_to_decision(
        db,
        history=history,
        item=item,
        knowledge_type=knowledge_type,
        permission_level=permission_level,
    )
    fingerprint = _add_canonical_decision_fingerprint(
        db,
        decision=decision,
        link=link,
        permission_level=permission_level,
    )
    return decision, item, source, link, fingerprint, store, document


def _seed_pg_canonical_knowledge_schedule(
    db,
    *,
    table_name: str,
    settings: Settings,
    knowledge_type: str,
):
    history, item, source, store, history_document = _seed_pg_schedule(
        db,
        table_name=table_name,
        settings=settings,
    )
    link = (
        db.query(TrustedKnowledgeApprovalLink)
        .filter_by(knowledge_type='history_event', knowledge_id=history.id)
        .one()
    )
    if knowledge_type == 'history_event':
        return history, link, store, history_document
    if knowledge_type == 'decision_record':
        decision, link, document = _convert_pg_schedule_to_decision(
            db,
            history=history,
            item=item,
            knowledge_type='decision_record',
            permission_level='internal',
        )
        return decision, link, store, document
    if knowledge_type == 'timeline_event':
        target = TimelineEvent(
            project_key='project-a',
            title='Canonical timeline',
            result_summary='Canonical timeline serving evidence',
            source_links=list(item.source_links),
            source_snippets=list(item.source_snippets),
            confidence_score=0.99,
            permission_level='internal',
            review_status='approved',
            source_review_item_id=item.id,
        )
    elif knowledge_type == 'todo':
        target = Todo(
            project_key='project-a',
            title='Canonical todo',
            assignee=None,
            due_date=None,
            priority='high',
            priority_reason='Canonical todo serving evidence',
            source_links=list(item.source_links),
            source_snippets=list(item.source_snippets),
            confidence_score=0.99,
            permission_level='internal',
            review_status='approved',
            source_review_item_id=item.id,
        )
    else:
        raise AssertionError(f'unsupported test knowledge type: {knowledge_type}')
    db.add(target)
    db.flush([target])
    item.item_type = knowledge_type
    link.knowledge_type = knowledge_type
    link.knowledge_id = target.id
    db.commit()
    document_id = f'{knowledge_type}:{target.id}'
    document = next(
        candidate
        for candidate in build_rag_index_documents(db)
        if candidate.document_id == document_id
    )
    return target, link, store, document


def _index_pg_test_document(
    db,
    *,
    document: VectorDocument,
    store: PgVectorStore,
    model_name: str,
    settings: Settings,
) -> None:
    indexed = index_changed_vector_documents(
        db=db,
        documents=[document],
        writer=store,
        embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
        embedding_model_name=model_name,
        settings=settings,
    )
    assert indexed.indexed_count == 1


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
                raw_metadata={'mime_type': 'message/rfc822'},
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
                parser_name='server_gmail_source_event',
                parser_status='parsed',
                parser_status_reason=None,
                mime_type='message/rfc822',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=signature,
                content_signature=signature,
                document_version_label='v1',
                parser_policy_version='server-source-parser-policy:v1',
                parser_version='source-event-paragraph-parser:v1',
                chunk_policy_version='paragraph-chunks:1200:v1',
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


@pytest.mark.parametrize('invalidity', ['non_string_mime', 'empty_revision'])
@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_live_filter_rejects_python_invalid_source_authority(
    invalidity: str,
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(
        database_url=database_url,
        openai_embedding_dimensions=8,
    )
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            history, _, source, store, document = _seed_pg_schedule(
                db,
                table_name=table_name,
                settings=settings,
            )
            indexed = index_changed_vector_documents(
                db=db,
                documents=[document],
                writer=store,
                embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
                embedding_model_name='deterministic-hash:authority-parity',
                settings=settings,
            )
            assert indexed.indexed_count == 1
            parser_run = (
                db.query(DocumentParserRun).filter_by(source_id=source.id).one()
            )
            evidence = (
                db.query(TrustedKnowledgeEvidenceLink)
                .join(
                    TrustedKnowledgeApprovalLink,
                    TrustedKnowledgeApprovalLink.id
                    == TrustedKnowledgeEvidenceLink.approval_link_id,
                )
                .filter(
                    TrustedKnowledgeApprovalLink.knowledge_id == history.id,
                    TrustedKnowledgeApprovalLink.knowledge_type == 'history_event',
                )
                .one()
            )
            if invalidity == 'non_string_mime':
                source.source_type = 'drive'
                source.raw_metadata = {'mime_type': ['text/plain']}
                parser_run.parser_name = 'server_drive_source_event'
                parser_run.mime_type = 'application/octet-stream'
                evidence.canonical_source_kind = 'drive'
            else:
                parser_run.revision_id = ''
                evidence.canonical_version_or_signature = ''
            db.commit()

            result = store.search_with_embedding(
                query_embedding=DeterministicHashEmbeddingModel(
                    dimensions=8
                ).embed('Exact current evidence'),
                user=USERS['viewer'],
                limit=5,
            )

            assert result.matches == []
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
def test_pgvector_live_filter_accepts_legacy_decision_link_under_canonical_id(
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    engine = create_engine(database_url)
    session_local = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    table_name = f'rag_vector_documents_test_{uuid4().hex[:8]}'
    settings = Settings(
        database_url=database_url,
        openai_embedding_dimensions=8,
    )
    Base.metadata.create_all(engine)
    try:
        with session_local() as db:
            history, item, _, store, _ = _seed_pg_schedule(
                db,
                table_name=table_name,
                settings=settings,
            )
            decision = DecisionRecord(
                project_key='project-a',
                title='Canonical decision',
                decision_summary='Legacy approval storage remains live',
                source_links=item.source_links,
                source_snippets=item.source_snippets,
                confidence_score=0.99,
                permission_level='internal',
                review_status='approved',
                source_review_item_id=item.id,
            )
            db.add(decision)
            db.flush()
            link = (
                db.query(TrustedKnowledgeApprovalLink)
                .filter_by(
                    knowledge_type='history_event',
                    knowledge_id=history.id,
                )
                .one()
            )
            link.knowledge_type = 'decision'
            link.knowledge_id = decision.id
            item.item_type = 'decision_record'
            db.commit()
            document_id = f'decision_record:{decision.id}'
            document = next(
                candidate
                for candidate in build_rag_index_documents(db)
                if candidate.document_id == document_id
            )
            indexed = index_changed_vector_documents(
                db=db,
                documents=[document],
                writer=store,
                embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
                embedding_model_name='deterministic-hash:legacy-decision',
                settings=settings,
            )
            assert indexed.indexed_count == 1

            result = store.search_with_embedding(
                query_embedding=DeterministicHashEmbeddingModel(
                    dimensions=8
                ).embed('Legacy approval storage remains live'),
                user=USERS['viewer'],
                limit=5,
            )

            assert [match.document.document_id for match in result.matches] == [
                document_id
            ]
            db.execute(text(f'DROP TABLE IF EXISTS {table_name}'))
            db.commit()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
@pytest.mark.parametrize('stored_knowledge_type', ['decision', 'decision_record'])
def test_pgvector_candidate_persists_exact_decision_link_dependency(
    stored_knowledge_type: str,
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    with _pgvector_test_db(database_url) as (db, table_name, settings):
        decision, _, _, link, _, store, document = _seed_pg_decision_schedule(
            db,
            table_name=table_name,
            settings=settings,
            knowledge_type=stored_knowledge_type,
            permission_level='internal',
        )
        _index_pg_test_document(
            db,
            document=document,
            store=store,
            model_name=f'deterministic-hash:{stored_knowledge_type}',
            settings=settings,
        )
        result = store.search_with_embedding(
            query_embedding=DeterministicHashEmbeddingModel(
                dimensions=8
            ).embed('Legacy decision mutation evidence'),
            user=USERS['viewer'],
            limit=5,
        )
        vector_candidates = candidates_from_vector_matches(result.matches)
        assert [row.source_id for row in vector_candidates] == [
            f'decision_record:{decision.id}'
        ]
        candidates = filter_live_serving_candidates(
            db=db,
            candidates=vector_candidates,
        )
        candidate = next(
            row
            for row in candidates
            if row.source_id == f'decision_record:{decision.id}'
        )
        dependency = build_serving_dependency_snapshot(db, candidate)

        assert dependency is not None
        assert dependency.serving_document_id == (
            f'decision_record:{decision.id}'
        )
        assert dependency.knowledge_type == stored_knowledge_type
        assert dependency.approval_link_id == link.id

        conversation = create_conversation(
            db, USERS['viewer'], title='PG exact dependency'
        )
        message = append_assistant_message(
            db,
            USERS['viewer'],
            conversation,
            content='Bound pgvector decision answer',
            citations=[{'source_id': candidate.source_id}],
            source_ids=[candidate.source_id],
            source_links=[candidate.source_url],
            source_snippets=[candidate.source_snippet],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={},
            serving_dependencies=(dependency,),
        )
        stored_dependency = db.query(
            AssistantMessageEvidenceDependency
        ).one()
        refs = db.query(AssistantMessageKnowledgeEvidenceRef).all()

        assert stored_dependency.serving_document_id == (
            f'decision_record:{decision.id}'
        )
        assert stored_dependency.knowledge_type == stored_knowledge_type
        assert stored_dependency.approval_link_id == link.id
        assert {ref.approval_link_id for ref in refs} == {link.id}
        assert assistant_message_evidence_is_live(
            db, user=USERS['viewer'], message=message
        )


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
@pytest.mark.parametrize(
    ('knowledge_type', 'expected_text'),
    [
        (
            'decision_record',
            'Canonical decision\nLegacy decision mutation evidence',
        ),
        ('history_event', 'Trusted history\nExact current evidence'),
        (
            'timeline_event',
            'Canonical timeline\nCanonical timeline serving evidence',
        ),
        ('todo', 'Canonical todo\nhigh\nCanonical todo serving evidence'),
    ],
)
def test_pgvector_candidate_snapshot_uses_canonical_text_for_every_knowledge_type(
    knowledge_type: str,
    expected_text: str,
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    with _pgvector_test_db(database_url) as (db, table_name, settings):
        target, link, store, document = _seed_pg_canonical_knowledge_schedule(
            db,
            table_name=table_name,
            settings=settings,
            knowledge_type=knowledge_type,
        )
        assert document.text == expected_text
        assert document.metadata['title'] == target.title
        assert document.source_snippet == 'Exact current evidence'
        _index_pg_test_document(
            db,
            document=document,
            store=store,
            model_name=f'deterministic-hash:canonical-{knowledge_type}',
            settings=settings,
        )
        result = store.search_with_embedding(
            query_embedding=DeterministicHashEmbeddingModel(
                dimensions=8
            ).embed(expected_text),
            user=USERS['viewer'],
            limit=5,
        )
        vector_candidates = candidates_from_vector_matches(result.matches)
        assert [row.source_id for row in vector_candidates] == [
            f'{knowledge_type}:{target.id}'
        ]
        assert vector_candidates[0].source_snippet == 'Exact current evidence'
        candidates = filter_live_serving_candidates(
            db=db,
            candidates=vector_candidates,
        )
        assert len(candidates) == 1
        dependency = build_serving_dependency_snapshot(db, candidates[0])

        assert dependency is not None
        assert dependency.serving_document_id == f'{knowledge_type}:{target.id}'
        assert dependency.knowledge_type == knowledge_type
        assert dependency.approval_link_id == link.id

        conversation = create_conversation(
            db, USERS['viewer'], title=f'PG {knowledge_type} dependency'
        )
        message = append_assistant_message(
            db,
            USERS['viewer'],
            conversation,
            content=f'Bound {knowledge_type} answer',
            citations=[{'source_id': candidates[0].source_id}],
            source_ids=[candidates[0].source_id],
            source_links=[candidates[0].source_url],
            source_snippets=[candidates[0].source_snippet],
            permission_level='internal',
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={},
            serving_dependencies=(dependency,),
        )
        assert assistant_message_evidence_is_live(
            db, user=USERS['viewer'], message=message
        )


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_legacy_decision_permission_recovery_narrows_canonical_fingerprint(
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    with _pgvector_test_db(database_url) as (db, table_name, settings):
        decision, item, source, link, fingerprint, store, document = (
            _seed_pg_decision_schedule(
                db,
                table_name=table_name,
                settings=settings,
                knowledge_type='decision',
                permission_level='public',
            )
        )
        _index_pg_test_document(
            db,
            document=document,
            store=store,
            model_name='deterministic-hash:legacy-permission',
            settings=settings,
        )
        source.permission_level = 'internal'
        db.commit()

        first = AutoReviewSourceReconciliationService(
            db,
            settings=settings,
            vector_writer=store,
        ).recover_stale_sources(limit=1)
        replay = AutoReviewSourceReconciliationService(
            db,
            settings=settings,
            vector_writer=store,
        ).recover_stale_sources(limit=1)

        assert first.narrowed_count == 1
        assert first.remaining_count == 0
        assert decision.permission_level == 'internal'
        assert link.permission_level == 'internal'
        assert item.permission_level == 'internal'
        assert fingerprint.permission_level == 'internal'
        assert db.scalar(
            text(
                f'SELECT permission_level FROM {table_name} '
                'WHERE document_id = :document_id'
            ),
            {'document_id': document.document_id},
        ) == 'internal'
        assert replay.reconciled_count == 0
        assert replay.remaining_count == 0


@pytest.mark.parametrize('knowledge_type', ['decision', 'decision_record'])
@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_decision_source_invalidation_revokes_canonical_artifacts(
    knowledge_type: str,
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    with _pgvector_test_db(database_url) as (db, table_name, settings):
        decision, item, source, link, fingerprint, store, document = (
            _seed_pg_decision_schedule(
                db,
                table_name=table_name,
                settings=settings,
                knowledge_type=knowledge_type,
                permission_level='internal',
            )
        )
        fingerprint_id = fingerprint.id
        _index_pg_test_document(
            db,
            document=document,
            store=store,
            model_name=f'deterministic-hash:revoke-{knowledge_type}',
            settings=settings,
        )
        source.permission_level = 'restricted'
        db.commit()

        first = AutoReviewSourceReconciliationService(
            db,
            settings=settings,
            vector_writer=store,
        ).recover_stale_sources(limit=1)
        replay = AutoReviewSourceReconciliationService(
            db,
            settings=settings,
            vector_writer=store,
        ).recover_stale_sources(limit=1)

        assert first.revoked_count == 1
        assert first.remaining_count == 0
        assert replay.reconciled_count == 0
        assert replay.remaining_count == 0
        assert item.status == 'revoked'
        assert link.active is False
        assert decision.review_status == 'revoked'
        assert db.scalar(text(f'SELECT count(*) FROM {table_name}')) == 0
        assert db.scalar(
            select(VectorServingTombstone.id).where(
                VectorServingTombstone.document_id == document.document_id
            )
        ) is not None
        assert db.get(TrustedKnowledgeFingerprint, fingerprint_id) is None
        assert db.scalar(
            select(VectorIndexState.id).where(
                VectorIndexState.document_id == document.document_id
            )
        ) is None
        if knowledge_type == 'decision':
            assert db.scalar(
                select(VectorServingTombstone.id).where(
                    VectorServingTombstone.document_id == f'decision:{decision.id}'
                )
            ) is None


@pytest.mark.skipif(
    not os.getenv('PARAWORKS_PGVECTOR_TEST_DATABASE_URL'),
    reason='set PARAWORKS_PGVECTOR_TEST_DATABASE_URL to run pgvector integration test',
)
def test_pgvector_legacy_decision_invalidation_preserves_canonical_provenance(
) -> None:
    database_url = os.environ['PARAWORKS_PGVECTOR_TEST_DATABASE_URL']
    with _pgvector_test_db(database_url) as (db, table_name, settings):
        decision, item, source, selected_link, fingerprint, store, document = (
            _seed_pg_decision_schedule(
                db,
                table_name=table_name,
                settings=settings,
                knowledge_type='decision',
                permission_level='internal',
            )
        )
        _, survivor_item, _, survivor_link = _seed_explicit_history(
            db,
            resolution_source='human',
            current_signature='b' * 64,
            evidence_signature='b' * 64,
        )
        survivor_item.item_type = 'decision_record'
        survivor_link.knowledge_type = 'decision_record'
        survivor_link.knowledge_id = decision.id
        db.commit()
        _index_pg_test_document(
            db,
            document=document,
            store=store,
            model_name='deterministic-hash:legacy-provenance',
            settings=settings,
        )
        source.permission_level = 'restricted'
        db.commit()

        first = AutoReviewSourceReconciliationService(
            db,
            settings=settings,
            vector_writer=store,
        ).recover_stale_sources(limit=1)
        replay = AutoReviewSourceReconciliationService(
            db,
            settings=settings,
            vector_writer=store,
        ).recover_stale_sources(limit=1)

        assert first.revoked_count == 1
        assert first.remaining_count == 0
        assert replay.reconciled_count == 0
        assert replay.remaining_count == 0
        assert item.status == 'revoked'
        assert selected_link.active is False
        assert survivor_link.active is True
        assert decision.review_status == 'approved'
        assert db.scalar(text(f'SELECT count(*) FROM {table_name}')) == 1
        assert db.query(VectorServingTombstone).count() == 0
        assert db.get(TrustedKnowledgeFingerprint, fingerprint.id) is not None


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
                raw_metadata={'mime_type': 'message/rfc822'},
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
                body='Restricted exact history',
            )
            db.add(version)
            db.flush([version])
            parser_run = DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='server_gmail_source_event',
                parser_status='parsed',
                parser_status_reason=None,
                mime_type='message/rfc822',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=source.server_content_signature,
                content_signature=source.server_content_signature,
                document_version_label='v1',
                parser_policy_version='server-source-parser-policy:v1',
                parser_version='source-event-paragraph-parser:v1',
                chunk_policy_version='paragraph-chunks:1200:v1',
            )
            db.add(parser_run)
            db.flush([parser_run])
            chunk = DocumentChunk(
                version_id=version.id,
                source_id=source.id,
                parser_run_id=parser_run.id,
                chunk_index=0,
                text=version.body,
                source_snippet=version.body,
                permission_level='public',
                metadata_={},
            )
            db.add(chunk)
            db.flush([chunk])
            document.current_document_version_id = version.id
            parser_run.chunk_count = 1
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
