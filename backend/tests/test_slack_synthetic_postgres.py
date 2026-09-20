"""Actual disposable PG, shared Slack sync/Review/pgvector; no external models."""

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import AutoReviewKeyBootstrapService
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.db.base import Base
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import Source
from backend.app.rag.embeddings import DeterministicHashEmbeddingModel
from backend.app.rag.indexing import (
    build_rag_index_documents,
    index_changed_vector_documents,
)
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.tests.postgres_isolation import lease_postgres_schema
from backend.tests.slack_synthetic_fixture import SyntheticSlackClient
from backend.tests.test_slack_synthetic_authority import (
    RecordingSlackModel,
    approve,
    draft,
    local_connector,
)


@pytest.fixture
def pg_slack():
    url = os.environ.get('PARAWORKS_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('explicit disposable PostgreSQL locator required')
    with lease_postgres_schema(
        url, run_id=uuid4().hex[:12], scope_name='slack_s2'
    ) as lease:
        engine = create_engine(lease.database_url)
        try:
            Base.metadata.create_all(engine)
            settings = Settings(
                _env_file=None,
                database_url=lease.database_url,
                agent_runtime_fingerprint_secret=uuid4().hex + uuid4().hex,
                openai_embedding_dimensions=8,
            )
            AutoReviewKeyBootstrapService(
                session_factory=sessionmaker(bind=engine), settings=settings
            ).ensure_initialized()
            with Session(engine) as db:
                writer = PgVectorStore(
                    session=db,
                    config=PgVectorConfig(embedding_dimensions=8),
                    settings=settings,
                )
                writer.ensure_schema()
                db.commit()
                yield db, settings, writer
        finally:
            engine.dispose()


def test_real_pg_shared_sync_known_thread_review_index_replay_and_edit(pg_slack):
    db, settings, writer = pg_slack
    fake = SyntheticSlackClient()
    first = sync_connector_events(
        db, local_connector(fake), settings=settings, vector_writer=writer
    )
    fake.add_late_reply()
    second = sync_connector_events(
        db, local_connector(fake), settings=settings, vector_writer=writer
    )
    assert (first.fetched_events, second.fetched_events) == (3, 1)
    assert second.changed_source_ids == [f'CPUBLIC:{fake.reply_ts}']
    replay = sync_connector_events(
        db, local_connector(fake), settings=settings, vector_writer=writer
    )
    assert replay.skipped_events == 1
    model = RecordingSlackModel()
    item = draft(db, model, source_ids=[f'CPUBLIC:{fake.reply_ts}'], settings=settings)[
        0
    ]
    assert (
        draft(db, model, source_ids=[f'CPUBLIC:{fake.reply_ts}'], settings=settings)
        == []
    )
    assert len(model.packets) == 1
    approve(db, item, settings=settings)
    assert approve(db, item, settings=settings).replayed
    embedding = DeterministicHashEmbeddingModel(dimensions=8)
    docs = build_rag_index_documents(db)
    indexed = index_changed_vector_documents(
        db=db,
        documents=docs,
        writer=writer,
        embedding_model=embedding,
        embedding_model_name='local-synthetic:8',
        settings=settings,
    )
    again = index_changed_vector_documents(
        db=db,
        documents=docs,
        writer=writer,
        embedding_model=embedding,
        embedding_model_name='local-synthetic:8',
        settings=settings,
    )
    assert indexed.indexed_count > 0
    assert (
        again.indexed_count == 0
        and again.saved_embedding_calls == indexed.indexed_count
    )
    assert writer.search_with_embedding(
        query_embedding=embedding.embed('pgvector'), user=USERS['admin']
    ).matches
    fake.histories = {
        'CPUBLIC': [fake.message(fake.parent_ts, '결정: 현재 부모 수정', 'UPARENT')]
    }
    fake.replies = {}
    sync_connector_events(
        db, local_connector(fake), settings=settings, vector_writer=writer
    )
    assert not any(
        d.document_id.startswith('history_event:')
        for d in build_rag_index_documents(db)
    )
    matches = writer.search_with_embedding(
        query_embedding=embedding.embed('pgvector'), user=USERS['admin']
    ).matches
    assert not any(m.document.document_id.startswith('history_event:') for m in matches)
    assert db.scalar(
        select(Source).where(Source.source_id == f'CPUBLIC:{fake.parent_ts}')
    ).server_content_signature


@pytest.mark.parametrize('mutation', ['delete', 'restrict'])
def test_real_pg_parent_lifecycle_filters_existing_vectors_and_v2_serving(
    pg_slack, mutation
):
    from backend.app.connectors.slack_synthetic import LocalSyntheticSlackConnector
    from backend.app.rag.indexing import build_rag_v2_index_documents

    db, settings, writer = pg_slack
    fake = SyntheticSlackClient()
    fake.histories['CPRIVATE'] = []
    sync_connector_events(
        db, local_connector(fake), settings=settings, vector_writer=writer
    )
    fake.add_late_reply()
    sync_connector_events(
        db, local_connector(fake), settings=settings, vector_writer=writer
    )
    item = draft(
        db,
        RecordingSlackModel(),
        source_ids=[f'CPUBLIC:{fake.reply_ts}'],
        settings=settings,
    )[0]
    approve(db, item, settings=settings)
    assert any(
        d.document_id.startswith('history_event:')
        for d in build_rag_v2_index_documents(db, settings=settings)
    )
    embedding = DeterministicHashEmbeddingModel(dimensions=8)
    index_changed_vector_documents(
        db=db,
        documents=build_rag_index_documents(db),
        writer=writer,
        embedding_model=embedding,
        embedding_model_name='local-synthetic:8',
        settings=settings,
    )
    channels = fake.conversations_list()
    deleted = [('CPUBLIC', fake.parent_ts)] if mutation == 'delete' else []
    if mutation == 'restrict':
        channels[0]['is_private'] = True
    connector = LocalSyntheticSlackConnector(
        channels=channels,
        messages_by_channel={}
        if deleted
        else {
            'CPUBLIC': [
                fake.message(fake.parent_ts, '결정: pgvector를 사용합니다.', 'UPARENT')
            ]
        },
        deleted_messages=deleted,
        users=fake.users_list(),
    )
    sync_connector_events(db, connector, settings=settings, vector_writer=writer)
    v2 = [
        d
        for d in build_rag_v2_index_documents(db, settings=settings)
        if d.document_id.startswith('history_event:')
    ]
    if mutation == 'delete':
        assert v2 == []
    else:
        assert all(d.permission_level == 'restricted' for d in v2)
    result = writer.search_with_embedding(
        query_embedding=embedding.embed('pgvector'), user=USERS['viewer']
    )
    assert not any(
        m.document.document_id.startswith('history_event:') for m in result.matches
    )
