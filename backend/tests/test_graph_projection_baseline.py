"""Deterministic cache-off actual pgvector baseline; no provider calls."""

import json
import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import security_scope_fingerprint
from backend.app.db.base import Base
from backend.app.rag.embeddings import validate_query_embedding_vector
from backend.app.rag.indexing import build_rag_v2_index_documents
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.retrieval import RetrievalRequest
from backend.app.rag.search_store import SqlAlchemyPgVectorSearchStore
from backend.tests.graph_projection_fixtures import (
    CASES,
    SCOPE,
    SETTINGS,
    seed_corpus,
    vector,
)
from backend.tests.postgres_isolation import lease_postgres_schema


@pytest.mark.parametrize('case', CASES)
def test_cache_off_pgvector_fixed_corpus(case):
    url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('disposable PostgreSQL required')
    with lease_postgres_schema(
        url, run_id=uuid4().hex[:12], scope_name='e_baseline'
    ) as lease:
        engine = create_engine(lease.database_url)
        try:
            Base.metadata.create_all(engine, checkfirst=False)
            with Session(engine) as db:
                seed_corpus(db, case)
                store = PgVectorStore(
                    session=db, config=PgVectorConfig(embedding_dimensions=1536)
                )
                store.ensure_schema()
                for document in build_rag_v2_index_documents(db, settings=SETTINGS):
                    axis = {
                        'history_event:1': 0,
                        'chunk:1': 0,
                        'chunk:2': 1,
                        'chunk:3': 2,
                    }[document.document_id]
                    db.execute(
                        text(
                            'INSERT INTO rag_vector_documents (document_id, text, source_url, source_snippet, permission_level, metadata_json, embedding) VALUES (:id, :body, :url, :snippet, :permission, CAST(:metadata AS jsonb), CAST(:embedding AS vector))'
                        ),
                        {
                            'id': document.document_id,
                            'body': document.text,
                            'url': document.source_url,
                            'snippet': document.source_snippet,
                            'permission': document.permission_level,
                            'metadata': json.dumps(document.metadata),
                            'embedding': json.dumps(vector(axis)),
                        },
                    )
                db.commit()
                request = RetrievalRequest(
                    retrieval_query_text=CASES[case]['query'],
                    security_scope=SCOPE,
                    security_scope_fingerprint=security_scope_fingerprint(
                        SCOPE, settings=SETTINGS
                    ),
                    query_embedding_result=None,
                    candidate_scan_limit=50,
                    visible_limit=5,
                    relevance_policy_version='rag-retrieval-policy:v2.0',
                )
                query = vector(2 if case == 'single' else 3 if case == 'absent' else 0)
                validated = validate_query_embedding_vector(
                    query, expected_dimensions=1536
                )
                reader = PgVectorStore(
                    session=db, config=store.config, settings=SETTINGS
                )
                rows = SqlAlchemyPgVectorSearchStore(
                    db=db, store=reader, settings=SETTINGS
                ).search(request, validated)
                visible = tuple(
                    row.evidence.serving_document_id
                    for row in rows
                    if row.access.permission_visibility == 'visible'
                )[:5]
                assert visible == CASES[case]['baseline_ids']
                print(
                    f'E baseline {case}: {visible}; cache=off; candidate=50; visible=5; embedding_calls=0'
                )
        finally:
            engine.dispose()
