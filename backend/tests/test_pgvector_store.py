from pathlib import Path

import pytest

from backend.app.core.demo_auth import USERS
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.vector_store import VectorDocument


def test_docker_postgres_init_creates_pgvector_rag_table() -> None:
    sql = Path('docker/postgres/init/002_rag_vector_documents.sql').read_text(encoding='utf-8')

    assert 'CREATE TABLE IF NOT EXISTS rag_vector_documents' in sql
    assert 'embedding vector(1536)' in sql
    assert 'USING ivfflat (embedding vector_cosine_ops)' in sql
    assert 'rag_vector_documents_permission_idx' in sql


class RecordingSession:
    def __init__(self, rows=None) -> None:
        self.calls = []
        self._rows = rows or []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))
        return RecordingResult(self._rows)


class RecordingResult:
    def __init__(self, rows) -> None:
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows

    @property
    def rowcount(self):
        return len(self._rows)


def test_pgvector_schema_sql_creates_extension_table_and_indexes() -> None:
    store = PgVectorStore(session=RecordingSession(), config=PgVectorConfig(embedding_dimensions=1536))

    schema_sql = '\n'.join(store.schema_sql())

    assert 'CREATE EXTENSION IF NOT EXISTS vector' in schema_sql
    assert 'embedding vector(1536)' in schema_sql
    assert 'rag_vector_documents_embedding_idx' in schema_sql
    assert 'vector_cosine_ops' in schema_sql
    assert 'rag_vector_documents_permission_idx' in schema_sql


def test_pgvector_upsert_writes_document_with_embedding_literal() -> None:
    session = RecordingSession()
    store = PgVectorStore(session=session)

    store.upsert_with_embedding(
        VectorDocument(
            document_id='knowledge:decision:1',
            text='Use Redis for queue state.',
            source_url='knowledge://decision_record:1',
            source_snippet='Use Redis for queue state.',
            permission_level='internal',
            metadata={'source_type': 'decision_record'},
        ),
        embedding=[0.1, 0.2, 0.3],
    )

    statement, params = session.calls[0]
    assert 'INSERT INTO rag_vector_documents' in statement
    assert 'ON CONFLICT (document_id) DO UPDATE' in statement
    assert params['document_id'] == 'knowledge:decision:1'
    assert params['embedding'] == '[0.1,0.2,0.3]'
    assert params['metadata_json'] == '{"source_type": "decision_record"}'


def test_pgvector_search_filters_by_user_permission_and_tracks_hidden_matches() -> None:
    rows = [
        {
            'document_id': 'gmail:redis',
            'text': 'Redis queue state',
            'source_url': 'https://gmail.mock/redis',
            'source_snippet': 'Redis queue state',
            'permission_level': 'internal',
            'metadata_json': {'source_type': 'gmail'},
            'score': 0.92,
            'hidden_match_count': 1,
        }
    ]
    session = RecordingSession(rows=rows)
    store = PgVectorStore(session=session)

    result = store.search_with_embedding(query_embedding=[0.3, 0.2, 0.1], user=USERS['viewer'], limit=5)

    statement, params = session.calls[0]
    assert 'permission_level = ANY(:allowed_permissions)' in statement
    assert 'embedding <=> CAST(:query_embedding AS vector)' in statement
    assert params['allowed_permissions'] == ['public', 'internal']
    assert params['query_embedding'] == '[0.3,0.2,0.1]'
    assert result.hidden_match_count == 1
    assert [match.document.document_id for match in result.matches] == ['gmail:redis']
    assert result.matches[0].score == 0.92


def test_pgvector_delete_many_uses_exact_document_ids() -> None:
    session = RecordingSession(rows=[{'deleted': True}, {'deleted': True}])
    store = PgVectorStore(session=session)

    deleted = store.delete_many(['todo:2', 'todo:1', 'todo:2'])

    statement, params = session.calls[0]
    assert 'DELETE FROM rag_vector_documents' in statement
    assert 'document_id = ANY(:document_ids)' in statement
    assert params == {'document_ids': ['todo:1', 'todo:2']}
    assert deleted == 2


def test_pgvector_narrow_permissions_rejects_broadening() -> None:
    session = RecordingSession(
        rows=[{'document_id': 'todo:1', 'permission_level': 'restricted'}]
    )
    store = PgVectorStore(session=session)

    with pytest.raises(ValueError, match='broadening'):
        store.narrow_permissions(['todo:1'], 'internal')

    assert len(session.calls) == 1

    session = RecordingSession(
        rows=[
            {'document_id': 'todo:1', 'permission_level': 'public'},
            {'document_id': 'todo:2', 'permission_level': 'internal'},
        ]
    )
    store = PgVectorStore(session=session)

    narrowed = store.narrow_permissions(['todo:2', 'todo:1'], 'restricted')

    statement, params = session.calls[1]
    assert 'UPDATE rag_vector_documents' in statement
    assert "WHEN 'public' THEN 0" in statement
    assert "WHEN 'internal' THEN 1" in statement
    assert params == {
        'document_ids': ['todo:1', 'todo:2'],
        'permission_level': 'restricted',
        'permission_rank': 2,
    }
    assert narrowed == 2


def test_pgvector_conditional_upsert_cannot_cross_a_tombstone() -> None:
    store = PgVectorStore(session=RecordingSession())

    statement = store._upsert_sql()

    assert 'INSERT INTO rag_vector_documents' in statement
    assert 'SELECT' in statement
    assert 'NOT EXISTS' in statement
    assert 'vector_serving_tombstones' in statement
    assert 'document_id = :document_id' in statement


def test_pgvector_search_excludes_stale_tombstoned_row_before_hidden_count() -> None:
    store = PgVectorStore(session=RecordingSession())

    statement = store._search_sql()

    ranked = statement.split('hidden AS', maxsplit=1)[0]
    assert 'vector_serving_tombstones' in ranked
    assert 'NOT EXISTS' in ranked


def test_pgvector_search_excludes_critical_audit_correction_before_ranking() -> None:
    statement = PgVectorStore(session=RecordingSession())._search_sql()

    ranked = statement.split('hidden AS', maxsplit=1)[0]
    assert 'auto_review_audit_corrections' in ranked
    assert 'effective_outcome' in ranked
