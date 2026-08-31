import struct
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
    store = PgVectorStore(
        session=session,
        config=PgVectorConfig(embedding_dimensions=3),
    )

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
    assert params['embedding'] == (
        '[0.10000000149011612,0.20000000298023224,0.30000001192092896]'
    )
    assert params['metadata_json'] == '{"source_type": "decision_record"}'


def test_pgvector_embedding_literal_round_trips_exact_float32_exponents() -> None:
    session = RecordingSession()
    store = PgVectorStore(
        session=session,
        config=PgVectorConfig(embedding_dimensions=6),
    )
    coordinates = [
        1.0e-10,
        -1.0e-10,
        1.0e10,
        -0.0,
        1.17549435e-38,
        3.4028235e38,
    ]

    store.upsert_with_embedding(
        VectorDocument(
            document_id='chunk:float32-boundaries',
            text='Exact float32 pgvector payload.',
            source_url='https://example.test/float32-boundaries',
            source_snippet='Exact float32 pgvector payload.',
            permission_level='internal',
            metadata={},
        ),
        embedding=coordinates,
    )

    literal = session.calls[0][1]['embedding']
    parsed = [float(value) for value in literal[1:-1].split(',')]
    expected = [
        struct.unpack('>f', struct.pack('>f', value))[0]
        for value in coordinates
    ]
    assert [struct.pack('>f', value) for value in parsed] == [
        struct.pack('>f', value) for value in expected
    ]
    assert 'e-10' in literal
    assert '-1.000000013351432e-10' in literal


@pytest.mark.parametrize(
    'embedding',
    (
        [1.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, float('nan'), 2.0],
        [True, 1.0, 2.0],
    ),
)
def test_pgvector_direct_writer_rejects_non_cosine_indexable_vectors(
    embedding: list[object],
) -> None:
    session = RecordingSession()
    store = PgVectorStore(
        session=session,
        config=PgVectorConfig(embedding_dimensions=3),
    )

    with pytest.raises(ValueError, match='cosine-indexable float32'):
        store.upsert_with_embedding(
            VectorDocument(
                document_id='chunk:1',
                text='Exact D vector payload',
                source_url='https://example.test/source',
                source_snippet='Exact D vector payload',
                permission_level='internal',
                metadata={},
            ),
            embedding=embedding,  # type: ignore[arg-type]
        )

    assert session.calls == []


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
    assert 'document_id = candidate.document_id' in statement


def test_pgvector_d_raw_write_is_actorless_without_widening_legacy_search() -> None:
    store = PgVectorStore(session=RecordingSession())

    upsert = store._upsert_sql()
    search = store._search_sql()
    normalized_upsert = ' '.join(upsert.split())

    assert "metadata_json->>'index_policy_version'" in normalized_upsert
    assert "'rag-v2-serving-index:v1'" in normalized_upsert
    assert "metadata_json->>'serving_kind' = 'raw_chunk'" in normalized_upsert
    assert "metadata_json->>'support_mode' = 'source_observation'" in normalized_upsert
    assert (
        "sources.source_type IN ( 'gmail', 'gmail_attachment', 'drive', 'calendar' )"
        in normalized_upsert
    )
    assert "metadata_json->>'index_policy_version'" not in search


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
