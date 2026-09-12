from backend.app.agent_runtime import rag_application as search_application
from backend.app.core.demo_auth import DemoUser
from backend.app.models import DocumentChunk
from backend.app.rag.vector_store import VectorDocument, VectorMatch, VectorSearchResult
from backend.tests.test_rag_orchestrator_service import seed_chunk


def test_search_response_discloses_default_deterministic_retrieval_backend(
    client, db_session
) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'gmail-search-backend',
        'Redis job state is stored as exact current evidence.',
        'internal',
    )

    response = client.post('/api/v1/search', json={'query': 'Redis job state'})

    assert response.status_code == 200
    payload = response.json()
    assert payload['retrieval_backend'] == 'deterministic_lexical'
    assert payload['cost_policy'] == {
        'embedding_query_call': False,
        'paid_llm_call': False,
        'requires_pgvector_flag': True,
    }
    assert payload['results']


def test_search_uses_pgvector_adapter_when_available(
    client, db_session, monkeypatch
) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'vector:gmail-redis',
        'Redis vector result text',
        'internal',
    )
    chunk = db_session.query(DocumentChunk).one()

    class FakeVectorStore:
        def search(
            self, *, query: str, user: DemoUser, limit: int = 5
        ) -> VectorSearchResult:
            assert query == 'Redis vector query'
            assert user.id == 'employee-mina'
            assert limit == 5
            return VectorSearchResult(
                matches=[
                    VectorMatch(
                        document=VectorDocument(
                            document_id='vector:gmail-redis',
                            text='Redis vector result text',
                            source_url='https://gmail.mock/vector:gmail-redis',
                            source_snippet='Redis vector result text',
                            permission_level='internal',
                            metadata={
                                'source_type': 'gmail',
                                'chunk_id': chunk.id,
                                'author': 'owner@example.com',
                                'timestamp': '2026-05-02T09:00:00+09:00',
                                'matched_terms': ['redis'],
                            },
                        ),
                        score=0.88,
                    )
                ],
                hidden_match_count=1,
            )

    monkeypatch.setattr(
        search_application,
        '_legacy_pgvector_search_store',
        lambda *, db, settings: FakeVectorStore(),
    )

    response = client.post(
        '/api/v1/search',
        headers={'X-Demo-User': 'viewer'},
        json={'query': 'Redis vector query'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['retrieval_backend'] == 'pgvector'
    assert payload['hidden_match_count'] == 1
    assert payload['permission_notice'] == 'Some sources may be hidden by permissions.'
    assert payload['cost_policy'] == {
        'embedding_query_call': True,
        'paid_llm_call': False,
        'requires_pgvector_flag': True,
    }
    assert payload['results'][0]['source_id'] == 'vector:gmail-redis'
    assert payload['results'][0]['relevance_score'] == 0.88
    assert payload['results'][0]['matched_terms'] == ['redis']


def test_search_pgvector_candidate_without_live_identity_is_dropped(
    client, monkeypatch
) -> None:
    class FakeVectorStore:
        def search(self, **kwargs) -> VectorSearchResult:
            return VectorSearchResult(
                matches=[
                    VectorMatch(
                        document=VectorDocument(
                            document_id='chunk:999999',
                            text='Stale vector content',
                            source_url='https://stale.invalid/evidence',
                            source_snippet='Stale source snippet',
                            permission_level='public',
                            metadata={'chunk_id': 999999},
                        ),
                        score=0.99,
                    )
                ],
                hidden_match_count=0,
            )

    monkeypatch.setattr(
        search_application,
        '_legacy_pgvector_search_store',
        lambda *, db, settings: FakeVectorStore(),
    )

    response = client.post('/api/v1/search', json={'query': 'stale'})

    assert response.status_code == 200
    assert response.json()['results'] == []


def test_search_pgvector_candidate_with_stale_content_for_live_identity_is_dropped(
    client, db_session, monkeypatch
) -> None:
    seed_chunk(
        db_session,
        'gmail',
        'gmail-current-vector',
        'Current canonical vector evidence.',
        'internal',
    )
    chunk = db_session.query(DocumentChunk).one()

    class FakeVectorStore:
        def search(self, **kwargs) -> VectorSearchResult:
            return VectorSearchResult(
                matches=[
                    VectorMatch(
                        document=VectorDocument(
                            document_id='gmail-current-vector',
                            text='Stale vector bytes from an older source state.',
                            source_url='https://gmail.mock/gmail-current-vector',
                            source_snippet=chunk.source_snippet,
                            permission_level='internal',
                            metadata={'chunk_id': chunk.id},
                        ),
                        score=0.99,
                    )
                ],
                hidden_match_count=0,
            )

    monkeypatch.setattr(
        search_application,
        '_legacy_pgvector_search_store',
        lambda *, db, settings: FakeVectorStore(),
    )

    response = client.post('/api/v1/search', json={'query': 'vector'})

    assert response.status_code == 200
    assert response.json()['results'] == []
