from backend.app.agents.rag_orchestrator_agent.service import (
    RagEvidenceCandidate,
    vector_documents_from_candidates,
)
from backend.app.core.demo_auth import USERS
from backend.app.rag.vector_store import InMemoryVectorStore, VectorDocument


def test_vector_store_ranks_semantic_candidates_and_tracks_hidden_matches() -> None:
    store = InMemoryVectorStore()
    store.upsert_many(
        [
            VectorDocument(
                document_id='gmail:redis',
                text='Redis stores transient queue state and job progress.',
                source_url='https://gmail.mock/redis',
                source_snippet='Redis stores transient queue state.',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            ),
            VectorDocument(
                document_id='drive:pricing',
                text='Confidential pricing requires finance approval.',
                source_url='https://drive.mock/pricing',
                source_snippet='Confidential pricing requires approval.',
                permission_level='restricted',
                metadata={'source_type': 'drive'},
            ),
            VectorDocument(
                document_id='slack:lunch',
                text='Team lunch is scheduled for Friday.',
                source_url='https://slack.mock/lunch',
                source_snippet='Team lunch is scheduled.',
                permission_level='internal',
                metadata={'source_type': 'slack'},
            ),
        ]
    )

    result = store.search(query='Redis queue progress pricing', user=USERS['viewer'], limit=5)

    assert [match.document.document_id for match in result.matches] == ['gmail:redis']
    assert result.hidden_match_count == 1
    assert result.matches[0].score > 0


def test_vector_store_can_export_documents_for_future_external_index() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        VectorDocument(
            document_id='knowledge:decision:1',
            text='Approved decision: use Redis for queue state.',
            source_url='knowledge://decision_record:1',
            source_snippet='Use Redis for queue state.',
            permission_level='internal',
            metadata={'source_type': 'decision_record'},
        )
    )

    exported = store.export_documents()

    assert exported == [
        {
            'document_id': 'knowledge:decision:1',
            'text': 'Approved decision: use Redis for queue state.',
            'source_url': 'knowledge://decision_record:1',
            'source_snippet': 'Use Redis for queue state.',
            'permission_level': 'internal',
            'metadata': {'source_type': 'decision_record'},
        }
    ]


def test_rag_candidates_can_be_projected_to_vector_documents() -> None:
    candidates = [
        RagEvidenceCandidate(
            source_id='decision_record:1',
            source_url='knowledge://decision_record:1',
            text='Approved Redis queue decision',
            source_snippet='Redis queue decision',
            author=None,
            timestamp='2026-05-01T10:00:00+09:00',
            permission_level='internal',
            metadata={'source_type': 'decision_record'},
        )
    ]

    documents = vector_documents_from_candidates(candidates)

    assert documents == [
        VectorDocument(
            document_id='decision_record:1',
            text='Approved Redis queue decision',
            source_url='knowledge://decision_record:1',
            source_snippet='Redis queue decision',
            permission_level='internal',
            metadata={
                'source_type': 'decision_record',
                'author': None,
                'timestamp': '2026-05-01T10:00:00+09:00',
            },
        )
    ]
