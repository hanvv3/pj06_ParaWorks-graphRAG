import math

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.models import (
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    SyncJob,
    TimelineEvent,
    Todo,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.embeddings import (
    DeterministicHashEmbeddingModel,
    EmbeddingBatchResult,
)
from backend.app.rag.indexing import (
    EmbeddingBudgetExceededError,
    build_rag_index_documents,
    compute_vector_document_hash,
    index_changed_vector_documents,
    index_vector_documents,
)
from backend.app.rag.vector_store import VectorDocument


class RecordingVectorWriter:
    def __init__(self) -> None:
        self.upserts: list[tuple[VectorDocument, list[float]]] = []

    def upsert_with_embedding(self, document: VectorDocument, embedding: list[float]) -> None:
        self.upserts.append((document, embedding))


class RecordingBatchEmbeddingModel:
    dimensions = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError('index_changed_vector_documents should call embed_many')

    def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
        self.batches.append(texts)
        return EmbeddingBatchResult(
            embeddings=[[float(index), float(index + 1)] for index, _ in enumerate(texts)],
            prompt_tokens=10 * len(texts),
            total_tokens=10 * len(texts),
            request_count=1 if texts else 0,
        )


class FailingEmbeddingModel:
    dimensions = 2

    def embed(self, text: str) -> list[float]:
        raise AssertionError('embedding call should be blocked by the budget gate')

    def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
        raise AssertionError('embedding call should be blocked by the budget gate')


def seed_chunk(
    db: Session,
    text: str,
    source_id: str = 'gmail-index-source',
    *,
    approve_for_rag: bool = True,
) -> int:
    source = Source(
        source_type='gmail',
        source_id=source_id,
        source_url=f'https://gmail.mock/{source_id}',
        title='Index source',
        author='owner@example.com',
        permission_level='internal',
        raw_metadata={'ts': '2026-04-30T10:00:00+00:00'},
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()

    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        chunk_index=0,
        text=text,
        source_snippet=text[:240],
        permission_level='internal',
        metadata_={'source_url': source.source_url, 'source_type': source.source_type},
    )
    db.add(chunk)
    if approve_for_rag:
        db.add(
            ReviewItem(
                item_type='history_event',
                payload={
                    'title': 'Approved source chunk',
                    'summary': 'Approved source should be indexed.',
                    'source_ids': [source_id],
                },
                source_links=[source.source_url],
                source_snippets=[text[:240]],
                confidence_score=0.88,
                permission_level='internal',
                status='approved',
            )
        )
    db.commit()
    return chunk.id


def test_hash_embedding_is_stable_and_normalized() -> None:
    model = DeterministicHashEmbeddingModel(dimensions=8)

    first = model.embed('Redis queue Redis 한국어 업무')
    second = model.embed('Redis queue Redis 한국어 업무')

    assert first == second
    assert len(first) == 8
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_index_vector_documents_writes_embeddings() -> None:
    writer = RecordingVectorWriter()
    documents = [
        VectorDocument(
            document_id='chunk:1',
            text='Redis queue state',
            source_url='https://gmail.mock/chunk-1',
            source_snippet='Redis queue state',
            permission_level='internal',
            metadata={'source_type': 'gmail'},
        )
    ]

    result = index_vector_documents(
        documents=documents,
        writer=writer,
        embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
    )

    assert result.indexed_count == 1
    assert result.embedding_dimensions == 8
    assert result.document_ids == ['chunk:1']
    assert writer.upserts[0][0] == documents[0]
    assert len(writer.upserts[0][1]) == 8


def test_raw_chunks_exclude_auto_policy_approval(db_session: Session) -> None:
    chunk_id = seed_chunk(db_session, 'Auto-approved raw evidence must not serve.')
    item = db_session.query(ReviewItem).one()
    item.resolution_source = 'auto_policy'
    db_session.commit()

    documents = build_rag_index_documents(db_session)

    assert f'chunk:{chunk_id}' not in {document.document_id for document in documents}


def test_human_approved_deferred_slack_chunk_never_gains_trusted_serving_authority(
    db_session: Session,
) -> None:
    chunk_id = seed_chunk(
        db_session,
        'Deferred Slack evidence remains legacy-dedupe-only.',
        source_id='slack-deferred-source',
    )
    source = db_session.query(Source).one()
    source.source_type = 'slack'
    db_session.commit()

    documents = build_rag_index_documents(db_session)

    assert f'chunk:{chunk_id}' not in {
        document.document_id for document in documents
    }


def test_tombstoned_documents_are_skipped_before_embedding(db_session: Session) -> None:
    document = VectorDocument(
        document_id='history_event:41',
        text='Revoked knowledge must never reach the embedding provider.',
        source_url='knowledge://history_event:41',
        source_snippet='Revoked knowledge',
        permission_level='internal',
        metadata={'source_type': 'history_event'},
    )
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Revoked knowledge'},
        source_links=['knowledge://history_event:41'],
        source_snippets=['Revoked knowledge'],
        confidence_score=1.0,
        permission_level='internal',
        status='revoked',
    )
    db_session.add(item)
    db_session.flush()
    db_session.add(
        VectorServingTombstone(
            document_id=document.document_id,
            source_review_item_id=item.id,
            reason_code='business_withdrawal',
        )
    )
    db_session.commit()
    embedding_model = RecordingBatchEmbeddingModel()

    result = index_changed_vector_documents(
        db=db_session,
        documents=[document],
        writer=RecordingVectorWriter(),
        embedding_model=embedding_model,
        embedding_model_name='deterministic-hash:v1',
    )

    assert embedding_model.batches == []
    assert result.indexed_count == 0
    assert result.skipped_document_ids == ['history_event:41']


def test_vector_document_hash_changes_when_serving_content_changes() -> None:
    original = VectorDocument(
        document_id='chunk:1',
        text='Redis queue state',
        source_url='https://gmail.mock/chunk-1',
        source_snippet='Redis queue state',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )
    changed = VectorDocument(
        document_id='chunk:1',
        text='Redis queue state with new approval rule',
        source_url='https://gmail.mock/chunk-1',
        source_snippet='Redis queue state',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )

    assert compute_vector_document_hash(original) != compute_vector_document_hash(changed)


def test_incremental_indexing_skips_unchanged_documents_after_success(db_session: Session) -> None:
    document = VectorDocument(
        document_id='chunk:1',
        text='Redis queue state',
        source_url='https://gmail.mock/chunk-1',
        source_snippet='Redis queue state',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )
    first_writer = RecordingVectorWriter()

    first_result = index_changed_vector_documents(
        db=db_session,
        documents=[document],
        writer=first_writer,
        embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
        embedding_model_name='deterministic-hash:test',
    )
    second_writer = RecordingVectorWriter()
    second_result = index_changed_vector_documents(
        db=db_session,
        documents=[document],
        writer=second_writer,
        embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
        embedding_model_name='deterministic-hash:test',
    )

    state = db_session.query(VectorIndexState).one()
    assert first_result.indexed_count == 1
    assert second_result.indexed_count == 0
    assert second_result.skipped_count == 1
    assert second_result.saved_embedding_calls == 1
    assert second_writer.upserts == []
    assert state.status == 'indexed'
    assert state.content_hash == compute_vector_document_hash(document)


def test_incremental_indexing_reindexes_changed_documents(db_session: Session) -> None:
    original = VectorDocument(
        document_id='chunk:1',
        text='Redis queue state',
        source_url='https://gmail.mock/chunk-1',
        source_snippet='Redis queue state',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )
    changed = VectorDocument(
        document_id='chunk:1',
        text='Redis queue state with changed retention policy',
        source_url='https://gmail.mock/chunk-1',
        source_snippet='Redis queue state with changed retention policy',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )

    index_changed_vector_documents(
        db=db_session,
        documents=[original],
        writer=RecordingVectorWriter(),
        embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
        embedding_model_name='deterministic-hash:test',
    )
    second_writer = RecordingVectorWriter()
    result = index_changed_vector_documents(
        db=db_session,
        documents=[changed],
        writer=second_writer,
        embedding_model=DeterministicHashEmbeddingModel(dimensions=8),
        embedding_model_name='deterministic-hash:test',
    )

    state = db_session.query(VectorIndexState).one()
    assert result.indexed_count == 1
    assert result.skipped_count == 0
    assert len(second_writer.upserts) == 1
    assert state.content_hash == compute_vector_document_hash(changed)


def test_incremental_indexing_batches_only_changed_documents(db_session: Session) -> None:
    unchanged = VectorDocument(
        document_id='chunk:1',
        text='Already indexed document',
        source_url='https://gmail.mock/chunk-1',
        source_snippet='Already indexed document',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )
    changed = VectorDocument(
        document_id='chunk:2',
        text='New document needs embedding',
        source_url='https://gmail.mock/chunk-2',
        source_snippet='New document needs embedding',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )
    index_changed_vector_documents(
        db=db_session,
        documents=[unchanged],
        writer=RecordingVectorWriter(),
        embedding_model=DeterministicHashEmbeddingModel(dimensions=2),
        embedding_model_name='deterministic-hash:test',
    )
    embedding_model = RecordingBatchEmbeddingModel()
    writer = RecordingVectorWriter()

    result = index_changed_vector_documents(
        db=db_session,
        documents=[unchanged, changed],
        writer=writer,
        embedding_model=embedding_model,
        embedding_model_name='deterministic-hash:test',
    )

    assert embedding_model.batches == [['New document needs embedding']]
    assert result.indexed_count == 1
    assert result.skipped_count == 1
    assert result.saved_embedding_calls == 1
    assert result.embedding_request_count == 1
    assert result.embedding_prompt_tokens == 10
    assert [document.document_id for document, _ in writer.upserts] == ['chunk:2']


def test_incremental_indexing_blocks_paid_embedding_when_estimated_budget_is_exceeded(
    db_session: Session,
) -> None:
    document = VectorDocument(
        document_id='chunk:expensive',
        text='Budget pressure from repeated company history. ' * 2_000,
        source_url='https://gmail.mock/expensive',
        source_snippet='Budget pressure from repeated company history.',
        permission_level='internal',
        metadata={'source_type': 'gmail'},
    )
    writer = RecordingVectorWriter()

    try:
        index_changed_vector_documents(
            db=db_session,
            documents=[document],
            writer=writer,
            embedding_model=FailingEmbeddingModel(),
            embedding_model_name='text-embedding-3-small',
            embedding_cost_per_1m_tokens=0.02,
            max_embedding_cost_usd=0.000001,
        )
    except EmbeddingBudgetExceededError as exc:
        assert exc.decision['budget_status'] == 'over_budget'
        assert exc.decision['estimated_cost_usd'] > exc.decision['budget_limit_usd']
        assert exc.decision['changed_document_count'] == 1
    else:
        raise AssertionError('expected embedding budget gate to block the run')

    assert writer.upserts == []
    assert db_session.query(VectorIndexState).count() == 0


def test_build_rag_index_documents_includes_chunks_and_approved_knowledge(db_session: Session) -> None:
    chunk_id = seed_chunk(db_session, 'Redis queue state should be indexed for RAG.')
    approved_source = ReviewItem(
        item_type='history_event',
        payload={
            'title': 'Approved source chunk',
            'summary': 'Approved Gmail source should be indexed.',
            'source_ids': ['gmail-index-source'],
        },
        source_links=['https://gmail.mock/gmail-index-source'],
        source_snippets=['Redis queue state should be indexed for RAG.'],
        confidence_score=0.88,
        permission_level='internal',
        status='approved',
    )
    approved = DecisionRecord(
        title='Use pgvector for RAG',
        decision_summary='PostgreSQL pgvector stores durable company memory embeddings.',
        source_links=['https://knowledge.mock/pgvector'],
        source_snippets=['pgvector decision snippet'],
        confidence_score=0.9,
        permission_level='internal',
        review_status='approved',
    )
    pending = Todo(
        title='Draft pending item',
        priority='medium',
        priority_reason='Pending records must not enter the serving index yet.',
        source_links=['https://knowledge.mock/pending'],
        source_snippets=['pending snippet'],
        confidence_score=0.7,
        permission_level='internal',
        review_status='pending_review',
    )
    db_session.add_all([approved_source, approved, pending])
    db_session.commit()

    documents = build_rag_index_documents(db_session)

    assert [document.document_id for document in documents] == [
        f'chunk:{chunk_id}',
        f'decision_record:{approved.id}',
    ]
    assert documents[0].metadata['source_type'] == 'gmail'
    assert documents[1].source_url == 'https://knowledge.mock/pgvector'


def test_build_rag_index_documents_excludes_unapproved_source_chunks(
    db_session: Session,
) -> None:
    seed_chunk(
        db_session,
        'Unapproved source chunk should not be indexed.',
        'gmail-unapproved-source',
        approve_for_rag=False,
    )

    documents = build_rag_index_documents(db_session)

    assert all('gmail-unapproved-source' not in document.document_id for document in documents)


def test_build_rag_index_documents_ignores_malformed_approved_source_ids(
    db_session: Session,
) -> None:
    seed_chunk(
        db_session,
        'Malformed approved source id payload should not index this chunk.',
        'gmail-malformed-source',
        approve_for_rag=False,
    )
    db_session.add(
        ReviewItem(
            item_type='history_event',
            payload={
                'title': 'Malformed source ids',
                'summary': 'source_ids must be a list of strings.',
                'source_ids': 'gmail-malformed-source',
            },
            source_links=['https://gmail.mock/gmail-malformed-source'],
            source_snippets=['Malformed approved source id payload should not index this chunk.'],
            confidence_score=0.8,
            permission_level='internal',
            status='approved',
        )
    )
    db_session.commit()

    documents = build_rag_index_documents(db_session)

    assert all(document.metadata.get('source_id') != 'gmail-malformed-source' for document in documents)


def test_build_rag_index_documents_includes_approved_timeline_events(
    db_session: Session,
) -> None:
    timeline = TimelineEvent(
        title='K-Tech pilot kickoff completed',
        result_summary='The K-Tech pilot kickoff meeting was completed with confirmed next steps.',
        source_links=['https://calendar.mock/k-tech-kickoff'],
        source_snippets=['K-Tech pilot kickoff meeting completed.'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
    )
    db_session.add(timeline)
    db_session.commit()

    documents = build_rag_index_documents(db_session)

    assert [document.document_id for document in documents] == [f'timeline_event:{timeline.id}']
    assert documents[0].metadata['source_type'] == 'timeline_event'


def test_build_rag_index_documents_preserves_project_key_on_approved_knowledge(
    db_session: Session,
) -> None:
    history = HistoryEvent(
        project_key='project-alpha',
        title='Gmail follow-up moved to review',
        reason='The customer follow-up from Gmail was approved for Project Alpha.',
        source_links=['https://mail.google.com/mail/u/0/#inbox/message-1'],
        source_snippets=['Customer follow-up should be reviewed under Project Alpha.'],
        confidence_score=0.89,
        permission_level='internal',
        review_status='approved',
    )
    db_session.add(history)
    db_session.commit()

    documents = build_rag_index_documents(db_session)

    assert [document.document_id for document in documents] == [f'history_event:{history.id}']
    assert documents[0].metadata['project_key'] == 'project-alpha'


def test_build_rag_index_documents_includes_document_parser_metadata(db_session: Session) -> None:
    source = Source(
        source_type='drive',
        source_id='drive:file-1',
        source_url='https://drive.google.com/file/d/file-1/view',
        title='휴가 정책',
        author='owner@example.com',
        permission_level='restricted',
        raw_metadata={'sync_cursor': '2026-05-01T09:00:00Z'},
    )
    db_session.add(source)
    db_session.flush()
    document = Document(source_id=source.id, title=source.title, current_version='43')
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version='43', body='휴가 신청 승인자가 인사팀으로 변경되었습니다.')
    db_session.add(version)
    db_session.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        chunk_index=0,
        text='휴가 신청 승인자가 인사팀으로 변경되었습니다.',
        source_snippet='휴가 신청 승인자가 인사팀으로 변경되었습니다.',
        permission_level='restricted',
        metadata_={
            'parser_name': 'google_drive_text_export',
            'parser_status': 'parsed',
            'parser_status_reason': None,
            'mime_type': 'application/vnd.google-apps.document',
            'document_version': '43',
            'revision_id': 'rev-43',
            'content_signature': 'drive:file-1:43:rev-43',
            'content_hash': 'hash-43',
            'section_path': '휴가 정책',
            'page_number': None,
        },
    )
    db_session.add(chunk)
    db_session.add(
        ReviewItem(
            item_type='history_event',
            payload={
                'title': 'Approved parser metadata source',
                'summary': 'Approved Drive source should preserve parser metadata.',
                'source_ids': ['drive:file-1'],
            },
            source_links=['https://drive.google.com/file/d/file-1/view'],
            source_snippets=['휴가 신청 승인자가 인사팀으로 변경되었습니다.'],
            confidence_score=0.88,
            permission_level='restricted',
            status='approved',
        )
    )
    db_session.commit()

    vector_document = build_rag_index_documents(db_session)[0]

    assert vector_document.permission_level == 'restricted'
    assert vector_document.metadata['parser_name'] == 'google_drive_text_export'
    assert vector_document.metadata['parser_status'] == 'parsed'
    assert vector_document.metadata['parser_status_reason'] is None
    assert vector_document.metadata['mime_type'] == 'application/vnd.google-apps.document'
    assert vector_document.metadata['document_version'] == '43'
    assert vector_document.metadata['revision_id'] == 'rev-43'
    assert vector_document.metadata['content_signature'] == 'drive:file-1:43:rev-43'
    assert vector_document.metadata['content_hash'] == 'hash-43'
    assert vector_document.metadata['section_path'] == '휴가 정책'
    assert vector_document.metadata['page_number'] is None


def test_reindex_endpoint_returns_dry_run_index_summary(client: TestClient, db_session: Session) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-api-index')

    response = client.post('/api/v1/rag/reindex')

    assert response.status_code == 200
    body = response.json()
    assert body['dry_run'] is True
    assert body['indexed_count'] == 1
    assert body['skipped_count'] == 0
    assert body['saved_embedding_calls'] == 0
    assert body['embedding_request_count'] == 1
    assert body['embedding_prompt_tokens'] == 0
    assert body['embedding_total_tokens'] == 0
    assert body['embedding_dimensions'] == 16
    assert body['document_ids'] == ['chunk:1']
    assert body['skipped_document_ids'] == []
    assert body['incremental'] is True
    assert body['storage_backend'] == 'preview'
    assert body['embedding_budget']['embedding_model'] == 'text-embedding-3-small'
    assert body['embedding_budget']['changed_document_count'] == 1
    assert body['embedding_budget']['estimated_input_tokens'] > 0
    assert body['embedding_budget']['estimated_cost_usd'] > 0
    assert body['embedding_budget']['budget_limit_usd'] == 0.001
    assert body['embedding_budget']['budget_status'] == 'within_budget'


def test_reindex_endpoint_reports_parser_status_counts(client: TestClient, db_session: Session) -> None:
    parsed_source = Source(
        source_type='drive',
        source_id='drive:parsed',
        source_url='https://drive.mock/parsed',
        title='Parsed Drive doc',
        author='owner@example.com',
        permission_level='restricted',
        raw_metadata={},
    )
    metadata_source = Source(
        source_type='drive',
        source_id='drive:metadata-only',
        source_url='https://drive.mock/metadata-only',
        title='Metadata-only Drive doc',
        author='owner@example.com',
        permission_level='restricted',
        raw_metadata={},
    )
    db_session.add_all([parsed_source, metadata_source])
    db_session.flush()

    parsed_document = Document(source_id=parsed_source.id, title=parsed_source.title, current_version='42')
    metadata_document = Document(source_id=metadata_source.id, title=metadata_source.title, current_version='42')
    db_session.add_all([parsed_document, metadata_document])
    db_session.flush()

    parsed_version = DocumentVersion(document_id=parsed_document.id, version='42', body='본문 파싱 완료')
    metadata_version = DocumentVersion(document_id=metadata_document.id, version='42', body='metadata only')
    db_session.add_all([parsed_version, metadata_version])
    db_session.flush()

    db_session.add_all(
        [
            DocumentChunk(
                version_id=parsed_version.id,
                source_id=parsed_source.id,
                chunk_index=0,
                text='본문 파싱 완료',
                source_snippet='본문 파싱 완료',
                permission_level='restricted',
                metadata_={'parser_status': 'parsed'},
            ),
            DocumentChunk(
                version_id=metadata_version.id,
                source_id=metadata_source.id,
                chunk_index=0,
                text='Metadata-only Drive file changed.',
                source_snippet='Metadata-only Drive file changed.',
                permission_level='restricted',
                metadata_={'parser_status': 'metadata_only'},
            ),
        ]
    )
    db_session.add(
        ReviewItem(
            item_type='history_event',
            payload={
                'title': 'Approved parser status sources',
                'summary': 'Approved Drive sources should report parser status counts.',
                'source_ids': ['drive:parsed', 'drive:metadata-only'],
            },
            source_links=['https://drive.mock/parsed', 'https://drive.mock/metadata-only'],
            source_snippets=['본문 파싱 완료', 'Metadata-only Drive file changed.'],
            confidence_score=0.88,
            permission_level='restricted',
            status='approved',
        )
    )
    db_session.commit()

    response = client.post('/api/v1/rag/reindex')

    assert response.status_code == 200
    assert response.json()['parser_status_counts'] == {
        'metadata_only': 1,
        'parsed': 1,
    }


def test_reindex_endpoint_requires_admin_role(client: TestClient, db_session: Session) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-employee-index')

    response = client.post('/api/v1/rag/reindex', headers={'X-Demo-User': 'hanvv-employee'})

    assert response.status_code == 403
    assert response.json()['detail'] == 'Admin permission required.'


def test_reindex_dry_run_reports_over_budget_without_provider_call(
    client: TestClient,
    db_session: Session,
) -> None:
    seed_chunk(db_session, 'Budget pressure from repeated company history. ' * 2_000, 'gmail-budget-preview')

    def override_settings() -> Settings:
        return Settings(
            database_url='sqlite://',
            rag_embedding_max_estimated_cost_usd=0.000001,
        )

    client.app.dependency_overrides[get_settings] = override_settings

    response = client.post('/api/v1/rag/reindex')

    assert response.status_code == 200
    body = response.json()
    assert body['dry_run'] is True
    assert body['storage_backend'] == 'preview'
    assert body['embedding_budget']['budget_status'] == 'over_budget'
    assert body['embedding_budget']['action'] == 'block'
    assert body['embedding_budget']['estimated_cost_usd'] > body['embedding_budget']['budget_limit_usd']


def test_reindex_endpoint_reports_skipped_documents_from_index_state(
    client: TestClient,
    db_session: Session,
) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-api-index')
    document = build_rag_index_documents(db_session)[0]
    index_changed_vector_documents(
        db=db_session,
        documents=[document],
        writer=RecordingVectorWriter(),
        embedding_model=DeterministicHashEmbeddingModel(dimensions=16),
        embedding_model_name='deterministic-hash:v1',
    )

    response = client.post('/api/v1/rag/reindex')

    assert response.status_code == 200
    assert response.json()['indexed_count'] == 0
    assert response.json()['skipped_count'] == 1
    assert response.json()['saved_embedding_calls'] == 1


def test_reindex_endpoint_rejects_pgvector_write_without_postgres(client: TestClient) -> None:
    response = client.post('/api/v1/rag/reindex?dry_run=false')

    assert response.status_code == 400
    assert response.json()['detail'] == 'pgvector writes require a PostgreSQL database.'


def test_reindex_job_endpoint_records_indexing_job(
    client: TestClient,
    db_session: Session,
) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-job-index')

    response = client.post('/api/v1/rag/reindex/jobs')

    assert response.status_code == 200
    body = response.json()
    assert body['job_id'].startswith('rag-index-')
    assert body['status'] == 'complete'
    assert body['indexed_count'] == 1
    assert body['skipped_count'] == 0
    assert body['saved_embedding_calls'] == 0
    job = db_session.query(SyncJob).one()
    assert job.job_id == body['job_id']
    assert job.connector_type == 'rag-index'
    assert job.status == 'complete'
    assert job.progress_pct == 100
    assert job.message == 'indexed=1 skipped=0 saved_embedding_calls=0'


def test_reindex_job_endpoint_requires_admin_role(client: TestClient, db_session: Session) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-employee-job-index')

    response = client.post('/api/v1/rag/reindex/jobs', headers={'X-Demo-User': 'hanvv-employee'})

    assert response.status_code == 403
    assert response.json()['detail'] == 'Admin permission required.'


def test_reindex_job_detail_endpoint_returns_status(
    client: TestClient,
    db_session: Session,
) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-job-detail')
    created = client.post('/api/v1/rag/reindex/jobs').json()

    response = client.get(f"/api/v1/rag/reindex/jobs/{created['job_id']}")

    assert response.status_code == 200
    assert response.json()['job_id'] == created['job_id']
    assert response.json()['status'] == 'complete'
    assert response.json()['indexed_count'] == 1


def test_reindex_job_detail_endpoint_requires_admin_role(client: TestClient, db_session: Session) -> None:
    job = SyncJob(
        job_id='rag-index-admin-only',
        connector_type='rag-index',
        status='complete',
        message='indexed=0 skipped=0 saved_embedding_calls=0',
        progress_pct=100,
    )
    db_session.add(job)
    db_session.commit()

    response = client.get('/api/v1/rag/reindex/jobs/rag-index-admin-only', headers={'X-Demo-User': 'hanvv-employee'})

    assert response.status_code == 403
    assert response.json()['detail'] == 'Admin permission required.'


def test_reindex_job_detail_endpoint_returns_failure_reason(
    client: TestClient,
    db_session: Session,
) -> None:
    job = SyncJob(
        job_id='rag-index-failed-detail',
        connector_type='rag-index',
        status='failed',
        message='failed: pgvector writes require a PostgreSQL database.',
        progress_pct=100,
    )
    db_session.add(job)
    db_session.commit()

    response = client.get('/api/v1/rag/reindex/jobs/rag-index-failed-detail')

    assert response.status_code == 200
    assert response.json()['status'] == 'failed'
    assert response.json()['failure_reason'] == 'pgvector writes require a PostgreSQL database.'


def test_reindex_job_detail_endpoint_returns_404_for_missing_job(client: TestClient) -> None:
    response = client.get('/api/v1/rag/reindex/jobs/missing-job')

    assert response.status_code == 404


def test_reindex_job_endpoint_queues_without_eager_execution(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    from backend.app.api.v1 import rag as rag_api

    enqueued_jobs: list[tuple[str, bool]] = []

    def record_enqueue(*, job_id: str, dry_run: bool) -> None:
        enqueued_jobs.append((job_id, dry_run))

    def override_settings() -> Settings:
        return Settings(database_url='sqlite://', celery_task_always_eager=False)

    monkeypatch.setattr(rag_api, 'enqueue_rag_reindex_job', record_enqueue)
    client.app.dependency_overrides[get_settings] = override_settings

    response = client.post('/api/v1/rag/reindex/jobs')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'queued'
    assert body['dry_run'] is True
    assert enqueued_jobs == [(body['job_id'], True)]
    job = db_session.query(SyncJob).one()
    assert job.status == 'queued'
    assert job.message == 'queued'
    assert job.progress_pct == 0


def test_rag_indexing_summary_returns_latest_jobs_and_state_counts(
    client: TestClient,
    db_session: Session,
) -> None:
    seed_chunk(db_session, 'Slack and Gmail history should be embedded for search.', 'gmail-summary-index')
    db_session.add(
        VectorIndexState(
            document_id='chunk:existing',
            embedding_model='deterministic-hash:v1',
            embedding_dimensions=16,
            content_hash='abc123',
            status='indexed',
        )
    )
    db_session.commit()
    client.post('/api/v1/rag/reindex/jobs')

    response = client.get('/api/v1/rag/indexing/summary')

    assert response.status_code == 200
    body = response.json()
    assert body['state_counts'] == {'indexed': 1}
    assert body['latest_jobs'][0]['connector_type'] == 'rag-index'
    assert body['latest_jobs'][0]['status'] == 'complete'
    assert body['latest_jobs'][0]['indexed_count'] == 1
    assert body['latest_jobs'][0]['skipped_count'] == 0
    assert body['latest_jobs'][0]['saved_embedding_calls'] == 0
    assert body['cost_policy'] == {
        'embedding_model': 'text-embedding-3-small',
        'embedding_input_cost_per_1m_tokens': 0.02,
        'max_estimated_embedding_cost_usd': 0.001,
        'preflight_budget_gate': True,
        'incremental_hash_skip': True,
    }


def test_rag_indexing_summary_requires_admin_role(client: TestClient) -> None:
    response = client.get('/api/v1/rag/indexing/summary', headers={'X-Demo-User': 'hanvv-employee'})

    assert response.status_code == 403
    assert response.json()['detail'] == 'Admin permission required.'
