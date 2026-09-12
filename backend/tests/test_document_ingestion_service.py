from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime import rag_application as search_application
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.agents.rag_orchestrator_agent import service as rag_service
from backend.app.connectors.base import SourceEvent
from backend.app.core.config import Settings
from backend.app.documents.service import (
    parsed_document_from_source_event,
    persist_parsed_document,
)
from backend.app.ingestion import service as ingestion_service
from backend.app.ingestion.service import ingest_events, ingest_events_with_result
from backend.app.ingestion.source_content_signature import (
    canonical_source_content_signature,
    server_parser_policy_for_event,
)
from backend.app.models import (
    AutoReviewRuntimeKeyState,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    RagServingCorpusGeneration,
    ReviewItem,
    Source,
    VectorIndexState,
)
from backend.app.rag.indexing import build_rag_index_documents
from backend.app.rag.serving_generation import lock_rag_serving_generation
from backend.app.rag.vector_store import VectorMatch, VectorSearchResult


def drive_event(
    *,
    version: str = '42',
    revision_id: str = 'rev-42',
    chunk_max_chars: int | None = None,
    body: str = '휴가 신청은 HR 시스템에서 진행합니다.\n승인은 팀장이 검토합니다.',
    permission_level: str = 'restricted',
    connector_signature: str | None = None,
    source_url: str = 'https://drive.google.com/file/d/file-1/view',
    extra_metadata: dict | None = None,
) -> SourceEvent:
    raw_metadata = {
        'mime_type': 'application/vnd.google-apps.document',
        'document_version': version,
        'revision_id': revision_id,
        'content_signature': connector_signature or f'drive:file-1:{version}:{revision_id}',
        'parser_name': 'google_drive_text_export',
        'parser_status': 'parsed',
        'parser_status_reason': None,
        'source_snippet': body.replace('\n', ' ')[:240],
    }
    if chunk_max_chars is not None:
        raw_metadata['chunk_max_chars'] = chunk_max_chars
    raw_metadata.update(extra_metadata or {})
    return SourceEvent(
        source_type='drive',
        source_id='drive:file-1',
        source_url=source_url,
        title='휴가 정책',
        body=body,
        author='owner@example.com',
        participants=['owner@example.com'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level=permission_level,
        raw_metadata=raw_metadata,
        semantic_timestamp_raw='2026-05-01T09:00:00Z',
    )


class RecordingVectorWriter:
    def __init__(self) -> None:
        self.deletes: list[tuple[str, ...]] = []
        self.narrowings: list[tuple[tuple[str, ...], str]] = []

    def upsert_with_embedding(self, document, embedding) -> None:
        raise AssertionError('ingestion must not call an embedding writer')

    def delete_many(self, document_ids) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.deletes.append(normalized)
        return len(normalized)

    def narrow_permissions(self, document_ids, permission_level: str) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.narrowings.append((normalized, permission_level))
        return len(normalized)


def slack_event_for_id(source_id: str) -> SourceEvent:
    return SourceEvent(
        source_type='slack',
        source_id=source_id,
        source_url=f'https://slack.example.test/{source_id}',
        title='Slack evidence',
        body='Slack body must remain on the deferred legacy path.',
        author='slack-user',
        participants=['slack-user'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={'content_signature': 'slack-connector-signature'},
    )


def _rag_settings() -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret=(
            'task-5-document-authority-secret-at-least-32-bytes'
        ),
        agent_runtime_fingerprint_key_version='task5-document-authority-v1',
        openai_embedding_model='fake-embedding:v1',
        openai_embedding_dimensions=2,
    )


def _seed_document_authority_rows(db: Session, *, settings: Settings) -> None:
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db.add_all(
        [
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                fingerprint_key_material_verifier=verifier,
                generation=1,
                ready=True,
            ),
            RagServingCorpusGeneration(
                id=1,
                corpus_generation=0,
                vector_index_generation=0,
                embedding_model=settings.openai_embedding_model,
                embedding_dimensions=settings.openai_embedding_dimensions,
                index_policy_version='rag-v2-serving-index:v1',
                pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
                fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                fingerprint_key_material_verifier=verifier,
            ),
        ]
    )


def _canonical_persist_fixture(db: Session) -> tuple[Source, SourceEvent]:
    event = drive_event()
    source = Source(
        source_type=event.source_type,
        source_id=event.source_id,
        source_url=event.source_url,
        title=event.title,
        author=event.author,
        permission_level=event.permission_level,
        raw_metadata={},
    )
    db.add(source)
    db.flush()
    return source, event


def _assert_no_parsed_document_rows(db: Session) -> None:
    assert db.query(Document).count() == 0
    assert db.query(DocumentVersion).count() == 0
    assert db.query(DocumentParserRun).count() == 0
    assert db.query(DocumentChunk).count() == 0


def test_canonical_persist_rejects_missing_context_before_malformed_pair(
    db_session: Session,
) -> None:
    source, event = _canonical_persist_fixture(db_session)
    parsed = parsed_document_from_source_event(event)
    server_signature = canonical_source_content_signature(event)

    with pytest.raises(TypeError, match='locked RAG generation context'):
        persist_parsed_document(
            db_session,
            source=source,
            title=event.title,
            parsed=parsed,
            metadata={},
            server_signature=server_signature,
            parser_policy=None,
            rag_generation_context=None,
        )

    _assert_no_parsed_document_rows(db_session)


def test_canonical_persist_rejects_share_context_before_malformed_pair(
    db_session: Session,
) -> None:
    settings = _rag_settings()
    source, event = _canonical_persist_fixture(db_session)
    parser_policy = server_parser_policy_for_event(event)
    _seed_document_authority_rows(db_session, settings=settings)
    db_session.commit()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        share_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
            for_update=False,
        )
        acquire_projection(db_session, key_context)
        with pytest.raises(TypeError, match='mutation-authorized'):
            persist_parsed_document(
                db_session,
                source=source,
                title=event.title,
                parsed=parsed_document_from_source_event(event),
                metadata={},
                server_signature=None,
                parser_policy=parser_policy,
                rag_generation_context=share_context,
            )
        _assert_no_parsed_document_rows(db_session)
        assert list(db_session.new) == []
        assert list(db_session.dirty) == []
        db_session.rollback()


def test_canonical_persist_rejects_foreign_context_before_malformed_pair(
    db_session: Session,
) -> None:
    settings = _rag_settings()
    source, event = _canonical_persist_fixture(db_session)
    server_signature = canonical_source_content_signature(event)
    _seed_document_authority_rows(db_session, settings=settings)
    db_session.commit()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        foreign_session = sessionmaker(bind=db_session.get_bind())()
        try:
            with pytest.raises(TypeError, match='another session'):
                persist_parsed_document(
                    foreign_session,
                    source=source,
                    title=event.title,
                    parsed=parsed_document_from_source_event(event),
                    metadata={},
                    server_signature=server_signature,
                    parser_policy=None,
                    rag_generation_context=generation_context,
                )
            _assert_no_parsed_document_rows(foreign_session)
            assert list(foreign_session.new) == []
            assert list(foreign_session.dirty) == []
        finally:
            foreign_session.close()
        db_session.rollback()


@pytest.mark.parametrize('close_root', ['rollback', 'commit'])
def test_canonical_persist_rejects_closed_context_before_malformed_pair(
    db_session: Session,
    close_root: str,
) -> None:
    settings = _rag_settings()
    source, event = _canonical_persist_fixture(db_session)
    server_signature = canonical_source_content_signature(event)
    _seed_document_authority_rows(db_session, settings=settings)
    db_session.commit()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        getattr(db_session, close_root)()
        with pytest.raises(TypeError, match='transaction'):
            persist_parsed_document(
                db_session,
                source=source,
                title=event.title,
                parsed=parsed_document_from_source_event(event),
                metadata={},
                server_signature=server_signature,
                parser_policy=None,
                rag_generation_context=generation_context,
            )

    _assert_no_parsed_document_rows(db_session)

def test_ingest_drive_parsed_document_preserves_parser_metadata(db_session: Session) -> None:
    event = drive_event()

    ingest_events(db_session, [event])

    document = db_session.query(Document).one()
    version = db_session.query(DocumentVersion).one()
    chunk = db_session.query(DocumentChunk).one()
    source = db_session.query(Source).one()
    parser_run = db_session.query(DocumentParserRun).one()
    assert document.current_version == '42'
    assert document.current_document_version_id == version.id
    assert version.version == '42'
    assert version.body == event.body
    assert chunk.text == event.body
    assert chunk.source_snippet == '휴가 신청은 HR 시스템에서 진행합니다. 승인은 팀장이 검토합니다.'
    assert chunk.permission_level == 'restricted'
    assert chunk.metadata_['source_id'] == 'drive:file-1'
    assert chunk.metadata_['source_url'] == 'https://drive.google.com/file/d/file-1/view'
    assert chunk.metadata_['source_type'] == 'drive'
    assert chunk.metadata_['permission_level'] == 'restricted'
    assert chunk.metadata_['mime_type'] == 'application/vnd.google-apps.document'
    assert chunk.metadata_['document_version'] == '42'
    assert chunk.metadata_['revision_id'] == 'rev-42'
    assert chunk.metadata_['content_signature'] == source.server_content_signature
    assert chunk.metadata_['parser_name'] == 'server_drive_source_event'
    assert chunk.metadata_['parser_status'] == 'parsed'
    assert chunk.metadata_['parser_status_reason'] is None
    assert len(chunk.metadata_['content_hash']) == 64
    assert source.server_content_signature_schema == 'server-source-content:v1'
    assert len(source.server_content_signature or '') == 64
    assert source.connector_content_signature == 'drive:file-1:42:rev-42'
    assert source.raw_metadata.get('content_signature') is None
    assert parser_run.server_content_signature == source.server_content_signature
    assert parser_run.server_content_signature_schema == 'server-source-content:v1'
    assert parser_run.parser_policy_version == 'server-source-parser-policy:v1'
    assert parser_run.parser_version == 'source-event-paragraph-parser:v1'
    assert parser_run.chunk_policy_version == 'paragraph-chunks:1200:v1'
    assert chunk.parser_run_id == parser_run.id


def test_hostile_connector_parser_metadata_never_reaches_chunk_index_or_search(
    client,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = drive_event(
        permission_level='internal',
        extra_metadata={
            'parser_name': 'connector_owned_parser',
            'parser_status': 'unsupported',
            'parser_status_reason': 'trust connector status',
            'mime_type': ' TEXT/PLAIN ',
            'chunk_max_chars': 1,
            'chunk_policy_version': 'connector-chunks:v999',
            'parser_policy_version': 'connector-parser-policy:v999',
            'parser_version': 'connector-parser:v999',
            'source_snippet': 'FORGED CONNECTOR SNIPPET',
            'content_hash': 'connector-content-hash',
            'section_path': 'connector-section',
            'page_number': 999,
        },
    )

    ingest_events(db_session, [event])
    source = db_session.query(Source).one()
    chunk = db_session.query(DocumentChunk).one()
    parser_run = db_session.query(DocumentParserRun).one()
    db_session.add(
        ReviewItem(
            item_type='history_event',
            payload={
                'title': 'Approved hostile-source regression fixture',
                'summary': event.body,
                'source_ids': [source.source_id],
            },
            source_links=[source.source_url],
            source_snippets=[chunk.source_snippet],
            confidence_score=0.99,
            permission_level='internal',
            status='approved',
            resolution_source='human',
        )
    )
    db_session.commit()

    for key in (
        'parser_name',
        'parser_status',
        'parser_status_reason',
        'chunk_max_chars',
        'chunk_policy_version',
        'parser_policy_version',
        'parser_version',
        'source_snippet',
        'content_hash',
        'section_path',
        'page_number',
    ):
        assert key not in source.raw_metadata
    assert source.raw_metadata['mime_type'] == 'text/plain'
    assert chunk.source_snippet != 'FORGED CONNECTOR SNIPPET'
    assert chunk.metadata_['parser_name'] == 'server_drive_source_event'
    assert chunk.metadata_['parser_status'] == 'parsed'
    assert chunk.metadata_['parser_status_reason'] is None
    assert chunk.metadata_['mime_type'] == 'text/plain'
    assert chunk.metadata_['content_hash'] != 'connector-content-hash'
    assert chunk.metadata_.get('page_number') is None

    # Even a stale/legacy chunk metadata snapshot is not parser authority for
    # indexing; the relational current server parser run remains authoritative.
    chunk.metadata_.update(
        {
            'parser_name': 'connector_owned_parser',
            'parser_status': 'unsupported',
            'parser_status_reason': 'trust connector status',
            'mime_type': 'application/x-connector-owned',
        }
    )
    db_session.commit()

    vector_document = next(
        document
        for document in build_rag_index_documents(db_session)
        if document.document_id == f'chunk:{chunk.id}'
    )
    assert vector_document.metadata['parser_name'] == parser_run.parser_name
    assert vector_document.metadata['parser_status'] == parser_run.parser_status
    assert vector_document.metadata['parser_status_reason'] is None
    assert vector_document.metadata['mime_type'] == parser_run.mime_type
    assert 'connector_owned_parser' not in repr(vector_document)
    assert 'trust connector status' not in repr(vector_document)

    class SingleDocumentSearchStore:
        def search(self, **_kwargs) -> VectorSearchResult:
            return VectorSearchResult(
                matches=[VectorMatch(document=vector_document, score=1.0)],
                hidden_match_count=0,
            )

    monkeypatch.setattr(
        search_application,
        '_legacy_pgvector_search_store',
        lambda **_kwargs: SingleDocumentSearchStore(),
    )
    # This regression targets response projection only. The separate live-serving
    # boundary has its own schedule-heavy tests and is left real elsewhere.
    monkeypatch.setattr(
        rag_service,
        'filter_live_serving_candidates',
        lambda *, db, candidates: candidates,
    )

    response = client.post('/api/v1/search', json={'query': '휴가 신청'})

    assert response.status_code == 200
    [result] = response.json()['results']
    assert result['parser_status'] == 'parsed'
    assert result['parser_status_reason'] is None
    assert 'connector_owned_parser' not in repr(result)
    assert 'trust connector status' not in repr(result)


def test_ingest_drive_parsed_document_splits_long_body_into_stable_chunks(db_session: Session) -> None:
    body = '\n\n'.join(
        [
            'Hiring policy',
            'Alpha team hiring plan keeps contractor review evidence close to the decision record.',
            'Budget policy',
            'Beta team budget policy requires approval evidence before finance updates.',
            'Launch policy',
            'Gamma launch policy keeps customer escalation notes restricted.',
        ]
    )

    ingest_events(db_session, [drive_event(body=body, chunk_max_chars=120)])

    chunks = db_session.query(DocumentChunk).order_by(DocumentChunk.chunk_index).all()
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert [chunk.metadata_['section_path'] for chunk in chunks] == [
        'Hiring policy',
        'Budget policy',
        'Launch policy',
    ]
    assert len({chunk.metadata_['content_hash'] for chunk in chunks}) == 3
    assert all(chunk.permission_level == 'restricted' for chunk in chunks)


def test_ingest_skips_same_content_signature_for_existing_document(db_session: Session) -> None:
    event = drive_event()

    ingest_events(db_session, [event])
    ingest_events(db_session, [event])

    assert db_session.query(Document).count() == 1
    assert db_session.query(DocumentVersion).count() == 1
    assert db_session.query(DocumentChunk).count() == 1


def test_ingest_adds_new_version_when_content_signature_changes(db_session: Session) -> None:
    ingest_events(db_session, [drive_event()])
    ingest_events(
        db_session,
        [
            drive_event(
                version='43',
                revision_id='rev-43',
                body='휴가 신청은 HR 시스템에서 진행합니다.\n승인자는 팀장에서 인사팀으로 변경되었습니다.',
            )
        ],
    )

    document = db_session.query(Document).one()
    versions = db_session.query(DocumentVersion).order_by(DocumentVersion.version).all()
    chunks = db_session.query(DocumentChunk).order_by(DocumentChunk.chunk_index, DocumentChunk.id).all()
    assert document.current_version == '43'
    assert [version.version for version in versions] == ['42', '43']
    assert len(chunks) == 2
    assert len(chunks[0].metadata_['content_signature']) == 64
    assert len(chunks[1].metadata_['content_signature']) == 64
    assert chunks[0].metadata_['content_signature'] != chunks[1].metadata_['content_signature']
    assert chunks[0].metadata_['content_hash'] != chunks[1].metadata_['content_hash']


def test_server_content_signature_detects_changed_body_with_same_external_version_and_signature(
    db_session: Session,
) -> None:
    first = drive_event(connector_signature='connector-stable')
    ingest_events(db_session, [first])

    result = ingest_events_with_result(
        db_session,
        [
            drive_event(
                connector_signature='connector-stable',
                body='Changed body with the same external revision.',
            )
        ],
    )

    assert result.changed_source_states[0].content_changed is True
    assert db_session.query(DocumentVersion).count() == 2
    source = db_session.query(Source).one()
    assert source.connector_content_signature == 'connector-stable'


def test_server_and_connector_signatures_are_separate_and_only_server_signature_is_authoritative(
    db_session: Session,
) -> None:
    event = drive_event(connector_signature='connector-a')
    ingest_events(db_session, [event])
    source = db_session.query(Source).one()
    original_server_signature = source.server_content_signature

    result = ingest_events_with_result(
        db_session,
        [drive_event(connector_signature='connector-b')],
    )

    db_session.refresh(source)
    assert result.changed_source_states == []
    assert source.server_content_signature == original_server_signature
    assert source.connector_content_signature == 'connector-b'
    assert db_session.query(DocumentParserRun).count() == 1


def test_legacy_connector_only_signature_cannot_authorize_c5_skip_reuse_or_serving(
    db_session: Session,
) -> None:
    legacy_signature = 'legacy-connector-signature'
    db_session.add(
        Source(
            source_type='drive',
            source_id='drive:file-1',
            source_url='https://legacy.example.test/file-1',
            title='Legacy drive source',
            author='legacy@example.com',
            permission_level='restricted',
            raw_metadata={'content_signature': legacy_signature},
            connector_content_signature=legacy_signature,
        )
    )
    db_session.commit()

    result = ingest_events_with_result(
        db_session,
        [drive_event(connector_signature=legacy_signature)],
    )

    source = db_session.query(Source).one()
    document = db_session.query(Document).one()
    parser_run = db_session.query(DocumentParserRun).one()
    assert result.skipped_events == 0
    assert result.changed_source_states[0].content_changed is True
    assert source.server_content_signature_schema == 'server-source-content:v1'
    assert source.server_content_signature != legacy_signature
    assert document.current_document_version_id is not None
    assert parser_run.server_content_signature == source.server_content_signature


def test_slack_collision_cannot_mutate_a_signed_c5_source(
    db_session: Session,
) -> None:
    ingest_events(db_session, [drive_event()])
    source = db_session.query(Source).one()
    document = db_session.query(Document).one()
    original_signature = source.server_content_signature
    original_pointer = document.current_document_version_id
    original_title = source.title

    with pytest.raises(ValueError, match='source type conflicts with existing source'):
        ingest_events_with_result(
            db_session,
            [slack_event_for_id('drive:file-1')],
        )

    db_session.refresh(source)
    db_session.refresh(document)
    assert source.source_type == 'drive'
    assert source.title == original_title
    assert source.server_content_signature == original_signature
    assert document.current_document_version_id == original_pointer
    assert db_session.query(DocumentVersion).count() == 1


def test_c5_collision_cannot_convert_a_legacy_slack_source(
    db_session: Session,
) -> None:
    ingest_events(db_session, [slack_event_for_id('legacy-slack-id')])
    source = db_session.query(Source).one()

    with pytest.raises(ValueError, match='source type conflicts with existing source'):
        ingest_events_with_result(
            db_session,
            [
                SourceEvent(
                    **{
                        **drive_event().__dict__,
                        'source_id': 'legacy-slack-id',
                    }
                )
            ],
        )

    db_session.refresh(source)
    assert source.source_type == 'slack'
    assert source.server_content_signature is None
    assert db_session.query(DocumentVersion).count() == 1


def test_supported_google_source_id_must_match_source_type_prefix(
    db_session: Session,
) -> None:
    malformed = SourceEvent(
        **{
            **drive_event().__dict__,
            'source_id': 'drive-file-1',
        }
    )

    with pytest.raises(ValueError, match='source id does not match source type'):
        ingest_events_with_result(db_session, [malformed])

    assert db_session.query(Source).count() == 0


def test_server_parser_registry_ignores_connector_parser_chunk_and_snippet_authority(
    db_session: Session,
) -> None:
    event = drive_event(
        chunk_max_chars=10,
        extra_metadata={
            'parser_name': 'malicious_connector_parser',
            'parser_status': 'unsupported',
            'source_snippet': 'connector-chosen-snippet',
        },
    )

    ingest_events(db_session, [event])

    parser_run = db_session.query(DocumentParserRun).one()
    chunk = db_session.query(DocumentChunk).one()
    assert parser_run.parser_name == 'server_drive_source_event'
    assert parser_run.parser_status == 'parsed'
    assert chunk.text == event.body
    assert chunk.source_snippet != 'connector-chosen-snippet'


def test_parser_or_chunk_policy_upgrade_is_not_unchanged_and_reparses_without_extraction_or_validation_call(
    db_session: Session,
    monkeypatch,
) -> None:
    event = drive_event()
    ingest_events(db_session, [event])
    real_policy = ingestion_service.server_parser_policy_for_event
    monkeypatch.setattr(
        ingestion_service,
        'server_parser_policy_for_event',
        lambda candidate: replace(
            real_policy(candidate),
            chunk_policy_version='paragraph-chunks:1200:v2',
        ),
    )

    result = ingest_events_with_result(db_session, [event])

    assert result.changed_source_states[0].content_changed is False
    assert result.changed_source_states[0].permission_changed is False
    assert result.changed_source_states[0].parser_policy_changed is True
    assert db_session.query(DocumentVersion).count() == 2
    assert db_session.query(DocumentParserRun).count() == 2
    current = db_session.query(Document).one()
    newest = db_session.query(DocumentVersion).order_by(DocumentVersion.id.desc()).first()
    assert current.current_document_version_id == newest.id


@pytest.mark.parametrize(
    'corruption',
    [
        'unbound_chunk',
        'chunk_count_mismatch',
        'missing_ordinal',
        'duplicate_ordinal',
        'wrong_source',
        'missing_chunk',
    ],
)
def test_malformed_current_parser_chunk_state_forces_bounded_reparse(
    db_session: Session,
    corruption: str,
) -> None:
    event = drive_event()
    ingest_events(db_session, [event])
    parser_run = db_session.query(DocumentParserRun).one()
    chunk = db_session.query(DocumentChunk).one()
    if corruption == 'unbound_chunk':
        chunk.parser_run_id = None
    elif corruption == 'chunk_count_mismatch':
        parser_run.chunk_count = 2
    elif corruption == 'missing_ordinal':
        chunk.chunk_index = 1
    elif corruption == 'duplicate_ordinal':
        parser_run.chunk_count = 2
        db_session.add(
            DocumentChunk(
                version_id=chunk.version_id,
                source_id=chunk.source_id,
                parser_run_id=None,
                chunk_index=chunk.chunk_index,
                text='duplicate legacy chunk',
                source_snippet='duplicate legacy chunk',
                permission_level=chunk.permission_level,
                metadata_={},
            )
        )
    elif corruption == 'wrong_source':
        wrong_source = Source(
            source_type='drive',
            source_id='drive:wrong-file',
            source_url='https://drive.google.com/file/d/wrong-file/view',
            title='Wrong source',
            author='wrong@example.com',
            permission_level='restricted',
            raw_metadata={},
        )
        db_session.add(wrong_source)
        db_session.flush()
        chunk.parser_run_id = None
        chunk.source_id = wrong_source.id
    else:
        db_session.delete(chunk)
    db_session.commit()

    result = ingest_events_with_result(db_session, [event])

    assert result.skipped_events == 0
    assert result.changed_source_states[0].content_changed is False
    assert result.changed_source_states[0].parser_policy_changed is True
    assert db_session.query(DocumentVersion).count() == 2
    document = db_session.query(Document).one()
    current_chunks = (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.version_id == document.current_document_version_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    current_run = current_chunks[0].parser_run_id
    assert [current.chunk_index for current in current_chunks] == [0]
    assert all(current.parser_run_id == current_run for current in current_chunks)


def test_ingestion_control_uses_flags_even_when_primary_projection_says_unchanged(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = drive_event()
    ingest_events(db_session, [event])
    monkeypatch.setattr(
        ingestion_service,
        'classify_source_state_change',
        lambda **_kwargs: ingestion_service.SourceStateChangeClassification(
            content_changed=True,
            permission_changed=False,
            parser_policy_changed=False,
            primary_code='unchanged',
        ),
    )

    result = ingest_events_with_result(db_session, [event])

    assert result.skipped_events == 0
    assert result.changed_source_states[0].content_changed is True
    assert result.changed_source_states[0].primary_code == 'content_changed'
    assert db_session.query(DocumentVersion).count() == 2


def test_unchanged_event_advances_safe_cursor_url_and_connector_signature_without_reparse_or_reconciliation(
    db_session: Session,
) -> None:
    ingest_events_with_result(
        db_session,
        [
            drive_event(
                connector_signature='connector-a',
                extra_metadata={
                    'sync_cursor': 'cursor-a',
                    'sync_partition': 'drive',
                },
            )
        ],
        authenticated_source_metadata_by_id={
            'drive:file-1': {
                'account_id': 'server-owner-a',
                'required_scopes': ['drive.readonly'],
            }
        },
    )
    source = db_session.query(Source).one()
    source.raw_metadata['server_security_scope'] = 'scope-a'
    db_session.commit()
    server_signature = source.server_content_signature
    permission = source.permission_level

    result = ingest_events_with_result(
        db_session,
        [
            drive_event(
                connector_signature='connector-b',
                source_url='https://drive.google.com/file/d/file-1/new-view',
                extra_metadata={
                    'sync_cursor': 'cursor-b',
                    'sync_partition': 'drive',
                    'connector_revision': 'revision-b',
                    'connector_updated_at': '2026-05-02T00:00:00Z',
                    'account_id': 'attacker-owner',
                    'required_scopes': ['drive.full'],
                    'server_security_scope': 'scope-b',
                    'server_content_signature': 'attacker-signature',
                    'current_document_version_id': 999,
                },
            )
        ],
    )

    db_session.refresh(source)
    assert result.changed_source_states == []
    assert source.source_url.endswith('/new-view')
    assert source.connector_content_signature == 'connector-b'
    assert source.server_content_signature == server_signature
    assert source.permission_level == permission
    assert source.raw_metadata['sync_cursor'] == 'cursor-b'
    assert source.raw_metadata['connector_revision'] == 'revision-b'
    assert source.raw_metadata['account_id'] == 'server-owner-a'
    assert source.raw_metadata['required_scopes'] == ['drive.readonly']
    assert source.raw_metadata['server_security_scope'] == 'scope-a'
    assert db_session.query(DocumentVersion).count() == 1


def test_connector_metadata_cannot_overwrite_server_signature_permission_scope_or_pointer(
    db_session: Session,
) -> None:
    ingest_events_with_result(
        db_session,
        [
            drive_event(
                permission_level='internal',
            )
        ],
        authenticated_source_metadata_by_id={
            'drive:file-1': {
                'account_id': 'server-account',
                'required_scopes': ['drive.readonly'],
            }
        },
    )
    source = db_session.query(Source).one()
    document = db_session.query(Document).one()
    original_signature = source.server_content_signature
    original_pointer = document.current_document_version_id

    ingest_events_with_result(
        db_session,
        [
            drive_event(
                permission_level='internal',
                extra_metadata={
                    'account_id': 'connector-account',
                    'required_scopes': ['drive.full'],
                    'permission_level': 'public',
                    'server_content_signature': 'connector-forgery',
                    'current_document_version_id': 999,
                },
            )
        ],
    )

    db_session.refresh(source)
    db_session.refresh(document)
    assert source.server_content_signature == original_signature
    assert source.permission_level == 'internal'
    assert source.raw_metadata.get('permission_level') is None
    assert source.raw_metadata['account_id'] == 'server-account'
    assert source.raw_metadata['required_scopes'] == ['drive.readonly']
    assert document.current_document_version_id == original_pointer


@pytest.mark.parametrize('existing_empty_source', [False, True])
def test_connector_reserved_metadata_forgery_is_stripped_for_new_or_empty_source(
    db_session: Session,
    existing_empty_source: bool,
) -> None:
    if existing_empty_source:
        db_session.add(
            Source(
                source_type='drive',
                source_id='drive:file-1',
                source_url='https://legacy.example.test/file-1',
                title='Empty legacy source',
                author='legacy@example.com',
                permission_level='restricted',
                raw_metadata={},
            )
        )
        db_session.commit()
    event = drive_event(
        extra_metadata={
            'account_id': 'forged-account',
            'oauth_scope': 'drive.full',
            'oauth_scopes': ['drive.full'],
            'required_scopes': ['drive.full'],
            'security_scope_id': 'forged-security-scope',
            'server_security_scope': 'forged-server-scope',
        }
    )

    ingest_events_with_result(db_session, [event])

    source = db_session.query(Source).one()
    for key in (
        'account_id',
        'oauth_scope',
        'oauth_scopes',
        'required_scopes',
        'security_scope_id',
        'server_security_scope',
    ):
        assert key not in source.raw_metadata


def test_authenticated_server_metadata_is_the_only_first_write_ownership_path(
    db_session: Session,
) -> None:
    event = drive_event(
        extra_metadata={
            'account_id': 'forged-account',
            'required_scopes': ['drive.full'],
            'security_scope_id': 'forged-scope',
        }
    )

    ingest_events_with_result(
        db_session,
        [event],
        authenticated_source_metadata_by_id={
            'drive:file-1': {
                'account_id': 'trusted-account',
                'required_scopes': ['drive.readonly'],
                'security_scope_id': 'trusted-scope',
            }
        },
    )

    source = db_session.query(Source).one()
    assert source.raw_metadata['account_id'] == 'trusted-account'
    assert source.raw_metadata['required_scopes'] == ['drive.readonly']
    assert source.raw_metadata['security_scope_id'] == 'trusted-scope'


def test_permission_only_event_updates_source_and_all_current_chunk_permissions(
    db_session: Session,
) -> None:
    first = drive_event(permission_level='public')
    ingest_events(db_session, [first])
    writer = RecordingVectorWriter()

    result = ingest_events_with_result(
        db_session,
        [drive_event(permission_level='internal')],
        vector_writer=writer,
    )

    source = db_session.query(Source).one()
    assert result.changed_source_states[0].permission_changed is True
    assert result.changed_source_states[0].content_changed is False
    assert source.permission_level == 'internal'
    assert {chunk.permission_level for chunk in source.documents[0].versions[0].chunks} == {
        'internal'
    }
    assert writer.narrowings == [(('chunk:1',), 'internal')]
    assert db_session.query(DocumentVersion).count() == 1


def test_c5_permission_normalization_keeps_source_and_chunks_equal_across_resync(
    db_session: Session,
) -> None:
    ingest_events(
        db_session,
        [drive_event(permission_level=' INTERNAL ')],
    )

    source = db_session.query(Source).one()
    chunk = db_session.query(DocumentChunk).one()
    assert source.permission_level == 'internal'
    assert chunk.permission_level == 'internal'
    assert chunk.metadata_['permission_level'] == 'internal'

    result = ingest_events_with_result(
        db_session,
        [drive_event(permission_level='internal')],
    )

    assert result.skipped_events == 1
    assert result.changed_source_states == []
    assert db_session.query(DocumentVersion).count() == 1


def test_permission_only_event_narrows_all_historical_chunks_and_chunk_vectors(
    db_session: Session,
) -> None:
    ingest_events(db_session, [drive_event(permission_level='public')])
    ingest_events(
        db_session,
        [drive_event(body='Second body version.', permission_level='public')],
    )
    writer = RecordingVectorWriter()

    ingest_events_with_result(
        db_session,
        [drive_event(body='Second body version.', permission_level='restricted')],
        vector_writer=writer,
    )

    chunks = db_session.query(DocumentChunk).order_by(DocumentChunk.id).all()
    assert [chunk.permission_level for chunk in chunks] == ['restricted', 'restricted']
    assert writer.narrowings == [(('chunk:1', 'chunk:2'), 'restricted')]


def test_permission_to_unknown_deletes_raw_chunk_vectors_and_index_states_without_provider_call(
    db_session: Session,
) -> None:
    ingest_events(db_session, [drive_event(permission_level='internal')])
    db_session.add(
        VectorIndexState(
            document_id='chunk:1',
            embedding_model='fake-embedding',
            content_hash='a' * 64,
            embedding_dimensions=3,
            status='indexed',
        )
    )
    db_session.commit()
    writer = RecordingVectorWriter()

    ingest_events_with_result(
        db_session,
        [drive_event(permission_level='unknown')],
        vector_writer=writer,
    )

    assert writer.deletes == [('chunk:1',)]
    assert db_session.query(VectorIndexState).count() == 0
    assert db_session.query(DocumentVersion).count() == 1


def test_content_supersession_indexes_only_current_document_version_and_removes_old_chunk_vectors(
    db_session: Session,
) -> None:
    ingest_events(db_session, [drive_event(permission_level='internal')])
    db_session.add(
        VectorIndexState(
            document_id='chunk:1',
            embedding_model='fake-embedding',
            content_hash='a' * 64,
            embedding_dimensions=3,
            status='indexed',
        )
    )
    db_session.commit()
    writer = RecordingVectorWriter()

    ingest_events_with_result(
        db_session,
        [drive_event(body='Superseding current body.', permission_level='internal')],
        vector_writer=writer,
    )

    document = db_session.query(Document).one()
    newest = db_session.query(DocumentVersion).order_by(DocumentVersion.id.desc()).first()
    assert document.current_document_version_id == newest.id
    assert writer.deletes == [('chunk:1',)]
    assert db_session.query(VectorIndexState).count() == 0


def test_duplicate_v1_labels_with_different_signatures_never_admit_old_chunk_vector(
    db_session: Session,
) -> None:
    first = drive_event(version='v1', revision_id='v1', body='First v1 body.')
    second = drive_event(version='v1', revision_id='v1', body='Second v1 body.')
    ingest_events(db_session, [first])
    old_chunk = db_session.query(DocumentChunk).one()
    db_session.add(
        VectorIndexState(
            document_id=f'chunk:{old_chunk.id}',
            embedding_model='fake-embedding',
            content_hash='a' * 64,
            embedding_dimensions=3,
            status='indexed',
        )
    )
    db_session.commit()
    writer = RecordingVectorWriter()

    ingest_events_with_result(db_session, [second], vector_writer=writer)

    document = db_session.query(Document).one()
    versions = db_session.query(DocumentVersion).order_by(DocumentVersion.id).all()
    current_chunk = (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.version_id == document.current_document_version_id)
        .one()
    )
    assert [version.version for version in versions] == ['v1', 'v1']
    assert current_chunk.text == 'Second v1 body.'
    assert current_chunk.id != old_chunk.id
    assert writer.deletes == [(f'chunk:{old_chunk.id}',)]
    assert db_session.query(VectorIndexState).count() == 0
