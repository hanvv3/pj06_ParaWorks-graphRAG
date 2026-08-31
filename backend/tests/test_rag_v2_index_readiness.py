from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.models import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    RagServingCorpusGeneration,
    ReviewItem,
    Source,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.index_readiness import (
    RagServingIndexReadiness,
    RagV2ServingIndexReadinessService,
)
from backend.app.rag.indexing import (
    build_rag_v2_index_documents,
    compute_rag_v2_document_hash,
    upsert_rag_v2_vector_index_state,
)
from backend.app.rag.vector_validation import CosineIndexableVectorValidator


def _settings() -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret=(
            'task-5-readiness-secret-with-at-least-32-bytes'
        ),
        agent_runtime_fingerprint_key_version='task5-readiness-v1',
        openai_embedding_model='fake-embedding:v1',
        openai_embedding_dimensions=2,
    )


def _seed_generation(
    db: Session,
    *,
    settings: Settings,
    corpus_generation: int = 1,
    vector_index_generation: int = 1,
) -> RagServingCorpusGeneration:
    row = RagServingCorpusGeneration(
        id=1,
        corpus_generation=corpus_generation,
        vector_index_generation=vector_index_generation,
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        index_policy_version='rag-v2-serving-index:v1',
        pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
    )
    db.add(row)
    return row


def _seed_raw_chunk(db: Session, *, ordinal: int) -> DocumentChunk:
    text = f'Canonical readiness body {ordinal}'
    signature = f'{ordinal + 1:064x}'[-64:]
    source = Source(
        source_type='drive',
        source_id=f'drive:readiness:{ordinal}',
        source_url=f'https://drive.example.test/readiness/{ordinal}',
        title=f'Readiness title {ordinal}',
        author='owner@example.test',
        permission_level='internal',
        raw_metadata={'mime_type': 'text/plain'},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db.add(source)
    db.flush()
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v1',
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body=text,
    )
    db.add(version)
    db.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_drive_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='text/plain',
        document_version_label='v1',
        revision_id=f'revision-{ordinal}',
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        parser_run_id=parser_run.id,
        chunk_index=0,
        text=text,
        source_snippet=text,
        permission_level='internal',
        metadata_={},
    )
    db.add(chunk)
    db.flush()
    document.current_document_version_id = version.id
    return chunk


def test_empty_exact_readiness_is_actor_independent_and_provider_free(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_generation(db_session, settings=settings, corpus_generation=0,
                     vector_index_generation=0)
    db_session.commit()
    service = RagV2ServingIndexReadinessService(settings=settings)

    first = service.inspect(db=db_session)
    second = service.inspect(db=db_session)

    assert isinstance(first, RagServingIndexReadiness)
    assert first == second
    assert first.ready is True
    assert first.expected_document_count == 0
    assert first.live_vector_count == 0
    assert first.tombstone_count == 0
    assert first.mismatch_count_capped_at_20 == 0
    assert len(first.readiness_snapshot_hmac) == 64


def test_readiness_reports_committed_tombstones_without_retagging_them_expected(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_generation(
        db_session,
        settings=settings,
        corpus_generation=1,
        vector_index_generation=1,
    )
    item = ReviewItem(
        item_type='decision_record',
        payload={},
        source_links=[],
        source_snippets=[],
        confidence_score=1.0,
        permission_level='internal',
        status='revoked',
    )
    db_session.add(item)
    db_session.flush()
    db_session.add(
        VectorServingTombstone(
            document_id='decision_record:999',
            source_review_item_id=item.id,
            reason_code='source_invalidated',
        )
    )
    db_session.commit()

    result = RagV2ServingIndexReadinessService(settings=settings).inspect(
        db=db_session
    )

    assert result.ready is True
    assert result.expected_document_count == 0
    assert result.tombstone_count == 1
    assert result.mismatch_count_capped_at_20 == 0


def test_exact_d_state_and_live_vector_are_ready_while_legacy_state_is_ignored(
    db_session: Session,
) -> None:
    settings = _settings()
    chunk = _seed_raw_chunk(db_session, ordinal=1)
    _seed_generation(db_session, settings=settings)
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    vector = CosineIndexableVectorValidator().validate(
        [0.25, 0.75],
        expected_dimensions=2,
    )
    upsert_rag_v2_vector_index_state(
        db=db_session,
        state=None,
        document=document,
        embedding_model_name=settings.openai_embedding_model,
        embedding=vector,
        content_hash=compute_rag_v2_document_hash(document),
        vector_index_generation=1,
        settings=settings,
    )
    db_session.add(
        VectorIndexState(
            document_id='decision_record:999',
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=2,
            content_hash='0' * 64,
            status='indexed',
        )
    )
    db_session.commit()
    service = RagV2ServingIndexReadinessService(
        settings=settings,
        live_vector_inspector=lambda _db, _ids: {
            f'chunk:{chunk.id}': vector,
            'decision_record:999': vector,
        },
    )

    result = service.inspect(db=db_session)

    assert result.ready is True
    assert result.expected_document_count == 1
    assert result.live_vector_count == 1
    assert result.tombstone_count == 0
    assert result.mismatch_count_capped_at_20 == 0


def test_missing_wrong_hash_and_invalid_live_vector_fail_closed(
    db_session: Session,
) -> None:
    settings = _settings()
    chunk = _seed_raw_chunk(db_session, ordinal=2)
    _seed_generation(db_session, settings=settings)
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    vector = CosineIndexableVectorValidator().validate(
        [0.25, 0.75],
        expected_dimensions=2,
    )
    state = upsert_rag_v2_vector_index_state(
        db=db_session,
        state=None,
        document=document,
        embedding_model_name=settings.openai_embedding_model,
        embedding=vector,
        content_hash='f' * 64,
        vector_index_generation=1,
        settings=settings,
    )
    db_session.commit()
    service = RagV2ServingIndexReadinessService(
        settings=settings,
        live_vector_inspector=lambda _db, _ids: {
            f'chunk:{chunk.id}': [float('nan'), 1.0],
        },
    )

    result = service.inspect(db=db_session)

    assert state.content_hash == 'f' * 64
    assert result.ready is False
    assert result.expected_document_count == 1
    assert result.live_vector_count == 1
    assert result.mismatch_count_capped_at_20 == 1


def test_readiness_mismatch_count_is_capped_and_snapshot_exposes_no_raw_ids(
    db_session: Session,
) -> None:
    settings = _settings()
    for ordinal in range(25):
        _seed_raw_chunk(db_session, ordinal=ordinal + 10)
    _seed_generation(db_session, settings=settings)
    db_session.commit()

    result = RagV2ServingIndexReadinessService(settings=settings).inspect(
        db=db_session
    )

    assert result.ready is False
    assert result.expected_document_count == 25
    assert result.live_vector_count == 0
    assert result.mismatch_count_capped_at_20 == 20
    assert 'drive:readiness' not in repr(result)
    assert 'chunk:' not in repr(result)
