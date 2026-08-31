from __future__ import annotations

import math
import struct
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.connectors.base import SourceEvent
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.ingestion.service import ingest_events_with_result
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.knowledge.promotion import promote_review_item
from backend.app.models import (
    AutoReviewRuntimeKeyState,
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    RagLexicalServingProjection,
    RagServingCorpusGeneration,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.embeddings import EmbeddingBatchResult
from backend.app.rag.index_readiness import RagV2ServingIndexReadinessService
from backend.app.rag.indexing import (
    _capture_rag_v2_pre_provider_snapshot,
    _persist_locked_pgvector_batch,
    build_rag_index_documents,
    build_rag_v2_index_documents,
    build_rag_v2_vector_index_state_hmac,
    canonical_float32_vector_sha256,
    compute_rag_v2_document_hash,
    index_rag_v2_serving_documents,
    upsert_rag_v2_vector_index_state,
)
from backend.app.rag.lexical_projection import (
    current_rag_lexical_projections,
    score_rag_lexical_candidate,
    tokenize_rag_lexical_query,
)
from backend.app.rag.serving_generation import (
    RagIndexMutationResult,
    advance_corpus_generation,
    arm_corpus_generation_refresh,
    assert_corpus_generation_refresh_armed,
    increment_vector_index_generation,
    lock_rag_serving_generation,
)
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    build_serving_lock_plan,
)
from backend.app.rag.vector_store import VectorDocument
from backend.app.rag.vector_validation import CosineIndexableVectorValidator
from backend.app.review.actors import human_review_actor
from backend.app.review.auto_review_quality_revoke import (
    AutoReviewQualityRevokeService,
)
from backend.app.review.auto_review_source_reconciliation import (
    AutoReviewSourceReconciliationService,
)
from backend.app.review.transitions import ReviewTransitionService


def _settings() -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret=(
            'task-5-indexing-secret-with-at-least-32-bytes'
        ),
        agent_runtime_fingerprint_key_version='task5-index-v1',
        openai_embedding_model='fake-embedding:v1',
        openai_embedding_dimensions=2,
    )


def _seed_canonical_raw_chunk(db: Session) -> tuple[Source, DocumentChunk]:
    text = 'Exact  raw\nobservation with % and _ literals'
    signature = 'a' * 64
    source = Source(
        source_type='gmail',
        source_id='gmail:task-5-raw',
        source_url='https://mail.example.test/messages/task-5-raw',
        title='Raw TITLE %_Case',
        author='owner@example.test',
        permission_level='internal',
        raw_metadata={},
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
    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id='revision-1',
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
        source_snippet='Exact raw observation with % and _ literals',
        permission_level='internal',
        metadata_={},
    )
    db.add(chunk)
    document.current_document_version_id = version.id
    db.flush()
    return source, chunk


def _seed_trusted_and_legacy_unbound(
    db: Session,
) -> tuple[DecisionRecord, DecisionRecord]:
    item = ReviewItem(
        item_type='decision_record',
        payload={'title': 'Bound trusted decision'},
        source_links=['https://knowledge.example.test/reviews/bound'],
        source_snippets=['Exact trusted citation'],
        confidence_score=0.98,
        permission_level='internal',
        status='approved',
        resolution_source='human',
    )
    db.add(item)
    db.flush()
    bound = DecisionRecord(
        title='Bound TRUSTED Decision',
        decision_summary='Canonical trusted decision body.',
        source_links=list(item.source_links),
        source_snippets=list(item.source_snippets),
        confidence_score=0.98,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=item.id,
    )
    unbound = DecisionRecord(
        title='Legacy unbound decision',
        decision_summary='This remains visible only to legacy indexing.',
        source_links=['https://knowledge.example.test/legacy-unbound'],
        source_snippets=['Legacy unbound citation'],
        confidence_score=0.91,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=None,
    )
    db.add_all([bound, unbound])
    db.commit()
    return bound, unbound


def test_d_builder_is_distinct_from_legacy_and_binds_exact_serving_metadata(
    db_session: Session,
) -> None:
    source, chunk = _seed_canonical_raw_chunk(db_session)
    bound, unbound = _seed_trusted_and_legacy_unbound(db_session)

    legacy = build_rag_index_documents(db_session)
    serving = build_rag_v2_index_documents(db_session, settings=_settings())

    assert [document.document_id for document in legacy] == [
        f'decision_record:{bound.id}',
        f'decision_record:{unbound.id}',
    ]
    assert [document.document_id for document in serving] == [
        f'chunk:{chunk.id}',
        f'decision_record:{bound.id}',
    ]
    raw, trusted = serving
    assert raw.metadata == {
        'canonical_citation_projection_hmac': raw.metadata[
            'canonical_citation_projection_hmac'
        ],
        'chunk_id': chunk.id,
        'index_policy_version': 'rag-v2-serving-index:v1',
        'model_content_hmac': raw.metadata['model_content_hmac'],
        'provenance_branch': 'raw_chunk',
        'public_source_id': source.source_id,
        'public_source_type': 'gmail',
        'serving_identity_hmac': raw.metadata['serving_identity_hmac'],
        'serving_kind': 'raw_chunk',
        'serving_version_fingerprint': raw.metadata[
            'serving_version_fingerprint'
        ],
        'source_pk': source.id,
        'support_mode': 'source_observation',
    }
    assert trusted.metadata == {
        'canonical_citation_projection_hmac': trusted.metadata[
            'canonical_citation_projection_hmac'
        ],
        'index_policy_version': 'rag-v2-serving-index:v1',
        'knowledge_id': bound.id,
        'model_content_hmac': trusted.metadata['model_content_hmac'],
        'provenance_branch': 'legacy_human_base',
        'public_source_id': f'decision_record:{bound.id}',
        'public_source_type': 'decision_record',
        'serving_identity_hmac': trusted.metadata['serving_identity_hmac'],
        'serving_kind': 'trusted_knowledge',
        'serving_version_fingerprint': trusted.metadata[
            'serving_version_fingerprint'
        ],
        'support_mode': 'trusted_fact',
    }
    assert all(
        len(document.metadata[key]) == 64
        for document in serving
        for key in (
            'canonical_citation_projection_hmac',
            'model_content_hmac',
            'serving_identity_hmac',
            'serving_version_fingerprint',
        )
    )


def test_cosine_validator_returns_exact_finite_nonzero_float32_tuple() -> None:
    vector = CosineIndexableVectorValidator().validate(
        [0.1, -2, 3.25],
        expected_dimensions=3,
    )

    assert isinstance(vector, tuple)
    assert vector == tuple(
        struct.unpack('>f', struct.pack('>f', value))[0]
        for value in (0.1, -2.0, 3.25)
    )
    assert all(math.isfinite(value) for value in vector)
    assert any(value != 0.0 for value in vector)


@pytest.mark.parametrize(
    ('vector', 'dimensions'),
    [
        ([1.0], 2),
        ([True, 1.0], 2),
        (['1.0', 2.0], 2),
        ([float('nan'), 1.0], 2),
        ([float('inf'), 1.0], 2),
        ([3.5e38, 1.0], 2),
        ([0.0, -0.0], 2),
        ([1.0e-50, -1.0e-50], 2),
        ([1.0], True),
        ([1.0], 0),
    ],
)
def test_cosine_validator_rejects_noncanonical_or_unindexable_vectors(
    vector: list[object],
    dimensions: int,
) -> None:
    with pytest.raises(ValueError, match='cosine-indexable float32'):
        CosineIndexableVectorValidator().validate(
            vector,
            expected_dimensions=dimensions,
        )


def test_cosine_validator_rejects_an_entire_batch_when_one_vector_is_zero() -> None:
    validator = CosineIndexableVectorValidator()

    with pytest.raises(ValueError, match='cosine-indexable float32'):
        validator.validate_batch(
            [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            expected_dimensions=2,
        )


class _FakeBatchEmbeddingModel:
    dimensions = 2

    def __init__(self, batches: list[list[list[float]]]) -> None:
        self._batches = list(batches)
        self.calls: list[tuple[str, ...]] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError('D indexing must use embed_many')

    def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
        self.calls.append(tuple(texts))
        return EmbeddingBatchResult(
            embeddings=self._batches.pop(0),
            request_count=1,
        )


class _RecordingWriter:
    def __init__(self) -> None:
        self.upserts: list[tuple[VectorDocument, list[float]]] = []
        self.deletes: list[tuple[str, ...]] = []

    def upsert_with_embedding(
        self,
        document: VectorDocument,
        embedding: list[float],
    ) -> None:
        self.upserts.append((document, embedding))

    def delete_many(self, document_ids: list[str] | tuple[str, ...]) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.deletes.append(normalized)
        return len(normalized)

    def narrow_permissions(
        self,
        document_ids: list[str] | tuple[str, ...],
        permission_level: str,
    ) -> int:
        return len(set(document_ids))


class _LockedRecordingWriter(_RecordingWriter):
    def __init__(
        self,
        *,
        live_embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__()
        self.live_embeddings = live_embeddings or {}
        self.embedding_reads: list[str] = []

    def upsert_with_embedding(
        self,
        document: VectorDocument,
        embedding: list[float],
        *,
        locked_context: object,
    ) -> None:
        del locked_context
        super().upsert_with_embedding(document, embedding)

    def load_embedding(
        self,
        document_id: str,
        *,
        locked_context: object,
    ) -> list[float] | None:
        del locked_context
        self.embedding_reads.append(document_id)
        return self.live_embeddings.get(document_id)


def _seed_corpus_generation(db: Session, *, settings: Settings) -> None:
    db.add(
        RagServingCorpusGeneration(
            id=1,
            corpus_generation=0,
            vector_index_generation=0,
            embedding_model='fake-embedding:v1',
            embedding_dimensions=2,
            index_policy_version='rag-v2-serving-index:v1',
            pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
        )
    )


def _seed_d_index_state(
    db: Session,
    *,
    document: VectorDocument,
    settings: Settings,
    content_hash: str | None = None,
) -> VectorIndexState:
    state = VectorIndexState(
        document_id=document.document_id,
        embedding_model='fake-embedding:v1',
        embedding_dimensions=2,
        content_hash=content_hash or compute_rag_v2_document_hash(document),
        status='indexed',
        corpus_generation_id=1,
        serving_kind=document.metadata['serving_kind'],
        support_mode=document.metadata['support_mode'],
        effective_permission=document.permission_level,
        serving_identity_hmac=document.metadata['serving_identity_hmac'],
        serving_version_fingerprint=document.metadata[
            'serving_version_fingerprint'
        ],
        model_content_hmac=document.metadata['model_content_hmac'],
        canonical_citation_projection_hmac=document.metadata[
            'canonical_citation_projection_hmac'
        ],
        index_policy_version='rag-v2-serving-index:v1',
        pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
        cosine_indexable=True,
        vector_index_generation=0,
        vector_index_state_hmac='f' * 64,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
    )
    db.add(state)
    return state


def test_d_incremental_skip_precedes_one_batch_provider_call_and_reports_counts(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    _seed_trusted_and_legacy_unbound(db_session)
    documents = build_rag_v2_index_documents(db_session, settings=settings)
    _seed_corpus_generation(db_session, settings=settings)
    _seed_d_index_state(db_session, document=documents[0], settings=settings)
    db_session.commit()
    provider = _FakeBatchEmbeddingModel([[[0.25, 0.75]]])
    writer = _RecordingWriter()

    result = index_rag_v2_serving_documents(
        db=db_session,
        documents=documents,
        writer=writer,
        embedding_model=provider,
        embedding_model_name='fake-embedding:v1',
        settings=settings,
        persist_state=False,
    )

    assert isinstance(result, RagIndexMutationResult)
    assert provider.calls == [(documents[1].text,)]
    assert [row[0].document_id for row in writer.upserts] == [
        documents[1].document_id
    ]
    assert result == RagIndexMutationResult(
        indexed_count=1,
        skipped_count=1,
        tombstoned_count=0,
        saved_embedding_calls=1,
        corpus_generation=0,
        vector_index_generation=0,
    )


def test_d_invalid_batch_rejects_every_write_and_state_change(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    _seed_trusted_and_legacy_unbound(db_session)
    documents = build_rag_v2_index_documents(db_session, settings=settings)
    _seed_corpus_generation(db_session, settings=settings)
    db_session.commit()
    before_states = tuple(db_session.scalars(select(VectorIndexState.id)).all())
    provider = _FakeBatchEmbeddingModel([[[1.0, 0.0], [0.0, 0.0]]])
    writer = _RecordingWriter()

    with pytest.raises(ValueError, match='cosine-indexable float32'):
        index_rag_v2_serving_documents(
            db=db_session,
            documents=documents,
            writer=writer,
            embedding_model=provider,
            embedding_model_name='fake-embedding:v1',
            settings=settings,
            persist_state=False,
        )

    assert provider.calls == [(documents[0].text, documents[1].text)]
    assert writer.upserts == []
    assert tuple(db_session.scalars(select(VectorIndexState.id)).all()) == before_states


def test_d_incremental_result_reports_existing_tombstone_skips(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    bound, _ = _seed_trusted_and_legacy_unbound(db_session)
    documents = build_rag_v2_index_documents(db_session, settings=settings)
    _seed_corpus_generation(db_session, settings=settings)
    _seed_d_index_state(
        db_session,
        document=documents[0],
        settings=settings,
    )
    db_session.add(
        VectorServingTombstone(
            document_id=documents[1].document_id,
            source_review_item_id=bound.source_review_item_id,
            reason_code='source_invalidated',
        )
    )
    db_session.commit()
    provider = _FakeBatchEmbeddingModel([])

    result = index_rag_v2_serving_documents(
        db=db_session,
        documents=documents,
        writer=_RecordingWriter(),
        embedding_model=provider,
        embedding_model_name=settings.openai_embedding_model,
        settings=settings,
        persist_state=False,
    )

    assert provider.calls == []
    assert result.indexed_count == 0
    assert result.skipped_count == 2
    assert result.tombstoned_count == 1
    assert result.saved_embedding_calls == 2


def test_d_sqlite_persistent_write_refuses_before_provider_work(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    documents = build_rag_v2_index_documents(db_session, settings=settings)
    provider = _FakeBatchEmbeddingModel([[[1.0, 0.0]]])

    with pytest.raises(ValueError, match='PostgreSQL.*pgvector'):
        index_rag_v2_serving_documents(
            db=db_session,
            documents=documents,
            writer=_RecordingWriter(),
            embedding_model=provider,
            embedding_model_name='fake-embedding:v1',
            settings=settings,
            persist_state=True,
            operator_authorized=True,
        )

    assert provider.calls == []


def test_generation_lock_follows_key_runtime_and_precedes_projection(
    db_session: Session,
) -> None:
    settings = _settings()
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        assert db_session.info['paraworks_c5_keyed_lock_order'] == [
            'generation_shared',
            'runtime_share',
            'rag_corpus_update',
            'projection',
        ]
        assert increment_vector_index_generation(
            db_session,
            context=generation_context,
        ) == 1
        db_session.commit()

    row = db_session.get(RagServingCorpusGeneration, 1)
    assert row is not None
    assert row.corpus_generation == 0
    assert row.vector_index_generation == 1
    assert row.embedding_model == settings.openai_embedding_model
    assert row.embedding_dimensions == settings.openai_embedding_dimensions
    with pytest.raises(TypeError, match='transaction'):
        increment_vector_index_generation(
            db_session,
            context=generation_context,
        )


def test_corpus_advance_refreshes_exact_lexical_rows_in_the_same_transaction(
    db_session: Session,
) -> None:
    settings = _settings()
    source, chunk = _seed_canonical_raw_chunk(db_session)
    bound, _ = _seed_trusted_and_legacy_unbound(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()
    documents = {
        document.document_id: document
        for document in build_rag_v2_index_documents(db_session, settings=settings)
    }

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        assert advance_corpus_generation(
            db_session,
            settings=settings,
            context=generation_context,
        ) == 1
        assert db_session.get(RagServingCorpusGeneration, 1).corpus_generation == 1
        assert {
            row.corpus_generation
            for row in db_session.scalars(select(RagLexicalServingProjection))
        } == {1}
        db_session.commit()

    rows = {
        row.serving_document_id: row
        for row in current_rag_lexical_projections(
            db_session,
            settings=settings,
        )
    }
    assert set(rows) == {f'chunk:{chunk.id}', f'decision_record:{bound.id}'}
    raw = rows[f'chunk:{chunk.id}']
    trusted = rows[f'decision_record:{bound.id}']
    assert raw.title_lower == source.title.lower()
    assert raw.searchable_lower == f'{source.title}\n{documents[raw.serving_document_id].text}'.lower()
    assert trusted.title_lower == bound.title.lower()
    assert trusted.searchable_lower == (
        f'{bound.title}\n{documents[trusted.serving_document_id].text}'.lower()
    )
    for document_id, row in rows.items():
        document = documents[document_id]
        assert row.serving_identity_hmac == document.metadata[
            'serving_identity_hmac'
        ]
        assert row.serving_version_fingerprint == document.metadata[
            'serving_version_fingerprint'
        ]
        assert row.model_content_hmac == document.metadata['model_content_hmac']
        assert row.canonical_citation_projection_hmac == document.metadata[
            'canonical_citation_projection_hmac'
        ]
        assert row.lexical_contract_version == 'rag-keyword-lexical-compat:v1'
        assert row.fingerprint_key_version == (
            settings.agent_runtime_fingerprint_key_version
        )
        assert row.fingerprint_key_material_verifier == verifier

    raw.corpus_generation = 0
    db_session.execute(
        delete(RagLexicalServingProjection).where(
            RagLexicalServingProjection.serving_document_id
            == trusted.serving_document_id
        )
    )
    db_session.commit()
    assert current_rag_lexical_projections(db_session, settings=settings) == ()


def test_lexical_oracle_preserves_literal_wildcards_and_duplicate_ordinality() -> None:
    question = '%_Case, %_Case.'

    terms = tokenize_rag_lexical_query(question)
    score, matched = score_rag_lexical_candidate(
        question=question,
        title='Raw TITLE %_Case',
        text='Exact raw observation with percent and underscore literals',
    )

    assert terms == ('%_case', '%_case')
    assert matched == ('%_case', '%_case')
    assert score == 1.3


def test_ingestion_advances_corpus_and_lexical_only_for_canonical_change(
    db_session: Session,
) -> None:
    settings = _settings()
    event = SourceEvent(
        source_type='drive',
        source_id='drive:task-5-incremental',
        source_url='https://drive.example.test/task-5-incremental',
        title='Generation TITLE',
        body='Canonical body version one.',
        author='owner@example.test',
        participants=['owner@example.test'],
        timestamp=datetime(2026, 8, 31, 9, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'document_version': 'v1',
            'revision_id': 'revision-v1',
            'mime_type': 'text/plain',
        },
        semantic_timestamp_raw='2026-08-31T09:00:00Z',
    )

    first = ingest_events_with_result(db_session, [event], settings=settings)
    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert first.changed_source_ids == [event.source_id]
    assert generation is not None
    assert generation.corpus_generation == 1
    assert generation.vector_index_generation == 0
    projection = db_session.scalars(select(RagLexicalServingProjection)).one()
    assert projection.corpus_generation == 1
    assert projection.title_lower == event.title.lower()
    assert projection.searchable_lower == f'{event.title}\n{event.body}'.lower()

    unchanged = ingest_events_with_result(db_session, [event], settings=settings)
    db_session.refresh(generation)
    assert unchanged.skipped_events == 1
    assert generation.corpus_generation == 1

    changed = SourceEvent(
        **{
            **event.__dict__,
            'body': 'Canonical body version two.',
            'raw_metadata': {
                **event.raw_metadata,
                'document_version': 'v2',
                'revision_id': 'revision-v2',
            },
        }
    )
    ingest_events_with_result(db_session, [changed], settings=settings)
    db_session.refresh(generation)
    assert generation.corpus_generation == 2
    projection = db_session.scalars(select(RagLexicalServingProjection)).one()
    assert projection.corpus_generation == 2
    assert projection.searchable_lower == (
        f'{event.title}\n{changed.body}'.lower()
    )


def test_bulk_review_savepoints_advance_corpus_only_at_outer_commit(
    db_session: Session,
) -> None:
    settings = _settings()
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    items = [
        ReviewItem(
            item_type='decision_record',
            payload={
                'title': f'Bulk generation decision {ordinal}',
                'decision_summary': f'Canonical bulk summary {ordinal}.',
            },
            source_links=[f'https://example.test/bulk-generation/{ordinal}'],
            source_snippets=[f'Exact bulk evidence {ordinal}.'],
            confidence_score=0.99,
            permission_level='internal',
            status='pending_review',
        )
        for ordinal in (1, 2)
    ]
    db_session.add_all(
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
            *items,
        ]
    )
    db_session.commit()

    result = ReviewTransitionService(settings=settings).transition_many(
        db=db_session,
        item_ids=[item.id for item in items],
        action='approve',
        actor=human_review_actor(USERS['admin']),
    )

    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert len(result.results) == 2
    assert generation is not None
    assert generation.corpus_generation == 0
    db_session.commit()
    db_session.refresh(generation)
    assert generation.corpus_generation == 1
    assert generation.vector_index_generation == 0


def test_locked_trusted_permission_and_revoke_auto_refresh_corpus_generation(
    db_session: Session,
) -> None:
    settings = _settings()
    bound, _ = _seed_trusted_and_legacy_unbound(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()
    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        advance_corpus_generation(
            db_session,
            settings=settings,
            context=generation_context,
        )
        db_session.commit()

    document_id = f'decision_record:{bound.id}'
    plan = build_serving_lock_plan(db_session, [document_id])
    db_session.rollback()
    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        ServingMutationLockCoordinator(db=db_session, settings=settings).acquire(
            key_context=key_context,
            plan=plan,
        )
        target = db_session.get(DecisionRecord, bound.id)
        target.permission_level = 'restricted'
        db_session.commit()

    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert generation.corpus_generation == 2
    projection = db_session.scalars(select(RagLexicalServingProjection)).one()
    assert projection.effective_permission == 'restricted'
    assert projection.corpus_generation == 2

    plan = build_serving_lock_plan(db_session, [document_id])
    db_session.rollback()
    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        ServingMutationLockCoordinator(db=db_session, settings=settings).acquire(
            key_context=key_context,
            plan=plan,
        )
        target = db_session.get(DecisionRecord, bound.id)
        target.review_status = 'revoked'
        db_session.commit()

    db_session.refresh(generation)
    assert generation.corpus_generation == 3
    assert db_session.scalar(select(RagLexicalServingProjection.id)) is None


def test_direct_trusted_promotion_refuses_without_same_transaction_generation_guard(
    db_session: Session,
) -> None:
    item = ReviewItem(
        item_type='decision_record',
        payload={
            'title': 'Guarded promotion',
            'decision_summary': 'The generation guard is mandatory.',
        },
        source_links=['https://example.test/guarded-promotion'],
        source_snippets=['Exact guarded evidence'],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    with pytest.raises(TypeError, match='generation guard'):
        promote_review_item(db_session, item)

    db_session.rollback()
    assert db_session.scalar(
        select(DecisionRecord.id).where(DecisionRecord.title == 'Guarded promotion')
    ) is None


def test_source_pointer_repair_advances_corpus_in_the_repair_transaction(
    db_session: Session,
) -> None:
    settings = _settings()
    _, chunk = _seed_canonical_raw_chunk(db_session)
    document = db_session.get(
        Document,
        db_session.get(DocumentVersion, chunk.version_id).document_id,
    )
    document.current_document_version_id = None
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=settings,
    ).repair_current_document_versions()

    assert result.repaired_count == 1
    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert generation is not None
    assert generation.corpus_generation == 1
    assert generation.vector_index_generation == 0
    projection = db_session.scalars(select(RagLexicalServingProjection)).one()
    assert projection.serving_document_id == f'chunk:{chunk.id}'
    assert projection.corpus_generation == 1


def test_d_tombstone_increments_only_vector_index_generation(
    db_session: Session,
) -> None:
    settings = _settings()
    bound, _ = _seed_trusted_and_legacy_unbound(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()
    document_id = f'decision_record:{bound.id}'
    plan = build_serving_lock_plan(db_session, [document_id])
    db_session.rollback()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        ServingMutationLockCoordinator(db=db_session, settings=settings).acquire(
            key_context=key_context,
            plan=plan,
        )
        db_session.add(
            VectorServingTombstone(
                document_id=document_id,
                source_review_item_id=bound.source_review_item_id,
                reason_code='source_invalidated',
            )
        )
        # Real revoke paths autoflush the tombstone before their final commit.
        db_session.flush()
        db_session.commit()

    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert generation is not None
    assert generation.corpus_generation == 1
    assert generation.vector_index_generation == 1

    plan = build_serving_lock_plan(db_session, [document_id])
    db_session.rollback()
    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        ServingMutationLockCoordinator(db=db_session, settings=settings).acquire(
            key_context=key_context,
            plan=plan,
        )
        db_session.commit()
    db_session.refresh(generation)
    assert generation.vector_index_generation == 1


def test_d_vector_state_identity_binds_exact_float32_payload_and_generation() -> None:
    settings = _settings()
    canonical = CosineIndexableVectorValidator().validate(
        [0.1, -2.0],
        expected_dimensions=2,
    )
    expected_digest = sha256(
        b'paraworks:pgvector-float32:v1\x00'
        + b''.join(struct.pack('>f', value) for value in canonical)
    ).hexdigest()

    digest = canonical_float32_vector_sha256(canonical)
    first = build_rag_v2_vector_index_state_hmac(
        document_id='chunk:7',
        embedding_model='fake-embedding:v1',
        embedding_dimensions=2,
        content_hash='a' * 64,
        canonical_document_sha256='b' * 64,
        canonical_float32_vector_sha256=digest,
        index_state='indexed',
        vector_index_generation=3,
        settings=settings,
    )
    changed_generation = build_rag_v2_vector_index_state_hmac(
        document_id='chunk:7',
        embedding_model='fake-embedding:v1',
        embedding_dimensions=2,
        content_hash='a' * 64,
        canonical_document_sha256='b' * 64,
        canonical_float32_vector_sha256=digest,
        index_state='indexed',
        vector_index_generation=4,
        settings=settings,
    )

    assert digest == expected_digest
    assert len(first) == 64
    assert first != changed_generation


def test_d_vector_state_write_binds_canonical_document_and_exact_generation(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    _seed_corpus_generation(db_session, settings=settings)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()
    canonical = CosineIndexableVectorValidator().validate(
        [0.25, 0.75],
        expected_dimensions=2,
    )

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        arm_corpus_generation_refresh(
            db_session,
            settings=settings,
            context=generation_context,
        )
        state = upsert_rag_v2_vector_index_state(
            db=db_session,
            state=None,
            document=document,
            embedding_model_name='fake-embedding:v1',
            embedding=canonical,
            content_hash=compute_rag_v2_document_hash(document),
            vector_index_generation=1,
            settings=settings,
            generation_context=generation_context,
        )
        db_session.commit()

    assert state.serving_kind == 'raw_chunk'
    assert state.support_mode == 'source_observation'
    assert state.effective_permission == document.permission_level
    assert state.serving_identity_hmac == document.metadata['serving_identity_hmac']
    assert state.serving_version_fingerprint == document.metadata[
        'serving_version_fingerprint'
    ]
    assert state.model_content_hmac == document.metadata['model_content_hmac']
    assert state.canonical_citation_projection_hmac == document.metadata[
        'canonical_citation_projection_hmac'
    ]
    assert state.index_policy_version == 'rag-v2-serving-index:v1'
    assert state.pgvector_cosine_policy_version == 'pgvector-cosine-indexable:v1'
    assert state.cosine_indexable is True
    assert state.vector_index_generation == 1
    assert len(state.vector_index_state_hmac) == 64
    assert state.fingerprint_key_version == (
        settings.agent_runtime_fingerprint_key_version
    )
    assert state.fingerprint_key_material_verifier == (
        fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        )
    )


def test_d_vector_state_write_rejects_wrong_configured_dimension(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        arm_corpus_generation_refresh(
            db_session,
            settings=settings,
            context=generation_context,
        )
        with pytest.raises(ValueError, match='cosine-indexable float32'):
            upsert_rag_v2_vector_index_state(
                db=db_session,
                state=None,
                document=document,
                embedding_model_name=settings.openai_embedding_model,
                embedding=[1.0],
                content_hash=compute_rag_v2_document_hash(document),
                vector_index_generation=1,
                settings=settings,
                generation_context=generation_context,
            )
        db_session.rollback()

    assert db_session.scalars(select(VectorIndexState)).all() == []


def test_direct_d_vector_state_write_refuses_before_validation_without_guard(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    _seed_corpus_generation(db_session, settings=settings)
    db_session.commit()

    with pytest.raises(TypeError, match='locked RAG generation context'):
        upsert_rag_v2_vector_index_state(
            db=db_session,
            state=None,
            document=document,
            embedding_model_name=settings.openai_embedding_model,
            embedding=[1.0],
            content_hash=compute_rag_v2_document_hash(document),
            vector_index_generation=1,
            settings=settings,
        )

    assert db_session.scalars(select(VectorIndexState)).all() == []


def test_direct_d_vector_state_write_marks_one_locked_vector_generation(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    db_session.commit()
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    db_session.rollback()

    with KeyedMutationGuard.generation_barrier(db_session):
        key_context = lock_runtime_state(db_session)
        generation_context = lock_rag_serving_generation(
            db_session,
            settings=settings,
            key_context=key_context,
        )
        acquire_projection(db_session, key_context)
        arm_corpus_generation_refresh(
            db_session,
            settings=settings,
            context=generation_context,
        )
        state = upsert_rag_v2_vector_index_state(
            db=db_session,
            state=None,
            document=document,
            embedding_model_name=settings.openai_embedding_model,
            embedding=[0.25, 0.75],
            content_hash=compute_rag_v2_document_hash(document),
            vector_index_generation=1,
            settings=settings,
            generation_context=generation_context,
        )
        db_session.commit()

    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert state.vector_index_generation == 1
    assert generation is not None
    assert generation.vector_index_generation == 1


def test_locked_d_vector_write_updates_state_and_generation_atomically(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    _seed_corpus_generation(db_session, settings=settings)
    db_session.commit()
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    content_hash = compute_rag_v2_document_hash(document)
    snapshot = _capture_rag_v2_pre_provider_snapshot(
        db=db_session,
        settings=settings,
        documents=[document],
    )
    writer = _LockedRecordingWriter()

    result = _persist_locked_pgvector_batch(
        db=db_session,
        writer=writer,
        settings=settings,
        changed_documents=[(document, content_hash, None)],
        embeddings=[[0.25, 0.75]],
        skipped_document_ids=[],
        embedding_model_name=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        persist_state=True,
        batch=EmbeddingBatchResult(
            embeddings=[[0.25, 0.75]],
            request_count=1,
        ),
        budget_decision={},
        rag_v2=True,
        pre_provider_snapshot=snapshot,
    )

    assert result.indexed_count == 1
    assert [row[0].document_id for row in writer.upserts] == [
        document.document_id
    ]
    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert generation.corpus_generation == 0
    assert generation.vector_index_generation == 1
    state = db_session.scalar(
        select(VectorIndexState).where(
            VectorIndexState.document_id == document.document_id
        )
    )
    assert state.serving_kind == 'raw_chunk'
    assert state.vector_index_generation == 1
    assert len(state.vector_index_state_hmac) == 64


@pytest.mark.parametrize(
    'drift_kind',
    (
        'corpus_generation',
        'vector_generation',
        'content',
        'permission',
        'provenance',
    ),
)
def test_d_post_provider_snapshot_discards_every_dependency_drift(
    db_session: Session,
    drift_kind: str,
) -> None:
    settings = _settings()
    source, chunk = _seed_canonical_raw_chunk(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    _seed_corpus_generation(db_session, settings=settings)
    db_session.commit()
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    content_hash = compute_rag_v2_document_hash(document)
    snapshot = _capture_rag_v2_pre_provider_snapshot(
        db=db_session,
        settings=settings,
        documents=[document],
    )
    db_session.rollback()

    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert generation is not None
    if drift_kind == 'corpus_generation':
        generation.corpus_generation += 1
    elif drift_kind == 'vector_generation':
        generation.vector_index_generation += 1
    elif drift_kind == 'content':
        db_session.get(DocumentChunk, chunk.id).text = 'Changed after provider.'
    elif drift_kind == 'permission':
        db_session.get(Source, source.id).permission_level = 'restricted'
    else:
        parser_run = db_session.get(DocumentParserRun, chunk.parser_run_id)
        parser_run.revision_id = 'revision-after-provider'
    db_session.commit()
    writer = _LockedRecordingWriter()

    result = _persist_locked_pgvector_batch(
        db=db_session,
        writer=writer,
        settings=settings,
        changed_documents=[(document, content_hash, None)],
        embeddings=[[0.25, 0.75]],
        skipped_document_ids=[],
        embedding_model_name=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        persist_state=True,
        batch=EmbeddingBatchResult(
            embeddings=[[0.25, 0.75]],
            request_count=1,
        ),
        budget_decision={},
        rag_v2=True,
        pre_provider_snapshot=snapshot,
    )

    assert writer.upserts == []
    assert result.indexed_count == 0
    assert result.saved_serving_writes == 1
    assert result.embedding_request_count == 1


@pytest.mark.parametrize('drift_kind', ('reaffirm', 'version', 'key'))
def test_stable_d_vector_refreshes_exact_metadata_without_generation_advance(
    db_session: Session,
    drift_kind: str,
) -> None:
    settings = _settings()
    _seed_canonical_raw_chunk(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    _seed_corpus_generation(db_session, settings=settings)
    db_session.commit()
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    content_hash = compute_rag_v2_document_hash(document)
    state = _seed_d_index_state(
        db_session,
        document=document,
        settings=settings,
    )
    state.vector_index_state_hmac = '0' * 64
    if drift_kind == 'reaffirm':
        state.serving_identity_hmac = '1' * 64
    elif drift_kind == 'version':
        state.serving_version_fingerprint = '2' * 64
    else:
        state.fingerprint_key_version = 'stale-key'
        state.fingerprint_key_material_verifier = '3' * 64
    db_session.commit()
    snapshot = _capture_rag_v2_pre_provider_snapshot(
        db=db_session,
        settings=settings,
        documents=[document],
    )
    vector = [0.25, 0.75]
    writer = _LockedRecordingWriter(
        live_embeddings={document.document_id: vector}
    )

    result = _persist_locked_pgvector_batch(
        db=db_session,
        writer=writer,
        settings=settings,
        changed_documents=[],
        metadata_refresh_documents=[(document, content_hash, state)],
        embeddings=[],
        skipped_document_ids=[document.document_id],
        embedding_model_name=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        persist_state=True,
        batch=EmbeddingBatchResult(embeddings=[], request_count=0),
        budget_decision={},
        rag_v2=True,
        pre_provider_snapshot=snapshot,
    )

    generation = db_session.get(RagServingCorpusGeneration, 1)
    db_session.refresh(state)
    readiness = RagV2ServingIndexReadinessService(
        settings=settings,
        live_vector_inspector=lambda _db, _ids: {
            document.document_id: vector
        },
    ).inspect(db=db_session)
    assert writer.upserts == []
    assert writer.embedding_reads == [document.document_id]
    assert result.indexed_count == 0
    assert result.embedding_request_count == 0
    assert result.saved_embedding_calls == 1
    assert generation is not None
    assert generation.vector_index_generation == 0
    assert state.serving_identity_hmac == document.metadata['serving_identity_hmac']
    assert state.serving_version_fingerprint == document.metadata[
        'serving_version_fingerprint'
    ]
    assert state.fingerprint_key_version == (
        settings.agent_runtime_fingerprint_key_version
    )
    assert readiness.ready is True


def test_quality_authority_mutation_enters_generation_guard_before_locked_rows(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    bound, _ = _seed_trusted_and_legacy_unbound(db_session)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add_all([
        TrustedKnowledgeApprovalLink(
            knowledge_type='decision_record',
            knowledge_id=bound.id,
            review_item_id=bound.source_review_item_id,
            security_scope_id='workspace-task-5',
            promotion_effect_kind='primary',
            resolution_source='human',
            claim_fingerprint='d' * 64,
            permission_level='internal',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            active=True,
        ),
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        ),
    ])
    db_session.commit()
    service = AutoReviewQualityRevokeService(
        db_session,
        settings=settings,
    )
    observed: list[list[str]] = []

    def fake_locked(**_kwargs: object) -> tuple[object, bool]:
        assert_corpus_generation_refresh_armed(db_session)
        observed.append(list(db_session.info['paraworks_c5_keyed_lock_order']))
        return object(), False

    monkeypatch.setattr(
        service,
        '_commit_quality_authority_locked',
        fake_locked,
        raising=False,
    )

    service._commit_quality_authority(
        review_item_id=bound.source_review_item_id,
        actor=object(),  # type: ignore[arg-type]
        reason_code='incorrect_content',
        reason='bounded reason',
    )

    assert observed == [[
        'generation_shared',
        'runtime_share',
        'rag_corpus_update',
        'projection',
    ]]
    db_session.rollback()


def test_ingestion_d_vector_delete_advances_only_vector_generation(
    db_session: Session,
) -> None:
    settings = _settings()
    event = SourceEvent(
        source_type='drive',
        source_id='drive:task-5-vector-delete',
        source_url='https://drive.example.test/task-5-vector-delete',
        title='Vector delete generation',
        body='Canonical first vector body.',
        author='owner@example.test',
        participants=['owner@example.test'],
        timestamp=datetime(2026, 8, 31, 10, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'document_version': 'v1',
            'revision_id': 'revision-v1',
            'mime_type': 'text/plain',
        },
        semantic_timestamp_raw='2026-08-31T10:00:00Z',
    )
    ingest_events_with_result(db_session, [event], settings=settings)
    document = build_rag_v2_index_documents(db_session, settings=settings)[0]
    _seed_d_index_state(
        db_session,
        document=document,
        settings=settings,
    )
    db_session.commit()
    changed = SourceEvent(
        **{
            **event.__dict__,
            'body': 'Canonical replacement vector body.',
            'raw_metadata': {
                **event.raw_metadata,
                'document_version': 'v2',
                'revision_id': 'revision-v2',
            },
        }
    )
    writer = _RecordingWriter()

    ingest_events_with_result(
        db_session,
        [changed],
        settings=settings,
        vector_writer=writer,
    )

    generation = db_session.get(RagServingCorpusGeneration, 1)
    assert generation.corpus_generation == 2
    assert generation.vector_index_generation == 1
    assert writer.deletes == [(document.document_id,)]
