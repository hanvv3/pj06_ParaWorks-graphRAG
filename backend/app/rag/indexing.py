import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Protocol

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.core.config import Settings
from backend.app.ingestion.source_authority import (
    exact_authority_contains_chunk,
    resolve_exact_source_authority,
)
from backend.app.knowledge.serving_text import canonical_knowledge_text
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
)
from backend.app.models import (
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    RagServingCorpusGeneration,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.embeddings import EmbeddingBatchResult, EmbeddingModel
from backend.app.rag.serving_contracts import (
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    ServingEvidence,
)
from backend.app.rag.serving_generation import (
    RAG_COSINE_POLICY_VERSION,
    RAG_INDEX_POLICY_VERSION,
    RagIndexMutationResult,
    RagServingGenerationLockedContext,
    assert_rag_serving_generation_mutation_context,
    mark_rag_vector_index_mutation,
)
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    ServingMutationLockPlan,
    build_serving_lock_plan,
)
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver
from backend.app.rag.vector_store import VectorDocument
from backend.app.rag.vector_validation import CosineIndexableVectorValidator


class VectorIndexWriter(Protocol):
    def upsert_with_embedding(self, document: VectorDocument, embedding: list[float]) -> None:
        raise NotImplementedError

    def delete_many(self, document_ids: Sequence[str]) -> int:
        raise NotImplementedError

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        raise NotImplementedError


class EmbeddingBudgetExceededError(ValueError):
    def __init__(self, decision: dict[str, float | int | str | None]) -> None:
        self.decision = decision
        super().__init__('embedding budget exceeded')


def _is_production_pgvector_writer(
    *,
    db: Session,
    writer: VectorIndexWriter,
) -> bool:
    return (
        writer.__class__.__name__ == 'PgVectorStore'
        and db.get_bind().dialect.name == 'postgresql'
    )


@dataclass(frozen=True, slots=True)
class _RagV2PreProviderSnapshot:
    corpus_generation: int
    vector_index_generation: int
    generation_updated_at: datetime | None
    canonical_document_hashes: tuple[tuple[str, str], ...]
    lock_plan: ServingMutationLockPlan


class _RagV2PreProviderDriftError(ValueError):
    pass


def _capture_rag_v2_pre_provider_snapshot(
    *,
    db: Session,
    settings: Settings,
    documents: Sequence[VectorDocument],
) -> _RagV2PreProviderSnapshot:
    provider_input_hashes = {
        document.document_id: compute_vector_document_hash(document)
        for document in documents
    }
    if len(provider_input_hashes) != len(documents):
        raise _RagV2PreProviderDriftError(
            'RAG V2 provider input contains duplicate document identities'
        )
    document_ids = tuple(sorted(provider_input_hashes))
    initial_generation = db.scalar(
        select(RagServingCorpusGeneration)
        .where(RagServingCorpusGeneration.id == 1)
        .execution_options(populate_existing=True)
    )
    if initial_generation is None:
        raise RuntimeError('RAG serving generation singleton is unavailable')
    initial_generation_snapshot = _rag_v2_generation_snapshot(initial_generation)
    initial_plan = build_serving_lock_plan(db, document_ids)
    canonical_documents = {
        document.document_id: document
        for document in build_rag_v2_index_documents(db, settings=settings)
        if document.document_id in document_ids
    }
    canonical_hashes = {
        document_id: compute_vector_document_hash(document)
        for document_id, document in canonical_documents.items()
    }
    final_plan = build_serving_lock_plan(db, document_ids)
    final_generation = db.scalar(
        select(RagServingCorpusGeneration)
        .where(RagServingCorpusGeneration.id == 1)
        .execution_options(populate_existing=True)
    )
    if (
        final_generation is None
        or _rag_v2_generation_snapshot(final_generation)
        != initial_generation_snapshot
        or final_plan != initial_plan
        or canonical_hashes != provider_input_hashes
    ):
        raise _RagV2PreProviderDriftError(
            'RAG V2 pre-provider snapshot drifted before dispatch'
        )
    return _RagV2PreProviderSnapshot(
        corpus_generation=initial_generation_snapshot[0],
        vector_index_generation=initial_generation_snapshot[1],
        generation_updated_at=initial_generation_snapshot[2],
        canonical_document_hashes=tuple(sorted(provider_input_hashes.items())),
        lock_plan=initial_plan,
    )


def _rag_v2_generation_snapshot(
    generation: RagServingCorpusGeneration,
) -> tuple[int, int, datetime | None]:
    return (
        generation.corpus_generation,
        generation.vector_index_generation,
        generation.updated_at,
    )


def canonical_float32_vector_sha256(vector: Sequence[float]) -> str:
    payload = b'paraworks:pgvector-float32:v1\x00' + b''.join(
        struct.pack('>f', coordinate) for coordinate in vector
    )
    return sha256(payload).hexdigest()


def build_rag_v2_vector_index_state_hmac(
    *,
    document_id: str,
    embedding_model: str,
    embedding_dimensions: int,
    content_hash: str,
    canonical_document_sha256: str,
    canonical_float32_vector_sha256: str,
    index_state: str,
    vector_index_generation: int,
    settings: Settings,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'canonical_document_sha256_bytes': exact_utf8_bytes(
                canonical_document_sha256
            ),
            'canonical_float32_vector_sha256': (
                canonical_float32_vector_sha256
            ),
            'content_hash_bytes': exact_utf8_bytes(content_hash),
            'cosine_indexable': index_state == 'indexed',
            'document_id_bytes': exact_utf8_bytes(document_id),
            'embedding_dimensions': embedding_dimensions,
            'embedding_model_bytes': exact_utf8_bytes(embedding_model),
            'index_state': index_state,
            'vector_index_generation': vector_index_generation,
        },
        secret=secret,
        schema_version='rag-vector-index-state:v1',
        policy_version=RAG_COSINE_POLICY_VERSION,
    )


def upsert_rag_v2_vector_index_state(
    *,
    db: Session,
    state: VectorIndexState | None,
    document: VectorDocument,
    embedding_model_name: str,
    embedding: Sequence[float],
    content_hash: str,
    vector_index_generation: int,
    settings: Settings,
    generation_context: RagServingGenerationLockedContext | None = None,
) -> VectorIndexState:
    assert_rag_serving_generation_mutation_context(db, generation_context)
    canonical = CosineIndexableVectorValidator().validate(
        embedding,
        expected_dimensions=settings.openai_embedding_dimensions,
    )
    required_metadata = _required_rag_v2_state_metadata(document)
    if (
        embedding_model_name != settings.openai_embedding_model
        or content_hash != compute_rag_v2_document_hash(document)
        or (
            state is not None
            and (
                state.document_id != document.document_id
                or state.embedding_model != embedding_model_name
            )
        )
    ):
        raise ValueError('RAG V2 vector state identity is not canonical')
    generation = db.get(RagServingCorpusGeneration, 1)
    if generation is None:
        raise RuntimeError('RAG serving generation singleton is unavailable')
    if vector_index_generation != generation.vector_index_generation + 1:
        raise ValueError('RAG V2 vector generation is not the locked next generation')
    mark_rag_vector_index_mutation(db)
    now = datetime.now(UTC)
    if state is None:
        state = VectorIndexState(
            document_id=document.document_id,
            embedding_model=embedding_model_name,
            embedding_dimensions=len(canonical),
            content_hash=content_hash,
            status='indexed',
            indexed_at=now,
        )
        db.add(state)
    return _apply_rag_v2_state_metadata(
        state=state,
        document=document,
        embedding_model_name=embedding_model_name,
        canonical_embedding=canonical,
        content_hash=content_hash,
        vector_index_generation=vector_index_generation,
        settings=settings,
        required_metadata=required_metadata,
        indexed_at=now,
    )


def refresh_rag_v2_vector_index_state_metadata(
    *,
    db: Session,
    state: VectorIndexState,
    document: VectorDocument,
    embedding_model_name: str,
    embedding: Sequence[float],
    content_hash: str,
    settings: Settings,
    generation_context: RagServingGenerationLockedContext | None = None,
) -> VectorIndexState:
    assert_rag_serving_generation_mutation_context(db, generation_context)
    canonical = CosineIndexableVectorValidator().validate(
        embedding,
        expected_dimensions=settings.openai_embedding_dimensions,
    )
    required_metadata = _required_rag_v2_state_metadata(document)
    if (
        embedding_model_name != settings.openai_embedding_model
        or content_hash != compute_rag_v2_document_hash(document)
    ):
        raise ValueError('RAG V2 vector state identity is not canonical')
    generation = db.get(RagServingCorpusGeneration, 1)
    if generation is None:
        raise RuntimeError('RAG serving generation singleton is unavailable')
    if (
        state.document_id != document.document_id
        or state.embedding_model != embedding_model_name
        or state.embedding_dimensions != len(canonical)
        or state.content_hash != content_hash
        or state.status != 'indexed'
        or state.cosine_indexable is not True
        or type(state.vector_index_generation) is not int
        or state.vector_index_generation < 0
        or state.vector_index_generation > generation.vector_index_generation
    ):
        raise ValueError('RAG V2 metadata refresh requires a stable indexed vector')
    return _apply_rag_v2_state_metadata(
        state=state,
        document=document,
        embedding_model_name=embedding_model_name,
        canonical_embedding=canonical,
        content_hash=content_hash,
        vector_index_generation=state.vector_index_generation,
        settings=settings,
        required_metadata=required_metadata,
        indexed_at=None,
    )


def _required_rag_v2_state_metadata(
    document: VectorDocument,
) -> dict[str, str]:
    required_metadata = {
        key: document.metadata.get(key)
        for key in (
            'serving_kind',
            'support_mode',
            'serving_identity_hmac',
            'serving_version_fingerprint',
            'model_content_hmac',
            'canonical_citation_projection_hmac',
        )
    }
    if any(
        type(value) is not str or not value
        for value in required_metadata.values()
    ):
        raise ValueError('RAG V2 vector state metadata is incomplete')
    return {key: str(value) for key, value in required_metadata.items()}


def _apply_rag_v2_state_metadata(
    *,
    state: VectorIndexState,
    document: VectorDocument,
    embedding_model_name: str,
    canonical_embedding: Sequence[float],
    content_hash: str,
    vector_index_generation: int,
    settings: Settings,
    required_metadata: dict[str, str],
    indexed_at: datetime | None,
) -> VectorIndexState:
    state_hmac = build_rag_v2_vector_index_state_hmac(
        document_id=document.document_id,
        embedding_model=embedding_model_name,
        embedding_dimensions=len(canonical_embedding),
        content_hash=content_hash,
        canonical_document_sha256=compute_vector_document_hash(document),
        canonical_float32_vector_sha256=(
            canonical_float32_vector_sha256(canonical_embedding)
        ),
        index_state='indexed',
        vector_index_generation=vector_index_generation,
        settings=settings,
    )
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    state.embedding_dimensions = len(canonical_embedding)
    state.embedding_model = embedding_model_name
    state.content_hash = content_hash
    state.status = 'indexed'
    state.last_error = None
    if indexed_at is not None:
        state.indexed_at = indexed_at
    state.corpus_generation_id = 1
    state.serving_kind = str(required_metadata['serving_kind'])
    state.support_mode = str(required_metadata['support_mode'])
    state.effective_permission = document.permission_level
    state.serving_identity_hmac = str(required_metadata['serving_identity_hmac'])
    state.serving_version_fingerprint = str(
        required_metadata['serving_version_fingerprint']
    )
    state.model_content_hmac = str(required_metadata['model_content_hmac'])
    state.canonical_citation_projection_hmac = str(
        required_metadata['canonical_citation_projection_hmac']
    )
    state.index_policy_version = RAG_INDEX_POLICY_VERSION
    state.pgvector_cosine_policy_version = RAG_COSINE_POLICY_VERSION
    state.cosine_indexable = True
    state.vector_index_generation = vector_index_generation
    state.vector_index_state_hmac = state_hmac
    state.fingerprint_key_version = (
        settings.agent_runtime_fingerprint_key_version
    )
    state.fingerprint_key_material_verifier = verifier
    return state


@dataclass(frozen=True)
class VectorIndexResult:
    indexed_count: int
    document_ids: list[str]
    embedding_dimensions: int
    skipped_count: int = 0
    skipped_document_ids: list[str] | None = None
    tombstoned_count: int = 0
    saved_embedding_calls: int = 0
    saved_serving_writes: int = 0
    embedding_request_count: int = 0
    embedding_prompt_tokens: int = 0
    embedding_total_tokens: int = 0
    embedding_budget: dict[str, float | int | str | None] | None = None


class PreviewVectorIndexWriter:
    def __init__(self) -> None:
        self.upserts: list[tuple[VectorDocument, list[float]]] = []
        self.deletes: list[tuple[str, ...]] = []
        self.permission_narrowings: list[tuple[tuple[str, ...], str]] = []

    def upsert_with_embedding(self, document: VectorDocument, embedding: list[float]) -> None:
        self.upserts.append((document, embedding))

    def delete_many(self, document_ids: Sequence[str]) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.deletes.append(normalized)
        return len(normalized)

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.permission_narrowings.append((normalized, permission_level))
        return len(normalized)


def index_vector_documents(
    *,
    documents: list[VectorDocument],
    writer: VectorIndexWriter,
    embedding_model: EmbeddingModel,
) -> VectorIndexResult:
    document_ids: list[str] = []
    embedding_dimensions = 0
    for document in documents:
        embedding = embedding_model.embed(document.text)
        embedding_dimensions = len(embedding)
        writer.upsert_with_embedding(document, embedding)
        document_ids.append(document.document_id)

    return VectorIndexResult(
        indexed_count=len(document_ids),
        document_ids=document_ids,
        embedding_dimensions=embedding_dimensions or _model_dimensions(embedding_model),
        skipped_document_ids=[],
    )


def index_changed_vector_documents(
    *,
    db: Session,
    documents: list[VectorDocument],
    writer: VectorIndexWriter,
    embedding_model: EmbeddingModel,
    embedding_model_name: str,
    persist_state: bool = True,
    embedding_cost_per_1m_tokens: float = 0.0,
    max_embedding_cost_usd: float | None = None,
    enforce_embedding_budget: bool = True,
    settings: Settings | None = None,
    rag_v2: bool = False,
    operator_authorized: bool = False,
) -> VectorIndexResult:
    changed_documents: list[tuple[VectorDocument, str, VectorIndexState | None]] = []
    metadata_refresh_documents: list[
        tuple[VectorDocument, str, VectorIndexState]
    ] = []
    skipped_document_ids: list[str] = []
    embedding_dimensions = _model_dimensions(embedding_model)
    production_pgvector = _is_production_pgvector_writer(
        db=db,
        writer=writer,
    )
    if rag_v2 and persist_state and not production_pgvector:
        raise ValueError(
            'RAG V2 persistent writes require PostgreSQL with pgvector'
        )
    if rag_v2 and production_pgvector and not operator_authorized:
        raise ValueError('RAG V2 live reindex requires operator authorization')
    if production_pgvector and settings is None:
        raise ValueError('PostgreSQL indexing requires serving-lock settings')
    canonical_live_documents = (
        {
            document.document_id: document
            for document in (
                build_rag_v2_index_documents(db, settings=settings)
                if rag_v2 and settings is not None
                else build_rag_index_documents(db)
            )
        }
        if production_pgvector
        else {}
    )

    tombstoned_document_ids = set(
        db.scalars(
            select(VectorServingTombstone.document_id).where(
                VectorServingTombstone.document_id.in_(
                    [document.document_id for document in documents]
                )
            )
        ).all()
    )
    encountered_tombstoned_count = sum(
        document.document_id in tombstoned_document_ids
        for document in documents
    )

    for document in documents:
        if document.document_id in tombstoned_document_ids:
            skipped_document_ids.append(document.document_id)
            continue
        if production_pgvector:
            canonical = canonical_live_documents.get(document.document_id)
            if (
                canonical is None
                or (
                    compute_vector_document_hash(canonical)
                    != compute_vector_document_hash(document)
                    if rag_v2
                    else _content_hash(canonical, rag_v2=False)
                    != _content_hash(document, rag_v2=False)
                )
            ):
                skipped_document_ids.append(document.document_id)
                continue
        content_hash = _content_hash(document, rag_v2=rag_v2)
        state = _get_index_state(
            db=db,
            document_id=document.document_id,
            embedding_model_name=embedding_model_name,
        )
        state_is_skippable = bool(
            state
            and state.status == 'indexed'
            and state.content_hash == content_hash
            and (
                not rag_v2
                or (
                    state.serving_kind is not None
                    and state.embedding_dimensions == embedding_dimensions
                    and state.index_policy_version == RAG_INDEX_POLICY_VERSION
                    and state.pgvector_cosine_policy_version
                    == RAG_COSINE_POLICY_VERSION
                    and state.cosine_indexable is True
                )
            )
        )
        if state_is_skippable:
            skipped_document_ids.append(document.document_id)
            if rag_v2 and production_pgvector and persist_state and state is not None:
                metadata_refresh_documents.append(
                    (document, content_hash, state)
                )
            continue

        changed_documents.append((document, content_hash, state))

    changed_texts = [document.text for document, _, _ in changed_documents]
    planned_documents = [
        document
        for document, _, _ in (
            [*changed_documents, *metadata_refresh_documents]
        )
    ]
    budget_decision = estimate_embedding_budget(
        texts=changed_texts,
        embedding_model_name=embedding_model_name,
        cost_per_1m_tokens=embedding_cost_per_1m_tokens,
        max_cost_usd=max_embedding_cost_usd,
    )
    if enforce_embedding_budget and budget_decision['action'] == 'block':
        raise EmbeddingBudgetExceededError(budget_decision)
    pre_provider_snapshot = None
    if (
        production_pgvector
        and rag_v2
        and settings is not None
        and planned_documents
    ):
        try:
            pre_provider_snapshot = _capture_rag_v2_pre_provider_snapshot(
                db=db,
                settings=settings,
                documents=planned_documents,
            )
        except _RagV2PreProviderDriftError:
            db.rollback()
            rejected_ids = sorted(
                set(skipped_document_ids)
                | {document.document_id for document in planned_documents}
            )
            return VectorIndexResult(
                indexed_count=0,
                document_ids=[],
                embedding_dimensions=embedding_dimensions,
                skipped_count=len(rejected_ids),
                skipped_document_ids=rejected_ids,
                tombstoned_count=encountered_tombstoned_count,
                saved_embedding_calls=len(rejected_ids),
                saved_serving_writes=len(planned_documents),
                embedding_request_count=0,
                embedding_budget=budget_decision,
            )

    if production_pgvector:
        # Provider work must not hold an application transaction or advisory lock.
        db.rollback()
    batch = (
        _embed_many(embedding_model, changed_texts)
        if changed_texts
        else EmbeddingBatchResult(embeddings=[], request_count=0)
    )
    if len(batch.embeddings) != len(changed_documents):
        raise ValueError('embedding batch is not a cosine-indexable float32 batch')
    canonical_embeddings = CosineIndexableVectorValidator().validate_batch(
        batch.embeddings,
        expected_dimensions=embedding_dimensions,
    )
    batch = replace(
        batch,
        embeddings=[list(vector) for vector in canonical_embeddings],
    )
    if production_pgvector:
        return _persist_locked_pgvector_batch(
            db=db,
            writer=writer,
            settings=settings,
            changed_documents=changed_documents,
            metadata_refresh_documents=metadata_refresh_documents,
            embeddings=batch.embeddings,
            skipped_document_ids=skipped_document_ids,
            tombstoned_count=encountered_tombstoned_count,
            embedding_model_name=embedding_model_name,
            embedding_dimensions=embedding_dimensions,
            persist_state=persist_state,
            batch=batch,
            budget_decision=budget_decision,
            rag_v2=rag_v2,
            pre_provider_snapshot=pre_provider_snapshot,
        )

    indexed_document_ids: list[str] = []
    for (document, content_hash, state), embedding in zip(changed_documents, batch.embeddings, strict=True):
        embedding_dimensions = len(embedding)
        writer.upsert_with_embedding(document, embedding)
        indexed_document_ids.append(document.document_id)
        if persist_state:
            _upsert_index_state(
                db=db,
                state=state,
                document=document,
                embedding_model_name=embedding_model_name,
                embedding_dimensions=embedding_dimensions,
                content_hash=content_hash,
            )

    if persist_state:
        db.commit()

    return VectorIndexResult(
        indexed_count=len(indexed_document_ids),
        document_ids=indexed_document_ids,
        embedding_dimensions=embedding_dimensions,
        skipped_count=len(skipped_document_ids),
        skipped_document_ids=skipped_document_ids,
        tombstoned_count=encountered_tombstoned_count,
        saved_embedding_calls=len(skipped_document_ids),
        embedding_request_count=batch.request_count,
        embedding_prompt_tokens=batch.prompt_tokens,
        embedding_total_tokens=batch.total_tokens,
        embedding_budget=budget_decision,
    )


def _persist_locked_pgvector_batch(
    *,
    db: Session,
    writer: VectorIndexWriter,
    settings: Settings,
    changed_documents: list[tuple[VectorDocument, str, VectorIndexState | None]],
    metadata_refresh_documents: list[
        tuple[VectorDocument, str, VectorIndexState]
    ] | None = None,
    embeddings: list[list[float]],
    skipped_document_ids: list[str],
    tombstoned_count: int = 0,
    embedding_model_name: str,
    embedding_dimensions: int,
    persist_state: bool,
    batch: EmbeddingBatchResult,
    budget_decision: dict[str, float | int | str | None],
    rag_v2: bool,
    pre_provider_snapshot: _RagV2PreProviderSnapshot | None = None,
) -> VectorIndexResult:
    metadata_refresh_documents = metadata_refresh_documents or []
    indexed: list[str] = []
    pre_provider_skip_count = len(skipped_document_ids)
    post_provider_skip_count = 0
    stale_skips = list(skipped_document_ids)
    document_ids = sorted(
        {
            document.document_id
            for document, _, _ in (
                [*changed_documents, *metadata_refresh_documents]
            )
        }
    )
    if not document_ids:
        return VectorIndexResult(
            indexed_count=0,
            document_ids=[],
            embedding_dimensions=embedding_dimensions,
            skipped_count=len(stale_skips),
            skipped_document_ids=stale_skips,
            tombstoned_count=tombstoned_count,
            saved_embedding_calls=pre_provider_skip_count,
            embedding_request_count=batch.request_count,
            embedding_prompt_tokens=batch.prompt_tokens,
            embedding_total_tokens=batch.total_tokens,
            embedding_budget=budget_decision,
        )
    if rag_v2 and pre_provider_snapshot is None:
        raise TypeError('RAG V2 persistence requires a pre-provider snapshot')
    if (
        pre_provider_snapshot is not None
        and tuple(sorted(document_ids))
        != tuple(
            document_id
            for document_id, _ in pre_provider_snapshot.canonical_document_hashes
        )
    ):
        raise ValueError('RAG V2 pre-provider snapshot does not match the batch')
    plan = (
        pre_provider_snapshot.lock_plan
        if pre_provider_snapshot is not None
        else build_serving_lock_plan(db, document_ids)
    )

    def discard_locked_batch() -> VectorIndexResult:
        nonlocal post_provider_skip_count
        db.rollback()
        stale_skips.extend(
            document_id
            for document_id in document_ids
            if document_id not in stale_skips
        )
        post_provider_skip_count += len(document_ids)
        return VectorIndexResult(
            indexed_count=0,
            document_ids=[],
            embedding_dimensions=embedding_dimensions,
            skipped_count=len(stale_skips),
            skipped_document_ids=stale_skips,
            tombstoned_count=tombstoned_count,
            saved_embedding_calls=pre_provider_skip_count,
            saved_serving_writes=post_provider_skip_count,
            embedding_request_count=batch.request_count,
            embedding_prompt_tokens=batch.prompt_tokens,
            embedding_total_tokens=batch.total_tokens,
            embedding_budget=budget_decision,
        )

    with KeyedMutationGuard.generation_barrier(db):
        key_context = lock_runtime_state(db)
        if key_context is None:
            raise ValueError('PostgreSQL indexing key runtime unavailable')
        coordinator = ServingMutationLockCoordinator(
            db=db, settings=settings
        )
        try:
            locked = coordinator.acquire(
                key_context=key_context, plan=plan
            )
        except RuntimeError as exc:
            if str(exc) != 'Serving mutation dependency plan changed':
                raise
            return discard_locked_batch()
        eligibility = TrustedServingEligibilityService(db)
        canonical_documents = {
            current.document_id: current
            for current in (
                build_rag_v2_index_documents(db, settings=settings)
                if rag_v2
                else build_rag_index_documents(db)
            )
            if current.document_id in document_ids
        }
        generation = db.get(RagServingCorpusGeneration, 1)
        if generation is None:
            raise RuntimeError('RAG serving generation singleton is unavailable')
        if pre_provider_snapshot is not None:
            exact_hashes = dict(
                pre_provider_snapshot.canonical_document_hashes
            )
            generation_changed = (
                generation.corpus_generation
                != pre_provider_snapshot.corpus_generation
                or generation.vector_index_generation
                != pre_provider_snapshot.vector_index_generation
                or generation.updated_at
                != pre_provider_snapshot.generation_updated_at
            )
            canonical_changed = any(
                canonical is None
                or compute_vector_document_hash(canonical)
                != exact_hashes.get(document_id)
                for document_id, canonical in (
                    (document_id, canonical_documents.get(document_id))
                    for document_id in document_ids
                )
            )
            tombstone_changed = db.scalar(
                select(VectorServingTombstone.id).where(
                    VectorServingTombstone.document_id.in_(document_ids)
                )
            ) is not None
            if generation_changed or canonical_changed or tombstone_changed:
                return discard_locked_batch()
        next_vector_generation = generation.vector_index_generation + 1
        refresh_vectors: list[
            tuple[VectorIndexState, VectorDocument, str, Sequence[float]]
        ] = []
        for document, content_hash, _ in metadata_refresh_documents:
            state = _get_index_state(
                db=db,
                document_id=document.document_id,
                embedding_model_name=embedding_model_name,
            )
            canonical = canonical_documents.get(document.document_id)
            if (
                state is None
                or canonical is None
                or state.status != 'indexed'
                or state.content_hash != content_hash
                or state.embedding_dimensions != embedding_dimensions
                or state.cosine_indexable is not True
                or db.scalar(
                    select(VectorServingTombstone.id).where(
                        VectorServingTombstone.document_id
                        == document.document_id
                    )
                )
                is not None
            ):
                return discard_locked_batch()
            try:
                live_embedding = writer.load_embedding(  # type: ignore[attr-defined]
                    document.document_id,
                    locked_context=locked,
                )
                canonical_live_embedding = (
                    CosineIndexableVectorValidator().validate(
                        live_embedding or (),
                        expected_dimensions=embedding_dimensions,
                    )
                )
            except (TypeError, ValueError):
                return discard_locked_batch()
            refresh_vectors.append(
                (state, canonical, content_hash, canonical_live_embedding)
            )
        for (document, content_hash, _), embedding in zip(
            changed_documents, embeddings, strict=True
        ):
            live = eligibility.for_document(document.document_id)
            canonical = canonical_documents.get(document.document_id)
            if (
                (not rag_v2 and not live.eligible)
                or (not rag_v2 and live.effective_permission is None)
                or canonical is None
                or db.scalar(
                    select(VectorServingTombstone.id).where(
                        VectorServingTombstone.document_id
                        == document.document_id
                    )
                )
                is not None
            ):
                stale_skips.append(document.document_id)
                post_provider_skip_count += 1
                continue
            canonical_hash = _content_hash(canonical, rag_v2=rag_v2)
            exact_snapshot = canonical == document and canonical_hash == content_hash
            permission_only_narrowing = bool(
                not rag_v2
                and _permission_only_narrowing(
                    embedded=document,
                    canonical=canonical,
                )
            )
            if not exact_snapshot and not permission_only_narrowing:
                stale_skips.append(document.document_id)
                post_provider_skip_count += 1
                continue
            writer.upsert_with_embedding(
                canonical, embedding, locked_context=locked  # type: ignore[call-arg]
            )
            indexed.append(document.document_id)
            embedding_dimensions = len(embedding)
            if persist_state:
                state = _get_index_state(
                    db=db,
                    document_id=document.document_id,
                    embedding_model_name=embedding_model_name,
                )
                if rag_v2:
                    upsert_rag_v2_vector_index_state(
                        db=db,
                        state=state,
                        document=canonical,
                        embedding_model_name=embedding_model_name,
                        embedding=embedding,
                        content_hash=canonical_hash,
                        vector_index_generation=next_vector_generation,
                        settings=settings,
                        generation_context=coordinator.generation_context,
                    )
                else:
                    _upsert_index_state(
                        db=db,
                        state=state,
                        document=canonical,
                        embedding_model_name=embedding_model_name,
                        embedding_dimensions=embedding_dimensions,
                        content_hash=canonical_hash,
                    )
        if persist_state:
            for state, canonical, content_hash, live_embedding in refresh_vectors:
                refresh_rag_v2_vector_index_state_metadata(
                    db=db,
                    state=state,
                    document=canonical,
                    embedding_model_name=embedding_model_name,
                    embedding=live_embedding,
                    content_hash=content_hash,
                    settings=settings,
                    generation_context=coordinator.generation_context,
                )
        db.commit()
    return VectorIndexResult(
        indexed_count=len(indexed),
        document_ids=indexed,
        embedding_dimensions=embedding_dimensions,
        skipped_count=len(stale_skips),
        skipped_document_ids=stale_skips,
        tombstoned_count=tombstoned_count,
        saved_embedding_calls=pre_provider_skip_count,
        saved_serving_writes=post_provider_skip_count,
        embedding_request_count=batch.request_count,
        embedding_prompt_tokens=batch.prompt_tokens,
        embedding_total_tokens=batch.total_tokens,
        embedding_budget=budget_decision,
    )


def _permission_only_narrowing(
    *, embedded: VectorDocument, canonical: VectorDocument
) -> bool:
    permission_rank = {'public': 0, 'internal': 1, 'restricted': 2}
    if (
        embedded.permission_level not in permission_rank
        or canonical.permission_level not in permission_rank
        or permission_rank[canonical.permission_level]
        <= permission_rank[embedded.permission_level]
    ):
        return False
    return replace(
        canonical, permission_level=embedded.permission_level
    ) == embedded


def compute_vector_document_hash(document: VectorDocument) -> str:
    payload = {
        'document_id': document.document_id,
        'text': document.text,
        'source_url': document.source_url,
        'source_snippet': document.source_snippet,
        'permission_level': document.permission_level,
        'metadata': document.metadata,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    return sha256(encoded).hexdigest()


def compute_rag_v2_document_hash(document: VectorDocument) -> str:
    """Hash the exact serving bytes that require a new stored vector payload."""
    metadata = {
        key: value
        for key, value in document.metadata.items()
        if key
        not in {
            'canonical_citation_projection_hmac',
            'model_content_hmac',
            'provenance_branch',
            'serving_identity_hmac',
            'serving_version_fingerprint',
        }
    }
    payload = {
        'document_id': document.document_id,
        'text': document.text,
        'source_url': document.source_url,
        'source_snippet': document.source_snippet,
        'permission_level': document.permission_level,
        'metadata': metadata,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    ).encode('utf-8')
    return sha256(encoded).hexdigest()


def _content_hash(document: VectorDocument, *, rag_v2: bool) -> str:
    return (
        compute_rag_v2_document_hash(document)
        if rag_v2
        else compute_vector_document_hash(document)
    )


def index_rag_v2_serving_documents(
    *,
    db: Session,
    documents: list[VectorDocument],
    writer: VectorIndexWriter,
    embedding_model: EmbeddingModel,
    embedding_model_name: str,
    settings: Settings,
    persist_state: bool,
    operator_authorized: bool = False,
) -> RagIndexMutationResult:
    result = index_changed_vector_documents(
        db=db,
        documents=documents,
        writer=writer,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
        persist_state=persist_state,
        settings=settings,
        rag_v2=True,
        operator_authorized=operator_authorized,
    )
    generation = db.get(RagServingCorpusGeneration, 1)
    return RagIndexMutationResult(
        indexed_count=result.indexed_count,
        skipped_count=result.skipped_count,
        tombstoned_count=result.tombstoned_count,
        saved_embedding_calls=result.saved_embedding_calls,
        corpus_generation=(generation.corpus_generation if generation else 0),
        vector_index_generation=(
            generation.vector_index_generation if generation else 0
        ),
    )


def estimate_embedding_budget(
    *,
    texts: list[str],
    embedding_model_name: str,
    cost_per_1m_tokens: float,
    max_cost_usd: float | None,
) -> dict[str, float | int | str | None]:
    estimated_input_tokens = sum(_estimate_embedding_tokens(text) for text in texts)
    estimated_cost = (
        Decimal(estimated_input_tokens) * Decimal(str(cost_per_1m_tokens)) / Decimal(1_000_000)
    )
    estimated_cost_usd = float(estimated_cost)

    if not texts:
        return {
            'embedding_model': embedding_model_name,
            'changed_document_count': 0,
            'estimated_input_tokens': 0,
            'estimated_cost_usd': 0.0,
            'budget_limit_usd': max_cost_usd,
            'budget_status': 'no_input',
            'action': 'skip',
            'reason': 'no_changed_documents',
        }

    if max_cost_usd is not None and estimated_cost_usd > max_cost_usd:
        return {
            'embedding_model': embedding_model_name,
            'changed_document_count': len(texts),
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_cost_usd': estimated_cost_usd,
            'budget_limit_usd': max_cost_usd,
            'budget_status': 'over_budget',
            'action': 'block',
            'reason': 'estimated_embedding_cost_exceeds_budget',
        }

    return {
        'embedding_model': embedding_model_name,
        'changed_document_count': len(texts),
        'estimated_input_tokens': estimated_input_tokens,
        'estimated_cost_usd': estimated_cost_usd,
        'budget_limit_usd': max_cost_usd,
        'budget_status': 'within_budget' if max_cost_usd is not None else 'not_limited',
        'action': 'run',
        'reason': 'within_embedding_budget',
    }


def build_rag_index_documents(db: Session) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    documents.extend(_chunk_documents(db))
    eligibility = TrustedServingEligibilityService(db)
    documents.extend(_decision_documents(db, eligibility))
    documents.extend(_history_documents(db, eligibility))
    documents.extend(_timeline_documents(db, eligibility))
    documents.extend(_todo_documents(db, eligibility))
    return documents


def build_rag_v2_index_documents(
    db: Session,
    *,
    settings: Settings,
) -> list[VectorDocument]:
    """Build the actor-independent D serving corpus without changing V1."""
    documents: list[VectorDocument] = []
    raw_resolver = CanonicalSourceObservationResolver(db=db, settings=settings)
    chunk_ids = tuple(db.scalars(select(DocumentChunk.id).order_by(DocumentChunk.id)))
    for chunk_id in chunk_ids:
        observation = raw_resolver.resolve_for_index(chunk_id)
        if observation is None:
            continue
        source = db.get(Source, observation.raw_version.source_row_id)
        chunk = db.get(DocumentChunk, observation.raw_version.document_chunk_id)
        if source is None or chunk is None:
            continue
        documents.append(
            _rag_v2_vector_document(
                evidence=observation.evidence,
                source_url=source.source_url,
                source_snippet=chunk.source_snippet,
                typed_row_id=chunk.id,
                source_pk=source.id,
            )
        )

    trusted_resolver = TrustedServingEnvelopeResolver(db=db, settings=settings)
    for knowledge_type, model in (
        ('decision_record', DecisionRecord),
        ('history_event', HistoryEvent),
        ('timeline_event', TimelineEvent),
        ('todo', Todo),
    ):
        knowledge_ids = tuple(db.scalars(select(model.id).order_by(model.id)))
        for knowledge_id in knowledge_ids:
            envelope = trusted_resolver.resolve_for_index(
                knowledge_type,
                knowledge_id,
            )
            if envelope is None:
                continue
            citation = _trusted_citation_bytes(db, evidence=envelope.evidence)
            if citation is None:
                continue
            documents.append(
                _rag_v2_vector_document(
                    evidence=envelope.evidence,
                    source_url=citation[0],
                    source_snippet=citation[1],
                    typed_row_id=knowledge_id,
                )
            )
    return documents


def _rag_v2_vector_document(
    *,
    evidence: ServingEvidence,
    source_url: str,
    source_snippet: str,
    typed_row_id: int,
    source_pk: int | None = None,
) -> VectorDocument:
    metadata: dict[str, object] = {
        'canonical_citation_projection_hmac': (
            evidence.canonical_citation_projection_hmac
        ),
        'index_policy_version': RAG_INDEX_POLICY_VERSION,
        'model_content_hmac': evidence.model_content_hmac,
        'provenance_branch': evidence.provenance.branch,
        'public_source_id': evidence.public_source_id,
        'public_source_type': evidence.public_source_type,
        'serving_identity_hmac': evidence.serving_identity_hmac,
        'serving_kind': evidence.serving_kind,
        'serving_version_fingerprint': evidence.serving_version_fingerprint,
        'support_mode': evidence.support_mode,
    }
    if evidence.serving_kind == 'raw_chunk':
        metadata['chunk_id'] = typed_row_id
        metadata['source_pk'] = source_pk
    else:
        metadata['knowledge_id'] = typed_row_id
    return VectorDocument(
        document_id=evidence.serving_document_id,
        text=evidence.model_content,
        source_url=source_url,
        source_snippet=source_snippet,
        permission_level=evidence.effective_permission,
        metadata=metadata,
    )


def _trusted_citation_bytes(
    db: Session,
    *,
    evidence: ServingEvidence,
) -> tuple[str, str] | None:
    provenance = evidence.provenance
    if isinstance(provenance, ExplicitApprovalProvenance):
        review_item_id = provenance.review_item_id
        ordinal = provenance.selected_citation_child.review_item_source_pair_ordinal
    elif isinstance(provenance, LegacyHumanProvenance):
        review_item_id = provenance.legacy_source_review_item_id
        ordinal = 0
    else:
        return None
    item = db.get(ReviewItem, review_item_id)
    if (
        item is None
        or not isinstance(item.source_links, list)
        or not isinstance(item.source_snippets, list)
        or ordinal < 0
        or ordinal >= len(item.source_links)
        or ordinal >= len(item.source_snippets)
    ):
        return None
    return item.source_links[ordinal], item.source_snippets[ordinal]


def _chunk_documents(db: Session) -> list[VectorDocument]:
    # Phase 2: 승인 기반 RAG (Approval-only RAG)
    # 사람이 '승인(approved)'한 ReviewItem에 포함된 source_id 목록만 수집
    approved_payloads = db.execute(
        select(ReviewItem.payload).where(
            ReviewItem.status == 'approved',
            or_(
                ReviewItem.resolution_source.is_(None),
                ReviewItem.resolution_source == 'human',
            ),
        )
    ).scalars().all()
    
    approved_sid_set: set[str] = set()
    for p in approved_payloads:
        if not isinstance(p, dict):
            continue
        source_ids = p.get('source_ids')
        if not isinstance(source_ids, list):
            continue
        approved_sid_set.update(
            source_id.strip()
            for source_id in source_ids
            if isinstance(source_id, str) and source_id.strip()
        )
            
    if not approved_sid_set:
        return []

    rows = db.execute(
        select(
            DocumentChunk,
            Source,
            DocumentVersion,
            Document,
            DocumentParserRun,
        )
        .join(Source, DocumentChunk.source_id == Source.id)
        .join(DocumentVersion, DocumentVersion.id == DocumentChunk.version_id)
        .join(
            Document,
            and_(
                Document.id == DocumentVersion.document_id,
                Document.source_id == Source.id,
                Document.current_document_version_id == DocumentVersion.id,
            ),
        )
        .join(
            DocumentParserRun,
            and_(
                DocumentParserRun.id == DocumentChunk.parser_run_id,
                DocumentParserRun.document_id == Document.id,
                DocumentParserRun.document_version_id == DocumentVersion.id,
                DocumentParserRun.source_id == Source.id,
            ),
        )
        .where(Source.source_id.in_(list(approved_sid_set)))
        .order_by(DocumentChunk.id)
    ).all()
    
    documents: list[VectorDocument] = []
    eligibility = TrustedServingEligibilityService(db)
    for chunk, source, version, document, parser_run in rows:
        serving = eligibility.for_document(f'chunk:{chunk.id}')
        if not serving.eligible or serving.effective_permission is None:
            continue
        if not _is_exact_current_server_chunk(
            db,
            source=source,
            document=document,
            version=version,
            parser_run=parser_run,
            chunk=chunk,
        ):
            continue
        timestamp = source.raw_metadata.get('ts') or source.created_at.isoformat()
        
        # 메타데이터 보강 (정적 태그 + 동적 태그)
        metadata = {
            'chunk_id': chunk.id,
            'source_pk': source.id,
            'source_id': source.source_id,
            'source_type': source.source_type,
            'author': source.author,
            'author_name': source.raw_metadata.get('author_name') or source.author,
            'channel_name': source.raw_metadata.get('channel_name'),
            'timestamp': str(timestamp),
            'created_at_date': source.raw_metadata.get('created_at_date'),
            'category': chunk.metadata_.get('category'),
            'topic_tag': chunk.metadata_.get('topic_tag'),
            'importance': chunk.metadata_.get('importance'),
            'scenario': source.raw_metadata.get('scenario'),
            **_document_parser_metadata(
                chunk=chunk,
                version=version,
                parser_run=parser_run,
            ),
        }
        
        documents.append(
            VectorDocument(
                document_id=f'chunk:{chunk.id}',
                text=chunk.text,
                source_url=source.source_url,
                source_snippet=chunk.source_snippet,
                permission_level=serving.effective_permission,
                metadata=metadata,
            )
        )
    return documents


def _is_exact_current_server_chunk(
    db: Session,
    *,
    source: Source,
    document: Document,
    version: DocumentVersion,
    parser_run: DocumentParserRun,
    chunk: DocumentChunk,
) -> bool:
    authority = resolve_exact_source_authority(db, source=source)
    return bool(
        authority is not None
        and authority.document.id == document.id
        and authority.version.id == version.id
        and authority.parser_run.id == parser_run.id
        and exact_authority_contains_chunk(authority, chunk)
    )


def _document_parser_metadata(
    *,
    chunk: DocumentChunk,
    version: DocumentVersion,
    parser_run: DocumentParserRun,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        'parser_name': parser_run.parser_name,
        'parser_status': parser_run.parser_status,
        'parser_status_reason': parser_run.parser_status_reason,
        'mime_type': parser_run.mime_type,
        'document_version': version.version,
        'revision_id': parser_run.revision_id,
        'content_signature': parser_run.content_signature,
        'content_hash': sha256(chunk.text.encode('utf-8')).hexdigest(),
    }
    for key in ('section_path', 'page_number'):
        if key in chunk.metadata_:
            metadata[key] = chunk.metadata_[key]
    return metadata


def _decision_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    # 결정사항 테이블 조회 (이미 승인된 것만 저장됨)
    decisions = db.scalars(
        select(DecisionRecord)
        .where(DecisionRecord.review_status == 'approved')
        .order_by(DecisionRecord.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'decision_record:{decision.id}',
            source_type='decision_record',
            title=decision.title,
            text=canonical_knowledge_text('decision_record', decision),
            source_links=decision.source_links,
            source_snippets=decision.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=decision.created_at.isoformat(),
            project_key=decision.project_key,
        )
        for decision in decisions
        if (result := eligibility.for_knowledge('decision_record', decision.id)).eligible
    ]


def _history_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    # 기록/공유 테이블 조회
    events = db.scalars(
        select(HistoryEvent)
        .where(HistoryEvent.review_status == 'approved')
        .order_by(HistoryEvent.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'history_event:{event.id}',
            source_type='history_event',
            title=event.title,
            text=canonical_knowledge_text('history_event', event),
            source_links=event.source_links,
            source_snippets=event.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=event.created_at.isoformat(),
            project_key=event.project_key,
        )
        for event in events
        if (result := eligibility.for_knowledge('history_event', event.id)).eligible
    ]


def _timeline_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    events = db.scalars(
        select(TimelineEvent)
        .where(TimelineEvent.review_status == 'approved')
        .order_by(TimelineEvent.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'timeline_event:{event.id}',
            source_type='timeline_event',
            title=event.title,
            text=canonical_knowledge_text('timeline_event', event),
            source_links=event.source_links,
            source_snippets=event.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=event.created_at.isoformat(),
            project_key=event.project_key,
        )
        for event in events
        if (result := eligibility.for_knowledge('timeline_event', event.id)).eligible
    ]


def _todo_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    # 할 일 테이블 조회
    todos = db.scalars(
        select(Todo)
        .where(Todo.review_status == 'approved')
        .order_by(Todo.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'todo:{todo.id}',
            source_type='todo',
            title=todo.title,
            text=canonical_knowledge_text('todo', todo),
            source_links=todo.source_links,
            source_snippets=todo.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=todo.created_at.isoformat(),
            project_key=todo.project_key,
        )
        for todo in todos
        if (result := eligibility.for_knowledge('todo', todo.id)).eligible
    ]


def _knowledge_document(
    *,
    document_id: str,
    source_type: str,
    title: str,
    text: str,
    source_links: list[str],
    source_snippets: list[str],
    permission_level: str,
    timestamp: str,
    project_key: str | None = None,
) -> VectorDocument:
    # 지식 항목 인덱싱 시에도 메타데이터 최대한 보강
    return VectorDocument(
        document_id=document_id,
        text=text,
        source_url=source_links[0] if source_links else f'knowledge://{document_id}',
        source_snippet=source_snippets[0] if source_snippets else text[:240],
        permission_level=permission_level,
        metadata={
            'knowledge_id': document_id,
            'source_type': source_type,
            'title': title,
            'author': 'ParaWorks AI (Verified)',
            'timestamp': timestamp,
            'project_key': project_key,
        },
    )


def _model_dimensions(embedding_model: EmbeddingModel) -> int:
    return int(getattr(embedding_model, 'dimensions', 0))


def _estimate_embedding_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative preflight estimate: UTF-8 bytes / 4 tracks English reasonably
    # and errs high for Korean before any paid embedding call is made.
    return max(1, (len(text.encode('utf-8')) + 3) // 4)


def _embed_many(embedding_model: EmbeddingModel, texts: list[str]) -> EmbeddingBatchResult:
    if hasattr(embedding_model, 'embed_many'):
        return embedding_model.embed_many(texts)
    return EmbeddingBatchResult(
        embeddings=[embedding_model.embed(text) for text in texts],
        request_count=len(texts),
    )


def _get_index_state(
    *,
    db: Session,
    document_id: str,
    embedding_model_name: str,
) -> VectorIndexState | None:
    return db.scalar(
        select(VectorIndexState).where(
            VectorIndexState.document_id == document_id,
            VectorIndexState.embedding_model == embedding_model_name,
        )
    )


def _upsert_index_state(
    *,
    db: Session,
    state: VectorIndexState | None,
    document: VectorDocument,
    embedding_model_name: str,
    embedding_dimensions: int,
    content_hash: str,
) -> None:
    now = datetime.now(UTC)
    if state is None:
        db.add(
            VectorIndexState(
                document_id=document.document_id,
                embedding_model=embedding_model_name,
                embedding_dimensions=embedding_dimensions,
                content_hash=content_hash,
                status='indexed',
                last_error=None,
                indexed_at=now,
            )
        )
        return

    state.embedding_dimensions = embedding_dimensions
    state.content_hash = content_hash
    state.status = 'indexed'
    state.last_error = None
    state.indexed_at = now
