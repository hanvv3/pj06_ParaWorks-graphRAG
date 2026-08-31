from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.core.config import Settings, get_settings
from backend.app.models import (
    RagServingCorpusGeneration,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.indexing import (
    build_rag_v2_index_documents,
    build_rag_v2_vector_index_state_hmac,
    canonical_float32_vector_sha256,
    compute_rag_v2_document_hash,
    compute_vector_document_hash,
)
from backend.app.rag.serving_generation import (
    RAG_COSINE_POLICY_VERSION,
    RAG_INDEX_POLICY_VERSION,
)
from backend.app.rag.vector_validation import CosineIndexableVectorValidator

LiveVectorInspector = Callable[
    [Session, tuple[str, ...]],
    Mapping[str, Sequence[object]],
]


@dataclass(frozen=True, slots=True)
class RagServingIndexReadiness:
    ready: bool
    corpus_generation: int
    vector_index_generation: int
    expected_document_count: int
    live_vector_count: int
    tombstone_count: int
    mismatch_count_capped_at_20: int
    embedding_model: str
    embedding_dimensions: int
    index_policy_version: str
    readiness_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class _LiveVectorSnapshot:
    dimensions: int | None
    cosine_indexable: bool
    canonical_float32_vector_sha256: str | None


class RagV2ServingIndexReadinessService:
    """Inspect the actor-independent D corpus without provider work."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        live_vector_inspector: LiveVectorInspector | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._live_vector_inspector = live_vector_inspector

    def inspect(self, *, db: Session) -> RagServingIndexReadiness:
        settings = self._settings
        generation = db.get(RagServingCorpusGeneration, 1)
        generation_barrier = _generation_barrier_snapshot(generation)
        corpus_generation = generation.corpus_generation if generation else 0
        vector_index_generation = (
            generation.vector_index_generation if generation else 0
        )
        expected_documents = {
            document.document_id: document
            for document in build_rag_v2_index_documents(db, settings=settings)
        }
        verifier = fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        )
        generation_identity_matches = bool(
            generation is not None
            and type(generation.corpus_generation) is int
            and generation.corpus_generation >= 0
            and type(generation.vector_index_generation) is int
            and generation.vector_index_generation >= 0
            and generation.embedding_model == settings.openai_embedding_model
            and generation.embedding_dimensions
            == settings.openai_embedding_dimensions
            and generation.index_policy_version == RAG_INDEX_POLICY_VERSION
            and generation.pgvector_cosine_policy_version
            == RAG_COSINE_POLICY_VERSION
            and generation.fingerprint_key_version
            == settings.agent_runtime_fingerprint_key_version
            and generation.fingerprint_key_material_verifier == verifier
        )

        tracked_states = tuple(
            db.scalars(
                select(VectorIndexState)
                .where(VectorIndexState.serving_kind.is_not(None))
                .order_by(VectorIndexState.document_id, VectorIndexState.id)
            ).all()
        )
        states_by_document: dict[str, list[VectorIndexState]] = {}
        for state in tracked_states:
            states_by_document.setdefault(state.document_id, []).append(state)
        relevant_document_ids = tuple(
            sorted(set(expected_documents) | set(states_by_document))
        )
        live_vectors = self._inspect_live_vectors(
            db,
            relevant_document_ids=relevant_document_ids,
        )
        if self._live_vector_inspector is not None:
            live_vectors = {
                document_id: snapshot
                for document_id, snapshot in live_vectors.items()
                if document_id in relevant_document_ids
            }

        explicit_tombstones = set(
            db.scalars(
                select(VectorServingTombstone.document_id)
            ).all()
        )
        state_tombstones = {
            state.document_id
            for state in tracked_states
            if state.status == 'tombstoned'
        }
        tombstone_ids = explicit_tombstones | state_tombstones
        mismatches: set[str] = set()
        if not generation_identity_matches:
            mismatches.add('generation_identity')

        exact_state_hmacs: list[str] = []
        for document_id, document in expected_documents.items():
            states = states_by_document.get(document_id, [])
            state = next(
                (
                    candidate
                    for candidate in states
                    if candidate.embedding_model
                    == settings.openai_embedding_model
                ),
                None,
            )
            live = live_vectors.get(document_id)
            if (
                state is None
                or live is None
                or document_id in tombstone_ids
                or not self._expected_state_matches(
                    state=state,
                    document=document,
                    live=live,
                    generation=vector_index_generation,
                    verifier=verifier,
                )
            ):
                mismatches.add(document_id)
                continue
            exact_state_hmacs.append(state.vector_index_state_hmac or '')

        for document_id, states in states_by_document.items():
            if document_id in expected_documents:
                continue
            has_live = document_id in live_vectors
            clean_tombstone = bool(
                not has_live
                and document_id in tombstone_ids
                and all(state.status == 'tombstoned' for state in states)
            )
            if not clean_tombstone:
                mismatches.add(document_id)

        for document_id in tombstone_ids & set(live_vectors):
            mismatches.add(document_id)
        for document_id in set(live_vectors) - set(expected_documents):
            if document_id not in states_by_document:
                mismatches.add(document_id)

        final_generation = db.scalar(
            select(RagServingCorpusGeneration)
            .where(RagServingCorpusGeneration.id == 1)
            .execution_options(populate_existing=True)
        )
        if _generation_barrier_snapshot(final_generation) != generation_barrier:
            generation_identity_matches = False
            mismatches.add('generation_changed_during_inspection')

        ready = generation_identity_matches and not mismatches
        mismatch_count = min(len(mismatches), 20)
        expected_identity_hmacs = sorted(
            str(document.metadata['serving_identity_hmac'])
            for document in expected_documents.values()
        )
        snapshot_hmac = self._snapshot_hmac(
            ready=ready,
            corpus_generation=corpus_generation,
            vector_index_generation=vector_index_generation,
            expected_identity_hmacs=expected_identity_hmacs,
            exact_state_hmacs=sorted(exact_state_hmacs),
            expected_document_count=len(expected_documents),
            live_vector_count=len(live_vectors),
            tombstone_count=len(tombstone_ids),
            mismatch_count=mismatch_count,
        )
        return RagServingIndexReadiness(
            ready=ready,
            corpus_generation=corpus_generation,
            vector_index_generation=vector_index_generation,
            expected_document_count=len(expected_documents),
            live_vector_count=len(live_vectors),
            tombstone_count=len(tombstone_ids),
            mismatch_count_capped_at_20=mismatch_count,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
            index_policy_version=RAG_INDEX_POLICY_VERSION,
            readiness_snapshot_hmac=snapshot_hmac,
        )

    def _expected_state_matches(
        self,
        *,
        state: VectorIndexState,
        document: Any,
        live: _LiveVectorSnapshot,
        generation: int,
        verifier: str,
    ) -> bool:
        settings = self._settings
        metadata = document.metadata
        if (
            state.status != 'indexed'
            or state.embedding_model != settings.openai_embedding_model
            or state.embedding_dimensions != settings.openai_embedding_dimensions
            or state.content_hash != compute_rag_v2_document_hash(document)
            or state.corpus_generation_id != 1
            or state.serving_kind != metadata.get('serving_kind')
            or state.support_mode != metadata.get('support_mode')
            or state.effective_permission != document.permission_level
            or state.serving_identity_hmac
            != metadata.get('serving_identity_hmac')
            or state.serving_version_fingerprint
            != metadata.get('serving_version_fingerprint')
            or state.model_content_hmac != metadata.get('model_content_hmac')
            or state.canonical_citation_projection_hmac
            != metadata.get('canonical_citation_projection_hmac')
            or state.index_policy_version != RAG_INDEX_POLICY_VERSION
            or state.pgvector_cosine_policy_version
            != RAG_COSINE_POLICY_VERSION
            or state.cosine_indexable is not True
            or type(state.vector_index_generation) is not int
            or state.vector_index_generation < 0
            or state.vector_index_generation > generation
            or state.fingerprint_key_version
            != settings.agent_runtime_fingerprint_key_version
            or state.fingerprint_key_material_verifier != verifier
            or live.dimensions != settings.openai_embedding_dimensions
            or not live.cosine_indexable
            or live.canonical_float32_vector_sha256 is None
        ):
            return False
        expected_state_hmac = build_rag_v2_vector_index_state_hmac(
            document_id=state.document_id,
            embedding_model=state.embedding_model,
            embedding_dimensions=state.embedding_dimensions,
            content_hash=state.content_hash,
            canonical_document_sha256=compute_vector_document_hash(document),
            canonical_float32_vector_sha256=(
                live.canonical_float32_vector_sha256
            ),
            index_state='indexed',
            vector_index_generation=state.vector_index_generation,
            settings=settings,
        )
        return state.vector_index_state_hmac == expected_state_hmac

    def _inspect_live_vectors(
        self,
        db: Session,
        *,
        relevant_document_ids: tuple[str, ...],
    ) -> dict[str, _LiveVectorSnapshot]:
        if self._live_vector_inspector is not None:
            values = self._live_vector_inspector(db, relevant_document_ids)
            return {
                str(document_id): self._snapshot_from_vector(vector)
                for document_id, vector in values.items()
            }
        if db.get_bind().dialect.name != 'postgresql':
            return {}
        rows = (
            db.execute(
                text(
                    'SELECT document_id, embedding::text AS embedding_text, '
                    'vector_dims(embedding) AS embedding_dimensions, '
                    'vector_norm(embedding) > 0 AS norm_positive '
                    'FROM rag_vector_documents '
                    'WHERE document_id = ANY(:document_ids) OR '
                    "metadata_json->>'index_policy_version' = :index_policy_version"
                ),
                {
                    'document_ids': list(relevant_document_ids),
                    'index_policy_version': RAG_INDEX_POLICY_VERSION,
                },
            )
            .mappings()
            .all()
        )
        snapshots: dict[str, _LiveVectorSnapshot] = {}
        for row in rows:
            document_id = str(row['document_id'])
            try:
                vector = _parse_pgvector_text(str(row['embedding_text']))
                snapshot = self._snapshot_from_vector(vector)
                if (
                    int(row['embedding_dimensions']) != snapshot.dimensions
                    or row['norm_positive'] is not True
                ):
                    snapshot = _LiveVectorSnapshot(
                        dimensions=int(row['embedding_dimensions']),
                        cosine_indexable=False,
                        canonical_float32_vector_sha256=None,
                    )
            except (TypeError, ValueError):
                snapshot = _LiveVectorSnapshot(
                    dimensions=None,
                    cosine_indexable=False,
                    canonical_float32_vector_sha256=None,
                )
            snapshots[document_id] = snapshot
        return snapshots

    def _snapshot_from_vector(
        self,
        vector: Sequence[object],
    ) -> _LiveVectorSnapshot:
        dimensions = (
            len(vector)
            if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes))
            else None
        )
        try:
            canonical = CosineIndexableVectorValidator().validate(
                vector,
                expected_dimensions=self._settings.openai_embedding_dimensions,
            )
        except (TypeError, ValueError):
            return _LiveVectorSnapshot(
                dimensions=dimensions,
                cosine_indexable=False,
                canonical_float32_vector_sha256=None,
            )
        return _LiveVectorSnapshot(
            dimensions=len(canonical),
            cosine_indexable=True,
            canonical_float32_vector_sha256=(
                canonical_float32_vector_sha256(canonical)
            ),
        )

    def _snapshot_hmac(
        self,
        *,
        ready: bool,
        corpus_generation: int,
        vector_index_generation: int,
        expected_identity_hmacs: list[str],
        exact_state_hmacs: list[str],
        expected_document_count: int,
        live_vector_count: int,
        tombstone_count: int,
        mismatch_count: int,
    ) -> str:
        settings = self._settings
        secret, _ = fingerprint_secret_bytes(settings)
        return keyed_fingerprint(
            {
                'corpus_generation': corpus_generation,
                'embedding_dimensions': settings.openai_embedding_dimensions,
                'embedding_model_bytes': exact_utf8_bytes(
                    settings.openai_embedding_model
                ),
                'exact_state_hmacs': exact_state_hmacs,
                'expected_document_count': expected_document_count,
                'expected_identity_hmacs': expected_identity_hmacs,
                'index_policy_version_bytes': exact_utf8_bytes(
                    RAG_INDEX_POLICY_VERSION
                ),
                'live_vector_count': live_vector_count,
                'mismatch_count_capped_at_20': mismatch_count,
                'pgvector_cosine_policy_version_bytes': exact_utf8_bytes(
                    RAG_COSINE_POLICY_VERSION
                ),
                'ready': ready,
                'tombstone_count': tombstone_count,
                'vector_index_generation': vector_index_generation,
            },
            secret=secret,
            schema_version='rag-serving-index-readiness:v1',
            policy_version=RAG_INDEX_POLICY_VERSION,
        )


def _generation_barrier_snapshot(
    generation: RagServingCorpusGeneration | None,
) -> tuple[object, ...] | None:
    if generation is None:
        return None
    return (
        generation.id,
        generation.corpus_generation,
        generation.vector_index_generation,
        generation.embedding_model,
        generation.embedding_dimensions,
        generation.index_policy_version,
        generation.pgvector_cosine_policy_version,
        generation.fingerprint_key_version,
        generation.fingerprint_key_material_verifier,
        generation.updated_at,
    )


def _parse_pgvector_text(value: str) -> tuple[float, ...]:
    if not value.startswith('[') or not value.endswith(']'):
        raise ValueError('live vector representation is invalid')
    body = value[1:-1]
    if not body:
        return ()
    return tuple(float(part) for part in body.split(','))
