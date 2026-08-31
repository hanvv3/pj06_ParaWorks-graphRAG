from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NoReturn

from sqlalchemy import event, select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyGenerationLockedContext,
    record_rag_corpus_lock,
)
from backend.app.core.config import Settings
from backend.app.models import RagServingCorpusGeneration, VectorServingTombstone

RAG_INDEX_POLICY_VERSION = 'rag-v2-serving-index:v1'
RAG_LEXICAL_COMPAT_VERSION = 'rag-keyword-lexical-compat:v1'
RAG_COSINE_POLICY_VERSION = 'pgvector-cosine-indexable:v1'


@dataclass(frozen=True, slots=True)
class RagIndexMutationResult:
    indexed_count: int
    skipped_count: int
    tombstoned_count: int
    saved_embedding_calls: int
    corpus_generation: int
    vector_index_generation: int


_GENERATION_CONTEXTS_INFO_KEY = 'paraworks_rag_generation_contexts'
_ARMED_CORPUS_MUTATION_INFO_KEY = 'paraworks_rag_armed_corpus_mutation'
_GENERATION_LISTENER_INFO_KEY = 'paraworks_rag_generation_listener'


@dataclass(frozen=True, slots=True, init=False)
class RagServingGenerationLockedContext:
    session_identity: int
    transaction_identity: int
    transaction_handle: object
    key_version: str
    key_material_verifier: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError('RAG generation contexts are minted only by the lock guard')

    def __copy__(self) -> NoReturn:
        raise TypeError('RAG generation contexts cannot be copied')

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError('RAG generation contexts cannot be copied')

    def __reduce__(self) -> NoReturn:
        raise TypeError('RAG generation contexts cannot be serialized')


@dataclass(slots=True)
class _ArmedCorpusMutation:
    context: RagServingGenerationLockedContext
    settings: Settings
    before_snapshot: tuple[tuple[str, ...], ...]
    vector_mutation: bool = False


def lock_rag_serving_generation(
    db: Session,
    *,
    settings: Settings,
    key_context: KeyGenerationLockedContext | None,
    for_update: bool = True,
) -> RagServingGenerationLockedContext:
    record_rag_corpus_lock(db, key_context, for_update=for_update)
    statement = (
        select(RagServingCorpusGeneration)
        .where(RagServingCorpusGeneration.id == 1)
        .execution_options(populate_existing=True)
    )
    if db.get_bind().dialect.name == 'postgresql':
        statement = statement.with_for_update(read=not for_update)
    row = db.scalar(statement)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    if row is None:
        if not for_update:
            raise RuntimeError('RAG serving generation singleton is unavailable')
        row = RagServingCorpusGeneration(
            id=1,
            corpus_generation=0,
            vector_index_generation=0,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
            index_policy_version=RAG_INDEX_POLICY_VERSION,
            pgvector_cosine_policy_version=RAG_COSINE_POLICY_VERSION,
            fingerprint_key_version=(
                settings.agent_runtime_fingerprint_key_version
            ),
            fingerprint_key_material_verifier=verifier,
        )
        db.add(row)
        db.flush([row])
    _validate_generation_identity(
        row,
        settings=settings,
        verifier=verifier,
    )
    transaction = db.get_transaction()
    if transaction is None:
        raise TypeError('RAG serving generation transaction is unavailable')
    context = object.__new__(RagServingGenerationLockedContext)
    object.__setattr__(context, 'session_identity', id(db))
    object.__setattr__(context, 'transaction_identity', id(transaction))
    object.__setattr__(context, 'transaction_handle', transaction)
    object.__setattr__(
        context,
        'key_version',
        settings.agent_runtime_fingerprint_key_version,
    )
    object.__setattr__(context, 'key_material_verifier', verifier)
    db.info.setdefault(_GENERATION_CONTEXTS_INFO_KEY, {})[id(context)] = context
    return context


def increment_vector_index_generation(
    db: Session,
    *,
    context: RagServingGenerationLockedContext,
) -> int:
    _validate_generation_context(db, context)
    row = db.get(RagServingCorpusGeneration, 1)
    if row is None:
        raise RuntimeError('RAG serving generation singleton is unavailable')
    row.vector_index_generation += 1
    row.updated_at = datetime.now(UTC)
    return row.vector_index_generation


def assert_rag_serving_generation_context(
    db: Session,
    context: RagServingGenerationLockedContext | None,
) -> None:
    if context is None:
        raise TypeError('A locked RAG generation context is required')
    _validate_generation_context(db, context)


def arm_corpus_generation_refresh(
    db: Session,
    *,
    settings: Settings,
    context: RagServingGenerationLockedContext,
) -> None:
    _validate_generation_context(db, context)
    current = db.info.get(_ARMED_CORPUS_MUTATION_INFO_KEY)
    if isinstance(current, _ArmedCorpusMutation):
        if current.context.transaction_handle is not context.transaction_handle:
            raise TypeError('RAG corpus mutation guard transaction is stale')
        return
    from backend.app.rag.lexical_projection import canonical_rag_corpus_snapshot

    db.info[_ARMED_CORPUS_MUTATION_INFO_KEY] = _ArmedCorpusMutation(
        context=context,
        settings=settings,
        before_snapshot=canonical_rag_corpus_snapshot(db, settings=settings),
    )
    _ensure_generation_listener(db)


def mark_rag_vector_index_mutation(db: Session) -> None:
    armed = db.info.get(_ARMED_CORPUS_MUTATION_INFO_KEY)
    if not isinstance(armed, _ArmedCorpusMutation):
        raise TypeError('RAG vector mutation requires an armed generation guard')
    _validate_generation_context(db, armed.context)
    armed.vector_mutation = True


def assert_corpus_generation_refresh_armed(db: Session) -> None:
    armed = db.info.get(_ARMED_CORPUS_MUTATION_INFO_KEY)
    if not isinstance(armed, _ArmedCorpusMutation):
        raise TypeError('Canonical mutation requires the RAG generation guard')
    _validate_generation_context(db, armed.context)


def advance_corpus_generation(
    db: Session,
    *,
    settings: Settings,
    context: RagServingGenerationLockedContext,
) -> int:
    """Advance the corpus and refresh its lexical projection atomically."""
    _validate_generation_context(db, context)
    row = db.get(RagServingCorpusGeneration, 1)
    if row is None:
        raise RuntimeError('RAG serving generation singleton is unavailable')
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    _validate_generation_identity(row, settings=settings, verifier=verifier)
    row.corpus_generation += 1
    row.updated_at = datetime.now(UTC)
    from backend.app.rag.lexical_projection import refresh_rag_lexical_projections

    refresh_rag_lexical_projections(
        db,
        settings=settings,
        corpus_generation=row.corpus_generation,
    )
    return row.corpus_generation


def _validate_generation_context(
    db: Session,
    context: RagServingGenerationLockedContext,
) -> None:
    if not isinstance(context, RagServingGenerationLockedContext):
        raise TypeError('A locked RAG generation context is required')
    if context.session_identity != id(db):
        raise TypeError('RAG generation context belongs to another session')
    transaction = db.get_transaction()
    if transaction is None or transaction is not context.transaction_handle:
        raise TypeError('RAG generation context transaction is no longer active')
    issued = db.info.get(_GENERATION_CONTEXTS_INFO_KEY, {}).get(id(context))
    if issued is not context:
        raise TypeError('RAG generation context was not issued for this session')


def _validate_generation_identity(
    row: RagServingCorpusGeneration,
    *,
    settings: Settings,
    verifier: str,
) -> None:
    expected = (
        settings.openai_embedding_model,
        settings.openai_embedding_dimensions,
        RAG_INDEX_POLICY_VERSION,
        RAG_COSINE_POLICY_VERSION,
        settings.agent_runtime_fingerprint_key_version,
        verifier,
    )
    actual = (
        row.embedding_model,
        row.embedding_dimensions,
        row.index_policy_version,
        row.pgvector_cosine_policy_version,
        row.fingerprint_key_version,
        row.fingerprint_key_material_verifier,
    )
    if actual != expected:
        raise RuntimeError('RAG serving generation identity mismatch')


def _ensure_generation_listener(db: Session) -> None:
    if db.info.get(_GENERATION_LISTENER_INFO_KEY):
        return

    def before_flush(
        session: Session,
        flush_context: object,
        instances: object,
    ) -> None:
        del flush_context, instances
        armed = session.info.get(_ARMED_CORPUS_MUTATION_INFO_KEY)
        if not isinstance(armed, _ArmedCorpusMutation):
            return
        if any(isinstance(row, VectorServingTombstone) for row in session.new):
            _validate_generation_context(session, armed.context)
            armed.vector_mutation = True

    def before_commit(session: Session) -> None:
        armed = session.info.get(_ARMED_CORPUS_MUTATION_INFO_KEY)
        if not isinstance(armed, _ArmedCorpusMutation):
            return
        if session.in_nested_transaction():
            return
        _validate_generation_context(session, armed.context)
        tombstone_created = any(
            isinstance(row, VectorServingTombstone) for row in session.new
        )
        from backend.app.rag.lexical_projection import canonical_rag_corpus_snapshot

        after_snapshot = canonical_rag_corpus_snapshot(
            session,
            settings=armed.settings,
        )
        if after_snapshot != armed.before_snapshot:
            advance_corpus_generation(
                session,
                settings=armed.settings,
                context=armed.context,
            )
        if armed.vector_mutation or tombstone_created:
            increment_vector_index_generation(session, context=armed.context)

    def after_transaction_end(session: Session, transaction: object) -> None:
        if getattr(transaction, 'parent', None) is None:
            session.info.pop(_ARMED_CORPUS_MUTATION_INFO_KEY, None)

    event.listen(db, 'before_flush', before_flush)
    event.listen(db, 'before_commit', before_commit)
    event.listen(db, 'after_transaction_end', after_transaction_end)
    db.info[_GENERATION_LISTENER_INFO_KEY] = True
