from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Literal, NoReturn

from sqlalchemy import Select, select, text
from sqlalchemy.orm import Session

from backend.app.models.auto_review import AutoReviewRuntimeKeyState

AUTO_REVIEW_KEY_GENERATION_LOCK_ID = 1066041229503628369
TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID = -2972884933094306491

GENERATION_BARRIER_SHARED_SQL = text(
    'SELECT pg_advisory_xact_lock_shared(:lock_id)'
)
GENERATION_BARRIER_EXCLUSIVE_SQL = text(
    'SELECT pg_advisory_xact_lock(:lock_id)'
)
PROJECTION_LOCK_SQL = text('SELECT pg_advisory_xact_lock(:lock_id)')

AUTO_REVIEW_GLOBAL_LOCK_ORDER = (
    'generation_barrier',
    'runtime_key_state',
    'rag_serving_corpus_generation',
    'provider_safety_states',
    'fingerprint_projection',
    'rollout_states',
    'sources',
    'workflow_threads',
    'calls_and_children',
    'review_items',
    'promotion_decisions_and_audits',
    'approval_and_evidence_links',
    'trusted_knowledge',
    'document_locks',
    'tombstones_and_vector_state',
)

_SQLITE_PROCESS_LOCAL_LOCK = RLock()
_LOCK_ORDER_INFO_KEY = 'paraworks_c5_keyed_lock_order'
_CONTEXTS_INFO_KEY = 'paraworks_c5_keyed_contexts'
_LATEST_CONTEXT_INFO_KEY = 'paraworks_c5_latest_keyed_context'
_RUNTIME_ABSENT = object()


def sqlite_keyed_mutation_mutex() -> RLock:
    """Return the single never-replaced SQLite C.5/RAG mutation mutex."""
    return _SQLITE_PROCESS_LOCAL_LOCK


@dataclass(frozen=True, slots=True, init=False)
class KeyGenerationLockedContext:
    session_identity: int
    generation: int
    key_version: str
    material_verifier: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError('Key-generation contexts are minted only by the lock guard')

    def __copy__(self) -> NoReturn:
        raise TypeError('Key-generation contexts cannot be copied')

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError('Key-generation contexts cannot be copied')

    def __reduce__(self) -> NoReturn:
        raise TypeError('Key-generation contexts cannot be serialized')


def _order(db: Session) -> list[str]:
    return db.info.setdefault(_LOCK_ORDER_INFO_KEY, [])


def _record_next(db: Session, step: str, *, allowed_previous: tuple[str, ...]) -> None:
    order = _order(db)
    previous = order[-1] if order else None
    if previous not in allowed_previous:
        raise RuntimeError('C.5 keyed mutation lock order violation')
    order.append(step)


def acquire_generation_shared(db: Session) -> None:
    _record_next(db, 'generation_shared', allowed_previous=(None,))
    if db.get_bind().dialect.name == 'postgresql':
        db.execute(
            GENERATION_BARRIER_SHARED_SQL,
            {'lock_id': AUTO_REVIEW_KEY_GENERATION_LOCK_ID},
        )


def acquire_generation_exclusive(db: Session) -> None:
    _record_next(db, 'generation_exclusive', allowed_previous=(None,))
    if db.get_bind().dialect.name == 'postgresql':
        db.execute(
            GENERATION_BARRIER_EXCLUSIVE_SQL,
            {'lock_id': AUTO_REVIEW_KEY_GENERATION_LOCK_ID},
        )


def lock_runtime_state(
    db: Session,
    *,
    mode: Literal['share', 'update'] = 'share',
) -> KeyGenerationLockedContext | None:
    if mode not in {'share', 'update'}:
        raise ValueError('runtime key state lock mode is unsupported')
    expected = 'generation_shared' if mode == 'share' else 'generation_exclusive'
    _record_next(db, f'runtime_{mode}', allowed_previous=(expected,))
    runtime = db.scalar(build_runtime_key_state_lock(for_update=mode == 'update'))
    if runtime is None:
        db.info[_LATEST_CONTEXT_INFO_KEY] = _RUNTIME_ABSENT
        return None
    context = object.__new__(KeyGenerationLockedContext)
    object.__setattr__(context, 'session_identity', id(db))
    object.__setattr__(context, 'generation', runtime.generation)
    object.__setattr__(context, 'key_version', runtime.fingerprint_key_version)
    object.__setattr__(
        context,
        'material_verifier',
        runtime.fingerprint_key_material_verifier,
    )
    db.info.setdefault(_CONTEXTS_INFO_KEY, {})[id(context)] = context
    db.info[_LATEST_CONTEXT_INFO_KEY] = context
    return context


def acquire_projection(
    db: Session,
    context: KeyGenerationLockedContext | None,
) -> None:
    issued = db.info.get(_LATEST_CONTEXT_INFO_KEY)
    if context is None:
        if issued is not _RUNTIME_ABSENT:
            raise TypeError('A locked runtime identity context is required')
    elif (
        context.session_identity != id(db)
        or db.info.get(_CONTEXTS_INFO_KEY, {}).get(id(context)) is not context
        or issued is not context
    ):
        raise TypeError('Key-generation context was not issued for this session')
    _record_next(
        db,
        'projection',
        allowed_previous=(
            'runtime_share',
            'runtime_update',
            'rag_corpus_share',
            'rag_corpus_update',
        ),
    )
    if db.get_bind().dialect.name == 'postgresql':
        db.execute(
            PROJECTION_LOCK_SQL,
            {'lock_id': TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID},
        )


def record_rag_corpus_lock(
    db: Session,
    context: KeyGenerationLockedContext | None,
    *,
    for_update: bool,
) -> None:
    issued = db.info.get(_LATEST_CONTEXT_INFO_KEY)
    if context is None:
        if issued is not _RUNTIME_ABSENT:
            raise TypeError('A locked runtime identity context is required')
    elif (
        context.session_identity != id(db)
        or db.info.get(_CONTEXTS_INFO_KEY, {}).get(id(context)) is not context
        or issued is not context
    ):
        raise TypeError('Key-generation context was not issued for this session')
    _record_next(
        db,
        'rag_corpus_update' if for_update else 'rag_corpus_share',
        allowed_previous=('runtime_share', 'runtime_update'),
    )


def build_runtime_key_state_lock(*, for_update: bool) -> Select:
    statement = select(AutoReviewRuntimeKeyState).where(
        AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
    )
    if for_update:
        return statement.with_for_update()
    return statement.with_for_update(read=True)


class KeyedMutationGuard:
    """The sole fixed-lock API for C.5 keyed database mutations."""

    @staticmethod
    @contextmanager
    def generation_barrier(
        db: Session,
        *,
        exclusive: bool = False,
    ) -> Iterator[None]:
        db.info.pop(_LOCK_ORDER_INFO_KEY, None)
        acquisition = acquire_generation_exclusive if exclusive else acquire_generation_shared
        if db.get_bind().dialect.name == 'postgresql':
            acquisition(db)
            try:
                yield
            finally:
                db.info.pop(_LOCK_ORDER_INFO_KEY, None)
                db.info.pop(_CONTEXTS_INFO_KEY, None)
                db.info.pop(_LATEST_CONTEXT_INFO_KEY, None)
            return
        with _SQLITE_PROCESS_LOCAL_LOCK:
            acquisition(db)
            try:
                yield
            finally:
                db.info.pop(_LOCK_ORDER_INFO_KEY, None)
                db.info.pop(_CONTEXTS_INFO_KEY, None)
                db.info.pop(_LATEST_CONTEXT_INFO_KEY, None)

    @staticmethod
    def lock_runtime_key_state(
        db: Session,
        *,
        for_update: bool = False,
    ) -> AutoReviewRuntimeKeyState | None:
        mode: Literal['share', 'update'] = 'update' if for_update else 'share'
        context = lock_runtime_state(db, mode=mode)
        if context is None:
            return None
        return db.scalar(
            select(AutoReviewRuntimeKeyState).where(
                AutoReviewRuntimeKeyState.component
                == 'auto_review_trust_promotion'
            )
        )

    @staticmethod
    def acquire_projection_lock(db: Session) -> None:
        order = _order(db)
        if not order or order[-1] not in {'runtime_share', 'runtime_update'}:
            raise RuntimeError('runtime key state must be locked before projection')
        issued = db.info.get(_LATEST_CONTEXT_INFO_KEY)
        acquire_projection(
            db,
            issued if isinstance(issued, KeyGenerationLockedContext) else None,
        )
