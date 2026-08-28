from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

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
        if db.get_bind().dialect.name == 'postgresql':
            statement = (
                GENERATION_BARRIER_EXCLUSIVE_SQL
                if exclusive
                else GENERATION_BARRIER_SHARED_SQL
            )
            db.execute(
                statement,
                {'lock_id': AUTO_REVIEW_KEY_GENERATION_LOCK_ID},
            )
            yield
            return
        with _SQLITE_PROCESS_LOCAL_LOCK:
            yield

    @staticmethod
    def lock_runtime_key_state(
        db: Session,
        *,
        for_update: bool = False,
    ) -> AutoReviewRuntimeKeyState | None:
        return db.scalar(build_runtime_key_state_lock(for_update=for_update))

    @staticmethod
    def acquire_projection_lock(db: Session) -> None:
        if db.get_bind().dialect.name == 'postgresql':
            db.execute(
                PROJECTION_LOCK_SQL,
                {'lock_id': TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID},
            )
