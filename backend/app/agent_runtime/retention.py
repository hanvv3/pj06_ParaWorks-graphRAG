from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from psycopg_pool import ConnectionPool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.checkpointing import (
    CheckpointUnavailableError,
    build_postgres_pool,
    sqlalchemy_url_to_psycopg_dsn,
)
from backend.app.core.config import Settings
from backend.app.models import AgentWorkflowThread

TERMINAL_PRUNABLE_STATUSES = (
    'completed',
    'needs_more_evidence',
    'failed',
    'cancelled',
)
_CHECKPOINT_UNAVAILABLE = 'checkpoint_unavailable'


@dataclass(frozen=True)
class CheckpointPruneResult:
    selected_thread_count: int
    deleted_write_count: int
    deleted_blob_count: int
    deleted_checkpoint_count: int
    cutoff: datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def _default_session_factory() -> Session:
    from backend.app.db.session import SessionLocal

    return SessionLocal()


def prune_expired_checkpoints(
    settings: Settings,
    *,
    session_factory: Callable[[], Session] = _default_session_factory,
    pool_factory: Callable[[str], ConnectionPool] = build_postgres_pool,
    now: Callable[[], datetime] = utc_now,
    limit: int = 100,
) -> CheckpointPruneResult:
    if limit < 1 or limit > 1000:
        raise ValueError('checkpoint prune limit must be between 1 and 1000')

    pool: object | None = None
    primary_error = False
    try:
        dsn = sqlalchemy_url_to_psycopg_dsn(settings.resolved_database_url())
        cutoff = now() - timedelta(
            days=settings.langgraph_checkpoint_retention_days
        )
        terminal_at = func.coalesce(
            AgentWorkflowThread.completed_at,
            AgentWorkflowThread.cancelled_at,
            AgentWorkflowThread.updated_at,
        )
        with session_factory() as session:
            checkpoint_thread_ids = list(
                session.scalars(
                    select(AgentWorkflowThread.checkpoint_thread_id)
                    .where(
                        AgentWorkflowThread.status.in_(
                            TERMINAL_PRUNABLE_STATUSES
                        )
                    )
                    .where(AgentWorkflowThread.checkpoint_store == 'postgres')
                    .where(terminal_at < cutoff)
                    .order_by(terminal_at, AgentWorkflowThread.thread_id)
                    .limit(limit)
                )
            )

        if not checkpoint_thread_ids:
            return CheckpointPruneResult(0, 0, 0, 0, cutoff)

        pool = pool_factory(dsn)
        pool.open(wait=True)
        with pool.connection() as connection:
            deleted_writes = connection.execute(
                'DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)',
                (checkpoint_thread_ids,),
            ).rowcount
            deleted_blobs = connection.execute(
                'DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)',
                (checkpoint_thread_ids,),
            ).rowcount
            deleted_checkpoints = connection.execute(
                'DELETE FROM checkpoints WHERE thread_id = ANY(%s)',
                (checkpoint_thread_ids,),
            ).rowcount
            connection.commit()

        return CheckpointPruneResult(
            selected_thread_count=len(checkpoint_thread_ids),
            deleted_write_count=deleted_writes,
            deleted_blob_count=deleted_blobs,
            deleted_checkpoint_count=deleted_checkpoints,
            cutoff=cutoff,
        )
    except Exception:
        primary_error = True
        raise CheckpointUnavailableError(_CHECKPOINT_UNAVAILABLE) from None
    finally:
        if pool is not None:
            try:
                pool.close()
            except Exception:
                if not primary_error:
                    raise CheckpointUnavailableError(
                        _CHECKPOINT_UNAVAILABLE
                    ) from None
