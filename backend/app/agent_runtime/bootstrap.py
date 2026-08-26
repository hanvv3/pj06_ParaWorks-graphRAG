from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.checkpointing import (
    build_postgres_pool,
    build_postgres_saver,
    build_strict_checkpoint_serializer,
    sqlalchemy_url_to_psycopg_dsn,
)
from backend.app.core.config import Settings
from backend.app.models import AgentRuntimeSchemaVersion


class CheckpointBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    component: str
    package_name: str
    package_version: str
    schema_revision: int
    applied_at: datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def _default_session_factory() -> Session:
    from backend.app.db.session import SessionLocal

    return SessionLocal()


def bootstrap_langgraph_checkpointer(
    settings: Settings,
    *,
    backup_confirmed: bool,
    pool_factory: Callable[[str], object] = build_postgres_pool,
    saver_factory: Callable[
        [object, SerializerProtocol], PostgresSaver
    ] = build_postgres_saver,
    session_factory: Callable[[], Session] = _default_session_factory,
    now: Callable[[], datetime] = utc_now,
) -> BootstrapResult:
    if not backup_confirmed:
        raise CheckpointBootstrapError('database backup confirmation is required')
    if not settings.langgraph_strict_msgpack:
        raise CheckpointBootstrapError('strict msgpack is required')

    pool: object | None = None
    primary_error: CheckpointBootstrapError | None = None
    try:
        dsn = sqlalchemy_url_to_psycopg_dsn(settings.resolved_database_url())
        pool = pool_factory(dsn)
        pool.open(wait=True)
        saver = saver_factory(pool, build_strict_checkpoint_serializer())
        saver.setup()
        expected_revision = len(PostgresSaver.MIGRATIONS) - 1
        with pool.connection() as connection:
            row = connection.execute(
                'SELECT MAX(v) AS v FROM checkpoint_migrations'
            ).fetchone()
        if row is None or row['v'] != expected_revision:
            raise CheckpointBootstrapError('checkpoint schema revision mismatch')

        applied_at = now()
        package_version = version('langgraph-checkpoint-postgres')
        with session_factory() as session:
            record = session.scalar(
                select(AgentRuntimeSchemaVersion).where(
                    AgentRuntimeSchemaVersion.component == 'langgraph_checkpoint'
                )
            )
            if record is None:
                record = AgentRuntimeSchemaVersion(component='langgraph_checkpoint')
                session.add(record)
            record.package_name = 'langgraph-checkpoint-postgres'
            record.package_version = package_version
            record.schema_revision = expected_revision
            record.applied_at = applied_at
            session.commit()

        return BootstrapResult(
            component='langgraph_checkpoint',
            package_name='langgraph-checkpoint-postgres',
            package_version=package_version,
            schema_revision=expected_revision,
            applied_at=applied_at,
        )
    except CheckpointBootstrapError as exc:
        primary_error = exc
        raise
    except Exception:
        primary_error = CheckpointBootstrapError('checkpoint bootstrap failed')
        raise primary_error from None
    finally:
        if pool is not None:
            try:
                pool.close()
            except Exception:
                if primary_error is None:
                    raise CheckpointBootstrapError(
                        'checkpoint bootstrap failed'
                    ) from None
