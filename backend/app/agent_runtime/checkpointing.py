from dataclasses import dataclass
from typing import Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from backend.app.core.config import Settings

CheckpointMode = Literal['disabled', 'memory', 'postgres']


class CheckpointUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointReadiness:
    enabled: bool
    mode: CheckpointMode
    ready: bool
    durable: bool
    checkpoint_store: str
    error_code: str | None = None


def resolve_checkpoint_mode(settings: Settings) -> CheckpointMode:
    if not settings.langgraph_review_v2_enabled:
        return 'disabled'
    try:
        backend = make_url(settings.resolved_database_url()).get_backend_name()
    except (ArgumentError, ValueError):
        raise ValueError('unsupported checkpoint database URL') from None
    if settings.paraworks_demo_mode or backend == 'sqlite':
        return 'memory'
    if backend == 'postgresql':
        return 'postgres'
    raise ValueError('unsupported checkpoint database URL')


def sqlalchemy_url_to_psycopg_dsn(database_url: str) -> str:
    try:
        url = make_url(database_url)
        if url.get_backend_name() != 'postgresql':
            raise ValueError
        return url.set(drivername='postgresql').render_as_string(
            hide_password=False,
        )
    except (ArgumentError, ValueError):
        raise ValueError('unsupported checkpoint database URL') from None


def build_strict_checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=None,
    )


def build_postgres_pool(dsn: str) -> ConnectionPool:
    return ConnectionPool(
        conninfo=dsn,
        kwargs={'autocommit': True, 'row_factory': dict_row},
        min_size=1,
        max_size=4,
        open=False,
    )


def build_postgres_saver(
    pool: ConnectionPool,
    serializer: SerializerProtocol,
) -> PostgresSaver:
    return PostgresSaver(pool, serde=serializer)


def checkpoint_tables_ready(pool: ConnectionPool) -> bool:
    with pool.connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
                SELECT
                    to_regclass('checkpoint_migrations') IS NOT NULL AS migrations,
                    to_regclass('checkpoints') IS NOT NULL AS checkpoints,
                    to_regclass('checkpoint_blobs') IS NOT NULL AS blobs,
                    to_regclass('checkpoint_writes') IS NOT NULL AS writes,
                    EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'checkpoint_writes'
                          AND column_name = 'task_path'
                    ) AS task_path
                """
        )
        row = cursor.fetchone()
    return bool(row) and all(bool(row[key]) for key in row)
