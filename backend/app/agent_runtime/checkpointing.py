import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from backend.app.core.config import Settings

CheckpointMode = Literal['disabled', 'memory', 'postgres']
_CHECKPOINT_JSON_ERROR = 'checkpoint value must be JSON-safe'
_CHECKPOINT_RUNTIME_CLOSED_ERROR = 'checkpoint runtime is closed'


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


def _validate_json_checkpoint_value(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(_CHECKPOINT_JSON_ERROR)
        return
    if type(value) is list:
        for item in value:
            _validate_json_checkpoint_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(_CHECKPOINT_JSON_ERROR)
            _validate_json_checkpoint_value(item)
        return
    raise ValueError(_CHECKPOINT_JSON_ERROR)


def _validate_checkpoint_serialization_input(value: object) -> None:
    # LangGraph 1.2.11 persists pending interrupts as tuple[Interrupt, ...].
    # This is the sole non-JSON transport envelope; its application value stays JSON-only.
    if type(value) is tuple:
        if not value or any(type(item) is not Interrupt for item in value):
            raise ValueError(_CHECKPOINT_JSON_ERROR)
        for item in value:
            if type(item.id) is not str:
                raise ValueError(_CHECKPOINT_JSON_ERROR)
            _validate_json_checkpoint_value(item.value)
        return
    _validate_json_checkpoint_value(value)


class _StrictCheckpointSerializer(JsonPlusSerializer):
    def dumps_typed(self, obj: object) -> tuple[str, bytes]:
        try:
            _validate_checkpoint_serialization_input(obj)
        except RecursionError:
            raise ValueError(_CHECKPOINT_JSON_ERROR) from None
        return super().dumps_typed(obj)


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
    return _StrictCheckpointSerializer(
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
                        FROM pg_catalog.pg_attribute
                        WHERE attrelid = to_regclass('checkpoint_writes')
                          AND attname = 'task_path'
                          AND NOT attisdropped
                    ) AS task_path
                """
        )
        row = cursor.fetchone()
    return bool(row) and all(bool(row[key]) for key in row)


class CheckpointRuntime:
    def __init__(
        self,
        settings: Settings,
        *,
        pool_factory: Callable[[str], ConnectionPool] = build_postgres_pool,
        saver_factory: Callable[
            [object, SerializerProtocol], BaseCheckpointSaver
        ] = build_postgres_saver,
        table_probe: Callable[[object], bool] = checkpoint_tables_ready,
    ) -> None:
        self._settings = settings
        self._mode = resolve_checkpoint_mode(settings)
        self._pool_factory = pool_factory
        self._saver_factory = saver_factory
        self._table_probe = table_probe
        self._pool: object | None = None
        self._saver: BaseCheckpointSaver | None = None
        self._started = False
        self._closed = False
        self._readiness = CheckpointReadiness(
            enabled=self._mode != 'disabled',
            mode=self._mode,
            ready=False,
            durable=False,
            checkpoint_store='none',
        )

    @property
    def saver(self) -> BaseCheckpointSaver | None:
        return self._saver

    @property
    def readiness(self) -> CheckpointReadiness:
        return self._readiness

    def start(self) -> None:
        if self._closed:
            raise CheckpointUnavailableError(_CHECKPOINT_RUNTIME_CLOSED_ERROR)
        if self._started:
            return
        self._started = True
        serializer = build_strict_checkpoint_serializer()
        if self._mode == 'disabled':
            return
        if self._mode == 'memory':
            self._saver = InMemorySaver(serde=serializer)
            self._readiness = CheckpointReadiness(
                enabled=True,
                mode='memory',
                ready=True,
                durable=False,
                checkpoint_store='memory',
            )
            return
        if not self._settings.langgraph_strict_msgpack:
            self._readiness = replace(
                self._readiness,
                error_code='strict_serializer_required',
            )
            return
        try:
            dsn = sqlalchemy_url_to_psycopg_dsn(
                self._settings.resolved_database_url()
            )
            self._pool = self._pool_factory(dsn)
            self._pool.open(wait=True)
            self._pool.check()
            if not self._table_probe(self._pool):
                raise CheckpointUnavailableError
            self._saver = self._saver_factory(self._pool, serializer)
            self._readiness = CheckpointReadiness(
                enabled=True,
                mode='postgres',
                ready=True,
                durable=True,
                checkpoint_store='postgres',
            )
        except Exception:
            self._readiness = CheckpointReadiness(
                enabled=True,
                mode='postgres',
                ready=False,
                durable=False,
                checkpoint_store='postgres',
                error_code='checkpoint_unavailable',
            )
            self._clear_resources()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        error_code = self._readiness.error_code
        if self._readiness.enabled and error_code is None:
            error_code = 'checkpoint_unavailable'
        self._readiness = replace(
            self._readiness,
            ready=False,
            durable=False,
            error_code=error_code,
        )
        self._clear_resources()

    def _clear_resources(self) -> None:
        pool = self._pool
        self._pool = None
        self._saver = None
        if pool is not None:
            with suppress(Exception):
                pool.close()


def build_checkpoint_runtime(settings: Settings) -> CheckpointRuntime:
    return CheckpointRuntime(settings)
