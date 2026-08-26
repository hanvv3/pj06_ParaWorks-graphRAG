from dataclasses import dataclass

import pytest
from psycopg.rows import dict_row
from pydantic import BaseModel, ValidationError

from backend.app.agent_runtime.checkpointing import (
    CheckpointReadiness,
    CheckpointRuntime,
    build_postgres_pool,
    build_postgres_saver,
    build_strict_checkpoint_serializer,
    checkpoint_tables_ready,
    resolve_checkpoint_mode,
    sqlalchemy_url_to_psycopg_dsn,
)
from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.state import ReviewGraphState
from backend.app.core.config import Settings


@dataclass
class _UnsafeDataclass:
    value: str


class _UnsafePydanticModel(BaseModel):
    value: str


class _FakePool:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def open(self, *, wait: bool) -> None:
        self.events.append('open')
        if wait:
            self.events.append('wait')

    def check(self) -> None:
        self.events.append('check')

    def connection(self) -> object:
        self.events.append('connection')
        return object()

    def close(self) -> None:
        self.events.append('close')


class _FakeSaver:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.setup_calls = 0

    def setup(self) -> None:
        self.events.append('setup')
        self.setup_calls += 1


def _postgres_checkpoint_settings(*, strict: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=None,
        paraworks_demo_database_url=None,
        database_url='postgresql+psycopg://runtime:secret@db/checkpoints',
        langgraph_review_v2_enabled=True,
        langgraph_strict_msgpack=strict,
    )


def _checkpoint_runtime(
    *,
    events: list[str],
    tables_ready: bool = True,
    strict: bool = True,
) -> tuple[CheckpointRuntime, _FakeSaver]:
    pool = _FakePool(events)
    saver = _FakeSaver(events)

    def pool_factory(dsn: str) -> _FakePool:
        assert dsn == 'postgresql://runtime:secret@db/checkpoints'
        return pool

    def saver_factory(pool_arg: object, serializer: object) -> _FakeSaver:
        assert pool_arg is pool
        assert serializer.pickle_fallback is False
        events.append('saver')
        return saver

    def table_probe(pool_arg: object) -> bool:
        assert pool_arg is pool
        pool.connection()
        return tables_ready

    return (
        CheckpointRuntime(
            _postgres_checkpoint_settings(strict=strict),
            pool_factory=pool_factory,  # type: ignore[arg-type]
            saver_factory=saver_factory,  # type: ignore[arg-type]
            table_probe=table_probe,  # type: ignore[arg-type]
        ),
        saver,
    )


def _valid_checkpoint_state() -> ReviewGraphState:
    return {
        'workflow_thread_id': 'thread-1',
        'graph_version': 'company-memory-review-v2.0',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_item_ids': [1, 2],
        'review_status_counts': {'pending_review': 2},
        'phase': 'created',
        'completed_nodes': ['validate_input'],
        'error_codes': [],
    }


def test_checkpoint_mode_matrix_preserves_disabled_smoke_and_postgres_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('PARAWORKS_DEMO_MODE', raising=False)
    monkeypatch.delenv('PARAWORKS_DATABASE_URL', raising=False)
    monkeypatch.delenv('PARAWORKS_DEMO_DATABASE_URL', raising=False)

    assert resolve_checkpoint_mode(
        Settings(_env_file=None, langgraph_review_v2_enabled=False)
    ) == 'disabled'
    assert resolve_checkpoint_mode(
        Settings(
            _env_file=None,
            paraworks_demo_mode=True,
            langgraph_review_v2_enabled=True,
        )
    ) == 'memory'
    assert resolve_checkpoint_mode(
        Settings(
            _env_file=None,
            database_url='sqlite://',
            langgraph_review_v2_enabled=True,
        )
    ) == 'memory'
    assert resolve_checkpoint_mode(
        Settings(
            _env_file=None,
            database_url='postgresql+psycopg://user:p%40ss@db/runtime',
            langgraph_review_v2_enabled=True,
        )
    ) == 'postgres'


def test_disabled_runtime_has_no_saver_pool_or_error() -> None:
    events: list[str] = []
    settings = Settings(
        _env_file=None,
        langgraph_review_v2_enabled=False,
    )
    runtime = CheckpointRuntime(
        settings,
        pool_factory=lambda _dsn: events.append('pool'),  # type: ignore[arg-type]
    )

    runtime.start()
    runtime.start()
    runtime.close()

    assert runtime.saver is None
    assert runtime.readiness == CheckpointReadiness(
        enabled=False,
        mode='disabled',
        ready=False,
        durable=False,
        checkpoint_store='none',
    )
    assert events == []


@pytest.mark.parametrize(
    ('demo_mode', 'database_url'),
    [
        (True, 'postgresql+psycopg://runtime:secret@db/checkpoints'),
        (False, 'sqlite:///runtime.db'),
    ],
)
def test_memory_runtime_owns_one_strict_saver_per_runtime(
    demo_mode: bool,
    database_url: str,
) -> None:
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=demo_mode,
        paraworks_database_url=None,
        paraworks_demo_database_url=None,
        database_url=database_url,
        langgraph_review_v2_enabled=True,
    )
    first = CheckpointRuntime(settings)
    second = CheckpointRuntime(settings)

    first.start()
    first.start()
    second.start()

    assert first.saver is not None
    assert second.saver is not None
    assert first.saver is not second.saver
    assert first.saver.serde.pickle_fallback is False
    assert first.readiness == CheckpointReadiness(
        enabled=True,
        mode='memory',
        ready=True,
        durable=False,
        checkpoint_store='memory',
    )


def test_postgres_runtime_opens_checks_and_builds_saver_without_setup() -> None:
    events: list[str] = []
    runtime, saver = _checkpoint_runtime(events=events)

    runtime.start()
    runtime.start()

    assert runtime.saver is saver
    assert runtime.readiness == CheckpointReadiness(
        enabled=True,
        mode='postgres',
        ready=True,
        durable=True,
        checkpoint_store='postgres',
    )
    assert saver.setup_calls == 0
    assert events == ['open', 'wait', 'check', 'connection', 'saver']


def test_postgres_runtime_missing_tables_fails_closed_without_memory_fallback() -> None:
    events: list[str] = []
    runtime, saver = _checkpoint_runtime(events=events, tables_ready=False)

    runtime.start()

    assert runtime.saver is None
    assert runtime.readiness == CheckpointReadiness(
        enabled=True,
        mode='postgres',
        ready=False,
        durable=False,
        checkpoint_store='postgres',
        error_code='checkpoint_unavailable',
    )
    assert saver.setup_calls == 0
    assert events == ['open', 'wait', 'check', 'connection', 'close']


def test_postgres_runtime_requires_strict_serializer_before_pool_access() -> None:
    events: list[str] = []
    runtime, saver = _checkpoint_runtime(events=events, strict=False)

    runtime.start()

    assert runtime.saver is None
    assert runtime.readiness == CheckpointReadiness(
        enabled=True,
        mode='postgres',
        ready=False,
        durable=False,
        checkpoint_store='none',
        error_code='strict_serializer_required',
    )
    assert saver.setup_calls == 0
    assert events == []


def test_postgres_runtime_close_is_idempotent_and_closes_pool_once() -> None:
    events: list[str] = []
    runtime, saver = _checkpoint_runtime(events=events)
    runtime.start()

    runtime.close()
    runtime.close()

    assert saver.setup_calls == 0
    assert runtime.saver is None
    assert events == ['open', 'wait', 'check', 'connection', 'saver', 'close']


def test_sqlalchemy_url_is_converted_to_psycopg_dsn_without_decoding_password() -> None:
    assert sqlalchemy_url_to_psycopg_dsn(
        'postgresql+psycopg://user:p%40ss@db/runtime'
    ) == 'postgresql://user:p%40ss@db/runtime'


def test_sqlite_url_rejection_does_not_echo_credentials_or_input_url() -> None:
    database_url = 'sqlite:///password-should-not-leak.db'

    with pytest.raises(ValueError) as exc_info:
        sqlalchemy_url_to_psycopg_dsn(database_url)

    error = str(exc_info.value)
    assert error == 'unsupported checkpoint database URL'
    assert 'password-should-not-leak' not in error
    assert database_url not in error


def test_malformed_database_url_rejection_does_not_echo_parser_input() -> None:
    database_url = 'postgresql+psycopg://user:credential-marker@db:bad/runtime'

    with pytest.raises(ValueError) as exc_info:
        sqlalchemy_url_to_psycopg_dsn(database_url)

    error = str(exc_info.value)
    assert error == 'unsupported checkpoint database URL'
    assert 'credential-marker' not in error
    assert database_url not in error


def test_enabled_checkpoint_mode_rejects_unsupported_backend_without_secret() -> None:
    database_url = 'mysql://user:credential-marker@db/runtime'
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        database_url=database_url,
        langgraph_review_v2_enabled=True,
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_checkpoint_mode(settings)

    error = str(exc_info.value)
    assert error == 'unsupported checkpoint database URL'
    assert 'credential-marker' not in error
    assert database_url not in error


def test_postgres_pool_uses_checkpoint_safe_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import checkpointing

    calls: list[dict[str, object]] = []
    expected_pool = object()

    def fake_connection_pool(*, conninfo: str, **kwargs: object) -> object:
        calls.append({'conninfo': conninfo, **kwargs})
        return expected_pool

    monkeypatch.setattr(checkpointing, 'ConnectionPool', fake_connection_pool)

    pool = build_postgres_pool('postgresql://user:secret@db/runtime')

    assert pool is expected_pool
    assert calls == [
        {
            'conninfo': 'postgresql://user:secret@db/runtime',
            'kwargs': {'autocommit': True, 'row_factory': dict_row},
            'min_size': 1,
            'max_size': 4,
            'open': False,
        }
    ]


def test_postgres_saver_factory_does_not_open_or_setup_the_saver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import checkpointing

    calls: list[tuple[object, object]] = []

    class FakePostgresSaver:
        setup_calls = 0

        def __init__(self, pool: object, *, serde: object) -> None:
            calls.append((pool, serde))

        def setup(self) -> None:
            self.setup_calls += 1

    monkeypatch.setattr(checkpointing, 'PostgresSaver', FakePostgresSaver)
    pool = object()
    serializer = build_strict_checkpoint_serializer()

    saver = build_postgres_saver(pool, serializer)  # type: ignore[arg-type]

    assert isinstance(saver, FakePostgresSaver)
    assert calls == [(pool, serializer)]
    assert saver.setup_calls == 0


@pytest.mark.parametrize(
    ('row', 'expected'),
    [
        (
            {
                'migrations': True,
                'checkpoints': True,
                'blobs': True,
                'writes': True,
                'task_path': True,
            },
            True,
        ),
        (
            {
                'migrations': True,
                'checkpoints': True,
                'blobs': True,
                'writes': True,
                'task_path': False,
            },
            False,
        ),
        (None, False),
    ],
)
def test_checkpoint_readiness_uses_the_resolved_writes_relation(
    row: dict[str, bool] | None,
    expected: bool,
) -> None:
    statements: list[str] = []

    class FakeCursor:
        def __enter__(self) -> 'FakeCursor':
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: str) -> None:
            statements.append(statement)

        def fetchone(self) -> dict[str, bool] | None:
            return row

    class FakeConnection:
        def __enter__(self) -> 'FakeConnection':
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    class FakePool:
        def connection(self) -> FakeConnection:
            return FakeConnection()

    assert checkpoint_tables_ready(FakePool()) is expected  # type: ignore[arg-type]
    assert len(statements) == 1
    normalized_sql = ' '.join(statements[0].split())
    assert 'FROM pg_catalog.pg_attribute' in normalized_sql
    assert "to_regclass('checkpoint_migrations')" in normalized_sql
    assert "to_regclass('checkpoints')" in normalized_sql
    assert "to_regclass('checkpoint_blobs')" in normalized_sql
    assert "to_regclass('checkpoint_writes')" in normalized_sql
    assert "attrelid = to_regclass('checkpoint_writes')" in normalized_sql
    assert "attname = 'task_path'" in normalized_sql
    assert 'NOT attisdropped' in normalized_sql
    assert 'information_schema.columns' not in normalized_sql


@pytest.mark.parametrize('retention_days', [0, 3651])
def test_checkpoint_retention_days_stay_within_operator_bounds(
    retention_days: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            langgraph_checkpoint_retention_days=retention_days,
        )


def test_strict_serializer_round_trips_only_json_safe_checkpoint_state() -> None:
    serializer = build_strict_checkpoint_serializer()
    state = _valid_checkpoint_state()

    serialized = serializer.dumps_typed(state)

    assert serializer.pickle_fallback is False
    assert serializer.loads_typed(serialized) == state


def test_strict_serializer_rejects_arbitrary_python_objects() -> None:
    serializer = build_strict_checkpoint_serializer()

    with pytest.raises(ValueError) as exc_info:
        serializer.dumps_typed(object())

    assert str(exc_info.value) == 'checkpoint value must be JSON-safe'


@pytest.mark.parametrize(
    'unsafe_value',
    [
        b'credential-marker',
        bytearray(b'credential-marker'),
        RuntimeError('credential-marker'),
        _UnsafeDataclass(value='credential-marker'),
        _UnsafePydanticModel(value='credential-marker'),
        (1, 2),
        float('inf'),
        {'nested': [RuntimeError('credential-marker')]},
        {1: 'credential-marker'},
    ],
)
def test_strict_serializer_rejects_non_json_values_without_echoing_secrets(
    unsafe_value: object,
) -> None:
    serializer = build_strict_checkpoint_serializer()

    with pytest.raises(ValueError) as exc_info:
        serializer.dumps_typed(unsafe_value)

    error = str(exc_info.value)
    assert error == 'checkpoint value must be JSON-safe'
    assert 'credential-marker' not in error


def test_strict_serializer_rejects_cyclic_or_excessively_nested_values() -> None:
    serializer = build_strict_checkpoint_serializer()
    recursive: list[object] = []
    recursive.append(recursive)
    nested: object = None
    for _ in range(2_000):
        nested = [nested]

    for unsafe_value in (recursive, nested):
        with pytest.raises(
            ValueError,
            match='^checkpoint value must be JSON-safe$',
        ):
            serializer.dumps_typed(unsafe_value)


def test_strict_serializer_preserves_the_langgraph_interrupt_transport() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
    from typing_extensions import TypedDict

    class InterruptState(TypedDict):
        resolution: str | None

    def await_review(_state: InterruptState) -> dict[str, str]:
        return {'resolution': interrupt({'kind': 'review_resolution'})}

    builder = StateGraph(InterruptState)
    builder.add_node('await_review', await_review)
    builder.add_edge(START, 'await_review')
    builder.add_edge('await_review', END)
    serializer = build_strict_checkpoint_serializer()
    graph = builder.compile(checkpointer=InMemorySaver(serde=serializer))
    config = {'configurable': {'thread_id': 'strict-serializer-interrupt'}}

    paused = graph.invoke({'resolution': None}, config, durability='sync')
    resumed = graph.invoke(Command(resume='approved'), config, durability='sync')

    assert paused['__interrupt__'][0].value == {'kind': 'review_resolution'}
    assert resumed['resolution'] == 'approved'


def test_strict_serializer_validates_nested_interrupt_values() -> None:
    from langgraph.types import Interrupt

    serializer = build_strict_checkpoint_serializer()
    envelope = (Interrupt(value=RuntimeError('credential-marker'), id='interrupt-1'),)

    with pytest.raises(ValueError) as exc_info:
        serializer.dumps_typed(envelope)

    error = str(exc_info.value)
    assert error == 'checkpoint value must be JSON-safe'
    assert 'credential-marker' not in error


def test_production_rejects_the_local_default_fingerprint_secret() -> None:
    settings = Settings(_env_file=None, paraworks_env='production')

    with pytest.raises(ValueError, match='dedicated fingerprint secret'):
        fingerprint_secret_bytes(settings)


def test_fingerprint_configuration_rejects_an_empty_key_version() -> None:
    settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='production-secret',
        agent_runtime_fingerprint_key_version=' ',
    )

    with pytest.raises(ValueError, match='configuration is incomplete'):
        fingerprint_secret_bytes(settings)


def test_production_fingerprint_configuration_returns_secret_and_key_version() -> None:
    settings = Settings(
        _env_file=None,
        paraworks_env='production',
        agent_runtime_fingerprint_secret='production-secret',
        agent_runtime_fingerprint_key_version='key-v2',
    )

    assert fingerprint_secret_bytes(settings) == (b'production-secret', 'key-v2')
