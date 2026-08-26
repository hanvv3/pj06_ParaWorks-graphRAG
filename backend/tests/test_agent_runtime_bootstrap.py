from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from backend.app.agent_runtime.bootstrap import (
    BootstrapResult,
    CheckpointBootstrapError,
    bootstrap_langgraph_checkpointer,
)
from backend.app.core.config import Settings
from backend.app.models import AgentRuntimeSchemaVersion
from scripts import bootstrap_langgraph_checkpointer as bootstrap_command

_PACKAGE_VERSION = '3.1.2-test'
_EXPECTED_REVISION = len(PostgresSaver.MIGRATIONS) - 1
_APPLIED_AT = datetime(2026, 8, 26, 9, 30, tzinfo=UTC)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP_SCRIPT = _REPO_ROOT / 'scripts' / 'bootstrap_langgraph_checkpointer.py'


class _FakeQueryResult:
    def __init__(self, revision: int | None) -> None:
        self._revision = revision

    def fetchone(self) -> dict[str, int | None]:
        return {'v': self._revision}


class _FakeConnection:
    def __init__(self, events: list[str], revision: int | None) -> None:
        self._events = events
        self._revision = revision

    def __enter__(self) -> _FakeConnection:
        self._events.append('connection_enter')
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append('connection_exit')

    def execute(self, statement: str) -> _FakeQueryResult:
        assert statement == 'SELECT MAX(v) AS v FROM checkpoint_migrations'
        self._events.append('verify_revision')
        return _FakeQueryResult(self._revision)


class _FakePool:
    def __init__(
        self,
        events: list[str],
        revision: int | None,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._revision = revision
        self._close_error = close_error
        self.close_calls = 0

    def open(self, *, wait: bool) -> None:
        assert wait is True
        self._events.extend(['open', 'wait'])

    def connection(self) -> _FakeConnection:
        self._events.append('connection')
        return _FakeConnection(self._events, self._revision)

    def close(self) -> None:
        self.close_calls += 1
        self._events.append('close')
        if self._close_error is not None:
            raise self._close_error


class _FakeSaver:
    def __init__(
        self,
        events: list[str],
        *,
        setup_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._setup_error = setup_error
        self.setup_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1
        self._events.append('setup')
        if self._setup_error is not None:
            raise self._setup_error


@dataclass
class _SessionStore:
    events: list[str]
    record: AgentRuntimeSchemaVersion | None = None
    add_calls: int = 0
    commit_calls: int = 0


class _FakeSession:
    def __init__(self, store: _SessionStore) -> None:
        self._store = store

    def __enter__(self) -> _FakeSession:
        self._store.events.append('session_enter')
        return self

    def __exit__(self, *_args: object) -> None:
        self._store.events.append('session_exit')

    def scalar(self, statement: Any) -> AgentRuntimeSchemaVersion | None:
        query_parameters = statement.compile().params
        assert 'langgraph_checkpoint' in query_parameters.values()
        self._store.events.append('schema_lookup')
        return self._store.record

    def add(self, record: AgentRuntimeSchemaVersion) -> None:
        assert record.component == 'langgraph_checkpoint'
        self._store.events.append('schema_add')
        self._store.add_calls += 1
        self._store.record = record

    def commit(self) -> None:
        self._store.events.append('schema_commit')
        self._store.commit_calls += 1


def _postgres_settings(*, strict: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=None,
        database_url='postgresql+psycopg://runtime:credential-marker@db/checkpoints',
        langgraph_strict_msgpack=strict,
    )


def _run_bootstrap_process(args: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment['PARAWORKS_DEMO_MODE'] = 'credential-marker'
    return subprocess.run(
        [sys.executable, str(_BOOTSTRAP_SCRIPT), *args],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_backup_confirmation_is_required_before_opening_a_pool() -> None:
    pool_calls: list[str] = []

    with pytest.raises(
        CheckpointBootstrapError,
        match='^database backup confirmation is required$',
    ):
        bootstrap_langgraph_checkpointer(
            _postgres_settings(),
            backup_confirmed=False,
            pool_factory=lambda dsn: pool_calls.append(dsn),  # type: ignore[arg-type]
        )

    assert pool_calls == []


def test_non_postgres_url_is_rejected_without_exposing_configuration() -> None:
    database_url = 'sqlite:///credential-marker.db'
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=None,
        database_url=database_url,
        langgraph_strict_msgpack=True,
    )
    pool_calls: list[str] = []

    with pytest.raises(CheckpointBootstrapError) as exc_info:
        bootstrap_langgraph_checkpointer(
            settings,
            backup_confirmed=True,
            pool_factory=lambda dsn: pool_calls.append(dsn),  # type: ignore[arg-type]
        )

    assert str(exc_info.value) == 'checkpoint bootstrap failed'
    assert database_url not in str(exc_info.value)
    assert 'credential-marker' not in str(exc_info.value)
    assert pool_calls == []


def test_strict_msgpack_is_required_before_opening_a_pool() -> None:
    pool_calls: list[str] = []

    with pytest.raises(
        CheckpointBootstrapError,
        match='^strict msgpack is required$',
    ):
        bootstrap_langgraph_checkpointer(
            _postgres_settings(strict=False),
            backup_confirmed=True,
            pool_factory=lambda dsn: pool_calls.append(dsn),  # type: ignore[arg-type]
        )

    assert pool_calls == []


def test_valid_bootstrap_sets_up_verifies_records_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import bootstrap

    events: list[str] = []
    pool = _FakePool(events, _EXPECTED_REVISION)
    saver = _FakeSaver(events)
    store = _SessionStore(events)

    def pool_factory(dsn: str) -> _FakePool:
        assert dsn == 'postgresql://runtime:credential-marker@db/checkpoints'
        events.append('pool_factory')
        return pool

    def saver_factory(pool_arg: object, serializer: object) -> _FakeSaver:
        assert pool_arg is pool
        assert serializer.pickle_fallback is False
        events.append('saver_factory')
        return saver

    def package_version(package_name: str) -> str:
        assert package_name == 'langgraph-checkpoint-postgres'
        events.append('package_version')
        return _PACKAGE_VERSION

    def now() -> datetime:
        events.append('now')
        return _APPLIED_AT

    monkeypatch.setattr(bootstrap, 'version', package_version)

    result = bootstrap_langgraph_checkpointer(
        _postgres_settings(),
        backup_confirmed=True,
        pool_factory=pool_factory,  # type: ignore[arg-type]
        saver_factory=saver_factory,  # type: ignore[arg-type]
        session_factory=lambda: _FakeSession(store),  # type: ignore[arg-type]
        now=now,
    )

    assert result == BootstrapResult(
        component='langgraph_checkpoint',
        package_name='langgraph-checkpoint-postgres',
        package_version=_PACKAGE_VERSION,
        schema_revision=_EXPECTED_REVISION,
        applied_at=_APPLIED_AT,
    )
    assert store.record is not None
    assert store.record.component == 'langgraph_checkpoint'
    assert store.record.package_name == 'langgraph-checkpoint-postgres'
    assert store.record.package_version == _PACKAGE_VERSION
    assert store.record.schema_revision == _EXPECTED_REVISION
    assert store.record.applied_at == _APPLIED_AT
    assert store.add_calls == 1
    assert store.commit_calls == 1
    assert saver.setup_calls == 1
    assert pool.close_calls == 1
    assert events == [
        'pool_factory',
        'open',
        'wait',
        'saver_factory',
        'setup',
        'connection',
        'connection_enter',
        'verify_revision',
        'connection_exit',
        'now',
        'package_version',
        'session_enter',
        'schema_lookup',
        'schema_add',
        'schema_commit',
        'session_exit',
        'close',
    ]
    result_text = repr(result)
    assert 'credential-marker' not in result_text
    assert 'postgresql://' not in result_text


def test_repeated_bootstrap_updates_the_single_component_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import bootstrap

    events: list[str] = []
    pools: list[_FakePool] = []
    savers: list[_FakeSaver] = []
    store = _SessionStore(events)
    applied_times = iter(
        [
            _APPLIED_AT,
            datetime(2026, 8, 26, 10, 30, tzinfo=UTC),
        ]
    )

    def pool_factory(_dsn: str) -> _FakePool:
        pool = _FakePool(events, _EXPECTED_REVISION)
        pools.append(pool)
        return pool

    def saver_factory(_pool: object, _serializer: object) -> _FakeSaver:
        saver = _FakeSaver(events)
        savers.append(saver)
        return saver

    monkeypatch.setattr(bootstrap, 'version', lambda _name: _PACKAGE_VERSION)

    first = bootstrap_langgraph_checkpointer(
        _postgres_settings(),
        backup_confirmed=True,
        pool_factory=pool_factory,  # type: ignore[arg-type]
        saver_factory=saver_factory,  # type: ignore[arg-type]
        session_factory=lambda: _FakeSession(store),  # type: ignore[arg-type]
        now=lambda: next(applied_times),
    )
    second = bootstrap_langgraph_checkpointer(
        _postgres_settings(),
        backup_confirmed=True,
        pool_factory=pool_factory,  # type: ignore[arg-type]
        saver_factory=saver_factory,  # type: ignore[arg-type]
        session_factory=lambda: _FakeSession(store),  # type: ignore[arg-type]
        now=lambda: next(applied_times),
    )

    assert store.record is not None
    assert first.applied_at == _APPLIED_AT
    assert second.applied_at == datetime(2026, 8, 26, 10, 30, tzinfo=UTC)
    assert store.record.applied_at == second.applied_at
    assert store.add_calls == 1
    assert store.commit_calls == 2
    assert [saver.setup_calls for saver in savers] == [1, 1]
    assert [pool.close_calls for pool in pools] == [1, 1]


def test_schema_revision_mismatch_fails_before_metadata_and_closes_pool() -> None:
    events: list[str] = []
    pool = _FakePool(events, _EXPECTED_REVISION - 1)
    saver = _FakeSaver(events)
    session_calls: list[str] = []

    with pytest.raises(
        CheckpointBootstrapError,
        match='^checkpoint schema revision mismatch$',
    ):
        bootstrap_langgraph_checkpointer(
            _postgres_settings(),
            backup_confirmed=True,
            pool_factory=lambda _dsn: pool,  # type: ignore[arg-type]
            saver_factory=lambda _pool, _serializer: saver,  # type: ignore[arg-type]
            session_factory=lambda: session_calls.append('session'),  # type: ignore[arg-type]
        )

    assert saver.setup_calls == 1
    assert session_calls == []
    assert pool.close_calls == 1


def test_provider_failure_is_sanitized_and_pool_is_always_closed() -> None:
    events: list[str] = []
    pool = _FakePool(events, _EXPECTED_REVISION)
    saver = _FakeSaver(events, setup_error=RuntimeError('credential-marker'))

    with pytest.raises(CheckpointBootstrapError) as exc_info:
        bootstrap_langgraph_checkpointer(
            _postgres_settings(),
            backup_confirmed=True,
            pool_factory=lambda _dsn: pool,  # type: ignore[arg-type]
            saver_factory=lambda _pool, _serializer: saver,  # type: ignore[arg-type]
        )

    assert str(exc_info.value) == 'checkpoint bootstrap failed'
    assert 'credential-marker' not in str(exc_info.value)
    assert saver.setup_calls == 1
    assert pool.close_calls == 1


def test_pool_close_failure_is_sanitized() -> None:
    events: list[str] = []
    pool = _FakePool(
        events,
        _EXPECTED_REVISION,
        close_error=RuntimeError('credential-marker'),
    )
    saver = _FakeSaver(events, setup_error=RuntimeError('setup-failed'))

    with pytest.raises(CheckpointBootstrapError) as exc_info:
        bootstrap_langgraph_checkpointer(
            _postgres_settings(),
            backup_confirmed=True,
            pool_factory=lambda _dsn: pool,  # type: ignore[arg-type]
            saver_factory=lambda _pool, _serializer: saver,  # type: ignore[arg-type]
        )

    assert str(exc_info.value) == 'checkpoint bootstrap failed'
    assert 'credential-marker' not in str(exc_info.value)
    assert 'setup-failed' not in str(exc_info.value)
    assert pool.close_calls == 1


def test_successful_commit_followed_by_pool_close_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import bootstrap

    events: list[str] = []
    pool = _FakePool(
        events,
        _EXPECTED_REVISION,
        close_error=RuntimeError('credential-marker'),
    )
    saver = _FakeSaver(events)
    store = _SessionStore(events)
    monkeypatch.setattr(bootstrap, 'version', lambda _name: _PACKAGE_VERSION)

    with pytest.raises(CheckpointBootstrapError) as exc_info:
        bootstrap_langgraph_checkpointer(
            _postgres_settings(),
            backup_confirmed=True,
            pool_factory=lambda _dsn: pool,  # type: ignore[arg-type]
            saver_factory=lambda _pool, _serializer: saver,  # type: ignore[arg-type]
            session_factory=lambda: _FakeSession(store),  # type: ignore[arg-type]
            now=lambda: _APPLIED_AT,
        )

    assert str(exc_info.value) == 'checkpoint bootstrap failed'
    assert 'credential-marker' not in str(exc_info.value)
    assert saver.setup_calls == 1
    assert store.add_calls == 1
    assert store.commit_calls == 1
    assert pool.close_calls == 1


def test_lazy_session_factory_failure_is_sanitized_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime import bootstrap

    events: list[str] = []
    pool = _FakePool(events, _EXPECTED_REVISION)
    saver = _FakeSaver(events)
    monkeypatch.setattr(bootstrap, 'version', lambda _name: _PACKAGE_VERSION)

    def failing_session_factory() -> _FakeSession:
        raise RuntimeError('credential-marker')

    with pytest.raises(CheckpointBootstrapError) as exc_info:
        bootstrap_langgraph_checkpointer(
            _postgres_settings(),
            backup_confirmed=True,
            pool_factory=lambda _dsn: pool,  # type: ignore[arg-type]
            saver_factory=lambda _pool, _serializer: saver,  # type: ignore[arg-type]
            session_factory=failing_session_factory,  # type: ignore[arg-type]
        )

    assert str(exc_info.value) == 'checkpoint bootstrap failed'
    assert 'credential-marker' not in str(exc_info.value)
    assert saver.setup_calls == 1
    assert pool.close_calls == 1


def test_importing_operator_command_has_no_settings_side_effect() -> None:
    environment = os.environ.copy()
    environment['PARAWORKS_DEMO_MODE'] = 'credential-marker'

    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            'import scripts.bootstrap_langgraph_checkpointer; print("imported")',
        ],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == 'imported\n'
    assert completed.stderr == ''


@pytest.mark.parametrize('args', [[], ['--unknown', 'credential-marker']])
def test_operator_argument_errors_happen_before_invalid_settings(
    args: list[str],
) -> None:
    completed = _run_bootstrap_process(args)

    assert completed.returncode == 2
    assert completed.stdout == ''
    assert 'invalid arguments' in completed.stderr
    assert 'credential-marker' not in completed.stderr
    assert 'Traceback' not in completed.stderr


def test_operator_invalid_settings_failure_is_bounded() -> None:
    completed = _run_bootstrap_process(['--confirm-backup'])

    assert completed.returncode == 1
    assert completed.stdout == ''
    assert completed.stderr == 'checkpoint bootstrap failed\n'
    assert 'credential-marker' not in completed.stderr
    assert 'Traceback' not in completed.stderr


def test_operator_command_requires_backup_confirmation_before_service_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_calls: list[object] = []
    monkeypatch.setattr(
        bootstrap_command,
        'bootstrap_langgraph_checkpointer',
        lambda *args, **kwargs: service_calls.append((args, kwargs)),
    )

    with pytest.raises(SystemExit) as exc_info:
        bootstrap_command.main([])

    assert exc_info.value.code != 0
    assert service_calls == []


def test_operator_command_accepts_only_the_confirmation_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service_calls: list[object] = []
    monkeypatch.setattr(
        bootstrap_command,
        'bootstrap_langgraph_checkpointer',
        lambda *args, **kwargs: service_calls.append((args, kwargs)),
    )

    with pytest.raises(SystemExit) as exc_info:
        bootstrap_command.main(
            ['--confirm-backup', '--dsn', 'postgresql://user:secret@db/runtime']
        )

    assert exc_info.value.code != 0
    assert service_calls == []
    output = capsys.readouterr()
    assert 'postgresql://' not in output.err
    assert 'secret' not in output.err


def test_operator_command_rejects_abbreviated_confirmation_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bootstrap_command,
        'get_settings',
        lambda: calls.append('settings'),
    )
    monkeypatch.setattr(
        bootstrap_command,
        'bootstrap_langgraph_checkpointer',
        lambda *args, **kwargs: calls.append('bootstrap'),
    )

    with pytest.raises(SystemExit) as exc_info:
        bootstrap_command.main(['--confirm'])

    assert exc_info.value.code == 2
    assert calls == []
    output = capsys.readouterr()
    assert 'invalid arguments' in output.err
    assert '--confirm' not in output.err


def test_operator_command_prints_only_the_safe_bootstrap_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = object()
    calls: list[tuple[object, bool]] = []
    result = BootstrapResult(
        component='langgraph_checkpoint',
        package_name='langgraph-checkpoint-postgres',
        package_version=_PACKAGE_VERSION,
        schema_revision=_EXPECTED_REVISION,
        applied_at=_APPLIED_AT,
    )
    monkeypatch.setattr(bootstrap_command, 'get_settings', lambda: settings)

    def bootstrap(settings_arg: object, *, backup_confirmed: bool) -> BootstrapResult:
        calls.append((settings_arg, backup_confirmed))
        return result

    monkeypatch.setattr(
        bootstrap_command,
        'bootstrap_langgraph_checkpointer',
        bootstrap,
    )

    assert bootstrap_command.main(['--confirm-backup']) == 0

    output = capsys.readouterr()
    assert calls == [(settings, True)]
    assert output.err == ''
    assert output.out.splitlines() == [
        'component: langgraph_checkpoint',
        f'package version: {_PACKAGE_VERSION}',
        f'schema revision: {_EXPECTED_REVISION}',
        'applied at: 2026-08-26T09:30:00+00:00',
    ]
    assert 'langgraph-checkpoint-postgres' not in output.out
    assert 'postgresql://' not in output.out
    assert 'credential-marker' not in output.out


def test_operator_command_sanitizes_bootstrap_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bootstrap_command, 'get_settings', object)

    def fail_bootstrap(*_args: object, **_kwargs: object) -> BootstrapResult:
        raise CheckpointBootstrapError('credential-marker')

    monkeypatch.setattr(
        bootstrap_command,
        'bootstrap_langgraph_checkpointer',
        fail_bootstrap,
    )

    assert bootstrap_command.main(['--confirm-backup']) == 1

    output = capsys.readouterr()
    assert output.out == ''
    assert output.err == 'checkpoint bootstrap failed\n'
    assert 'credential-marker' not in output.err
