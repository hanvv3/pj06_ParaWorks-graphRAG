from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy import event
from sqlalchemy.dialects import registry
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    InvalidRequestError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
    StatementError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLAlchemyTimeoutError,
)

from backend.app.db import initialization


def test_initialize_database_runtime_preserves_exact_options_without_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    connect_events: list[str] = []
    engine = sqlalchemy_create_engine('sqlite:///:memory:')
    event.listen(engine, 'connect', lambda *_args: connect_events.append('connect'))

    def create_engine_probe(database_url: str, **kwargs: object):
        observed['database_url'] = database_url
        observed['kwargs'] = kwargs
        return engine

    monkeypatch.setattr(initialization, 'create_engine', create_engine_probe)
    runtime = initialization.initialize_database_runtime('sqlite:///:memory:')

    assert runtime.engine is engine
    assert runtime.session_factory.kw['bind'] is engine
    assert runtime.session_factory.kw['autoflush'] is False
    assert runtime.session_factory.kw['autocommit'] is False
    assert runtime.session_factory.kw['expire_on_commit'] is True
    assert observed == {
        'database_url': 'sqlite:///:memory:',
        'kwargs': {'pool_pre_ping': True},
    }
    assert connect_events == []
    assert 'Engine(' not in repr(runtime)
    engine.dispose()


@pytest.mark.parametrize(
    'database_url',
    (
        'not-a-sqlalchemy-url-sensitive',
        'paraworks_missing_dialect://user:secret@host/database',
    ),
)
def test_configuration_failures_are_typed_and_sanitized(database_url: str) -> None:
    with pytest.raises(initialization.DatabaseConfigurationError) as captured:
        initialization.initialize_database_runtime(database_url)

    error = captured.value
    rendered = ''.join(traceback.format_exception(error))
    assert error.code == 'database_configuration_invalid'
    assert error.args == ()
    assert error.__dict__ == {}
    assert error.__cause__ is None
    assert error.__context__ is None
    assert database_url not in repr(error)
    assert database_url not in rendered
    assert 'secret' not in rendered


def test_typed_error_codes_are_class_level_only() -> None:
    configuration = initialization.DatabaseConfigurationError()
    storage = initialization.DatabaseInitializationError()

    assert configuration.code == 'database_configuration_invalid'
    assert storage.code == 'database_initialization_failed'
    assert configuration.args == storage.args == ()
    assert configuration.__dict__ == storage.__dict__ == {}


def _dbapi_failure(
    error_type: type[DBAPIError],
    *,
    connection_invalidated: bool,
) -> DBAPIError:
    return error_type(
        'SELECT sensitive_statement',
        {'secret': 'sensitive-param'},
        RuntimeError('sensitive-dbapi-original'),
        connection_invalidated=connection_invalidated,
    )


@pytest.mark.parametrize(
    'failure',
    (
        ModuleNotFoundError('sensitive-driver-module'),
        ImportError('sensitive-native-module'),
        OSError('sensitive-native-loader'),
        OperationalError(
            'statement', {}, RuntimeError('db'), connection_invalidated=False
        ),
        OperationalError(
            'statement', {}, RuntimeError('db'), connection_invalidated=True
        ),
        InterfaceError(
            'statement', {}, RuntimeError('db'), connection_invalidated=False
        ),
        InterfaceError(
            'statement', {}, RuntimeError('db'), connection_invalidated=True
        ),
        SQLAlchemyTimeoutError('sensitive-pool-timeout'),
        DisconnectionError('sensitive-disconnect'),
        _dbapi_failure(IntegrityError, connection_invalidated=True),
        _dbapi_failure(ProgrammingError, connection_invalidated=True),
        _dbapi_failure(DataError, connection_invalidated=True),
        _dbapi_failure(DBAPIError, connection_invalidated=True),
    ),
)
def test_engine_availability_failures_become_sanitized_typed_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    error = captured.value
    rendered = ''.join(traceback.format_exception(error))
    assert error.args == ()
    assert error.__cause__ is None
    assert error.__context__ is None
    assert 'sensitive' not in rendered


@pytest.mark.parametrize(
    'failure',
    (
        _dbapi_failure(IntegrityError, connection_invalidated=False),
        _dbapi_failure(ProgrammingError, connection_invalidated=False),
        _dbapi_failure(DataError, connection_invalidated=False),
        _dbapi_failure(DBAPIError, connection_invalidated=False),
        StatementError('sensitive-statement', 'SELECT 1', {}, RuntimeError('raw')),
        InvalidRequestError('sensitive-invalid-request'),
        SQLAlchemyError('sensitive-generic-sqlalchemy-error'),
        TypeError('sensitive-programmer-error'),
    ),
)
def test_nonavailability_failures_remain_original_operation_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(type(failure)) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    assert captured.value is failure


def _write_initializer_dialect_probe(tmp_path: Path) -> Path:
    (tmp_path / 'paraworks_initializer_probe_dialect.py').write_text(
        """
import os

from sqlalchemy.engine.default import DefaultDialect


class ProbeDialect(DefaultDialect):
    name = 'paraworks_initializer_probe'
    driver = 'probe'

    @classmethod
    def import_dbapi(cls):
        failure = os.environ['PARAWORKS_TEST_DBAPI_FAILURE']
        if failure == 'module':
            raise ModuleNotFoundError('dbapi-module-sensitive-sentinel')
        if failure == 'import':
            raise ImportError('dbapi-native-sensitive-sentinel')
        if failure == 'oserror':
            raise OSError('dbapi-loader-sensitive-sentinel')
        raise RuntimeError('unknown-probe-failure')
""".strip()
        + '\n',
        encoding='utf-8',
    )
    return tmp_path


@pytest.mark.parametrize('failure', ('module', 'import', 'oserror'))
def test_real_dialect_dbapi_load_failures_are_typed_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    probe_path = _write_initializer_dialect_probe(tmp_path)
    monkeypatch.syspath_prepend(str(probe_path))
    monkeypatch.setenv('PARAWORKS_TEST_DBAPI_FAILURE', failure)
    registry.register(
        'paraworks_initializer_probe',
        'paraworks_initializer_probe_dialect',
        'ProbeDialect',
    )
    database_url = (
        'paraworks_initializer_probe://probe_role_test:probe_password@'
        '127.0.0.1:55432/probe_database_test'
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        initialization.initialize_database_runtime(database_url)

    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert database_url not in rendered
    assert 'probe_password' not in rendered
    assert 'sensitive-sentinel' not in rendered


def test_real_dialect_programmer_failure_remains_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_path = _write_initializer_dialect_probe(tmp_path)
    monkeypatch.syspath_prepend(str(probe_path))
    monkeypatch.setenv('PARAWORKS_TEST_DBAPI_FAILURE', 'runtime')
    registry.register(
        'paraworks_initializer_probe',
        'paraworks_initializer_probe_dialect',
        'ProbeDialect',
    )

    with pytest.raises(RuntimeError) as captured:
        initialization.initialize_database_runtime(
            'paraworks_initializer_probe://probe_role_test@localhost/probe_database_test'
        )

    assert str(captured.value) == 'unknown-probe-failure'


class _DisposeProbe:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def dispose(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure


def test_runtime_dispose_latches_before_availability_failure() -> None:
    engine = _DisposeProbe(
        OperationalError(
            'dispose-sensitive-statement',
            {'secret': 'dispose-sensitive-param'},
            RuntimeError('dispose-sensitive-original'),
            connection_invalidated=False,
        )
    )
    runtime = initialization.DatabaseRuntime(
        engine=engine,  # type: ignore[arg-type]
        session_factory=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        runtime.dispose()
    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'dispose-sensitive' not in rendered
    runtime.dispose()
    assert engine.calls == 1


def test_runtime_dispose_success_is_idempotent() -> None:
    engine = _DisposeProbe()
    runtime = initialization.DatabaseRuntime(
        engine=engine,  # type: ignore[arg-type]
        session_factory=object(),  # type: ignore[arg-type]
    )

    runtime.dispose()
    runtime.dispose()

    assert engine.calls == 1


def test_runtime_dispose_preserves_programmer_failure_and_does_not_retry() -> None:
    failure = TypeError('dispose-programmer-sentinel')
    engine = _DisposeProbe(failure)
    runtime = initialization.DatabaseRuntime(
        engine=engine,  # type: ignore[arg-type]
        session_factory=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError) as captured:
        runtime.dispose()
    assert captured.value is failure
    runtime.dispose()
    assert engine.calls == 1


def _raise(error: Exception) -> None:
    raise error


def _chain_members(error: BaseException) -> tuple[BaseException, ...]:
    members: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in members:
        members.append(current)
        current = current.__cause__ or current.__context__
    return tuple(members)


def test_partial_engine_cleanup_success_preserves_sessionmaker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker_failure = TypeError('sessionmaker-programmer-sentinel')
    engine = _DisposeProbe()
    monkeypatch.setattr(initialization, 'create_engine', lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        initialization,
        'sessionmaker',
        lambda *_args, **_kwargs: _raise(sessionmaker_failure),
    )

    with pytest.raises(TypeError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    assert captured.value is sessionmaker_failure
    assert engine.calls == 1


def test_partial_cleanup_availability_failure_overrides_without_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker_failure = TypeError('sessionmaker-sensitive-sentinel')
    cleanup_failure = OperationalError(
        'dispose-sensitive-statement',
        {'secret': 'dispose-sensitive-param'},
        RuntimeError('dispose-sensitive-original'),
        connection_invalidated=False,
    )
    engine = _DisposeProbe(cleanup_failure)
    monkeypatch.setattr(initialization, 'create_engine', lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        initialization,
        'sessionmaker',
        lambda *_args, **_kwargs: _raise(sessionmaker_failure),
    )

    with pytest.raises(initialization.DatabaseInitializationError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'sensitive' not in rendered
    assert engine.calls == 1


def test_partial_cleanup_programmer_failure_overrides_without_factory_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessionmaker_failure = TypeError('sessionmaker-sensitive-sentinel')
    cleanup_failure = TypeError('dispose-programmer-sentinel')
    engine = _DisposeProbe(cleanup_failure)
    monkeypatch.setattr(initialization, 'create_engine', lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        initialization,
        'sessionmaker',
        lambda *_args, **_kwargs: _raise(sessionmaker_failure),
    )

    with pytest.raises(TypeError) as captured:
        initialization.initialize_database_runtime('sqlite:///:memory:')

    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value is cleanup_failure
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert 'sessionmaker-sensitive-sentinel' not in rendered
    assert engine.calls == 1


def test_typed_storage_error_retains_only_caller_active_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_failure = OperationalError(
        'storage-sensitive-statement',
        {'secret': 'storage-sensitive-param'},
        RuntimeError('storage-sensitive-original'),
        connection_invalidated=False,
    )
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: _raise(storage_failure),
    )
    caller_failure = ValueError('caller-active-sentinel')

    try:
        raise caller_failure
    except ValueError:
        with pytest.raises(initialization.DatabaseInitializationError) as captured:
            initialization.initialize_database_runtime('sqlite:///:memory:')

    chain = _chain_members(captured.value)
    rendered = ''.join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is caller_failure
    assert storage_failure not in chain
    assert 'storage-sensitive' not in rendered
    assert 'caller-active-sentinel' in rendered
