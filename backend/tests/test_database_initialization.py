from __future__ import annotations

import ast
import os
import subprocess
import sys
import traceback
from collections.abc import Generator
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
from sqlalchemy.pool import NullPool

from backend.app.db import initialization
from backend.app.db import session as database_session


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


def test_postgres_runtime_mints_a_trusted_dedicated_engine_bootstrap() -> None:
    policy_type = getattr(initialization, 'DatabaseConnectionPolicy', None)
    assert policy_type is not None
    assert 'rag_postgres_bootstrap' in initialization.DatabaseRuntime.__annotations__


def test_postgres_bootstrap_preserves_the_resolved_connection_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = sqlalchemy_create_engine('sqlite+pysqlite:///:memory:')
    dedicated = sqlalchemy_create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    application.dialect.name = 'postgresql'
    dedicated.dialect.name = 'postgresql'
    created: list[tuple[str, dict[str, object]]] = []
    initialized: list[object] = []
    engines = iter((application, dedicated))
    creator = object()
    tls_context = object()
    mutable_connect_args: dict[str, object] = {
        'sslmode': 'verify-full',
        'sslrootcert': tls_context,
    }

    def create_engine_probe(database_url: str, **kwargs: object):
        created.append((database_url, kwargs))
        return next(engines)

    monkeypatch.setattr(initialization, 'create_engine', create_engine_probe)
    policy = initialization.DatabaseConnectionPolicy(
        engine_options={
            'creator': creator,
            'connect_args': mutable_connect_args,
        },
        initialize_engine=initialized.append,
    )
    mutable_connect_args['sslmode'] = 'disable'
    runtime = initialization.initialize_database_runtime(
        'postgresql+psycopg://invalid.example/authority',
        connection_policy=policy,
    )
    assert runtime.rag_postgres_bootstrap is not None
    issued = runtime.rag_postgres_bootstrap._issue(application)
    issued_engine = runtime.rag_postgres_bootstrap._require_issued(
        issued,
        application,
    )

    assert issued_engine is dedicated
    assert created == [
        (
            'postgresql+psycopg://invalid.example/authority',
            {
                'pool_pre_ping': True,
                'creator': creator,
                'connect_args': {
                    'sslmode': 'verify-full',
                    'sslrootcert': tls_context,
                },
            },
        ),
        (
            'postgresql+psycopg://invalid.example/authority',
            {
                'pool_pre_ping': True,
                'creator': creator,
                'connect_args': {
                    'sslmode': 'verify-full',
                    'sslrootcert': tls_context,
                },
                'poolclass': NullPool,
            },
        ),
    ]
    assert initialized == [application, dedicated]

    issued_engine.dispose()
    runtime.dispose()
    with pytest.raises(TypeError, match='attestation changed'):
        runtime.rag_postgres_bootstrap._require_issued(issued, application)
    with pytest.raises(TypeError, match='bootstrap authority changed'):
        runtime.rag_postgres_bootstrap._issue(application)


def test_postgres_bootstrap_cleans_up_when_engine_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = sqlalchemy_create_engine('sqlite+pysqlite:///:memory:')
    application.dialect.name = 'postgresql'
    disposals: list[object] = []
    event.listen(application, 'engine_disposed', lambda engine: disposals.append(engine))
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: application,
    )

    def fail_initialization(_engine: object) -> None:
        raise RuntimeError('injected engine initialization failure')

    with pytest.raises(RuntimeError, match='injected engine initialization'):
        initialization.initialize_database_runtime(
            'postgresql+psycopg://invalid.example/authority',
            connection_policy=initialization.DatabaseConnectionPolicy(
                initialize_engine=fail_initialization,
            ),
        )

    assert disposals == [application]


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


def test_session_adapter_preserves_public_engine_and_factory_contract() -> None:
    assert database_session.engine.pool._pre_ping is True
    assert database_session.SessionLocal.kw['bind'] is database_session.engine
    assert database_session.SessionLocal.kw['autoflush'] is False
    assert database_session.SessionLocal.kw['autocommit'] is False
    assert database_session.SessionLocal.kw['expire_on_commit'] is True


def test_get_db_still_closes_the_request_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionProbe:
        closed = False

        def close(self) -> None:
            self.closed = True

    session = SessionProbe()
    monkeypatch.setattr(database_session, 'SessionLocal', lambda: session)
    dependency: Generator[object, None, None] = database_session.get_db()
    assert next(dependency) is session
    dependency.close()
    assert session.closed is True


def _run_isolated_python(script: str) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[2]
    isolated_cwd = Path(__file__).resolve().parent
    assert not (isolated_cwd / '.env').exists()
    allowed_parent_names = (
        'COMSPEC',
        'PATH',
        'PATHEXT',
        'SYSTEMROOT',
        'TEMP',
        'TMP',
        'WINDIR',
    )
    env = {
        name: os.environ[name]
        for name in allowed_parent_names
        if name in os.environ
    }
    env.update(
        {
            'AUTO_REVIEW_MODE': 'disabled',
            'AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS': '60',
            'AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS': '5',
            'AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS': '120',
            'AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS': '30',
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': 'sqlite:///:memory:',
            'DATABASE_URL': 'sqlite:///:memory:',
            'PYTHONPATH': str(repository_root),
        }
    )
    return subprocess.run(
        [sys.executable, '-c', script],
        cwd=isolated_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize(
    ('case_inputs', 'expected_url'),
    (
        (
            {
                'paraworks_demo_mode': True,
                'paraworks_demo_database_url': 'sqlite:///demo-precedence.db',
                'paraworks_database_url': 'sqlite:///override-precedence.db',
                'database_url': 'sqlite:///fallback-precedence.db',
            },
            'sqlite:///demo-precedence.db',
        ),
        (
            {
                'paraworks_demo_mode': False,
                'paraworks_demo_database_url': 'sqlite:///demo-precedence.db',
                'paraworks_database_url': 'sqlite:///override-precedence.db',
                'database_url': 'sqlite:///fallback-precedence.db',
            },
            'sqlite:///override-precedence.db',
        ),
        (
            {
                'paraworks_demo_mode': False,
                'paraworks_demo_database_url': None,
                'paraworks_database_url': None,
                'database_url': 'sqlite:///fallback-precedence.db',
            },
            'sqlite:///fallback-precedence.db',
        ),
    ),
)
def test_session_adapter_forwards_the_resolved_database_url_in_a_fresh_process(
    case_inputs: dict[str, object],
    expected_url: str,
) -> None:
    script = (
        'import importlib\n'
        'from types import SimpleNamespace\n'
        "config = importlib.import_module('backend.app.core.config')\n"
        "initialization = importlib.import_module('backend.app.db.initialization')\n"
        f'case_inputs = {case_inputs!r}\n'
        'settings = config.Settings(_env_file=None, **case_inputs)\n'
        'observed_urls = []\n'
        'def initialize(database_url):\n'
        '    observed_urls.append(database_url)\n'
        '    return SimpleNamespace(engine=object(), session_factory=object())\n'
        'config.get_settings = lambda: settings\n'
        'initialization.initialize_database_runtime = initialize\n'
        "importlib.import_module('backend.app.db.session')\n"
        f'assert settings.resolved_database_url() == {expected_url!r}\n'
        f'assert observed_urls == [{expected_url!r}]\n'
    )
    completed = _run_isolated_python(script)

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''


CONSUMER_MODULES = (
    'backend.app.db.init_db',
    'backend.app.tasks.sync',
    'backend.app.tasks.rag_indexing',
    'backend.app.agent_runtime.bootstrap',
    'backend.app.agent_runtime.retention',
)


@pytest.mark.parametrize('module_name', CONSUMER_MODULES)
@pytest.mark.parametrize('application_first', (False, True))
def test_database_consumers_import_direct_and_application_first(
    module_name: str,
    application_first: bool,
) -> None:
    statements = ['import importlib']
    if application_first:
        statements.append("importlib.import_module('backend.app.main')")
    statements.append(f'importlib.import_module({module_name!r})')
    completed = _run_isolated_python(';'.join(statements))

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''


@pytest.mark.parametrize('application_first', (False, True))
def test_fastapi_get_db_override_keeps_identity_across_import_order(
    application_first: bool,
) -> None:
    if application_first:
        ordered_imports = (
            'from backend.app.main import app\n'
            'from backend.app.db.session import get_db\n'
        )
    else:
        ordered_imports = (
            'from backend.app.db.session import get_db\n'
            'from backend.app.main import app\n'
        )
    script = ordered_imports + (
        'from fastapi.routing import APIRoute\n'
        'def walk_dependencies(dependant):\n'
        '    yield dependant\n'
        '    for child in dependant.dependencies:\n'
        '        yield from walk_dependencies(child)\n'
        'route = next(\n'
        '    route for route in app.routes\n'
        '    if isinstance(route, APIRoute)\n'
        "    and route.path == '/api/v1/documents'\n"
        "    and 'GET' in route.methods\n"
        ')\n'
        'matches = [\n'
        '    dependant for dependant in walk_dependencies(route.dependant)\n'
        '    if dependant.call is get_db\n'
        ']\n'
        'assert len(matches) == 1\n'
        'def override_db():\n'
        '    yield object()\n'
        'app.dependency_overrides[get_db] = override_db\n'
        'assert app.dependency_overrides[matches[0].call] is override_db\n'
        'app.dependency_overrides.clear()\n'
    )
    completed = _run_isolated_python(script)

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''


@pytest.mark.parametrize(
    'script',
    (
        'import backend.app.db.initialization; import backend.app.db.session',
        'import backend.app.db.session; import backend.app.db.initialization',
        'import backend.app.db.session; import backend.app.main',
        'import backend.app.main; import backend.app.db.session',
    ),
)
def test_database_initialization_and_session_import_order_is_cycle_free(
    script: str,
) -> None:
    completed = _run_isolated_python(script)

    assert completed.returncode == 0
    assert completed.stdout == ''
    assert completed.stderr == ''


def test_database_initialization_is_a_settings_and_environment_free_leaf() -> None:
    initializer_path = Path(__file__).resolve().parents[1] / 'app/db/initialization.py'
    tree = ast.parse(initializer_path.read_text(encoding='utf-8'))
    prohibited_names = {'environ', 'getenv', 'get_settings', 'Settings'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != 'os' for alias in node.names)
            assert all(
                alias.name
                not in {'backend.app.core.config', 'backend.app.db.session'}
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            assert node.module != 'os'
            assert node.module not in {
                'backend.app.core.config',
                'backend.app.db.session',
            }
        if isinstance(node, ast.Name):
            assert node.id not in prohibited_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in prohibited_names
