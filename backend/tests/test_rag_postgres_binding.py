from __future__ import annotations

import inspect
import threading

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from backend.app.agent_runtime import rag_postgres_binding as binding_module
from backend.app.agent_runtime.rag_advisory_locks import RegisteredAdvisoryLock
from backend.app.agent_runtime.rag_postgres_binding import (
    RagPostgresDatabaseBusyError,
    RagPostgresDatabaseIdentity,
    _bind_rag_postgres_database,
)
from backend.app.db import initialization


def _fake_postgres_engines():
    application = create_engine('sqlite+pysqlite:///:memory:')
    dedicated = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    application.dialect.name = 'postgresql'
    dedicated.dialect.name = 'postgresql'
    return application, dedicated


def _identity(schema: str = 'authority_schema') -> RagPostgresDatabaseIdentity:
    return RagPostgresDatabaseIdentity(
        database_name='authority_database',
        current_schema=schema,
        effective_search_path=(schema, 'public'),
        search_path_setting=f'{schema}, public',
        current_role='authority_role',
        database_oid=16_384,
    )


def _trusted_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    application,
    dedicated,
):
    engines = iter((application, dedicated))
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: next(engines),
    )
    runtime = initialization.initialize_database_runtime(
        'postgresql+psycopg://authority-role@localhost/authority-database'
    )
    assert runtime.rag_postgres_bootstrap is not None
    return runtime.rag_postgres_bootstrap


def test_binding_api_requires_explicit_dedicated_engine_and_bootstrap_capability():
    assert tuple(inspect.signature(_bind_rag_postgres_database).parameters) == (
        'session',
        'trusted_bootstrap',
        'bootstrap_capability',
    )


def test_identity_probe_is_least_privilege_and_has_no_control_file_dependency():
    assert 'pg_control_system' not in str(binding_module._IDENTITY_SQL)


def test_database_authority_requires_an_active_operation_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    identity = _identity()
    monkeypatch.setattr(binding_module, '_session_identity', lambda _session: identity)
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )

    assert callable(getattr(authority, 'operation_lease', None))

    authority.close()
    session.close()
    application.dispose()


def test_explicit_dedicated_engine_is_owned_closed_once_and_never_url_cloned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    identity = _identity()
    disposed = 0

    def record_dispose(*_args: object) -> None:
        nonlocal disposed
        disposed += 1

    event.listen(dedicated, 'engine_disposed', record_dispose)
    monkeypatch.setattr(binding_module, '_session_identity', lambda _session: identity)
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        binding_module,
        'create_engine',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('binding must not reconstruct URL credentials')
        ),
        raising=False,
    )
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    with authority.operation_lease():
        connection = authority.connect()
        with pytest.raises(RagPostgresDatabaseBusyError, match='active operation'):
            authority.close()
        assert connection.closed is False
        assert disposed == 0
    authority.close()

    assert connection.closed is True
    assert authority.closed is True
    assert disposed == 1
    with pytest.raises(TypeError, match='closed'):
        authority.connect()
    session.close()
    application.dispose()


def test_concurrent_close_cannot_release_an_active_operation_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    identity = _identity()
    monkeypatch.setattr(binding_module, '_session_identity', lambda _session: identity)
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    close_errors: list[BaseException] = []
    lease_errors: list[BaseException] = []

    with authority.operation_lease():
        connection = authority.connect()

        def contend_for_lease() -> None:
            try:
                with authority.operation_lease():
                    raise AssertionError('request authority must be single-owner')
            except BaseException as exc:
                lease_errors.append(exc)

        first_contender = threading.Thread(target=contend_for_lease)
        first_contender.start()
        first_contender.join(timeout=5)
        assert first_contender.is_alive() is False
        assert len(lease_errors) == 1
        assert isinstance(lease_errors[0], TypeError)
        lease_errors.clear()

        def close_while_active() -> None:
            try:
                authority.close()
            except BaseException as exc:
                close_errors.append(exc)

        closer = threading.Thread(target=close_while_active)
        closer.start()
        closer.join(timeout=5)
        assert closer.is_alive() is False
        assert len(close_errors) == 1
        assert isinstance(close_errors[0], RagPostgresDatabaseBusyError)
        assert connection.closed is False

        def acquire_after_close_started() -> None:
            try:
                with authority.operation_lease():
                    raise AssertionError('closing authority must reject new lease')
            except BaseException as exc:
                lease_errors.append(exc)

        contender = threading.Thread(target=acquire_after_close_started)
        contender.start()
        contender.join(timeout=5)
        assert contender.is_alive() is False
        assert len(lease_errors) == 1
        assert isinstance(lease_errors[0], TypeError)
        assert connection.closed is False

    authority.close()
    assert connection.closed is True
    session.close()
    application.dispose()


def test_binding_identity_failure_disposes_injected_engine_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    disposed = 0

    def record_dispose(*_args: object) -> None:
        nonlocal disposed
        disposed += 1

    event.listen(dedicated, 'engine_disposed', record_dispose)
    monkeypatch.setattr(
        binding_module,
        '_session_identity',
        lambda _session: _identity('schema_a'),
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: _identity('schema_b'),
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    with pytest.raises(TypeError, match='does not match session'):
        _bind_rag_postgres_database(
            session,
            trusted_bootstrap=_trusted_bootstrap(
                monkeypatch,
                application,
                dedicated,
            ),
            bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
        )

    assert disposed == 1
    session.close()
    application.dispose()


def test_close_disposes_once_even_when_connection_invalidation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    identity = _identity()
    disposed = 0

    def record_dispose(*_args: object) -> None:
        nonlocal disposed
        disposed += 1

    event.listen(dedicated, 'engine_disposed', record_dispose)
    monkeypatch.setattr(binding_module, '_session_identity', lambda _session: identity)
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    with authority.operation_lease():
        connection = authority.connect()

    def fail_invalidate(_self) -> None:
        raise RuntimeError('injected invalidate failure')

    monkeypatch.setattr(type(connection), 'invalidate', fail_invalidate)
    with pytest.raises(RuntimeError, match='injected invalidate failure'):
        authority.close()

    assert connection.closed is True
    assert authority.closed is True
    assert disposed == 1
    authority.close()
    assert disposed == 1
    session.close()
    application.dispose()
