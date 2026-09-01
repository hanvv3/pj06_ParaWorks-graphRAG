from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from backend.app.agent_runtime import rag_postgres_binding as binding_module
from backend.app.agent_runtime.rag_advisory_locks import RegisteredAdvisoryLock
from backend.app.agent_runtime.rag_finalization import (
    RagFinalizationService,
    SqlAlchemyRagFinalizationBoundary,
)
from backend.app.agent_runtime.rag_postgres_binding import (
    RagPostgresDatabaseBusyError,
    RagPostgresDatabaseIdentity,
    _bind_rag_postgres_database,
)
from backend.app.core.config import Settings
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
    *,
    preserve_server_identity: bool = False,
):
    if not preserve_server_identity:
        server_identity = binding_module.RagPostgresWritableServerIdentity(
            server_address='127.0.0.1',
            server_port=5432,
            postmaster_start_time='2026-09-01 00:00:00+00',
        )
        monkeypatch.setattr(
            binding_module,
            '_session_server_identity',
            lambda _session: server_identity,
        )
        monkeypatch.setattr(
            binding_module,
            '_connection_server_identity',
            lambda _connection: server_identity,
        )
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


def test_provider_advisory_transport_uses_fresh_dedicated_engine_without_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = create_engine('sqlite+pysqlite:///:memory:')
    dedicated_one = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    dedicated_two = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    for engine in (application, dedicated_one, dedicated_two):
        engine.dialect.name = 'postgresql'
    engines = iter((application, dedicated_one, dedicated_two))
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: next(engines),
    )
    runtime = initialization.initialize_database_runtime(
        'postgresql+psycopg://authority-role@localhost/authority-database'
    )
    bootstrap = runtime.rag_postgres_bootstrap
    assert bootstrap is not None
    identity = _identity()
    server = binding_module.RagPostgresWritableServerIdentity(
        server_address='127.0.0.1',
        server_port=5432,
        postmaster_start_time='2026-09-01T00:00:00Z',
    )
    monkeypatch.setattr(binding_module, '_session_identity', lambda _session: identity)
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_session_server_identity',
        lambda _session: server,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_server_identity',
        lambda _connection: server,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )
    transport = binding_module._bind_rag_postgres_advisory_transport(
        Session(application),
        trusted_bootstrap=bootstrap,
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )

    with transport() as first:
        first_engine = first.engine
    with transport() as second:
        second_engine = second.engine

    assert first_engine is dedicated_one
    assert second_engine is dedicated_two
    assert first_engine is not second_engine
    assert transport.runtime_health_authority is bootstrap._runtime_effect_authority(
        application
    )


def test_cost_ledger_rejects_provider_transport_from_another_application_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from backend.app.agent_runtime.rag_cost_ledger import _assemble_rag_cost_ledger
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agent_runtime.rag_provider_safety import (
        RagProviderSafetyService,
    )

    application, dedicated = _fake_postgres_engines()
    other_application = create_engine('sqlite+pysqlite:///:memory:')
    other_application.dialect.name = 'postgresql'
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
    monkeypatch.setattr(
        RegisteredAdvisoryLock,
        'matches',
        lambda *_args, **_kwargs: True,
    )
    transport = binding_module._bind_rag_postgres_advisory_transport(
        Session(application),
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    safety = RagProviderSafetyService(
        latch_path=tmp_path / 'different-engine.json',
        identity_secret=b'provider-advisory-consumer-secret',
        designated_environment_id='test',
        advisory_capability=object.__new__(RegisteredAdvisoryLock),
        advisory_transport=transport,
    )
    policy = RagCostPolicy(
        settings=Settings(_env_file=None),
        answer_output_schema_hmac='a' * 64,
        answer_prompt_renderer_hmac='b' * 64,
    )

    with pytest.raises(ValueError, match='advisory transport'):
        _assemble_rag_cost_ledger(
            Session(other_application),
            identity_secret=b'provider-advisory-consumer-secret',
            cost_policy=policy,
            provider_safety=safety,
            provider_connection_factory=transport,
            designated_environment_id='test',
            designated_host_id='pytest-host',
            runtime_health=transport.runtime_health_authority,
        )

    other_application.dispose()
    application.dispose()


def test_provider_advisory_cleanup_uncertainty_poisons_before_escape(
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
    bootstrap = _trusted_bootstrap(monkeypatch, application, dedicated)
    transport = binding_module._bind_rag_postgres_advisory_transport(
        session,
        trusted_bootstrap=bootstrap,
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    capability = object.__new__(RegisteredAdvisoryLock)
    monkeypatch.setattr(
        binding_module,
        'acquire_advisory_lock',
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        binding_module,
        'release_advisory_lock',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError('sensitive unlock failure')
        ),
        raising=False,
    )
    monkeypatch.setattr(
        Connection,
        'invalidate',
        lambda _self: (_ for _ in ()).throw(
            RuntimeError('sensitive invalidate failure')
        ),
    )
    monkeypatch.setattr(
        Connection,
        'close',
        lambda _self: (_ for _ in ()).throw(
            RuntimeError('sensitive close failure')
        ),
    )

    with (
        pytest.raises(TypeError, match='advisory cleanup') as captured,
        transport.advisory(capability, shared=False),
    ):
        pass

    snapshot = transport.runtime_health_authority.snapshot
    assert snapshot.healthy is False
    assert snapshot.failure_count == 1
    assert 'sensitive' not in repr(captured.value)
    assert 'sensitive' not in repr(snapshot)
    with pytest.raises(TypeError, match='runtime health'), transport():
        raise AssertionError('poisoned transport must not connect')


def test_identity_probe_is_least_privilege_and_has_no_control_file_dependency():
    assert 'pg_control_system' not in str(binding_module._IDENTITY_SQL)


def test_binding_rejects_different_authoritative_writable_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    logical_identity = _identity()
    monkeypatch.setattr(
        binding_module,
        '_session_identity',
        lambda _session: logical_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: logical_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_session_server_identity',
        lambda _session: binding_module.RagPostgresWritableServerIdentity(
            server_address='10.0.0.11',
            server_port=5432,
            postmaster_start_time='2026-09-01T00:00:00Z',
        ),
        raising=False,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_server_identity',
        lambda _connection: binding_module.RagPostgresWritableServerIdentity(
            server_address='10.0.0.12',
            server_port=5432,
            postmaster_start_time='2026-09-01T00:00:01Z',
        ),
        raising=False,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(TypeError, match='writable server'):
        _bind_rag_postgres_database(
            session,
            trusted_bootstrap=_trusted_bootstrap(
                monkeypatch,
                application,
                dedicated,
                preserve_server_identity=True,
            ),
            bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
        )

    session.close()
    application.dispose()


@pytest.mark.parametrize(
    'consumer',
    ('provider_safety', 'projection_owner', 'evidence'),
)
def test_each_paid_advisory_consumer_fail_stops_on_unlock_and_close_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    consumer: str,
) -> None:
    from backend.app.agent_runtime.provider_send_fence import (
        _assemble_rag_evidence_barrier,
    )
    from backend.app.agent_runtime.rag_advisory_locks import begin_rag_lock_order
    from backend.app.agent_runtime.rag_cost_ledger import _assemble_rag_cost_ledger
    from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
    from backend.app.agent_runtime.rag_provider_safety import (
        RagProviderSafetyService,
    )

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
    monkeypatch.setattr(
        RegisteredAdvisoryLock,
        'matches',
        lambda *_args, **_kwargs: True,
    )
    transport = binding_module._bind_rag_postgres_advisory_transport(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    capability = object.__new__(RegisteredAdvisoryLock)
    safety = RagProviderSafetyService(
        latch_path=tmp_path / f'{consumer}.json',
        identity_secret=b'provider-advisory-consumer-secret',
        designated_environment_id='test',
        advisory_capability=capability,
        advisory_transport=transport,
    )
    monkeypatch.setattr(
        binding_module,
        'acquire_advisory_lock',
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        binding_module,
        'release_advisory_lock',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError('sensitive unlock detail')
        ),
    )
    cleanup_calls: list[str] = []

    def fail_invalidate(_connection: Connection) -> None:
        cleanup_calls.append('invalidate')
        raise RuntimeError('sensitive invalidate detail')

    def fail_close(_connection: Connection) -> None:
        cleanup_calls.append('close')
        raise RuntimeError('sensitive close detail')

    monkeypatch.setattr(Connection, 'invalidate', fail_invalidate)
    monkeypatch.setattr(Connection, 'close', fail_close)
    yielded: list[str] = []

    with pytest.raises(TypeError, match='advisory cleanup') as captured:
        if consumer == 'provider_safety':
            with (
                transport() as connection,
                safety._registered_advisory(connection),
            ):
                yielded.append(consumer)
        elif consumer == 'evidence':
            barrier = _assemble_rag_evidence_barrier(
                load_current_identity=lambda: 'a' * 64,
                connection_factory=transport,
                registered_lock=capability,
                advisory_transport=transport,
            )
            order = begin_rag_lock_order('ordinary')
            order.acquire('provider_stable_sidecar')
            order.acquire('provider_safety_rows')
            order.acquire('projection_owner')
            evidence_capability = order.acquire('evidence_shared_barrier')
            c5_capability = order.acquire('c5_key_corpus')
            barrier._run(
                expected_identity_hmac='a' * 64,
                operation=lambda: yielded.append(consumer),
                order=order,
                evidence_capability=evidence_capability,
                c5_capability=c5_capability,
            )
        else:
            policy = RagCostPolicy(
                settings=Settings(_env_file=None),
                answer_output_schema_hmac='a' * 64,
                answer_prompt_renderer_hmac='b' * 64,
            )
            ledger = _assemble_rag_cost_ledger(
                session,
                identity_secret=b'provider-advisory-consumer-secret',
                cost_policy=policy,
                provider_safety=safety,
                provider_connection_factory=transport,
                designated_environment_id='test',
                designated_host_id='pytest-host',
                projection_lock_capability_factory=lambda _run_id: capability,
                runtime_health=transport.runtime_health_authority,
            )
            order = begin_rag_lock_order('ordinary')
            order.acquire('provider_stable_sidecar')
            order.acquire('provider_safety_rows')
            owner_capability = order.acquire('projection_owner')
            with ledger.projection_owner_barrier(
                1,
                order=order,
                order_capability=owner_capability,
            ):
                yielded.append(consumer)

    assert yielded == [consumer]
    assert cleanup_calls == ['invalidate', 'close']
    snapshot = transport.runtime_health_authority.snapshot
    assert snapshot.healthy is False
    assert snapshot.failure_count == 1
    assert 'sensitive' not in str(captured.value)


@pytest.mark.parametrize(
    'row',
    (
        ('127.0.0.1', 5432, '2026-09-01T00:00:00Z', True, 'off'),
        ('127.0.0.1', 5432, '2026-09-01T00:00:00Z', False, 'on'),
    ),
)
def test_server_identity_refuses_replica_or_read_only_session(row: tuple) -> None:
    with pytest.raises(TypeError, match='writable PostgreSQL server'):
        binding_module._server_identity_from_row(row)


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


def test_operation_lease_pins_exact_application_connection_through_commit(
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
    pinned: Connection | None = None

    with authority.operation_lease():
        bound = session.get_bind()
        assert isinstance(bound, Connection)
        pinned = bound
        assert session.in_transaction() is True
        session.execute(text('SELECT 1'))
        session.commit()
        assert pinned.closed is False
        assert session.get_bind() is pinned

    assert pinned is not None and pinned.closed is True
    assert session.get_bind() is application
    authority.close()
    session.close()
    application.dispose()


def _file_postgres_engines(tmp_path: Path):
    application = create_engine(
        f'sqlite+pysqlite:///{(tmp_path / "application.db").as_posix()}'
    )
    dedicated = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    with application.begin() as connection:
        connection.execute(text('CREATE TABLE durable_probe (value INTEGER)'))
    application.dialect.name = 'postgresql'
    dedicated.dialect.name = 'postgresql'
    return application, dedicated


def _patch_raw_sql_identity_probe(
    monkeypatch: pytest.MonkeyPatch,
    identity: RagPostgresDatabaseIdentity,
) -> None:
    monkeypatch.setattr(binding_module, '_session_identity', lambda _session: identity)

    def connection_identity(connection: Connection):
        connection.execute(text('SELECT 1'))
        return identity

    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        connection_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )


def test_session_owned_pinned_transaction_commits_physically_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application, dedicated = _file_postgres_engines(tmp_path)
    session = Session(application)
    _patch_raw_sql_identity_probe(monkeypatch, _identity())
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    pinned: Connection | None = None

    with authority.operation_lease():
        pinned = session.connection()
        session.execute(text('INSERT INTO durable_probe VALUES (7)'))
        session.commit()
        assert pinned.in_transaction() is False
        assert pinned.in_nested_transaction() is False
        with application.connect() as independent:
            assert independent.scalar(text('SELECT COUNT(*) FROM durable_probe')) == 1

    assert pinned is not None and pinned.closed is True
    authority.close()
    session.close()
    application.dispose()


def test_clean_prebound_connection_is_enlisted_and_committed_without_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application, dedicated = _file_postgres_engines(tmp_path)
    external = application.connect()
    session = Session(bind=external)
    _patch_raw_sql_identity_probe(monkeypatch, _identity())
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
        assert session.connection() is external
        session.execute(text('INSERT INTO durable_probe VALUES (8)'))
        session.commit()
        assert external.in_transaction() is False
        assert external.in_nested_transaction() is False
        with application.connect() as independent:
            assert independent.scalar(text('SELECT COUNT(*) FROM durable_probe')) == 1

    assert external.closed is False
    authority.close()
    session.close()
    external.close()
    application.dispose()


@pytest.mark.parametrize('nested', (False, True))
def test_prebound_active_transaction_is_rejected_without_touching_caller_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    nested: bool,
) -> None:
    application, dedicated = _file_postgres_engines(tmp_path)
    external = application.connect()
    root = external.begin()
    external.execute(text('INSERT INTO durable_probe VALUES (9)'))
    savepoint = external.begin_nested() if nested else None
    session = Session(bind=external)
    _patch_raw_sql_identity_probe(monkeypatch, _identity())

    with pytest.raises(TypeError, match='fresh.*transaction'):
        _bind_rag_postgres_database(
            session,
            trusted_bootstrap=_trusted_bootstrap(
                monkeypatch,
                application,
                dedicated,
            ),
            bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
        )

    assert root.is_active is True
    assert external.in_transaction() is True
    assert external.in_nested_transaction() is nested
    assert external.scalar(text('SELECT COUNT(*) FROM durable_probe')) == 1
    if savepoint is not None:
        savepoint.rollback()
    root.rollback()
    session.close()
    external.close()
    dedicated.dispose()
    application.dispose()


def test_session_owned_pinned_transaction_rolls_back_without_visibility_or_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application, dedicated = _file_postgres_engines(tmp_path)
    session = Session(application)
    _patch_raw_sql_identity_probe(monkeypatch, _identity())
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    pinned: Connection | None = None

    with (
        pytest.raises(RuntimeError, match='rollback probe'),
        authority.operation_lease(),
    ):
        pinned = session.connection()
        session.execute(text('INSERT INTO durable_probe VALUES (10)'))
        raise RuntimeError('rollback probe')

    assert pinned is not None
    assert pinned.in_transaction() is False
    assert pinned.in_nested_transaction() is False
    assert pinned.closed is True
    with application.connect() as independent:
        assert independent.scalar(text('SELECT COUNT(*) FROM durable_probe')) == 0
    authority.close()
    session.close()
    application.dispose()


def test_operation_lease_rejects_would_be_transaction_server_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated = _fake_postgres_engines()
    session = Session(application)
    logical_identity = _identity()
    server_a = binding_module.RagPostgresWritableServerIdentity(
        server_address='10.0.0.11',
        server_port=5432,
        postmaster_start_time='2026-09-01T00:00:00Z',
    )
    restarted_server_a = binding_module.RagPostgresWritableServerIdentity(
        server_address='10.0.0.11',
        server_port=5432,
        postmaster_start_time='2026-09-01T00:00:01Z',
    )
    connection_servers = iter((server_a, restarted_server_a))
    monkeypatch.setattr(
        binding_module,
        '_session_identity',
        lambda _session: logical_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: logical_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_session_server_identity',
        lambda _session: server_a,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_server_identity',
        lambda _connection: next(connection_servers),
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
            preserve_server_identity=True,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    effects: list[str] = []

    with (
        pytest.raises(TypeError, match='writable PostgreSQL server'),
        authority.operation_lease(),
    ):
        effects.append('mutation')

    assert effects == []
    authority.close()
    session.close()
    application.dispose()


def test_operation_lease_rolls_back_same_pinned_connection_on_error(
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
    pinned: list[Connection] = []

    with (
        pytest.raises(RuntimeError, match='projection failed'),
        authority.operation_lease(),
    ):
        bound = session.get_bind()
        assert isinstance(bound, Connection)
        pinned.append(bound)
        session.execute(text('SELECT 1'))
        raise RuntimeError('projection failed')

    assert len(pinned) == 1 and pinned[0].closed is True
    assert session.in_transaction() is False
    assert session.get_bind() is application
    authority.close()
    session.close()
    application.dispose()


def test_unix_socket_server_identity_is_explicitly_unsupported() -> None:
    with pytest.raises(TypeError, match='writable PostgreSQL server'):
        binding_module._server_identity_from_row(
            (None, None, '2026-09-01T00:00:00Z', False, 'off')
        )


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


class _CommitUnknownProbe(BaseException):
    pass


@pytest.mark.parametrize(
    ('foreign_kind', 'primary'),
    (
        ('cleanup', None),
        ('poison', KeyboardInterrupt('cancelled projection')),
        ('cleanup', ValueError('validation primary')),
        ('poison', _CommitUnknownProbe('commit state unknown')),
    ),
)
def test_owned_operation_close_continues_under_outer_cleanup_owner(
    monkeypatch: pytest.MonkeyPatch,
    foreign_kind: str,
    primary: BaseException | None,
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
    disposed: list[str] = []
    event.listen(dedicated, 'engine_disposed', lambda *_args: disposed.append('dispose'))
    authority = _bind_rag_postgres_database(
        session,
        trusted_bootstrap=_trusted_bootstrap(
            monkeypatch,
            application,
            dedicated,
        ),
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    health = authority.runtime_health_authority
    release_entered = threading.Event()
    resume_release = threading.Event()
    foreign_enqueued = threading.Event()
    foreign_finished = threading.Event()
    order: list[str] = []
    errors: list[BaseException] = []
    products: list[str] = []
    advisory_connections: list[Connection] = []
    foreign_thread_ids: list[int] = []
    invalidations = 0
    closes = 0
    original_release = type(authority)._release_application_transaction
    original_enqueue = type(health)._enqueue_exclusive_ticket
    original_invalidate = Connection.invalidate
    original_close = Connection.close

    def pause_release(self, lease) -> None:
        release_entered.set()
        assert resume_release.wait(timeout=5)
        original_release(self, lease)

    def observe_enqueue(self, *, thread_id: int, purpose: str):
        ticket = original_enqueue(self, thread_id=thread_id, purpose=purpose)
        if foreign_thread_ids and thread_id == foreign_thread_ids[0]:
            foreign_enqueued.set()
        return ticket

    def count_invalidate(connection: Connection, *args, **kwargs):
        nonlocal invalidations
        if connection in advisory_connections:
            invalidations += 1
        return original_invalidate(connection, *args, **kwargs)

    def count_close(connection: Connection, *args, **kwargs):
        nonlocal closes
        if connection in advisory_connections:
            closes += 1
        return original_close(connection, *args, **kwargs)

    monkeypatch.setattr(
        type(authority),
        '_release_application_transaction',
        pause_release,
    )
    monkeypatch.setattr(
        type(health),
        '_enqueue_exclusive_ticket',
        observe_enqueue,
    )
    monkeypatch.setattr(Connection, 'invalidate', count_invalidate)
    monkeypatch.setattr(Connection, 'close', count_close)

    def run_owned_operation() -> None:
        try:
            with authority.owned_operation():
                advisory_connections.append(authority.connect())
                products.append('known_durable_result')
                if primary is not None:
                    raise primary
        except BaseException as error:
            errors.append(error)

    def run_foreign() -> None:
        foreign_thread_ids.append(threading.get_ident())
        if foreign_kind == 'cleanup':
            with health._cleanup_boundary():
                order.append('foreign_cleanup')
        else:
            health._poison()
            order.append('foreign_poison')
        foreign_finished.set()

    owner = threading.Thread(target=run_owned_operation)
    foreign = threading.Thread(target=run_foreign)
    owner.start()
    assert release_entered.wait(timeout=2)
    foreign.start()
    assert foreign_enqueued.wait(timeout=2)
    assert foreign_finished.is_set() is False
    resume_release.set()
    owner.join(timeout=5)
    foreign.join(timeout=5)
    assert owner.is_alive() is False
    assert foreign.is_alive() is False

    closed_before_fallback = authority.closed
    disposed_before_fallback = len(disposed)
    connection_closed_before_fallback = advisory_connections[0].closed
    if not authority.closed:
        authority.close()

    assert products == ['known_durable_result']
    if primary is None:
        assert errors == []
    else:
        assert errors == [primary]
    assert closed_before_fallback is True
    assert disposed_before_fallback == 1
    assert connection_closed_before_fallback is True
    assert invalidations == 1
    assert closes == 1
    assert order == [f'foreign_{foreign_kind}']
    assert health._exclusive_waiters == 0
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert authority._connections == set()
    if foreign_kind == 'poison':
        assert health.snapshot.healthy is False
        with (
            pytest.raises(TypeError, match='runtime health'),
            health._effect('future-admission'),
        ):
            raise AssertionError('poisoned runtime must reject admission')
    session.close()
    application.dispose()


def test_cleanup_owner_capability_rejects_forged_expired_cross_thread_and_gate(
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
    health = authority.runtime_health_authority

    with pytest.raises(TypeError, match='cleanup owner'):
        authority._close_under_cleanup_owner(object())

    cross_thread_errors: list[BaseException] = []
    with health._cleanup_boundary() as health_owner:
        owner = authority._mint_cleanup_owner(health_owner)

        with (
            health._cleanup_boundary(),
            pytest.raises(TypeError, match='cleanup owner'),
        ):
            authority._close_under_cleanup_owner(owner)

        copied_owner = replace(owner)
        with pytest.raises(TypeError, match='cleanup owner'):
            authority._close_under_cleanup_owner(copied_owner)

        def cross_thread_close() -> None:
            try:
                authority._close_under_cleanup_owner(owner)
            except BaseException as error:
                cross_thread_errors.append(error)

        thread = threading.Thread(target=cross_thread_close)
        thread.start()
        thread.join(timeout=5)
        assert thread.is_alive() is False
        assert len(cross_thread_errors) == 1
        assert isinstance(cross_thread_errors[0], TypeError)

        wrong_authority = replace(owner, authority=object())
        with pytest.raises(TypeError, match='cleanup owner'):
            authority._close_under_cleanup_owner(wrong_authority)

    with pytest.raises(TypeError, match='cleanup owner'):
        authority._close_under_cleanup_owner(owner)

    foreign_health = initialization.TrustedPostgresRuntimeHealth(
        _seal=initialization._POSTGRES_RUNTIME_HEALTH_SEAL
    )
    with (
        foreign_health._cleanup_boundary() as foreign_owner,
        pytest.raises(TypeError, match='cleanup owner'),
    ):
        authority._mint_cleanup_owner(foreign_owner)

    authority.close()
    session.close()
    application.dispose()


@pytest.mark.parametrize(
    'primary',
    (None, KeyboardInterrupt('cleanup cancellation primary')),
)
def test_owned_operation_cleanup_failure_poisons_before_escape_with_waiter(
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException | None,
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
    health = authority.runtime_health_authority
    release_entered = threading.Event()
    resume_release = threading.Event()
    foreign_enqueued = threading.Event()
    foreign_finished = threading.Event()
    foreign_thread_ids: list[int] = []
    errors: list[BaseException] = []
    returned_health: list[bool] = []
    advisory_connections: list[Connection] = []
    order: list[str] = []
    dispose_calls = 0
    original_release = type(authority)._release_application_transaction
    original_enqueue = type(health)._enqueue_exclusive_ticket

    def pause_release(self, lease) -> None:
        release_entered.set()
        assert resume_release.wait(timeout=5)
        original_release(self, lease)

    def observe_enqueue(self, *, thread_id: int, purpose: str):
        ticket = original_enqueue(self, thread_id=thread_id, purpose=purpose)
        if foreign_thread_ids and thread_id == foreign_thread_ids[0]:
            foreign_enqueued.set()
        return ticket

    def fail_dispose(*_args, **_kwargs) -> None:
        nonlocal dispose_calls
        dispose_calls += 1
        order.append('dispose_failed')
        raise RuntimeError('sensitive dedicated transport failure')

    monkeypatch.setattr(
        type(authority),
        '_release_application_transaction',
        pause_release,
    )
    monkeypatch.setattr(
        type(health),
        '_enqueue_exclusive_ticket',
        observe_enqueue,
    )
    monkeypatch.setattr(dedicated, 'dispose', fail_dispose)

    def run_owned_operation() -> None:
        try:
            with authority.owned_operation():
                advisory_connections.append(authority.connect())
                if primary is not None:
                    raise primary
            returned_health.append(health.snapshot.healthy)
        except BaseException as error:
            returned_health.append(health.snapshot.healthy)
            errors.append(error)

    def run_foreign_cleanup() -> None:
        foreign_thread_ids.append(threading.get_ident())
        with health._cleanup_boundary():
            order.append('foreign_cleanup')
        foreign_finished.set()

    owner = threading.Thread(target=run_owned_operation)
    foreign = threading.Thread(target=run_foreign_cleanup)
    owner.start()
    assert release_entered.wait(timeout=2)
    foreign.start()
    assert foreign_enqueued.wait(timeout=2)
    resume_release.set()
    owner.join(timeout=5)
    foreign.join(timeout=5)
    assert owner.is_alive() is False
    assert foreign.is_alive() is False
    assert foreign_finished.is_set() is True

    if primary is None:
        assert errors == []
    else:
        assert errors == [primary]
    assert returned_health == [False]
    assert order == ['dispose_failed', 'foreign_cleanup']
    assert dispose_calls == 1
    assert authority.closed is True
    assert advisory_connections[0].closed is True
    assert authority.cleanup_failure is not None
    assert authority.cleanup_failure.code == 'rag_postgres_transport_cleanup_failed'
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert 'sensitive' not in repr(health.snapshot)
    assert health._exclusive_waiters == 0
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    with (
        pytest.raises(TypeError, match='runtime health'),
        health._effect('future-effect'),
    ):
        raise AssertionError('poisoned runtime must refuse future effects')
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
    cleanup_failure = authority.close()

    assert connection.closed is True
    assert authority.closed is True
    assert cleanup_failure is authority.cleanup_failure
    assert cleanup_failure is not None
    assert cleanup_failure.code == 'rag_postgres_transport_cleanup_failed'
    assert 'injected invalidate failure' not in repr(cleanup_failure)
    assert disposed == 1
    assert authority.close() is cleanup_failure
    assert disposed == 1
    session.close()
    application.dispose()


def test_cleanup_failure_poisons_shared_runtime_and_stale_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = create_engine('sqlite+pysqlite:///:memory:')
    dedicated_one = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    dedicated_two = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    dedicated_three = create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    for engine in (application, dedicated_one, dedicated_two, dedicated_three):
        engine.dialect.name = 'postgresql'
    engines = iter((application, dedicated_one, dedicated_two, dedicated_three))
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: next(engines),
    )
    runtime = initialization.initialize_database_runtime(
        'postgresql+psycopg://authority-role@localhost/authority-database'
    )
    bootstrap = runtime.rag_postgres_bootstrap
    assert bootstrap is not None
    logical_identity = _identity()
    server_identity = binding_module.RagPostgresWritableServerIdentity(
        server_address='127.0.0.1',
        server_port=5432,
        postmaster_start_time='2026-09-01T00:00:00Z',
    )
    monkeypatch.setattr(
        binding_module,
        '_session_identity',
        lambda _session: logical_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_identity',
        lambda _connection: logical_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_session_server_identity',
        lambda _session: server_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_connection_server_identity',
        lambda _connection: server_identity,
    )
    monkeypatch.setattr(
        binding_module,
        '_require_bootstrap_capability',
        lambda *_args, **_kwargs: None,
    )
    first_session = Session(application)
    second_session = Session(application)
    third_session = Session(application)
    first = _bind_rag_postgres_database(
        first_session,
        trusted_bootstrap=bootstrap,
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    second = _bind_rag_postgres_database(
        second_session,
        trusted_bootstrap=bootstrap,
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    third = _bind_rag_postgres_database(
        third_session,
        trusted_bootstrap=bootstrap,
        bootstrap_capability=object.__new__(RegisteredAdvisoryLock),
    )
    cleanup_results: list[object] = []

    def fail_dispose(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError('sensitive transport failure')

    monkeypatch.setattr(dedicated_one, 'dispose', fail_dispose)
    monkeypatch.setattr(dedicated_three, 'dispose', fail_dispose)
    product_results: list[object] = []

    class FirstBoundary:
        def __init__(self) -> None:
            self.dispositions: list[object] = []

        def acquire_request_database_authority(self):
            return first.owned_operation()

        def close_request_database_authority(self):
            return first.close()

        def record_request_database_cleanup_failure(self, disposition) -> None:
            self.dispositions.append(disposition)

        def finalize_inter_component_failure(self):
            return SimpleNamespace(outcome='committed_terminal')

    first_boundary = FirstBoundary()
    first_service = RagFinalizationService(
        transaction_boundary=first_boundary,
        settings=Settings(
            _env_file=None,
            agent_runtime_fingerprint_secret='secret',
        ),
    )
    with second.operation_lease():
        first_closer = threading.Thread(
            target=lambda: product_results.append(
                first_service.finalize_inter_component_failure()
            )
        )
        third_closer = threading.Thread(
            target=lambda: cleanup_results.append(third.close())
        )
        first_closer.start()
        third_closer.start()
        first_closer.join(timeout=5)
        third_closer.join(timeout=5)
        assert not first_closer.is_alive() and not third_closer.is_alive()
        assert len(cleanup_results) == 1 and cleanup_results[0] is not None
        assert len(product_results) == 1
        assert product_results[0].outcome == 'committed_terminal'
        assert len(first_boundary.dispositions) == 1
        disposition = first_boundary.dispositions[0]
        assert disposition.operation_state == 'acknowledged_product'
        assert disposition.delivery_permitted is True
        assert disposition.retry_permitted is False
        snapshot = bootstrap.runtime_health_snapshot
        assert snapshot.healthy is False
        assert snapshot.failure_count == 1
        assert snapshot.code == 'rag_postgres_transport_cleanup_failed'
        assert 'sensitive transport failure' not in repr(snapshot)
        with pytest.raises(TypeError, match='runtime health'):
            second.connect()

    first_cleanup = first.close()
    assert first_cleanup is not None
    assert bootstrap.runtime_health_snapshot.failure_count == 1
    with pytest.raises(TypeError, match='runtime health'):
        bootstrap._issue(application)

    class SeparateBoundary:
        def __init__(self) -> None:
            self.effects = 0

        def acquire_request_database_authority(self):
            return second.owned_operation()

        def close_request_database_authority(self):
            return second.close()

        def record_request_database_cleanup_failure(self, _disposition) -> None:
            return None

        def finalize_inter_component_failure(self):
            self.effects += 1
            return SimpleNamespace(outcome='should-not-run')

    separate = SeparateBoundary()
    service = RagFinalizationService(
        transaction_boundary=separate,
        settings=Settings(
            _env_file=None,
            agent_runtime_fingerprint_secret='secret',
        ),
    )
    with pytest.raises(TypeError, match='runtime health'):
        service.finalize_inter_component_failure()
    assert separate.effects == 0

    second.close()
    assert bootstrap.runtime_health_snapshot.failure_count == 1
    first_session.close()
    second_session.close()
    third_session.close()
    application.dispose()


def test_sql_finalization_refuses_all_phase2_effects_when_poison_wins_after_pin(
) -> None:
    pinned = threading.Event()
    resume = threading.Event()
    poisoned = threading.Event()
    effects: list[str] = []
    errors: list[BaseException] = []

    class Authority:
        @contextmanager
        def owned_operation(self):
            pinned.set()
            assert resume.wait(timeout=5)
            yield

        @contextmanager
        def health_effect(self):
            if poisoned.is_set():
                raise TypeError('RAG PostgreSQL runtime health is fail-stopped')
            yield

    boundary = object.__new__(SqlAlchemyRagFinalizationBoundary)
    boundary._postgres_database = Authority()

    def finalize() -> None:
        try:
            with boundary.acquire_request_database_authority():
                effects.extend(
                    ('advisory', 'safety', 'evidence', 'c5', 'retrieval', 'mutation')
                )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=finalize)
    worker.start()
    assert pinned.wait(timeout=5)
    poisoned.set()
    resume.set()
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert effects == []
    assert len(errors) == 1
    assert isinstance(errors[0], TypeError)


class _CleanupStateMachineFault(BaseException):
    pass


class _CleanupBodyCommitUnknown(BaseException):
    pass


@pytest.mark.parametrize(
    'fault_stage',
    (
        'enter_before',
        'mint_before',
        'mint_after',
        'release_before',
        'release_after',
        'close_before',
        'close_after',
        'exit_before',
        'exit_after',
    ),
)
@pytest.mark.parametrize(
    'body_outcome',
    ('success', 'validation', 'keyboard_interrupt', 'commit_unknown'),
)
def test_owned_operation_cleanup_state_machine_preserves_body_and_drains(
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
    body_outcome: str,
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
    health = authority.runtime_health_authority
    advisory_connections: list[Connection] = []
    dispose_calls = 0

    def record_dispose(*_args: object) -> None:
        nonlocal dispose_calls
        dispose_calls += 1

    event.listen(dedicated, 'engine_disposed', record_dispose)

    stage_owner = health if fault_stage.startswith(('enter_', 'exit_')) else authority
    stage_name = {
        'enter_before': '_enter_cleanup',
        'mint_before': '_mint_cleanup_owner',
        'mint_after': '_mint_cleanup_owner',
        'release_before': '_release_application_transaction',
        'release_after': '_release_application_transaction',
        'close_before': '_close_under_cleanup_owner',
        'close_after': '_close_under_cleanup_owner',
        'exit_before': '_exit_cleanup',
        'exit_after': '_exit_cleanup',
    }[fault_stage]
    timing = fault_stage.rsplit('_', 1)[1]
    original = getattr(type(stage_owner), stage_name)
    injected = False

    def one_shot(self, *args, **kwargs):
        nonlocal injected
        should_inject = not injected
        if should_inject:
            injected = True
        if should_inject and timing == 'before':
            raise _CleanupStateMachineFault('secret cleanup failure')
        result = original(self, *args, **kwargs)
        if should_inject and timing == 'after':
            raise _CleanupStateMachineFault('secret cleanup failure')
        return result

    monkeypatch.setattr(type(stage_owner), stage_name, one_shot)
    primary: BaseException | None = {
        'success': None,
        'validation': ValueError('validation primary'),
        'keyboard_interrupt': KeyboardInterrupt('cancellation primary'),
        'commit_unknown': _CleanupBodyCommitUnknown('commit unknown primary'),
    }[body_outcome]
    returned: list[str] = []
    raised: list[BaseException] = []

    try:
        with authority.owned_operation():
            advisory_connections.append(authority.connect())
            if primary is not None:
                raise primary
            returned.append('durable-product')
    except BaseException as exc:
        raised.append(exc)

    assert injected is True
    if primary is None:
        assert returned == ['durable-product']
        assert raised == []
    else:
        assert raised == [primary]
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert advisory_connections[0].closed is True
    assert dispose_calls == 1
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert 'secret' not in repr(health.snapshot)
    assert health._exclusive_waiters == 0
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert tuple(health._exclusive_tickets) == ()
    cleanup_owner = authority._cleanup_owner_capability
    assert cleanup_owner is None or cleanup_owner.health_owner.active is False
    with (
        pytest.raises(TypeError, match='runtime health'),
        authority.owned_operation(),
    ):
        raise AssertionError('poisoned authority must refuse future work')
    session.close()
    application.dispose()


def test_owned_operation_retries_cleanup_responsibility_after_queued_cancel(
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
    health = authority.runtime_health_authority
    foreign_entered = threading.Event()
    release_foreign = threading.Event()
    queued_cancelled = threading.Event()
    owner_thread_id: list[int] = []
    returned: list[str] = []
    errors: list[BaseException] = []
    body_ready = threading.Event()
    allow_body_exit = threading.Event()

    def hold_foreign_cleanup() -> None:
        with health._cleanup_boundary():
            foreign_entered.set()
            assert release_foreign.wait(timeout=5)

    original_enter = type(health)._enter_cleanup
    cancelled_once = False

    def cancel_after_enqueue(self):
        nonlocal cancelled_once
        if threading.get_ident() == owner_thread_id[0] and not cancelled_once:
            cancelled_once = True
            with self._condition:
                ticket = self._enqueue_exclusive_ticket(
                    thread_id=threading.get_ident(),
                    purpose='cleanup',
                )
                assert self._exclusive_waiters == 1
                self._cancel_exclusive_ticket(ticket)
            queued_cancelled.set()
            assert release_foreign.wait(timeout=5)
            raise KeyboardInterrupt('queued cleanup cancellation')
        return original_enter(self)

    monkeypatch.setattr(type(health), '_enter_cleanup', cancel_after_enqueue)

    def run_owned() -> None:
        owner_thread_id.append(threading.get_ident())
        try:
            with authority.owned_operation():
                authority.connect()
                body_ready.set()
                assert allow_body_exit.wait(timeout=5)
                returned.append('durable-product')
        except BaseException as exc:
            errors.append(exc)

    owner = threading.Thread(target=run_owned)
    owner.start()
    assert body_ready.wait(timeout=2)
    foreign = threading.Thread(target=hold_foreign_cleanup)
    foreign.start()
    assert foreign_entered.wait(timeout=2)
    allow_body_exit.set()
    assert queued_cancelled.wait(timeout=2)
    release_foreign.set()
    owner.join(timeout=5)
    foreign.join(timeout=5)

    assert owner.is_alive() is False
    assert foreign.is_alive() is False
    assert errors == []
    assert returned == ['durable-product']
    assert authority.closed is True
    assert authority._active_leases == 0
    assert session.get_bind() is application
    assert health.snapshot.healthy is False
    assert health._exclusive_waiters == 0
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert tuple(health._exclusive_tickets) == ()
    session.close()
    application.dispose()


def test_emergency_cleanup_state_is_exact_thread_bound_and_expired(
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
    captured_states: list[object] = []
    cross_thread_errors: list[BaseException] = []

    with authority.owned_operation():
        state = authority._emergency_cleanup_state
        assert state is not None
        captured_states.append(state)
        forged = replace(state)
        with pytest.raises(TypeError, match='emergency cleanup state'):
            authority._mark_cleanup_uncertain(forged)

        def cross_thread_use() -> None:
            try:
                authority._mark_cleanup_uncertain(state)
            except BaseException as exc:
                cross_thread_errors.append(exc)

        thread = threading.Thread(target=cross_thread_use)
        thread.start()
        thread.join(timeout=5)
        assert thread.is_alive() is False

    state = captured_states[0]
    assert len(cross_thread_errors) == 1
    assert isinstance(cross_thread_errors[0], TypeError)
    assert state.active is False
    assert state.health_capability.active is False
    with pytest.raises(TypeError, match='emergency cleanup state'):
        authority._mark_cleanup_uncertain(state)
    session.close()
    application.dispose()


@pytest.mark.parametrize(
    'fault_point',
    (
        'session_rollback',
        'session_bind_restore',
        'application_close',
        'owner_validate',
        'advisory_close',
        'dedicated_dispose',
    ),
)
@pytest.mark.parametrize(
    'body_outcome',
    ('success', 'keyboard_interrupt', 'commit_unknown'),
)
def test_owned_operation_lower_cleanup_faults_are_secondary_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
    body_outcome: str,
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
    health = authority.runtime_health_authority
    application_connection: list[Connection] = []
    advisory_connection: list[Connection] = []
    dispose_calls = 0
    fault_calls = 0

    def record_dispose(*_args: object) -> None:
        nonlocal dispose_calls
        dispose_calls += 1

    event.listen(dedicated, 'engine_disposed', record_dispose)

    if fault_point == 'session_rollback':
        original_rollback = Session.rollback

        def rollback_once(self):
            nonlocal fault_calls
            if self is session and fault_calls == 0:
                fault_calls += 1
                raise _CleanupStateMachineFault('secret rollback failure')
            return original_rollback(self)

        monkeypatch.setattr(Session, 'rollback', rollback_once)
    elif fault_point == 'session_bind_restore':
        original_setattr = Session.__setattr__

        def setattr_once(self, name, value):
            nonlocal fault_calls
            if (
                self is session
                and name == 'bind'
                and value is application
                and fault_calls == 0
            ):
                fault_calls += 1
                raise _CleanupStateMachineFault('secret bind failure')
            return original_setattr(self, name, value)

        monkeypatch.setattr(Session, '__setattr__', setattr_once)
    elif fault_point in {'application_close', 'advisory_close'}:
        original_close = Connection.close

        def close_once(self):
            nonlocal fault_calls
            target = (
                application_connection
                if fault_point == 'application_close'
                else advisory_connection
            )
            if target and self is target[0] and fault_calls == 0:
                fault_calls += 1
                raise _CleanupStateMachineFault('secret close failure')
            return original_close(self)

        monkeypatch.setattr(Connection, 'close', close_once)
    elif fault_point == 'owner_validate':
        original_require_owner = type(health)._require_cleanup_owner
        owner_checks = 0

        def validate_once(self, capability, *, outermost=False):
            nonlocal fault_calls, owner_checks
            owner_checks += 1
            if owner_checks == 2:
                fault_calls += 1
                raise _CleanupStateMachineFault('secret owner failure')
            return original_require_owner(
                self,
                capability,
                outermost=outermost,
            )

        monkeypatch.setattr(
            type(health),
            '_require_cleanup_owner',
            validate_once,
        )
    else:
        original_dispose = dedicated.dispose

        def dispose_once(*args, **kwargs):
            nonlocal fault_calls
            result = original_dispose(*args, **kwargs)
            if fault_calls == 0:
                fault_calls += 1
                raise _CleanupStateMachineFault('secret dispose failure')
            return result

        monkeypatch.setattr(dedicated, 'dispose', dispose_once)

    primary: BaseException | None = {
        'success': None,
        'keyboard_interrupt': KeyboardInterrupt('cancellation primary'),
        'commit_unknown': _CleanupBodyCommitUnknown('commit unknown primary'),
    }[body_outcome]
    returned: list[str] = []
    raised: list[BaseException] = []

    try:
        with authority.owned_operation():
            lease = authority._lease_context.get()
            assert lease is not None
            application_connection.append(lease.application_connection)
            advisory_connection.append(authority.connect())
            if primary is not None:
                raise primary
            returned.append('durable-product')
    except BaseException as exc:
        raised.append(exc)

    assert fault_calls == 1
    if primary is None:
        assert returned == ['durable-product']
        assert raised == []
    else:
        assert raised == [primary]
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert application_connection[0].closed is True
    assert advisory_connection[0].closed is True
    assert dispose_calls == 1
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert 'secret' not in repr(health.snapshot)
    assert health._exclusive_waiters == 0
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    session.close()
    application.dispose()
