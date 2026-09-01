from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager, suppress
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
    first_started = threading.Event()
    third_started = threading.Event()

    def close_first() -> None:
        first_started.set()
        product_results.append(
            first_service.finalize_inter_component_failure()
        )

    def close_third() -> None:
        third_started.set()
        cleanup_results.append(third.close())

    with second.operation_lease():
        first_closer = threading.Thread(target=close_first)
        third_closer = threading.Thread(target=close_third)
        first_closer.start()
        third_closer.start()
        assert first_started.wait(timeout=2)
        assert third_started.wait(timeout=2)
        first_closer.join(timeout=0.1)
        third_closer.join(timeout=0.1)
        assert first_closer.is_alive() and third_closer.is_alive()

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
    original_exit_effect = type(health)._exit_effect
    cancelled_once = False

    def cancel_after_enqueue(self, *args, **kwargs):
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
        return original_enter(self, *args, **kwargs)

    def pause_after_effect_exit(self):
        original_exit_effect(self)
        if (
            threading.get_ident() == owner_thread_id[0]
            and body_ready.is_set()
            and self._effect_depths.get(owner_thread_id[0], 0) == 0
        ):
            assert foreign_entered.wait(timeout=5)

    monkeypatch.setattr(type(health), '_enter_cleanup', cancel_after_enqueue)
    monkeypatch.setattr(type(health), '_exit_effect', pause_after_effect_exit)

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
    allow_body_exit.set()
    assert foreign_entered.wait(timeout=2)
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
    assert (
        authority.runtime_health_authority
        ._emergency_cleanup_capability_is_active(state.health_capability)
        is False
    )
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


def _cleanup_responsibility_authority(
    monkeypatch: pytest.MonkeyPatch,
):
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
    return application, dedicated, session, authority


@pytest.mark.parametrize(
    'primary',
    (None, KeyboardInterrupt('pin-race cancellation primary')),
)
def test_owned_operation_pin_is_inside_compound_health_effect_before_poison(
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException | None,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    pinned = threading.Event()
    resume = threading.Event()
    poison_started = threading.Event()
    poison_finished = threading.Event()
    body_effects: list[str] = []
    returned: list[str] = []
    errors: list[BaseException] = []
    pinned_connections: list[Connection] = []
    original_pin = type(authority)._pin_application_transaction

    def pause_after_pin(self, health_lease, *args, **kwargs):
        lease = original_pin(self, health_lease, *args, **kwargs)
        pinned_connections.append(lease.application_connection)
        pinned.set()
        assert resume.wait(timeout=5)
        return lease

    monkeypatch.setattr(
        type(authority),
        '_pin_application_transaction',
        pause_after_pin,
    )

    def poison_runtime() -> None:
        poison_started.set()
        health._poison()
        poison_finished.set()

    def run_owned() -> None:
        try:
            with authority.owned_operation():
                body_effects.append('body')
                if primary is not None:
                    raise primary
                returned.append('durable-product')
        except BaseException as exc:
            errors.append(exc)

    owner = threading.Thread(target=run_owned)
    owner.start()
    assert pinned.wait(timeout=2)
    poisoner = threading.Thread(target=poison_runtime)
    poisoner.start()
    assert poison_started.wait(timeout=2)
    resume.set()
    owner.join(timeout=5)
    poisoner.join(timeout=5)

    assert owner.is_alive() is False
    assert poisoner.is_alive() is False
    assert poison_finished.is_set() is True
    assert body_effects == ['body']
    if primary is None:
        assert returned == ['durable-product']
        assert errors == []
    else:
        assert errors == [primary]
    assert health.snapshot.healthy is False
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert pinned_connections[0].closed is True
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    session.close()
    application.dispose()


@pytest.mark.parametrize('timing', ('before', 'after'))
def test_emergency_responsibility_mint_fault_is_sanitized_and_drained(
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_mint = type(health)._mint_emergency_cleanup_capability
    injected = False

    def mint_once(self, operation_lease, *args, **kwargs):
        nonlocal injected
        should_inject = not injected
        injected = True
        if should_inject and timing == 'before':
            raise _CleanupStateMachineFault('secret mint failure')
        capability = original_mint(self, operation_lease, *args, **kwargs)
        if should_inject and timing == 'after':
            raise _CleanupStateMachineFault('secret mint failure')
        return capability

    monkeypatch.setattr(
        type(health),
        '_mint_emergency_cleanup_capability',
        mint_once,
    )
    body_effects: list[str] = []

    with (
        pytest.raises(TypeError, match='runtime health') as captured,
        authority.owned_operation(),
    ):
        body_effects.append('must-not-run')

    assert injected is True
    assert 'secret' not in str(captured.value)
    assert body_effects == []
    assert health.snapshot.healthy is False
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    session.close()
    application.dispose()


@pytest.mark.parametrize('normal_hook_behavior', ('raise', 'noop'))
def test_persistent_normal_poison_and_finish_faults_use_sealed_fallback(
    monkeypatch: pytest.MonkeyPatch,
    normal_hook_behavior: str,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_release = type(authority)._release_application_transaction
    original_fail_stop = type(health)._emergency_fail_stop
    captured_capabilities: list[object] = []
    release_faults = 0

    def fail_release(self, lease) -> None:
        nonlocal release_faults
        release_faults += 1
        raise _CleanupStateMachineFault('secret release failure')

    def fail_stop_hook(*_args, **_kwargs) -> None:
        if normal_hook_behavior == 'raise':
            raise _CleanupStateMachineFault('secret poison hook failure')

    def finish_hook(_self, capability) -> None:
        captured_capabilities.append(capability)
        if normal_hook_behavior == 'raise':
            raise _CleanupStateMachineFault('secret finish hook failure')

    monkeypatch.setattr(
        type(authority),
        '_release_application_transaction',
        fail_release,
    )
    monkeypatch.setattr(type(health), '_emergency_fail_stop', fail_stop_hook)
    monkeypatch.setattr(
        type(health),
        '_finish_emergency_cleanup',
        finish_hook,
    )
    returned: list[str] = []

    with authority.owned_operation():
        authority.connect()
        returned.append('durable-product')

    assert returned == ['durable-product']
    assert release_faults == 1
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    assert len(captured_capabilities) >= 1
    assert (
        health._emergency_cleanup_capability_is_active(
            captured_capabilities[0]
        )
        is False
    )
    monkeypatch.setattr(
        type(health),
        '_emergency_fail_stop',
        original_fail_stop,
    )
    with pytest.raises(TypeError, match='emergency cleanup capability'):
        health._emergency_fail_stop(captured_capabilities[0])
    monkeypatch.setattr(
        type(authority),
        '_release_application_transaction',
        original_release,
    )
    assert authority.close() is authority.cleanup_failure
    session.close()
    application.dispose()


@pytest.mark.parametrize('timing', ('before', 'after'))
def test_emergency_state_install_fault_precedes_all_external_effects(
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_install = type(authority)._install_emergency_cleanup_state
    injected = False
    body_effects: list[str] = []

    def install_once(self, capability, *args, **kwargs):
        nonlocal injected
        should_inject = not injected
        injected = True
        if should_inject and timing == 'before':
            raise _CleanupStateMachineFault('secret install failure')
        state = original_install(self, capability, *args, **kwargs)
        if should_inject and timing == 'after':
            raise _CleanupStateMachineFault('secret install failure')
        return state

    monkeypatch.setattr(
        type(authority),
        '_install_emergency_cleanup_state',
        install_once,
    )

    with (
        pytest.raises(TypeError, match='runtime health') as captured,
        authority.owned_operation(),
    ):
        body_effects.append('must-not-run')

    assert injected is True
    assert 'secret' not in str(captured.value)
    assert body_effects == []
    assert health.snapshot.healthy is False
    assert authority.closed is True
    assert authority._active_leases == 0
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    session.close()
    application.dispose()


@pytest.mark.parametrize('timing', ('before', 'after'))
@pytest.mark.parametrize(
    'primary',
    (None, KeyboardInterrupt('snapshot cancellation primary')),
)
def test_connection_snapshot_fault_preserves_body_and_cleanup_responsibility(
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
    primary: BaseException | None,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_capture = type(authority)._capture_emergency_connections
    injected = False
    returned: list[str] = []
    errors: list[BaseException] = []

    def capture_once(self, state):
        nonlocal injected
        should_inject = not injected
        injected = True
        if should_inject and timing == 'before':
            raise _CleanupStateMachineFault('secret snapshot failure')
        result = original_capture(self, state)
        if should_inject and timing == 'after':
            raise _CleanupStateMachineFault('secret snapshot failure')
        return result

    monkeypatch.setattr(
        type(authority),
        '_capture_emergency_connections',
        capture_once,
    )

    try:
        with authority.owned_operation():
            authority.connect()
            if primary is not None:
                raise primary
            returned.append('durable-product')
    except BaseException as exc:
        errors.append(exc)

    assert injected is True
    if primary is None:
        assert returned == ['durable-product']
        assert errors == []
    else:
        assert errors == [primary]
    assert health.snapshot.healthy is False
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    session.close()
    application.dispose()


def test_emergency_capability_attests_exact_active_operation_and_revokes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    captured: list[object] = []

    with health._operation('rag_finalization_or_recovery') as operation_lease:
        for forged in (
            replace(operation_lease),
            replace(operation_lease, active=False),
            replace(operation_lease, epoch=operation_lease.epoch + 1),
            replace(operation_lease, purpose='wrong-purpose'),
            replace(operation_lease, _seal=object()),
        ):
            with pytest.raises(TypeError, match='runtime health|cleanup capability'):
                health._force_mint_emergency_cleanup_capability(
                    forged,
                    authority=authority,
                )
        capability = health._force_mint_emergency_cleanup_capability(
            operation_lease,
            authority=authority,
        )
        captured.append(capability)
        with pytest.raises(TypeError, match='emergency cleanup capability'):
            health._require_emergency_cleanup_capability(
                replace(capability),
                authority=authority,
            )
        with pytest.raises(TypeError, match='emergency cleanup capability'):
            health._require_emergency_cleanup_capability(
                capability,
                authority=object(),
            )
        health._force_finish_emergency_cleanup(capability)
        assert health._emergency_cleanup_capability_is_active(capability) is False
        assert tuple(health._emergency_cleanup_capabilities.values()) == ()

    capability = captured[0]
    with pytest.raises(TypeError, match='emergency cleanup capability'):
        health._force_emergency_fail_stop(capability)
    with health._operation('rag_finalization_or_recovery') as later_operation:
        later = health._force_mint_emergency_cleanup_capability(
            later_operation,
            authority=authority,
        )
        with pytest.raises(TypeError, match='emergency cleanup capability'):
            health._force_emergency_fail_stop(capability)
        health._force_finish_emergency_cleanup(later)

    session.close()
    application.dispose()


def test_emergency_revocation_clears_only_exact_ticket_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority

    with health._operation('rag_finalization_or_recovery') as operation_lease:
        capability = health._force_mint_emergency_cleanup_capability(
            operation_lease,
            authority=authority,
        )
        owner = health._enter_cleanup(
            emergency_capability=capability,
        )
        with health._condition:
            later = health._enqueue_exclusive_ticket(
                thread_id=threading.get_ident(),
                purpose='cleanup',
            )
        health._force_finish_emergency_cleanup(capability)

        assert owner.active is False
        assert health._exclusive_owner is None
        assert health._exclusive_depth == 0
        assert tuple(health._exclusive_tickets) == (later,)
        assert health._exclusive_waiters == 1
        with health._condition:
            health._cancel_exclusive_ticket(later)
        with health._cleanup_boundary():
            assert health._exclusive_owner == threading.get_ident()

    assert health._exclusive_waiters == 0
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    session.close()
    application.dispose()


def test_finish_after_revoke_fault_uses_current_operation_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_finish = type(health)._finish_emergency_cleanup
    captured: list[object] = []

    def finish_after(self, capability) -> None:
        original_finish(self, capability)
        captured.append(capability)
        raise _CleanupStateMachineFault('secret post-revoke failure')

    monkeypatch.setattr(
        type(health),
        '_finish_emergency_cleanup',
        finish_after,
    )
    returned: list[str] = []

    with authority.owned_operation():
        returned.append('durable-product')

    assert returned == ['durable-product']
    assert len(captured) == 1
    assert health._emergency_cleanup_capability_is_active(captured[0]) is False
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    assert tuple(health._active_operation_authorities.values()) == ()
    assert authority.closed is True
    assert authority._active_leases == 0
    assert session.get_bind() is application
    session.close()
    application.dispose()


@pytest.mark.parametrize(
    'checkpoint',
    ('after_connect', 'after_context', 'after_count', 'after_token'),
)
def test_pin_publication_fault_is_sanitized_and_fully_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    body_effects: list[str] = []
    injected = False

    def publication_checkpoint(self, current: str) -> None:
        nonlocal injected
        if current == checkpoint and not injected:
            injected = True
            raise _CleanupStateMachineFault('secret publication failure')

    monkeypatch.setattr(
        type(authority),
        '_pin_publication_checkpoint',
        publication_checkpoint,
        raising=False,
    )

    with (
        pytest.raises(TypeError, match='runtime health') as captured,
        authority.owned_operation(),
    ):
        body_effects.append('must-not-run')

    assert injected is True
    assert 'secret' not in str(captured.value)
    assert body_effects == []
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    session.close()
    application.dispose()


def test_emergency_capability_mutation_cannot_retarget_cleanup_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import FrozenInstanceError

    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority

    with health._operation('rag_finalization_or_recovery') as operation_lease:
        capability = health._force_mint_emergency_cleanup_capability(
            operation_lease,
            authority=authority,
        )
        owner = health._enter_cleanup(
            emergency_capability=capability,
        )
        with health._condition:
            later = health._enqueue_exclusive_ticket(
                thread_id=threading.get_ident(),
                purpose='cleanup',
            )
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            capability.cleanup_ticket = later
        with suppress(AttributeError, TypeError):
            object.__setattr__(capability, 'cleanup_ticket', later)
        health._force_finish_emergency_cleanup(capability)

        assert owner.active is False
        assert tuple(health._exclusive_tickets) == (later,)
        assert health._exclusive_waiters == 1
        with health._condition:
            health._cancel_exclusive_ticket(later)

    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    session.close()
    application.dispose()


@pytest.mark.parametrize('cleanup_fault', ('release', 'close'))
@pytest.mark.parametrize(
    'primary',
    (
        None,
        ValueError('finish matrix validation primary'),
        KeyboardInterrupt('finish matrix cancellation primary'),
        _CleanupBodyCommitUnknown('finish matrix commit unknown primary'),
    ),
)
def test_prior_poison_then_finish_after_revoke_preserves_outcome_and_drain(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fault: str,
    primary: BaseException | None,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_finish = type(health)._finish_emergency_cleanup
    original_release = type(authority)._release_application_transaction
    original_close = type(authority)._close_under_cleanup_owner
    injected_cleanup = False
    captured_capabilities: list[object] = []
    returned: list[str] = []
    errors: list[BaseException] = []

    def release_once(self, lease) -> None:
        nonlocal injected_cleanup
        if cleanup_fault == 'release' and not injected_cleanup:
            injected_cleanup = True
            raise _CleanupStateMachineFault('secret release fault')
        original_release(self, lease)

    def close_once(self, cleanup_owner):
        nonlocal injected_cleanup
        if cleanup_fault == 'close' and not injected_cleanup:
            injected_cleanup = True
            raise _CleanupStateMachineFault('secret close fault')
        return original_close(self, cleanup_owner)

    def finish_after(self, capability) -> None:
        original_finish(self, capability)
        captured_capabilities.append(capability)
        raise _CleanupStateMachineFault('secret finish-after-revoke fault')

    monkeypatch.setattr(
        type(authority),
        '_release_application_transaction',
        release_once,
    )
    monkeypatch.setattr(
        type(authority),
        '_close_under_cleanup_owner',
        close_once,
    )
    monkeypatch.setattr(
        type(health),
        '_finish_emergency_cleanup',
        finish_after,
    )

    try:
        with authority.owned_operation():
            authority.connect()
            if primary is not None:
                raise primary
            returned.append('durable-product')
    except BaseException as exc:
        errors.append(exc)

    assert injected_cleanup is True
    if primary is None:
        assert returned == ['durable-product']
        assert errors == []
    else:
        assert errors == [primary]
    assert len(captured_capabilities) == 1
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert authority.closed is True
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert authority._emergency_cleanup_state is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert health._exclusive_waiters == 0
    session.close()
    application.dispose()


def test_checkout_ownership_is_registered_below_engine_connect_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_connect = type(application).connect
    captured_connections: list[Connection] = []
    checkins: list[str] = []
    checkout_proxies: list[object] = []

    event.listen(application.pool, 'checkin', lambda *_args: checkins.append('checkin'))
    event.listen(
        application.pool,
        'checkout',
        lambda _dbapi, _record, proxy: checkout_proxies.append(proxy),
    )

    def connector_fault(self) -> Connection:
        connection = original_connect(self)
        if self is application:
            captured_connections.append(connection)
            raise _CleanupStateMachineFault('secret connector return fault')
        return connection

    monkeypatch.setattr(type(application), 'connect', connector_fault)

    with (
        pytest.raises(TypeError, match='runtime health') as captured,
        authority.owned_operation(),
    ):
        raise AssertionError('body must not run')

    assert 'secret' not in str(captured.value)
    assert len(captured_connections) == 1
    assert checkins == ['checkin']
    assert len(checkout_proxies) == 1
    assert checkout_proxies[0].is_valid is False
    registry = authority._assembly.trusted_bootstrap._checkout_registry
    assert registry._pending == {}
    assert registry._captured == {}
    assert health.snapshot.healthy is False
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert authority._emergency_cleanup_state is None
    with (
        pytest.raises(TypeError, match='fail-stopped'),
        health._operation('later-effect'),
    ):
        pass
    with suppress(BaseException):
        captured_connections[0].close()
    session.close()
    application.dispose()


@pytest.mark.parametrize(
    'checkpoint',
    (
        'after_ticket_append',
        'before_ticket_record',
        'after_ticket_record',
        'after_wait',
        'after_claim',
    ),
)
def test_emergency_cleanup_ticket_transition_rolls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    injected = False

    def transition_checkpoint(self, current: str) -> None:
        nonlocal injected
        if current == checkpoint and not injected:
            injected = True
            raise _CleanupStateMachineFault('secret ticket transition fault')

    monkeypatch.setattr(
        type(health),
        '_emergency_cleanup_transition_checkpoint',
        transition_checkpoint,
        raising=False,
    )

    with health._operation('rag_finalization_or_recovery') as operation_lease:
        capability = health._force_mint_emergency_cleanup_capability(
            operation_lease,
            authority=authority,
        )
        foreign = None
        with health._condition:
            foreign = health._enqueue_exclusive_ticket(
                thread_id=threading.get_ident() + 1,
                purpose='cleanup',
            )
            health._cancel_exclusive_ticket(foreign)
        with pytest.raises(_CleanupStateMachineFault):
            health._enter_cleanup(emergency_capability=capability)
        assert injected is True
        assert tuple(health._exclusive_tickets) == ()
        assert health._exclusive_waiters == 0
        assert health._exclusive_owner is None
        assert health._exclusive_depth == 0
        record = health._emergency_cleanup_record(capability)
        assert record.cleanup_ticket is None
        assert record.cleanup_generation is None
        health._force_finish_emergency_cleanup(capability)

    session.close()
    application.dispose()


def test_generic_emergency_cleanup_record_updater_is_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority

    assert not hasattr(health, '_update_emergency_cleanup_record')

    session.close()
    application.dispose()


def test_typed_cleanup_transitions_reject_cross_thread_and_cross_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    seal = initialization._POSTGRES_EMERGENCY_TRANSITION_SEAL
    cross_thread_errors: list[BaseException] = []

    with (
        health._operation('rag_finalization_or_recovery') as first_lease,
        health._operation('rag_finalization_or_recovery') as second_lease,
    ):
        first = health._force_mint_emergency_cleanup_capability(
            first_lease,
            authority=authority,
        )
        second = health._force_mint_emergency_cleanup_capability(
            second_lease,
            authority=authority,
        )
        before = (
            tuple(health._exclusive_tickets),
            health._emergency_cleanup_record(first),
            health._emergency_cleanup_record(second),
        )
        with pytest.raises(TypeError, match='transition'):
            health._begin_emergency_cleanup_ticket(first, _seal=object())
        assert (
            tuple(health._exclusive_tickets),
            health._emergency_cleanup_record(first),
            health._emergency_cleanup_record(second),
        ) == before

        def cross_thread_transition() -> None:
            try:
                health._begin_emergency_cleanup_ticket(first, _seal=seal)
            except BaseException as exc:
                cross_thread_errors.append(exc)

        thread = threading.Thread(target=cross_thread_transition)
        thread.start()
        thread.join(timeout=5)
        assert thread.is_alive() is False
        assert len(cross_thread_errors) == 1
        assert isinstance(cross_thread_errors[0], TypeError)
        assert (
            tuple(health._exclusive_tickets),
            health._emergency_cleanup_record(first),
            health._emergency_cleanup_record(second),
        ) == before

        ticket = health._begin_emergency_cleanup_ticket(first, _seal=seal)
        queued = (
            tuple(health._exclusive_tickets),
            health._emergency_cleanup_record(first),
            health._emergency_cleanup_record(second),
        )
        with pytest.raises(TypeError, match='ticket'):
            health._claim_emergency_cleanup_ticket(second, ticket, _seal=seal)
        assert (
            tuple(health._exclusive_tickets),
            health._emergency_cleanup_record(first),
            health._emergency_cleanup_record(second),
        ) == queued
        health._rollback_emergency_cleanup_transition(first, ticket, _seal=seal)
        health._force_finish_emergency_cleanup(first)
        health._force_finish_emergency_cleanup(second)

    assert tuple(health._exclusive_tickets) == ()
    assert health._exclusive_waiters == 0
    session.close()
    application.dispose()


def test_revoked_cleanup_disposition_ignores_mutable_state_lease_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    original_finish = type(health)._finish_emergency_cleanup
    wrong_context = health._operation('rag_finalization_or_recovery')
    wrong_lease = wrong_context.__enter__()

    def finish_after_revoke(self, capability) -> None:
        original_finish(self, capability)
        state = authority._emergency_cleanup_state
        assert state is not None
        if hasattr(state, 'runtime_health_lease'):
            state.runtime_health_lease = wrong_lease
        raise _CleanupStateMachineFault('secret finish-after-revoke fault')

    monkeypatch.setattr(type(health), '_finish_emergency_cleanup', finish_after_revoke)
    returned: list[str] = []
    try:
        with authority.owned_operation():
            returned.append('durable-product')
    finally:
        wrong_context.__exit__(None, None, None)

    assert returned == ['durable-product']
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1
    assert authority._emergency_cleanup_state is None
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    session.close()
    application.dispose()


@pytest.mark.parametrize('fault_point', ('active_probe', 'state_clear'))
@pytest.mark.parametrize(
    'primary',
    (
        None,
        ValueError('terminal drain validation primary'),
        KeyboardInterrupt('terminal drain cancellation primary'),
        _CleanupBodyCommitUnknown('terminal drain commit unknown primary'),
    ),
)
def test_terminal_drain_helper_fault_never_masks_body_outcome(
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
    primary: BaseException | None,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    injected = False
    original_probe = type(health)._emergency_cleanup_capability_is_active

    def probe_once(self, capability) -> bool:
        nonlocal injected
        if fault_point == 'active_probe' and not injected:
            injected = True
            raise _CleanupStateMachineFault('secret active probe fault')
        return original_probe(self, capability)

    def terminal_checkpoint(self, stage: str) -> None:
        nonlocal injected
        if fault_point == 'state_clear' and stage == 'before_state_clear' and not injected:
            injected = True
            raise _CleanupStateMachineFault('secret state clear fault')

    monkeypatch.setattr(
        type(health),
        '_emergency_cleanup_capability_is_active',
        probe_once,
    )
    monkeypatch.setattr(
        type(authority),
        '_terminal_cleanup_checkpoint',
        terminal_checkpoint,
        raising=False,
    )
    returned: list[str] = []
    errors: list[BaseException] = []

    try:
        with authority.owned_operation():
            if primary is not None:
                raise primary
            returned.append('durable-product')
    except BaseException as exc:
        errors.append(exc)

    assert injected is True
    if primary is None:
        assert returned == ['durable-product']
        assert errors == []
    else:
        assert errors == [primary]
    assert health.snapshot.healthy is False
    assert authority._emergency_cleanup_state is None
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert health._exclusive_waiters == 0
    session.close()
    application.dispose()


@pytest.mark.parametrize('fault_point', ('state_validate', 'terminal_finalize'))
@pytest.mark.parametrize(
    'primary',
    (
        None,
        ValueError('outer drain validation primary'),
        KeyboardInterrupt('outer drain cancellation primary'),
        _CleanupBodyCommitUnknown('outer drain commit unknown primary'),
    ),
)
def test_outer_cleanup_shell_is_nonthrowing_and_drains_exact_state(
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
    primary: BaseException | None,
) -> None:
    application, dedicated, session, authority = _cleanup_responsibility_authority(
        monkeypatch
    )
    health = authority.runtime_health_authority
    target = (
        '_require_emergency_cleanup_state'
        if fault_point == 'state_validate'
        else '_finalize_terminal_cleanup_state'
    )
    original = getattr(type(authority), target)
    injected = False

    def fail_once(self, *args, **kwargs):
        nonlocal injected
        if (
            fault_point == 'state_validate'
            and args
            and getattr(args[0], 'lease', None) is None
        ):
            return original(self, *args, **kwargs)
        if not injected:
            injected = True
            raise _CleanupStateMachineFault('secret outer drain fault')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(authority), target, fail_once)
    returned: list[str] = []
    errors: list[BaseException] = []

    try:
        with authority.owned_operation():
            if primary is not None:
                raise primary
            returned.append('durable-product')
    except BaseException as exc:
        errors.append(exc)

    assert injected is True
    if primary is None:
        assert returned == ['durable-product']
        assert errors == []
    else:
        assert errors == [primary]
    assert health.snapshot.healthy is False
    assert authority._emergency_cleanup_state is None
    assert authority._active_leases == 0
    assert authority._lease_context.get() is None
    assert session.get_bind() is application
    assert session.in_transaction() is False
    assert tuple(health._emergency_cleanup_capabilities.values()) == ()
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert health._exclusive_waiters == 0
    session.close()
    application.dispose()
