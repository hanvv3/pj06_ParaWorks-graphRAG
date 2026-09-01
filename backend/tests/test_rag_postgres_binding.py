from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
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
