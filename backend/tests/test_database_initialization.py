from __future__ import annotations

import ast
import os
import subprocess
import sys
import traceback
from collections.abc import Generator
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Thread, current_thread, get_ident
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


class _RuntimeWaitCancelled(BaseException):
    pass


class _ControlledCondition(Condition):
    """Deterministically cancel or reorder selected health-latch waiters."""

    def __init__(self, lock: object) -> None:
        super().__init__(lock)  # type: ignore[arg-type]
        self.cancel_gates: dict[str, tuple[Event, type[BaseException]]] = {}
        self.return_gates: dict[str, Event] = {}
        self.wait_counts: dict[str, int] = {}
        self.wait_events: dict[tuple[str, int], Event] = {}

    def watch(self, thread_name: str, occurrence: int = 1) -> Event:
        event = Event()
        self.wait_events[(thread_name, occurrence)] = event
        return event

    def cancel(
        self,
        thread_name: str,
        gate: Event,
        error_type: type[BaseException],
    ) -> None:
        self.cancel_gates[thread_name] = (gate, error_type)

    def gate_return(self, thread_name: str, gate: Event) -> None:
        self.return_gates[thread_name] = gate

    def wait(self, timeout: float | None = None) -> bool:
        thread_name = current_thread().name
        occurrence = self.wait_counts.get(thread_name, 0) + 1
        self.wait_counts[thread_name] = occurrence
        event = self.wait_events.get((thread_name, occurrence))
        if event is not None:
            event.set()

        cancellation = self.cancel_gates.get(thread_name)
        if cancellation is not None:
            gate, error_type = cancellation
            state = self._release_save()
            try:
                if not gate.wait(timeout=5):
                    raise AssertionError('cancellation gate timed out')
            finally:
                self._acquire_restore(state)
            raise error_type()

        notified = super().wait(timeout)
        return_gate = self.return_gates.get(thread_name)
        if return_gate is not None and not return_gate.is_set():
            state = self._release_save()
            try:
                if not return_gate.wait(timeout=5):
                    raise AssertionError('condition return gate timed out')
            finally:
                self._acquire_restore(state)
        return notified


def _controlled_runtime_health() -> tuple[
    initialization.TrustedPostgresRuntimeHealth,
    _ControlledCondition,
]:
    health = initialization.TrustedPostgresRuntimeHealth(
        _seal=initialization._POSTGRES_RUNTIME_HEALTH_SEAL
    )
    condition = _ControlledCondition(health._lock)
    health._condition = condition
    return health, condition


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


def test_runtime_health_effects_overlap_and_poison_waits_exclusively() -> None:
    health = initialization.TrustedPostgresRuntimeHealth(
        _seal=initialization._POSTGRES_RUNTIME_HEALTH_SEAL
    )
    effect = getattr(health, '_effect', None)
    assert callable(effect)
    entered = (Event(), Event())
    release = Event()

    def reader(index: int) -> None:
        with effect(f'healthy-reader-{index}'):
            entered[index].set()
            assert release.wait(timeout=5)

    readers = [Thread(target=reader, args=(index,)) for index in range(2)]
    for reader in readers:
        reader.start()
    assert entered[0].wait(timeout=2)
    assert entered[1].wait(timeout=2)

    poison_finished = Event()

    def poison() -> None:
        health._poison()
        poison_finished.set()

    poisoner = Thread(target=poison)
    poisoner.start()
    assert poison_finished.wait(timeout=0.1) is False
    assert health.snapshot.healthy is True
    release.set()
    for reader in readers:
        reader.join(timeout=5)
        assert reader.is_alive() is False
    poisoner.join(timeout=5)
    assert poisoner.is_alive() is False
    assert health.snapshot.healthy is False
    assert health.snapshot.failure_count == 1


def test_runtime_health_poison_inside_effect_is_latched_before_escape() -> None:
    health = initialization.TrustedPostgresRuntimeHealth(
        _seal=initialization._POSTGRES_RUNTIME_HEALTH_SEAL
    )
    effect = getattr(health, '_effect', None)
    assert callable(effect)

    with effect('cleanup-failure'):
        health._poison()
        assert health.snapshot.healthy is True

    assert health.snapshot.healthy is False
    with (
        pytest.raises(TypeError, match='runtime health'),
        effect('future-effect'),
    ):
        raise AssertionError('poisoned effect must not run')


def test_cancelled_cleanup_waiter_wakes_a_blocked_shared_effect() -> None:
    health, condition = _controlled_runtime_health()
    active_entered = Event()
    release_active = Event()
    follower_entered = Event()
    cancel_cleanup = Event()
    cleanup_waiting = condition.watch('cancelled-cleanup')
    follower_waiting = condition.watch('blocked-follower')
    condition.cancel(
        'cancelled-cleanup',
        cancel_cleanup,
        _RuntimeWaitCancelled,
    )
    cancellations: Queue[BaseException] = Queue()

    def active_effect() -> None:
        with health._effect('active-effect'):
            active_entered.set()
            assert release_active.wait(timeout=5)

    def cancelled_cleanup() -> None:
        try:
            with health._cleanup_boundary():
                raise AssertionError('cancelled cleanup must not acquire')
        except BaseException as error:
            cancellations.put(error)

    def blocked_follower() -> None:
        with health._effect('blocked-follower'):
            follower_entered.set()

    active = Thread(target=active_effect, name='active-effect')
    cleanup = Thread(target=cancelled_cleanup, name='cancelled-cleanup')
    follower = Thread(target=blocked_follower, name='blocked-follower')
    active.start()
    assert active_entered.wait(timeout=2)
    cleanup.start()
    assert cleanup_waiting.wait(timeout=2)
    follower.start()
    assert follower_waiting.wait(timeout=2)

    try:
        cancel_cleanup.set()
        cleanup.join(timeout=2)
        assert cleanup.is_alive() is False
        assert isinstance(cancellations.get_nowait(), _RuntimeWaitCancelled)
        assert follower_entered.wait(timeout=2)
        assert health._active_effects == 1
        assert health._exclusive_waiters == 0
    finally:
        release_active.set()
        cancel_cleanup.set()
        active.join(timeout=5)
        cleanup.join(timeout=5)
        follower.join(timeout=5)
    assert active.is_alive() is False
    assert follower.is_alive() is False


def test_cleanup_and_poison_exclusive_tickets_are_fifo_under_late_wakeups() -> None:
    health, condition = _controlled_runtime_health()
    active_entered = Event()
    release_active = Event()
    cleanup_one_return = Event()
    cleanup_three_return = Event()
    cleanup_one_entered = Event()
    cleanup_three_entered = Event()
    release_cleanup_one = Event()
    release_cleanup_three = Event()
    poison_finished = Event()
    cleanup_one_waiting = condition.watch('cleanup-one')
    poison_waiting = condition.watch('poison-two')
    poison_waiting_again = condition.watch('poison-two', 2)
    cleanup_three_waiting = condition.watch('cleanup-three')
    condition.gate_return('cleanup-one', cleanup_one_return)
    condition.gate_return('cleanup-three', cleanup_three_return)
    order: list[str] = []

    def active_effect() -> None:
        with health._effect('fifo-active-effect'):
            active_entered.set()
            assert release_active.wait(timeout=5)

    def cleanup_one() -> None:
        with health._cleanup_boundary():
            order.append('cleanup-one')
            cleanup_one_entered.set()
            assert release_cleanup_one.wait(timeout=5)

    def poison_two() -> None:
        health._poison()
        order.append('poison-two')
        poison_finished.set()

    def cleanup_three() -> None:
        with health._cleanup_boundary():
            order.append('cleanup-three')
            cleanup_three_entered.set()
            assert release_cleanup_three.wait(timeout=5)

    threads = (
        Thread(target=active_effect, name='fifo-active-effect'),
        Thread(target=cleanup_one, name='cleanup-one'),
        Thread(target=poison_two, name='poison-two'),
        Thread(target=cleanup_three, name='cleanup-three'),
    )
    threads[0].start()
    assert active_entered.wait(timeout=2)
    threads[1].start()
    assert cleanup_one_waiting.wait(timeout=2)
    threads[2].start()
    assert poison_waiting.wait(timeout=2)
    threads[3].start()
    assert cleanup_three_waiting.wait(timeout=2)

    try:
        release_active.set()
        assert poison_waiting_again.wait(timeout=2)
        assert poison_finished.is_set() is False

        cleanup_one_return.set()
        assert cleanup_one_entered.wait(timeout=2)
        assert order == ['cleanup-one']
        release_cleanup_one.set()

        assert poison_finished.wait(timeout=2)
        assert order == ['cleanup-one', 'poison-two']
        cleanup_three_return.set()
        assert cleanup_three_entered.wait(timeout=2)
        assert order == ['cleanup-one', 'poison-two', 'cleanup-three']
    finally:
        release_active.set()
        cleanup_one_return.set()
        cleanup_three_return.set()
        release_cleanup_one.set()
        release_cleanup_three.set()
        for thread in threads:
            thread.join(timeout=5)
    assert all(thread.is_alive() is False for thread in threads)
    assert health.snapshot.healthy is False


@pytest.mark.parametrize('operation', ('cleanup', 'poison'))
@pytest.mark.parametrize('blocker', ('active_effect', 'cleanup_owner'))
def test_cancelled_exclusive_wait_preserves_health_latch_state(
    operation: str,
    blocker: str,
) -> None:
    health, condition = _controlled_runtime_health()
    worker_name = f'cancel-{operation}-behind-{blocker}'
    cancel = Event()
    cancel.set()
    waiting = condition.watch(worker_name)
    condition.cancel(worker_name, cancel, KeyboardInterrupt)
    failures: Queue[BaseException] = Queue()

    def waiter() -> None:
        try:
            if operation == 'cleanup':
                with health._cleanup_boundary():
                    raise AssertionError('cancelled cleanup must not acquire')
            else:
                health._poison()
        except BaseException as error:
            failures.put(error)

    worker = Thread(target=waiter, name=worker_name)
    if blocker == 'active_effect':
        boundary = health._effect('cancellation-blocker')
    else:
        boundary = health._cleanup_boundary()

    with boundary:
        worker.start()
        assert waiting.wait(timeout=2)
        worker.join(timeout=2)
        assert worker.is_alive() is False
        assert isinstance(failures.get_nowait(), KeyboardInterrupt)
        assert health._exclusive_waiters == 0
        assert health._poison_requested is False
        assert health.snapshot.healthy is True
        if blocker == 'active_effect':
            assert health._active_effects == 1
            assert health._effect_depths == {get_ident(): 1}
            assert health._exclusive_owner is None
            assert health._exclusive_depth == 0
        else:
            assert health._active_effects == 0
            assert health._effect_depths == {}
            assert health._exclusive_owner == get_ident()
            assert health._exclusive_depth == 1

    assert health._active_effects == 0
    assert health._effect_depths == {}
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert health._exclusive_waiters == 0


def test_cancelled_shared_effect_wait_preserves_cleanup_owner_state() -> None:
    health, condition = _controlled_runtime_health()
    cancel = Event()
    cancel.set()
    waiting = condition.watch('cancel-shared-effect')
    condition.cancel('cancel-shared-effect', cancel, KeyboardInterrupt)
    failures: Queue[BaseException] = Queue()

    def waiter() -> None:
        try:
            with health._effect('cancel-shared-effect'):
                raise AssertionError('cancelled effect must not acquire')
        except BaseException as error:
            failures.put(error)

    worker = Thread(target=waiter, name='cancel-shared-effect')
    with health._cleanup_boundary():
        worker.start()
        assert waiting.wait(timeout=2)
        worker.join(timeout=2)
        assert worker.is_alive() is False
        assert isinstance(failures.get_nowait(), KeyboardInterrupt)
        assert health._active_effects == 0
        assert health._effect_depths == {}
        assert health._exclusive_owner == get_ident()
        assert health._exclusive_depth == 1
        assert health._exclusive_waiters == 0

    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    with health._effect('still-healthy'):
        assert health.snapshot.healthy is True


def test_nested_cleanup_does_not_bypass_an_older_foreign_waiter() -> None:
    health, condition = _controlled_runtime_health()
    foreign_waiting = condition.watch('older-foreign-cleanup')
    foreign_entered = Event()
    release_foreign = Event()
    nested_failure: RuntimeError | None = None
    nested_acquired = False

    def foreign_cleanup() -> None:
        with health._cleanup_boundary():
            foreign_entered.set()
            assert release_foreign.wait(timeout=5)

    foreign = Thread(target=foreign_cleanup, name='older-foreign-cleanup')
    with health._cleanup_boundary():
        with health._cleanup_boundary():
            assert health._exclusive_depth == 2
        assert health._exclusive_depth == 1

        foreign.start()
        assert foreign_waiting.wait(timeout=2)
        try:
            with health._cleanup_boundary():
                nested_acquired = True
        except RuntimeError as error:
            nested_failure = error

    try:
        assert foreign_entered.wait(timeout=2)
    finally:
        release_foreign.set()
        foreign.join(timeout=5)
    assert foreign.is_alive() is False
    assert nested_acquired is False
    assert nested_failure is not None
    assert 'queued' in str(nested_failure)
    assert health._exclusive_owner is None
    assert health._exclusive_depth == 0
    assert health._exclusive_waiters == 0


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
    engines = iter((application, dedicated))
    mutable_connect_args: dict[str, object] = {
        'sslmode': 'verify-full',
        'sslrootcert': '/run/secrets/database-ca.pem',
        'server_settings': {
            'application_name': 'paraworks-rag',
            'statement_timeout': '5000',
        },
    }

    def create_engine_probe(database_url: str, **kwargs: object):
        created.append((database_url, kwargs))
        return next(engines)

    monkeypatch.setattr(initialization, 'create_engine', create_engine_probe)
    policy = initialization.DatabaseConnectionPolicy(
        engine_options={'connect_args': mutable_connect_args},
    )
    mutable_connect_args['sslmode'] = 'disable'
    runtime = initialization.initialize_database_runtime(
        'postgresql+psycopg://invalid.example/authority',
        connection_policy=policy,
    )
    mutable_connect_args['server_settings']['statement_timeout'] = '0'
    assert runtime.rag_postgres_bootstrap is not None
    issued = runtime.rag_postgres_bootstrap._issue(application)
    issued_engine = issued.engine

    assert issued_engine is dedicated
    assert created == [
        (
            'postgresql+psycopg://invalid.example/authority',
            {
                'pool_pre_ping': True,
                'connect_args': {
                    'sslmode': 'verify-full',
                    'sslrootcert': '/run/secrets/database-ca.pem',
                    'server_settings': {
                        'application_name': 'paraworks-rag',
                        'statement_timeout': '5000',
                    },
                },
            },
        ),
        (
            'postgresql+psycopg://invalid.example/authority',
            {
                'pool_pre_ping': True,
                'connect_args': {
                    'sslmode': 'verify-full',
                    'sslrootcert': '/run/secrets/database-ca.pem',
                    'server_settings': {
                        'application_name': 'paraworks-rag',
                        'statement_timeout': '5000',
                    },
                },
                'poolclass': NullPool,
            },
        ),
    ]
    assert len(issued.policy_capability_id) == 64

    issued_engine.dispose()
    runtime.dispose()
    with pytest.raises(TypeError, match='bootstrap authority changed'):
        runtime.rag_postgres_bootstrap._issue(application)


def test_postgres_bootstrap_uses_distinct_sealed_policy_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_ids: list[str] = []
    variants = (
        (
            'postgresql+psycopg://first.example/authority',
            {'sslmode': 'verify-full', 'application_name': 'one'},
        ),
        (
            'postgresql+psycopg://second.example/authority',
            {'sslmode': 'verify-ca', 'application_name': 'two'},
        ),
    )
    for database_url, connect_args in variants:
        application = sqlalchemy_create_engine('sqlite+pysqlite:///:memory:')
        dedicated = sqlalchemy_create_engine(
            'sqlite+pysqlite:///:memory:',
            poolclass=NullPool,
        )
        application.dialect.name = 'postgresql'
        dedicated.dialect.name = 'postgresql'
        engines = iter((application, dedicated))
        monkeypatch.setattr(
            initialization,
            'create_engine',
            lambda *_args, _engines=engines, **_kwargs: next(_engines),
        )
        runtime = initialization.initialize_database_runtime(
            database_url,
            connection_policy=initialization.DatabaseConnectionPolicy(
                engine_options={'connect_args': connect_args},
            ),
        )
        assert runtime.rag_postgres_bootstrap is not None
        issued = runtime.rag_postgres_bootstrap._issue(application)
        capability_ids.append(issued.policy_capability_id)
        issued.engine.dispose()
        runtime.dispose()

    assert len(set(capability_ids)) == 2
    assert all(len(value) == 64 for value in capability_ids)


@pytest.mark.parametrize(
    'policy',
    (
        initialization.DatabaseConnectionPolicy(
            engine_options={'connect_args': {'sslmode': 'verify-full'}}
        ),
        {'engine_options': {'creator': lambda: object()}},
        {'initialize_engine': lambda _engine: None},
        {'engine_options': {'connect_args': {'ssl_context': object()}}},
        {'engine_options': {'connect_args': {'fallbacks': ['unsafe']}}},
    ),
)
def test_connection_policy_rejects_mutable_or_stateful_authority(
    policy: object,
) -> None:
    if isinstance(policy, initialization.DatabaseConnectionPolicy):
        assert policy.engine_options['connect_args']['sslmode'] == 'verify-full'
        return
    with pytest.raises(initialization.DatabaseConfigurationError):
        initialization.DatabaseConnectionPolicy(**policy)


def test_postgres_revoke_during_issue_rejects_and_disposes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = sqlalchemy_create_engine('sqlite+pysqlite:///:memory:')
    dedicated = sqlalchemy_create_engine(
        'sqlite+pysqlite:///:memory:',
        poolclass=NullPool,
    )
    application.dialect.name = 'postgresql'
    dedicated.dialect.name = 'postgresql'
    dedicated_created = Event()
    resume_issue = Event()
    revoke_finished = Event()
    issue_errors: list[BaseException] = []
    disposals: list[object] = []
    event.listen(dedicated, 'engine_disposed', lambda engine: disposals.append(engine))
    calls = 0

    def create_engine_probe(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            return application
        dedicated_created.set()
        assert resume_issue.wait(5)
        return dedicated

    monkeypatch.setattr(initialization, 'create_engine', create_engine_probe)
    runtime = initialization.initialize_database_runtime(
        'postgresql+psycopg://invalid.example/authority'
    )
    assert runtime.rag_postgres_bootstrap is not None

    def issue() -> None:
        try:
            runtime.rag_postgres_bootstrap._issue(application)
        except BaseException as exc:
            issue_errors.append(exc)

    issuer = Thread(target=issue)
    issuer.start()
    assert dedicated_created.wait(5)
    revoker = Thread(target=lambda: (runtime.dispose(), revoke_finished.set()))
    revoker.start()
    assert revoke_finished.wait(5)
    resume_issue.set()
    issuer.join(timeout=5)
    revoker.join(timeout=5)

    assert issuer.is_alive() is False and revoker.is_alive() is False
    assert len(issue_errors) == 1
    assert isinstance(issue_errors[0], TypeError)
    assert disposals == [dedicated]


def test_postgres_policy_rejects_stateful_initializer_before_engine_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_creations: list[object] = []
    monkeypatch.setattr(
        initialization,
        'create_engine',
        lambda *_args, **_kwargs: engine_creations.append(object()),
    )

    def fail_initialization(_engine: object) -> None:
        raise RuntimeError('injected engine initialization failure')

    with pytest.raises(initialization.DatabaseConfigurationError):
        initialization.DatabaseConnectionPolicy(
            initialize_engine=fail_initialization,
        )

    assert engine_creations == []


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
