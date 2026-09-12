from __future__ import annotations

import dis
import sys
import threading

import pytest
from sqlalchemy import create_engine, event

from backend.app.db import initialization


class PublicationInterrupted(BaseException):
    pass


@pytest.mark.parametrize(
    'stage', ('bootstrap_return', 'runtime_return', 'before_return')
)
def test_runtime_publication_failure_removes_exact_listeners_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    application = create_engine('sqlite:///:memory:')
    application.dialect.name = 'postgresql'
    installed = []
    disposed = []
    primary = PublicationInterrupted('construction primary')
    original_listen = event.listen
    event.listen(application, 'engine_disposed', lambda *_: disposed.append(True))

    def listen(target, identifier, callback):
        original_listen(target, identifier, callback)
        installed.append((target, identifier, callback))

    monkeypatch.setattr(initialization, 'create_engine', lambda *_a, **_k: application)
    monkeypatch.setattr(event, 'listen', listen)
    if stage != 'before_return':
        cls = (
            initialization.TrustedPostgresEngineBootstrap
            if stage == 'bootstrap_return'
            else initialization.DatabaseRuntime
        )
        original_init = cls.__init__

        def initialize_then_raise(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            raise primary

        monkeypatch.setattr(cls, '__init__', initialize_then_raise)

    return_line = next(
        instruction.positions.lineno
        for instruction in dis.Bytecode(initialization.initialize_database_runtime)
        if instruction.opname == 'RETURN_VALUE'
    )

    def interrupt_return(frame, event_name, _arg):
        if (
            frame.f_code is initialization.initialize_database_runtime.__code__
            and event_name == 'line'
            and frame.f_lineno == return_line
        ):
            sys.settrace(None)
            raise primary
        return interrupt_return

    previous_trace = sys.gettrace()
    try:
        if stage == 'before_return':
            sys.settrace(interrupt_return)
        with pytest.raises(PublicationInterrupted) as captured:
            initialization.initialize_database_runtime(
                'postgresql+psycopg://test@localhost/test'
            )
        assert captured.value is primary
        assert len(installed) == (0 if stage == 'runtime_return' else 3)
        traceback = captured.value.__traceback__
        frames = []
        while traceback is not None:
            frames.append(traceback.tb_frame.f_code.co_name)
            traceback = traceback.tb_next
        assert (
            'interrupt_return' if stage == 'before_return' else 'initialize_then_raise'
        ) in frames
        assert disposed == [True]
        assert initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS == {}
        assert all(not event.contains(*listener) for listener in installed)
    finally:
        sys.settrace(previous_trace)
        for listener in installed:
            if event.contains(*listener):
                event.remove(*listener)
        application.dispose()


@pytest.mark.parametrize(
    'stage',
    ('cleanup_entry', 'cleanup_global_lock', 'cleanup_self_lock', 'after_map_pop'),
)
def test_interrupted_construction_cleanup_retains_exact_bounded_obligation(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    application = create_engine('sqlite:///:memory:')
    application.dialect.name = 'postgresql'
    installed = []
    primary = PublicationInterrupted('original construction failure')
    secondary = PublicationInterrupted('cleanup interrupted')
    original_listen = event.listen
    original_remove = event.remove
    responsibility_seen = []
    displaced_locks = {}
    original_failed = (
        initialization._CheckoutListenerConstructionResponsibility.construction_failed
    )

    class InterruptedLock:
        def __enter__(self):
            raise secondary

        def __exit__(self, *_args):
            return False

    def listen(target, identifier, callback):
        original_listen(target, identifier, callback)
        installed.append((target, identifier, callback))
        if stage != 'after_map_pop':
            raise primary

    def interrupted_cleanup(self):
        responsibility_seen.append(self)
        if stage == 'cleanup_entry' and len(responsibility_seen) == 1:
            raise secondary
        original_lock = displaced_locks.get(id(self), self._lock)
        original_condition = initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS_CONDITION
        try:
            if stage == 'cleanup_global_lock':
                initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS_CONDITION = (
                    InterruptedLock()
                )
            elif stage != 'cleanup_entry':
                self._lock = InterruptedLock()
            original_failed(self)
        finally:
            self._lock = original_lock
            initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS_CONDITION = (
                original_condition
            )

    def retire(self, checkpoint):
        if checkpoint == 'after_map_pop':
            displaced_locks[id(self)] = self._lock
            self._lock = InterruptedLock()
            raise primary

    def remove_unavailable(*_args):
        raise secondary

    monkeypatch.setattr(initialization, 'create_engine', lambda *_a, **_k: application)
    monkeypatch.setattr(event, 'listen', listen)
    monkeypatch.setattr(event, 'remove', remove_unavailable)
    monkeypatch.setattr(
        initialization._CheckoutListenerConstructionResponsibility,
        'construction_failed',
        interrupted_cleanup,
    )
    if stage == 'after_map_pop':
        monkeypatch.setattr(
            initialization._CheckoutListenerConstructionResponsibility,
            '_retire_checkpoint',
            retire,
        )
    try:
        with pytest.raises(PublicationInterrupted) as captured:
            initialization.initialize_database_runtime(
                'postgresql+psycopg://test@localhost/test'
            )
        assert captured.value is primary
        retained = initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS
        assert len(retained) == 1
        responsibility = next(iter(retained.values()))
        assert responsibility in responsibility_seen
        assert responsibility._state == 'QUARANTINED'
        assert responsibility.remaining_listeners == tuple(installed)
        assert responsibility.runtime_health.snapshot.healthy is False
        with pytest.raises(initialization.PostgresRuntimeHealthUnavailableError):
            initialization.initialize_database_runtime(
                'postgresql+psycopg://test@localhost/test'
            )
        assert len(installed) <= 3
        monkeypatch.setattr(event, 'remove', original_remove)
        initialization._drain_failed_checkout_listener_cleanups()
        assert retained == {}
        assert all(not event.contains(*listener) for listener in installed)
    finally:
        monkeypatch.setattr(event, 'remove', original_remove)
        for responsibility in tuple(
            initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS.values()
        ):
            responsibility._state = 'QUARANTINED'
        initialization._drain_failed_checkout_listener_cleanups()
        application.dispose()


def test_registration_has_a_deadline_without_adopting_stalled_foreign_construction():
    health = initialization.TrustedPostgresRuntimeHealth(
        _seal=initialization._POSTGRES_RUNTIME_HEALTH_SEAL,
    )
    cls = initialization._CheckoutListenerConstructionResponsibility
    stalled = cls(
        owner=object(),
        runtime_health=health,
        _seal=initialization._POSTGRES_LISTENER_CONSTRUCTION_SEAL,
    )
    contender = cls(
        owner=object(),
        runtime_health=health,
        _seal=initialization._POSTGRES_LISTENER_CONSTRUCTION_SEAL,
    )
    stalled.register()
    result = []

    def register():
        try:
            contender.register()
        except BaseException as error:
            result.append(error)

    thread = threading.Thread(target=register, daemon=True)
    thread.start()
    try:
        thread.join(2)
        assert not thread.is_alive(), 'registration waited indefinitely'
        assert len(result) == 1
        assert isinstance(
            result[0], initialization.PostgresRuntimeHealthUnavailableError
        )
        assert list(initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS.values()) == [
            stalled
        ]
        assert stalled._state == 'INSTALLING'
    finally:
        stalled.construction_failed()
        thread.join(2)
        contender.construction_failed()


def test_failed_outer_publication_does_not_remove_foreign_installation(monkeypatch):
    applications = [create_engine('sqlite:///:memory:') for _ in range(2)]
    for application in applications:
        application.dialect.name = 'postgresql'
    installed = [[], []]
    original_listen = event.listen
    original_init = initialization.TrustedPostgresEngineBootstrap.__init__
    primary = PublicationInterrupted('outer publication failed')
    foreign_installed = threading.Event()
    release_foreign = threading.Event()
    runtimes = []
    errors = []

    def create(*_args, **_kwargs):
        return applications[1 if threading.current_thread().name == 'foreign' else 0]

    def listen(target, identifier, callback):
        original_listen(target, identifier, callback)
        index = 0 if target is applications[0].pool else 1
        installed[index].append((target, identifier, callback))

    def pause_foreign(self, stage):
        if (
            self._application_engine is applications[1]
            and stage == 'after_registry_install'
        ):
            foreign_installed.set()
            assert release_foreign.wait(3)

    def initialize_foreign():
        try:
            runtimes.append(
                initialization.initialize_database_runtime(
                    'postgresql+psycopg://test@localhost/test'
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=initialize_foreign, name='foreign')

    def construct(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self._application_engine is applications[0]:
            thread.start()
            assert foreign_installed.wait(3)
            raise primary

    monkeypatch.setattr(initialization, 'create_engine', create)
    monkeypatch.setattr(event, 'listen', listen)
    monkeypatch.setattr(
        initialization.TrustedPostgresEngineBootstrap, '__init__', construct
    )
    monkeypatch.setattr(
        initialization.TrustedPostgresEngineBootstrap,
        '_listener_handoff_checkpoint',
        pause_foreign,
    )
    try:
        with pytest.raises(PublicationInterrupted) as captured:
            initialization.initialize_database_runtime(
                'postgresql+psycopg://test@localhost/test'
            )
        assert captured.value is primary
        assert len(installed[0]) == len(installed[1]) == 3
        assert all(not event.contains(*listener) for listener in installed[0])
        assert all(event.contains(*listener) for listener in installed[1])
        release_foreign.set()
        thread.join(3)
        assert not thread.is_alive()
        assert errors == []
        assert len(runtimes) == 1
        assert runtimes[0].rag_postgres_bootstrap._runtime_health.snapshot.healthy
        assert initialization._FAILED_CHECKOUT_LISTENER_CLEANUPS == {}
    finally:
        release_foreign.set()
        if thread.ident is not None:
            thread.join(3)
        for runtime in runtimes:
            runtime.dispose()
        for listeners in installed:
            for listener in listeners:
                if event.contains(*listener):
                    event.remove(*listener)
        for application in applications:
            application.dispose()
