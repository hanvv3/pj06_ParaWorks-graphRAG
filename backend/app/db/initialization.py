from __future__ import annotations

import secrets
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, RLock, get_ident
from time import monotonic_ns
from types import MappingProxyType
from typing import Literal

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import (
    ArgumentError,
    DBAPIError,
    DisconnectionError,
    InterfaceError,
    NoSuchModuleError,
    OperationalError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLAlchemyTimeoutError,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool


class DatabaseConfigurationError(RuntimeError):
    code = 'database_configuration_invalid'


class DatabaseInitializationError(RuntimeError):
    code = 'database_initialization_failed'


class PostgresRuntimeHealthUnavailableError(TypeError):
    """Sanitized refusal after the trusted runtime is fail-stopped."""


_POSTGRES_BOOTSTRAP_SEAL = object()
_POSTGRES_RUNTIME_HEALTH_SEAL = object()
_POSTGRES_RUNTIME_HEALTH_LEASE_SEAL = object()
_POSTGRES_CLEANUP_OWNER_SEAL = object()


@dataclass(frozen=True, slots=True)
class PostgresRuntimeHealthSnapshot:
    healthy: bool
    failure_count: int
    code: Literal['rag_postgres_transport_cleanup_failed'] | None
    first_failure_monotonic_ns: int | None


@dataclass(slots=True, repr=False)
class _PostgresRuntimeHealthLease:
    health: TrustedPostgresRuntimeHealth = field(repr=False)
    epoch: int
    purpose: str
    active: bool = field(default=True, repr=False)
    _seal: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PostgresRuntimeExclusiveTicket:
    sequence: int
    owner_thread_id: int
    purpose: Literal['cleanup', 'poison']


@dataclass(slots=True, repr=False)
class _PostgresCleanupOwnerCapability:
    health: TrustedPostgresRuntimeHealth = field(repr=False)
    owner_thread_id: int
    generation: int
    active: bool = field(default=True, repr=False)
    _seal: object = field(default=None, repr=False, compare=False)


class TrustedPostgresRuntimeHealth:
    """Process-local fail-stop admission shared by one DB runtime."""

    __slots__ = (
        '_active_effects',
        '_condition',
        '_effect_depths',
        '_epoch',
        '_exclusive_depth',
        '_exclusive_generation',
        '_exclusive_owner',
        '_exclusive_ticket_sequence',
        '_exclusive_tickets',
        '_exclusive_waiters',
        '_cleanup_owner_capability',
        '_failure_count',
        '_first_failure_monotonic_ns',
        '_healthy',
        '_lock',
        '_poison_requested',
        '_seal',
    )

    def __init__(self, *, _seal: object) -> None:
        if _seal is not _POSTGRES_RUNTIME_HEALTH_SEAL:
            raise TypeError('trusted PostgreSQL runtime health is unavailable')
        self._epoch = 0
        self._failure_count = 0
        self._first_failure_monotonic_ns: int | None = None
        self._healthy = True
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._active_effects = 0
        self._effect_depths: dict[int, int] = {}
        self._exclusive_owner: int | None = None
        self._exclusive_depth = 0
        self._exclusive_generation = 0
        self._exclusive_ticket_sequence = 0
        self._exclusive_tickets: deque[_PostgresRuntimeExclusiveTicket] = deque()
        self._exclusive_waiters = 0
        self._cleanup_owner_capability: _PostgresCleanupOwnerCapability | None = None
        self._poison_requested = False
        self._seal = _seal

    @property
    def snapshot(self) -> PostgresRuntimeHealthSnapshot:
        with self._lock:
            return PostgresRuntimeHealthSnapshot(
                healthy=self._healthy,
                failure_count=self._failure_count,
                code=(
                    None
                    if self._healthy
                    else 'rag_postgres_transport_cleanup_failed'
                ),
                first_failure_monotonic_ns=self._first_failure_monotonic_ns,
            )

    @contextmanager
    def _operation(self, purpose: str):
        if type(purpose) is not str or not purpose:
            raise TypeError('PostgreSQL runtime health purpose is invalid')
        with self._condition:
            if not self._healthy or self._poison_requested:
                raise PostgresRuntimeHealthUnavailableError(
                    'RAG PostgreSQL runtime health is fail-stopped'
                )
            lease = _PostgresRuntimeHealthLease(
                health=self,
                epoch=self._epoch,
                purpose=purpose,
                _seal=_POSTGRES_RUNTIME_HEALTH_LEASE_SEAL,
            )
        try:
            yield lease
        finally:
            with self._lock:
                lease.active = False

    @contextmanager
    def _guard(self, lease: _PostgresRuntimeHealthLease):
        self._enter_effect(lease)
        try:
            yield
        finally:
            self._exit_effect()

    @contextmanager
    def _effect(self, purpose: str):
        """Admit one compound effect under the shared healthy epoch."""
        with self._operation(purpose) as lease, self._guard(lease):
            yield

    def _enter_effect(self, lease: _PostgresRuntimeHealthLease) -> None:
        thread_id = get_ident()
        with self._condition:
            depth = self._effect_depths.get(thread_id, 0)
            if depth:
                self._require_effect_lease(lease)
                self._effect_depths[thread_id] = depth + 1
                return
            while self._exclusive_owner is not None or self._exclusive_tickets:
                self._condition.wait()
            self._require_effect_lease(lease)
            self._active_effects += 1
            self._effect_depths[thread_id] = 1

    def _exit_effect(self) -> None:
        thread_id = get_ident()
        with self._condition:
            depth = self._effect_depths.get(thread_id, 0)
            if depth <= 0:
                raise RuntimeError('PostgreSQL runtime effect lease underflow')
            if depth > 1:
                self._effect_depths[thread_id] = depth - 1
                return
            del self._effect_depths[thread_id]
            self._active_effects -= 1
            if self._active_effects < 0:
                raise RuntimeError('PostgreSQL runtime effect count underflow')
            if self._active_effects == 0 and self._poison_requested:
                self._apply_poison()
            self._condition.notify_all()

    def _require_effect_lease(
        self,
        lease: _PostgresRuntimeHealthLease,
    ) -> None:
        if (
            type(lease) is not _PostgresRuntimeHealthLease
            or lease._seal is not _POSTGRES_RUNTIME_HEALTH_LEASE_SEAL
            or lease.health is not self
            or not lease.active
            or lease.epoch != self._epoch
            or not self._healthy
            or self._poison_requested
        ):
            raise PostgresRuntimeHealthUnavailableError(
                'RAG PostgreSQL runtime health lease changed'
            )

    def _poison(self) -> None:
        thread_id = get_ident()
        with self._condition:
            if not self._healthy:
                return
            if self._exclusive_owner == thread_id:
                self._apply_poison()
                return
            if self._effect_depths.get(thread_id, 0):
                self._poison_requested = True
                self._condition.notify_all()
                return
            ticket = self._enqueue_exclusive_ticket(
                thread_id=thread_id,
                purpose='poison',
            )
            try:
                while self._healthy and (
                    not self._is_head_exclusive_ticket(ticket)
                    or self._active_effects
                    or self._exclusive_owner is not None
                ):
                    self._condition.wait()
                if not self._healthy:
                    self._cancel_exclusive_ticket(ticket)
                    return
                self._claim_head_exclusive_ticket(ticket)
                self._apply_poison()
                self._condition.notify_all()
            except BaseException:
                self._cancel_exclusive_ticket(ticket)
                raise

    def _apply_poison(self) -> None:
        if not self._healthy:
            return
        self._healthy = False
        self._poison_requested = False
        self._epoch += 1
        self._failure_count = 1
        self._first_failure_monotonic_ns = monotonic_ns()

    @contextmanager
    def _cleanup_boundary(self):
        """Linearize cleanup success or poison before another admission."""
        capability = self._enter_cleanup()
        try:
            yield capability
        finally:
            self._exit_cleanup(capability)

    def _enter_cleanup(self) -> _PostgresCleanupOwnerCapability:
        thread_id = get_ident()
        with self._condition:
            if self._effect_depths.get(thread_id, 0):
                raise RuntimeError(
                    'cleanup cannot run inside a PostgreSQL runtime effect'
                )
            if self._exclusive_owner == thread_id:
                if self._exclusive_tickets:
                    raise RuntimeError(
                        'cleanup reentrancy cannot bypass queued authority'
                    )
                self._exclusive_depth += 1
                capability = self._cleanup_owner_capability
                if capability is None:
                    raise RuntimeError('PostgreSQL cleanup owner is unavailable')
                return capability
            ticket = self._enqueue_exclusive_ticket(
                thread_id=thread_id,
                purpose='cleanup',
            )
            try:
                while (
                    not self._is_head_exclusive_ticket(ticket)
                    or self._active_effects
                    or self._exclusive_owner is not None
                ):
                    self._condition.wait()
                self._claim_head_exclusive_ticket(ticket)
                self._exclusive_owner = thread_id
                self._exclusive_depth = 1
                self._exclusive_generation += 1
                capability = _PostgresCleanupOwnerCapability(
                    health=self,
                    owner_thread_id=thread_id,
                    generation=self._exclusive_generation,
                    _seal=_POSTGRES_CLEANUP_OWNER_SEAL,
                )
                self._cleanup_owner_capability = capability
                return capability
            except BaseException:
                self._cancel_exclusive_ticket(ticket)
                raise

    def _exit_cleanup(
        self,
        capability: _PostgresCleanupOwnerCapability,
    ) -> None:
        with self._condition:
            self._require_cleanup_owner(capability)
            self._exclusive_depth -= 1
            if self._exclusive_depth == 0:
                capability.active = False
                self._cleanup_owner_capability = None
                self._exclusive_owner = None
                self._condition.notify_all()

    def _require_cleanup_owner(
        self,
        capability: object,
        *,
        outermost: bool = False,
    ) -> None:
        with self._condition:
            if (
                type(capability) is not _PostgresCleanupOwnerCapability
                or capability._seal is not _POSTGRES_CLEANUP_OWNER_SEAL
                or capability.health is not self
                or not capability.active
                or self._cleanup_owner_capability is not capability
                or capability.owner_thread_id != get_ident()
                or self._exclusive_owner != get_ident()
                or self._exclusive_depth <= 0
                or (outermost and self._exclusive_depth != 1)
                or capability.generation != self._exclusive_generation
            ):
                raise TypeError('PostgreSQL cleanup owner capability changed')

    def _enqueue_exclusive_ticket(
        self,
        *,
        thread_id: int,
        purpose: Literal['cleanup', 'poison'],
    ) -> _PostgresRuntimeExclusiveTicket:
        self._exclusive_ticket_sequence += 1
        ticket = _PostgresRuntimeExclusiveTicket(
            sequence=self._exclusive_ticket_sequence,
            owner_thread_id=thread_id,
            purpose=purpose,
        )
        self._exclusive_tickets.append(ticket)
        self._exclusive_waiters += 1
        return ticket

    def _is_head_exclusive_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> bool:
        return bool(self._exclusive_tickets) and self._exclusive_tickets[0] is ticket

    def _claim_head_exclusive_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> None:
        if not self._is_head_exclusive_ticket(ticket):
            raise RuntimeError('PostgreSQL cleanup authority ticket changed')
        self._exclusive_tickets.popleft()
        self._exclusive_waiters -= 1
        if self._exclusive_waiters < 0:
            raise RuntimeError('PostgreSQL cleanup waiter count underflow')

    def _cancel_exclusive_ticket(
        self,
        ticket: _PostgresRuntimeExclusiveTicket,
    ) -> None:
        queued_ticket = next(
            (
                queued
                for queued in self._exclusive_tickets
                if queued is ticket
            ),
            None,
        )
        if queued_ticket is None:
            return
        self._exclusive_tickets.remove(queued_ticket)
        self._exclusive_waiters -= 1
        if self._exclusive_waiters < 0:
            raise RuntimeError('PostgreSQL cleanup waiter count underflow')
        self._condition.notify_all()


@dataclass(frozen=True, slots=True, repr=False)
class DatabaseConnectionPolicy:
    """One resolved application/dedicated Engine construction policy."""

    engine_options: Mapping[str, object] = field(
        default_factory=dict,
        repr=False,
    )
    initialize_engine: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        options = dict(self.engine_options)
        if 'poolclass' in options or 'creator' in options:
            raise DatabaseConfigurationError()
        if self.initialize_engine is not None:
            raise DatabaseConfigurationError()
        object.__setattr__(
            self,
            'engine_options',
            _freeze_policy_mapping(options),
        )


@dataclass(frozen=True, slots=True, repr=False)
class _IssuedDedicatedPostgresEngine:
    engine: Engine = field(repr=False)
    policy_capability_id: str
    bootstrap: TrustedPostgresEngineBootstrap = field(repr=False)
    runtime_health: TrustedPostgresRuntimeHealth = field(repr=False)
    _seal: object = field(repr=False, compare=False)


class TrustedPostgresEngineBootstrap:
    """Bootstrap-minted factory for per-request advisory transport."""

    __slots__ = (
        '_application_engine',
        '_dedicated_factory',
        '_policy_capability_id',
        '_revoked',
        '_runtime_health',
        '_seal',
        '_state_lock',
    )

    def __init__(
        self,
        *,
        application_engine: Engine,
        dedicated_factory: Callable[[], Engine],
        policy_capability_id: str,
        runtime_health: TrustedPostgresRuntimeHealth,
        _seal: object,
    ) -> None:
        if (
            _seal is not _POSTGRES_BOOTSTRAP_SEAL
            or not isinstance(application_engine, Engine)
            or application_engine.dialect.name != 'postgresql'
            or not callable(dedicated_factory)
            or len(policy_capability_id) != 64
            or type(runtime_health) is not TrustedPostgresRuntimeHealth
        ):
            raise TypeError('trusted PostgreSQL Engine bootstrap is unavailable')
        self._application_engine = application_engine
        self._dedicated_factory = dedicated_factory
        self._policy_capability_id = policy_capability_id
        self._revoked = False
        self._runtime_health = runtime_health
        self._seal = _seal
        self._state_lock = RLock()

    def _issue(self, application_engine: Engine) -> _IssuedDedicatedPostgresEngine:
        with self._runtime_health._operation('bootstrap_issue') as health_lease:
            with self._runtime_health._guard(health_lease), self._state_lock:
                if (
                    self._revoked
                    or self._seal is not _POSTGRES_BOOTSTRAP_SEAL
                    or application_engine is not self._application_engine
                ):
                    raise TypeError('PostgreSQL Engine bootstrap authority changed')
                dedicated_factory = self._dedicated_factory
            dedicated_engine = dedicated_factory()
            try:
                with self._runtime_health._guard(health_lease), self._state_lock:
                    accepted = (
                        not self._revoked
                        and self._seal is _POSTGRES_BOOTSTRAP_SEAL
                        and application_engine is self._application_engine
                        and isinstance(dedicated_engine, Engine)
                        and dedicated_engine is not application_engine
                        and dedicated_engine.dialect.name == 'postgresql'
                        and type(dedicated_engine.pool) is NullPool
                    )
                    if accepted:
                        return _IssuedDedicatedPostgresEngine(
                            engine=dedicated_engine,
                            policy_capability_id=self._policy_capability_id,
                            bootstrap=self,
                            runtime_health=self._runtime_health,
                            _seal=_POSTGRES_BOOTSTRAP_SEAL,
                        )
            except BaseException:
                self._dispose_rejected_engine(dedicated_engine)
                raise
            self._dispose_rejected_engine(dedicated_engine)
            raise TypeError('trusted dedicated PostgreSQL Engine is unavailable')

    def _dispose_rejected_engine(self, engine: object) -> None:
        if not isinstance(engine, Engine):
            return
        try:
            engine.dispose()
        except BaseException:
            self._runtime_health._poison()

    @property
    def runtime_health_snapshot(self) -> PostgresRuntimeHealthSnapshot:
        return self._runtime_health.snapshot

    def _runtime_effect_authority(
        self,
        application_engine: Engine,
    ) -> TrustedPostgresRuntimeHealth:
        with self._state_lock:
            if (
                self._revoked
                or self._seal is not _POSTGRES_BOOTSTRAP_SEAL
                or application_engine is not self._application_engine
            ):
                raise TypeError('PostgreSQL Engine bootstrap authority changed')
            return self._runtime_health

    def _revoke(self) -> None:
        with self._state_lock:
            self._revoked = True

def _is_database_availability_error(error: Exception) -> bool:
    if isinstance(error, (ModuleNotFoundError, ImportError, OSError)):
        return True
    if isinstance(error, DBAPIError) and bool(error.connection_invalidated):
        return True
    return isinstance(
        error,
        (
            OperationalError,
            InterfaceError,
            SQLAlchemyTimeoutError,
            DisconnectionError,
        ),
    )


@dataclass(slots=True, repr=False)
class DatabaseRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]
    rag_postgres_bootstrap: TrustedPostgresEngineBootstrap | None = field(
        default=None,
        repr=False,
    )
    _dispose_attempted: bool = field(default=False, init=False, repr=False)

    def dispose(self) -> None:
        if self._dispose_attempted:
            return
        self._dispose_attempted = True
        if self.rag_postgres_bootstrap is not None:
            self.rag_postgres_bootstrap._revoke()
        cleanup_failure: Exception | None = None
        try:
            self.engine.dispose()
        except Exception as error:
            cleanup_failure = error
        if cleanup_failure is None:
            return
        if _is_database_availability_error(cleanup_failure):
            raise DatabaseInitializationError()
        raise cleanup_failure


def initialize_database_runtime(
    database_url: str,
    *,
    connection_policy: DatabaseConnectionPolicy | None = None,
) -> DatabaseRuntime:
    configuration_failure = False
    try:
        parsed_url = make_url(database_url)
    except ArgumentError:
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    configuration_failure = False
    engine_failure: Exception | None = None
    policy = connection_policy or DatabaseConnectionPolicy()
    engine_options = {'pool_pre_ping': True, **dict(policy.engine_options)}
    try:
        engine = create_engine(
            database_url,
            **_engine_options_for_create(engine_options),
        )
    except (NoSuchModuleError, ArgumentError):
        configuration_failure = True
    except Exception as error:
        engine_failure = error
    if configuration_failure:
        raise DatabaseConfigurationError()
    if engine_failure is not None:
        if _is_database_availability_error(engine_failure):
            raise DatabaseInitializationError()
        raise engine_failure

    session_factory: sessionmaker[Session] | None = None
    session_factory_failure: Exception | None = None
    try:
        session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=True,
        )
    except Exception as error:
        session_factory_failure = error

    if session_factory_failure is not None:
        cleanup_failure: Exception | None = None
        try:
            engine.dispose()
        except Exception as error:
            cleanup_failure = error
        if cleanup_failure is not None:
            if _is_database_availability_error(cleanup_failure):
                raise DatabaseInitializationError()
            raise cleanup_failure
        raise session_factory_failure

    assert session_factory is not None
    postgres_bootstrap: TrustedPostgresEngineBootstrap | None = None
    if parsed_url.get_backend_name() == 'postgresql':
        policy_capability_id = secrets.token_hex(32)
        runtime_health = TrustedPostgresRuntimeHealth(
            _seal=_POSTGRES_RUNTIME_HEALTH_SEAL
        )

        def dedicated_factory() -> Engine:
            dedicated_options = _engine_options_for_create(engine_options)
            dedicated_options['poolclass'] = NullPool
            return create_engine(database_url, **dedicated_options)

        postgres_bootstrap = TrustedPostgresEngineBootstrap(
            application_engine=engine,
            dedicated_factory=dedicated_factory,
            policy_capability_id=policy_capability_id,
            runtime_health=runtime_health,
            _seal=_POSTGRES_BOOTSTRAP_SEAL,
        )
    return DatabaseRuntime(
        engine=engine,
        session_factory=session_factory,
        rag_postgres_bootstrap=postgres_bootstrap,
    )


def _engine_options_for_create(
    engine_options: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: _thaw_policy_value(value)
        for key, value in engine_options.items()
    }


def _freeze_policy_mapping(
    value: Mapping[object, object],
) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise DatabaseConfigurationError()
        frozen[key] = _freeze_policy_value(item)
    return MappingProxyType(frozen)


def _freeze_policy_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if isinstance(value, Mapping):
        return _freeze_policy_mapping(value)
    if type(value) is tuple:
        return tuple(_freeze_policy_value(item) for item in value)
    raise DatabaseConfigurationError()


def _thaw_policy_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_policy_value(item)
            for key, item in value.items()
        }
    if type(value) is tuple:
        return tuple(_thaw_policy_value(item) for item in value)
    return value
