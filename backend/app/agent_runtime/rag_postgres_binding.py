from __future__ import annotations

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
    RegisteredAdvisoryLock,
    _require_registered_capability,
    acquire_advisory_lock,
    release_advisory_lock,
)
from backend.app.db.initialization import (
    PostgresRuntimeHealthUnavailableError,
    TrustedPostgresEngineBootstrap,
    TrustedPostgresRuntimeHealth,
    _PostgresCleanupOwnerCapability,
    _PostgresEmergencyCleanupCapability,
    _PostgresRuntimeHealthLease,
)

_POSTGRES_DATABASE_AUTHORITY_SEAL = object()
_POSTGRES_ADVISORY_TRANSPORT_SEAL = object()
_OPERATION_LEASE_SEAL = object()
_CLEANUP_OWNER_SEAL = object()
_EMERGENCY_CLEANUP_STATE_SEAL = object()
_IDENTITY_SQL = text(
    "SELECT current_database(), current_schema(), current_schemas(false), "
    "current_setting('search_path'), current_user, "
    "(SELECT oid FROM pg_database WHERE datname = current_database())"
)
_SERVER_IDENTITY_SQL = text(
    "SELECT inet_server_addr()::text, inet_server_port(), "
    "pg_postmaster_start_time()::text, pg_is_in_recovery(), "
    "current_setting('transaction_read_only')"
)


class RagPostgresDatabaseBusyError(RuntimeError):
    """The request authority is draining an already active operation."""


@dataclass(frozen=True, slots=True)
class RagPostgresDatabaseCleanupFailure:
    code: Literal['rag_postgres_transport_cleanup_failed'] = (
        'rag_postgres_transport_cleanup_failed'
    )
    transport_fail_stopped: Literal[True] = True


class RagPostgresAdvisoryCleanupError(TypeError):
    """Sanitized fail-stop for uncertain provider advisory cleanup."""


@dataclass(frozen=True, slots=True)
class _RagPostgresOperationLease:
    authority: RagPostgresDatabaseAuthority = field(repr=False)
    application_connection: Connection = field(repr=False)
    original_session_bind: Engine | Connection = field(repr=False)
    owns_application_connection: bool
    runtime_health_lease: _PostgresRuntimeHealthLease = field(repr=False)
    _seal: object = field(repr=False, compare=False)
    _context_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class RagPostgresDatabaseIdentity:
    database_name: str
    current_schema: str
    effective_search_path: tuple[str, ...]
    search_path_setting: str
    current_role: str
    database_oid: int

    def __post_init__(self) -> None:
        if (
            not self.database_name
            or not self.current_schema
            or not self.effective_search_path
            or not self.search_path_setting
            or not self.current_role
            or type(self.database_oid) is not int
            or self.database_oid <= 0
        ):
            raise TypeError('RAG PostgreSQL database identity is incomplete')


@dataclass(frozen=True, slots=True)
class RagPostgresWritableServerIdentity:
    server_address: str
    server_port: int
    postmaster_start_time: str

    def __post_init__(self) -> None:
        if (
            not self.server_address
            or type(self.server_port) is not int
            or self.server_port <= 0
            or not self.postmaster_start_time
        ):
            raise TypeError('authoritative writable PostgreSQL server is unavailable')


@dataclass(frozen=True, slots=True)
class _PostgresDatabaseAssembly:
    session: Session = field(repr=False)
    application_engine: Engine = field(repr=False)
    dedicated_engine: Engine = field(repr=False)
    trusted_bootstrap: TrustedPostgresEngineBootstrap = field(repr=False)
    policy_capability_id: str
    bootstrap_capability: RegisteredAdvisoryLock = field(repr=False)
    identity: RagPostgresDatabaseIdentity
    server_identity: RagPostgresWritableServerIdentity
    runtime_health: TrustedPostgresRuntimeHealth = field(repr=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PostgresAdvisoryTransportAssembly:
    application_engine: Engine = field(repr=False)
    trusted_bootstrap: TrustedPostgresEngineBootstrap = field(repr=False)
    bootstrap_capability: RegisteredAdvisoryLock = field(repr=False)
    identity: RagPostgresDatabaseIdentity
    server_identity: RagPostgresWritableServerIdentity
    runtime_health: TrustedPostgresRuntimeHealth = field(repr=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _RagPostgresCleanupOwner:
    authority: RagPostgresDatabaseAuthority = field(repr=False)
    health_owner: _PostgresCleanupOwnerCapability = field(repr=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _RagPinnedConnectionResponsibility:
    original_session_bind: Engine | Connection = field(repr=False)
    application_connection: Connection | None = field(repr=False)
    owns_application_connection: bool


@dataclass(slots=True, repr=False)
class _RagPostgresEmergencyCleanupState:
    authority: RagPostgresDatabaseAuthority = field(repr=False)
    health_capability: _PostgresEmergencyCleanupCapability = field(repr=False)
    close_on_exit: bool
    captured_connections: tuple[Connection, ...] = field(repr=False)
    lease: _RagPostgresOperationLease | None = field(default=None, repr=False)
    connection_responsibility: _RagPinnedConnectionResponsibility | None = field(
        default=None,
        repr=False,
    )
    cleanup_uncertain: bool = False
    health_owner: _PostgresCleanupOwnerCapability | None = field(
        default=None,
        repr=False,
    )
    cleanup_owner: _RagPostgresCleanupOwner | None = field(
        default=None,
        repr=False,
    )
    active: bool = field(default=True, repr=False)
    _seal: object = field(default=None, repr=False, compare=False)


class RagPostgresAdvisoryTransport:
    """Bootstrap-bound one-physical-use advisory connection transport."""

    __slots__ = ('_active_connections', '_assembly', '_lock')

    def __init__(self, assembly: object) -> None:
        if (
            type(assembly) is not _PostgresAdvisoryTransportAssembly
            or assembly._seal is not _POSTGRES_ADVISORY_TRANSPORT_SEAL
            or not isinstance(assembly.application_engine, Engine)
            or assembly.application_engine.dialect.name != 'postgresql'
            or type(assembly.trusted_bootstrap)
            is not TrustedPostgresEngineBootstrap
            or type(assembly.bootstrap_capability) is not RegisteredAdvisoryLock
            or type(assembly.identity) is not RagPostgresDatabaseIdentity
            or type(assembly.server_identity)
            is not RagPostgresWritableServerIdentity
            or type(assembly.runtime_health) is not TrustedPostgresRuntimeHealth
        ):
            raise TypeError('RAG PostgreSQL advisory transport is unavailable')
        self._assembly = assembly
        self._active_connections: set[Connection] = set()
        self._lock = RLock()

    @property
    def runtime_health_authority(self) -> TrustedPostgresRuntimeHealth:
        return self._assembly.runtime_health

    @property
    def application_engine_authority(self) -> Engine:
        return self._assembly.application_engine

    @contextmanager
    def __call__(self):
        """Yield one dedicated connection and physically discard it exactly once."""
        health = self._assembly.runtime_health
        with health._effect('rag_provider_advisory_connection'):
            dedicated_engine: Engine | None = None
            connection: Connection | None = None
            primary: BaseException | None = None
            cleanup_failed = False
            try:
                issued = self._assembly.trusted_bootstrap._issue(
                    self._assembly.application_engine
                )
                if issued.runtime_health is not health:
                    raise TypeError('RAG PostgreSQL advisory health changed')
                dedicated_engine = issued.engine
                connection = dedicated_engine.connect()
                if connection.engine is not dedicated_engine:
                    raise TypeError('RAG PostgreSQL advisory connection changed')
                if _connection_identity(connection) != self._assembly.identity:
                    raise TypeError('RAG PostgreSQL advisory database changed')
                if (
                    _connection_server_identity(connection)
                    != self._assembly.server_identity
                ):
                    raise TypeError('RAG PostgreSQL advisory server changed')
                _require_bootstrap_capability(
                    connection,
                    self._assembly.bootstrap_capability,
                )
                with self._lock:
                    self._active_connections.add(connection)
                yield connection
            except BaseException as exc:
                primary = exc
            finally:
                if connection is not None:
                    with self._lock:
                        self._active_connections.discard(connection)
                    if primary is not None and not connection.closed:
                        try:
                            connection.invalidate()
                        except BaseException:
                            cleanup_failed = True
                    try:
                        connection.close()
                    except BaseException:
                        cleanup_failed = True
                if dedicated_engine is not None:
                    try:
                        dedicated_engine.dispose()
                    except BaseException:
                        cleanup_failed = True
                if cleanup_failed:
                    health._poison()
            if primary is not None:
                raise primary
            if cleanup_failed:
                raise RagPostgresAdvisoryCleanupError(
                    'RAG PostgreSQL advisory cleanup failed'
                )

    @contextmanager
    def advisory(self, capability: RegisteredAdvisoryLock, *, shared: bool):
        if type(capability) is not RegisteredAdvisoryLock or type(shared) is not bool:
            raise TypeError('registered advisory transport capability is required')
        with (
            self() as connection,
            self.advisory_connection(
                connection,
                capability,
                shared=shared,
            ),
        ):
            yield connection

    @contextmanager
    def advisory_connection(
        self,
        connection: Connection,
        capability: RegisteredAdvisoryLock,
        *,
        shared: bool,
    ):
        if type(capability) is not RegisteredAdvisoryLock or type(shared) is not bool:
            raise TypeError('registered advisory transport capability is required')
        with self._lock:
            if connection not in self._active_connections:
                raise TypeError('provider advisory connection authority changed')
        primary: BaseException | None = None
        cleanup_failed = False
        locked = False
        try:
            acquire_advisory_lock(connection, capability, shared=shared)
            locked = True
            yield
        except BaseException as exc:
            primary = exc
        finally:
            if locked:
                try:
                    release_advisory_lock(
                        connection,
                        capability,
                        shared=shared,
                    )
                except BaseException:
                    cleanup_failed = True
                    self._assembly.runtime_health._poison()
        if primary is not None:
            raise primary
        if cleanup_failed:
            raise RagPostgresAdvisoryCleanupError(
                'RAG PostgreSQL advisory cleanup failed'
            )


class RagPostgresDatabaseAuthority:
    """Sealed exact-identity authority with physical-close lock connections."""

    __slots__ = (
        '_active_leases',
        '_assembly',
        '_connections',
        '_cleanup_failure',
        '_cleanup_owner_capability',
        '_emergency_cleanup_state',
        '_lease_context',
        '_lifecycle_lock',
        '_state',
        '_transport_disposal_complete',
    )

    def __init__(self, assembly: object) -> None:
        if (
            type(assembly) is not _PostgresDatabaseAssembly
            or assembly._seal is not _POSTGRES_DATABASE_AUTHORITY_SEAL
            or not isinstance(assembly.session, Session)
            or not isinstance(assembly.application_engine, Engine)
            or not isinstance(assembly.dedicated_engine, Engine)
            or type(assembly.dedicated_engine.pool) is not NullPool
            or assembly.application_engine.dialect.name != 'postgresql'
            or assembly.dedicated_engine.dialect.name != 'postgresql'
            or type(assembly.trusted_bootstrap)
            is not TrustedPostgresEngineBootstrap
            or len(assembly.policy_capability_id) != 64
            or type(assembly.identity) is not RagPostgresDatabaseIdentity
            or type(assembly.server_identity)
            is not RagPostgresWritableServerIdentity
            or type(assembly.bootstrap_capability) is not RegisteredAdvisoryLock
            or type(assembly.runtime_health) is not TrustedPostgresRuntimeHealth
        ):
            raise TypeError('RAG PostgreSQL database authority is unavailable')
        self._assembly = assembly
        self._active_leases = 0
        self._state: Literal['open', 'closing', 'closed'] = 'open'
        self._connections: set[Connection] = set()
        self._cleanup_failure: RagPostgresDatabaseCleanupFailure | None = None
        self._cleanup_owner_capability: _RagPostgresCleanupOwner | None = None
        self._emergency_cleanup_state: _RagPostgresEmergencyCleanupState | None = (
            None
        )
        self._lease_context: ContextVar[_RagPostgresOperationLease | None] = (
            ContextVar(
                f'rag_postgres_operation_lease_{id(self)}',
                default=None,
            )
        )
        self._lifecycle_lock = RLock()
        self._transport_disposal_complete = False

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._state == 'closed'

    @property
    def cleanup_failure(self) -> RagPostgresDatabaseCleanupFailure | None:
        with self._lifecycle_lock:
            return self._cleanup_failure

    @property
    def runtime_health_authority(self) -> TrustedPostgresRuntimeHealth:
        return self._assembly.runtime_health

    def __enter__(self) -> RagPostgresDatabaseAuthority:
        with self._lifecycle_lock:
            self._require_usable()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self) -> Connection:
        """Open a never-pooled physical connection and validate DB identity."""
        with self._lifecycle_lock:
            lease = self._require_active_operation()
            with self._assembly.runtime_health._guard(
                lease.runtime_health_lease
            ):
                self._connections = {
                    connection
                    for connection in self._connections
                    if not connection.closed
                }
                connection = self._assembly.dedicated_engine.connect()
                try:
                    if connection.engine is not self._assembly.dedicated_engine:
                        raise TypeError('RAG PostgreSQL connection authority changed')
                    current = _connection_identity(connection)
                    if current != self._assembly.identity:
                        raise TypeError('RAG PostgreSQL connection identity changed')
                    if (
                        _connection_server_identity(connection)
                        != self._assembly.server_identity
                    ):
                        raise TypeError(
                            'authoritative writable PostgreSQL server changed'
                        )
                    _require_bootstrap_capability(
                        connection,
                        self._assembly.bootstrap_capability,
                    )
                    self._connections.add(connection)
                    return connection
                except BaseException:
                    try:
                        connection.invalidate()
                    finally:
                        connection.close()
                    raise

    def require_session(self, session: Session) -> None:
        with self._lifecycle_lock:
            self._require_usable()
            if session is not self._assembly.session:
                raise TypeError('RAG PostgreSQL session authority changed')
            bind = session.get_bind()
            engine = bind.engine if isinstance(bind, Connection) else bind
            if engine is not self._assembly.application_engine:
                raise TypeError('RAG PostgreSQL engine authority changed')
            lease = self._lease_context.get()
            if type(lease) is _RagPostgresOperationLease:
                if (
                    lease._seal is not _OPERATION_LEASE_SEAL
                    or lease.authority is not self
                    or bind is not lease.application_connection
                    or not session.in_transaction()
                ):
                    raise TypeError('pinned PostgreSQL transaction changed')
                with self._assembly.runtime_health._guard(
                    lease.runtime_health_lease
                ):
                    self._validate_application_connection(bind)
                return
            try:
                if _session_identity(session) != self._assembly.identity:
                    raise TypeError('RAG PostgreSQL session identity changed')
                if (
                    _session_server_identity(session)
                    != self._assembly.server_identity
                ):
                    raise TypeError(
                        'authoritative writable PostgreSQL server changed'
                    )
                _require_bootstrap_capability(
                    session,
                    self._assembly.bootstrap_capability,
                )
            finally:
                session.rollback()

    def require_session_transaction_ended(self, session: Session) -> None:
        """Prove the Session-owned root transaction physically ended."""
        with self._lifecycle_lock:
            lease = self._require_active_operation()
            if (
                session is not self._assembly.session
                or session.get_bind() is not lease.application_connection
                or session.in_transaction()
                or lease.application_connection.in_transaction()
                or lease.application_connection.in_nested_transaction()
            ):
                raise TypeError('pinned PostgreSQL transaction did not end')

    @contextmanager
    def operation_lease(self):
        with self._leased_operation(close_on_exit=False) as lease:
            yield lease

    @contextmanager
    def owned_operation(self):
        """One-shot request owner including transport cleanup and health poison."""
        with self._leased_operation(close_on_exit=True) as lease:
            yield lease

    @contextmanager
    def _leased_operation(self, *, close_on_exit: bool):
        health = self._assembly.runtime_health
        with health._operation('rag_finalization_or_recovery') as health_lease:
            setup_uncertain = False
            try:
                emergency_capability = (
                    health._mint_emergency_cleanup_capability(
                        health_lease,
                        authority=self,
                    )
                )
            except BaseException:
                setup_uncertain = True
                emergency_capability = (
                    health._force_mint_emergency_cleanup_capability(
                        health_lease,
                        authority=self,
                    )
                )
            try:
                state = self._install_emergency_cleanup_state(
                    emergency_capability,
                    close_on_exit=close_on_exit,
                )
            except BaseException:
                setup_uncertain = True
                state = self._force_install_emergency_cleanup_state(
                    emergency_capability,
                    close_on_exit=close_on_exit,
                )
            if setup_uncertain:
                self._mark_cleanup_uncertain(state)
            primary: BaseException | None = None
            primary_traceback = None
            try:
                with health._guard(health_lease):
                    lease = self._pin_application_transaction(
                        health_lease,
                        state,
                    )
                    try:
                        yield lease
                    except BaseException as exc:
                        primary = exc
                        primary_traceback = exc.__traceback__
                    finally:
                        try:
                            self._capture_emergency_connections(state)
                        except BaseException:
                            self._mark_cleanup_uncertain(state)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                    primary_traceback = exc.__traceback__
            finally:
                try:
                    self._run_cleanup_state_machine(state)
                except BaseException:
                    self._force_terminal_cleanup_noexcept(state)
            if primary is not None:
                raise primary.with_traceback(primary_traceback)

    def _install_emergency_cleanup_state(
        self,
        emergency_capability: _PostgresEmergencyCleanupCapability,
        *,
        close_on_exit: bool,
    ) -> _RagPostgresEmergencyCleanupState:
        state = _RagPostgresEmergencyCleanupState(
            authority=self,
            health_capability=emergency_capability,
            close_on_exit=close_on_exit,
            captured_connections=(),
            _seal=_EMERGENCY_CLEANUP_STATE_SEAL,
        )
        with self._lifecycle_lock:
            if self._emergency_cleanup_state is not None:
                raise RuntimeError('RAG PostgreSQL cleanup state changed')
            self._emergency_cleanup_state = state
        return state

    def _force_install_emergency_cleanup_state(
        self,
        emergency_capability: _PostgresEmergencyCleanupCapability,
        *,
        close_on_exit: bool,
    ) -> _RagPostgresEmergencyCleanupState:
        with self._lifecycle_lock:
            current = self._emergency_cleanup_state
            if current is not None:
                if (
                    type(current) is not _RagPostgresEmergencyCleanupState
                    or current._seal is not _EMERGENCY_CLEANUP_STATE_SEAL
                    or current.authority is not self
                    or current.health_capability is not emergency_capability
                ):
                    raise TypeError('RAG PostgreSQL cleanup state changed')
                return current
            state = _RagPostgresEmergencyCleanupState(
                authority=self,
                health_capability=emergency_capability,
                close_on_exit=close_on_exit,
                captured_connections=(),
                _seal=_EMERGENCY_CLEANUP_STATE_SEAL,
            )
            self._emergency_cleanup_state = state
            return state

    def _capture_emergency_connections(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        self._require_emergency_cleanup_state(state)
        with self._lifecycle_lock:
            state.captured_connections = tuple(self._connections)

    def _run_cleanup_state_machine(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        """Drain one request authority without allowing cleanup to become primary."""
        self._require_emergency_cleanup_state(state)
        health = self._assembly.runtime_health
        try:
            for _attempt in range(2):
                try:
                    state.health_owner = health._enter_cleanup(
                        emergency_capability=state.health_capability,
                    )
                    break
                except BaseException:
                    self._mark_cleanup_uncertain(state)
            if state.health_owner is not None:
                try:
                    state.cleanup_owner = self._mint_cleanup_owner(
                        state.health_owner
                    )
                except BaseException:
                    self._mark_cleanup_uncertain(state)
                    state.cleanup_owner = self._recover_cleanup_owner(state)
                    if state.cleanup_owner is None:
                        try:
                            state.cleanup_owner = self._mint_cleanup_owner(
                                state.health_owner
                            )
                        except BaseException:
                            self._mark_cleanup_uncertain(state)
            try:
                if state.lease is not None:
                    self._release_application_transaction(state.lease)
            except BaseException:
                self._mark_cleanup_uncertain(state)
                self._emergency_release_application_transaction(state)
            if state.close_on_exit:
                try:
                    if state.cleanup_owner is None:
                        raise TypeError('RAG PostgreSQL cleanup owner is unavailable')
                    self._close_under_cleanup_owner(state.cleanup_owner)
                except BaseException:
                    self._mark_cleanup_uncertain(state)
                    self._emergency_close_owned_transport(state)
        except BaseException:
            self._mark_cleanup_uncertain(state)
        finally:
            self._emergency_release_application_transaction(state)
            if state.close_on_exit:
                self._emergency_close_owned_transport(state)
            if state.health_owner is not None:
                try:
                    health._exit_cleanup(state.health_owner)
                except BaseException:
                    self._mark_cleanup_uncertain(state)
            self._finalize_terminal_cleanup_state(state)

    def _finalize_terminal_cleanup_state(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        """Never allow terminal probes/hooks to replace the captured outcome."""
        health = self._assembly.runtime_health
        uncertain = False
        try:
            health._finish_emergency_cleanup(state.health_capability)
        except BaseException:
            uncertain = True
        try:
            health._emergency_cleanup_capability_is_active(
                state.health_capability
            )
        except BaseException:
            uncertain = True
        try:
            self._terminal_cleanup_checkpoint('before_state_clear')
        except BaseException:
            uncertain = True
        try:
            with self._lifecycle_lock:
                if self._emergency_cleanup_state is state:
                    self._emergency_cleanup_state = None
                self._cleanup_owner_capability = None
        except BaseException:
            uncertain = True
        finally:
            with suppress(BaseException):
                if self._emergency_cleanup_state is state:
                    self._emergency_cleanup_state = None
                self._cleanup_owner_capability = None
            state.active = False
        try:
            self._terminal_cleanup_checkpoint('after_state_clear')
        except BaseException:
            uncertain = True
        if uncertain:
            with self._lifecycle_lock:
                if self._cleanup_failure is None:
                    self._cleanup_failure = RagPostgresDatabaseCleanupFailure()
        with suppress(BaseException):
            health._force_emergency_cleanup_disposition(
                state.health_capability,
                authority=self,
                poison=uncertain,
            )

    def _terminal_cleanup_checkpoint(self, _stage: str) -> None:
        """Test seam around the terminal non-throwing drain."""

    def _force_terminal_cleanup_noexcept(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        """Sealed last-resort drain; every individual action is best-effort."""
        health = self._assembly.runtime_health
        with suppress(BaseException):
            with self._lifecycle_lock:
                if self._cleanup_failure is None:
                    self._cleanup_failure = RagPostgresDatabaseCleanupFailure()
                self._active_leases = 0
            self._lease_context.set(None)
        with suppress(BaseException):
            self._assembly.trusted_bootstrap._drain_registered_application_checkout(
                state.health_capability
            )
        session = self._assembly.session
        with suppress(BaseException):
            if session.in_transaction():
                session.rollback()
        lease = state.lease
        responsibility = state.connection_responsibility
        connection = (
            lease.application_connection
            if lease is not None
            else (
                None
                if responsibility is None
                else responsibility.application_connection
            )
        )
        original_bind = (
            lease.original_session_bind
            if lease is not None
            else (
                None
                if responsibility is None
                else responsibility.original_session_bind
            )
        )
        owns_connection = (
            lease.owns_application_connection
            if lease is not None
            else bool(
                responsibility is not None
                and responsibility.owns_application_connection
            )
        )
        if connection is not None:
            with suppress(BaseException):
                if connection.in_transaction() or connection.in_nested_transaction():
                    connection.rollback()
            if owns_connection:
                with suppress(BaseException):
                    connection.invalidate()
                with suppress(BaseException):
                    connection.close()
        if original_bind is not None:
            with suppress(BaseException):
                session.bind = original_bind
        if state.close_on_exit:
            with suppress(BaseException):
                with self._lifecycle_lock:
                    connections = tuple(self._connections)
                    self._connections.clear()
                    self._state = 'closed'
                for advisory_connection in connections:
                    with suppress(BaseException):
                        advisory_connection.invalidate()
                    with suppress(BaseException):
                        advisory_connection.close()
                if not self._transport_disposal_complete:
                    with suppress(BaseException):
                        self._assembly.dedicated_engine.dispose()
                    self._transport_disposal_complete = True
        with suppress(BaseException):
            health._force_emergency_cleanup_disposition(
                state.health_capability,
                authority=self,
                poison=True,
            )
        with suppress(BaseException), self._lifecycle_lock:
            if self._emergency_cleanup_state is state:
                self._emergency_cleanup_state = None
            self._cleanup_owner_capability = None
            self._active_leases = 0
            state.active = False

    def _recover_cleanup_owner(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> _RagPostgresCleanupOwner | None:
        with self._lifecycle_lock:
            owner = self._cleanup_owner_capability
        if (
            type(owner) is not _RagPostgresCleanupOwner
            or owner._seal is not _CLEANUP_OWNER_SEAL
            or owner.authority is not self
            or owner.health_owner is not state.health_owner
        ):
            return None
        try:
            self._assembly.runtime_health._require_cleanup_owner(
                owner.health_owner,
                outermost=True,
            )
        except BaseException:
            return None
        return owner

    def _mark_cleanup_uncertain(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        self._require_emergency_cleanup_state(state)
        state.cleanup_uncertain = True
        with self._lifecycle_lock:
            if self._cleanup_failure is None:
                self._cleanup_failure = RagPostgresDatabaseCleanupFailure()
        try:
            self._assembly.runtime_health._emergency_fail_stop(
                state.health_capability
            )
        except BaseException:
            pass
        finally:
            self._assembly.runtime_health._force_emergency_fail_stop(
                state.health_capability
            )

    def _emergency_release_application_transaction(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        self._require_emergency_cleanup_state(state)
        session = self._assembly.session
        lease = state.lease
        responsibility = state.connection_responsibility
        connection = (
            lease.application_connection
            if lease is not None
            else (
                None
                if responsibility is None
                else responsibility.application_connection
            )
        )
        original_bind = (
            lease.original_session_bind
            if lease is not None
            else (
                None
                if responsibility is None
                else responsibility.original_session_bind
            )
        )
        owns_connection = (
            lease.owns_application_connection
            if lease is not None
            else (
                False
                if responsibility is None
                else responsibility.owns_application_connection
            )
        )
        failed = self._assembly.trusted_bootstrap._drain_registered_application_checkout(
            state.health_capability
        )
        with self._lifecycle_lock:
            current = self._lease_context.get()
            if lease is not None and current is lease:
                token = lease._context_token
                if token is not None:
                    try:
                        self._lease_context.reset(token)
                    except BaseException:
                        failed = True
                        self._lease_context.set(None)
                else:
                    self._lease_context.set(None)
            elif current is not None:
                failed = True
                self._lease_context.set(None)
            self._active_leases = 0
        for _attempt in range(2):
            try:
                if session.in_transaction():
                    session.rollback()
                break
            except BaseException:
                failed = True
        if connection is not None:
            for _attempt in range(2):
                try:
                    if (
                        connection.in_transaction()
                        or connection.in_nested_transaction()
                    ):
                        connection.rollback()
                    break
                except BaseException:
                    failed = True
                    with suppress(BaseException):
                        connection.invalidate()
        if original_bind is not None:
            for _attempt in range(2):
                try:
                    session.bind = original_bind
                    break
                except BaseException:
                    failed = True
        if (
            connection is not None
            and owns_connection
            and not connection.closed
        ):
            for _attempt in range(2):
                try:
                    connection.close()
                    break
                except BaseException:
                    failed = True
                    with suppress(BaseException):
                        connection.invalidate()
        if failed:
            self._mark_cleanup_uncertain(state)

    def _emergency_close_owned_transport(
        self,
        state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        self._require_emergency_cleanup_state(state)
        failed = False
        with self._lifecycle_lock:
            self._active_leases = 0
            self._state = 'closed'
            connections = tuple(
                dict.fromkeys((*state.captured_connections, *self._connections))
            )
            self._connections.clear()
        for connection in connections:
            if connection.closed:
                continue
            try:
                connection.invalidate()
            except BaseException:
                failed = True
            for _attempt in range(2):
                try:
                    connection.close()
                    break
                except BaseException:
                    failed = True
        if not self._transport_disposal_complete:
            try:
                self._assembly.dedicated_engine.dispose()
            except BaseException:
                failed = True
            finally:
                self._transport_disposal_complete = True
        if failed:
            self._mark_cleanup_uncertain(state)

    def _require_emergency_cleanup_state(
        self,
        state: object,
    ) -> None:
        health = self._assembly.runtime_health
        if (
            type(state) is not _RagPostgresEmergencyCleanupState
            or state._seal is not _EMERGENCY_CLEANUP_STATE_SEAL
            or state.authority is not self
            or not state.active
            or self._emergency_cleanup_state is not state
            or (state.lease is not None and state.lease.authority is not self)
        ):
            raise TypeError('RAG PostgreSQL emergency cleanup state changed')
        try:
            health._require_emergency_cleanup_capability(
                state.health_capability,
                authority=self,
            )
        except BaseException as exc:
            raise TypeError(
                'RAG PostgreSQL emergency cleanup state changed'
            ) from exc

    def _pin_application_transaction(
        self,
        health_lease: _PostgresRuntimeHealthLease,
        cleanup_state: _RagPostgresEmergencyCleanupState | None = None,
    ) -> _RagPostgresOperationLease:
        with self._lifecycle_lock:
            if self._state != 'open':
                raise TypeError('RAG PostgreSQL database authority is closing')
            if self._lease_context.get() is not None or self._active_leases:
                raise TypeError('RAG PostgreSQL operation is already leased')
            session = self._assembly.session
            if session.in_transaction():
                raise TypeError('fresh pinned PostgreSQL transaction is required')
            original_bind = session.get_bind()
            if cleanup_state is not None:
                self._require_emergency_cleanup_state(cleanup_state)
            original_engine = (
                original_bind.engine
                if isinstance(original_bind, Connection)
                else original_bind
            )
            if original_engine is not self._assembly.application_engine:
                raise TypeError('RAG PostgreSQL engine authority changed')
            owns_connection = not isinstance(original_bind, Connection)
            connection = self._checkout_application_connection(
                original_bind=original_bind,
                owns_connection=owns_connection,
                cleanup_state=cleanup_state,
            )
            if connection.in_transaction() or connection.in_nested_transaction():
                if owns_connection:
                    connection.close()
                raise TypeError('fresh pinned PostgreSQL transaction is required')
            session.bind = connection
            lease = _RagPostgresOperationLease(
                authority=self,
                application_connection=connection,
                original_session_bind=original_bind,
                owns_application_connection=owns_connection,
                runtime_health_lease=health_lease,
                _seal=_OPERATION_LEASE_SEAL,
            )
            if cleanup_state is not None:
                cleanup_state.lease = lease
        try:
            with self._assembly.runtime_health._guard(health_lease):
                session.begin()
                enlisted = session.connection()
                if enlisted is not connection:
                    raise TypeError('pinned PostgreSQL connection was not enlisted')
                self._validate_application_connection(enlisted)
            self._publish_operation_lease(lease, cleanup_state=cleanup_state)
            return lease
        except BaseException:
            if 'lease' in locals():
                self._cleanup_pinned_connection(lease)
            raise

    def _checkout_application_connection(
        self,
        *,
        original_bind: Engine | Connection,
        owns_connection: bool,
        cleanup_state: _RagPostgresEmergencyCleanupState | None,
    ) -> Connection:
        """Use bootstrap-owned pool checkout registration before caller return."""
        if cleanup_state is None:
            raise TypeError('RAG PostgreSQL cleanup state is required')
        cleanup_state.connection_responsibility = (
            _RagPinnedConnectionResponsibility(
                original_session_bind=original_bind,
                application_connection=None,
                owns_application_connection=owns_connection,
            )
        )
        connection: Connection | None = None
        try:
            connection = (
                self._assembly.trusted_bootstrap._connect_registered_application(
                    self._assembly.application_engine,
                    cleanup_state.health_capability,
                )
                if owns_connection
                else original_bind
            )
            cleanup_state.connection_responsibility = (
                _RagPinnedConnectionResponsibility(
                    original_session_bind=original_bind,
                    application_connection=connection,
                    owns_application_connection=owns_connection,
                )
            )
            if owns_connection:
                self._assembly.trusted_bootstrap._release_registered_application_checkout(
                    cleanup_state.health_capability,
                    connection,
                )
            self._run_pin_publication_checkpoint(
                'after_connect',
                cleanup_state,
            )
            return connection
        except BaseException:
            self._mark_cleanup_uncertain(cleanup_state)
            raise PostgresRuntimeHealthUnavailableError(
                'RAG PostgreSQL runtime health is fail-stopped'
            ) from None

    def _publish_operation_lease(
        self,
        lease: _RagPostgresOperationLease,
        *,
        cleanup_state: _RagPostgresEmergencyCleanupState | None,
    ) -> None:
        context_token = None
        with self._lifecycle_lock:
            starting_active_leases = self._active_leases
        try:
            with self._lifecycle_lock:
                context_token = self._lease_context.set(lease)
                if cleanup_state is not None:
                    self._run_pin_publication_checkpoint(
                        'after_context',
                        cleanup_state,
                    )
                self._active_leases += 1
                if cleanup_state is not None:
                    self._run_pin_publication_checkpoint(
                        'after_count',
                        cleanup_state,
                    )
                object.__setattr__(lease, '_context_token', context_token)
                if cleanup_state is not None:
                    self._run_pin_publication_checkpoint(
                        'after_token',
                        cleanup_state,
                    )
        except BaseException:
            with self._lifecycle_lock:
                object.__setattr__(lease, '_context_token', None)
                self._active_leases = starting_active_leases
                if context_token is not None:
                    with suppress(BaseException):
                        self._lease_context.reset(context_token)
                if self._lease_context.get() is lease:
                    self._lease_context.set(None)
            raise

    def _run_pin_publication_checkpoint(
        self,
        stage: str,
        cleanup_state: _RagPostgresEmergencyCleanupState,
    ) -> None:
        try:
            self._pin_publication_checkpoint(stage)
        except BaseException:
            self._mark_cleanup_uncertain(cleanup_state)
            raise PostgresRuntimeHealthUnavailableError(
                'RAG PostgreSQL runtime health is fail-stopped'
            ) from None

    def _pin_publication_checkpoint(self, _stage: str) -> None:
        """Test seam; production publication is uninterrupted."""

    def _release_application_transaction(
        self,
        lease: _RagPostgresOperationLease,
    ) -> None:
        with self._lifecycle_lock:
            current = self._lease_context.get()
            if current is not lease:
                raise RuntimeError('RAG PostgreSQL authority lease changed')
            context_token = lease._context_token
            if context_token is None:
                raise RuntimeError('RAG PostgreSQL authority lease token is missing')
            self._lease_context.reset(context_token)
            self._active_leases -= 1
            if self._active_leases < 0:
                raise RuntimeError('RAG PostgreSQL authority lease underflow')
        self._cleanup_pinned_connection(lease)

    def _cleanup_pinned_connection(
        self,
        lease: _RagPostgresOperationLease,
    ) -> None:
        session = self._assembly.session
        connection = lease.application_connection
        failed = False
        try:
            if session.in_transaction():
                session.rollback()
            if connection.in_transaction() or connection.in_nested_transaction():
                failed = True
                with suppress(BaseException):
                    connection.rollback()
        except BaseException:
            failed = True
            with suppress(BaseException):
                connection.invalidate()
        finally:
            session.bind = lease.original_session_bind
            if lease.owns_application_connection:
                try:
                    connection.close()
                except BaseException:
                    failed = True
        if failed:
            self._record_cleanup_failure()

    def _validate_application_connection(self, connection: Connection) -> None:
        if connection.engine is not self._assembly.application_engine:
            raise TypeError('RAG PostgreSQL application connection changed')
        if _connection_identity(connection) != self._assembly.identity:
            raise TypeError('RAG PostgreSQL connection identity changed')
        if _connection_server_identity(connection) != self._assembly.server_identity:
            raise TypeError('authoritative writable PostgreSQL server changed')
        _require_bootstrap_capability(
            connection,
            self._assembly.bootstrap_capability,
        )

    @contextmanager
    def health_effect(self):
        with self._lifecycle_lock:
            lease = self._require_active_operation()
        with self._assembly.runtime_health._guard(lease.runtime_health_lease):
            yield

    def close(self) -> RagPostgresDatabaseCleanupFailure | None:
        """Stop new leases, then close idle request-owned transport once."""
        with self._lifecycle_lock:
            if self._active_leases:
                self._state = 'closing'
                raise RagPostgresDatabaseBusyError(
                    'RAG PostgreSQL database authority has an active operation'
                )
        health = self._assembly.runtime_health
        with health._cleanup_boundary() as health_owner:
            return self._close_under_cleanup_owner(
                self._mint_cleanup_owner(health_owner)
            )

    def _mint_cleanup_owner(
        self,
        health_owner: object,
    ) -> _RagPostgresCleanupOwner:
        health = self._assembly.runtime_health
        health._require_cleanup_owner(health_owner, outermost=True)
        cleanup_owner = _RagPostgresCleanupOwner(
            authority=self,
            health_owner=health_owner,
            _seal=_CLEANUP_OWNER_SEAL,
        )
        with self._lifecycle_lock:
            self._cleanup_owner_capability = cleanup_owner
        return cleanup_owner

    def _close_under_cleanup_owner(
        self,
        cleanup_owner: object,
    ) -> RagPostgresDatabaseCleanupFailure | None:
        health = self._assembly.runtime_health
        if (
            type(cleanup_owner) is not _RagPostgresCleanupOwner
            or cleanup_owner._seal is not _CLEANUP_OWNER_SEAL
            or cleanup_owner.authority is not self
            or self._cleanup_owner_capability is not cleanup_owner
        ):
            raise TypeError('RAG PostgreSQL cleanup owner changed')
        health._require_cleanup_owner(
            cleanup_owner.health_owner,
            outermost=True,
        )
        with self._lifecycle_lock:
            if self._state == 'closed':
                return self._cleanup_failure
            if self._active_leases:
                self._state = 'closing'
                raise RagPostgresDatabaseBusyError(
                    'RAG PostgreSQL database authority has an active operation'
                )
            self._state = 'closed'
        failure = self._dispose_owned_transport()
        with self._lifecycle_lock:
            if failure is not None:
                self._cleanup_failure = failure
            cleanup_failure = self._cleanup_failure
        if failure is not None:
            health._poison()
        return cleanup_failure

    def _record_cleanup_failure(self) -> None:
        with self._lifecycle_lock:
            self._cleanup_failure = RagPostgresDatabaseCleanupFailure()
        self._assembly.runtime_health._poison()

    def _dispose_owned_transport(
        self,
    ) -> RagPostgresDatabaseCleanupFailure | None:
        connections = tuple(self._connections)
        self._connections.clear()
        failure: BaseException | None = None
        for connection in connections:
            if connection.invalidated:
                failure = failure or RuntimeError(
                    'uncertain advisory connection was invalidated'
                )
            if connection.closed:
                continue
            try:
                connection.invalidate()
            except BaseException as exc:
                failure = failure or exc
            finally:
                try:
                    connection.close()
                except BaseException as exc:
                    failure = failure or exc
        try:
            self._assembly.dedicated_engine.dispose()
        except BaseException as exc:
            failure = failure or exc
        finally:
            self._transport_disposal_complete = True
        if failure is None:
            return None
        return RagPostgresDatabaseCleanupFailure()

    def _require_usable(self) -> None:
        if self._state == 'closed' or (
            self._state == 'closing' and self._active_leases == 0
        ):
            raise TypeError('RAG PostgreSQL database authority is closed')

    def _require_active_operation(self) -> _RagPostgresOperationLease:
        self._require_usable()
        lease = self._lease_context.get()
        if (
            self._active_leases <= 0
            or type(lease) is not _RagPostgresOperationLease
            or lease._seal is not _OPERATION_LEASE_SEAL
            or lease.authority is not self
        ):
            raise TypeError('RAG PostgreSQL operation lease is required')
        return lease


def _bind_rag_postgres_database(
    session: Session,
    *,
    trusted_bootstrap: TrustedPostgresEngineBootstrap,
    bootstrap_capability: RegisteredAdvisoryLock,
) -> RagPostgresDatabaseAuthority:
    if not isinstance(session, Session):
        raise TypeError('RAG PostgreSQL session authority is required')
    bind = session.get_bind()
    if session.in_transaction() or (
        isinstance(bind, Connection)
        and (bind.in_transaction() or bind.in_nested_transaction())
    ):
        raise TypeError('fresh PostgreSQL transaction is required')
    engine = bind.engine if isinstance(bind, Connection) else bind
    if not isinstance(engine, Engine) or engine.dialect.name != 'postgresql':
        raise TypeError('RAG PostgreSQL database authority requires PostgreSQL')
    if type(trusted_bootstrap) is not TrustedPostgresEngineBootstrap:
        raise TypeError('trusted PostgreSQL Engine bootstrap is required')
    if type(bootstrap_capability) is not RegisteredAdvisoryLock:
        raise TypeError('RAG PostgreSQL bootstrap capability is required')
    dedicated_engine: Engine | None = None
    try:
        issued = trusted_bootstrap._issue(engine)
        dedicated_engine = issued.engine
        try:
            identity = _session_identity(session)
            server_identity = _session_server_identity(session)
            _require_bootstrap_capability(session, bootstrap_capability)
        finally:
            session.rollback()
        with dedicated_engine.connect() as connection:
            if _connection_identity(connection) != identity:
                raise TypeError(
                    'dedicated PostgreSQL database identity does not match session'
                )
            if _connection_server_identity(connection) != server_identity:
                raise TypeError(
                    'dedicated PostgreSQL writable server does not match session'
                )
            _require_bootstrap_capability(connection, bootstrap_capability)
        return RagPostgresDatabaseAuthority(
            _PostgresDatabaseAssembly(
                session=session,
                application_engine=engine,
                dedicated_engine=dedicated_engine,
                trusted_bootstrap=trusted_bootstrap,
                policy_capability_id=issued.policy_capability_id,
                bootstrap_capability=bootstrap_capability,
                identity=identity,
                server_identity=server_identity,
                runtime_health=issued.runtime_health,
                _seal=_POSTGRES_DATABASE_AUTHORITY_SEAL,
            )
        )
    except BaseException:
        if dedicated_engine is not None:
            dedicated_engine.dispose()
        raise


def _bind_rag_postgres_advisory_transport(
    session: Session,
    *,
    trusted_bootstrap: TrustedPostgresEngineBootstrap,
    bootstrap_capability: RegisteredAdvisoryLock,
) -> RagPostgresAdvisoryTransport:
    """Bind per-use NullPool advisory connections to one trusted DB runtime."""
    if not isinstance(session, Session):
        raise TypeError('RAG PostgreSQL advisory session is required')
    bind = session.get_bind()
    if session.in_transaction() or (
        isinstance(bind, Connection)
        and (bind.in_transaction() or bind.in_nested_transaction())
    ):
        raise TypeError('fresh PostgreSQL advisory binding is required')
    engine = bind.engine if isinstance(bind, Connection) else bind
    if not isinstance(engine, Engine) or engine.dialect.name != 'postgresql':
        raise TypeError('RAG PostgreSQL advisory transport requires PostgreSQL')
    if type(trusted_bootstrap) is not TrustedPostgresEngineBootstrap:
        raise TypeError('trusted PostgreSQL Engine bootstrap is required')
    if type(bootstrap_capability) is not RegisteredAdvisoryLock:
        raise TypeError('RAG PostgreSQL bootstrap capability is required')
    try:
        identity = _session_identity(session)
        server_identity = _session_server_identity(session)
        _require_bootstrap_capability(session, bootstrap_capability)
    finally:
        session.rollback()
    runtime_health = trusted_bootstrap._runtime_effect_authority(engine)
    return RagPostgresAdvisoryTransport(
        _PostgresAdvisoryTransportAssembly(
            application_engine=engine,
            trusted_bootstrap=trusted_bootstrap,
            bootstrap_capability=bootstrap_capability,
            identity=identity,
            server_identity=server_identity,
            runtime_health=runtime_health,
            _seal=_POSTGRES_ADVISORY_TRANSPORT_SEAL,
        )
    )


def _require_bootstrap_capability(
    connection: object,
    capability: RegisteredAdvisoryLock,
) -> None:
    if (
        type(capability) is not RegisteredAdvisoryLock
        or not capability.matches(
            RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID,
            identity_namespace='static',
        )
    ):
        raise TypeError('RAG PostgreSQL bootstrap capability is invalid')
    _require_registered_capability(connection, capability)


def _session_identity(session: Session) -> RagPostgresDatabaseIdentity:
    if session.in_transaction():
        raise TypeError('fresh PostgreSQL session identity requires no transaction')
    row = session.execute(_IDENTITY_SQL).one()
    return _identity_from_row(row)


def _connection_identity(connection: Connection) -> RagPostgresDatabaseIdentity:
    row = connection.execute(_IDENTITY_SQL).one()
    return _identity_from_row(row)


def _session_server_identity(
    session: Session,
) -> RagPostgresWritableServerIdentity:
    row = session.execute(_SERVER_IDENTITY_SQL).one()
    return _server_identity_from_row(row)


def _connection_server_identity(
    connection: Connection,
) -> RagPostgresWritableServerIdentity:
    row = connection.execute(_SERVER_IDENTITY_SQL).one()
    return _server_identity_from_row(row)


def _identity_from_row(row: object) -> RagPostgresDatabaseIdentity:
    values = tuple(row)  # type: ignore[arg-type]
    if len(values) != 6 or not isinstance(values[2], (list, tuple)):
        raise TypeError('RAG PostgreSQL database identity is unavailable')
    return RagPostgresDatabaseIdentity(
        database_name=values[0],
        current_schema=values[1],
        effective_search_path=tuple(values[2]),
        search_path_setting=values[3],
        current_role=values[4],
        database_oid=values[5],
    )


def _server_identity_from_row(
    row: object,
) -> RagPostgresWritableServerIdentity:
    values = tuple(row)  # type: ignore[arg-type]
    if (
        len(values) != 5
        or values[3] is not False
        or values[4] != 'off'
    ):
        raise TypeError('authoritative writable PostgreSQL server is unavailable')
    return RagPostgresWritableServerIdentity(
        server_address=values[0],
        server_port=values[1],
        postmaster_start_time=values[2],
    )
