from __future__ import annotations

from contextlib import contextmanager
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
)
from backend.app.db.initialization import TrustedPostgresEngineBootstrap

_POSTGRES_DATABASE_AUTHORITY_SEAL = object()
_OPERATION_LEASE_SEAL = object()
_IDENTITY_SQL = text(
    "SELECT current_database(), current_schema(), current_schemas(false), "
    "current_setting('search_path'), current_user, "
    "(SELECT oid FROM pg_database WHERE datname = current_database())"
)


class RagPostgresDatabaseBusyError(RuntimeError):
    """The request authority is draining an already active operation."""


@dataclass(frozen=True, slots=True)
class _RagPostgresOperationLease:
    authority: RagPostgresDatabaseAuthority = field(repr=False)
    _seal: object = field(repr=False, compare=False)


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
class _PostgresDatabaseAssembly:
    session: Session = field(repr=False)
    application_engine: Engine = field(repr=False)
    dedicated_engine: Engine = field(repr=False)
    trusted_bootstrap: TrustedPostgresEngineBootstrap = field(repr=False)
    policy_fingerprint: str
    bootstrap_capability: RegisteredAdvisoryLock = field(repr=False)
    identity: RagPostgresDatabaseIdentity
    _seal: object = field(repr=False, compare=False)


class RagPostgresDatabaseAuthority:
    """Sealed exact-identity authority with physical-close lock connections."""

    __slots__ = (
        '_active_leases',
        '_assembly',
        '_connections',
        '_lease_context',
        '_lifecycle_lock',
        '_state',
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
            or len(assembly.policy_fingerprint) != 64
            or type(assembly.identity) is not RagPostgresDatabaseIdentity
            or type(assembly.bootstrap_capability) is not RegisteredAdvisoryLock
        ):
            raise TypeError('RAG PostgreSQL database authority is unavailable')
        self._assembly = assembly
        self._active_leases = 0
        self._state: Literal['open', 'closing', 'closed'] = 'open'
        self._connections: set[Connection] = set()
        self._lease_context: ContextVar[_RagPostgresOperationLease | None] = (
            ContextVar(
                f'rag_postgres_operation_lease_{id(self)}',
                default=None,
            )
        )
        self._lifecycle_lock = RLock()

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._state == 'closed'

    def __enter__(self) -> RagPostgresDatabaseAuthority:
        with self._lifecycle_lock:
            self._require_usable()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self) -> Connection:
        """Open a never-pooled physical connection and validate DB identity."""
        with self._lifecycle_lock:
            self._require_active_operation()
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
            if _session_identity(session) != self._assembly.identity:
                raise TypeError('RAG PostgreSQL session identity changed')
            try:
                _require_bootstrap_capability(
                    session,
                    self._assembly.bootstrap_capability,
                )
            finally:
                session.rollback()

    @contextmanager
    def operation_lease(self):
        with self._lifecycle_lock:
            if self._state != 'open':
                raise TypeError('RAG PostgreSQL database authority is closing')
            if self._lease_context.get() is not None or self._active_leases:
                raise TypeError('RAG PostgreSQL operation is already leased')
            self.require_session(self._assembly.session)
            lease = _RagPostgresOperationLease(
                authority=self,
                _seal=_OPERATION_LEASE_SEAL,
            )
            context_token = self._lease_context.set(lease)
            self._active_leases += 1
        try:
            yield lease
        finally:
            with self._lifecycle_lock:
                self._lease_context.reset(context_token)
                self._active_leases -= 1
                if self._active_leases < 0:
                    raise RuntimeError('RAG PostgreSQL authority lease underflow')

    def close(self) -> None:
        """Stop new leases, then close idle request-owned transport once."""
        with self._lifecycle_lock:
            if self._state == 'closed':
                return
            if self._active_leases:
                self._state = 'closing'
                raise RagPostgresDatabaseBusyError(
                    'RAG PostgreSQL database authority has an active operation'
                )
            self._state = 'closed'
        self._dispose_owned_transport()

    def _dispose_owned_transport(self) -> None:
        connections = tuple(self._connections)
        self._connections.clear()
        failure: BaseException | None = None
        for connection in connections:
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
        if failure is not None:
            raise failure

    def _require_usable(self) -> None:
        if self._state == 'closed' or (
            self._state == 'closing' and self._active_leases == 0
        ):
            raise TypeError('RAG PostgreSQL database authority is closed')

    def _require_active_operation(self) -> None:
        self._require_usable()
        lease = self._lease_context.get()
        if (
            self._active_leases <= 0
            or type(lease) is not _RagPostgresOperationLease
            or lease._seal is not _OPERATION_LEASE_SEAL
            or lease.authority is not self
        ):
            raise TypeError('RAG PostgreSQL operation lease is required')


def _bind_rag_postgres_database(
    session: Session,
    *,
    trusted_bootstrap: TrustedPostgresEngineBootstrap,
    bootstrap_capability: RegisteredAdvisoryLock,
) -> RagPostgresDatabaseAuthority:
    if not isinstance(session, Session):
        raise TypeError('RAG PostgreSQL session authority is required')
    bind = session.get_bind()
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
        dedicated_engine = trusted_bootstrap._require_issued(issued, engine)
        identity = _session_identity(session)
        try:
            _require_bootstrap_capability(session, bootstrap_capability)
        finally:
            session.rollback()
        with dedicated_engine.connect() as connection:
            if _connection_identity(connection) != identity:
                raise TypeError(
                    'dedicated PostgreSQL database identity does not match session'
                )
            _require_bootstrap_capability(connection, bootstrap_capability)
        return RagPostgresDatabaseAuthority(
            _PostgresDatabaseAssembly(
                session=session,
                application_engine=engine,
                dedicated_engine=dedicated_engine,
                trusted_bootstrap=trusted_bootstrap,
                policy_fingerprint=issued.policy_fingerprint,
                bootstrap_capability=bootstrap_capability,
                identity=identity,
                _seal=_POSTGRES_DATABASE_AUTHORITY_SEAL,
            )
        )
    except BaseException:
        if dedicated_engine is not None:
            dedicated_engine.dispose()
        raise


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
    try:
        row = session.execute(_IDENTITY_SQL).one()
        return _identity_from_row(row)
    finally:
        session.rollback()


def _connection_identity(connection: Connection) -> RagPostgresDatabaseIdentity:
    row = connection.execute(_IDENTITY_SQL).one()
    return _identity_from_row(row)


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
