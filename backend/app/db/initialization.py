from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType

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


_POSTGRES_BOOTSTRAP_SEAL = object()


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
    _seal: object = field(repr=False, compare=False)


class TrustedPostgresEngineBootstrap:
    """Bootstrap-minted factory for per-request advisory transport."""

    __slots__ = (
        '_application_engine',
        '_dedicated_factory',
        '_policy_capability_id',
        '_revoked',
        '_seal',
        '_state_lock',
    )

    def __init__(
        self,
        *,
        application_engine: Engine,
        dedicated_factory: Callable[[], Engine],
        policy_capability_id: str,
        _seal: object,
    ) -> None:
        if (
            _seal is not _POSTGRES_BOOTSTRAP_SEAL
            or not isinstance(application_engine, Engine)
            or application_engine.dialect.name != 'postgresql'
            or not callable(dedicated_factory)
            or len(policy_capability_id) != 64
        ):
            raise TypeError('trusted PostgreSQL Engine bootstrap is unavailable')
        self._application_engine = application_engine
        self._dedicated_factory = dedicated_factory
        self._policy_capability_id = policy_capability_id
        self._revoked = False
        self._seal = _seal
        self._state_lock = RLock()

    def _issue(self, application_engine: Engine) -> _IssuedDedicatedPostgresEngine:
        with self._state_lock:
            if (
                self._revoked
                or self._seal is not _POSTGRES_BOOTSTRAP_SEAL
                or application_engine is not self._application_engine
            ):
                raise TypeError('PostgreSQL Engine bootstrap authority changed')
            dedicated_factory = self._dedicated_factory
        dedicated_engine = dedicated_factory()
        with self._state_lock:
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
                    _seal=_POSTGRES_BOOTSTRAP_SEAL,
                )
        if isinstance(dedicated_engine, Engine):
            dedicated_engine.dispose()
        raise TypeError('trusted dedicated PostgreSQL Engine is unavailable')

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

        def dedicated_factory() -> Engine:
            dedicated_options = _engine_options_for_create(engine_options)
            dedicated_options['poolclass'] = NullPool
            return create_engine(database_url, **dedicated_options)

        postgres_bootstrap = TrustedPostgresEngineBootstrap(
            application_engine=engine,
            dedicated_factory=dedicated_factory,
            policy_capability_id=policy_capability_id,
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
