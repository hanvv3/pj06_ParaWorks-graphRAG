from __future__ import annotations

from dataclasses import dataclass, field

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


class DatabaseConfigurationError(RuntimeError):
    code = 'database_configuration_invalid'


class DatabaseInitializationError(RuntimeError):
    code = 'database_initialization_failed'


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
    _dispose_attempted: bool = field(default=False, init=False, repr=False)

    def dispose(self) -> None:
        if self._dispose_attempted:
            return
        self._dispose_attempted = True
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


def initialize_database_runtime(database_url: str) -> DatabaseRuntime:
    configuration_failure = False
    try:
        make_url(database_url)
    except ArgumentError:
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    configuration_failure = False
    engine_failure: Exception | None = None
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
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
    return DatabaseRuntime(engine=engine, session_factory=session_factory)
