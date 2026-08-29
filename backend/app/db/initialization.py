from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, NoSuchModuleError
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    code = 'database_configuration_invalid'


class DatabaseInitializationError(RuntimeError):
    code = 'database_initialization_failed'


@dataclass(slots=True, repr=False)
class DatabaseRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]
    _dispose_attempted: bool = field(default=False, init=False, repr=False)

    def dispose(self) -> None:
        if self._dispose_attempted:
            return
        self._dispose_attempted = True
        self.engine.dispose()


def initialize_database_runtime(database_url: str) -> DatabaseRuntime:
    configuration_failure = False
    try:
        make_url(database_url)
    except ArgumentError:
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    configuration_failure = False
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
    except (NoSuchModuleError, ArgumentError):
        configuration_failure = True
    if configuration_failure:
        raise DatabaseConfigurationError()

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=True,
    )
    return DatabaseRuntime(engine=engine, session_factory=session_factory)
