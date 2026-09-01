from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

_POSTGRES_DATABASE_AUTHORITY_SEAL = object()
_IDENTITY_SQL = text(
    "SELECT current_database(), current_schema(), current_schemas(false), "
    "current_setting('search_path'), current_user"
)


@dataclass(frozen=True, slots=True)
class RagPostgresDatabaseIdentity:
    database_name: str
    current_schema: str
    effective_search_path: tuple[str, ...]
    search_path_setting: str
    current_role: str

    def __post_init__(self) -> None:
        if (
            not self.database_name
            or not self.current_schema
            or not self.effective_search_path
            or not self.search_path_setting
            or not self.current_role
        ):
            raise TypeError('RAG PostgreSQL database identity is incomplete')


@dataclass(frozen=True, slots=True)
class _PostgresDatabaseAssembly:
    session: Session = field(repr=False)
    application_engine: Engine = field(repr=False)
    dedicated_engine: Engine = field(repr=False)
    identity: RagPostgresDatabaseIdentity
    _seal: object = field(repr=False, compare=False)


class RagPostgresDatabaseAuthority:
    """Sealed exact-identity authority with physical-close lock connections."""

    __slots__ = ('_assembly',)

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
            or type(assembly.identity) is not RagPostgresDatabaseIdentity
        ):
            raise TypeError('RAG PostgreSQL database authority is unavailable')
        self._assembly = assembly

    def connect(self) -> Connection:
        """Open a never-pooled physical connection and validate DB identity."""
        connection = self._assembly.dedicated_engine.connect()
        try:
            if connection.engine is not self._assembly.dedicated_engine:
                raise TypeError('RAG PostgreSQL connection authority changed')
            current = _connection_identity(connection)
            if current != self._assembly.identity:
                raise TypeError('RAG PostgreSQL connection identity changed')
            return connection
        except BaseException:
            try:
                connection.invalidate()
            finally:
                connection.close()
            raise

    def require_session(self, session: Session) -> None:
        if session is not self._assembly.session:
            raise TypeError('RAG PostgreSQL session authority changed')
        bind = session.get_bind()
        engine = bind.engine if isinstance(bind, Connection) else bind
        if engine is not self._assembly.application_engine:
            raise TypeError('RAG PostgreSQL engine authority changed')
        if _session_identity(session) != self._assembly.identity:
            raise TypeError('RAG PostgreSQL session identity changed')


def _bind_rag_postgres_database(session: Session) -> RagPostgresDatabaseAuthority:
    if not isinstance(session, Session):
        raise TypeError('RAG PostgreSQL session authority is required')
    bind = session.get_bind()
    engine = bind.engine if isinstance(bind, Connection) else bind
    if not isinstance(engine, Engine) or engine.dialect.name != 'postgresql':
        raise TypeError('RAG PostgreSQL database authority requires PostgreSQL')
    identity = _session_identity(session)
    dedicated_engine = create_engine(engine.url, poolclass=NullPool)
    try:
        with dedicated_engine.connect() as connection:
            if _connection_identity(connection) != identity:
                raise TypeError(
                    'dedicated PostgreSQL database identity does not match session'
                )
        return RagPostgresDatabaseAuthority(
            _PostgresDatabaseAssembly(
                session=session,
                application_engine=engine,
                dedicated_engine=dedicated_engine,
                identity=identity,
                _seal=_POSTGRES_DATABASE_AUTHORITY_SEAL,
            )
        )
    except BaseException:
        dedicated_engine.dispose()
        raise


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
    if len(values) != 5 or not isinstance(values[2], (list, tuple)):
        raise TypeError('RAG PostgreSQL database identity is unavailable')
    return RagPostgresDatabaseIdentity(
        database_name=values[0],
        current_schema=values[1],
        effective_search_path=tuple(values[2]),
        search_path_setting=values[3],
        current_role=values[4],
    )
