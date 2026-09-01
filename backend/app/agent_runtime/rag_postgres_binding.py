from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

_POSTGRES_DATABASE_AUTHORITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class _PostgresDatabaseAssembly:
    session: Session = field(repr=False)
    engine: Engine = field(repr=False)
    _seal: object = field(repr=False, compare=False)


class RagPostgresDatabaseAuthority:
    """Sealed exact-Session/Engine capability for RAG PostgreSQL locks."""

    __slots__ = ('_assembly',)

    def __init__(self, assembly: object) -> None:
        if (
            type(assembly) is not _PostgresDatabaseAssembly
            or assembly._seal is not _POSTGRES_DATABASE_AUTHORITY_SEAL
            or not isinstance(assembly.session, Session)
            or not isinstance(assembly.engine, Engine)
            or assembly.engine.dialect.name != 'postgresql'
        ):
            raise TypeError('RAG PostgreSQL database authority is unavailable')
        self._assembly = assembly

    def connect(self) -> Connection:
        connection = self._assembly.engine.connect()
        if connection.engine is not self._assembly.engine:
            connection.close()
            raise TypeError('RAG PostgreSQL connection authority changed')
        return connection

    def require_session(self, session: Session) -> None:
        if session is not self._assembly.session:
            raise TypeError('RAG PostgreSQL session authority changed')
        bind = session.get_bind()
        engine = bind.engine if isinstance(bind, Connection) else bind
        if engine is not self._assembly.engine:
            raise TypeError('RAG PostgreSQL engine authority changed')


def _bind_rag_postgres_database(session: Session) -> RagPostgresDatabaseAuthority:
    if not isinstance(session, Session):
        raise TypeError('RAG PostgreSQL session authority is required')
    bind = session.get_bind()
    engine = bind.engine if isinstance(bind, Connection) else bind
    if not isinstance(engine, Engine) or engine.dialect.name != 'postgresql':
        raise TypeError('RAG PostgreSQL database authority requires PostgreSQL')
    return RagPostgresDatabaseAuthority(
        _PostgresDatabaseAssembly(
            session=session,
            engine=engine,
            _seal=_POSTGRES_DATABASE_AUTHORITY_SEAL,
        )
    )
