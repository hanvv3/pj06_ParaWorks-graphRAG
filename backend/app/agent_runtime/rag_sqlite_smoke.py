from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

from backend.app.agent_runtime.rag_finalization import (
    AssistantProjectionTarget,
    CanonicalRagProjection,
    RagFinalProjection,
)
from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.agents.rag_orchestrator_agent.v2_input import PreparedRagRequestText

_T = TypeVar('_T')
_SQLITE_RAG_SMOKE_MUTEX = threading.RLock()
_PROCESS_LOCKS: dict[Path, object] = {}
_DATABASE_IDENTITIES: dict[tuple[int, int], Path] = {}


class SQLiteRagSmokeUnavailable(RuntimeError):  # noqa: N818 - approved API name
    pass


@dataclass(frozen=True, slots=True)
class SQLiteRagSmokeOperationResult:
    parent_agent_run_id: int
    projection: RagFinalProjection

    def __post_init__(self) -> None:
        if (
            type(self.parent_agent_run_id) is not int
            or self.parent_agent_run_id <= 0
            or not isinstance(self.projection, CanonicalRagProjection)
            and self.projection.__class__.__name__ != 'AssistantFinalizationRecord'
        ):
            raise ValueError('SQLite smoke operation result is invalid')


def sqlite_rag_smoke_mutex() -> threading.RLock:
    """The single never-replaced process mutex shared by smoke writers."""
    return _SQLITE_RAG_SMOKE_MUTEX


class SQLiteRagSmokeCoordinator:
    """Single-process provider-free smoke; never emulates PostgreSQL authority."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        database_path: Path | str | None,
        configured_backend: str,
        provider_dispatch_count: int,
        production_mode: bool,
        automated_test: bool = True,
        keyword_operation: Callable[..., SQLiteRagSmokeOperationResult] | None = None,
    ) -> None:
        if (
            not isinstance(connection, sqlite3.Connection)
            or configured_backend != 'keyword'
            or type(provider_dispatch_count) is not int
            or provider_dispatch_count != 0
            or type(production_mode) is not bool
            or production_mode
        ):
            raise SQLiteRagSmokeUnavailable(
                'SQLite RAG smoke requires keyword, provider-free, non-production mode'
            )
        self._connection = connection
        self._keyword_operation = keyword_operation
        self._path = self._validated_path(database_path, automated_test)
        if self._path is not None:
            self._validate_connection_path(self._path)
            self._acquire_process_lock(self._path)

    def run_atomic(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        if not callable(operation):
            raise TypeError('SQLite smoke operation must be callable')
        with _SQLITE_RAG_SMOKE_MUTEX:
            if self._connection.in_transaction:
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke requires sole transaction ownership'
                )
            try:
                self._connection.execute('BEGIN IMMEDIATE')
                result = operation(self._connection)
                if not self._connection.in_transaction:
                    raise SQLiteRagSmokeUnavailable(
                        'SQLite smoke operation escaped its transaction'
                    )
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def run_keyword(
        self,
        request: PreparedRagRequestText,
        *,
        scope: SecurityScope,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagFinalProjection:
        if (
            type(request) is not PreparedRagRequestText
            or type(scope) is not SecurityScope
            or (
                assistant_target is not None
                and type(assistant_target) is not AssistantProjectionTarget
            )
            or not callable(self._keyword_operation)
        ):
            raise SQLiteRagSmokeUnavailable(
                'SQLite keyword smoke finalizer is unavailable'
            )
        result = self.run_atomic(
            lambda db: self._checked_keyword_operation(
                db,
                request=request,
                scope=scope,
                assistant_target=assistant_target,
            )
        )
        return result.projection

    def _checked_keyword_operation(
        self,
        db: sqlite3.Connection,
        *,
        request: PreparedRagRequestText,
        scope: SecurityScope,
        assistant_target: AssistantProjectionTarget | None,
    ) -> SQLiteRagSmokeOperationResult:
        result = self._keyword_operation(
            db,
            request=request,
            scope=scope,
            assistant_target=assistant_target,
        )
        if type(result) is not SQLiteRagSmokeOperationResult:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke operation did not return committed authority'
            )
        self._assert_final_invariants(db, result)
        return result

    @staticmethod
    def _assert_final_invariants(
        db: sqlite3.Connection,
        result: SQLiteRagSmokeOperationResult,
    ) -> None:
        parent = db.execute(
            'SELECT status,run_contract_version,run_record_phase,'
            'total_charged_cost_usd,projection_owner_fence_hmac '
            'FROM agent_runs WHERE id = ?',
            (result.parent_agent_run_id,),
        ).fetchone()
        children = tuple(
            db.execute(
                'SELECT component,component_ordinal,dispatch_state,attempted,'
                'dispatch_count,charged_cost_usd FROM agent_run_cost_components '
                'WHERE agent_run_id = ? ORDER BY component_ordinal',
                (result.parent_agent_run_id,),
            )
        )
        if (
            parent is None
            or parent[0] != 'complete'
            or parent[1] != 'rag-run:v2'
            or parent[2] != 'final'
            or Decimal(str(parent[3])) != Decimal('0.000000')
            or parent[4] is not None
            or tuple((row[0], row[1]) for row in children)
            != (('query_embedding', 0), ('answer_generation', 1))
            or any(
                row[2] != 'terminal'
                or row[3] != 0
                or row[4] != 0
                or Decimal(str(row[5])) != Decimal('0.000000')
                for row in children
            )
        ):
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke exact-two final invariant failed'
            )
        residue = db.execute(
            "SELECT count(*) FROM agent_runs WHERE run_contract_version = 'rag-run:v2' "
            "AND run_record_phase = 'cost_finalized_pending_projection'"
        ).fetchone()
        if residue is None or residue[0] != 0:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke cannot commit pending projection residue'
            )

    def _validate_connection_path(self, database_path: Path) -> None:
        try:
            rows = tuple(self._connection.execute('PRAGMA database_list'))
            main = next(row for row in rows if row[1] == 'main')
            actual = Path(main[2]).absolute()
            if (
                not main[2]
                or actual.resolve(strict=True)
                != database_path.resolve(strict=True)
                or actual.stat().st_dev != database_path.stat().st_dev
                or actual.stat().st_ino != database_path.stat().st_ino
            ):
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke connection/path identity mismatch'
                )
        except SQLiteRagSmokeUnavailable:
            raise
        except (OSError, sqlite3.Error, StopIteration) as exc:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke connection identity is unavailable'
            ) from exc

    @staticmethod
    def _validated_path(
        database_path: Path | str | None, automated_test: bool
    ) -> Path | None:
        if database_path is None or str(database_path) == ':memory:':
            if automated_test is not True:
                raise SQLiteRagSmokeUnavailable(
                    'in-memory SQLite is automated-test-only'
                )
            return None
        path = Path(database_path)
        absolute = path.absolute()
        try:
            if path.is_symlink() or absolute.is_symlink():
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke database path cannot be a symlink'
                )
            if path.exists() and absolute.resolve(strict=True) != absolute:
                raise SQLiteRagSmokeUnavailable(
                    'SQLite smoke database path identity is ambiguous'
                )
        except OSError as exc:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke database path is unavailable'
            ) from exc
        return absolute

    @staticmethod
    def _acquire_process_lock(database_path: Path) -> None:
        lock_path = database_path.with_name(database_path.name + '.rag-smoke-process.lock')
        if lock_path in _PROCESS_LOCKS:
            return
        try:
            stat = database_path.stat()
            identity = (stat.st_dev, stat.st_ino)
        except OSError as exc:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke database identity is unavailable'
            ) from exc
        existing_path = _DATABASE_IDENTITIES.get(identity)
        if existing_path is not None and existing_path != database_path:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke hardlink/path identity mismatch'
            )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            handle = os.fdopen(descriptor, 'a+b', buffering=0)
            if os.name == 'nt':
                import msvcrt

                if os.path.getsize(lock_path) == 0:
                    handle.write(b'0')
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ValueError) as exc:
            raise SQLiteRagSmokeUnavailable(
                'SQLite smoke process lock is unavailable'
            ) from exc
        _PROCESS_LOCKS[lock_path] = handle
        _DATABASE_IDENTITIES[identity] = database_path
