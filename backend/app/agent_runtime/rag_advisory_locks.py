from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import Connection, insert, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.models.rag_runtime import RagAdvisoryLockKey


class AdvisoryLockCollisionError(RuntimeError):
    pass


class RagLockOrderError(RuntimeError):
    """A required RAG authority was acquired outside the frozen order."""


_CAPABILITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class RegisteredAdvisoryLock:
    """Opaque proof that an exact advisory identity was checked in the DB registry."""

    key: tuple[int, int]
    identity_namespace: str
    identity_digest: str
    canonical_identity: bytes = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def _require_authentic(self) -> None:
        if self._seal is not _CAPABILITY_SEAL:
            raise TypeError('registered advisory capability is required')

    def matches(self, value: object, *, identity_namespace: str) -> bool:
        self._require_authentic()
        canonical = advisory_identity_bytes(value)
        return (
            self.identity_namespace == identity_namespace
            and self.identity_digest == hashlib.sha256(canonical).hexdigest()
            and self.canonical_identity == canonical
            and self.key == advisory_int4_pair(value)
        )


ORDINARY_RAG_LOCK_ORDER = (
    'provider_stable_sidecar',
    'provider_safety_rows',
    'projection_owner',
    'evidence_shared_barrier',
    'c5_key_corpus',
    'agent_run_cost',
    'optional_assistant',
)
LIVE_RELEASE_LOCK_ORDER = (
    'provider_stable_sidecar',
    'release_stable_sidecar_advisory',
    'provider_safety_rows',
    'release_rows',
    'projection_owner',
    'evidence_shared_barrier',
    'c5_key_corpus',
    'agent_run_cost',
    'optional_assistant',
)
_ORDER_CAPABILITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class RagLockOrderCapability:
    path: Literal['ordinary', 'live_release']
    stage: str
    ordinal: int
    _coordinator: object = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


class RagLockOrderCoordinator:
    """Issues non-forgeable stage capabilities in the one audited order."""

    __slots__ = ('_next', '_order', '_path')

    def __init__(
        self,
        path: Literal['ordinary', 'live_release'],
        order: tuple[str, ...],
    ) -> None:
        self._path = path
        self._order = order
        self._next = 0

    def acquire(self, stage: str) -> RagLockOrderCapability:
        if self._next >= len(self._order):
            raise RagLockOrderError('RAG lock order is already complete')
        expected = self._order[self._next]
        if stage != expected:
            raise RagLockOrderError(
                f'RAG lock order expected {expected}, received {stage}'
            )
        capability = RagLockOrderCapability(
            path=self._path,
            stage=stage,
            ordinal=self._next,
            _coordinator=self,
            _seal=_ORDER_CAPABILITY_SEAL,
        )
        self._next += 1
        return capability

    def require(self, capability: object, *, stage: str) -> None:
        if (
            type(capability) is not RagLockOrderCapability
            or capability._seal is not _ORDER_CAPABILITY_SEAL
            or capability._coordinator is not self
            or capability.path != self._path
            or capability.stage != stage
            or capability.ordinal >= self._next
        ):
            raise RagLockOrderError('RAG lock-order capability is invalid')

    def finish(self) -> None:
        if self._next != len(self._order):
            raise RagLockOrderError('RAG lock order is incomplete')


def begin_rag_lock_order(
    path: Literal['ordinary', 'live_release'],
) -> RagLockOrderCoordinator:
    if path == 'ordinary':
        return RagLockOrderCoordinator(path, ORDINARY_RAG_LOCK_ORDER)
    if path == 'live_release':
        return RagLockOrderCoordinator(path, LIVE_RELEASE_LOCK_ORDER)
    raise ValueError('RAG lock-order path is invalid')
RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID = {
    'lock_name': 'provider_safety_authority',
    'scope': 'database',
}
RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID = {
    'lock_name': 'release_ledger_authority',
    'scope': 'database',
}
RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID = {
    'lock_name': 'evidence_provider_send',
    'scope': 'database',
}
RAG_PROJECTION_OWNER_REGISTRY_LOCK_ID = {
    'lock_name': 'projection_owner_registry',
    'scope': 'database',
}
RAG_C5_KEY_CORPUS_AUTHORITY_LOCK_ID = {
    'lock_name': 'c5_key_corpus_authority',
    'scope': 'database',
}
RAG_AGENT_RUN_COST_AUTHORITY_LOCK_ID = {
    'lock_name': 'agent_run_cost_authority',
    'scope': 'database',
}


def rag_projection_owner_lock_id(agent_run_id: int) -> dict[str, int | str]:
    if type(agent_run_id) is not int or agent_run_id <= 0:
        raise ValueError('agent run id must be a positive integer')
    return {'agent_run_id': agent_run_id, 'lock_name': 'projection_owner'}


def advisory_identity_bytes(value: object) -> bytes:
    return canonical_json_bytes(
        {
            'domain': 'paraworks:postgres-advisory-int4-pair:v1',
            'policy_version': 'rag-lock-order:v1',
            'value': value,
        }
    )


def advisory_int4_pair(value: object) -> tuple[int, int]:
    digest = hashlib.sha256(advisory_identity_bytes(value)).digest()
    return (
        int.from_bytes(digest[:4], 'big', signed=True),
        int.from_bytes(digest[4:8], 'big', signed=True),
    )


def register_advisory_identity(
    registry: MutableMapping[tuple[int, int], bytes], value: object
) -> bytes:
    canonical = advisory_identity_bytes(value)
    pair = advisory_int4_pair(value)
    existing = registry.get(pair)
    if existing is not None and existing != canonical:
        raise AdvisoryLockCollisionError('advisory lock key collision')
    registry[pair] = canonical
    return canonical


def register_advisory_identity_db(
    connection: Connection, value: object, *, identity_namespace: str
) -> None:
    """Durably register a lock identity before any advisory acquisition."""
    if identity_namespace not in {'static', 'dynamic'}:
        raise ValueError('advisory identity namespace is invalid')
    canonical = advisory_identity_bytes(value)
    digest = hashlib.sha256(canonical).hexdigest()
    pair = advisory_int4_pair(value)
    existing = (
        connection.execute(
            select(RagAdvisoryLockKey.__table__).where(
                RagAdvisoryLockKey.key1 == pair[0], RagAdvisoryLockKey.key2 == pair[1]
            )
        )
        .mappings()
        .one_or_none()
    )
    if existing is None:
        values = {
            'key1': pair[0],
            'key2': pair[1],
            'identity_namespace': identity_namespace,
            'lock_identity_digest': digest,
            'lock_identity_canonical_bytes': canonical,
        }
        if connection.dialect.name == 'postgresql':
            statement = postgresql_insert(RagAdvisoryLockKey).values(**values)
            statement = statement.on_conflict_do_nothing(
                index_elements=['key1', 'key2']
            )
        elif connection.dialect.name == 'sqlite':
            statement = sqlite_insert(RagAdvisoryLockKey).values(**values)
            statement = statement.on_conflict_do_nothing(
                index_elements=['key1', 'key2']
            )
        else:
            statement = insert(RagAdvisoryLockKey).values(**values)
        result = connection.execute(statement)
        if result.rowcount == 1:
            return None
        existing = (
            connection.execute(
                select(RagAdvisoryLockKey.__table__).where(
                    RagAdvisoryLockKey.key1 == pair[0],
                    RagAdvisoryLockKey.key2 == pair[1],
                )
            )
            .mappings()
            .one_or_none()
        )
    if (
        existing['identity_namespace'] != identity_namespace
        or existing['lock_identity_digest'] != digest
        or bytes(existing['lock_identity_canonical_bytes']) != canonical
    ):
        raise AdvisoryLockCollisionError('advisory lock key collision')
    return None


def load_registered_advisory_capability(
    connection: Connection, value: object, *, identity_namespace: str
) -> RegisteredAdvisoryLock:
    """Load a capability only from a fresh post-commit registry read."""
    if connection.in_transaction():
        raise AdvisoryLockCollisionError(
            'registered advisory capability requires a committed read'
        )
    if identity_namespace not in {'static', 'dynamic'}:
        raise ValueError('advisory identity namespace is invalid')
    canonical = advisory_identity_bytes(value)
    digest = hashlib.sha256(canonical).hexdigest()
    pair = advisory_int4_pair(value)
    existing = (
        connection.execute(
            select(RagAdvisoryLockKey.__table__).where(
                RagAdvisoryLockKey.key1 == pair[0],
                RagAdvisoryLockKey.key2 == pair[1],
            )
        )
        .mappings()
        .one_or_none()
    )
    if (
        existing is None
        or existing['identity_namespace'] != identity_namespace
        or existing['lock_identity_digest'] != digest
        or bytes(existing['lock_identity_canonical_bytes']) != canonical
    ):
        raise AdvisoryLockCollisionError(
            'advisory identity is not committed in this database'
        )
    return RegisteredAdvisoryLock(
        key=pair,
        identity_namespace=identity_namespace,
        identity_digest=digest,
        canonical_identity=canonical,
        _seal=_CAPABILITY_SEAL,
    )


def _require_registered_capability(
    connection: object, capability: RegisteredAdvisoryLock
) -> tuple[int, int]:
    if type(capability) is not RegisteredAdvisoryLock:
        raise TypeError('registered advisory capability is required')
    capability._require_authentic()
    row = (
        connection.execute(  # type: ignore[attr-defined]
            select(RagAdvisoryLockKey.__table__).where(
                RagAdvisoryLockKey.key1 == capability.key[0],
                RagAdvisoryLockKey.key2 == capability.key[1],
            )
        )
        .mappings()
        .one_or_none()
    )
    if (
        row is None
        or row['identity_namespace'] != capability.identity_namespace
        or row['lock_identity_digest'] != capability.identity_digest
        or bytes(row['lock_identity_canonical_bytes']) != capability.canonical_identity
    ):
        raise AdvisoryLockCollisionError(
            'advisory capability is not registered in this database'
        )
    return capability.key


def acquire_advisory_lock(
    connection: object, capability: RegisteredAdvisoryLock, *, shared: bool
) -> None:
    key = _require_registered_capability(connection, capability)
    fn = 'pg_advisory_lock_shared' if shared else 'pg_advisory_lock'
    connection.exec_driver_sql(f'SELECT {fn}(%s, %s)', key)  # type: ignore[attr-defined]


def try_acquire_advisory_lock(
    connection: object,
    capability: RegisteredAdvisoryLock,
    *,
    shared: bool,
) -> bool:
    """Nonblocking session-lock proof used by bounded owner recovery only."""
    key = _require_registered_capability(connection, capability)
    fn = 'pg_try_advisory_lock_shared' if shared else 'pg_try_advisory_lock'
    result = connection.exec_driver_sql(  # type: ignore[attr-defined]
        f'SELECT {fn}(%s, %s)', key
    )
    return result.scalar_one() is True


def release_advisory_lock(
    connection: object, capability: RegisteredAdvisoryLock, *, shared: bool
) -> None:
    if type(capability) is not RegisteredAdvisoryLock:
        raise TypeError('registered advisory capability is required')
    capability._require_authentic()
    key = capability.key
    fn = 'pg_advisory_unlock_shared' if shared else 'pg_advisory_unlock'
    result = connection.exec_driver_sql(f'SELECT {fn}(%s, %s)', key)  # type: ignore[attr-defined]
    if result.scalar_one() is not True:
        try:
            connection.invalidate()  # type: ignore[attr-defined]
        finally:
            connection.close()  # type: ignore[attr-defined]
        raise RuntimeError('advisory unlock was not confirmed')
