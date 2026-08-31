from __future__ import annotations

import hashlib
from collections.abc import MutableMapping

from sqlalchemy import Connection, insert, select

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.models.rag_runtime import RagAdvisoryLockKey


class AdvisoryLockCollisionError(RuntimeError):
    pass


ORDINARY_RAG_LOCK_ORDER = (
    'provider_stable_sidecar', 'provider_safety_rows', 'projection_owner',
    'evidence_shared_barrier', 'c5_key_corpus', 'agent_run_cost',
    'optional_assistant',
)
LIVE_RELEASE_LOCK_ORDER = (
    'provider_stable_sidecar', 'release_stable_sidecar_advisory',
    'provider_safety_rows', 'release_rows', 'projection_owner',
    'evidence_shared_barrier', 'c5_key_corpus', 'agent_run_cost',
    'optional_assistant',
)
RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID = {
    'lock_name': 'provider_safety_authority', 'scope': 'database'
}
RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID = {
    'lock_name': 'release_ledger_authority', 'scope': 'database'
}
RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID = {
    'lock_name': 'evidence_provider_send', 'scope': 'database'
}


def rag_projection_owner_lock_id(agent_run_id: int) -> dict[str, int | str]:
    if type(agent_run_id) is not int or agent_run_id <= 0:
        raise ValueError('agent run id must be a positive integer')
    return {'agent_run_id': agent_run_id, 'lock_name': 'projection_owner'}


def advisory_identity_bytes(value: object) -> bytes:
    return canonical_json_bytes({
        'domain': 'paraworks:postgres-advisory-int4-pair:v1',
        'policy_version': 'rag-lock-order:v1',
        'value': value,
    })


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
) -> tuple[int, int]:
    """Durably register a lock identity before any advisory acquisition."""
    if identity_namespace not in {'static', 'dynamic'}:
        raise ValueError('advisory identity namespace is invalid')
    canonical = advisory_identity_bytes(value)
    digest = hashlib.sha256(canonical).hexdigest()
    pair = advisory_int4_pair(value)
    existing = connection.execute(
        select(RagAdvisoryLockKey.__table__).where(
            RagAdvisoryLockKey.key1 == pair[0], RagAdvisoryLockKey.key2 == pair[1]
        )
    ).mappings().one_or_none()
    if existing is None:
        try:
            connection.execute(insert(RagAdvisoryLockKey).values(
                key1=pair[0], key2=pair[1],
                identity_namespace=identity_namespace,
                lock_identity_digest=digest,
                lock_identity_canonical_bytes=canonical,
            ))
            connection.commit()
        except Exception:
            connection.rollback()
            raise AdvisoryLockCollisionError(
                'advisory lock identity registration failed'
            ) from None
        return pair
    if (
        existing['identity_namespace'] != identity_namespace
        or existing['lock_identity_digest'] != digest
        or bytes(existing['lock_identity_canonical_bytes']) != canonical
    ):
        connection.rollback()
        raise AdvisoryLockCollisionError('advisory lock key collision')
    connection.rollback()
    return pair


def acquire_advisory_lock(connection: object, key: tuple[int, int], *, shared: bool) -> None:
    fn = 'pg_advisory_lock_shared' if shared else 'pg_advisory_lock'
    connection.exec_driver_sql(f'SELECT {fn}(%s, %s)', key)  # type: ignore[attr-defined]


def release_advisory_lock(connection: object, key: tuple[int, int], *, shared: bool) -> None:
    fn = 'pg_advisory_unlock_shared' if shared else 'pg_advisory_unlock'
    result = connection.exec_driver_sql(f'SELECT {fn}(%s, %s)', key)  # type: ignore[attr-defined]
    if result.scalar_one() is not True:
        try:
            connection.invalidate()  # type: ignore[attr-defined]
        finally:
            connection.close()  # type: ignore[attr-defined]
        raise RuntimeError('advisory unlock was not confirmed')
