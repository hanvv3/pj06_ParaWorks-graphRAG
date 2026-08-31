from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import create_engine, func, select

from backend.app.agent_runtime.rag_advisory_locks import (
    LIVE_RELEASE_LOCK_ORDER,
    ORDINARY_RAG_LOCK_ORDER,
    RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID,
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    AdvisoryLockCollisionError,
    acquire_advisory_lock,
    advisory_int4_pair,
    register_advisory_identity,
    register_advisory_identity_db,
)
from backend.app.db.base import Base
from backend.app.models.rag_runtime import RagAdvisoryLockKey


def test_advisory_pair_is_exact_signed_int4_sha256_prefix():
    value = {
        'namespace': 'static',
        'identity': 'provider-safety-authority',
    }
    pair = advisory_int4_pair(value)
    digest = hashlib.sha256(
        b'{"domain":"paraworks:postgres-advisory-int4-pair:v1",'
        b'"policy_version":"rag-lock-order:v1","value":'
        b'{"identity":"provider-safety-authority","namespace":"static"}}'
    ).digest()
    assert pair == (
        int.from_bytes(digest[:4], 'big', signed=True),
        int.from_bytes(digest[4:8], 'big', signed=True),
    )


def test_collision_registry_is_append_only_and_never_salts():
    registry: dict[tuple[int, int], bytes] = {}
    canonical = register_advisory_identity(
        registry, {'namespace': 'dynamic', 'identity': 'run:17'}
    )
    assert (
        register_advisory_identity(
            registry, {'namespace': 'dynamic', 'identity': 'run:17'}
        )
        == canonical
    )
    pair = advisory_int4_pair({'namespace': 'dynamic', 'identity': 'run:17'})
    registry[pair] = b'other canonical identity'
    with pytest.raises(AdvisoryLockCollisionError):
        register_advisory_identity(
            registry, {'namespace': 'dynamic', 'identity': 'run:17'}
        )


def test_global_lock_orders_are_executable_exact_contracts():
    assert RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID == {
        'lock_name': 'provider_safety_authority',
        'scope': 'database',
    }
    assert RAG_EVIDENCE_PROVIDER_SEND_LOCK_ID == {
        'lock_name': 'evidence_provider_send',
        'scope': 'database',
    }
    assert ORDINARY_RAG_LOCK_ORDER == (
        'provider_stable_sidecar',
        'provider_safety_rows',
        'projection_owner',
        'evidence_shared_barrier',
        'c5_key_corpus',
        'agent_run_cost',
        'optional_assistant',
    )
    assert LIVE_RELEASE_LOCK_ORDER == (
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


def test_db_registration_never_commits_or_rolls_back_callers_transaction():
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    with engine.connect() as connection:
        transaction = connection.begin()
        capability = register_advisory_identity_db(
            connection,
            {'lock_name': 'provider_safety_authority', 'scope': 'database'},
            identity_namespace='static',
        )
        assert transaction.is_active
        transaction.rollback()
        assert (
            connection.scalar(select(func.count()).select_from(RagAdvisoryLockKey)) == 0
        )
        with pytest.raises(TypeError, match='registered advisory capability'):
            acquire_advisory_lock(connection, capability.key, shared=False)
