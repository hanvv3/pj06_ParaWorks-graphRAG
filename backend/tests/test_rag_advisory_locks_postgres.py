from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.rag_advisory_locks import (
    acquire_advisory_lock,
    register_advisory_identity_db,
    release_advisory_lock,
)

POSTGRES_URL = os.environ.get('PARAWORKS_TEST_POSTGRES_URL')


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason='disposable PostgreSQL gate is not configured',
)
def test_two_argument_postgres_advisory_lock_and_durable_registry():
    engine = create_engine(POSTGRES_URL)
    identity = {'lock_name': 'task12-gated-test', 'nonce': str(uuid4())}
    with engine.connect() as connection:
        pair = register_advisory_identity_db(
            connection, identity, identity_namespace='dynamic'
        )
        assert register_advisory_identity_db(
            connection, identity, identity_namespace='dynamic'
        ) == pair
        acquire_advisory_lock(connection, pair, shared=False)
        release_advisory_lock(connection, pair, shared=False)
