from __future__ import annotations

import os
from threading import Barrier, Thread
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.rag_advisory_locks import (
    acquire_advisory_lock,
    load_registered_advisory_capability,
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
        assert register_advisory_identity_db(
            connection, identity, identity_namespace='dynamic'
        ) is None
        connection.commit()
        capability = load_registered_advisory_capability(
            connection, identity, identity_namespace='dynamic'
        )
        connection.rollback()
        assert register_advisory_identity_db(
            connection, identity, identity_namespace='dynamic'
        ) is None
        connection.commit()
        acquire_advisory_lock(connection, capability, shared=False)
        release_advisory_lock(connection, capability, shared=False)


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason='disposable PostgreSQL gate is not configured',
)
def test_concurrent_identical_registration_is_idempotent_and_postcommit_loadable():
    engine = create_engine(POSTGRES_URL)
    identity = {'lock_name': 'task12-concurrent-gated-test', 'nonce': str(uuid4())}
    barrier = Barrier(2)
    failures: list[BaseException] = []

    def register() -> None:
        try:
            with engine.connect() as connection:
                barrier.wait(timeout=5)
                register_advisory_identity_db(
                    connection, identity, identity_namespace='dynamic'
                )
                connection.commit()
        except BaseException as exc:
            failures.append(exc)

    workers = [Thread(target=register) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
    assert all(not worker.is_alive() for worker in workers)
    assert failures == []
    with engine.connect() as connection:
        capability = load_registered_advisory_capability(
            connection, identity, identity_namespace='dynamic'
        )
        assert capability.matches(identity, identity_namespace='dynamic')
    engine.dispose()
