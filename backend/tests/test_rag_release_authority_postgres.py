from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from threading import Barrier, Thread
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import DBAPIError

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
    load_registered_advisory_capability,
    register_advisory_identity_db,
)
from backend.app.models.rag_runtime import RagAdvisoryLockKey
from backend.app.rag.release_authority import (
    RagReleaseAuthority,
    RagReleaseAuthorityError,
)
from backend.app.rag.release_schema import (
    RAG_RELEASE_TABLE_NAMES,
    build_rag_release_metadata,
    release_tables,
)

_RUNTIME_KEY = b'task23-postgres-release-runtime-key-32-bytes'
_SAFE_DATABASE = re.compile(r'^rag_task23_[0-9a-f]{20}$', re.ASCII)


@pytest.fixture
def postgres_release_db() -> Iterator[Engine]:
    base_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not base_url:
        pytest.skip('PARAWORKS_TEST_POSTGRES_URL is unavailable for Task 23')
    parsed = make_url(base_url)
    if (
        parsed.get_backend_name() != 'postgresql'
        or parsed.host != '127.0.0.1'
        or parsed.port != 55432
        or not parsed.database
        or not parsed.database.endswith('_test')
    ):
        pytest.fail(
            'Task 23 PostgreSQL checks require disposable '
            '127.0.0.1:55432/*_test authority'
        )
    database_name = f'rag_task23_{uuid4().hex[:20]}'
    assert _SAFE_DATABASE.fullmatch(database_name)
    admin = create_engine(base_url, isolation_level='AUTOCOMMIT')
    quoted = admin.dialect.identifier_preparer.quote(database_name)
    with admin.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE {quoted}')
    target_url = parsed.set(database=database_name).render_as_string(
        hide_password=False
    )
    engine = create_engine(target_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            RagAdvisoryLockKey.__table__.create(connection)
            register_advisory_identity_db(
                connection,
                RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
                identity_namespace='static',
            )
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as connection:
            connection.execute(
                text(
                    'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
                    'WHERE datname=:name AND pid <> pg_backend_pid()'
                ),
                {'name': database_name},
            )
            connection.exec_driver_sql(f'DROP DATABASE {quoted}')
        admin.dispose()


def _authority(engine: Engine, tmp_path: Path, *, callback=None):
    provider = tmp_path / 'provider' / 'state.json'
    if not provider.exists():
        DurableFileAuthority(provider).write({'provider': 'test-only'})
    marker = tmp_path / 'release' / 'marker.json'
    with engine.connect() as connection:
        capability = load_registered_advisory_capability(
            connection,
            RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
    return RagReleaseAuthority(
        marker_path=marker,
        provider_safety_latch_path=provider,
        identity_secret=_RUNTIME_KEY,
        fingerprint_key_version='v1',
        designated_environment_id='task23-live-validation',
        designated_host_id='task23-host',
        repository_roots=(Path.cwd(),),
        database_backup_roots=(),
        advisory_capability=capability,
        after_marker_replace=callback,
    )


def _review(seed: str) -> dict[str, str]:
    return {
        'review_envelope_hmac': seed * 64,
        'review_nonce_hmac': format((int(seed, 16) + 1) % 16, 'x') * 64,
    }


def test_postgresql_creates_exact_six_in_default_schema_and_binds_oid(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.begin() as connection:
        snapshot = authority.initialize(connection, **_review('1'))
        current_oid = connection.scalar(
            text('SELECT oid FROM pg_database WHERE datname=current_database()')
        )
    with postgres_release_db.connect() as connection:
        table_names = set(inspect(connection).get_table_names(schema='public'))
        assert {
            name for name in table_names if name.startswith('rag_live_gate_')
        } == set(RAG_RELEASE_TABLE_NAMES)
        assert all(
            table.schema is None
            for table in build_rag_release_metadata().tables.values()
        )
        row = connection.execute(
            select(release_tables(build_rag_release_metadata()).ledgers)
        ).mappings().one()
        assert row['validation_database_oid'] == current_oid
        assert row['generation'] == snapshot.generation == 0

    with postgres_release_db.begin() as connection:
        table = release_tables(build_rag_release_metadata()).ledgers
        stale = connection.execute(
            table.update()
            .where(
                table.c.ledger_uuid == str(snapshot.ledger_uuid),
                table.c.ledger_epoch == snapshot.ledger_epoch,
                table.c.generation == 1,
            )
            .values(
                generation=2,
                last_transition_digest='e' * 64,
                marker_file_digest='f' * 64,
            )
        )
        assert stale.rowcount == 0

    with pytest.raises(DBAPIError), postgres_release_db.begin() as connection:
        table = release_tables(build_rag_release_metadata()).ledgers
        connection.execute(
            table.update().values(designated_host_id_hmac='f' * 64)
        )


def test_postgresql_advisory_serializes_concurrent_initialization(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    first = _authority(postgres_release_db, tmp_path)
    second = _authority(postgres_release_db, tmp_path)
    barrier = Barrier(3)
    outcomes: list[str] = []

    def initialize(authority: RagReleaseAuthority, seed: str) -> None:
        barrier.wait()
        try:
            with postgres_release_db.begin() as connection:
                authority.initialize(connection, **_review(seed))
            outcomes.append('committed')
        except RagReleaseAuthorityError:
            outcomes.append('refused')

    workers = [
        Thread(target=initialize, args=(first, '1')),
        Thread(target=initialize, args=(second, '3')),
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(15)
        assert not worker.is_alive()
    assert sorted(outcomes) == ['committed', 'refused']


def test_postgresql_marker_first_fault_is_not_repaired_by_init(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    def crash() -> None:
        raise RuntimeError('simulated marker-first crash')

    authority = _authority(postgres_release_db, tmp_path, callback=crash)
    with pytest.raises(
        RuntimeError, match='marker-first'
    ), postgres_release_db.begin() as connection:
        authority.initialize(connection, **_review('1'))
    assert authority.marker_path.exists()
    with postgres_release_db.connect() as connection:
        assert not (
            set(inspect(connection).get_table_names()) & set(RAG_RELEASE_TABLE_NAMES)
        )
    retry = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.begin() as connection, pytest.raises(
        RagReleaseAuthorityError
    ):
        retry.initialize(connection, **_review('3'))


def test_postgresql_rebootstrap_and_disaster_preserve_prior_rows(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.begin() as connection:
        first = authority.initialize(connection, **_review('1'))
    tables = release_tables(build_rag_release_metadata())
    with postgres_release_db.begin() as connection:
        connection.execute(
            tables.ledgers.update()
            .where(
                tables.ledgers.c.ledger_uuid == str(first.ledger_uuid),
                tables.ledgers.c.ledger_epoch == 1,
            )
            .values(generation=1, last_transition_digest='a' * 64)
        )
    with postgres_release_db.connect() as connection:
        prior_row = dict(
            connection.execute(
                select(tables.ledgers).where(
                    tables.ledgers.c.ledger_uuid == str(first.ledger_uuid),
                    tables.ledgers.c.ledger_epoch == 1,
                )
            ).mappings().one()
        )
    with postgres_release_db.begin() as connection:
        second = authority.rebootstrap(
            connection,
            rebootstrap_reason_hmac='b' * 64,
            **_review('3'),
        )
    assert second.ledger_uuid == first.ledger_uuid
    assert second.ledger_epoch == 2
    authority.marker_path.write_bytes(b'corrupt')
    disaster = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.begin() as connection:
        third = disaster.disaster_initialize(
            connection,
            rebootstrap_reason_hmac='c' * 64,
            **_review('5'),
        )
    assert third.ledger_uuid != first.ledger_uuid
    assert third.ledger_epoch == 1
    with postgres_release_db.connect() as connection:
        preserved = dict(
            connection.execute(
                select(tables.ledgers).where(
                    tables.ledgers.c.ledger_uuid == str(first.ledger_uuid),
                    tables.ledgers.c.ledger_epoch == 1,
                )
            ).mappings().one()
        )
        rows = connection.execute(
            select(
                tables.ledgers.c.ledger_uuid,
                tables.ledgers.c.ledger_epoch,
            ).order_by(tables.ledgers.c.ledger_uuid, tables.ledgers.c.ledger_epoch)
        ).all()
    assert preserved == prior_row
    assert len(rows) == 3
