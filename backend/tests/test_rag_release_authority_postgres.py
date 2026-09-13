from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterator
from pathlib import Path
from threading import Barrier, Thread
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
    load_registered_advisory_capability,
    register_advisory_identity_db,
)
from backend.app.models.rag_runtime import (
    RagAdvisoryLockKey,
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)
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
_PROVIDER_REVIEW_KEY = b'task23-provider-review-key-material-32-bytes'
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
            RagProviderSafetyAuthority.__table__.create(connection)
            RagProviderReadiness.__table__.create(connection)
            RagProviderSafetyTransition.__table__.create(connection)
            register_advisory_identity_db(
                connection,
                RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
                identity_namespace='static',
            )
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
    from backend.app.admin.rag_provider_safety import (
        ProviderSafetyAdminTarget,
        RagProviderSafetyAdminService,
        review_key_material_verifier,
    )
    from backend.app.agent_runtime.fingerprints import canonical_json_bytes
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyService
    from backend.tests.test_rag_v2_costs import _snapshot

    provider = tmp_path / 'provider' / 'state.json'
    marker = tmp_path / 'release' / 'marker.json'
    with engine.connect() as connection:
        capability = load_registered_advisory_capability(
            connection,
            RAG_RELEASE_LEDGER_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
        provider_capability = load_registered_advisory_capability(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
    target = ProviderSafetyAdminTarget.build(
        kind='live_validation',
        database_url=engine.url.render_as_string(hide_password=False),
        latch_path=provider,
        designated_environment_id='task23-live-validation',
        review_secret=_PROVIDER_REVIEW_KEY,
    )
    provider_service = RagProviderSafetyService(
        latch_path=provider,
        identity_secret=_RUNTIME_KEY,
        designated_environment_id='task23-live-validation',
        advisory_capability=provider_capability,
    )
    provider_admin = RagProviderSafetyAdminService(
        target=target,
        connection_factory=engine.connect,
        provider_safety=provider_service,
        snapshots=(_snapshot('query_embedding'), _snapshot('answer_generation')),
        runtime_identity_secret=_RUNTIME_KEY,
        review_secret=_PROVIDER_REVIEW_KEY,
        review_key_id='provider-safety-review-v1',
        implementation_plan_reference_hmac='9' * 64,
        successor_registry={},
        review_key_registry={
            'provider-safety-review-v1': review_key_material_verifier(
                _PROVIDER_REVIEW_KEY
            )
        },
    )
    if not provider.exists():
        signed = {
            'actor_subject_hmac': '4' * 64,
            'expected_context': None,
            'historical_block_acknowledged': False,
            'implementation_plan_reference_hmac': '9' * 64,
            'nonce': str(uuid4()),
            'operation': 'provider-safety-init',
            'review_authority_key_id': 'provider-safety-review-v1',
            'schema_version': 'rag-provider-safety-admin-review:v1',
            'successor': None,
            'target': target.review_identity,
        }
        signature = hmac.new(
            _PROVIDER_REVIEW_KEY,
            b'paraworks:provider-safety-admin-review:v1\x00'
            + canonical_json_bytes(signed),
            hashlib.sha256,
        ).hexdigest()
        provider_admin.initialize(
            canonical_json_bytes(
                {'hmac_sha256': signature, 'signed_payload': signed}
            )
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
        provider_safety_release_peer=provider_admin.release_peer(),
        after_marker_replace=callback,
    )


def _review(seed: str) -> dict[str, str]:
    return {
        'review_envelope_hmac': seed * 64,
        'review_nonce_hmac': format((int(seed, 16) + 1) % 16, 'x') * 64,
    }


def _recovery_review(seed: str, reason: str):
    review = _review(seed)
    return lambda _connection, _raw: (
        review['review_envelope_hmac'],
        review['review_nonce_hmac'],
        reason,
    )


def test_postgresql_creates_exact_six_in_default_schema_and_binds_oid(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection:
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
            with postgres_release_db.connect() as connection:
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
    ), postgres_release_db.connect() as connection:
        authority.initialize(connection, **_review('1'))
    assert authority.marker_path.exists()
    failed_snapshot = authority._parse(authority.marker_path.read_bytes())[1]
    with postgres_release_db.connect() as connection:
        assert not (
            set(inspect(connection).get_table_names()) & set(RAG_RELEASE_TABLE_NAMES)
        )
    retry = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection, pytest.raises(
        RagReleaseAuthorityError
    ):
        retry.initialize(connection, **_review('3'))
    with postgres_release_db.connect() as connection, pytest.raises(
        RagReleaseAuthorityError, match='nonce'
    ):
        retry.disaster_initialize(
            connection,
            review_verifier=_recovery_review('1', 'a' * 64),
        )
    with postgres_release_db.connect() as connection:
        recovered = retry.disaster_initialize(
            connection,
            review_verifier=_recovery_review('3', 'b' * 64),
        )
    assert recovered.bootstrap_operation == 'release-ledger-disaster-init'
    assert recovered.ledger_uuid != failed_snapshot.ledger_uuid


def test_postgresql_rebootstrap_preserves_and_same_db_disaster_refuses(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection:
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
    with postgres_release_db.connect() as connection:
        second = authority.rebootstrap(
            connection,
            review_verifier=_recovery_review('3', 'b' * 64),
        )
    assert second.ledger_uuid == first.ledger_uuid
    assert second.ledger_epoch == 2
    authority.marker_path.write_bytes(b'corrupt')
    disaster = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection, pytest.raises(
        RagReleaseAuthorityError, match='existing validation'
    ):
        disaster.disaster_initialize(
            connection,
            review_verifier=_recovery_review('5', 'c' * 64),
        )
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
    assert len(rows) == 2


def test_postgresql_claimed_case_concurrency_allows_exactly_one(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection:
        snapshot = authority.initialize(connection, **_review('1'))
    tables = release_tables(build_rag_release_metadata())
    approval_id = '8' * 64
    with postgres_release_db.begin() as connection:
        connection.execute(
            tables.authorizations.insert().values(
                ledger_uuid=str(snapshot.ledger_uuid),
                ledger_epoch=1,
                approval_id_hmac=approval_id,
                approval_hmac='9' * 64,
                base_generation=0,
                state='started',
                approved_corpus_snapshot_hmac='a' * 64,
                approved_provider_safety_snapshot_hmac='b' * 64,
                provider_safety_envelope_digest='c' * 64,
                validation_database_identity_hmac=(
                    snapshot.validation_database_identity_hmac
                ),
                manifest_hmac='d' * 64,
                baseline_hmac='e' * 64,
                reviewer_roster_hmac='f' * 64,
                execution_process_instance_hmac='1' * 64,
                execution_runner_fence_hmac='2' * 64,
                case_claim_count=0,
                embedding_dispatch_count=0,
                generation_dispatch_count=0,
                total_dispatch_count=0,
                reserved_cost_usd='0.000000',
                charged_cost_usd='0.000000',
            )
        )
    barrier = Barrier(3)
    outcomes: list[str] = []

    def claim(ordinal: int) -> None:
        barrier.wait()
        try:
            with postgres_release_db.begin() as connection:
                connection.execute(
                    tables.cases.insert().values(
                        ledger_uuid=str(snapshot.ledger_uuid),
                        ledger_epoch=1,
                        approval_id_hmac=approval_id,
                        case_id_hmac=format(ordinal + 3, 'x') * 64,
                        manifest_ordinal=ordinal,
                        case_projection_hmac=None,
                        state='claimed',
                        runtime_agent_run_id_hmac=format(ordinal + 5, 'x') * 64,
                        embedding_reserved_cost_usd='0.000000',
                        generation_reserved_cost_usd='0.000000',
                        total_reserved_cost_usd='0.000000',
                    )
                )
            outcomes.append('committed')
        except IntegrityError:
            outcomes.append('refused')

    workers = [Thread(target=claim, args=(ordinal,)) for ordinal in (0, 1)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(15)
        assert not worker.is_alive()
    assert sorted(outcomes) == ['committed', 'refused']


def test_postgresql_missing_release_trigger_fails_status_without_repair(
    postgres_release_db: Engine, tmp_path: Path
) -> None:
    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection:
        authority.initialize(connection, **_review('1'))
    with postgres_release_db.begin() as connection:
        connection.exec_driver_sql(
            'DROP TRIGGER rag_release_guard_transition '
            'ON rag_live_gate_transitions'
        )
    marker_before = authority.marker_path.read_bytes()
    with postgres_release_db.connect() as connection, pytest.raises(
        RagReleaseAuthorityError, match='physical schema'
    ):
        authority.disaster_initialize(
            connection,
            review_verifier=_recovery_review('3', 'b' * 64),
        )
    assert authority.marker_path.read_bytes() == marker_before
    with postgres_release_db.connect() as connection, pytest.raises(
        RagReleaseAuthorityError, match='physical schema'
    ):
        authority.inspect(connection)
    with postgres_release_db.connect() as connection:
        triggers = connection.scalar(
            text(
                "SELECT count(*) FROM pg_trigger WHERE tgname="
                "'rag_release_guard_transition'"
            )
        )
    assert triggers == 0
