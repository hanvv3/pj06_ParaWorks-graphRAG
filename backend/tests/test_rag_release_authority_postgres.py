from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterator
from decimal import Decimal
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
from backend.app.models.agent_runs import AgentRun
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


@pytest.mark.parametrize(
    'field,value',
    [
        ('permission_level', 'public'),
        ('generation_provider', 'forged'),
        ('metadata', {'forged': True}),
    ],
)
def test_postgresql_runtime_piggyback_is_zero_sql_and_marker_change(
    postgres_release_db, tmp_path, field, value
):
    from sqlalchemy import event

    from backend.app.rag.release_ledger import RagReleaseLedgerError
    from backend.tests.release_ledger_fixtures import ReleaseHarness
    from backend.tests.test_rag_release_ledger_round5 import failure_changes

    harness = ReleaseHarness(
        postgres_release_db,
        _authority(postgres_release_db, tmp_path),
        _RUNTIME_KEY,
        None,
    )
    harness.claim()
    changes, failed = failure_changes(harness)
    changes[1][2][field] = value
    marker = harness.authority.marker_path.read_bytes()
    before = harness.records('agent_run')
    writes = []

    def record(_conn, _cursor, statement, _params, _ctx, _many):
        if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            writes.append(statement)

    with postgres_release_db.connect() as connection:
        payload, mutations = harness.prepare(
            connection,
            'case_failure',
            changes,
            case=failed,
            outcome='model_unavailable',
        )
        event.listen(postgres_release_db, 'before_cursor_execute', record)
        try:
            with pytest.raises(RagReleaseLedgerError):
                harness.ledger.append(connection, payload, actual_mutations=mutations)
        finally:
            event.remove(postgres_release_db, 'before_cursor_execute', record)
    assert writes == []
    assert harness.records('agent_run') == before
    assert harness.authority.marker_path.read_bytes() == marker


def test_postgresql_release_generation_and_pending_failure_keep_sealed_cost_roster(
    postgres_release_db: Engine,
    tmp_path: Path,
) -> None:
    from backend.tests.release_ledger_fixtures import ReleaseHarness

    harness = ReleaseHarness(
        postgres_release_db,
        _authority(postgres_release_db, tmp_path),
        _RUNTIME_KEY,
        None,
    )
    harness.claim()
    harness.claim_generation()
    harness.generation_outcome()
    costs = harness.records('cost_component')
    payload = harness.fail_case()
    assert harness.records('cost_component') == costs
    assert harness.records('agent_run')[0]['total_charged_cost_usd'] == Decimal(
        '0.000400'
    )
    assert (
        sum(row['row_kind'] == 'cost_component' for row in payload['observation_set'])
        == 2
    )
    assert not any(
        row['row_kind'] == 'cost_component' for row in payload['affected_rows']
    )


def test_postgresql_current_provider_incident_allows_observed_snapshot_abort(
    postgres_release_db: Engine,
    tmp_path: Path,
) -> None:
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.tests.release_ledger_fixtures import ReleaseHarness

    authority = _authority(postgres_release_db, tmp_path)
    harness = ReleaseHarness(postgres_release_db, authority, _RUNTIME_KEY, None)
    harness.claim()
    harness.claim_generation()
    harness.generation_outcome()
    approved = harness.records('authorization')[0]['provider_safety_envelope_digest']
    costs = harness.records('cost_component')
    peer = authority._provider_safety_release_peer
    with postgres_release_db.connect() as connection:
        incident = peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_response_identity_invalid',
            agent_run_id=41,
            input_tokens=1,
            output_tokens=2,
            cost_usd=Decimal('0.000400'),
        )
    marker = DurableFileAuthority.open_runtime(authority.marker_path)
    with (
        postgres_release_db.begin() as connection,
        authority._authority_barrier(connection, marker=marker) as guard,
    ):
        guard.apply_provider_incident(incident)
    payload = harness.fail_case(
        'authorization_abort_control', 'provider_safety_unavailable'
    )
    assert payload['provider_safety_envelope_digest'] != approved
    assert (
        harness.records('authorization')[0]['provider_safety_envelope_digest']
        == approved
    )
    assert harness.records('cost_component') == costs


def test_postgresql_release_component_incident_binds_approved_before_and_blocked_after(
    postgres_release_db: Engine,
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from backend.app.rag.release_ledger import release_row_identity_hmac
    from backend.tests.release_ledger_fixtures import ReleaseHarness, row_key

    authority = _authority(postgres_release_db, tmp_path)
    harness = ReleaseHarness(postgres_release_db, authority, _RUNTIME_KEY, None)
    harness.claim()
    harness.claim_generation()
    auth, case, run = (
        harness.records('authorization')[0],
        harness.records('case')[0],
        harness.records('agent_run')[0],
    )
    child, dispatch = (
        harness.records('cost_component')[-1],
        harness.records('dispatch')[0],
    )
    peer = authority._provider_safety_release_peer
    with postgres_release_db.connect() as connection:
        incident = peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=41,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    failed = {**case, 'state': 'failed'}
    with postgres_release_db.connect() as connection:
        payload, mutations = harness.prepare(
            connection,
            'authorization_abort_component',
            [
                (
                    'authorization',
                    auth,
                    {
                        **auth,
                        'state': 'aborted_overrun',
                        'charged_cost_usd': Decimal('0.100000'),
                    },
                ),
                ('case', case, failed),
                (
                    'agent_run',
                    run,
                    {
                        **run,
                        'status': 'failed',
                        'run_record_phase': 'final',
                        'completed_at': datetime.now(UTC),
                        'total_charged_cost_usd': Decimal('0.100000'),
                    },
                ),
                (
                    'cost_component',
                    child,
                    {
                        **child,
                        'dispatch_state': 'terminal',
                        'charge_basis': 'actual',
                        'charged_cost_usd': Decimal('0.100000'),
                        'actual_input_tokens': 20,
                        'actual_output_tokens': 30,
                        'overrun': True,
                        'terminal_outcome': 'provider_usage_overrun',
                    },
                ),
                (
                    'dispatch',
                    dispatch,
                    {
                        **dispatch,
                        'state': 'terminal',
                        'charge_basis': 'actual',
                        'charged_cost_usd': Decimal('0.100000'),
                    },
                ),
            ],
            case=failed,
            component='answer_generation',
            outcome='provider_usage_overrun',
        )
        provider_keys = [
            row_key(
                'provider_safety_authority',
                harness.records('provider_safety_authority')[0],
            ),
            row_key(
                'provider_readiness',
                next(
                    row
                    for row in harness.records('provider_readiness')
                    if row['component'] == 'answer_generation' and row['active']
                ),
            ),
        ]
        for key in provider_keys:
            identity = release_row_identity_hmac(
                key.row_kind, key.primary_key, identity_secret=_RUNTIME_KEY
            )
            mutations._observation_rows.remove(key)
            payload['observation_set'] = [
                row
                for row in payload['observation_set']
                if (row['row_kind'], row['row_identity_hmac'])
                != (key.row_kind, identity)
            ]
            payload['affected_rows'].append(
                {'row_kind': key.row_kind, 'row_identity_hmac': identity}
            )
        payload['affected_rows'].sort(
            key=lambda item: (item['row_kind'], item['row_identity_hmac'])
        )
        payload['provider_safety_envelope_digest'] = incident.new_envelope_digest
        advanced = harness.ledger.append(
            connection, payload, actual_mutations=mutations, provider_incident=incident
        )
    assert advanced.generation == 4
    assert (
        harness.records('authorization')[0]['provider_safety_envelope_digest']
        == auth['provider_safety_envelope_digest']
    )
    assert (
        harness.records('provider_safety_authority')[0]['envelope_digest']
        == incident.new_envelope_digest
    )
    assert harness.records('cost_component')[-1]['charged_cost_usd'] == Decimal(
        '0.100000'
    )


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
            AgentRun.__table__.create(connection)
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
            canonical_json_bytes({'hmac_sha256': signature, 'signed_payload': signed})
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


def _insert_incident_parent(connection, run_id: int = 41) -> None:
    connection.execute(
        AgentRun.__table__.insert().values(
            id=run_id,
            agent_name='rag_orchestrator',
            prompt_version='rag-answer:v2',
            status='running',
            source_window='task23-provider-incident',
            cache_key=f'task23-provider-incident-{run_id}',
            model_name='gpt-5.4-mini-2026-03-17',
            generation_provider='openai',
            generation_reasoning_effort='low',
            generation_route_version='rag-route:v2',
            generation_output_contract_version='rag-answer:v2',
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            permission_level='internal',
            metadata={},
            workflow_thread_id=f'task23-provider-incident-{run_id}',
            effect_key=f'task23-provider-incident-{run_id}',
            run_contract_version='rag-run:v2',
            run_record_phase='admission',
            total_charged_cost_usd=Decimal('0.000000'),
            projection_owner_fence_hmac=None,
            completed_at=None,
        )
    )


def test_postgresql_release_barrier_applies_external_first_incident_once(
    postgres_release_db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.admin.rag_provider_safety import ProviderSafetyReviewError
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority

    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection:
        authority.initialize(connection, **_review('1'))
    peer = authority._provider_safety_release_peer
    service = peer._provider_safety
    with postgres_release_db.begin() as connection:
        _insert_incident_parent(connection)
    with postgres_release_db.connect() as connection:
        plan = peer.prepare_incident(
            connection,
            component='query_embedding',
            category='provider_usage_overrun',
            agent_run_id=41,
            input_tokens=7,
            output_tokens=0,
            cost_usd=Decimal('0.100000'),
        )
    observed_generation: list[int] = []
    original_replace = service._authority._replace_unlocked

    def observe_external_first(envelope):
        original_replace(envelope)
        with postgres_release_db.connect() as observer:
            observed_generation.append(
                int(
                    observer.scalar(
                        select(RagProviderSafetyAuthority.global_safety_generation)
                    )
                )
            )

    monkeypatch.setattr(service._authority, '_replace_unlocked', observe_external_first)
    marker = DurableFileAuthority.open_runtime(authority.marker_path)
    with (
        postgres_release_db.begin() as connection,
        authority._authority_barrier(connection, marker=marker) as guard,
    ):
        evidence = guard.apply_provider_incident(plan)
        assert evidence.new_body['envelope_digest'] == plan.new_envelope_digest
        with pytest.raises(ProviderSafetyReviewError, match='incident plan'):
            guard.apply_provider_incident(plan)
    assert observed_generation == [0]
    with postgres_release_db.connect() as connection:
        authority_row = (
            connection.execute(select(RagProviderSafetyAuthority.__table__))
            .mappings()
            .one()
        )
        readiness = (
            connection.execute(
                select(RagProviderReadiness.__table__).where(
                    RagProviderReadiness.component == 'query_embedding',
                    RagProviderReadiness.active.is_(True),
                )
            )
            .mappings()
            .one()
        )
        assert authority_row['global_safety_generation'] == 1
        assert authority_row['envelope_digest'] == plan.new_envelope_digest
        assert readiness['state'] == 'blocked_overrun'
        assert readiness['state_version'] == 2
        assert readiness['family_safety_generation'] == 1


def test_postgresql_incident_db_failure_leaves_external_mismatch_fail_stopped(
    postgres_release_db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    authority = _authority(postgres_release_db, tmp_path)
    with postgres_release_db.connect() as connection:
        authority.initialize(connection, **_review('1'))
    peer = authority._provider_safety_release_peer
    service = peer._provider_safety
    with postgres_release_db.begin() as connection:
        _insert_incident_parent(connection)
    with postgres_release_db.connect() as connection:
        plan = peer.prepare_incident(
            connection,
            component='query_embedding',
            category='provider_safety_unavailable',
            agent_run_id=41,
            input_tokens=0,
            output_tokens=0,
            cost_usd=Decimal('0.000000'),
        )
    monkeypatch.setattr(
        service,
        '_commit_transition',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RagProviderSafetyError('simulated provider DB failure')
        ),
    )
    marker = DurableFileAuthority.open_runtime(authority.marker_path)
    with (
        postgres_release_db.connect() as connection,
        pytest.raises(RagReleaseAuthorityError, match='provider safety authority'),
        authority._authority_barrier(connection, marker=marker) as guard,
    ):
        guard.apply_provider_incident(plan)
    assert service._read_unlocked()['envelope_digest'] == plan.new_envelope_digest
    with postgres_release_db.connect() as connection:
        assert (
            connection.scalar(
                select(RagProviderSafetyAuthority.global_safety_generation)
            )
            == 0
        )
        with pytest.raises(RagReleaseAuthorityError, match='provider safety authority'):
            authority.inspect(connection)


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
        row = (
            connection.execute(
                select(release_tables(build_rag_release_metadata()).ledgers)
            )
            .mappings()
            .one()
        )
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
        connection.execute(table.update().values(designated_host_id_hmac='f' * 64))


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
    with (
        pytest.raises(RuntimeError, match='marker-first'),
        postgres_release_db.connect() as connection,
    ):
        authority.initialize(connection, **_review('1'))
    assert authority.marker_path.exists()
    failed_snapshot = authority._parse(authority.marker_path.read_bytes())[1]
    with postgres_release_db.connect() as connection:
        assert not (
            set(inspect(connection).get_table_names()) & set(RAG_RELEASE_TABLE_NAMES)
        )
    retry = _authority(postgres_release_db, tmp_path)
    with (
        postgres_release_db.connect() as connection,
        pytest.raises(RagReleaseAuthorityError),
    ):
        retry.initialize(connection, **_review('3'))
    with (
        postgres_release_db.connect() as connection,
        pytest.raises(RagReleaseAuthorityError, match='nonce'),
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
            )
            .mappings()
            .one()
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
    with (
        postgres_release_db.connect() as connection,
        pytest.raises(RagReleaseAuthorityError, match='existing validation'),
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
            )
            .mappings()
            .one()
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
            'DROP TRIGGER rag_release_guard_transition ON rag_live_gate_transitions'
        )
    marker_before = authority.marker_path.read_bytes()
    with (
        postgres_release_db.connect() as connection,
        pytest.raises(RagReleaseAuthorityError, match='physical schema'),
    ):
        authority.disaster_initialize(
            connection,
            review_verifier=_recovery_review('3', 'b' * 64),
        )
    assert authority.marker_path.read_bytes() == marker_before
    with (
        postgres_release_db.connect() as connection,
        pytest.raises(RagReleaseAuthorityError, match='physical schema'),
    ):
        authority.inspect(connection)
    with postgres_release_db.connect() as connection:
        triggers = connection.scalar(
            text(
                'SELECT count(*) FROM pg_trigger WHERE tgname='
                "'rag_release_guard_transition'"
            )
        )
    assert triggers == 0
