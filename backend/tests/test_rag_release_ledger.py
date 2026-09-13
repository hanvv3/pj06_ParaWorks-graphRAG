from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables

_SECRET = b'task23-release-runtime-key-material-32-bytes'
_REVIEW = {'review_envelope_hmac': '1' * 64, 'review_nonce_hmac': '2' * 64}


class _TestProviderPeer:
    def _assert_active_guard(self, guard, connection):
        assert guard is self

    def revalidate_database_peer(self, connection):
        pass


@pytest.fixture(autouse=True)
def _deterministic_non_product_database_seam(monkeypatch) -> None:
    from backend.app.admin import rag_provider_safety
    from backend.app.rag import release_authority

    monkeypatch.setattr(
        rag_provider_safety, 'RagProviderSafetyReleasePeer', _TestProviderPeer
    )
    monkeypatch.setattr(
        release_authority,
        'assert_rag_release_physical_contract',
        lambda _connection: None,
    )

    @contextmanager
    def barrier(_self, _connection, *, marker):
        with marker.locked():
            yield _self._provider_safety_release_peer

    monkeypatch.setattr(
        release_authority.RagReleaseAuthority, '_authority_transport', barrier
    )
    monkeypatch.setattr(
        release_authority.RagReleaseAuthority,
        '_current_database_identity',
        staticmethod(lambda _connection, supplied: supplied),
    )


def _authority(tmp_path: Path):
    from backend.app.rag.release_authority import RagReleaseAuthority

    provider = tmp_path / 'provider' / 'state.json'
    DurableFileAuthority(provider).write({'provider': 'test-only'})
    return RagReleaseAuthority(
        marker_path=tmp_path / 'release' / 'marker.json',
        provider_safety_latch_path=provider,
        identity_secret=_SECRET,
        fingerprint_key_version='v1',
        designated_environment_id='release-validation',
        designated_host_id='host-one',
        repository_roots=(Path.cwd(),),
        database_backup_roots=(),
        provider_safety_release_peer=_TestProviderPeer(),
    )


def _identity():
    from backend.app.rag.release_authority import ValidationDatabaseIdentity

    return ValidationDatabaseIdentity('validation', 53)


def _capture_bootstrap_authorization(connection, ledger, payload):
    from backend.app.rag.release_ledger import ReleaseRowPrimaryKey

    tables = release_tables(build_rag_release_metadata())
    row = ReleaseRowPrimaryKey(
        'authorization',
        {
            'ledger_uuid': payload['ledger_uuid'],
            'ledger_epoch': payload['ledger_epoch'],
            'approval_id_hmac': payload['approval_id_hmac'],
        },
    )
    mutations = ledger.mutation_set(connection)
    from backend.tests.release_ledger_fixtures import observe_provider_fixture

    observe_provider_fixture(connection, mutations)
    mutations.plan(
        insert(tables.authorizations).values(
            ledger_uuid=payload['ledger_uuid'],
            ledger_epoch=payload['ledger_epoch'],
            approval_id_hmac=payload['approval_id_hmac'],
            approval_hmac=payload['approval_hmac'],
            base_generation=payload['from_generation'],
            state='unused',
            approved_corpus_snapshot_hmac=payload['approved_corpus_snapshot_hmac'],
            approved_provider_safety_snapshot_hmac=(
                payload['approved_provider_safety_snapshot_hmac']
            ),
            provider_safety_envelope_digest=(
                payload['provider_safety_envelope_digest']
            ),
            validation_database_identity_hmac=(
                payload['validation_database_identity_hmac']
            ),
            manifest_hmac='a' * 64,
            baseline_hmac='b' * 64,
            reviewer_roster_hmac='c' * 64,
            execution_process_instance_hmac=None,
            execution_runner_fence_hmac=None,
            case_claim_count=0,
            embedding_dispatch_count=0,
            generation_dispatch_count=0,
            total_dispatch_count=0,
            reserved_cost_usd='0.000000',
            charged_cost_usd='0.000000',
        ),
        row,
    )
    return mutations


def _bootstrap_payload(snapshot) -> dict[str, object]:
    from backend.app.rag.release_ledger import release_row_identity_hmac

    payload = {
        'affected_rows': [],
        'approval_hmac': '4' * 64,
        'approval_id_hmac': '5' * 64,
        'approved_corpus_snapshot_hmac': '6' * 64,
        'approved_provider_safety_snapshot_hmac': '7' * 64,
        'authorization_charged_cost_usd': '0.000000',
        'authorization_reserved_cost_usd': '0.000000',
        'authorization_state_after': 'unused',
        'authorization_state_before': None,
        'case_claim_count': 0,
        'case_embedding_reserved_cost_usd': None,
        'case_generation_reserved_cost_usd': None,
        'case_id_hmac': None,
        'case_projection_hmac': None,
        'case_state_after': None,
        'case_state_before': None,
        'case_total_reserved_cost_usd': None,
        'charge_basis_after': None,
        'charged_cost_usd': None,
        'component': None,
        'current_corpus_snapshot_hmac': '6' * 64,
        'dispatch_count_after': None,
        'dispatch_count_before': None,
        'dispatch_fence_hmac': None,
        'dispatch_state_after': None,
        'dispatch_state_before': None,
        'execution_crash_attestation_hmac': None,
        'execution_process_instance_hmac': None,
        'execution_runner_fence_hmac': None,
        'embedding_dispatch_count': 0,
        'from_generation': 0,
        'generation_dispatch_count': 0,
        'ledger_epoch': snapshot.ledger_epoch,
        'ledger_uuid': str(snapshot.ledger_uuid),
        'observation_set': [],
        'outcome': None,
        'provider_safety_envelope_digest': '8' * 64,
        'quality_report_hmac': None,
        'reserved_cost_usd': None,
        'runtime_agent_run_id_hmac': None,
        'to_generation': 1,
        'total_dispatch_count': 0,
        'transition_kind': 'authorization_bootstrap',
        'validation_database_identity_hmac': snapshot.validation_database_identity_hmac,
    }
    row_keys = {
        'authorization': {
            'ledger_uuid': str(snapshot.ledger_uuid),
            'ledger_epoch': snapshot.ledger_epoch,
            'approval_id_hmac': payload['approval_id_hmac'],
        },
        'release_ledger': {
            'ledger_uuid': str(snapshot.ledger_uuid),
            'ledger_epoch': snapshot.ledger_epoch,
        },
        'release_transition': {
            'ledger_uuid': str(snapshot.ledger_uuid),
            'ledger_epoch': snapshot.ledger_epoch,
            'to_generation': 1,
        },
    }
    payload['affected_rows'] = sorted(
        (
            {
                'row_identity_hmac': release_row_identity_hmac(
                    kind, primary_key, identity_secret=_SECRET
                ),
                'row_kind': kind,
            }
            for kind, primary_key in row_keys.items()
        ),
        key=lambda item: (item['row_kind'], item['row_identity_hmac']),
    )
    from backend.tests.release_ledger_fixtures import provider_observations

    payload['observation_set'] = provider_observations(_SECRET)
    return payload


def test_transition_validator_accepts_exact_bootstrap_and_is_deterministic(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_ledger import validate_transition_payload

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    first = validate_transition_payload(payload, identity_secret=_SECRET)
    second = validate_transition_payload(deepcopy(payload), identity_secret=_SECRET)
    assert first.payload_canonical_bytes == second.payload_canonical_bytes
    assert first.transition_digest == second.transition_digest
    assert len(first.transition_digest) == 64


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda value: value.update(extra='forbidden'), 'keys'),
        (lambda value: value.update(to_generation=2), 'generation'),
        (lambda value: value.update(case_claim_count=True), 'count'),
        (
            lambda value: value.update(authorization_reserved_cost_usd='0E-6'),
            'six-place',
        ),
        (lambda value: value.update(outcome='quality_gate_green'), 'outcome'),
        (lambda value: value.update(case_id_hmac='9' * 64), 'affected'),
        (
            lambda value: value['affected_rows'].reverse(),
            'affected',
        ),
    ],
)
def test_transition_validator_rejects_noncanonical_or_wrong_matrix(
    tmp_path: Path, mutation, message: str
) -> None:
    from backend.app.rag.release_ledger import (
        RagReleaseLedgerError,
        validate_transition_payload,
    )

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    mutation(payload)
    with pytest.raises(RagReleaseLedgerError, match=message):
        validate_transition_payload(payload, identity_secret=_SECRET)


@pytest.mark.parametrize(
    ('kind', 'outcome'),
    [
        ('case_claim', None),
        ('component_claim', None),
        ('authorization_complete', 'quality_gate_green'),
        ('authorization_abort_execution_crash', 'abandoned_unknown'),
    ],
)
def test_transition_validator_rejects_cross_kind_null_and_state_matrix(
    tmp_path: Path,
    kind: str,
    outcome: str | None,
) -> None:
    from backend.app.rag.release_ledger import (
        RagReleaseLedgerError,
        validate_transition_payload,
    )

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    payload['transition_kind'] = kind
    payload['outcome'] = outcome
    with pytest.raises(RagReleaseLedgerError, match='matrix'):
        validate_transition_payload(payload, identity_secret=_SECRET)


def test_append_transition_is_gapless_cas_and_exact_affected_set(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        RagReleaseLedgerError,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    with engine.begin() as connection:
        mutations = _capture_bootstrap_authorization(connection, ledger, payload)
        advanced = ledger.append(
            connection,
            payload,
            actual_mutations=mutations,
            database_identity=_identity(),
        )
    assert advanced.generation == 1
    assert advanced.last_transition_digest is not None
    tables = release_tables(build_rag_release_metadata())
    with engine.connect() as connection:
        assert (
            connection.scalar(select(func.count()).select_from(tables.transitions)) == 1
        )
        row = connection.execute(select(tables.transitions)).mappings().one()
        assert row['generation'] == 1
        assert bytes(row['payload_canonical_bytes'])

    with (
        engine.begin() as connection,
        pytest.raises(RagReleaseLedgerError, match='generation'),
    ):
        ledger.append(
            connection,
            payload,
            actual_mutations=ledger.mutation_set(connection),
            database_identity=_identity(),
        )
    next_payload = deepcopy(payload)
    next_payload['from_generation'] = 1
    next_payload['to_generation'] = 2
    next_payload['affected_rows'][-1]['row_identity_hmac'] = 'f' * 64
    with (
        engine.begin() as connection,
        pytest.raises(RagReleaseLedgerError, match='identity'),
    ):
        ledger.append(
            connection, next_payload, actual_mutations=ledger.mutation_set(connection)
        )


def test_runtime_links_are_hmac_only_and_not_foreign_keys() -> None:
    tables = release_tables(build_rag_release_metadata())
    assert 'runtime_agent_run_id_hmac' in tables.cases.c
    assert 'runtime_agent_run_id' not in tables.cases.c
    assert all(
        foreign_key.column.table.name.startswith('rag_live_gate_')
        for table in (
            tables.authorizations,
            tables.cases,
            tables.dispatches,
            tables.transitions,
            tables.quality_reports,
        )
        for foreign_key in table.foreign_keys
    )


def test_inspection_reconstructs_transition_history_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_authority import RagReleaseAuthorityError
    from backend.app.rag.release_ledger import RagReleaseLedger

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    with engine.begin() as connection:
        mutations = _capture_bootstrap_authorization(connection, ledger, payload)
        ledger.append(
            connection,
            payload,
            actual_mutations=mutations,
            database_identity=_identity(),
        )
    tables = release_tables(build_rag_release_metadata())
    with engine.begin() as connection:
        connection.execute(
            tables.transitions.update().values(payload_canonical_bytes=b'{}')
        )
    with (
        engine.connect() as connection,
        pytest.raises(RagReleaseAuthorityError, match='transition'),
    ):
        authority.inspect(connection, database_identity=_identity())


def test_reviewer_repro_arbitrary_row_hmac_and_impossible_terminal_are_refused(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_ledger import (
        RagReleaseLedgerError,
        release_row_identity_hmac,
        validate_transition_payload,
    )

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    arbitrary = deepcopy(payload)
    for index, row in enumerate(arbitrary['affected_rows'], start=1):
        row['row_identity_hmac'] = format(index, 'x') * 64
    with pytest.raises(RagReleaseLedgerError, match='identity'):
        validate_transition_payload(arbitrary, identity_secret=_SECRET)

    extra = deepcopy(payload)
    extra['affected_rows'].append(
        {
            'row_kind': 'provider_safety_authority',
            'row_identity_hmac': release_row_identity_hmac(
                'provider_safety_authority',
                {'authority_uuid': '33333333-3333-3333-3333-333333333333'},
                identity_secret=_SECRET,
            ),
        }
    )
    extra['affected_rows'].sort(
        key=lambda item: (item['row_kind'], item['row_identity_hmac'])
    )
    with pytest.raises(RagReleaseLedgerError, match='matrix|overlap'):
        validate_transition_payload(extra, identity_secret=_SECRET)

    impossible = deepcopy(payload)
    impossible.update(
        transition_kind='authorization_finish_failed',
        outcome='ordinary_execution_failed',
        authorization_state_before='started',
        authorization_state_after='finished_failed',
        execution_process_instance_hmac='d' * 64,
        execution_runner_fence_hmac='e' * 64,
        authorization_reserved_cost_usd='10.000000',
        authorization_charged_cost_usd='999.000000',
    )
    with pytest.raises(RagReleaseLedgerError, match='cost|aggregate'):
        validate_transition_payload(impossible, identity_secret=_SECRET)


def test_release_mutation_set_rejects_noop_and_wrong_transaction(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        RagReleaseLedgerError,
        ReleaseRowPrimaryKey,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    with engine.begin() as connection:
        ledger.append(
            connection,
            payload,
            actual_mutations=_capture_bootstrap_authorization(
                connection, ledger, payload
            ),
            database_identity=_identity(),
        )
    with engine.begin() as connection:
        mutations = ledger.mutation_set(connection)
        row = ReleaseRowPrimaryKey(
            'authorization',
            {
                'ledger_uuid': payload['ledger_uuid'],
                'ledger_epoch': payload['ledger_epoch'],
                'approval_id_hmac': payload['approval_id_hmac'],
            },
        )
        authorization_table = release_tables(
            build_rag_release_metadata()
        ).authorizations
        mutations.plan(
            update(authorization_table)
            .where(
                authorization_table.c.approval_id_hmac == payload['approval_id_hmac']
            )
            .values(state='unused'),
            row,
        )
        marker = DurableFileAuthority.open_runtime(authority.marker_path)
        with (
            pytest.raises(RagReleaseLedgerError, match='not exact'),
            authority._authority_barrier(connection, marker=marker) as guard,
        ):
            mutations._execute_under_barrier(
                connection,
                authority=authority,
                barrier_guard=guard,
            )


def test_mutation_plan_executes_only_inside_authority_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.rag import release_authority
    from backend.app.rag.release_ledger import RagReleaseLedger

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    tables = release_tables(build_rag_release_metadata())
    entered = False
    original = release_authority.RagReleaseAuthority._authority_barrier

    @contextmanager
    def observed(self, connection, *, marker):
        nonlocal entered
        assert (
            connection.scalar(select(func.count()).select_from(tables.authorizations))
            == 0
        )
        entered = True
        with original(self, connection, marker=marker) as guard:
            yield guard

    monkeypatch.setattr(
        release_authority.RagReleaseAuthority, '_authority_barrier', observed
    )
    with engine.begin() as connection:
        mutations = _capture_bootstrap_authorization(connection, ledger, payload)
        assert (
            connection.scalar(select(func.count()).select_from(tables.authorizations))
            == 0
        )
        ledger.append(
            connection,
            payload,
            actual_mutations=mutations,
            database_identity=_identity(),
        )
    assert entered is True


def test_terminal_payload_counts_require_actual_complete_child_roster(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        RagReleaseLedgerError,
        ReleaseRowPrimaryKey,
        release_row_identity_hmac,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    bootstrap = _bootstrap_payload(snapshot)
    with engine.begin() as connection:
        ledger.append(
            connection,
            bootstrap,
            actual_mutations=_capture_bootstrap_authorization(
                connection, ledger, bootstrap
            ),
            database_identity=_identity(),
        )

    terminal = deepcopy(bootstrap)
    terminal.update(
        transition_kind='authorization_complete',
        outcome='quality_gate_green',
        from_generation=1,
        to_generation=2,
        authorization_state_before='started',
        authorization_state_after='complete',
        execution_process_instance_hmac='d' * 64,
        execution_runner_fence_hmac='e' * 64,
        case_claim_count=30,
        embedding_dispatch_count=10,
        generation_dispatch_count=30,
        total_dispatch_count=40,
        quality_report_hmac='f' * 64,
    )
    common = {
        'ledger_uuid': terminal['ledger_uuid'],
        'ledger_epoch': terminal['ledger_epoch'],
    }
    row_keys = {
        'authorization': {
            **common,
            'approval_id_hmac': terminal['approval_id_hmac'],
        },
        'quality_report': {
            **common,
            'approval_id_hmac': terminal['approval_id_hmac'],
        },
        'release_ledger': common,
        'release_transition': {**common, 'to_generation': 2},
    }
    terminal['affected_rows'] = sorted(
        (
            {
                'row_kind': kind,
                'row_identity_hmac': release_row_identity_hmac(
                    kind, keys, identity_secret=_SECRET
                ),
            }
            for kind, keys in row_keys.items()
        ),
        key=lambda item: (item['row_kind'], item['row_identity_hmac']),
    )
    authorization = release_tables(build_rag_release_metadata()).authorizations
    with engine.begin() as connection:
        connection.execute(
            update(authorization)
            .where(
                authorization.c.ledger_uuid == terminal['ledger_uuid'],
                authorization.c.ledger_epoch == terminal['ledger_epoch'],
                authorization.c.approval_id_hmac == terminal['approval_id_hmac'],
            )
            .values(
                state='started',
                execution_process_instance_hmac='d' * 64,
                execution_runner_fence_hmac='e' * 64,
            )
        )
    marker_before = authority.marker_path.read_bytes()
    with engine.begin() as connection:
        mutations = ledger.mutation_set(connection)
        mutations.plan(
            update(authorization)
            .where(
                authorization.c.ledger_uuid == terminal['ledger_uuid'],
                authorization.c.ledger_epoch == terminal['ledger_epoch'],
                authorization.c.approval_id_hmac == terminal['approval_id_hmac'],
            )
            .values(
                state='complete',
                execution_process_instance_hmac='d' * 64,
                execution_runner_fence_hmac='e' * 64,
                case_claim_count=30,
                embedding_dispatch_count=10,
                generation_dispatch_count=30,
                total_dispatch_count=40,
            ),
            ReleaseRowPrimaryKey('authorization', row_keys['authorization']),
        )
        quality = release_tables(build_rag_release_metadata()).quality_reports
        mutations.plan(
            insert(quality).values(
                **row_keys['quality_report'],
                quality_report_hmac='f' * 64,
                manifest_hmac='a' * 64,
                baseline_hmac='b' * 64,
                reviewer_roster_hmac='c' * 64,
                payload_canonical_bytes=b'{}',
            ),
            ReleaseRowPrimaryKey('quality_report', row_keys['quality_report']),
        )
        with pytest.raises(RagReleaseLedgerError, match='terminal roster'):
            ledger.append(
                connection,
                terminal,
                actual_mutations=mutations,
                database_identity=_identity(),
            )
    assert authority.marker_path.read_bytes() == marker_before
    with engine.connect() as connection:
        assert connection.scalar(select(authorization.c.state)) == 'started'
        quality = release_tables(build_rag_release_metadata()).quality_reports
        assert connection.scalar(select(func.count()).select_from(quality)) == 0


@pytest.mark.parametrize(
    'kind',
    [
        'authorization_abort_control',
        'authorization_abort_component',
        'authorization_abort_component_snapshot',
        'authorization_abort_corpus_drift',
        'authorization_abort_execution_crash',
        'authorization_abort_final',
        'authorization_abort_snapshot',
        'authorization_complete',
        'authorization_finish_failed',
        'authorization_finish_quality_failed',
    ],
)
def test_every_terminal_kind_rejects_self_reported_missing_case_roster(
    tmp_path: Path, kind: str
) -> None:
    from backend.app.rag.release_ledger import RagReleaseLedger, RagReleaseLedgerError

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    payload['transition_kind'] = kind
    payload['case_claim_count'] = 1
    with (
        engine.connect() as connection,
        pytest.raises(RagReleaseLedgerError, match='terminal roster'),
    ):
        ledger._assert_database_roster(connection, payload)


def test_mutation_plan_refuses_unsealed_barrier_without_executing(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_authority import RagReleaseAuthorityError
    from backend.app.rag.release_ledger import RagReleaseLedger

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_REVIEW
        )
    payload = _bootstrap_payload(snapshot)
    tables = release_tables(build_rag_release_metadata())
    with engine.begin() as connection:
        mutations = _capture_bootstrap_authorization(connection, ledger, payload)
        with pytest.raises(RagReleaseAuthorityError, match='barrier guard'):
            mutations._execute_under_barrier(
                connection, authority=authority, barrier_guard=object()
            )
        assert (
            connection.scalar(select(func.count()).select_from(tables.authorizations))
            == 0
        )


def test_authorization_owner_abort_and_single_claim_are_database_enforced() -> None:
    tables = release_tables(build_rag_release_metadata())
    engine = create_engine('sqlite+pysqlite:///:memory:')
    build_rag_release_metadata().create_all(engine)
    ledger_values = {
        'ledger_uuid': '11111111-1111-1111-1111-111111111111',
        'ledger_epoch': 1,
        'generation': 0,
        'last_transition_digest': None,
        'predecessor_marker_digest': None,
        'rebootstrap_reason_hmac': None,
        'marker_file_digest': '1' * 64,
        'bootstrap_review_envelope_hmac': '2' * 64,
        'bootstrap_review_nonce_hmac': '3' * 64,
        'bootstrap_operation': 'release-ledger-init',
        'fingerprint_key_version': 'v1',
        'fingerprint_key_material_verifier': '4' * 64,
        'designated_environment_id_hmac': '5' * 64,
        'designated_host_id_hmac': '6' * 64,
        'validation_database_identity_hmac': '7' * 64,
        'validation_database_locator_hmac': '8' * 64,
        'validation_database_identity_uuid': ('22222222-2222-2222-2222-222222222222'),
        'validation_database_oid': 1,
    }
    authorization_values = {
        'ledger_uuid': ledger_values['ledger_uuid'],
        'ledger_epoch': 1,
        'approval_id_hmac': '8' * 64,
        'approval_hmac': '9' * 64,
        'base_generation': 0,
        'state': 'unused',
        'approved_corpus_snapshot_hmac': 'a' * 64,
        'approved_provider_safety_snapshot_hmac': 'b' * 64,
        'provider_safety_envelope_digest': 'c' * 64,
        'validation_database_identity_hmac': '7' * 64,
        'manifest_hmac': 'd' * 64,
        'baseline_hmac': 'e' * 64,
        'reviewer_roster_hmac': 'f' * 64,
        'execution_process_instance_hmac': None,
        'execution_runner_fence_hmac': None,
        'case_claim_count': 0,
        'embedding_dispatch_count': 0,
        'generation_dispatch_count': 0,
        'total_dispatch_count': 0,
        'reserved_cost_usd': '0.000000',
        'charged_cost_usd': '0.000000',
    }
    with engine.begin() as connection:
        connection.execute(insert(tables.ledgers).values(**ledger_values))
        connection.execute(insert(tables.authorizations).values(**authorization_values))
        connection.execute(
            update(tables.authorizations).values(state='aborted_corpus_drift')
        )
        connection.execute(
            update(tables.authorizations).values(
                state='started',
                execution_process_instance_hmac='1' * 64,
                execution_runner_fence_hmac='2' * 64,
            )
        )
        case_common = {
            'ledger_uuid': ledger_values['ledger_uuid'],
            'ledger_epoch': 1,
            'approval_id_hmac': authorization_values['approval_id_hmac'],
            'case_projection_hmac': None,
            'state': 'claimed',
            'runtime_agent_run_id_hmac': '3' * 64,
            'embedding_reserved_cost_usd': '0.000000',
            'generation_reserved_cost_usd': '0.000000',
            'total_reserved_cost_usd': '0.000000',
        }
        connection.execute(
            insert(tables.cases).values(
                **case_common, case_id_hmac='4' * 64, manifest_ordinal=0
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                insert(tables.cases).values(
                    **case_common,
                    case_id_hmac='5' * 64,
                    manifest_ordinal=1,
                )
            )
