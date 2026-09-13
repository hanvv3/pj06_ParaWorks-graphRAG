from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables

_SECRET = b'task23-release-runtime-key-material-32-bytes'
_REVIEW = {'review_envelope_hmac': '1' * 64, 'review_nonce_hmac': '2' * 64}


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
    )


def _identity():
    from backend.app.rag.release_authority import ValidationDatabaseIdentity

    return ValidationDatabaseIdentity('validation', 53)


def _bootstrap_payload(snapshot) -> dict[str, object]:
    affected = [
        {'row_identity_hmac': '1' * 64, 'row_kind': 'authorization'},
        {'row_identity_hmac': '2' * 64, 'row_kind': 'release_ledger'},
        {'row_identity_hmac': '3' * 64, 'row_kind': 'release_transition'},
    ]
    return {
        'affected_rows': affected,
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
        (lambda value: value.update(authorization_reserved_cost_usd='0E-6'), 'six-place'),
        (lambda value: value.update(outcome='quality_gate_green'), 'outcome'),
        (lambda value: value.update(case_id_hmac='9' * 64), 'null'),
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
    actual = tuple(
        (row['row_kind'], row['row_identity_hmac'])
        for row in payload['affected_rows']
    )
    with engine.begin() as connection:
        advanced = ledger.append(
            connection,
            payload,
            actual_affected_rows=actual,
            database_identity=_identity(),
        )
    assert advanced.generation == 1
    assert advanced.last_transition_digest is not None
    tables = release_tables(build_rag_release_metadata())
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(tables.transitions)) == 1
        row = connection.execute(select(tables.transitions)).mappings().one()
        assert row['generation'] == 1
        assert bytes(row['payload_canonical_bytes'])

    with engine.begin() as connection, pytest.raises(
        RagReleaseLedgerError, match='generation'
    ):
        ledger.append(
            connection,
            payload,
            actual_affected_rows=actual,
            database_identity=_identity(),
        )
    wrong_actual = (*actual[:-1], ('release_transition', 'f' * 64))
    next_payload = deepcopy(payload)
    next_payload['from_generation'] = 1
    next_payload['to_generation'] = 2
    with engine.begin() as connection, pytest.raises(
        RagReleaseLedgerError, match='affected'
    ):
        ledger.append(
            connection,
            next_payload,
            actual_affected_rows=wrong_actual,
            database_identity=_identity(),
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
    affected = tuple(
        (row['row_kind'], row['row_identity_hmac'])
        for row in payload['affected_rows']
    )
    with engine.begin() as connection:
        ledger.append(
            connection,
            payload,
            actual_affected_rows=affected,
            database_identity=_identity(),
        )
    tables = release_tables(build_rag_release_metadata())
    with engine.begin() as connection:
        connection.execute(
            tables.transitions.update().values(payload_canonical_bytes=b'{}')
        )
    with engine.connect() as connection, pytest.raises(
        RagReleaseAuthorityError, match='transition'
    ):
        authority.inspect(connection, database_identity=_identity())
