"""Actual append security/audit regressions for the final scoped review."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import event

from backend.app.rag.release_ledger import (
    RagReleaseLedgerError,
    validate_transition_payload,
)
from backend.tests.test_rag_release_ledger import _SECRET
from backend.tests.test_rag_release_ledger_review_q import (
    _release_test_seam,  # noqa: F401
)
from backend.tests.test_rag_release_ledger_round4 import harness as harness


def failure_changes(harness):
    case, run = harness.records('case')[-1], harness.records('agent_run')[-1]
    failed = {**case, 'state': 'failed'}
    changes = [
        ('case', case, failed),
        (
            'agent_run',
            run,
            {
                **run,
                'status': 'failed',
                'run_record_phase': 'final',
                'completed_at': run['started_at'] + timedelta(seconds=2),
            },
        ),
    ]
    for child in harness.records('cost_component')[-2:]:
        if child['dispatch_state'] == 'not_attempted':
            changes.append(
                (
                    'cost_component',
                    child,
                    {
                        **child,
                        'dispatch_state': 'terminal',
                        'reserved_input_tokens': 0,
                        'reserved_output_tokens': 0,
                        'reserved_cost_usd': Decimal(0),
                    },
                )
            )
    return changes, failed


FORBIDDEN_RUN_FIELDS = [
    ('id', 99),
    ('agent_name', 'forged'),
    ('prompt_version', 'forged'),
    ('source_window', 'forged'),
    ('cache_key', 'forged'),
    ('model_name', 'forged'),
    ('generation_provider', 'forged'),
    ('generation_reasoning_effort', 'forged'),
    ('generation_route_version', 'forged'),
    ('generation_output_contract_version', 'forged'),
    ('input_tokens', 1),
    ('output_tokens', 1),
    ('total_tokens', 1),
    ('estimated_cost_usd', 0.5),
    ('permission_level', 'public'),
    ('metadata', {'raw_sensitive': 'must-never-be-signed-in-plaintext'}),
    ('workflow_thread_id', 'forged'),
    ('effect_key', 'forged'),
    ('run_contract_version', 'forged'),
    ('started_at', 'clock'),
    ('projection_owner_fence_hmac', '9' * 64),
    ('status', 'complete'),
    ('run_record_phase', 'admission_only'),
    ('total_charged_cost_usd', Decimal('0.001000')),
    ('completed_at', 'past'),
]


@pytest.mark.parametrize('field,value', FORBIDDEN_RUN_FIELDS)
def test_agent_run_piggyback_rejected_before_any_sql_or_marker(harness, field, value):
    """Removing a deny-by-default audit guard must permit no signed lifecycle write."""
    harness.claim()
    changes, failed = failure_changes(harness)
    before = {
        kind: harness.records(kind)
        for kind in ('authorization', 'case', 'agent_run', 'cost_component')
    }
    marker = harness.authority.marker_path.read_bytes()
    changes[1][2][field] = (
        changes[1][1]['started_at'] + timedelta(seconds=1)
        if value == 'clock'
        else changes[1][1]['started_at'] - timedelta(seconds=1)
        if value == 'past'
        else value
    )
    with harness.engine.begin() as connection:
        payload, mutations = harness.prepare(
            connection,
            'case_failure',
            changes,
            case=failed,
            outcome='model_unavailable',
        )
        writes = []

        def record(_conn, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                writes.append(statement)

        event.listen(harness.engine, 'before_cursor_execute', record)
        try:
            with pytest.raises(RagReleaseLedgerError):
                harness.ledger.append(
                    connection,
                    payload,
                    actual_mutations=mutations,
                    database_identity=harness.database_identity,
                )
        finally:
            event.remove(harness.engine, 'before_cursor_execute', record)
        assert writes == []
    assert harness.authority.marker_path.read_bytes() == marker
    assert {kind: harness.records(kind) for kind in before} == before


@pytest.mark.parametrize('field', ['created_at', 'updated_at'])
def test_cost_cancellation_cannot_piggyback_unrelated_timestamp(harness, field):
    harness.claim()
    changes, failed = failure_changes(harness)
    changes[-1][2][field] += timedelta(seconds=1)
    marker = harness.authority.marker_path.read_bytes()
    before = harness.records('cost_component')
    with pytest.raises(RagReleaseLedgerError):
        harness.append(
            'case_failure', changes, case=failed, outcome='model_unavailable'
        )
    assert harness.records('cost_component') == before
    assert harness.authority.marker_path.read_bytes() == marker


def test_affected_runtime_signature_binds_exact_completion_timestamp(harness):
    harness.claim()
    changes, failed = failure_changes(harness)
    with harness.engine.begin() as connection:
        first, _ = harness.prepare(
            connection,
            'case_failure',
            changes,
            case=failed,
            outcome='model_unavailable',
        )
        changes[1][2]['completed_at'] += timedelta(seconds=1)
        second, _ = harness.prepare(
            connection,
            'case_failure',
            changes,
            case=failed,
            outcome='model_unavailable',
        )
    assert (
        validate_transition_payload(first, identity_secret=_SECRET).transition_digest
        != validate_transition_payload(
            second, identity_secret=_SECRET
        ).transition_digest
    )


def test_completion_cannot_precede_start_within_the_same_second(harness, monkeypatch):
    from backend.app.rag import release_review

    class Clock:
        @staticmethod
        def now(_zone):
            return datetime(2026, 9, 13, 1, 2, 3, 500000, tzinfo=UTC)

    monkeypatch.setattr(
        release_review,
        'datetime',
        Clock,
    )
    harness.claim()
    changes, failed = failure_changes(harness)
    assert changes[1][1]['started_at'].microsecond > 0
    changes[1][2]['completed_at'] = changes[1][1]['started_at'].replace(microsecond=0)
    marker = harness.authority.marker_path.read_bytes()
    with pytest.raises(RagReleaseLedgerError):
        harness.append(
            'case_failure', changes, case=failed, outcome='model_unavailable'
        )
    assert harness.authority.marker_path.read_bytes() == marker


@pytest.mark.parametrize(
    'field',
    [
        'id',
        'agent_run_id',
        'component',
        'component_ordinal',
        'dispatch_state',
        'dispatch_fence_hmac',
        'process_instance_hmac',
        'attempted',
        'dispatch_count',
        'reserved_input_tokens',
        'reserved_output_tokens',
        'actual_input_tokens',
        'actual_output_tokens',
        'reserved_cost_usd',
        'charged_cost_usd',
        'charge_basis',
        'overrun',
        'provider',
        'model',
        'authorized_model_config_version',
        'authorized_model_config_snapshot_hmac',
        'authorized_cost_policy_version',
        'authorized_token_estimator_version',
        'authorized_policy_snapshot_hmac',
        'terminal_outcome',
        'created_at',
        'updated_at',
    ],
)
def test_every_cost_child_column_has_exact_cancellation_delta_before_sql(
    harness, field
):
    """Any extra delta on a cancelled child must reject before the parent write."""
    harness.claim()
    changes, failed = failure_changes(harness)
    child = changes[-1][2]
    old = child[field]
    if field.endswith('_at'):
        value = old + timedelta(seconds=1)
    elif field in {'actual_input_tokens', 'actual_output_tokens'}:
        value = 1
    elif isinstance(old, bool):
        value = not old
    elif isinstance(old, (int, Decimal)):
        value = old + 1
    else:
        value = 'forged' if old is None or isinstance(old, str) else 1
    child[field] = value
    before = harness.records('cost_component')
    marker = harness.authority.marker_path.read_bytes()
    with harness.engine.connect() as connection:
        payload, mutations = harness.prepare(
            connection,
            'case_failure',
            changes,
            case=failed,
            outcome='model_unavailable',
        )
        writes = []

        def record(_conn, _cursor, statement, _params, _ctx, _many):
            if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                writes.append(statement)

        event.listen(harness.engine, 'before_cursor_execute', record)
        try:
            with pytest.raises(RagReleaseLedgerError):
                harness.ledger.append(
                    connection,
                    payload,
                    actual_mutations=mutations,
                    database_identity=harness.database_identity,
                )
        finally:
            event.remove(harness.engine, 'before_cursor_execute', record)
        assert writes == []
    assert harness.records('cost_component') == before
    assert harness.authority.marker_path.read_bytes() == marker


@pytest.mark.parametrize('tamper', ['before', 'after', 'hmac', 'missing', 'expression'])
def test_mutation_seal_rejects_stale_or_changed_plan_before_sql(harness, tamper):
    from dataclasses import replace

    from sqlalchemy import update

    from backend.app.models.agent_runs import AgentRun

    harness.claim()
    changes, failed = failure_changes(harness)
    marker = harness.authority.marker_path.read_bytes()
    with harness.engine.connect() as connection:
        payload, mutations = harness.prepare(
            connection,
            'case_failure',
            changes,
            case=failed,
            outcome='model_unavailable',
        )
        entry = next(
            row for row in payload['affected_rows'] if row['row_kind'] == 'agent_run'
        )
        if tamper == 'before':
            connection.execute(
                update(AgentRun.__table__)
                .where(AgentRun.id == harness.records('agent_run')[0]['id'])
                .values(metadata={'revision': 2})
            )
            connection.commit()
        elif tamper == 'after':
            plan = mutations._plans[1]
            mutations._plans[1] = replace(
                plan,
                statement=plan.statement.values(
                    completed_at=changes[1][2]['completed_at'] + timedelta(seconds=1)
                ),
            )
        elif tamper == 'expression':
            plan = mutations._plans[1]
            mutations._plans[1] = replace(
                plan,
                statement=plan.statement.values(input_tokens=AgentRun.input_tokens + 1),
            )
        elif tamper == 'missing':
            del entry['row_mutation_hmac']
        else:
            entry['row_mutation_hmac'] = 'f' * 64
        before = harness.records('agent_run')
        writes = []

        def record(_conn, _cursor, statement, _params, _ctx, _many):
            if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                writes.append(statement)

        event.listen(harness.engine, 'before_cursor_execute', record)
        try:
            with pytest.raises(RagReleaseLedgerError):
                harness.ledger.append(
                    connection,
                    payload,
                    actual_mutations=mutations,
                    database_identity=harness.database_identity,
                )
        finally:
            event.remove(harness.engine, 'before_cursor_execute', record)
        assert writes == []
    assert harness.records('agent_run') == before
    assert harness.authority.marker_path.read_bytes() == marker
