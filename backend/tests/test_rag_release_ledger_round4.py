from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine

from backend.app.rag.release_ledger import (
    RagReleaseLedger,
    RagReleaseLedgerError,
    _assert_exact_transition_snapshots,
    release_row_identity_hmac,
    validate_transition_payload,
)
from backend.tests.test_rag_release_ledger import _SECRET
from backend.tests.test_rag_release_ledger_review_q import (
    _case_claim_projection,
    _release_test_seam,  # noqa: F401
    _valid_payload_for_kind,
)


def _payload(kind):
    return _valid_payload_for_kind(
        SimpleNamespace(
            ledger_uuid='11111111-1111-1111-1111-111111111111',
            ledger_epoch=1,
            validation_database_identity_hmac='2' * 64,
        ),
        kind,
    )


def _observe_affected(payload, row_kind):
    for row in list(payload['affected_rows']):
        if row['row_kind'] == row_kind:
            payload['affected_rows'].remove(row)
            payload['observation_set'].append({**row, 'row_projection_hmac': 'a' * 64})
    payload['observation_set'].sort(
        key=lambda row: (row['row_kind'], row['row_identity_hmac'])
    )


def test_ordinary_transition_cannot_omit_current_provider_proof():
    payload = _payload('component_claim')
    payload['observation_set'] = [
        row
        for row in payload['observation_set']
        if not row['row_kind'].startswith('provider_')
    ]
    with pytest.raises(RagReleaseLedgerError, match='provider'):
        validate_transition_payload(payload, identity_secret=_SECRET)


def test_generation_outcome_requires_mutated_pending_parent():
    payload = _payload('component_outcome')
    payload['component'] = 'answer_generation'
    payload['embedding_dispatch_count'] = 0
    payload['generation_dispatch_count'] = 1
    for row in payload['affected_rows']:
        if row['row_kind'] == 'dispatch':
            row['row_identity_hmac'] = release_row_identity_hmac(
                'dispatch',
                {
                    'ledger_uuid': payload['ledger_uuid'],
                    'ledger_epoch': 1,
                    'approval_id_hmac': payload['approval_id_hmac'],
                    'case_id_hmac': payload['case_id_hmac'],
                    'component': 'answer_generation',
                },
                identity_secret=_SECRET,
            )
    for row in list(payload['observation_set']):
        if row['row_kind'] == 'agent_run':
            payload['observation_set'].remove(row)
            payload['affected_rows'].append(
                {
                    key: value
                    for key, value in row.items()
                    if key != 'row_projection_hmac'
                }
            )
    payload['affected_rows'].sort(
        key=lambda row: (row['row_kind'], row['row_identity_hmac'])
    )
    validate_transition_payload(payload, identity_secret=_SECRET)


def test_post_generation_failure_observes_all_terminal_costs():
    payload = _payload('case_failure')
    payload['outcome'] = 'persistence_failed'
    _observe_affected(payload, 'cost_component')
    validate_transition_payload(payload, identity_secret=_SECRET)


@pytest.mark.parametrize(
    'kind', ['case_failure', 'authorization_abort_execution_crash']
)
def test_pending_parent_can_fail_without_rewriting_terminal_costs(kind):
    payload = _payload('case_failure')
    payload['transition_kind'] = kind
    payload['outcome'] = 'persistence_failed'
    before = {
        'status': 'running',
        'run_record_phase': 'cost_finalized_pending_projection',
        'projection_owner_fence_hmac': payload['execution_runner_fence_hmac'],
        'completed_at': None,
    }
    after = {
        **before,
        'status': 'failed',
        'run_record_phase': 'final',
        'completed_at': datetime.now(UTC),
    }
    costs = [
        {'component': component, 'dispatch_state': 'terminal'}
        for component in ('query_embedding', 'answer_generation')
    ]
    _assert_exact_transition_snapshots(
        payload,
        {'agent_run': [after], 'cost_component': costs},
        {'agent_run': [before], 'cost_component': deepcopy(costs)},
        {'agent_run': [True], 'cost_component': [False, False]},
    )


def test_later_case_claim_cannot_rotate_execution_owner():
    payload = _payload('case_claim')
    payload['authorization_state_before'] = 'started'
    payload['case_claim_count'] = 2
    with create_engine('sqlite://').connect() as connection:
        mutations = RagReleaseLedger.mutation_set(connection)
        _case_claim_projection(mutations, payload)
    before = mutations._before_snapshots[0]
    before.update(
        state='started',
        execution_process_instance_hmac='9' * 64,
        execution_runner_fence_hmac='0' * 64,
        case_claim_count=1,
    )
    mutations._after_snapshots[0]['case_claim_count'] = 2
    for component in mutations._after_snapshots[3:]:
        component['dispatch_state'] = 'not_attempted'
    with pytest.raises(RagReleaseLedgerError, match='owner'):
        mutations.assert_payload_projection(payload, identity_secret=_SECRET)


def test_thirty_zero_dispatch_ordinary_failures_have_terminal_payload():
    payload = _payload('authorization_finish_failed')
    payload.update(
        embedding_dispatch_count=0, generation_dispatch_count=0, total_dispatch_count=0
    )
    payload['observation_set'] = [
        row for row in payload['observation_set'] if row['row_kind'] != 'dispatch'
    ]
    validate_transition_payload(payload, identity_secret=_SECRET)


def test_terminal_payload_requires_all_thirty_runtime_roster_observations():
    payload = _payload('authorization_complete')
    payload['observation_set'] = [
        row
        for row in payload['observation_set']
        if row['row_kind'].startswith('provider_')
    ]
    with pytest.raises(RagReleaseLedgerError, match='roster'):
        validate_transition_payload(payload, identity_secret=_SECRET)


def test_component_outcome_cannot_rotate_claimed_child_process():
    payload = _payload('component_outcome')
    run = {
        'status': 'running',
        'run_record_phase': 'admission',
        'completed_at': None,
        'projection_owner_fence_hmac': None,
    }
    dispatch = {
        'state': 'terminal',
        'dispatch_count': 1,
        'dispatch_fence_hmac': '5' * 64,
        'reserved_cost_usd': '0.000000',
        'charged_cost_usd': '0.000000',
        'charge_basis': 'reserved',
    }
    child = {
        **dispatch,
        'component': 'query_embedding',
        'dispatch_state': 'terminal',
        'attempted': True,
        'terminal_outcome': 'component_succeeded',
        'process_instance_hmac': 'd' * 64,
    }
    before = {
        **child,
        'dispatch_state': 'dispatching',
        'terminal_outcome': None,
        'process_instance_hmac': '0' * 64,
    }
    with pytest.raises(RagReleaseLedgerError, match='owner'):
        _assert_exact_transition_snapshots(
            payload,
            {'agent_run': [run], 'cost_component': [child], 'dispatch': [dispatch]},
            {
                'agent_run': [run],
                'cost_component': [before],
                'dispatch': [{**dispatch, 'state': 'dispatching'}],
            },
            {'agent_run': [False], 'cost_component': [True], 'dispatch': [True]},
        )


@pytest.fixture
def harness(tmp_path):
    from backend.tests.release_ledger_fixtures import ReleaseHarness
    from backend.tests.test_rag_release_ledger import _authority, _identity

    engine = create_engine('sqlite://')
    yield ReleaseHarness(engine, _authority(tmp_path), _SECRET, _identity())
    engine.dispose()


def test_actual_generation_pending_and_persistence_failure_preserve_charge(harness):
    from decimal import Decimal

    harness.claim()
    claim = harness.claim_generation()
    assert not any(row['row_kind'] == 'agent_run' for row in claim['affected_rows'])
    outcome = harness.generation_outcome()
    assert any(row['row_kind'] == 'agent_run' for row in outcome['affected_rows'])
    children = harness.records('cost_component')
    failed = harness.fail_case()
    assert not any(
        row['row_kind'] == 'cost_component' for row in failed['affected_rows']
    )
    assert (
        sum(row['row_kind'] == 'cost_component' for row in failed['observation_set'])
        == 2
    )
    assert harness.records('cost_component') == children
    parent = harness.records('agent_run')[0]
    assert (
        parent['status'],
        parent['run_record_phase'],
        parent['total_charged_cost_usd'],
    ) == ('failed', 'final', Decimal('0.000400'))


def test_admission_failure_cannot_invent_projection_owner(harness):
    harness.claim()
    case, run = harness.records('case')[0], harness.records('agent_run')[0]
    child = harness.records('cost_component')[-1]
    failed = {**case, 'state': 'failed'}
    with pytest.raises(RagReleaseLedgerError, match='projection.*fence'):
        harness.append(
            'case_failure',
            [
                ('case', case, failed),
                (
                    'agent_run',
                    run,
                    {
                        **run,
                        'status': 'failed',
                        'run_record_phase': 'final',
                        'completed_at': datetime.now(UTC),
                        'projection_owner_fence_hmac': '9' * 64,
                    },
                ),
                (
                    'cost_component',
                    child,
                    {
                        **child,
                        'dispatch_state': 'terminal',
                        'reserved_input_tokens': 0,
                        'reserved_output_tokens': 0,
                        'reserved_cost_usd': 0,
                    },
                ),
            ],
            case=failed,
            outcome='model_unavailable',
        )


def test_actual_thirty_zero_dispatch_failures_finish_with_full_roster(harness):
    for ordinal in range(30):
        harness.claim(ordinal)
        harness.fail_case(outcome='model_unavailable')
    auth = harness.records('authorization')[0]
    payload = harness.append(
        'authorization_finish_failed',
        [('authorization', auth, {**auth, 'state': 'finished_failed'})],
        outcome='ordinary_execution_failed',
    )
    assert payload['total_dispatch_count'] == 0
    assert payload['authorization_charged_cost_usd'] == '0.000000'
    assert sum(row['row_kind'] == 'case' for row in payload['observation_set']) == 30
    assert (
        sum(row['row_kind'] == 'agent_run' for row in payload['observation_set']) == 30
    )
    assert (
        sum(row['row_kind'] == 'cost_component' for row in payload['observation_set'])
        == 60
    )


def _provider_drift(harness, *, changed_digest=True, blocked=True):
    from sqlalchemy import update

    from backend.app.models.rag_runtime import (
        RagProviderReadiness,
        RagProviderSafetyAuthority,
    )

    with harness.engine.begin() as connection:
        if changed_digest:
            connection.execute(
                update(RagProviderSafetyAuthority).values(
                    envelope_digest='9' * 64, global_safety_generation=1
                )
            )
        if blocked:
            connection.execute(
                update(RagProviderReadiness)
                .where(RagProviderReadiness.component == 'answer_generation')
                .values(state='rebind_required')
            )


@pytest.mark.parametrize(
    'changed_digest,blocked', [(True, False), (False, True), (True, True)]
)
def test_provider_drift_allows_post_generation_abort_without_cost_mutation(
    harness, changed_digest, blocked
):
    harness.claim()
    harness.claim_generation()
    harness.generation_outcome()
    children = harness.records('cost_component')
    _provider_drift(harness, changed_digest=changed_digest, blocked=blocked)
    payload = harness.fail_case(
        'authorization_abort_control', 'provider_safety_unavailable'
    )
    assert (
        harness.records('authorization')[0]['provider_safety_envelope_digest']
        == '8' * 64
    )
    assert harness.records('cost_component') == children
    assert all(
        row['row_kind']
        not in {'provider_safety_authority', 'provider_readiness', 'cost_component'}
        for row in payload['affected_rows']
    )


def test_provider_drift_rejects_ordinary_component_claim_before_marker(harness):
    harness.claim()
    before_marker = harness.authority.marker_path.read_bytes()
    before_costs = harness.records('cost_component')
    _provider_drift(harness)
    with pytest.raises(RagReleaseLedgerError, match='provider.*approval'):
        harness.claim_generation()
    assert harness.authority.marker_path.read_bytes() == before_marker
    assert harness.records('cost_component') == before_costs
    assert harness.records('dispatch') == []


def test_equal_ready_provider_cannot_fabricate_control_abort(harness):
    harness.claim()
    with pytest.raises(RagReleaseLedgerError, match='drift or non-ready'):
        harness.fail_case('authorization_abort_control', 'provider_safety_unavailable')
    assert harness.records('case')[0]['state'] == 'claimed'


def test_query_success_observes_parent_and_generation_sibling(harness):
    from decimal import Decimal

    harness.claim(query_reserve='0.001000')
    harness.claim_generation('query_embedding')
    auth = harness.records('authorization')[0]
    case = harness.records('case')[0]
    query = harness.records('cost_component')[0]
    dispatch = harness.records('dispatch')[0]
    before_parent = harness.records('agent_run')[0]
    payload = harness.append(
        'component_outcome',
        [
            ('authorization', auth, {**auth, 'charged_cost_usd': Decimal('0.000400')}),
            (
                'cost_component',
                query,
                {
                    **query,
                    'dispatch_state': 'terminal',
                    'charge_basis': 'actual',
                    'charged_cost_usd': Decimal('0.000400'),
                    'actual_input_tokens': 1,
                    'actual_output_tokens': 0,
                    'terminal_outcome': 'component_succeeded',
                },
            ),
            (
                'dispatch',
                dispatch,
                {
                    **dispatch,
                    'state': 'terminal',
                    'charge_basis': 'actual',
                    'charged_cost_usd': Decimal('0.000400'),
                },
            ),
        ],
        case=case,
        component='query_embedding',
        outcome='component_succeeded',
    )
    assert harness.records('agent_run')[0] == before_parent
    assert not any(row['row_kind'] == 'agent_run' for row in payload['affected_rows'])
    assert (
        sum(row['row_kind'] == 'cost_component' for row in payload['observation_set'])
        == 1
    )


def test_actual_case_projection_observes_charged_children(harness):
    harness.claim()
    harness.claim_generation()
    harness.generation_outcome()
    case = harness.records('case')[0]
    run = harness.records('agent_run')[0]
    complete = {**case, 'state': 'complete', 'case_projection_hmac': '4' * 64}
    payload = harness.append(
        'case_outcome',
        [
            ('case', case, complete),
            (
                'agent_run',
                run,
                {
                    **run,
                    'status': 'complete',
                    'run_record_phase': 'final',
                    'completed_at': datetime.now(UTC),
                },
            ),
        ],
        case=complete,
        outcome='supported',
    )
    assert (
        sum(row['row_kind'] == 'cost_component' for row in payload['observation_set'])
        == 2
    )
    assert not any(
        row['row_kind'] in {'authorization', 'cost_component'}
        for row in payload['affected_rows']
    )


def test_generation_success_cannot_hide_provider_overrun(harness):
    harness.claim()
    harness.claim_generation()
    with pytest.raises(RagReleaseLedgerError, match='overrun'):
        harness.generation_outcome(overrun=True)
    assert harness.records('agent_run')[0]['run_record_phase'] == 'admission'


@pytest.mark.parametrize(
    'kind',
    [
        'authorization_abort_snapshot',
        'authorization_abort_final',
        'authorization_abort_corpus_drift',
        'authorization_abort_execution_crash',
    ],
)
def test_case_null_abort_preserves_roster_under_current_provider_proof(harness, kind):
    harness.claim()
    harness.fail_case(outcome='model_unavailable')
    auth = harness.records('authorization')[0]
    state = {
        'authorization_abort_snapshot': 'aborted_provider_safety',
        'authorization_abort_final': 'aborted_provider_safety',
        'authorization_abort_corpus_drift': 'aborted_corpus_drift',
        'authorization_abort_execution_crash': 'aborted_execution_crash',
    }[kind]
    outcome = 'provider_safety_unavailable'
    overrides = {}
    if kind == 'authorization_abort_corpus_drift':
        outcome = 'live_corpus_snapshot_changed'
        overrides['current_corpus_snapshot_hmac'] = '9' * 64
        _provider_drift(harness)
    elif kind == 'authorization_abort_execution_crash':
        outcome = 'abandoned_unknown'
        overrides['execution_crash_attestation_hmac'] = 'a' * 64
        _provider_drift(harness)
    else:
        _provider_drift(harness)
    payload = harness.append(
        kind,
        [('authorization', auth, {**auth, 'state': state})],
        outcome=outcome,
        payload_overrides=overrides,
    )
    assert {row['row_kind'] for row in payload['affected_rows']} == {
        'authorization',
        'release_ledger',
        'release_transition',
    }


def test_post_call_snapshot_abort_preserves_actual_charge_and_owner(harness):
    from decimal import Decimal

    harness.claim()
    harness.claim_generation()
    _provider_drift(harness)
    auth, case, run = (
        harness.records('authorization')[0],
        harness.records('case')[0],
        harness.records('agent_run')[0],
    )
    child, dispatch = (
        harness.records('cost_component')[-1],
        harness.records('dispatch')[0],
    )
    failed = {**case, 'state': 'failed'}
    payload = harness.append(
        'authorization_abort_component_snapshot',
        [
            (
                'authorization',
                auth,
                {
                    **auth,
                    'state': 'aborted_provider_safety',
                    'charged_cost_usd': Decimal('0.000400'),
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
                    'total_charged_cost_usd': Decimal('0.000400'),
                },
            ),
            (
                'cost_component',
                child,
                {
                    **child,
                    'dispatch_state': 'terminal',
                    'charge_basis': 'actual',
                    'charged_cost_usd': Decimal('0.000400'),
                    'actual_input_tokens': 1,
                    'actual_output_tokens': 2,
                    'terminal_outcome': 'provider_safety_unavailable',
                },
            ),
            (
                'dispatch',
                dispatch,
                {
                    **dispatch,
                    'state': 'terminal',
                    'charge_basis': 'actual',
                    'charged_cost_usd': Decimal('0.000400'),
                },
            ),
        ],
        case=failed,
        component='answer_generation',
        outcome='provider_safety_unavailable',
    )
    assert payload['authorization_charged_cost_usd'] == '0.000400'
    assert all(
        not row['row_kind'].startswith('provider_') for row in payload['affected_rows']
    )


@pytest.mark.parametrize('tamper', ['missing', 'extra', 'changed'])
def test_observation_roster_tamper_is_zero_marker_and_database_change(harness, tamper):
    harness.claim()
    harness.fail_case(outcome='model_unavailable')
    before_marker = harness.authority.marker_path.read_bytes()
    auth = harness.records('authorization')[0]
    with harness.engine.begin() as connection:
        payload, mutations = harness.prepare(
            connection,
            'authorization_abort_execution_crash',
            [('authorization', auth, {**auth, 'state': 'aborted_execution_crash'})],
            outcome='abandoned_unknown',
        )
        payload['execution_crash_attestation_hmac'] = 'f' * 64
        target = next(
            row
            for row in payload['observation_set']
            if row['row_kind'] == 'cost_component'
        )
        if tamper == 'missing':
            payload['observation_set'].remove(target)
        elif tamper == 'extra':
            payload['observation_set'].append({**target, 'row_identity_hmac': 'f' * 64})
            payload['observation_set'].sort(
                key=lambda row: (row['row_kind'], row['row_identity_hmac'])
            )
        else:
            target['row_projection_hmac'] = 'f' * 64
        with pytest.raises(RagReleaseLedgerError, match='matrix|observation'):
            harness.ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=harness.database_identity,
            )
    assert harness.authority.marker_path.read_bytes() == before_marker
    assert harness.records('authorization')[0] == auth


def test_prior_runtime_projection_changes_terminal_transition_digest(harness):
    from sqlalchemy import update

    from backend.app.models.agent_runs import AgentRun

    harness.claim()
    harness.fail_case(outcome='model_unavailable')
    auth = harness.records('authorization')[0]
    changes = [('authorization', auth, {**auth, 'state': 'aborted_execution_crash'})]
    with harness.engine.begin() as connection:
        first, _mutations = harness.prepare(
            connection,
            'authorization_abort_execution_crash',
            changes,
            outcome='abandoned_unknown',
            payload_overrides={'execution_crash_attestation_hmac': 'a' * 64},
        )
    with harness.engine.begin() as connection:
        connection.execute(
            update(AgentRun.__table__)
            .where(AgentRun.id == 41)
            .values(metadata={'fixture_revision': 2})
        )
    with harness.engine.begin() as connection:
        second, mutations = harness.prepare(
            connection,
            'authorization_abort_execution_crash',
            changes,
            outcome='abandoned_unknown',
            payload_overrides={'execution_crash_attestation_hmac': 'a' * 64},
        )
        assert (
            validate_transition_payload(
                first, identity_secret=_SECRET
            ).transition_digest
            != validate_transition_payload(
                second, identity_secret=_SECRET
            ).transition_digest
        )
        harness.ledger.append(
            connection,
            second,
            actual_mutations=mutations,
            database_identity=harness.database_identity,
        )
    assert b'fixture_revision' not in harness.authority.marker_path.read_bytes()


def test_wrong_provider_observation_is_rejected_before_any_mutation_sql(harness):
    from sqlalchemy import event

    harness.claim()
    harness.fail_case(outcome='model_unavailable')
    auth = harness.records('authorization')[0]
    with harness.engine.begin() as connection:
        payload, mutations = harness.prepare(
            connection,
            'authorization_abort_execution_crash',
            [('authorization', auth, {**auth, 'state': 'aborted_execution_crash'})],
            outcome='abandoned_unknown',
            payload_overrides={'execution_crash_attestation_hmac': 'a' * 64},
        )
        next(
            row
            for row in payload['observation_set']
            if row['row_kind'] == 'provider_safety_authority'
        )['row_projection_hmac'] = 'f' * 64
        writes = []

        def record_write(
            _connection, _cursor, statement, _parameters, _context, _executemany
        ):
            if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                writes.append(statement.split()[0])

        event.listen(harness.engine, 'before_cursor_execute', record_write)
        try:
            with pytest.raises(RagReleaseLedgerError, match='observation'):
                harness.ledger.append(
                    connection,
                    payload,
                    actual_mutations=mutations,
                    database_identity=harness.database_identity,
                )
        finally:
            event.remove(harness.engine, 'before_cursor_execute', record_write)
        assert writes == []
