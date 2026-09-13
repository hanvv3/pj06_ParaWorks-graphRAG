"""Task24 carryover: a signature over caller INSERT values is not approval."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import event, text, update

from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent, RagProviderReadiness
from backend.app.rag.release_ledger import RagReleaseLedgerError
from backend.tests.test_rag_release_ledger_review_q import (
    _release_test_seam,  # noqa: F401
)
from backend.tests.test_rag_release_ledger_round4 import harness as harness


def claim_changes(harness, monkeypatch):
    """Get a legitimate plan, then let the attacker re-sign different literals."""
    captured = []
    with monkeypatch.context() as patch:
        patch.setattr(
            harness,
            'append',
            lambda kind, changes, **kwargs: captured.append((kind, changes, kwargs)),
        )
        harness.claim()
    return captured[0]


def forged(value):
    if isinstance(value, datetime):
        return value + timedelta(seconds=1)
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float, Decimal)):
        return value + 1
    if isinstance(value, dict):
        return {'unreviewed_metadata': 'synthetic-test-only'}
    if isinstance(value, str) and len(value) == 64:
        return '9' * 64
    return 'public' if value == 'restricted' or value == 'internal' else 'forged'


def assert_rejected_before_write(harness, kind, changes, kwargs):
    before = {
        item: harness.records(item)
        for item in (
            'authorization',
            'case',
            'dispatch',
            'agent_run',
            'cost_component',
            'provider_safety_authority',
            'provider_readiness',
            'quality_report',
        )
    }
    marker = harness.authority.marker_path.read_bytes()
    generation = harness.snapshot.generation
    writes = []

    def record(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            writes.append(statement)

    event.listen(harness.engine, 'before_cursor_execute', record)
    try:
        with pytest.raises(RagReleaseLedgerError):
            harness.append(kind, changes, **kwargs)
    finally:
        event.remove(harness.engine, 'before_cursor_execute', record)
    assert writes == []
    assert harness.authority.marker_path.read_bytes() == marker
    assert harness.snapshot.generation == generation
    assert {item: harness.records(item) for item in before} == before


@pytest.mark.parametrize('field', list(AgentRun.__table__.c.keys()))
def test_signed_initial_parent_field_must_match_approved_projection(
    harness, monkeypatch, field
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    parent = next(after for row, _, after in changes if row == 'agent_run')
    parent[field] = forged(parent[field])
    assert_rejected_before_write(harness, kind, changes, kwargs)


@pytest.mark.parametrize('component', ['query_embedding', 'answer_generation'])
@pytest.mark.parametrize('field', list(AgentRunCostComponent.__table__.c.keys()))
def test_signed_initial_child_field_must_match_approved_projection(
    harness, monkeypatch, component, field
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    child = next(
        after
        for row, _, after in changes
        if row == 'cost_component' and after['component'] == component
    )
    child[field] = forged(child[field])
    assert_rejected_before_write(harness, kind, changes, kwargs)


@pytest.mark.parametrize('roster', ['missing', 'extra', 'duplicate', 'reordered'])
def test_initial_children_require_exact_approved_order_and_membership(
    harness, monkeypatch, roster
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    if roster == 'missing':
        changes.pop()
    elif roster == 'duplicate':
        changes.append(changes[-1])
    elif roster == 'extra':
        extra = {**changes[-1][2], 'component': 'unreviewed', 'id': 999}
        changes.append(('cost_component', None, extra))
    else:
        changes[-2:] = changes[-2:][::-1]
    assert_rejected_before_write(harness, kind, changes, kwargs)


def test_valid_exact_initial_projection_commits_once(harness):
    harness.claim()
    assert harness.snapshot.generation == 2
    assert len(harness.records('agent_run')) == 1
    assert len(harness.records('cost_component')) == 2
    assert harness.records('authorization')[0]['case_claim_count'] == 1


@pytest.mark.parametrize(
    'index,field',
    [(2, 'total_charged_cost_usd'), (3, 'charged_cost_usd'), (4, 'reserved_cost_usd')],
)
def test_insert_cannot_hide_subprecision_cost_in_canonical_rounding(
    harness, monkeypatch, index, field
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    changes[index][2][field] += Decimal('0.0000001')
    assert_rejected_before_write(harness, kind, changes, kwargs)


def test_manifest_ordinal_mismatch_rejects_before_case_insert(harness, monkeypatch):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    changes[1][2]['manifest_ordinal'] = 1
    assert_rejected_before_write(harness, kind, changes, kwargs)


@pytest.mark.parametrize('capability', ['missing', 'foreign', 'unissued'])
def test_projection_requires_issued_approved_manifest_authority(
    harness, monkeypatch, capability
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    projection = kwargs['approved_case_claim']
    kwargs['approved_case_claim'] = (
        None
        if capability == 'missing'
        else object()
        if capability == 'foreign'
        else object.__new__(type(projection))
    )
    assert_rejected_before_write(harness, kind, changes, kwargs)


@pytest.mark.parametrize('component', ['query_embedding', 'answer_generation'])
@pytest.mark.parametrize(
    'field,value',
    [
        ('authorized_model_config_snapshot_hmac', '9' * 64),
        ('authorized_policy_snapshot_hmac', '9' * 64),
        ('reasoning_or_config_identity', 'unreviewed'),
        ('state', 'rebind_required'),
        ('active', False),
    ],
)
def test_fresh_readiness_must_match_frozen_manifest_before_sql(
    harness, monkeypatch, component, field, value
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    with harness.engine.begin() as connection:
        connection.execute(
            update(RagProviderReadiness.__table__)
            .where(RagProviderReadiness.component == component)
            .values(**{field: value})
        )
    assert_rejected_before_write(harness, kind, changes, kwargs)


@pytest.mark.parametrize(
    'change', ['version', 'provider', 'tokens', 'case_id', 'duplicate', 'reordered']
)
def test_resigned_manifest_does_not_replace_locked_reviewed_manifest(
    harness, monkeypatch, change
):
    from dataclasses import replace

    from backend.app.rag.release_review import prepare_case_claim_projection

    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    manifest = harness.manifest
    case = manifest.cases[0]
    if change == 'version':
        manifest = replace(manifest, fixture_manifest_version='unreviewed:v2')
    elif change == 'reordered':
        manifest = replace(manifest, cases=manifest.cases[::-1])
    else:
        if change == 'provider':
            child = case.components[1]
            child = replace(
                child,
                policy=replace(child.policy, authorized_policy_snapshot_hmac='9' * 64),
            )
            case = replace(case, components=(case.components[0], child))
        elif change == 'tokens':
            case = replace(
                case,
                components=(
                    case.components[0],
                    replace(case.components[1], reserved_input_tokens=11),
                ),
            )
        elif change == 'case_id':
            case = replace(case, case_id_hmac='9' * 64)
        else:
            case = replace(case, case_id_hmac=manifest.cases[1].case_id_hmac)
        manifest = replace(manifest, cases=(case, *manifest.cases[1:]))
    with harness.engine.connect() as connection:
        payload, _ = harness.prepare(connection, kind, changes, case=kwargs['case'])
        with pytest.raises(RagReleaseLedgerError):
            prepare_case_claim_projection(
                connection,
                manifest=manifest,
                payload=payload,
                identity_secret=harness.secret,
            )
    assert harness.records('agent_run') == []
    assert harness.snapshot.generation == 1


@pytest.mark.parametrize('harness', [(), ('0.001000',)], indirect=True)
def test_exact_projection_uses_runtime_admission_and_reviewed_provider_inputs(harness):
    query = harness.manifest.cases[0].components[0].reserved_cost_usd
    harness.claim(query_reserve=format(query, '.6f'))
    parent = harness.records('agent_run')[0]
    assert {
        key: parent[key]
        for key in (
            'agent_name',
            'prompt_version',
            'permission_level',
            'model_name',
            'generation_provider',
            'generation_reasoning_effort',
            'generation_route_version',
            'generation_output_contract_version',
            'workflow_thread_id',
            'effect_key',
            'completed_at',
            'projection_owner_fence_hmac',
            'input_tokens',
            'output_tokens',
            'total_tokens',
        )
    } == {
        'agent_name': 'rag_orchestrator_agent',
        'prompt_version': 'rag-answer:v2',
        'permission_level': 'restricted',
        'model_name': 'rag-v2-admission',
        'generation_provider': None,
        'generation_reasoning_effort': None,
        'generation_route_version': None,
        'generation_output_contract_version': None,
        'workflow_thread_id': None,
        'effect_key': None,
        'completed_at': None,
        'projection_owner_fence_hmac': None,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
    }
    children = harness.records('cost_component')
    assert [
        (row['component'], row['component_ordinal'], row['dispatch_state'])
        for row in children
    ] == [
        ('query_embedding', 0, 'not_attempted' if query else 'terminal'),
        ('answer_generation', 1, 'not_attempted'),
    ]
    assert children[1]['reserved_input_tokens'] == 10
    assert children[1]['reserved_output_tokens'] == 10
    assert children[1]['reserved_cost_usd'] == Decimal('0.001000')
    assert all(
        row['authorized_model_config_snapshot_hmac'] == '1' * 64
        and row['authorized_policy_snapshot_hmac'] == '2' * 64
        for row in children
    )


def test_actual_sql_after_image_is_rechecked_before_marker_commit(harness, monkeypatch):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    marker = harness.authority.marker_path.read_bytes()
    with harness.engine.begin() as connection:
        connection.execute(
            text(
                'CREATE TRIGGER tamper_initial_parent AFTER INSERT ON agent_runs '
                "BEGIN UPDATE agent_runs SET permission_level = 'public' WHERE id = NEW.id; END"
            )
        )
    with pytest.raises(RagReleaseLedgerError):
        harness.append(kind, changes, **kwargs)
    assert harness.records('agent_run') == []
    assert harness.records('cost_component') == []
    assert harness.records('case') == []
    assert harness.authority.marker_path.read_bytes() == marker
    assert harness.snapshot.generation == 1


def test_prepared_projection_cannot_be_replayed_at_next_generation(
    harness, monkeypatch
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    harness.append(kind, changes, **kwargs)
    assert_rejected_before_write(harness, kind, changes, kwargs)


def test_runtime_allocation_rejects_occupied_child_id_before_sql(harness, monkeypatch):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    from sqlalchemy import insert

    child = {**changes[3][2], 'agent_run_id': 999}
    with harness.engine.begin() as connection:
        connection.execute(insert(AgentRunCostComponent.__table__).values(**child))
    assert_rejected_before_write(harness, kind, changes, kwargs)
