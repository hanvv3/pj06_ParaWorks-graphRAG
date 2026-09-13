from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, insert

from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables
from backend.tests.test_rag_release_ledger import (
    _SECRET,
    _authority,
    _bootstrap_payload,
    _identity,
    _TestProviderPeer,
)


@pytest.fixture(autouse=True)
def _release_test_seam(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Mirror Task 23's deterministic non-product barrier seam."""
    from contextlib import contextmanager

    from backend.app.admin import rag_provider_safety
    from backend.app.rag import release_authority

    if request.node.name == 'test_task22_release_peer_issues_sealed_incident_plan':
        return

    monkeypatch.setattr(
        rag_provider_safety, 'RagProviderSafetyReleasePeer', _TestProviderPeer
    )
    monkeypatch.setattr(
        release_authority,
        'assert_rag_release_physical_contract',
        lambda _connection: None,
    )

    @contextmanager
    def barrier(_self, connection, *, marker):
        with marker.locked():
            yield release_authority._RagReleaseBarrierGuard(
                connection, seal=release_authority._RELEASE_BARRIER_SEAL
            )

    monkeypatch.setattr(
        release_authority.RagReleaseAuthority, '_authority_barrier', barrier
    )
    monkeypatch.setattr(
        release_authority.RagReleaseAuthority,
        '_current_database_identity',
        staticmethod(lambda _connection, supplied: supplied),
    )


def _agent_run_values(run_id: int = 41) -> dict[str, object]:
    return {
        'id': run_id,
        'agent_name': 'rag_orchestrator',
        'prompt_version': 'rag-answer:v2',
        'status': 'running',
        'source_window': 'release-case',
        'cache_key': f'release-case-{run_id}',
        'model_name': 'gpt-5.4-mini-2026-03-17',
        'generation_provider': 'openai',
        'generation_reasoning_effort': 'low',
        'generation_route_version': 'rag-route:v2',
        'generation_output_contract_version': 'rag-answer:v2',
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'estimated_cost_usd': 0.0,
        'permission_level': 'internal',
        'metadata': {},
        'workflow_thread_id': f'release-case-{run_id}',
        'effect_key': f'release-case-{run_id}',
        'run_contract_version': 'rag-run:v2',
        'run_record_phase': 'admission',
        'total_charged_cost_usd': Decimal('0.000000'),
        'projection_owner_fence_hmac': None,
        'started_at': datetime.now(UTC),
        'completed_at': None,
    }


def test_agent_run_logical_identity_maps_only_to_physical_id() -> None:
    """Catches snapshot code looking for a nonexistent agent_run_id column."""
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        RagReleaseLedgerError,
        ReleaseRowPrimaryKey,
        release_row_identity_hmac,
    )

    engine = create_engine('sqlite+pysqlite:///:memory:')
    AgentRun.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(insert(AgentRun.__table__).values(**_agent_run_values()))
        snapshot = RagReleaseLedger.mutation_set(connection)._snapshot(
            connection,
            ReleaseRowPrimaryKey('agent_run', {'agent_run_id': 41}),
        )
    assert snapshot is not None and snapshot['id'] == 41
    with pytest.raises(RagReleaseLedgerError, match='primary key'):
        release_row_identity_hmac('agent_run', {'id': 41}, identity_secret=_SECRET)


def test_bootstrap_rejects_authorization_identity_different_from_payload(
    tmp_path: Path,
) -> None:
    """Catches an authorization row attesting a different reviewed approval."""
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
            connection, database_identity=_identity(), **{
                'review_envelope_hmac': '1' * 64,
                'review_nonce_hmac': '2' * 64,
            }
        )
    payload = _bootstrap_payload(snapshot)
    tables = release_tables(build_rag_release_metadata())
    key = {
        'ledger_uuid': payload['ledger_uuid'],
        'ledger_epoch': payload['ledger_epoch'],
        'approval_id_hmac': payload['approval_id_hmac'],
    }
    with engine.begin() as connection:
        mutations = ledger.mutation_set(connection)
        mutations.plan(
            insert(tables.authorizations).values(
                **key,
                approval_hmac='9' * 64,
                base_generation=payload['from_generation'],
                state='unused',
                approved_corpus_snapshot_hmac='a' * 64,
                approved_provider_safety_snapshot_hmac='b' * 64,
                provider_safety_envelope_digest='c' * 64,
                validation_database_identity_hmac='d' * 64,
                manifest_hmac='e' * 64,
                baseline_hmac='f' * 64,
                reviewer_roster_hmac='0' * 64,
                execution_process_instance_hmac=None,
                execution_runner_fence_hmac=None,
                case_claim_count=0,
                embedding_dispatch_count=0,
                generation_dispatch_count=0,
                total_dispatch_count=0,
                reserved_cost_usd='0.000000',
                charged_cost_usd='0.000000',
            ),
            ReleaseRowPrimaryKey('authorization', key),
        )
        with pytest.raises(RagReleaseLedgerError, match='authorization.*identity'):
            ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=_identity(),
            )


def _case_claim_projection(mutations, payload: dict[str, object]) -> None:
    run = _agent_run_values()
    authorization = {
        'ledger_uuid': payload['ledger_uuid'],
        'ledger_epoch': payload['ledger_epoch'],
        'approval_id_hmac': payload['approval_id_hmac'],
        'approval_hmac': payload['approval_hmac'],
        'base_generation': 0,
        'state': 'started',
        'approved_corpus_snapshot_hmac': payload['approved_corpus_snapshot_hmac'],
        'approved_provider_safety_snapshot_hmac': payload[
            'approved_provider_safety_snapshot_hmac'
        ],
        'provider_safety_envelope_digest': payload['provider_safety_envelope_digest'],
        'validation_database_identity_hmac': payload[
            'validation_database_identity_hmac'
        ],
        'manifest_hmac': 'a' * 64,
        'baseline_hmac': 'b' * 64,
        'reviewer_roster_hmac': 'c' * 64,
        'execution_process_instance_hmac': payload['execution_process_instance_hmac'],
        'execution_runner_fence_hmac': payload['execution_runner_fence_hmac'],
        'case_claim_count': 1,
        'embedding_dispatch_count': 0,
        'generation_dispatch_count': 0,
        'total_dispatch_count': 0,
        'reserved_cost_usd': Decimal('0.000000'),
        'charged_cost_usd': Decimal('0.000000'),
    }
    authorization_before = {
        **authorization,
        'state': 'unused',
        'execution_process_instance_hmac': None,
        'execution_runner_fence_hmac': None,
        'case_claim_count': 0,
    }
    case = {
        'ledger_uuid': payload['ledger_uuid'],
        'ledger_epoch': payload['ledger_epoch'],
        'approval_id_hmac': payload['approval_id_hmac'],
        'case_id_hmac': payload['case_id_hmac'],
        'manifest_ordinal': 0,
        'case_projection_hmac': None,
        'state': 'claimed',
        'runtime_agent_run_id_hmac': payload['runtime_agent_run_id_hmac'],
        'embedding_reserved_cost_usd': Decimal('0.000000'),
        'generation_reserved_cost_usd': Decimal('0.000000'),
        'total_reserved_cost_usd': Decimal('0.000000'),
    }
    components = []
    for ordinal, component in enumerate(('query_embedding', 'answer_generation')):
        components.append(
            {
                'id': ordinal + 1,
                'agent_run_id': 41,
                'component': component,
                'component_ordinal': ordinal,
                'dispatch_state': 'terminal',
                'dispatch_fence_hmac': None,
                'process_instance_hmac': None,
                'attempted': False,
                'dispatch_count': 0,
                'reserved_input_tokens': 0,
                'reserved_output_tokens': 0,
                'actual_input_tokens': None,
                'actual_output_tokens': None,
                'reserved_cost_usd': Decimal('0.000000'),
                'charged_cost_usd': Decimal('0.000000'),
                'charge_basis': 'zero',
                'overrun': False,
                'provider': 'openai',
                'model': (
                    'text-embedding-3-small'
                    if component == 'query_embedding'
                    else 'gpt-5.4-mini-2026-03-17'
                ),
                'authorized_model_config_version': (
                    'rag-query-embedding-config:v1'
                    if component == 'query_embedding'
                    else 'rag-answer-model-config:v1'
                ),
                'authorized_model_config_snapshot_hmac': '1' * 64,
                'authorized_cost_policy_version': (
                    'rag-query-embedding-cost:v1'
                    if component == 'query_embedding'
                    else 'rag-answer-cost:v1'
                ),
                'authorized_token_estimator_version': (
                    'openai-cl100k-text-embedding-3-small:v1'
                    if component == 'query_embedding'
                    else 'openai-o200k-rag-answer:v1'
                ),
                'authorized_policy_snapshot_hmac': '2' * 64,
                'terminal_outcome': None,
                'created_at': datetime.now(UTC),
                'updated_at': datetime.now(UTC),
            }
        )
    mutations._rows = [
        type('Row', (), {'row_kind': 'authorization'})(),
        type('Row', (), {'row_kind': 'case'})(),
        type('Row', (), {'row_kind': 'agent_run'})(),
        type('Row', (), {'row_kind': 'cost_component'})(),
        type('Row', (), {'row_kind': 'cost_component'})(),
    ]
    mutations._before_snapshots = [authorization_before, None, None, None, None]
    mutations._after_snapshots = [authorization, case, run, *components]


def test_case_claim_rejects_two_terminal_zero_children(tmp_path: Path) -> None:
    """Catches case admission accepting no later-admissible component."""
    from backend.app.agent_runtime.rag_safety_identity import rag_identity_hmac
    from backend.app.rag.release_ledger import RagReleaseLedger, RagReleaseLedgerError

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **{
                'review_envelope_hmac': '1' * 64,
                'review_nonce_hmac': '2' * 64,
            }
        )
        mutations = ledger.mutation_set(connection)
    payload = _bootstrap_payload(snapshot)
    payload.update(
        transition_kind='case_claim',
        authorization_state_before='unused',
        authorization_state_after='started',
        case_claim_count=1,
        case_id_hmac='3' * 64,
        case_projection_hmac=None,
        case_state_before=None,
        case_state_after='claimed',
        case_embedding_reserved_cost_usd='0.000000',
        case_generation_reserved_cost_usd='0.000000',
        case_total_reserved_cost_usd='0.000000',
        execution_process_instance_hmac='d' * 64,
        execution_runner_fence_hmac='e' * 64,
        runtime_agent_run_id_hmac=rag_identity_hmac(
            {'agent_run_id': 41},
            secret=_SECRET,
            schema_version='rag-runtime-agent-run-id:v1',
            policy_version='rag-run:v2',
        ),
    )
    _case_claim_projection(mutations, payload)
    with pytest.raises(RagReleaseLedgerError, match='case claim runtime'):
        mutations.assert_payload_projection(payload, identity_secret=_SECRET)


def _not_attempted_component(run_id: int, component: str) -> dict[str, object]:
    query = component == 'query_embedding'
    return {
        'agent_run_id': run_id,
        'component': component,
        'component_ordinal': 0 if query else 1,
        'dispatch_state': 'not_attempted',
        'dispatch_fence_hmac': None,
        'process_instance_hmac': None,
        'attempted': False,
        'dispatch_count': 0,
        'reserved_input_tokens': 0,
        'reserved_output_tokens': 0,
        'actual_input_tokens': None,
        'actual_output_tokens': None,
        'reserved_cost_usd': Decimal('0.000000'),
        'charged_cost_usd': Decimal('0.000000'),
        'charge_basis': 'zero',
        'overrun': False,
        'provider': 'openai',
        'model': (
            'text-embedding-3-small' if query else 'gpt-5.4-mini-2026-03-17'
        ),
        'authorized_model_config_version': (
            'rag-query-embedding-config:v1'
            if query
            else 'rag-answer-model-config:v1'
        ),
        'authorized_model_config_snapshot_hmac': '1' * 64,
        'authorized_cost_policy_version': (
            'rag-query-embedding-cost:v1' if query else 'rag-answer-cost:v1'
        ),
        'authorized_token_estimator_version': (
            'openai-cl100k-text-embedding-3-small:v1'
            if query
            else 'openai-o200k-rag-answer:v1'
        ),
        'authorized_policy_snapshot_hmac': '2' * 64,
        'terminal_outcome': None,
    }


def test_case_claim_appends_with_logical_run_identity_and_exact_runtime_roster(
    tmp_path: Path,
) -> None:
    """Catches the logical/physical PK bridge failing in a real append path."""
    from sqlalchemy import update

    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        ReleaseRowPrimaryKey,
    )
    from backend.tests.test_rag_release_ledger import (
        _capture_bootstrap_authorization,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        AgentRun.__table__.create(connection)
        AgentRunCostComponent.__table__.create(connection)
        initial = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    bootstrap = _bootstrap_payload(initial)
    with engine.begin() as connection:
        first = ledger.append(
            connection,
            bootstrap,
            actual_mutations=_capture_bootstrap_authorization(
                connection, ledger, bootstrap
            ),
            database_identity=_identity(),
        )
    payload = _valid_payload_for_kind(first, 'case_claim')
    tables = release_tables(build_rag_release_metadata())
    common = {
        'ledger_uuid': payload['ledger_uuid'],
        'ledger_epoch': payload['ledger_epoch'],
        'approval_id_hmac': payload['approval_id_hmac'],
    }
    with engine.begin() as connection:
        mutations = ledger.mutation_set(connection)
        mutations.plan(
            update(tables.authorizations)
            .where(
                tables.authorizations.c.ledger_uuid == payload['ledger_uuid'],
                tables.authorizations.c.ledger_epoch == payload['ledger_epoch'],
                tables.authorizations.c.approval_id_hmac
                == payload['approval_id_hmac'],
            )
            .values(
                state='started',
                execution_process_instance_hmac=payload[
                    'execution_process_instance_hmac'
                ],
                execution_runner_fence_hmac=payload['execution_runner_fence_hmac'],
                case_claim_count=1,
            ),
            ReleaseRowPrimaryKey('authorization', common),
        )
        case_key = {**common, 'case_id_hmac': payload['case_id_hmac']}
        mutations.plan(
            insert(tables.cases).values(
                **case_key,
                manifest_ordinal=0,
                case_projection_hmac=None,
                state='claimed',
                runtime_agent_run_id_hmac=payload['runtime_agent_run_id_hmac'],
                embedding_reserved_cost_usd='0.000000',
                generation_reserved_cost_usd='0.000000',
                total_reserved_cost_usd='0.000000',
            ),
            ReleaseRowPrimaryKey('case', case_key),
        )
        mutations.plan(
            insert(AgentRun.__table__).values(**_agent_run_values()),
            ReleaseRowPrimaryKey('agent_run', {'agent_run_id': 41}),
        )
        for component in ('query_embedding', 'answer_generation'):
            mutations.plan(
                insert(AgentRunCostComponent.__table__).values(
                    **_not_attempted_component(41, component)
                ),
                ReleaseRowPrimaryKey(
                    'cost_component',
                    {'agent_run_id': 41, 'component': component},
                ),
            )
        advanced = ledger.append(
            connection,
            payload,
            actual_mutations=mutations,
            database_identity=_identity(),
        )
    assert advanced.generation == 2
    with engine.connect() as connection:
        assert connection.scalar(
            tables.cases.select()
            .with_only_columns(tables.cases.c.manifest_ordinal)
        ) == 0
        assert connection.scalar(
            AgentRun.__table__.select().with_only_columns(AgentRun.id)
        ) == 41


def test_task22_release_peer_issues_sealed_incident_plan(tmp_path: Path) -> None:
    """Catches release code falling back to self-reported provider SQL updates."""
    from backend.app.admin.rag_provider_safety import (
        ProviderSafetyReviewError,
        RagProviderSafetyIncidentPlan,
    )
    from backend.tests.test_rag_provider_safety_admin import (
        _review_bytes,
        _service,
    )

    engine, target, _runtime, admin = _service(tmp_path, kind='live_validation')
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    peer = admin.release_peer()
    with engine.connect() as connection:
        plan = peer.prepare_incident(
            connection,
            component='query_embedding',
            category='provider_usage_overrun',
            agent_run_id=41,
            input_tokens=7,
            output_tokens=0,
            cost_usd=Decimal('0.100000'),
        )
    assert type(plan) is RagProviderSafetyIncidentPlan
    assert len(plan.new_envelope_digest) == 64
    with pytest.raises(ProviderSafetyReviewError):
        RagProviderSafetyIncidentPlan(
            component='query_embedding',
            category='provider_usage_overrun',
            agent_run_id=41,
            input_tokens=7,
            output_tokens=0,
            cost_usd=Decimal('0.100000'),
            seal=object(),
        )


def test_release_append_rejects_foreign_incident_capability_before_mutation(
    tmp_path: Path,
) -> None:
    """Catches accepting caller-built objects as provider incident evidence."""
    from backend.app.rag.release_ledger import RagReleaseLedger, RagReleaseLedgerError
    from backend.tests.test_rag_release_ledger import (
        _capture_bootstrap_authorization,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    payload = _bootstrap_payload(snapshot)
    with engine.begin() as connection:
        mutations = _capture_bootstrap_authorization(connection, ledger, payload)
        with pytest.raises(RagReleaseLedgerError, match='incident capability'):
            ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=_identity(),
                provider_incident=object(),
            )


_KIND_OUTCOME = {
    'authorization_bootstrap': None,
    'case_claim': None,
    'component_claim': None,
    'component_outcome': 'component_succeeded',
    'case_failure': 'retriever_unavailable',
    'case_safe_outcome': 'no_match',
    'case_outcome': 'supported',
    'authorization_abort_control': 'provider_safety_unavailable',
    'authorization_abort_component': 'provider_usage_overrun',
    'authorization_abort_component_snapshot': 'provider_safety_unavailable',
    'authorization_abort_corpus_drift': 'live_corpus_snapshot_changed',
    'authorization_abort_execution_crash': 'abandoned_unknown',
    'authorization_abort_final': 'provider_safety_unavailable',
    'authorization_abort_snapshot': 'provider_safety_unavailable',
    'authorization_complete': 'quality_gate_green',
    'authorization_finish_failed': 'ordinary_execution_failed',
    'authorization_finish_quality_failed': 'quality_gate_failed',
}


def _valid_payload_for_kind(snapshot, kind: str) -> dict[str, object]:
    from backend.app.agent_runtime.rag_safety_identity import rag_identity_hmac
    from backend.app.rag.release_ledger import release_row_identity_hmac

    payload = _bootstrap_payload(snapshot)
    if kind == 'authorization_bootstrap':
        return payload
    payload.update(
        transition_kind=kind,
        outcome=_KIND_OUTCOME[kind],
        from_generation=1,
        to_generation=2,
        authorization_state_before='started',
        authorization_state_after='started',
        execution_process_instance_hmac='d' * 64,
        execution_runner_fence_hmac='e' * 64,
    )
    if kind == 'case_claim':
        payload['authorization_state_before'] = 'unused'
    terminal_state = {
        'authorization_abort_control': 'aborted_provider_safety',
        'authorization_abort_component': 'aborted_overrun',
        'authorization_abort_component_snapshot': 'aborted_provider_safety',
        'authorization_abort_corpus_drift': 'aborted_corpus_drift',
        'authorization_abort_execution_crash': 'aborted_execution_crash',
        'authorization_abort_final': 'aborted_provider_safety',
        'authorization_abort_snapshot': 'aborted_provider_safety',
        'authorization_complete': 'complete',
        'authorization_finish_failed': 'finished_failed',
        'authorization_finish_quality_failed': 'finished_failed',
    }.get(kind)
    if terminal_state is not None:
        payload['authorization_state_after'] = terminal_state
    if kind == 'authorization_abort_snapshot':
        payload['authorization_state_before'] = 'unused'
        payload['execution_process_instance_hmac'] = None
        payload['execution_runner_fence_hmac'] = None
    if kind == 'authorization_abort_corpus_drift':
        payload['current_corpus_snapshot_hmac'] = '9' * 64
    if kind == 'authorization_abort_execution_crash':
        payload['execution_crash_attestation_hmac'] = '0' * 64

    case_kinds = {
        'case_claim',
        'component_claim',
        'component_outcome',
        'case_failure',
        'case_safe_outcome',
        'case_outcome',
        'authorization_abort_control',
        'authorization_abort_component',
        'authorization_abort_component_snapshot',
    }
    if kind in case_kinds:
        payload.update(
            case_claim_count=1,
            case_id_hmac='3' * 64,
            case_projection_hmac=(
                '4' * 64 if kind in {'case_safe_outcome', 'case_outcome'} else None
            ),
            case_state_before=None if kind == 'case_claim' else 'claimed',
            case_state_after=(
                'claimed'
                if kind in {'case_claim', 'component_claim', 'component_outcome'}
                else 'complete'
                if kind in {'case_safe_outcome', 'case_outcome'}
                else 'failed'
            ),
            case_embedding_reserved_cost_usd='0.000000',
            case_generation_reserved_cost_usd='0.000000',
            case_total_reserved_cost_usd='0.000000',
            runtime_agent_run_id_hmac=rag_identity_hmac(
                {'agent_run_id': 41},
                secret=_SECRET,
                schema_version='rag-runtime-agent-run-id:v1',
                policy_version='rag-run:v2',
            ),
        )
    component_kinds = {
        'component_claim',
        'component_outcome',
        'authorization_abort_component',
        'authorization_abort_component_snapshot',
    }
    if kind in component_kinds:
        payload.update(
            component='query_embedding',
            dispatch_state_before=(
                'not_attempted' if kind == 'component_claim' else 'dispatching'
            ),
            dispatch_state_after=(
                'dispatching' if kind == 'component_claim' else 'terminal'
            ),
            dispatch_count_before=0 if kind == 'component_claim' else 1,
            dispatch_count_after=1,
            dispatch_fence_hmac='5' * 64,
            reserved_cost_usd='0.000000',
            charged_cost_usd='0.000000',
            charge_basis_after='reserved',
            embedding_dispatch_count=1,
            total_dispatch_count=1,
        )
    if kind in {'authorization_complete', 'authorization_finish_quality_failed'}:
        payload.update(
            case_claim_count=30,
            embedding_dispatch_count=10,
            generation_dispatch_count=30,
            total_dispatch_count=40,
            quality_report_hmac='f' * 64,
        )
    elif kind == 'authorization_finish_failed':
        payload.update(
            case_claim_count=30,
            embedding_dispatch_count=1,
            total_dispatch_count=1,
        )

    common = {
        'ledger_uuid': payload['ledger_uuid'],
        'ledger_epoch': payload['ledger_epoch'],
    }
    keys: list[tuple[str, dict[str, object]]] = [
        (
            'authorization',
            {**common, 'approval_id_hmac': payload['approval_id_hmac']},
        ),
        ('release_ledger', dict(common)),
        ('release_transition', {**common, 'to_generation': 2}),
    ]
    if payload['case_id_hmac'] is not None:
        keys.extend(
            [
                (
                    'case',
                    {
                        **common,
                        'approval_id_hmac': payload['approval_id_hmac'],
                        'case_id_hmac': payload['case_id_hmac'],
                    },
                ),
                ('agent_run', {'agent_run_id': 41}),
            ]
        )
    if payload['component'] is not None:
        keys.append(
            (
                'dispatch',
                {
                    **common,
                    'approval_id_hmac': payload['approval_id_hmac'],
                    'case_id_hmac': payload['case_id_hmac'],
                    'component': payload['component'],
                },
            )
        )
    cost_count = {
        'case_claim': 2,
        'component_claim': 1,
        'component_outcome': 1,
        'case_failure': 1,
        'case_safe_outcome': 1,
        'authorization_abort_control': 1,
        'authorization_abort_component': 1,
        'authorization_abort_component_snapshot': 1,
    }.get(kind, 0)
    for component in ('query_embedding', 'answer_generation')[:cost_count]:
        keys.append(
            ('cost_component', {'agent_run_id': 41, 'component': component})
        )
    if kind == 'authorization_abort_component':
        keys.extend(
            [
                (
                    'provider_safety_authority',
                    {'authority_uuid': '33333333-3333-3333-3333-333333333333'},
                ),
                (
                    'provider_readiness',
                    {
                        'component': 'query_embedding',
                        'provider_bytes': 'openai',
                        'model_bytes': 'text-embedding-3-small',
                        'reasoning_or_config_identity_bytes': 'dimensions:1536',
                    },
                ),
            ]
        )
    if payload['quality_report_hmac'] is not None:
        keys.append(
            (
                'quality_report',
                {**common, 'approval_id_hmac': payload['approval_id_hmac']},
            )
        )
    payload['affected_rows'] = sorted(
        (
            {
                'row_kind': row_kind,
                'row_identity_hmac': release_row_identity_hmac(
                    row_kind, key, identity_secret=_SECRET
                ),
            }
            for row_kind, key in keys
        ),
        key=lambda item: (item['row_kind'], item['row_identity_hmac']),
    )
    observation_kinds = set()
    if kind in {
        'component_outcome',
        'case_failure',
        'case_safe_outcome',
        'case_outcome',
    }:
        observation_kinds.add('authorization')
    if kind in {'component_claim', 'component_outcome'}:
        observation_kinds.update({'case', 'agent_run'})
    observations = []
    for row in list(payload['affected_rows']):
        if row['row_kind'] in observation_kinds:
            payload['affected_rows'].remove(row)
            observations.append({**row, 'row_projection_hmac': 'a' * 64})
    provider_observation_kinds = {
        'authorization_abort_control',
        'authorization_abort_component_snapshot',
        'authorization_abort_corpus_drift',
        'authorization_abort_execution_crash',
        'authorization_abort_final',
        'authorization_abort_snapshot',
        'authorization_complete',
        'authorization_finish_failed',
        'authorization_finish_quality_failed',
    }
    if kind in provider_observation_kinds:
        provider_keys = [
            (
                'provider_safety_authority',
                {'authority_uuid': '33333333-3333-3333-3333-333333333333'},
            )
        ]
        if kind == 'authorization_abort_component_snapshot':
            provider_keys.append(
                (
                    'provider_readiness',
                    {
                        'component': 'query_embedding',
                        'provider_bytes': 'openai',
                        'model_bytes': 'text-embedding-3-small',
                        'reasoning_or_config_identity_bytes': 'dimensions:1536',
                    },
                )
            )
        observations.extend(
            {
                'row_kind': row_kind,
                'row_identity_hmac': release_row_identity_hmac(
                    row_kind, key, identity_secret=_SECRET
                ),
                'row_projection_hmac': 'b' * 64,
            }
            for row_kind, key in provider_keys
        )
    payload['observation_set'] = sorted(
        observations,
        key=lambda item: (
            item['row_kind'],
            item['row_identity_hmac'],
            item['row_projection_hmac'],
        ),
    )
    return payload


@pytest.mark.parametrize('kind', tuple(_KIND_OUTCOME))
def test_every_frozen_transition_kind_has_a_minimal_positive_payload(
    tmp_path: Path, kind: str
) -> None:
    """Catches a frozen transition kind becoming internally unrepresentable."""
    from backend.app.rag.release_ledger import _OUTCOMES, validate_transition_payload

    assert set(_KIND_OUTCOME) == set(_OUTCOMES)
    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    validated = validate_transition_payload(
        _valid_payload_for_kind(snapshot, kind), identity_secret=_SECRET
    )
    assert validated.transition_kind == kind


@pytest.mark.parametrize('kind', tuple(_KIND_OUTCOME))
def test_every_frozen_transition_kind_rejects_wrong_authorization_after_state(
    tmp_path: Path, kind: str
) -> None:
    """Catches a per-kind authorization state transition becoming permissive."""
    from backend.app.rag.release_ledger import (
        RagReleaseLedgerError,
        validate_transition_payload,
    )

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    payload = _valid_payload_for_kind(snapshot, kind)
    payload['authorization_state_after'] = (
        'started' if kind == 'authorization_bootstrap' else 'unused'
    )
    with pytest.raises(RagReleaseLedgerError, match='matrix'):
        validate_transition_payload(payload, identity_secret=_SECRET)


def test_observation_projection_hmac_is_typed_and_changes_with_projection() -> None:
    """Catches read-only peer evidence being identity-only or unkeyed."""
    from backend.app.rag.release_ledger import release_observation_projection_hmac

    first = {
        'ledger_uuid': '11111111-1111-1111-1111-111111111111',
        'ledger_epoch': 1,
        'approval_id_hmac': '1' * 64,
        'approval_hmac': '2' * 64,
        'state': 'started',
        'case_claim_count': 1,
    }
    changed = {**first, 'case_claim_count': 2}
    first_hmac = release_observation_projection_hmac(
        'authorization', first, identity_secret=_SECRET
    )
    changed_hmac = release_observation_projection_hmac(
        'authorization', changed, identity_secret=_SECRET
    )
    assert len(first_hmac) == 64
    assert first_hmac != changed_hmac
    assert first_hmac != release_observation_projection_hmac(
        'case', first, identity_secret=_SECRET
    )
    assert len(
        release_observation_projection_hmac(
            'agent_run', _agent_run_values(), identity_secret=_SECRET
        )
    ) == 64


def test_case_outcome_observes_unchanged_authorization_and_binds_digest(
    tmp_path: Path,
) -> None:
    """The unchanged authorization is evidence, never an affected no-op row."""
    from backend.app.rag.release_ledger import (
        RagReleaseLedgerError,
        validate_transition_payload,
    )

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    payload = _valid_payload_for_kind(snapshot, 'case_outcome')
    first = validate_transition_payload(payload, identity_secret=_SECRET)
    changed = deepcopy(payload)
    changed['observation_set'][0]['row_projection_hmac'] = 'b' * 64
    second = validate_transition_payload(changed, identity_secret=_SECRET)
    assert first.transition_digest != second.transition_digest
    missing = deepcopy(payload)
    missing['observation_set'] = []
    with pytest.raises(RagReleaseLedgerError, match='identity.*unbound|observation'):
        validate_transition_payload(missing, identity_secret=_SECRET)


@pytest.mark.parametrize('kind', ['component_claim', 'component_outcome'])
def test_component_transition_observes_unchanged_case_and_parent(
    tmp_path: Path, kind: str
) -> None:
    """Component work binds its parent peers without pretending to mutate them."""
    from backend.app.rag.release_ledger import validate_transition_payload

    authority = _authority(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    payload = _valid_payload_for_kind(snapshot, kind)
    validated = validate_transition_payload(payload, identity_secret=_SECRET)
    expected = {'case', 'agent_run'}
    if kind == 'component_outcome':
        expected.add('authorization')
    assert {item[0] for item in validated.observation_set} == expected


def test_observed_peer_is_locked_under_barrier_and_never_counted_as_affected(
    tmp_path: Path,
) -> None:
    """Catches TOCTOU and no-op peers leaking into semantic affected rows."""
    from sqlalchemy import update

    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        RagReleaseLedgerError,
        ReleaseRowPrimaryKey,
        release_row_identity_hmac,
    )
    from backend.tests.test_rag_release_ledger import (
        _capture_bootstrap_authorization,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        initial = authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    bootstrap = _bootstrap_payload(initial)
    with engine.begin() as connection:
        ledger.append(
            connection,
            bootstrap,
            actual_mutations=_capture_bootstrap_authorization(
                connection, ledger, bootstrap
            ),
            database_identity=_identity(),
        )
    tables = release_tables(build_rag_release_metadata())
    auth_key = {
        'ledger_uuid': bootstrap['ledger_uuid'],
        'ledger_epoch': bootstrap['ledger_epoch'],
        'approval_id_hmac': bootstrap['approval_id_hmac'],
    }
    report_key = dict(auth_key)
    with engine.begin() as connection:
        mutations = ledger.mutation_set(connection)
        mutations.observe(ReleaseRowPrimaryKey('authorization', auth_key))
        mutations.plan(
            insert(tables.quality_reports).values(
                **report_key,
                quality_report_hmac='f' * 64,
                manifest_hmac='a' * 64,
                baseline_hmac='b' * 64,
                reviewer_roster_hmac='c' * 64,
                payload_canonical_bytes=b'{}',
            ),
            ReleaseRowPrimaryKey('quality_report', report_key),
        )
        marker = DurableFileAuthority.open_runtime(authority.marker_path)
        with authority._authority_barrier(connection, marker=marker) as guard:
            mutations._execute_under_barrier(
                connection, authority=authority, barrier_guard=guard
            )
            assert [row.row_kind for row in mutations.rows] == ['quality_report']
            assert [row.row_kind for row in mutations.observation_rows] == [
                'authorization'
            ]
            with pytest.raises(
                RagReleaseLedgerError, match='observation set differs'
            ):
                mutations.assert_payload_projection(
                    {'observation_set': []}, identity_secret=_SECRET
                )
            with pytest.raises(
                RagReleaseLedgerError, match='observation set differs'
            ):
                mutations.assert_payload_projection(
                    {
                        'observation_set': [
                            {
                                'row_kind': 'authorization',
                                'row_identity_hmac': release_row_identity_hmac(
                                    'authorization',
                                    auth_key,
                                    identity_secret=_SECRET,
                                ),
                                'row_projection_hmac': 'f' * 64,
                            }
                        ]
                    },
                    identity_secret=_SECRET,
                )
            connection.execute(
                update(tables.authorizations)
                .where(
                    tables.authorizations.c.approval_id_hmac
                    == bootstrap['approval_id_hmac']
                )
                .values(state='aborted_provider_safety')
            )
            with pytest.raises(RagReleaseLedgerError, match='observed.*changed'):
                mutations.assert_current(connection)


def test_affected_agent_run_rejects_started_at_only_update(tmp_path: Path) -> None:
    """A clock-only write is not a semantic affected-row mutation."""
    from sqlalchemy import update

    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.app.rag.release_ledger import (
        RagReleaseLedger,
        RagReleaseLedgerError,
        ReleaseRowPrimaryKey,
    )

    authority = _authority(tmp_path)
    ledger = RagReleaseLedger(authority=authority, identity_secret=_SECRET)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    AgentRun.__table__.create(engine)
    original = _agent_run_values()
    with engine.begin() as connection:
        authority.initialize(
            connection,
            database_identity=_identity(),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    with engine.begin() as connection:
        connection.execute(insert(AgentRun.__table__).values(**original))
    with engine.begin() as connection:
        mutations = ledger.mutation_set(connection)
        mutations.plan(
            update(AgentRun.__table__)
            .where(AgentRun.__table__.c.id == 41)
            .values(started_at=original['started_at'] + timedelta(seconds=1)),
            ReleaseRowPrimaryKey('agent_run', {'agent_run_id': 41}),
        )
        marker = DurableFileAuthority.open_runtime(authority.marker_path)
        with pytest.raises(
            RagReleaseLedgerError, match='not exact'
        ), authority._authority_barrier(connection, marker=marker) as guard:
            mutations._execute_under_barrier(
                connection, authority=authority, barrier_guard=guard
            )
