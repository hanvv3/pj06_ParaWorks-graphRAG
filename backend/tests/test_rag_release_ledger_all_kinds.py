"""Every registry kind executes mutation plans and a real SQL ledger append."""

import hashlib
import hmac
import json
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from functools import cache

import pytest
from sqlalchemy import create_engine, select

from backend.app.admin.rag_provider_safety import (
    RagProviderSafetyReleasePeer as RealPeer,
)
from backend.app.rag.release_ledger import (
    RagReleaseLedger,
    RagReleaseLedgerError,
    release_row_identity_hmac,
)
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables
from backend.tests.release_ledger_fixtures import ReleaseHarness, row_key
from backend.tests.test_rag_release_ledger import _SECRET, _authority, _identity
from backend.tests.test_rag_release_ledger_review_q import (
    _KIND_OUTCOME,
)
from backend.tests.test_rag_release_ledger_review_q import (
    _release_test_seam as _release_test_seam,
)
from backend.tests.test_rag_release_ledger_round4 import (
    _provider_drift,
)
from backend.tests.test_rag_release_ledger_round4 import (
    test_case_null_abort_preserves_roster_under_current_provider_proof as drive_null_abort,
)
from backend.tests.test_rag_release_ledger_round4 import (
    test_post_call_snapshot_abort_preserves_actual_charge_and_owner as drive_snapshot_abort,
)


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode()


@pytest.fixture
def append_audit(monkeypatch):
    """Every positive has an invalid sibling and independent SQL/digest checks."""
    original = RagReleaseLedger.append
    from backend.app.rag.release_ledger import RagReleaseMutationSet

    # The immutable table registry is materialized once per test. Every read,
    # statement, row lock, mutation and SQL verification still executes normally.
    monkeypatch.setattr(
        RagReleaseMutationSet,
        '_table',
        staticmethod(cache(RagReleaseMutationSet._table)),
    )
    recorded = []

    def audited(self, connection, payload, *, actual_mutations, **kwargs):
        tables = release_tables(build_rag_release_metadata())
        old_marker = self._authority.marker_path.read_bytes()
        old_history = [
            dict(row)
            for row in connection.execute(select(tables.transitions)).mappings()
        ]
        old_ledger = dict(connection.execute(select(tables.ledgers)).mappings().one())
        before = [
            actual_mutations._snapshot(connection, plan.row)
            for plan in actual_mutations._plans
        ]
        observations = [
            actual_mutations._snapshot(connection, row)
            for row in actual_mutations.observation_rows
        ]
        invalid = deepcopy(payload)
        invalid['authorization_state_after'] = 'invalid'
        with pytest.raises(RagReleaseLedgerError):
            original(
                self, connection, invalid, actual_mutations=actual_mutations, **kwargs
            )
        assert self._authority.marker_path.read_bytes() == old_marker
        assert [
            dict(row)
            for row in connection.execute(select(tables.transitions)).mappings()
        ] == old_history
        assert (
            dict(connection.execute(select(tables.ledgers)).mappings().one())
            == old_ledger
        )
        assert [
            actual_mutations._snapshot(connection, plan.row)
            for plan in actual_mutations._plans
        ] == before
        advanced = original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )
        with connection.engine.connect() as verification:
            after = [
                actual_mutations._snapshot(verification, plan.row)
                for plan in actual_mutations._plans
            ]
            assert all(
                old != new and new is not None
                for old, new in zip(before, after, strict=True)
            )
            assert [
                actual_mutations._snapshot(verification, row)
                for row in actual_mutations.observation_rows
            ] == observations
            history = [
                dict(row)
                for row in verification.execute(
                    select(tables.transitions).order_by(tables.transitions.c.generation)
                ).mappings()
            ]
            assert history[:-1] == old_history
            assert [row['generation'] for row in history] == list(
                range(1, advanced.generation + 1)
            )
            assert history[-1]['payload_canonical_bytes'] == _bytes(payload)
            digest = hmac.new(
                self._secret,
                _bytes(
                    {
                        'domain': 'paraworks:keyed-fingerprint:v1',
                        'policy_version': 'rag-live-gate:v1',
                        'schema_version': 'rag-release-ledger-transition:v2',
                        'value': payload,
                    }
                ),
                hashlib.sha256,
            ).hexdigest()
            assert (
                history[-1]['transition_digest']
                == advanced.last_transition_digest
                == digest
            )
            ledger = dict(verification.execute(select(tables.ledgers)).mappings().one())
            assert (
                ledger['generation']
                == advanced.generation
                == old_ledger['generation'] + 1
            )
            assert ledger['last_transition_digest'] == digest
        expected = {
            (
                row.row_kind,
                release_row_identity_hmac(
                    row.row_kind, row.primary_key, identity_secret=self._secret
                ),
            )
            for row in actual_mutations.rows
        }
        signed = {
            (row['row_kind'], row['row_identity_hmac'])
            for row in payload['affected_rows']
        }
        assert signed - expected == {
            (row['row_kind'], row['row_identity_hmac'])
            for row in payload['affected_rows']
            if row['row_kind'] in {'release_ledger', 'release_transition'}
        }
        assert expected <= signed
        assert {
            (
                row.row_kind,
                release_row_identity_hmac(
                    row.row_kind, row.primary_key, identity_secret=self._secret
                ),
            )
            for row in actual_mutations.observation_rows
        } == {
            (row['row_kind'], row['row_identity_hmac'])
            for row in payload['observation_set']
        }
        assert self._authority.marker_path.read_bytes() != old_marker
        recorded.append((payload['transition_kind'], payload['outcome']))
        return advanced

    monkeypatch.setattr(RagReleaseLedger, 'append', audited)
    return recorded


def safe_case(harness):
    case, run = harness.records('case')[-1], harness.records('agent_run')[-1]
    done = {**case, 'state': 'complete', 'case_projection_hmac': '4' * 64}
    changes = [
        ('case', case, done),
        (
            'agent_run',
            run,
            {
                **run,
                'status': 'complete',
                'run_record_phase': 'final',
                'completed_at': run['started_at'] + timedelta(seconds=1),
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
    return harness.append('case_safe_outcome', changes, case=done, outcome='no_match')


def project_case(harness):
    case, run = harness.records('case')[-1], harness.records('agent_run')[-1]
    done = {**case, 'state': 'complete', 'case_projection_hmac': '4' * 64}
    return harness.append(
        'case_outcome',
        [
            ('case', case, done),
            (
                'agent_run',
                run,
                {
                    **run,
                    'status': 'complete',
                    'run_record_phase': 'final',
                    'completed_at': run['started_at'] + timedelta(seconds=1),
                },
            ),
        ],
        case=done,
        outcome='supported',
    )


def query_outcome(harness):
    auth, case = harness.records('authorization')[0], harness.records('case')[-1]
    child, dispatch = (
        harness.records('cost_component')[-2],
        harness.records('dispatch')[-1],
    )
    return harness.append(
        'component_outcome',
        [
            (
                'authorization',
                auth,
                {
                    **auth,
                    'charged_cost_usd': auth['charged_cost_usd']
                    - child['charged_cost_usd']
                    + Decimal('0.000400'),
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


def generation_outcome(harness):
    # The existing one-case helper's parent total assumes the query was impossible.
    # A real 40-dispatch lifecycle must preserve the already-charged query cost.
    auth, case, run = (
        harness.records('authorization')[0],
        harness.records('case')[-1],
        harness.records('agent_run')[-1],
    )
    child, dispatch = (
        harness.records('cost_component')[-1],
        harness.records('dispatch')[-1],
    )
    total = sum(
        item['charged_cost_usd'] for item in harness.records('cost_component')[-2:-1]
    ) + Decimal('0.000400')
    return harness.append(
        'component_outcome',
        [
            (
                'authorization',
                auth,
                {
                    **auth,
                    'charged_cost_usd': auth['charged_cost_usd']
                    - child['charged_cost_usd']
                    + Decimal('0.000400'),
                },
            ),
            (
                'agent_run',
                run,
                {
                    **run,
                    'run_record_phase': 'cost_finalized_pending_projection',
                    'projection_owner_fence_hmac': 'e' * 64,
                    'total_charged_cost_usd': total,
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
        component='answer_generation',
        outcome='component_succeeded',
    )


def finish(harness, kind, outcome):
    auth = harness.records('authorization')[0]
    changes = [
        (
            'authorization',
            auth,
            {
                **auth,
                'state': 'complete'
                if kind == 'authorization_complete'
                else 'finished_failed',
            },
        )
    ]
    overrides = {}
    if kind != 'authorization_finish_failed':
        report = {
            key: auth[key]
            for key in (
                'ledger_uuid',
                'ledger_epoch',
                'approval_id_hmac',
                'manifest_hmac',
                'baseline_hmac',
                'reviewer_roster_hmac',
            )
        }
        report.update(
            quality_report_hmac='7' * 64,
            payload_canonical_bytes=_bytes(
                {'quality_pass': outcome == 'quality_gate_green'}
            ),
        )
        changes.append(('quality_report', None, report))
        overrides['quality_report_hmac'] = '7' * 64
    return harness.append(kind, changes, outcome=outcome, payload_overrides=overrides)


def incident_harness(tmp_path, monkeypatch):
    """Real sealed Task22 service; only PostgreSQL transport/advisory is substituted."""
    from backend.app.admin import rag_provider_safety as admin_module
    from backend.app.rag import release_authority as authority_module
    from backend.tests.test_rag_provider_safety_admin import _review_bytes, _service

    monkeypatch.setattr(admin_module, 'RagProviderSafetyReleasePeer', RealPeer)
    engine, target, runtime, admin = _service(tmp_path, kind='live_validation')
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    peer = admin.release_peer()

    @contextmanager
    def provider_transport(_self, connection, **options):
        with runtime._authority.locked():
            peer._ledger.assert_pin()
            body = runtime._read_unlocked()
            runtime._match_db_whole_set(connection, body, for_update=False)
            guard = object.__new__(admin_module.RagProviderSafetyReleasePeerGuard)
            guard._provider_safety, guard._ledger = runtime, peer._ledger
            guard._body, guard._owner = body, peer
            yield guard

    def provider_revalidate(guard, connection):
        peer._assert_active_guard(guard, connection)
        peer._ledger.assert_pin()
        body = runtime._read_unlocked()
        assert body['envelope_digest'] == guard._body['envelope_digest']
        runtime._match_db_whole_set(connection, body, for_update=False)

    monkeypatch.setattr(RealPeer, '_locked_transport', provider_transport)
    monkeypatch.setattr(
        admin_module.RagProviderSafetyReleasePeerGuard,
        'revalidate_database_peer',
        provider_revalidate,
    )

    @contextmanager
    def barrier(_self, connection, *, marker):
        with peer.locked(connection) as provider_guard, marker.locked():
            yield provider_guard

    monkeypatch.setattr(
        authority_module.RagReleaseAuthority, '_authority_transport', barrier
    )
    authority = authority_module.RagReleaseAuthority(
        marker_path=tmp_path / 'release' / 'marker.json',
        provider_safety_latch_path=target.latch_path,
        identity_secret=_SECRET,
        fingerprint_key_version='v1',
        designated_environment_id='release-validation',
        designated_host_id='host-one',
        repository_roots=(),
        database_backup_roots=(),
        provider_safety_release_peer=peer,
    )
    return ReleaseHarness(engine, authority, _SECRET, _identity())


@pytest.mark.parametrize('exceptional_exit', [False, True])
def test_real_provider_and_release_guard_lifetimes(
    tmp_path, monkeypatch, exceptional_exit
):
    from backend.app.admin.rag_provider_safety import RagProviderSafetyReleasePeerGuard
    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    harness = incident_harness(tmp_path, monkeypatch)
    peer = harness.authority._provider_safety_release_peer
    other = RealPeer(
        provider_safety=peer._provider_safety, ledger=peer._ledger, seal=peer._seal
    )
    with harness.engine.connect() as connection, harness.engine.connect() as another:
        with pytest.raises(TypeError):
            RagProviderSafetyReleasePeerGuard(
                provider_safety=peer._provider_safety,
                ledger=peer._ledger,
                body={},
                seal=peer._seal,
            )
        with pytest.raises(RagProviderSafetyError):
            peer._assert_active_guard(
                object.__new__(RagProviderSafetyReleasePeerGuard), connection
            )
        try:
            with peer.locked(connection) as saved:
                peer._assert_active_guard(saved, connection)
                for owner, target in ((other, connection), (peer, another)):
                    with pytest.raises(RagProviderSafetyError):
                        owner._assert_active_guard(saved, target)
                with pytest.raises(RagProviderSafetyError), peer.locked(connection):
                    pytest.fail('nested provider lock transport was entered')
                with monkeypatch.context() as nested_patch:

                    def forbidden_transport(*args, **options):
                        pytest.fail('nested provider barrier attempted a sidecar lock')

                    nested_patch.setattr(
                        RealPeer, '_locked_transport', forbidden_transport
                    )
                    with pytest.raises(RagProviderSafetyError), peer.locked(another):
                        pytest.fail(
                            'cross-connection nested provider barrier was accepted'
                        )
                if exceptional_exit:
                    raise RuntimeError('test-only unwind')
        except RuntimeError:
            assert exceptional_exit
        with pytest.raises(RagProviderSafetyError):
            saved.revalidate_database_peer(connection)
        with peer.locked(connection) as current:
            peer._assert_active_guard(current, connection)
            with pytest.raises(RagProviderSafetyError):
                peer._assert_active_guard(saved, connection)

        @contextmanager
        def expired_provider_transport(_self, _connection, *, marker):
            yield saved

        monkeypatch.setattr(
            type(harness.authority), '_authority_transport', expired_provider_transport
        )
        with (
            pytest.raises(RagProviderSafetyError),
            harness.authority._authority_barrier(
                connection,
                marker=DurableFileAuthority.open_runtime(harness.authority.marker_path),
            ),
        ):
            pytest.fail('release accepted an expired underlying provider guard')


def incident_abort(harness):
    harness.claim()
    harness.claim_generation()
    auth, case, run = (
        harness.records('authorization')[0],
        harness.records('case')[-1],
        harness.records('agent_run')[-1],
    )
    child, dispatch = (
        harness.records('cost_component')[-1],
        harness.records('dispatch')[-1],
    )
    with harness.engine.connect() as connection:
        incident = harness.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=run['id'],
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    failed = {**case, 'state': 'failed'}
    with harness.engine.connect() as connection:
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
                        'completed_at': run['started_at'] + timedelta(seconds=1),
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
        for kind in ('provider_safety_authority', 'provider_readiness'):
            record = next(
                row
                for row in harness.records(kind)
                if kind == 'provider_safety_authority'
                or row['component'] == 'answer_generation'
            )
            key = row_key(kind, record)
            mutations._observation_rows.remove(key)
            identity = release_row_identity_hmac(
                kind, key.primary_key, identity_secret=_SECRET
            )
            payload['observation_set'] = [
                row
                for row in payload['observation_set']
                if (row['row_kind'], row['row_identity_hmac']) != (kind, identity)
            ]
            payload['affected_rows'].append(
                {'row_kind': kind, 'row_identity_hmac': identity}
            )
        payload['affected_rows'].sort(
            key=lambda row: (row['row_kind'], row['row_identity_hmac'])
        )
        payload['provider_safety_envelope_digest'] = incident.new_envelope_digest
        harness.snapshot = harness.ledger.append(
            connection,
            payload,
            actual_mutations=mutations,
            database_identity=harness.database_identity,
            provider_incident=incident,
        )
    assert (
        harness.records('provider_safety_authority')[0]['envelope_digest']
        == incident.new_envelope_digest
    )
    assert harness.records('provider_readiness')[-1]['state'] == 'blocked_overrun'


MATRIX = [(kind, outcome) for kind, outcome in _KIND_OUTCOME.items()] + [
    ('authorization_finish_failed', 'execution_contract_failed')
]


@pytest.mark.parametrize('kind,outcome', MATRIX)
def test_every_kind_appends_real_sql_with_paired_invalid_proof(
    tmp_path, monkeypatch, append_audit, kind, outcome
):
    from backend.app.rag.release_ledger import _OUTCOMES

    assert {item[0] for item in MATRIX} == set(_OUTCOMES)
    if kind == 'authorization_abort_component':
        harness = incident_harness(tmp_path, monkeypatch)
        incident_abort(harness)
    else:
        harness = ReleaseHarness(
            create_engine('sqlite://'),
            _authority(tmp_path),
            _SECRET,
            _identity(),
            query_reserves=('0.001000',) * 10
            if kind in {'authorization_complete', 'authorization_finish_quality_failed'}
            else (),
        )
        if kind == 'authorization_bootstrap':
            pass
        elif kind in {'authorization_complete', 'authorization_finish_quality_failed'}:
            for ordinal in range(30):
                harness.claim(
                    ordinal, query_reserve='0.001000' if ordinal < 10 else '0.000000'
                )
                if ordinal < 10:
                    harness.claim_generation('query_embedding')
                    query_outcome(harness)
                harness.claim_generation()
                generation_outcome(harness)
                project_case(harness)
            payload = finish(harness, kind, outcome)
            assert (
                payload['case_claim_count'],
                payload['embedding_dispatch_count'],
                payload['generation_dispatch_count'],
                payload['total_dispatch_count'],
            ) == (30, 10, 30, 40)
            assert payload['authorization_charged_cost_usd'] == '0.016000'
            assert Counter(row['row_kind'] for row in payload['observation_set']) == {
                'case': 30,
                'agent_run': 30,
                'cost_component': 60,
                'dispatch': 40,
                'provider_safety_authority': 1,
                'provider_readiness': 2,
            }
        elif kind == 'authorization_finish_failed':
            for ordinal in range(30):
                harness.claim(ordinal)
                if outcome == 'execution_contract_failed':
                    safe_case(harness)
                else:
                    harness.fail_case(outcome='model_unavailable')
            payload = finish(harness, kind, outcome)
            assert payload['total_dispatch_count'] == 0
            assert payload['authorization_charged_cost_usd'] == '0.000000'
            assert Counter(row['row_kind'] for row in payload['observation_set']) == {
                'case': 30,
                'agent_run': 30,
                'cost_component': 60,
                'provider_safety_authority': 1,
                'provider_readiness': 2,
            }
        elif kind in {
            'authorization_abort_final',
            'authorization_abort_snapshot',
            'authorization_abort_corpus_drift',
            'authorization_abort_execution_crash',
        }:
            drive_null_abort(harness, kind)
        elif kind == 'authorization_abort_component_snapshot':
            drive_snapshot_abort(harness)
        else:
            harness.claim()
            if kind == 'case_claim':
                pass
            elif kind == 'case_failure':
                harness.fail_case(outcome=outcome)
            elif kind == 'case_safe_outcome':
                safe_case(harness)
            elif kind == 'authorization_abort_control':
                _provider_drift(harness)
                harness.fail_case(kind, outcome)
            else:
                harness.claim_generation()
                if kind != 'component_claim':
                    harness.generation_outcome()
                if kind == 'case_outcome':
                    project_case(harness)
    assert (kind, outcome) in append_audit
    harness.engine.dispose()
