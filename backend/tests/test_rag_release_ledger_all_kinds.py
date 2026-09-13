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
from backend.app.admin.rag_provider_safety import (
    _freeze_release_peer as real_freeze_release_peer,
)
from backend.app.admin.rag_provider_safety import (
    _require_provider_guard as real_require_provider_guard,
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
    roster_probed = set()

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
        transition = payload['transition_kind'], payload['outcome']
        if transition not in roster_probed:
            from sqlalchemy import event

            if payload['transition_kind'] == 'authorization_bootstrap':
                # The synthetic harness seeds provider peers just before this
                # first append. Preserve only that test setup across refusal.
                connection.commit()
            roster_probed.add(transition)
            missing = self.mutation_set(connection)
            for plan in actual_mutations._plans[1:]:
                missing.plan(plan.statement, plan.row)
            for row in actual_mutations.observation_rows:
                missing.observe(row)
            dml = []

            def count_dml(_connection, _cursor, sql, *_):
                if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                    dml.append(sql)

            event.listen(connection.engine, 'before_cursor_execute', count_dml)
            try:
                with pytest.raises(RagReleaseLedgerError):
                    original(
                        self, connection, payload, actual_mutations=missing, **kwargs
                    )
            finally:
                event.remove(connection.engine, 'before_cursor_execute', count_dml)
            assert dml == []
            assert self._authority.marker_path.read_bytes() == old_marker
            assert [
                dict(row)
                for row in connection.execute(select(tables.transitions)).mappings()
            ] == old_history
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


def incident_harness(tmp_path, monkeypatch, *, prepare_only=False):
    """Real sealed Task22 service; only PostgreSQL transport/advisory is substituted."""
    from backend.app.admin import rag_provider_safety as admin_module
    from backend.app.rag import release_authority as authority_module
    from backend.tests.test_rag_provider_safety_admin import _review_bytes, _service

    monkeypatch.setattr(admin_module, 'RagProviderSafetyReleasePeer', RealPeer)
    monkeypatch.setattr(admin_module, '_freeze_release_peer', real_freeze_release_peer)
    monkeypatch.setattr(
        admin_module, '_require_provider_guard', real_require_provider_guard
    )
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
    if prepare_only:
        return engine, authority
    return ReleaseHarness(engine, authority, _SECRET, _identity())


@pytest.mark.parametrize(
    'operation', ['initialize', 'disaster_initialize', 'rebootstrap', 'inspect']
)
def test_nonappend_provider_entry_drift_never_publishes(
    tmp_path, monkeypatch, operation
):
    from sqlalchemy import event, func, update

    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
    from backend.app.models.rag_runtime import RagProviderSafetyAuthority
    from backend.app.rag.release_authority import RagReleaseAuthorityError

    engine, authority = incident_harness(tmp_path, monkeypatch, prepare_only=True)
    tables = release_tables(build_rag_release_metadata())
    if operation in {'inspect', 'rebootstrap'}:
        with engine.connect() as connection:
            authority.initialize(
                connection,
                database_identity=_identity(),
                review_envelope_hmac='1' * 64,
                review_nonce_hmac='2' * 64,
            )
            if operation == 'rebootstrap':
                connection.execute(
                    update(tables.ledgers).values(
                        generation=9, last_transition_digest='9' * 64
                    )
                )
                connection.commit()
    old_marker = (
        authority.marker_path.read_bytes() if authority.marker_path.exists() else None
    )
    original = type(authority)._authority_transport
    callbacks, release_dml = [], []

    @contextmanager
    def drift_transport(owner, connection, *, marker):
        with original(owner, connection, marker=marker) as guard:
            callbacks.append('entry')
            connection.execute(
                update(RagProviderSafetyAuthority).values(global_safety_generation=99)
            )
            yield guard

    def observe(_conn, _cursor, sql, *_):
        if (
            sql.lstrip()
            .upper()
            .startswith(('INSERT INTO RAG_LIVE_GATE', 'UPDATE RAG_LIVE_GATE'))
        ):
            release_dml.append(sql)

    monkeypatch.setattr(type(authority), '_authority_transport', drift_transport)
    event.listen(engine, 'before_cursor_execute', observe)
    refused = False
    with engine.connect() as connection:
        try:
            kwargs = {'database_identity': _identity()}
            if operation == 'initialize':
                kwargs.update(review_envelope_hmac='3' * 64, review_nonce_hmac='4' * 64)
            elif operation != 'inspect':
                kwargs['review_verifier'] = lambda *_: ('3' * 64, '4' * 64, '5' * 64)
            getattr(authority, operation)(connection, **kwargs)
        except (RagReleaseAuthorityError, RagProviderSafetyError):
            refused = True
        finally:
            connection.rollback()
    with engine.connect() as check:
        if operation in {'inspect', 'rebootstrap'}:
            assert check.scalar(select(func.count()).select_from(tables.ledgers)) == 1
            assert check.scalar(select(tables.ledgers.c.generation)) == (
                9 if operation == 'rebootstrap' else 0
            )
        else:
            assert (
                not __import__('sqlalchemy')
                .inspect(check)
                .has_table(tables.ledgers.name)
            )
    assert callbacks == ['entry']
    assert release_dml == []
    assert (
        authority.marker_path.read_bytes() if authority.marker_path.exists() else None
    ) == old_marker
    assert refused


def test_all_plans_freeze_before_reordered_authorization_callable_commit(
    tmp_path, monkeypatch
):
    from sqlalchemy import bindparam, event

    harness = incident_harness(tmp_path, monkeypatch)
    old_auth = harness.records('authorization')
    old_marker = harness.authority.marker_path.read_bytes()
    original = RagReleaseLedger.append
    calls, dml = [], []

    def attack(self, connection, payload, *, actual_mutations, **kwargs):
        if payload['transition_kind'] == 'case_claim':
            plan = actual_mutations._plans.pop(0)

            def commit_value():
                calls.append(len(dml))
                connection.commit()
                return 1

            actual_mutations.plan(
                plan.statement.values(
                    case_claim_count=bindparam('late_count', callable_=commit_value)
                ),
                plan.row,
            )
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    def observe(_conn, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    monkeypatch.setattr(RagReleaseLedger, 'append', attack)
    event.listen(harness.engine, 'before_cursor_execute', observe)
    with pytest.raises(RagReleaseLedgerError):
        harness.claim()
    assert harness.records('case') == []
    assert harness.records('agent_run') == []
    assert harness.records('cost_component') == []
    assert harness.records('authorization') == old_auth
    assert harness.authority.marker_path.read_bytes() == old_marker
    assert calls == []
    assert dml == []


@pytest.mark.parametrize(
    'row_kind',
    [
        'authorization',
        'case',
        'dispatch',
        'quality_report',
        'release_ledger',
        'release_transition',
        'agent_run',
        'cost_component',
    ],
)
@pytest.mark.parametrize(
    'attack_kind', ['callable', 'expression', 'processor', 'default', 'delete']
)
def test_all_table_plan_surfaces_refuse_before_first_dml(
    tmp_path, monkeypatch, row_kind, attack_kind
):
    from sqlalchemy import (
        Integer,
        MetaData,
        bindparam,
        delete,
        event,
        insert,
        literal,
        update,
    )
    from sqlalchemy.types import TypeDecorator

    from backend.app.rag.release_ledger import ReleaseRowPrimaryKey

    harness = incident_harness(tmp_path, monkeypatch)
    old_marker = harness.authority.marker_path.read_bytes()
    old = {
        kind: harness.records(kind)
        for kind in (
            'authorization',
            'case',
            'dispatch',
            'quality_report',
            'agent_run',
            'cost_component',
        )
    }
    original = RagReleaseLedger.append
    calls, dml = [], []

    def attack(self, connection, payload, *, actual_mutations, **kwargs):
        plans = actual_mutations._plans
        if row_kind in {'release_ledger', 'release_transition'}:
            tables = release_tables(build_rag_release_metadata())
            table = (
                tables.ledgers if row_kind == 'release_ledger' else tables.transitions
            )
            key = {
                'ledger_uuid': payload['ledger_uuid'],
                'ledger_epoch': payload['ledger_epoch'],
            }
            if row_kind == 'release_transition':
                key['to_generation'] = payload['to_generation']
            row = ReleaseRowPrimaryKey(row_kind, key)
            statement = insert(table).values(generation=2)
        elif row_kind in {'dispatch', 'quality_report'}:
            table, _ = actual_mutations._table(row_kind)
            key = {
                'ledger_uuid': payload['ledger_uuid'],
                'ledger_epoch': payload['ledger_epoch'],
                'approval_id_hmac': payload['approval_id_hmac'],
            }
            if row_kind == 'dispatch':
                key.update(
                    case_id_hmac=payload['case_id_hmac'], component='answer_generation'
                )
            row = ReleaseRowPrimaryKey(row_kind, key)
            statement = insert(table).values(**key)
        else:
            plan = next(plan for plan in plans if plan.row.row_kind == row_kind)
            plans.remove(plan)
            statement, row = plan.statement, plan.row
            table = statement.table
        name = next(iter(statement._values))
        value = statement._values[name].value

        def callback(*_):
            calls.append(len(dml))
            connection.commit()
            return value

        class Processor(TypeDecorator):
            impl = Integer
            cache_ok = False

            def process_bind_param(self, value, dialect):
                callback()
                return value

        if attack_kind == 'callable':
            statement = statement.values(
                {name: bindparam('attack', callable_=callback)}
            )
        elif attack_kind == 'expression':
            statement = statement.values({name: literal(value) + literal(0)})
        elif attack_kind == 'processor':
            statement = statement.values(
                {name: bindparam('attack', value=value, type_=Processor())}
            )
        elif attack_kind == 'default':
            from sqlalchemy import ColumnDefault

            clone = table.to_metadata(MetaData())
            clone.c[str(name)].default = ColumnDefault(callback)
            values = {
                str(k): v.value for k, v in statement._values.items() if k != name
            }
            statement = (
                insert(clone) if row_kind != 'authorization' else update(clone)
            ).values(**values)
        else:
            statement = delete(table)
        actual_mutations.plan(statement, row)
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    def observe(_conn, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    monkeypatch.setattr(RagReleaseLedger, 'append', attack)
    event.listen(harness.engine, 'before_cursor_execute', observe)
    with pytest.raises(RagReleaseLedgerError):
        harness.claim()
    assert {kind: harness.records(kind) for kind in old} == old
    assert harness.authority.marker_path.read_bytes() == old_marker
    assert calls == [] and dml == []


@pytest.mark.parametrize(
    'operation', ['initialize', 'disaster_initialize', 'rebootstrap']
)
@pytest.mark.parametrize(
    'attack_kind', ['provider', 'commit', 'rollback', 'close', 'replace_transaction']
)
def test_marker_first_hook_never_publishes_release_db_after_callback_drift(
    tmp_path, monkeypatch, operation, attack_kind
):
    from sqlalchemy import event, update

    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
    from backend.app.models.rag_runtime import RagProviderSafetyAuthority
    from backend.app.rag.release_authority import RagReleaseAuthorityError

    engine, authority = incident_harness(tmp_path, monkeypatch, prepare_only=True)
    tables = release_tables(build_rag_release_metadata())
    if operation == 'rebootstrap':
        with engine.connect() as connection:
            authority.initialize(
                connection,
                database_identity=_identity(),
                review_envelope_hmac='1' * 64,
                review_nonce_hmac='2' * 64,
            )
            connection.execute(
                update(tables.ledgers).values(
                    generation=9, last_transition_digest='9' * 64
                )
            )
            connection.commit()
    old_rows = None
    if operation == 'rebootstrap':
        with engine.connect() as connection:
            old_rows = [
                dict(row)
                for row in connection.execute(select(tables.ledgers)).mappings()
            ]
    calls, dml = [], []
    with engine.connect() as connection:

        def hook():
            calls.append(len(dml))
            assert authority.marker_path.exists()
            if attack_kind == 'provider':
                connection.execute(
                    update(RagProviderSafetyAuthority).values(
                        global_safety_generation=99
                    )
                )
            elif attack_kind == 'replace_transaction':
                connection.rollback()
                connection.begin()
            else:
                getattr(connection, attack_kind)()

        authority._after_marker_replace = hook

        def observe(_conn, _cursor, sql, *_):
            if (
                sql.lstrip()
                .upper()
                .startswith(('INSERT INTO RAG_LIVE_GATE', 'UPDATE RAG_LIVE_GATE'))
            ):
                dml.append(sql)

        event.listen(engine, 'before_cursor_execute', observe)
        kwargs = {'database_identity': _identity()}
        if operation == 'initialize':
            kwargs.update(review_envelope_hmac='3' * 64, review_nonce_hmac='4' * 64)
        else:
            kwargs['review_verifier'] = lambda *_: ('3' * 64, '4' * 64, '5' * 64)
        with pytest.raises((RagReleaseAuthorityError, RagProviderSafetyError)):
            getattr(authority, operation)(connection, **kwargs)
    with engine.connect() as check:
        if old_rows is None:
            assert (
                not __import__('sqlalchemy')
                .inspect(check)
                .has_table(tables.ledgers.name)
            )
        else:
            assert [
                dict(row) for row in check.execute(select(tables.ledgers)).mappings()
            ] == old_rows
    assert calls == [0] and dml == []
    assert authority.marker_path.exists()  # deliberate marker-first crash evidence


def test_literal_plan_reordering_is_safe_and_executes_no_callback(
    tmp_path, monkeypatch
):
    harness = incident_harness(tmp_path, monkeypatch)
    original = RagReleaseLedger.append

    def reordered(self, connection, payload, *, actual_mutations, **kwargs):
        actual_mutations._plans.append(actual_mutations._plans.pop(0))
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    monkeypatch.setattr(RagReleaseLedger, 'append', reordered)
    harness.claim()
    assert harness.snapshot.generation == 2
    assert harness.records('authorization')[0]['case_claim_count'] == 1
    assert len(harness.records('case')) == len(harness.records('agent_run')) == 1
    assert len(harness.records('cost_component')) == 2


@pytest.mark.parametrize(
    'mutation',
    [
        'extra',
        'missing',
        'duplicate',
        'child_order',
        'semantic',
        'wrong_table',
        'wrong_operation',
        'wrong_identity',
    ],
)
def test_frozen_plan_roster_refuses_before_any_dml(tmp_path, monkeypatch, mutation):
    from sqlalchemy import event, insert

    harness = incident_harness(tmp_path, monkeypatch)
    old_auth = harness.records('authorization')
    old_marker = harness.authority.marker_path.read_bytes()
    original = RagReleaseLedger.append
    dml = []

    def attack(self, connection, payload, *, actual_mutations, **kwargs):
        plans = actual_mutations._plans
        if mutation == 'extra':
            row = {
                **old_auth[0],
                'approval_id_hmac': 'f' * 64,
                'approval_hmac': 'e' * 64,
            }
            table, _ = actual_mutations._table('authorization')
            actual_mutations.plan(
                insert(table).values(**row), row_key('authorization', row)
            )
        elif mutation == 'missing':
            plans.pop(0)
        elif mutation == 'duplicate':
            plans.append(plans[0])
        elif mutation == 'child_order':
            plans[-2:] = reversed(plans[-2:])
        elif mutation == 'wrong_table':
            plans[0] = type(plans[0])(plans[1].statement, plans[0].row)
        elif mutation == 'wrong_operation':
            table, _ = actual_mutations._table('authorization')
            plans[0] = type(plans[0])(insert(table).values(**old_auth[0]), plans[0].row)
        elif mutation == 'wrong_identity':
            from backend.app.rag.release_ledger import ReleaseRowPrimaryKey

            plan = plans[1]
            row = ReleaseRowPrimaryKey(
                'case', {**plan.row.primary_key, 'case_id_hmac': 'f' * 64}
            )
            plans[1] = type(plan)(plan.statement, row)
        else:
            plan = plans[0]
            plans[0] = type(plan)(
                plan.statement.values(total_dispatch_count=1), plan.row
            )
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    monkeypatch.setattr(RagReleaseLedger, 'append', attack)
    event.listen(
        harness.engine,
        'before_cursor_execute',
        lambda _c, _u, sql, *_: (
            dml.append(sql)
            if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE'))
            else None
        ),
    )
    with pytest.raises(RagReleaseLedgerError):
        harness.claim()
    assert harness.records('authorization') == old_auth
    assert (
        harness.records('case')
        == harness.records('agent_run')
        == harness.records('cost_component')
        == []
    )
    assert harness.authority.marker_path.read_bytes() == old_marker
    assert dml == []


@pytest.mark.parametrize(
    'row_kind',
    [
        'authorization',
        'case',
        'dispatch',
        'quality_report',
        'release_ledger',
        'release_transition',
        'agent_run',
        'cost_component',
    ],
)
def test_extra_native_plan_for_every_table_refuses_before_dml(
    tmp_path, monkeypatch, row_kind
):
    from sqlalchemy import event, insert

    from backend.app.rag.release_ledger import (
        ReleaseRowPrimaryKey,
        _ReleaseMutationPlan,
    )

    harness = incident_harness(tmp_path, monkeypatch)
    old_marker = harness.authority.marker_path.read_bytes()
    kinds = (
        'authorization',
        'case',
        'dispatch',
        'quality_report',
        'agent_run',
        'cost_component',
    )
    before = {kind: harness.records(kind) for kind in kinds}
    original = RagReleaseLedger.append
    dml = []

    def attack(self, connection, payload, *, actual_mutations, **kwargs):
        if row_kind in {'release_ledger', 'release_transition'}:
            tables = release_tables(build_rag_release_metadata())
            table = (
                tables.ledgers if row_kind == 'release_ledger' else tables.transitions
            )
            values = dict(connection.execute(select(table)).mappings().first())
            key = {
                'ledger_uuid': payload['ledger_uuid'],
                'ledger_epoch': payload['ledger_epoch'],
            }
            if row_kind == 'release_transition':
                key['to_generation'] = payload['to_generation']
                values['generation'] = payload['to_generation']
            row = ReleaseRowPrimaryKey(row_kind, key)
        else:
            table, _ = actual_mutations._table(row_kind)
            key = {
                key: payload[key]
                for key in ('ledger_uuid', 'ledger_epoch', 'approval_id_hmac')
            }
            if row_kind == 'authorization':
                values = {
                    **before['authorization'][0],
                    'approval_id_hmac': 'f' * 64,
                    'approval_hmac': 'e' * 64,
                }
            elif row_kind == 'dispatch':
                values = {
                    **key,
                    'case_id_hmac': payload['case_id_hmac'],
                    'component': 'answer_generation',
                    'state': 'not_attempted',
                    'dispatch_count': 0,
                    'dispatch_fence_hmac': None,
                    'charge_basis': 'reserved',
                    'reserved_cost_usd': Decimal('0.000000'),
                    'charged_cost_usd': Decimal('0.000000'),
                }
            elif row_kind == 'quality_report':
                values = {
                    **key,
                    'quality_report_hmac': 'a' * 64,
                    'manifest_hmac': '9' * 64,
                    'baseline_hmac': 'b' * 64,
                    'reviewer_roster_hmac': 'c' * 64,
                    'payload_canonical_bytes': b'{}',
                }
            else:
                plan = next(
                    plan
                    for plan in actual_mutations._plans
                    if plan.row.row_kind == row_kind
                )
                values = {
                    str(key): binding.value
                    for key, binding in plan.statement._values.items()
                }
                if row_kind == 'case':
                    values.update(
                        case_id_hmac='f' * 64,
                        manifest_ordinal=1,
                        runtime_agent_run_id_hmac='e' * 64,
                    )
                elif row_kind == 'agent_run':
                    values.update(id=999, cache_key='extra-native-run')
                else:
                    values['id'] = 999
            row = row_key(row_kind, values)
        # Include even a duplicate key to exercise the final authority boundary,
        # not merely the public collector's earlier convenience check.
        actual_mutations._plans.append(
            _ReleaseMutationPlan(insert(table).values(**values), row)
        )
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    monkeypatch.setattr(RagReleaseLedger, 'append', attack)
    event.listen(
        harness.engine,
        'before_cursor_execute',
        lambda _c, _u, sql, *_: (
            dml.append(sql)
            if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE'))
            else None
        ),
    )
    with pytest.raises(RagReleaseLedgerError):
        harness.claim()
    assert dml == []
    assert {kind: harness.records(kind) for kind in kinds} == before
    assert harness.authority.marker_path.read_bytes() == old_marker


@pytest.mark.parametrize('subclass', [False, True])
def test_append_never_reuses_caller_database_identity_after_dml(
    tmp_path, monkeypatch, subclass
):
    from sqlalchemy import event

    from backend.app.rag.release_authority import (
        RagReleaseAuthorityError,
        ValidationDatabaseIdentity,
    )

    harness = incident_harness(tmp_path, monkeypatch)
    identity = harness.database_identity
    old_marker = harness.authority.marker_path.read_bytes()
    old_auth = harness.records('authorization')
    calls, dml = [], []

    class CallbackIdentity(ValidationDatabaseIdentity):
        def __getattribute__(self, name):
            if dml:
                calls.append(name)
                active[0].commit()
            return object.__getattribute__(self, name)

    if subclass:
        identity = CallbackIdentity(identity.database_name, identity.database_oid)
    harness.database_identity = identity
    active = [None]

    def observe(connection, _u, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            active[0] = connection
            dml.append(sql)
            if not subclass:
                object.__setattr__(identity, 'database_name', 'mutated-after-entry')

    event.listen(harness.engine, 'before_cursor_execute', observe)
    if subclass:
        with pytest.raises((RagReleaseAuthorityError, RagReleaseLedgerError)):
            harness.claim()
        assert dml == []
        assert harness.records('authorization') == old_auth
        assert harness.records('case') == []
        assert harness.authority.marker_path.read_bytes() == old_marker
    else:
        harness.claim()
        assert harness.snapshot.generation == 2
        assert len(harness.records('case')) == 1
    assert calls == []


@pytest.mark.parametrize(
    'row_kind,column',
    [
        ('authorization', 'reserved_cost_usd'),
        ('authorization', 'charged_cost_usd'),
        ('case', 'embedding_reserved_cost_usd'),
        ('case', 'generation_reserved_cost_usd'),
        ('case', 'total_reserved_cost_usd'),
        ('dispatch', 'reserved_cost_usd'),
        ('dispatch', 'charged_cost_usd'),
        ('agent_run', 'total_charged_cost_usd'),
        ('cost_component', 'reserved_cost_usd'),
        ('cost_component', 'charged_cost_usd'),
    ],
)
@pytest.mark.parametrize(
    'representation',
    [
        'str',
        'int',
        'float',
        'bool',
        'subclass',
        'subprecision',
        'nan',
        'infinity',
        'negative',
        'negative_zero',
        'excess_scale',
        'overflow',
    ],
)
def test_every_numeric_caller_literal_refuses_laundering_before_dml(
    tmp_path, monkeypatch, row_kind, column, representation
):
    from sqlalchemy import event

    harness = incident_harness(tmp_path, monkeypatch)
    if row_kind == 'dispatch':
        harness.claim()
    old_marker = harness.authority.marker_path.read_bytes()
    kinds = ('authorization', 'case', 'dispatch', 'agent_run', 'cost_component')
    old_rows = {kind: harness.records(kind) for kind in kinds}
    original = RagReleaseLedger.append
    dml = []

    class DecimalSubclass(Decimal):
        pass

    def attack(self, connection, payload, *, actual_mutations, **kwargs):
        plan = next(
            item for item in actual_mutations._plans if item.row.row_kind == row_kind
        )
        binding = plan.statement._values.get(column)
        value = binding.value if binding is not None else old_rows[row_kind][0][column]
        conversions = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'subclass': DecimalSubclass,
            'subprecision': lambda v: v + Decimal('0.0000001'),
            'nan': lambda v: Decimal('NaN'),
            'infinity': lambda v: Decimal('Infinity'),
            'negative': lambda v: Decimal('-0.000001'),
            'negative_zero': lambda v: Decimal('-0.000000'),
            'excess_scale': lambda v: v.quantize(Decimal('0.0000000')),
            'overflow': lambda v: Decimal('1e30'),
        }
        actual_mutations._plans[actual_mutations._plans.index(plan)] = type(plan)(
            plan.statement.values({column: conversions[representation](value)}),
            plan.row,
        )
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    monkeypatch.setattr(RagReleaseLedger, 'append', attack)
    event.listen(
        harness.engine,
        'before_cursor_execute',
        lambda _c, _u, sql, *_: (
            dml.append(sql)
            if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE'))
            else None
        ),
    )
    with pytest.raises(RagReleaseLedgerError):
        harness.claim_generation() if row_kind == 'dispatch' else harness.claim()
    assert {kind: harness.records(kind) for kind in kinds} == old_rows
    assert harness.authority.marker_path.read_bytes() == old_marker
    assert dml == []


@pytest.mark.parametrize(
    'operation', ['initialize', 'disaster_initialize', 'rebootstrap', 'inspect']
)
def test_nonappend_final_checkpoint_refuses_provider_db_drift(
    tmp_path, monkeypatch, operation
):
    from sqlalchemy import event, update

    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
    from backend.app.models.rag_runtime import RagProviderSafetyAuthority

    engine, authority = incident_harness(tmp_path, monkeypatch, prepare_only=True)
    tables = release_tables(build_rag_release_metadata())
    if operation in {'inspect', 'rebootstrap'}:
        with engine.connect() as connection:
            authority.initialize(
                connection,
                database_identity=_identity(),
                review_envelope_hmac='1' * 64,
                review_nonce_hmac='2' * 64,
            )
            if operation == 'rebootstrap':
                connection.execute(
                    update(tables.ledgers).values(
                        generation=9, last_transition_digest='9' * 64
                    )
                )
                connection.commit()
    with engine.connect() as connection:
        old_provider = dict(
            connection.execute(select(RagProviderSafetyAuthority)).mappings().one()
        )
        old_ledgers = (
            [dict(row) for row in connection.execute(select(tables.ledgers)).mappings()]
            if operation in {'inspect', 'rebootstrap'}
            else []
        )
    attacks = []
    original = type(authority)._inspect_locked
    if operation == 'inspect':

        def inspect_then_drift(owner, connection, *args, **kwargs):
            result = original(owner, connection, *args, **kwargs)
            attacks.append('inspection')
            connection.execute(
                update(RagProviderSafetyAuthority).values(global_safety_generation=99)
            )
            return result

        monkeypatch.setattr(type(authority), '_inspect_locked', inspect_then_drift)
    else:

        def after_insert(connection, _cursor, sql, *_):
            if sql.lstrip().upper().startswith('INSERT INTO RAG_LIVE_GATE_LEDGERS'):
                attacks.append('ledger_insert')
                connection.execute(
                    update(RagProviderSafetyAuthority).values(
                        global_safety_generation=99
                    )
                )

        event.listen(engine, 'after_cursor_execute', after_insert)
    with engine.connect() as connection:
        kwargs = {'database_identity': _identity()}
        if operation == 'initialize':
            kwargs.update(review_envelope_hmac='3' * 64, review_nonce_hmac='4' * 64)
        elif operation != 'inspect':
            kwargs['review_verifier'] = lambda *_: ('3' * 64, '4' * 64, '5' * 64)
        with pytest.raises(RagProviderSafetyError):
            getattr(authority, operation)(connection, **kwargs)
    with engine.connect() as connection:
        assert [
            dict(row) for row in connection.execute(select(tables.ledgers)).mappings()
        ] == old_ledgers
        assert (
            dict(
                connection.execute(select(RagProviderSafetyAuthority)).mappings().one()
            )
            == old_provider
        )
    assert len(attacks) == 1


@pytest.mark.parametrize('operation', ['disaster_initialize', 'rebootstrap'])
def test_recovery_review_callback_drift_cannot_publish_marker(
    tmp_path, monkeypatch, operation
):
    from sqlalchemy import update

    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
    from backend.app.models.rag_runtime import RagProviderSafetyAuthority

    engine, authority = incident_harness(tmp_path, monkeypatch, prepare_only=True)
    tables = release_tables(build_rag_release_metadata())
    if operation == 'rebootstrap':
        with engine.connect() as connection:
            authority.initialize(
                connection,
                database_identity=_identity(),
                review_envelope_hmac='1' * 64,
                review_nonce_hmac='2' * 64,
            )
            connection.execute(
                update(tables.ledgers).values(
                    generation=9, last_transition_digest='9' * 64
                )
            )
            connection.commit()
    old_marker = (
        authority.marker_path.read_bytes() if authority.marker_path.exists() else None
    )
    calls = []

    def reviewed(connection, raw):
        calls.append(raw)
        connection.execute(
            update(RagProviderSafetyAuthority).values(global_safety_generation=99)
        )
        return '3' * 64, '4' * 64, '5' * 64

    with engine.connect() as connection, pytest.raises(RagProviderSafetyError):
        getattr(authority, operation)(
            connection, database_identity=_identity(), review_verifier=reviewed
        )
    assert calls == [old_marker]
    assert (
        authority.marker_path.read_bytes() if authority.marker_path.exists() else None
    ) == old_marker


@pytest.mark.parametrize(
    'effect', ['commit', 'rollback', 'close', 'begin', 'replace_transaction']
)
def test_plan_materialization_transaction_callbacks_refuse_before_dml(
    tmp_path, monkeypatch, effect
):
    from sqlalchemy import event
    from sqlalchemy.exc import SQLAlchemyError

    harness = incident_harness(tmp_path, monkeypatch)
    before = harness.records('authorization')
    old_marker = harness.authority.marker_path.read_bytes()
    original = RagReleaseLedger.append
    calls, dml = [], []

    def callback_plan(self, connection, payload, *, actual_mutations, **kwargs):
        plan = actual_mutations._plans.pop(0)

        class Values(dict):
            def items(self):
                calls.append(len(dml))
                if effect == 'replace_transaction':
                    connection.rollback()
                    connection.begin()
                else:
                    getattr(connection, effect)()
                return super().items()

        statement = plan.statement._clone()
        statement._values = Values(statement._values)
        actual_mutations.plan(statement, plan.row)
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    def observe(_conn, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    monkeypatch.setattr(RagReleaseLedger, 'append', callback_plan)
    event.listen(harness.engine, 'before_cursor_execute', observe)
    with pytest.raises((RagReleaseLedgerError, SQLAlchemyError)):
        harness.claim()
    assert harness.records('authorization') == before
    assert (
        harness.records('case')
        == harness.records('agent_run')
        == harness.records('cost_component')
        == []
    )
    assert harness.authority.marker_path.read_bytes() == old_marker
    assert calls == [0] and dml == []


@pytest.mark.parametrize('zone', ['driver_zoneinfo', 'fixed_utc', 'custom'])
def test_native_driver_datetime_plan_preserves_normalization_without_tz_callbacks(
    tmp_path, monkeypatch, zone
):
    from datetime import UTC, timezone, tzinfo

    from psycopg._tz import get_tzinfo
    from sqlalchemy import event

    harness = incident_harness(tmp_path, monkeypatch)
    harness.claim()
    before = {
        kind: harness.records(kind)
        for kind in ('authorization', 'case', 'dispatch', 'agent_run', 'cost_component')
    }
    old_marker = harness.authority.marker_path.read_bytes()
    original = RagReleaseLedger.append
    calls, dml = [], []

    class DriverConnection:
        def parameter_status(self, name):
            assert name == b'TimeZone'
            return b'Etc/UTC'

    class CallbackZone(tzinfo):
        def utcoffset(self, value):
            calls.append('utcoffset')
            return timedelta(0)

        def dst(self, value):
            calls.append('dst')
            return timedelta(0)

    selected_zone = (
        get_tzinfo(DriverConnection())
        if zone == 'driver_zoneinfo'
        else timezone(timedelta(0), 'driver UTC')
        if zone == 'fixed_utc'
        else CallbackZone()
    )

    def use_driver_timestamp(self, connection, payload, *, actual_mutations, **kwargs):
        plan = next(
            plan
            for plan in actual_mutations._plans
            if plan.row.row_kind == 'cost_component'
        )
        current = actual_mutations._snapshot(connection, plan.row)
        timestamp = (
            current['updated_at']
            .replace(tzinfo=UTC)
            .astimezone(UTC)
            .replace(tzinfo=selected_zone)
        )
        index = actual_mutations._plans.index(plan)
        actual_mutations._plans[index] = type(plan)(
            plan.statement.values(updated_at=timestamp), plan.row
        )
        return original(
            self, connection, payload, actual_mutations=actual_mutations, **kwargs
        )

    def observe(_conn, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    monkeypatch.setattr(RagReleaseLedger, 'append', use_driver_timestamp)
    event.listen(harness.engine, 'before_cursor_execute', observe)
    if zone == 'custom':
        with pytest.raises(RagReleaseLedgerError):
            harness.claim_generation()
        assert {kind: harness.records(kind) for kind in before} == before
        assert harness.authority.marker_path.read_bytes() == old_marker
        assert dml == []
    else:
        harness.claim_generation()
        assert harness.snapshot.generation == 3
        assert harness.records('authorization')[0]['total_dispatch_count'] == 1
        assert len(harness.records('dispatch')) == 1
        assert dml
    assert calls == []


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


@pytest.mark.parametrize('fail_after_sql', [False, True])
def test_real_append_ends_transaction_before_trusted_provider_cleanup(
    tmp_path, monkeypatch, fail_after_sql
):
    from sqlalchemy import event

    from backend.app.admin.rag_provider_safety import RagProviderSafetyReleasePeerGuard

    harness = incident_harness(tmp_path, monkeypatch)
    before = harness.records('authorization')
    old_marker = harness.authority.marker_path.read_bytes()
    phase = {'dml': False}
    external_calls = []
    cleanup_transactions = []
    original = RealPeer._locked_transport
    original_check = RagProviderSafetyReleasePeerGuard.revalidate_database_peer

    @contextmanager
    def transport(peer, connection, **options):
        with original(peer, connection, **options) as guard:
            try:
                yield guard
            finally:
                if phase['dml']:
                    cleanup_transactions.append(connection.get_transaction() is None)

    def callback(guard, connection):
        if phase['dml']:
            external_calls.append('provider_revalidation')
        return original_check(guard, connection)

    def observe(_connection, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            phase['dml'] = True
        if fail_after_sql and sql.lstrip().upper().startswith(
            'INSERT INTO RAG_LIVE_GATE_CASES'
        ):
            raise RuntimeError('test-only before-publication failure')

    monkeypatch.setattr(RealPeer, '_locked_transport', transport)
    monkeypatch.setattr(
        RagProviderSafetyReleasePeerGuard, 'revalidate_database_peer', callback
    )
    event.listen(harness.engine, 'after_cursor_execute', observe)
    if fail_after_sql:
        with pytest.raises(RuntimeError, match='test-only before-publication failure'):
            harness.claim()
        assert harness.records('authorization') == before
        assert harness.records('case') == []
        assert harness.authority.marker_path.read_bytes() == old_marker
    else:
        harness.claim()
        assert len(harness.records('case')) == 1
        assert harness.records('authorization')[0]['case_claim_count'] == 1
        assert harness.snapshot.generation == 2
    assert phase['dml'] and cleanup_transactions == [True]
    assert external_calls == []


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
