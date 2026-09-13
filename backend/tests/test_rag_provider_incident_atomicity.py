"""Task25 carried P1: caller SQL metadata must never publish partial incidents."""

from contextlib import contextmanager

import pytest
from sqlalchemy import ColumnDefault, TypeDecorator, event, text

from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError
from backend.app.models.rag_runtime import (
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)
from backend.app.rag.release_ledger import RagReleaseLedger
from backend.tests.test_rag_release_ledger import (  # noqa: F401
    _deterministic_non_product_database_seam,
)
from backend.tests.test_rag_release_ledger_all_kinds import (
    incident_abort,
    incident_harness,
)

PROVIDER = (
    'rag_provider_safety_authorities',
    'rag_provider_readiness',
    'rag_provider_safety_transitions',
)
RELEASE = (
    'rag_live_gate_ledgers',
    'rag_live_gate_transitions',
    'rag_live_gate_authorizations',
    'rag_live_gate_cases',
    'rag_live_gate_dispatches',
    'rag_live_gate_quality_reports',
    'agent_runs',
    'agent_run_cost_components',
)
MODELS = (RagProviderSafetyAuthority, RagProviderReadiness, RagProviderSafetyTransition)
COLUMNS = [
    (model, str(column.name)) for model in MODELS for column in model.__table__.c
]


def database_image(connection):
    # Independent physical SQL bypasses the metadata under attack and its processors.
    return {
        name: [
            dict(row)
            for row in connection.exec_driver_sql(
                'SELECT * FROM ' + name + ' ORDER BY 1'
            ).mappings()
        ]
        for name in PROVIDER + RELEASE
    }


def assert_complete_incident(h, before, after, marker, latch, category):
    authority = after[PROVIDER[0]][0]
    assert authority['global_safety_generation'] == 1
    assert len(after[PROVIDER[1]]) == 2
    untouched, blocked = after[PROVIDER[1]]
    assert untouched == before[PROVIDER[1]][0]
    state = (
        'blocked_overrun'
        if category == 'provider_usage_overrun'
        else 'blocked_remediation'
    )
    assert blocked['state'] == state
    assert blocked['state_version'] == 2
    assert blocked['family_safety_generation'] == 1
    assert blocked['overrun_input_tokens'] == 20
    assert blocked['overrun_output_tokens'] == 30
    expected_cost = 0.1 if category == 'provider_usage_overrun' else 0.0004
    assert blocked['overrun_cost_usd'] == expected_cost
    assert after[PROVIDER[2]][:-1] == before[PROVIDER[2]]
    history = after[PROVIDER[2]][-1]
    assert history['global_safety_generation'] == 1
    assert history['readiness_id'] == blocked['id']
    assert history['new_state'] == state
    assert history['new_state_version'] == 2
    assert history['envelope_digest'] == authority['envelope_digest']
    assert h.authority.marker_path.read_bytes() != marker
    runtime = h.authority._provider_safety_release_peer._provider_safety
    raw = runtime._latch_path.read_bytes()
    assert raw != latch
    import hashlib
    import json

    body = json.loads(raw)['signed_payload']['body']
    assert body['global_safety_generation'] == 1
    assert (
        hashlib.sha256(
            b'paraworks:provider-safety-envelope-file:v1\x00' + raw
        ).hexdigest()
        == authority['envelope_digest']
    )
    assert after['rag_live_gate_authorizations'][0]['state'] == (
        'aborted_overrun'
        if category == 'provider_usage_overrun'
        else 'aborted_provider_safety'
    )
    assert after['rag_live_gate_cases'][0]['state'] == 'failed'
    assert after['agent_runs'][0]['status'] == 'failed'
    assert after['agent_run_cost_components'][-1]['terminal_outcome'] == category
    assert (
        after['rag_live_gate_ledgers'][0]['generation']
        == before['rag_live_gate_ledgers'][0]['generation'] + 1
    )
    assert (
        len(after['rag_live_gate_transitions'])
        == len(before['rag_live_gate_transitions']) + 1
    )


@pytest.mark.parametrize(
    'model,column_name', COLUMNS, ids=[f'{m.__tablename__}.{n}' for m, n in COLUMNS]
)
@pytest.mark.parametrize('processor', ['bind', 'result'])
def test_incident_never_executes_shared_column_processors(
    tmp_path, monkeypatch, model, column_name, processor
):
    _metadata_attack(tmp_path, monkeypatch, model, column_name, processor, 'commit')


@pytest.mark.parametrize('action', ['rollback', 'close', 'replace'])
@pytest.mark.parametrize('processor', ['bind', 'result'])
def test_incident_shared_numeric_cannot_end_transaction(
    tmp_path, monkeypatch, action, processor
):
    _metadata_attack(
        tmp_path,
        monkeypatch,
        RagProviderReadiness,
        'overrun_cost_usd',
        processor,
        action,
    )


def _metadata_attack(
    tmp_path,
    monkeypatch,
    model,
    column_name,
    processor,
    action,
    *,
    category='provider_usage_overrun',
):
    h = incident_harness(tmp_path, monkeypatch)
    original_append, original_transport = (
        RagReleaseLedger.append,
        h.authority._authority_transport,
    )
    phase = {'incident': False, 'calls': [], 'dml': []}
    column = model.__table__.c[column_name]
    original_type = column.type
    metadata_patch = pytest.MonkeyPatch()

    def attack():
        if phase['dml'] and not phase['calls']:
            phase['calls'].append(action)
            connection = phase['connection']
            if action == 'replace':
                connection.commit()
                connection.begin()
            else:
                getattr(connection, action)()

    class CallbackType(TypeDecorator):
        impl = type(original_type)
        cache_ok = False

        @property
        def python_type(self):
            return original_type.python_type

        def process_bind_param(self, value, dialect):
            if processor == 'bind':
                attack()
            return value

        def process_result_value(self, value, dialect):
            if processor == 'result':
                attack()
            return value

    @contextmanager
    def transport(connection, **kwargs):
        with original_transport(connection, **kwargs) as guard:
            if phase['incident']:
                phase['attacked'] = True
                if processor in {'bind', 'result'}:
                    metadata_patch.setattr(column, 'type', CallbackType())
                elif processor == 'type_bind':
                    metadata_patch.setattr(
                        original_type,
                        'bind_processor',
                        lambda dialect: lambda value: (attack(), value)[1],
                    )
                elif processor == 'type_result':
                    metadata_patch.setattr(
                        original_type,
                        'result_processor',
                        lambda dialect, coltype: lambda value: (attack(), value)[1],
                    )
                elif processor == 'default':
                    metadata_patch.setattr(
                        column, 'default', ColumnDefault(lambda: (attack(), None)[1])
                    )
                elif processor == 'onupdate':
                    metadata_patch.setattr(
                        column, 'onupdate', ColumnDefault(lambda: (attack(), None)[1])
                    )
                elif processor == 'column_name':
                    metadata_patch.setattr(column, 'name', 'unexpected_column')
                elif processor == 'table_name':
                    metadata_patch.setattr(model.__table__, 'name', 'unexpected_table')
                elif processor == 'table_schema':
                    metadata_patch.setattr(
                        model.__table__, 'schema', 'unexpected_schema'
                    )
                elif processor == 'type_options':
                    metadata_patch.setattr(original_type, 'precision', 1)
                    metadata_patch.setattr(original_type, 'scale', 0)
            yield guard

    def append(self, connection, payload, **kwargs):
        if kwargs.get('provider_incident') is None:
            return original_append(self, connection, payload, **kwargs)
        phase.update(
            incident=True,
            connection=connection,
            before=database_image(connection),
            marker=self._authority.marker_path.read_bytes(),
            latch=h.authority._provider_safety_release_peer._provider_safety._latch_path.read_bytes(),
        )
        try:
            return original_append(self, connection, payload, **kwargs)
        finally:
            metadata_patch.undo()

    def observe(_connection, _cursor, sql, *_):
        if phase['incident'] and sql.lstrip().upper().startswith(
            ('INSERT', 'UPDATE', 'DELETE')
        ):
            phase['dml'].append(sql)

    monkeypatch.setattr(h.authority, '_authority_transport', transport)
    monkeypatch.setattr(RagReleaseLedger, 'append', append)
    event.listen(h.engine, 'before_cursor_execute', observe)
    failure = None
    try:
        incident_abort(h, category=category)
    except Exception as exc:
        failure = exc
    with h.engine.connect() as connection:
        after = database_image(connection)
    assert phase.get('attacked'), repr(failure)
    assert phase['calls'] == [], 'shared SQL processor ran after first DML'
    assert failure is None, repr(failure)
    assert_complete_incident(
        h, phase['before'], after, phase['marker'], phase['latch'], category
    )


@pytest.mark.parametrize('model', MODELS, ids=[m.__tablename__ for m in MODELS])
@pytest.mark.parametrize(
    'attack',
    [
        'default',
        'onupdate',
        'column_name',
        'table_name',
        'table_schema',
        'type_bind',
        'type_result',
    ],
)
def test_incident_shared_schema_mutation_cannot_change_owned_sql(
    tmp_path, monkeypatch, model, attack
):
    _metadata_attack(tmp_path, monkeypatch, model, 'created_at', attack, 'commit')


def test_incident_shared_numeric_options_do_not_change_accounting(
    tmp_path, monkeypatch
):
    _metadata_attack(
        tmp_path,
        monkeypatch,
        RagProviderReadiness,
        'overrun_cost_usd',
        'type_options',
        'commit',
    )


@pytest.mark.parametrize(
    'category',
    [
        'provider_usage_overrun',
        'provider_response_identity_invalid',
        'provider_embedding_payload_invalid',
        'provider_safety_unavailable',
    ],
)
def test_incident_outcomes_commit_complete_cross_authority_state(
    tmp_path, monkeypatch, category
):
    _metadata_attack(
        tmp_path,
        monkeypatch,
        RagProviderReadiness,
        'overrun_cost_usd',
        'bind',
        'commit',
        category=category,
    )


def test_incident_accepts_complete_prior_provider_history(tmp_path, monkeypatch):
    h = incident_harness(tmp_path, monkeypatch, prior_provider_history=True)
    incident_abort(h)
    with h.engine.connect() as connection:
        image = database_image(connection)
    assert image[PROVIDER[0]][0]['global_safety_generation'] == 3
    assert [row['global_safety_generation'] for row in image[PROVIDER[2]]] == [
        0,
        1,
        2,
        3,
    ]
    assert [row['transition_kind'] for row in image[PROVIDER[2]]] == [
        'bootstrap',
        'block_overrun',
        'reset',
        'block_overrun',
    ]


def test_incident_preparation_refuses_preexisting_historical_digest_corruption(
    tmp_path, monkeypatch
):
    from decimal import Decimal

    h = incident_harness(tmp_path, monkeypatch, prior_provider_history=True)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        connection.execute(
            text(
                'UPDATE rag_provider_safety_transitions '
                'SET envelope_digest = :value WHERE global_safety_generation = 1'
            ),
            {'value': 'f' * 64},
        )
        connection.commit()
        before = database_image(connection)
    latch = runtime._latch_path.read_bytes()
    with h.engine.connect() as connection, pytest.raises(RagProviderSafetyError):
        h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=99,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    with h.engine.connect() as connection:
        assert database_image(connection) == before
    assert runtime._latch_path.read_bytes() == latch


@pytest.mark.parametrize(
    'damage',
    [
        'history_missing',
        'history_digest',
        'history_roster',
        'history_actor',
        'history_review_reference',
        'history_chain',
        'history_old_digest',
        'history_reordered',
        'history_kind',
        'inactive_extra',
    ],
)
def test_incident_refuses_inconsistent_complete_roster_before_latch_or_dml(
    tmp_path, monkeypatch, damage
):
    h = incident_harness(
        tmp_path,
        monkeypatch,
        prior_provider_history=damage
        in {'history_chain', 'history_old_digest', 'history_reordered', 'history_kind'},
    )
    original = RagReleaseLedger.append
    phase = {'dml': []}

    def append(self, connection, payload, **kwargs):
        if kwargs.get('provider_incident') is None:
            return original(self, connection, payload, **kwargs)
        if damage == 'history_missing':
            connection.exec_driver_sql('DELETE FROM rag_provider_safety_transitions')
        elif damage == 'history_digest':
            connection.execute(
                text(
                    'UPDATE rag_provider_safety_transitions SET envelope_digest = :value'
                ),
                {'value': 'f' * 64},
            )
        elif damage == 'history_roster':
            connection.exec_driver_sql(
                'UPDATE rag_provider_safety_transitions SET authority_id = 2'
            )
        elif damage == 'history_actor':
            connection.execute(
                text(
                    'UPDATE rag_provider_safety_transitions '
                    'SET actor_subject_hmac = :value'
                ),
                {'value': 'f' * 64},
            )
        elif damage == 'history_review_reference':
            connection.execute(
                text(
                    'UPDATE rag_provider_safety_transitions '
                    'SET reviewed_transition_reference_hmac = :value'
                ),
                {'value': 'f' * 64},
            )
        elif damage == 'history_chain':
            connection.exec_driver_sql(
                'UPDATE rag_provider_safety_transitions '
                'SET prior_state_version = 99 WHERE global_safety_generation = 2'
            )
        elif damage == 'history_old_digest':
            connection.execute(
                text(
                    'UPDATE rag_provider_safety_transitions '
                    'SET envelope_digest = :value WHERE global_safety_generation = 1'
                ),
                {'value': 'f' * 64},
            )
        elif damage == 'history_reordered':
            connection.exec_driver_sql(
                'UPDATE rag_provider_safety_transitions '
                'SET id = 99 WHERE global_safety_generation = 0'
            )
        elif damage == 'history_kind':
            connection.exec_driver_sql(
                "UPDATE rag_provider_safety_transitions SET transition_kind = 'reset' "
                'WHERE global_safety_generation = 1'
            )
        else:
            connection.exec_driver_sql(
                "UPDATE rag_provider_readiness SET active = 0 WHERE component = 'query_embedding'"
            )
        connection.commit()
        phase['before'] = database_image(connection)
        phase['marker'] = h.authority.marker_path.read_bytes()
        phase['latch'] = (
            h.authority._provider_safety_release_peer._provider_safety._latch_path.read_bytes()
        )
        phase['armed'] = True
        return original(self, connection, payload, **kwargs)

    def observe(_connection, _cursor, sql, *_):
        if phase.get('armed') and sql.lstrip().upper().startswith(
            ('INSERT', 'UPDATE', 'DELETE')
        ):
            phase['dml'].append(sql)

    monkeypatch.setattr(RagReleaseLedger, 'append', append)
    event.listen(h.engine, 'before_cursor_execute', observe)
    with pytest.raises(RagProviderSafetyError):
        incident_abort(h)
    with h.engine.connect() as connection:
        assert database_image(connection) == phase['before']
    assert not phase['dml']
    assert h.authority.marker_path.read_bytes() == phase['marker']
    assert (
        h.authority._provider_safety_release_peer._provider_safety._latch_path.read_bytes()
        == phase['latch']
    )


@pytest.mark.parametrize(
    'damage', ['authority', 'readiness_peer', 'history', 'commit_failure']
)
def test_incident_database_failure_rolls_back_every_row_and_retains_fail_stop_latch(
    tmp_path, monkeypatch, damage
):
    h = incident_harness(tmp_path, monkeypatch)
    original = RagReleaseLedger.append
    phase = {}

    def append(self, connection, payload, **kwargs):
        if kwargs.get('provider_incident') is None:
            return original(self, connection, payload, **kwargs)
        phase['before'] = database_image(connection)
        phase['marker'] = h.authority.marker_path.read_bytes()
        phase['latch'] = (
            h.authority._provider_safety_release_peer._provider_safety._latch_path.read_bytes()
        )
        if damage == 'commit_failure':

            def fail_commit():
                raise RuntimeError('injected database commit failure')

            monkeypatch.setattr(connection, 'commit', fail_commit)
        else:
            # SQLite triggers simulate DB-side corruption; no caller SQL metadata
            # is used for the oracle. Trigger effects share the incident transaction.
            delta = {
                'authority': "UPDATE rag_provider_safety_authorities SET designated_environment_id = 'tampered'",
                'readiness_peer': "UPDATE rag_provider_readiness SET authorized_cost_policy_version = 'tampered' WHERE component = 'query_embedding'",
                'history': "UPDATE rag_provider_safety_transitions SET actor_subject_hmac = '"
                + 'f' * 64
                + "' WHERE global_safety_generation = 1",
            }[damage]
            connection.exec_driver_sql(
                'CREATE TEMP TRIGGER incident_damage AFTER INSERT ON rag_provider_safety_transitions BEGIN '
                + delta
                + '; END'
            )
        return original(self, connection, payload, **kwargs)

    monkeypatch.setattr(RagReleaseLedger, 'append', append)
    with pytest.raises((RagProviderSafetyError, RuntimeError)):
        incident_abort(h)
    with h.engine.connect() as connection:
        assert database_image(connection) == phase['before']
    runtime = h.authority._provider_safety_release_peer._provider_safety
    assert runtime._latch_path.read_bytes() != phase['latch']
    if damage != 'commit_failure':
        assert h.authority.marker_path.read_bytes() == phase['marker']
    with h.engine.connect() as connection, pytest.raises(RagProviderSafetyError):
        h.authority.inspect(connection, database_identity=h.database_identity)


def test_incident_owned_sql_matches_physical_native_columns_without_aliases():
    from backend.app.agent_runtime.rag_provider_schema import _provider_tables

    first, second = _provider_tables(), _provider_tables()
    for native, fresh, model in zip(first, second, MODELS, strict=True):
        declared = model.__table__
        assert native is not declared and native is not fresh
        assert native.name == declared.name
        assert [c.name for c in native.c] == [c.name for c in declared.c]
        for column, original, separate in zip(
            native.c, declared.c, fresh.c, strict=True
        ):
            assert column is not original and column.type is not original.type
            assert column is not separate and column.type is not separate.type
            assert type(column.type) is type(original.type)
            assert str(column.type) == str(original.type)
            for attribute in ('length', 'precision', 'scale', 'timezone'):
                assert getattr(column.type, attribute, None) == getattr(
                    original.type, attribute, None
                )
            assert column.nullable == original.nullable
            assert column.primary_key == original.primary_key
            assert column.autoincrement == original.autoincrement
            assert column.default is None and column.onupdate is None


@pytest.mark.parametrize(
    'target',
    [
        '_match_db_whole_set',
        '_match_database_rows',
        '_validate_envelope',
        '_signature',
        '_file_digest_bytes',
        '_apply_release_incident',
    ],
)
def test_incident_does_not_reenter_overridable_service_after_dml(
    tmp_path, monkeypatch, target
):
    h = incident_harness(tmp_path, monkeypatch)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    original_append = RagReleaseLedger.append
    original_method = getattr(runtime, target)
    phase = {'incident': False, 'dml': [], 'calls': []}

    def append(self, connection, payload, **kwargs):
        if kwargs.get('provider_incident') is not None:
            phase['incident'] = True
            phase['connection'] = connection
            phase['before'] = database_image(connection)
            phase['marker'] = self._authority.marker_path.read_bytes()
            phase['latch'] = runtime._latch_path.read_bytes()
        return original_append(self, connection, payload, **kwargs)

    def callback(*args, **kwargs):
        if target == '_apply_release_incident':
            result = original_method(*args, **kwargs)
            attack()
            return result
        attack()
        return original_method(*args, **kwargs)

    def attack():
        if phase['incident'] and phase['dml'] and not phase['calls']:
            phase['calls'].append(target)
            phase['connection'].commit()

    def observe(_connection, _cursor, sql, *_):
        if phase['incident'] and sql.lstrip().upper().startswith(
            ('INSERT', 'UPDATE', 'DELETE')
        ):
            phase['dml'].append(sql)

    monkeypatch.setattr(RagReleaseLedger, 'append', append)
    monkeypatch.setattr(runtime, target, callback)
    event.listen(h.engine, 'before_cursor_execute', observe)
    failure = None
    try:
        incident_abort(h)
    except Exception as exc:
        failure = exc
    assert phase['calls'] == [], 'overridable service callback ran after first DML'
    assert failure is None, repr(failure)
    with h.engine.connect() as connection:
        after = database_image(connection)
    assert_complete_incident(
        h,
        phase['before'],
        after,
        phase['marker'],
        phase['latch'],
        'provider_usage_overrun',
    )


def test_release_inspection_rejects_committed_provider_only_incident(
    tmp_path, monkeypatch
):
    from decimal import Decimal

    from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
    from backend.app.rag.release_authority import RagReleaseAuthorityError

    h = incident_harness(tmp_path, monkeypatch)
    h.claim()
    h.claim_generation()
    run = h.records('agent_run')[-1]
    with h.engine.connect() as connection:
        incident = h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=run['id'],
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    marker_before = h.authority.marker_path.read_bytes()
    marker = DurableFileAuthority.open_runtime(h.authority.marker_path)
    with (
        h.engine.connect() as connection,
        h.authority._authority_barrier(connection, marker=marker) as guard,
    ):
        guard.apply_provider_incident(incident)
        connection.commit()
    with h.engine.connect() as connection:
        image = database_image(connection)
        with pytest.raises(RagReleaseAuthorityError):
            h.authority.inspect(connection, database_identity=h.database_identity)
    assert image[PROVIDER[0]][0]['global_safety_generation'] == 1
    assert image['rag_live_gate_authorizations'][0]['state'] == 'started'
    assert h.authority.marker_path.read_bytes() == marker_before


def test_reviewed_snapshot_abort_accepts_exact_provider_drift_and_preserves_terminal_rows(
    tmp_path, monkeypatch
):
    from decimal import Decimal

    h = incident_harness(tmp_path, monkeypatch)
    h.claim()
    h.fail_case(outcome='model_unavailable')
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        runtime.block_overrun(
            connection,
            'answer_generation',
            agent_run_id=h.records('agent_run')[-1]['id'],
            input_tokens=1,
            output_tokens=2,
            cost_usd=Decimal('0.000001'),
        )
        before = database_image(connection)
    authorization = h.records('authorization')[0]
    marker_before = h.authority.marker_path.read_bytes()
    latch_before = runtime._latch_path.read_bytes()
    dml = []

    def observe(_connection, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    event.listen(h.engine, 'before_cursor_execute', observe)
    h.append(
        'authorization_abort_snapshot',
        [
            (
                'authorization',
                authorization,
                {**authorization, 'state': 'aborted_provider_safety'},
            )
        ],
        outcome='provider_safety_unavailable',
    )
    with h.engine.connect() as connection:
        after = database_image(connection)

    assert len(dml) == 3
    assert all(after[name] == before[name] for name in PROVIDER)
    assert runtime._latch_path.read_bytes() == latch_before
    assert h.authority.marker_path.read_bytes() != marker_before
    assert after['rag_live_gate_authorizations'][0] == {
        **before['rag_live_gate_authorizations'][0],
        'state': 'aborted_provider_safety',
    }
    for name in (
        'rag_live_gate_cases',
        'rag_live_gate_dispatches',
        'agent_runs',
        'agent_run_cost_components',
        'rag_live_gate_quality_reports',
    ):
        assert after[name] == before[name]


@pytest.mark.parametrize(
    'damage', ['actor_subject_hmac', 'agent_run_id', 'overrun_observed_at']
)
def test_incident_preparation_authenticates_historical_blocker_attribution(
    tmp_path, monkeypatch, damage
):
    from decimal import Decimal

    h = incident_harness(tmp_path, monkeypatch, prior_provider_history=True)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        from datetime import UTC, datetime

        value = (
            'f' * 64
            if damage == 'actor_subject_hmac'
            else 999
            if damage == 'agent_run_id'
            else datetime(2000, 1, 1, tzinfo=UTC)
        )
        table = (
            'rag_provider_safety_transitions'
            if damage != 'overrun_observed_at'
            else 'rag_provider_readiness'
        )
        predicate = (
            'global_safety_generation = 1'
            if damage != 'overrun_observed_at'
            else "component = 'answer_generation'"
        )
        connection.execute(
            text(f'UPDATE {table} SET {damage} = :value WHERE {predicate}'),
            {'value': value},
        )
        connection.commit()
        before = database_image(connection)
    marker_before = h.authority.marker_path.read_bytes()
    latch_before = runtime._latch_path.read_bytes()
    dml = []

    def observe(_connection, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    event.listen(h.engine, 'before_cursor_execute', observe)
    with h.engine.connect() as connection, pytest.raises(RagProviderSafetyError):
        h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=99,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    with h.engine.connect() as connection:
        assert database_image(connection) == before
    assert dml == []
    assert h.authority.marker_path.read_bytes() == marker_before
    assert runtime._latch_path.read_bytes() == latch_before


@pytest.mark.parametrize('boundary', ['rebind', 'supersession'])
def test_incident_preparation_refuses_unauthenticated_history_prefix(
    tmp_path, monkeypatch, boundary
):
    from dataclasses import replace
    from decimal import Decimal

    from backend.tests.test_rag_v2_costs import _snapshot
    from backend.tests.test_rag_v2_provider_safety import _reviewed_command

    h = incident_harness(tmp_path, monkeypatch, prior_provider_history=True)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        if boundary == 'rebind':
            command = _reviewed_command(
                runtime,
                connection,
                'answer_generation',
                'mark_rebind_required',
            )
            runtime.mark_rebind_required(connection, command)
            successor = _snapshot('answer_generation')
            command = _reviewed_command(
                runtime,
                connection,
                'answer_generation',
                'rebind',
                successor=successor,
            )
            runtime.reviewed_rebind(connection, command, successor)
        else:
            successor = replace(
                _snapshot('answer_generation'),
                model='gpt-5.6-luna',
                reasoning_or_config_identity='reasoning:low',
                authorized_policy_snapshot_hmac='e' * 64,
            )
            command = _reviewed_command(
                runtime,
                connection,
                'answer_generation',
                'supersession',
                successor=successor,
                historical_block_acknowledged=True,
            )
            runtime.reviewed_supersession(connection, command, successor)
        connection.execute(
            text(
                'UPDATE rag_provider_safety_transitions '
                'SET envelope_digest = :value WHERE global_safety_generation = 1'
            ),
            {'value': 'f' * 64},
        )
        connection.commit()
        before = database_image(connection)
    marker_before = h.authority.marker_path.read_bytes()
    latch_before = runtime._latch_path.read_bytes()
    dml = []

    def observe(_connection, _cursor, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(sql)

    event.listen(h.engine, 'before_cursor_execute', observe)
    with h.engine.connect() as connection, pytest.raises(RagProviderSafetyError):
        h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=99,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    with h.engine.connect() as connection:
        assert database_image(connection) == before
    assert dml == []
    assert h.authority.marker_path.read_bytes() == marker_before
    assert runtime._latch_path.read_bytes() == latch_before


def test_incident_preparation_requires_rebootstrap_after_legacy_rebind(
    tmp_path, monkeypatch
):
    from decimal import Decimal

    from backend.tests.test_rag_v2_costs import _snapshot
    from backend.tests.test_rag_v2_provider_safety import _reviewed_command

    h = incident_harness(tmp_path, monkeypatch, prior_provider_history=True)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        command = _reviewed_command(
            runtime,
            connection,
            'answer_generation',
            'mark_rebind_required',
        )
        runtime.mark_rebind_required(connection, command)
        successor = _snapshot('answer_generation')
        command = _reviewed_command(
            runtime,
            connection,
            'answer_generation',
            'rebind',
            successor=successor,
        )
        runtime.reviewed_rebind(connection, command, successor)
        before = database_image(connection)
    latch_before = runtime._latch_path.read_bytes()
    with (
        h.engine.connect() as connection,
        pytest.raises(RagProviderSafetyError, match='rebootstrap'),
    ):
        h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=99,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    with h.engine.connect() as connection:
        assert database_image(connection) == before
    assert runtime._latch_path.read_bytes() == latch_before


def test_incident_preparation_authenticates_signed_remediation_category(
    tmp_path, monkeypatch
):
    from decimal import Decimal

    from backend.tests.test_rag_v2_provider_safety import _reviewed_command

    h = incident_harness(tmp_path, monkeypatch)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        runtime.block_remediation(
            connection,
            'answer_generation',
            category='provider_response_identity_invalid',
            agent_run_id=17,
            input_tokens=1,
            output_tokens=2,
            cost_usd=Decimal('0.000001'),
        )
        reset = _reviewed_command(
            runtime,
            connection,
            'answer_generation',
            'reset',
            historical_block_acknowledged=True,
        )
        runtime.reviewed_reset(connection, reset)
        wrong_reference = runtime._incident_reference(
            component='answer_generation',
            state='blocked_remediation',
            agent_run_id=17,
            category='provider_safety_unavailable',
        )
        connection.execute(
            text(
                'UPDATE rag_provider_safety_transitions '
                'SET actor_subject_hmac = :value '
                'WHERE global_safety_generation = 1'
            ),
            {'value': wrong_reference},
        )
        connection.commit()
        before = database_image(connection)
    latch_before = runtime._latch_path.read_bytes()
    with h.engine.connect() as connection, pytest.raises(RagProviderSafetyError):
        h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=99,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    with h.engine.connect() as connection:
        assert database_image(connection) == before
    assert runtime._latch_path.read_bytes() == latch_before


def test_incident_preparation_authenticates_history_through_supersession(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from decimal import Decimal

    from backend.tests.test_rag_v2_costs import _snapshot
    from backend.tests.test_rag_v2_provider_safety import _reviewed_command

    h = incident_harness(tmp_path, monkeypatch, prior_provider_history=True)
    runtime = h.authority._provider_safety_release_peer._provider_safety
    with h.engine.connect() as connection:
        successor = replace(
            _snapshot('answer_generation'),
            model='gpt-5.6-luna',
            reasoning_or_config_identity='reasoning:low',
            authorized_policy_snapshot_hmac='e' * 64,
        )
        command = _reviewed_command(
            runtime,
            connection,
            'answer_generation',
            'supersession',
            successor=successor,
            historical_block_acknowledged=True,
        )
        runtime.reviewed_supersession(connection, command, successor)
        before = database_image(connection)
        plan = h.authority._provider_safety_release_peer.prepare_incident(
            connection,
            component='answer_generation',
            category='provider_usage_overrun',
            agent_run_id=99,
            input_tokens=20,
            output_tokens=30,
            cost_usd=Decimal('0.100000'),
        )
    with h.engine.connect() as connection:
        assert database_image(connection) == before
    assert plan.component == 'answer_generation'
    assert plan.category == 'provider_usage_overrun'
