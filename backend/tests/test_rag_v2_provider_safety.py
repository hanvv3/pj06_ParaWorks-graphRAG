from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from threading import Event, Thread

import pytest
from sqlalchemy import create_engine, func, select

from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyReviewAuthority,
    RagProviderSafetyService,
)
from backend.app.db.base import Base
from backend.app.models.rag_runtime import (
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)
from backend.tests.test_rag_v2_costs import _snapshot

_PLAN_REFERENCE = '9' * 64
_REVIEW_SECRET = b'provider-safety-test-secret'


def _bootstrap(service, connection, snapshots=None):
    selected = snapshots or (
        _snapshot('query_embedding'),
        _snapshot('answer_generation'),
    )
    service.bootstrap(
        connection,
        selected,
        reviewed_transition_reference_hmac=_PLAN_REFERENCE,
    )
    return selected


def _reviewed_command(
    service,
    connection,
    component,
    operation,
    *,
    successor=None,
    historical_block_acknowledged=False,
):
    context = service.review_context(connection, component)
    return RagProviderSafetyReviewAuthority(
        identity_secret=_REVIEW_SECRET
    ).issue(
        context,
        operation=operation,
        successor=successor,
        actor_subject_hmac='4' * 64,
        reviewed_gate_reference_hmac='5' * 64,
        historical_block_acknowledged=historical_block_acknowledged,
    )


def test_exact_two_families_bootstrap_and_external_first_block_persists(tmp_path: Path):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    service = RagProviderSafetyService(
        latch_path=tmp_path / 'provider.json',
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    _bootstrap(service, connection)
    assert (
        service.require_ready(
            connection, 'query_embedding', _snapshot('query_embedding')
        ).family_state
        == 'ready'
    )
    service.block_overrun(
        connection,
        'query_embedding',
        agent_run_id=17,
        input_tokens=20,
        output_tokens=0,
        cost_usd=Decimal('0.000011'),
    )
    with pytest.raises(RagProviderSafetyError, match='blocked'):
        service.require_ready(
            connection, 'query_embedding', _snapshot('query_embedding')
        )

    restarted = RagProviderSafetyService(
        latch_path=tmp_path / 'provider.json',
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    with pytest.raises(RagProviderSafetyError, match='blocked'):
        restarted.require_ready(
            connection, 'query_embedding', _snapshot('query_embedding')
        )


def test_disabled_path_validation_does_not_create_latch(tmp_path: Path):
    path = tmp_path / 'disabled.json'
    RagProviderSafetyService.validate_disabled_path(path)
    assert not path.exists()
    assert not Path(str(path) + '.lock').exists()


def test_bootstrap_rejects_non_frozen_active_family_without_creating_artifacts(
    tmp_path: Path,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    snapshots = (
        _snapshot('query_embedding'),
        replace(_snapshot('answer_generation'), model='unreviewed-model'),
    )
    with pytest.raises(RagProviderSafetyError, match='frozen provider family'):
        _bootstrap(service, connection, snapshots)
    assert not path.exists()
    assert not Path(str(path) + '.lock').exists()


def test_db_rollback_cannot_clear_external_first_blocker(tmp_path: Path):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    service = RagProviderSafetyService(
        latch_path=tmp_path / 'provider.json',
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    snapshots = _bootstrap(service, connection)

    class FailingCommit:
        def execute(self, *args, **kwargs):
            return connection.execute(*args, **kwargs)

        def commit(self):
            raise RuntimeError('simulated DB commit failure')

        def rollback(self):
            connection.rollback()

    with pytest.raises(RagProviderSafetyError, match='persisted'):
        service.block_remediation(
            FailingCommit(),  # type: ignore[arg-type]
            'answer_generation',
            category='provider_safety_unavailable',
            agent_run_id=18,
            input_tokens=4,
            output_tokens=2,
            cost_usd=Decimal('0.000050'),
        )
    with pytest.raises(RagProviderSafetyError, match='blocked'):
        service.require_ready(connection, 'answer_generation', snapshots[1])


def test_sidecar_stays_locked_through_db_commit_and_other_worker_makes_zero_call(
    tmp_path: Path,
):
    engine = create_engine(
        f'sqlite+pysqlite:///{(tmp_path / "provider.db").as_posix()}',
        connect_args={'check_same_thread': False},
    )
    Base.metadata.create_all(engine)
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    with engine.connect() as connection:
        snapshots = _bootstrap(service, connection)

    commit_entered = Event()
    allow_commit = Event()
    observer_started = Event()
    observer_finished = Event()
    blocker_errors: list[BaseException] = []
    observer_errors: list[BaseException] = []
    provider_calls: list[str] = []
    block_connection = engine.connect()
    observer_connection = engine.connect()

    class PausingCommit:
        dialect = block_connection.dialect

        def execute(self, *args, **kwargs):
            return block_connection.execute(*args, **kwargs)

        def scalar(self, *args, **kwargs):
            return block_connection.scalar(*args, **kwargs)

        def commit(self):
            commit_entered.set()
            if not allow_commit.wait(5):
                raise RuntimeError('test did not release DB commit')
            block_connection.commit()

        def rollback(self):
            block_connection.rollback()

    def block() -> None:
        try:
                service.block_remediation(
                    PausingCommit(),  # type: ignore[arg-type]
                    'answer_generation',
                    category='provider_safety_unavailable',
                agent_run_id=18,
                input_tokens=4,
                output_tokens=2,
                cost_usd=Decimal('0.000050'),
            )
        except BaseException as exc:
            blocker_errors.append(exc)

    def observe() -> None:
        observer_started.set()
        try:
            service.require_ready(
                observer_connection, 'answer_generation', snapshots[1]
            )
            provider_calls.append('called')
        except BaseException as exc:
            observer_errors.append(exc)
        finally:
            observer_finished.set()

    blocker = Thread(target=block)
    observer = Thread(target=observe)
    blocker.start()
    assert commit_entered.wait(5)
    observer.start()
    assert observer_started.wait(5)
    assert not observer_finished.wait(0.2)
    allow_commit.set()
    blocker.join(5)
    observer.join(5)
    block_connection.close()
    observer_connection.close()

    assert not blocker.is_alive()
    assert not observer.is_alive()
    assert blocker_errors == []
    assert len(observer_errors) == 1
    assert isinstance(observer_errors[0], RagProviderSafetyError)
    assert provider_calls == []


def test_bootstrap_writes_exact_whole_set_envelope_and_one_history_generation(
    tmp_path: Path,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    _bootstrap(service, connection)

    envelope = json.loads(path.read_text(encoding='utf-8'))
    assert set(envelope) == {'hmac_sha256', 'signed_payload'}
    assert set(envelope['signed_payload']) == {
        'body',
        'fingerprint_key_material_verifier',
        'fingerprint_key_version',
    }
    body = envelope['signed_payload']['body']
    assert set(body) == {
        'active_family_by_component',
        'authority_uuid',
        'designated_environment_id',
        'family_records',
        'global_safety_generation',
        'latch_schema_version',
    }
    assert set(body['active_family_by_component']) == {
        'answer_generation',
        'query_embedding',
    }
    expected_record_keys = {
        'authorized_policy_snapshot_hmac',
        'component',
        'family_safety_generation',
        'first_blocker_agent_run_hmac',
        'first_blocker_category',
        'first_blocker_observed_at',
        'model',
        'provider',
        'reasoning_or_config_identity',
        'reviewed_transition_reference_hmac',
        'state',
        'state_version',
    }
    assert [record['component'] for record in body['family_records']] == [
        'answer_generation',
        'query_embedding',
    ]
    assert all(set(record) == expected_record_keys for record in body['family_records'])
    assert (
        connection.scalar(select(func.count()).select_from(RagProviderReadiness)) == 2
    )
    transitions = connection.execute(
        select(
            RagProviderSafetyTransition.global_safety_generation,
            RagProviderSafetyTransition.transition_kind,
            RagProviderSafetyTransition.readiness_id,
        )
    ).all()
    assert transitions == [(0, 'bootstrap', None)]


def test_bootstrap_refuses_second_init_and_recovers_only_external_generation_zero(
    tmp_path: Path,
):
    path = tmp_path / 'provider.json'
    first_engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(first_engine)
    first = first_engine.connect()
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )

    class FailingCommit:
        def execute(self, *args, **kwargs):
            return first.execute(*args, **kwargs)

        def scalar(self, *args, **kwargs):
            return first.scalar(*args, **kwargs)

        def commit(self):
            raise RuntimeError('simulated bootstrap commit failure')

        def rollback(self):
            first.rollback()

    with pytest.raises(RagProviderSafetyError, match='bootstrap failed'):
        _bootstrap(service, FailingCommit())
    assert path.is_file()
    original_envelope = json.loads(path.read_text(encoding='utf-8'))
    assert (
        first.scalar(select(func.count()).select_from(RagProviderSafetyAuthority)) == 0
    )

    recovery_engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(recovery_engine)
    recovery = recovery_engine.connect()
    recovered = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    bad_body = json.loads(json.dumps(original_envelope['signed_payload']['body']))
    bad_body['family_records'][0]['state_version'] = 2
    recovered._authority.write(
        recovered._wrap(
            bad_body,
            key_version=original_envelope['signed_payload'][
                'fingerprint_key_version'
            ],
            key_verifier=original_envelope['signed_payload'][
                'fingerprint_key_material_verifier'
            ],
        )
    )
    with pytest.raises(RagProviderSafetyError, match='recovery shape'):
        recovered.recover_partial_bootstrap(
            recovery,
            (_snapshot('query_embedding'), _snapshot('answer_generation')),
        )
    recovered._authority.write(original_envelope)
    recovered.recover_partial_bootstrap(
        recovery,
        (_snapshot('query_embedding'), _snapshot('answer_generation')),
    )
    assert (
        recovered.require_ready(
            recovery, 'answer_generation', _snapshot('answer_generation')
        ).global_safety_generation
        == 0
    )
    with pytest.raises(RagProviderSafetyError, match='already exists'):
        _bootstrap(recovered, recovery)


def test_rebind_reset_and_supersession_are_cas_bound_and_preserve_history(
    tmp_path: Path,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'provider-safety-test-secret',
        designated_environment_id='test',
    )
    snapshots = _bootstrap(service, connection)

    mark_command = _reviewed_command(
        service, connection, 'answer_generation', 'mark_rebind_required'
    )
    service.mark_rebind_required(connection, mark_command)
    with pytest.raises(RagProviderSafetyError, match='already consumed'):
        service.mark_rebind_required(connection, mark_command)
    rebound = replace(
        snapshots[1],
        authorized_model_config_snapshot_hmac='d' * 64,
        authorized_policy_snapshot_hmac='e' * 64,
    )
    rebind_command = _reviewed_command(
        service,
        connection,
        'answer_generation',
        'rebind',
        successor=rebound,
    )
    service.reviewed_rebind(connection, rebind_command, rebound)
    assert (
        service.require_ready(
            connection, 'answer_generation', rebound
        ).global_safety_generation
        == 2
    )

    service.block_overrun(
        connection,
        'query_embedding',
        agent_run_id=17,
        input_tokens=20,
        output_tokens=0,
        cost_usd=Decimal('0.000011'),
    )
    reset_command = _reviewed_command(
        service,
        connection,
        'query_embedding',
        'reset',
        historical_block_acknowledged=True,
    )
    service.reviewed_reset(connection, reset_command)
    reset_row = connection.execute(
        select(RagProviderReadiness.reset_by, RagProviderReadiness.reset_at).where(
            RagProviderReadiness.component == 'query_embedding',
            RagProviderReadiness.active.is_(True),
        )
    ).one()
    assert reset_row.reset_by == '4' * 64
    assert reset_row.reset_at is not None
    assert (
        service.require_ready(
            connection, 'query_embedding', snapshots[0]
        ).global_safety_generation
        == 4
    )

    successor = replace(
        rebound,
        model='gpt-5.6-luna',
        reasoning_or_config_identity='reasoning:low',
        authorized_policy_snapshot_hmac='f' * 64,
    )
    supersession_command = _reviewed_command(
        service,
        connection,
        'answer_generation',
        'supersession',
        successor=successor,
    )
    service.reviewed_supersession(
        connection, supersession_command, successor
    )
    assert (
        service.require_ready(
            connection, 'answer_generation', successor
        ).global_safety_generation
        == 5
    )
    with pytest.raises(RagProviderSafetyError):
        stale = RagProviderSafetyReviewAuthority(
            identity_secret=_REVIEW_SECRET
        ).issue(
            service.review_context(connection, 'query_embedding'),
            operation='reset',
            successor=None,
            actor_subject_hmac='7' * 64,
            reviewed_gate_reference_hmac='8' * 64,
            historical_block_acknowledged=True,
        )
        service.reviewed_reset(connection, stale)
    with pytest.raises(RagProviderSafetyError, match='capability'):
        service.reviewed_reset(connection, '8' * 64)  # type: ignore[arg-type]

    body = json.loads(path.read_text(encoding='utf-8'))['signed_payload']['body']
    assert len(body['family_records']) == 3
    query_record = next(
        record
        for record in body['family_records']
        if record['component'] == 'query_embedding'
    )
    assert query_record['state'] == 'ready'
    assert query_record['first_blocker_category'] == 'provider_usage_overrun'
    assert query_record['first_blocker_agent_run_hmac'] is not None
    answer_rows = connection.execute(
        select(RagProviderReadiness.model, RagProviderReadiness.active)
        .where(RagProviderReadiness.component == 'answer_generation')
        .order_by(RagProviderReadiness.id)
    ).all()
    assert [tuple(row) for row in answer_rows] == [
        ('gpt-5.4-mini-2026-03-17', False),
        ('gpt-5.6-luna', True),
    ]
    assert connection.execute(
        select(RagProviderSafetyTransition.global_safety_generation).order_by(
            RagProviderSafetyTransition.global_safety_generation
        )
    ).scalars().all() == [0, 1, 2, 3, 4, 5]


def test_inactive_same_component_blocker_requires_ack_for_later_supersession(
    tmp_path: Path,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=_REVIEW_SECRET,
        designated_environment_id='test',
    )
    snapshots = _bootstrap(service, connection)
    service.block_overrun(
        connection,
        'query_embedding',
        agent_run_id=31,
        input_tokens=2,
        output_tokens=0,
        cost_usd=Decimal('0.000001'),
    )
    successor_b = replace(
        snapshots[0],
        model='text-embedding-3-large',
        reasoning_or_config_identity='dimensions:3072',
        authorized_policy_snapshot_hmac='d' * 64,
    )
    acknowledged = _reviewed_command(
        service,
        connection,
        'query_embedding',
        'supersession',
        successor=successor_b,
        historical_block_acknowledged=True,
    )
    service.reviewed_supersession(connection, acknowledged, successor_b)

    context = service.review_context(connection, 'query_embedding')
    assert context.has_historical_blocker is True
    successor_c = replace(
        successor_b,
        model='text-embedding-4',
        authorized_policy_snapshot_hmac='e' * 64,
    )
    with pytest.raises(RagProviderSafetyError, match='acknowledgement is required'):
        RagProviderSafetyReviewAuthority(identity_secret=_REVIEW_SECRET).issue(
            context,
            operation='supersession',
            successor=successor_c,
            actor_subject_hmac='4' * 64,
            reviewed_gate_reference_hmac='5' * 64,
            historical_block_acknowledged=False,
        )
    body = json.loads(path.read_text(encoding='utf-8'))['signed_payload']['body']
    assert len(body['family_records']) == 3


def test_bootstrap_plan_reference_is_signed_and_recovery_rejects_wrong_shape(
    tmp_path: Path,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=_REVIEW_SECRET,
        designated_environment_id='test',
    )
    _bootstrap(service, connection)
    body = json.loads(path.read_text(encoding='utf-8'))['signed_payload']['body']
    assert {
        record['reviewed_transition_reference_hmac']
        for record in body['family_records']
    } == {_PLAN_REFERENCE}
    assert set(
        connection.execute(
            select(RagProviderReadiness.reviewed_gate_reference_hmac)
        ).scalars()
    ) == {_PLAN_REFERENCE}


def test_automatic_block_preserves_review_reference_and_separates_incident(
    tmp_path: Path,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=_REVIEW_SECRET,
        designated_environment_id='test',
    )
    _bootstrap(service, connection)
    service.block_remediation(
        connection,
        'query_embedding',
        category='provider_embedding_payload_invalid',
        agent_run_id=21,
        input_tokens=1,
        output_tokens=0,
        cost_usd=Decimal('0.000001'),
    )
    body = json.loads(path.read_text(encoding='utf-8'))['signed_payload']['body']
    record = next(
        item for item in body['family_records']
        if item['component'] == 'query_embedding'
    )
    assert record['reviewed_transition_reference_hmac'] == _PLAN_REFERENCE
    assert record['first_blocker_category'] == 'provider_embedding_payload_invalid'
    transition = connection.execute(
        select(RagProviderSafetyTransition)
        .where(RagProviderSafetyTransition.global_safety_generation == 1)
    ).mappings().one()
    assert transition['reviewed_transition_reference_hmac'] == _PLAN_REFERENCE
    assert transition['actor_subject_hmac'] != _PLAN_REFERENCE
    with pytest.raises(RagProviderSafetyError, match='category'):
        service.block_remediation(
            connection,
            'answer_generation',
            category='provider_usage_overrun',  # type: ignore[arg-type]
            agent_run_id=22,
            input_tokens=1,
            output_tokens=0,
            cost_usd=Decimal('0.000001'),
        )


def test_external_write_failure_commits_distinct_db_only_remediation_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    connection = engine.connect()
    path = tmp_path / 'provider.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=_REVIEW_SECRET,
        designated_environment_id='test',
    )
    snapshots = _bootstrap(service, connection)
    original = path.read_bytes()
    monkeypatch.setattr(
        service._authority,
        '_replace_unlocked',
        lambda _value: (_ for _ in ()).throw(OSError('simulated replace failure')),
    )
    with pytest.raises(RagProviderSafetyError, match='external transition failed'):
        service.block_remediation(
            connection,
            'answer_generation',
            category='provider_response_identity_invalid',
            agent_run_id=23,
            input_tokens=1,
            output_tokens=1,
            cost_usd=Decimal('0.000001'),
        )
    authority = connection.execute(
        select(RagProviderSafetyAuthority)
    ).mappings().one()
    transition = connection.execute(
        select(RagProviderSafetyTransition)
        .where(RagProviderSafetyTransition.global_safety_generation == 1)
    ).mappings().one()
    external_digest = RagProviderSafetyService._file_digest_bytes(original)
    assert authority['global_safety_generation'] == 1
    assert authority['envelope_digest'] != external_digest
    assert transition['envelope_digest'] == authority['envelope_digest']
    with pytest.raises(RagProviderSafetyError, match='drift'):
        service.require_ready(connection, 'answer_generation', snapshots[1])
