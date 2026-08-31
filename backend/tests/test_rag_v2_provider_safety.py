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
    recovered.recover_partial_bootstrap(
        recovery,
        (_snapshot('query_embedding'), _snapshot('answer_generation')),
        reviewed_transition_reference_hmac=_PLAN_REFERENCE,
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

    service.mark_rebind_required(
        connection,
        'answer_generation',
        expected_global_safety_generation=0,
        expected_state_version=1,
        reviewed_transition_reference_hmac='1' * 64,
    )
    rebound = replace(
        snapshots[1],
        authorized_model_config_snapshot_hmac='d' * 64,
        authorized_policy_snapshot_hmac='e' * 64,
    )
    service.reviewed_rebind(
        connection,
        'answer_generation',
        rebound,
        expected_global_safety_generation=1,
        expected_state_version=2,
        reviewed_transition_reference_hmac='2' * 64,
    )
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
    service.reviewed_reset(
        connection,
        'query_embedding',
        expected_global_safety_generation=3,
        expected_state_version=2,
        reviewed_transition_reference_hmac='3' * 64,
        actor_subject_hmac='4' * 64,
    )
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
    service.reviewed_supersession(
        connection,
        'answer_generation',
        successor,
        expected_global_safety_generation=4,
        expected_state_version=3,
        reviewed_transition_reference_hmac='5' * 64,
        actor_subject_hmac='6' * 64,
    )
    assert (
        service.require_ready(
            connection, 'answer_generation', successor
        ).global_safety_generation
        == 5
    )
    with pytest.raises(RagProviderSafetyError):
        service.reviewed_reset(
            connection,
            'query_embedding',
            expected_global_safety_generation=4,
            expected_state_version=2,
            reviewed_transition_reference_hmac='7' * 64,
            actor_subject_hmac='8' * 64,
        )

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
