from __future__ import annotations

import hashlib
import hmac
import io
import socket
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

from backend.app.admin.rag_provider_safety import (
    MAX_REVIEW_ENVELOPE_BYTES,
    ProviderSafetyAdminResult,
    ProviderSafetyAdminStatus,
    ProviderSafetyAdminTarget,
    ProviderSafetyReviewError,
    RagProviderSafetyAdminService,
    _configured_target,
    _run_cli,
    build_cli_parser,
    successor_registry_from_snapshots,
    verify_review_envelope,
)
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyService,
)
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models.rag_runtime import (
    AgentRunCostComponent,
    RagProviderReadiness,
    RagProviderSafetyAuthority,
    RagProviderSafetyTransition,
)
from backend.tests.test_rag_v2_costs import _snapshot

_REVIEW_KEY = b'external-review-authority-key-material'
_RUNTIME_KEY = b'provider-safety-test-secret'
_PLAN_HMAC = '9' * 64


def _target(tmp_path: Path, *, kind: str = 'production') -> ProviderSafetyAdminTarget:
    return ProviderSafetyAdminTarget.build(
        kind=kind,
        database_url=f'sqlite+pysqlite:///{(tmp_path / f"{kind}.db").as_posix()}',
        latch_path=tmp_path / f'{kind}.json',
        designated_environment_id=f'{kind}-environment',
        review_secret=_REVIEW_KEY,
    )


def _successor(snapshot) -> dict[str, object]:
    return {
        'authorized_cost_policy_version': snapshot.authorized_cost_policy_version,
        'authorized_model_config_snapshot_hmac': (
            snapshot.authorized_model_config_snapshot_hmac
        ),
        'authorized_model_config_version': snapshot.authorized_model_config_version,
        'authorized_policy_snapshot_hmac': snapshot.authorized_policy_snapshot_hmac,
        'authorized_token_estimator_version': (
            snapshot.authorized_token_estimator_version
        ),
        'component': snapshot.component,
        'fingerprint_key_material_verifier': (
            snapshot.fingerprint_key_material_verifier
        ),
        'fingerprint_key_version': snapshot.fingerprint_key_version,
        'model': snapshot.model,
        'provider': snapshot.provider,
        'reasoning_or_config_identity': snapshot.reasoning_or_config_identity,
    }


def _context(context) -> dict[str, object]:
    return {
        'authority_uuid': context.authority_uuid,
        'component': context.component,
        'current_family_identity': list(context.family_identity),
        'current_state': context.state,
        'current_state_version': context.state_version,
        'designated_environment_id': context.designated_environment_id,
        'global_safety_generation': context.global_safety_generation,
        'has_historical_blocker': context.has_historical_blocker,
    }


def _review_bytes(
    target: ProviderSafetyAdminTarget,
    operation: str,
    *,
    context: object = None,
    successor: object = None,
    key: bytes = _REVIEW_KEY,
    key_id: str = 'provider-safety-review-v1',
    plan_hmac: str = _PLAN_HMAC,
    acknowledged: bool = False,
) -> bytes:
    payload = {
        'actor_subject_hmac': '4' * 64,
        'expected_context': context,
        'historical_block_acknowledged': acknowledged,
        'implementation_plan_reference_hmac': plan_hmac,
        'nonce': '879c1400-2f11-4ac6-82a1-112f8f2431cb',
        'operation': operation,
        'review_authority_key_id': key_id,
        'schema_version': 'rag-provider-safety-admin-review:v1',
        'successor': successor,
        'target': target.review_identity,
    }
    signature = hmac.new(
        key,
        b'paraworks:provider-safety-admin-review:v1\x00'
        + canonical_json_bytes(payload),
        hashlib.sha256,
    ).hexdigest()
    return canonical_json_bytes(
        {'hmac_sha256': signature, 'signed_payload': payload}
    )


def _service(tmp_path: Path, *, kind: str = 'production'):
    target = _target(tmp_path, kind=kind)
    engine = create_engine(target.database_url)
    Base.metadata.create_all(engine)
    runtime = RagProviderSafetyService(
        latch_path=target.latch_path,
        identity_secret=_RUNTIME_KEY,
        designated_environment_id=target.designated_environment_id,
    )
    admin = RagProviderSafetyAdminService(
        target=target,
        connection_factory=engine.connect,
        provider_safety=runtime,
        snapshots=(_snapshot('query_embedding'), _snapshot('answer_generation')),
        runtime_identity_secret=_RUNTIME_KEY,
        review_secret=_REVIEW_KEY,
        review_key_id='provider-safety-review-v1',
        implementation_plan_reference_hmac=_PLAN_HMAC,
        successor_registry={},
    )
    return engine, target, runtime, admin


def test_review_envelope_is_exact_bounded_and_binds_external_authority_target_and_plan(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    raw = _review_bytes(target, 'provider-safety-init')
    reviewed = verify_review_envelope(
        raw,
        expected_operation='provider-safety-init',
        expected_target=target,
        expected_context=None,
        expected_successor=None,
        review_secret=_REVIEW_KEY,
        review_key_id='provider-safety-review-v1',
        implementation_plan_reference_hmac=_PLAN_HMAC,
        successor_registry={},
    )
    assert reviewed.operation == 'provider-safety-init'
    assert reviewed.reviewed_gate_reference_hmac == _PLAN_HMAC

    refused = (
        raw + b' ',
        _review_bytes(target, 'provider-safety-init', key=b'wrong-review-key'),
        _review_bytes(target, 'provider-safety-reset'),
        _review_bytes(_target(tmp_path, kind='live_validation'), 'provider-safety-init'),
        _review_bytes(target, 'provider-safety-init', plan_hmac='8' * 64),
        _review_bytes(target, 'provider-safety-init', key_id='rotated-unapproved-key'),
        b'{' + b'x' * MAX_REVIEW_ENVELOPE_BYTES + b'}',
    )
    for candidate in refused:
        with pytest.raises(ProviderSafetyReviewError):
            verify_review_envelope(
                candidate,
                expected_operation='provider-safety-init',
                expected_target=target,
                expected_context=None,
                expected_successor=None,
                review_secret=_REVIEW_KEY,
                review_key_id='provider-safety-review-v1',
                implementation_plan_reference_hmac=_PLAN_HMAC,
                successor_registry={},
            )


def test_review_authority_key_must_be_distinct_from_runtime_latch_key(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    engine = create_engine(target.database_url)
    Base.metadata.create_all(engine)
    runtime = RagProviderSafetyService(
        latch_path=target.latch_path,
        identity_secret=_REVIEW_KEY,
        designated_environment_id=target.designated_environment_id,
    )
    with pytest.raises(ProviderSafetyReviewError, match='distinct'):
        RagProviderSafetyAdminService(
            target=target,
            connection_factory=engine.connect,
            provider_safety=runtime,
            snapshots=(_snapshot('query_embedding'), _snapshot('answer_generation')),
            runtime_identity_secret=_REVIEW_KEY,
            review_secret=_REVIEW_KEY,
            review_key_id='provider-safety-review-v1',
            implementation_plan_reference_hmac=_PLAN_HMAC,
            successor_registry={},
        )


def test_mutating_parser_accepts_no_review_payload_or_secret_arguments() -> None:
    parser = build_cli_parser()
    parsed = parser.parse_args(['provider-safety-reset'])
    assert vars(parsed) == {'command': 'provider-safety-reset'}
    with pytest.raises(ProviderSafetyReviewError, match='arguments'):
        parser.parse_args(['provider-safety-reset', '--review', 'secret'])


def test_status_is_read_only_and_does_not_create_authority_artifacts(
    tmp_path: Path,
) -> None:
    engine, target, _runtime, admin = _service(tmp_path)
    status = admin.status()
    assert status.authority_present is False
    assert status.ready_family_count == 0
    assert status.blocked_family_count == 0
    assert not target.latch_path.exists()
    assert not Path(str(target.latch_path) + '.lock').exists()
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(RagProviderSafetyAuthority)) == 0


def test_status_reports_db_latch_drift_without_repairing_or_recreating_file(
    tmp_path: Path,
) -> None:
    _engine, target, _runtime, admin = _service(tmp_path)
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    target.latch_path.unlink()
    with pytest.raises(RagProviderSafetyError, match='inconsistent'):
        admin.status()
    assert not target.latch_path.exists()


def test_init_requires_external_review_and_refuses_second_init(tmp_path: Path) -> None:
    engine, target, _runtime, admin = _service(tmp_path)
    with pytest.raises(ProviderSafetyReviewError):
        admin.initialize(b'')
    result = admin.initialize(_review_bytes(target, 'provider-safety-init'))
    assert result.operation == 'provider-safety-init'
    assert result.global_safety_generation == 0
    assert result.ready_family_count == 2
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(RagProviderReadiness)) == 2
        assert connection.scalar(select(func.count()).select_from(RagProviderSafetyTransition)) == 1
    with pytest.raises(RagProviderSafetyError, match='already exists'):
        admin.initialize(_review_bytes(target, 'provider-safety-init'))


def test_rebind_is_two_reviewed_transitions_and_rejects_direct_ready_rebind(
    tmp_path: Path,
) -> None:
    _engine, target, runtime, admin = _service(tmp_path)
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    with admin.connection_factory() as connection:
        context = runtime.review_context(connection, 'answer_generation')
    context_payload = _context(context)
    snapshot = _snapshot('answer_generation')
    with pytest.raises(RagProviderSafetyError, match='CAS'):
        admin.rebind(
            _review_bytes(
                target,
                'provider-safety-rebind',
                context=context_payload,
                successor=_successor(snapshot),
            )
        )
    admin.mark_rebind_required(
        _review_bytes(
            target,
            'provider-safety-mark-rebind-required',
            context=context_payload,
        )
    )
    with admin.connection_factory() as connection:
        marked = runtime.review_context(connection, 'answer_generation')
    marked_payload = {
        **context_payload,
        'current_state': 'rebind_required',
        'current_state_version': marked.state_version,
        'global_safety_generation': marked.global_safety_generation,
    }
    admin.rebind(
        _review_bytes(
            target,
            'provider-safety-rebind',
            context=marked_payload,
            successor=_successor(snapshot),
        )
    )
    with admin.connection_factory() as connection:
        assert runtime.review_context(connection, 'answer_generation').state == 'ready'


def test_supersession_requires_committed_registry_in_addition_to_signed_snapshot(
    tmp_path: Path,
) -> None:
    _engine, target, runtime, admin = _service(tmp_path)
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    successor = replace(
        _snapshot('answer_generation'),
        model='unregistered-successor',
        authorized_policy_snapshot_hmac='7' * 64,
    )
    with admin.connection_factory() as connection:
        context = runtime.review_context(connection, 'answer_generation')
    context_payload = _context(context)
    with pytest.raises(ProviderSafetyReviewError, match='registry'):
        admin.supersede(
            _review_bytes(
                target,
                'provider-safety-supersede',
                context=context_payload,
                successor=_successor(successor),
            )
        )


def test_production_review_cannot_authorize_live_validation_target(
    tmp_path: Path,
) -> None:
    _engine, production, _runtime, admin = _service(tmp_path)
    live = _target(tmp_path, kind='live_validation')
    with pytest.raises(ProviderSafetyReviewError, match='target'):
        admin.initialize(_review_bytes(live, 'provider-safety-init'))
    assert not production.latch_path.exists()
    assert not live.latch_path.exists()


def test_bootstrap_recovery_requires_a_separate_signed_review_and_only_repairs_file_first_shape(
    tmp_path: Path,
) -> None:
    engine, target, runtime, admin = _service(tmp_path)
    snapshots = (_snapshot('query_embedding'), _snapshot('answer_generation'))
    real_connection = engine.connect()

    class FailingCommit:
        dialect = real_connection.dialect

        def execute(self, *args, **kwargs):
            return real_connection.execute(*args, **kwargs)

        def scalar(self, *args, **kwargs):
            return real_connection.scalar(*args, **kwargs)

        def commit(self):
            raise RuntimeError('simulated crash')

        def rollback(self):
            real_connection.rollback()

    with pytest.raises(RagProviderSafetyError, match='bootstrap failed'):
        runtime.bootstrap(
            FailingCommit(),  # type: ignore[arg-type]
            snapshots,
            reviewed_transition_reference_hmac=_PLAN_HMAC,
        )
    real_connection.close()
    assert target.latch_path.exists()
    with pytest.raises(ProviderSafetyReviewError):
        admin.recover_bootstrap(b'')
    recovered = admin.recover_bootstrap(
        _review_bytes(target, 'provider-safety-bootstrap-recovery')
    )
    assert recovered.global_safety_generation == 0
    with pytest.raises(RagProviderSafetyError, match='not eligible'):
        admin.recover_bootstrap(
            _review_bytes(target, 'provider-safety-bootstrap-recovery')
        )


def test_reset_preserves_blocker_and_appends_monotonic_reviewed_history(
    tmp_path: Path,
) -> None:
    engine, target, runtime, admin = _service(tmp_path)
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    with admin.connection_factory() as connection:
        runtime.block_remediation(
            connection,
            'query_embedding',
            category='provider_safety_unavailable',
            agent_run_id=91,
            input_tokens=0,
            output_tokens=0,
            cost_usd=Decimal('0'),
        )
    with admin.connection_factory() as connection:
        context = runtime.review_context(connection, 'query_embedding')
    result = admin.reset(
        _review_bytes(
            target,
            'provider-safety-reset',
            context=_context(context),
            acknowledged=True,
        )
    )
    assert result.global_safety_generation == 2
    with engine.connect() as connection:
        row = connection.execute(
            select(RagProviderReadiness.__table__).where(
                RagProviderReadiness.component == 'query_embedding',
                RagProviderReadiness.active.is_(True),
            )
        ).mappings().one()
        assert row is not None
        assert row['state'] == 'ready'
        assert row['overrun_agent_run_id'] == 91
        assert connection.execute(
            select(RagProviderSafetyTransition.transition_kind).order_by(
                RagProviderSafetyTransition.global_safety_generation
            )
        ).scalars().all() == ['bootstrap', 'block_remediation', 'reset']


def test_registered_successor_can_supersede_without_widening_other_family(
    tmp_path: Path,
) -> None:
    engine, target, runtime, original = _service(tmp_path)
    original.initialize(_review_bytes(target, 'provider-safety-init'))
    successor = replace(
        _snapshot('answer_generation'),
        model='reviewed-model-snapshot',
        authorized_policy_snapshot_hmac='7' * 64,
    )
    admin = RagProviderSafetyAdminService(
        target=target,
        connection_factory=engine.connect,
        provider_safety=runtime,
        snapshots=(_snapshot('query_embedding'), _snapshot('answer_generation')),
        runtime_identity_secret=_RUNTIME_KEY,
        review_secret=_REVIEW_KEY,
        review_key_id='provider-safety-review-v1',
        implementation_plan_reference_hmac=_PLAN_HMAC,
        successor_registry=successor_registry_from_snapshots((successor,)),
    )
    with admin.connection_factory() as connection:
        context = runtime.review_context(connection, 'answer_generation')
    result = admin.supersede(
        _review_bytes(
            target,
            'provider-safety-supersede',
            context=_context(context),
            successor=_successor(successor),
        )
    )
    assert result.global_safety_generation == 1
    with engine.connect() as connection:
        active = connection.execute(
            select(
                RagProviderReadiness.component,
                RagProviderReadiness.model,
                RagProviderReadiness.active,
            ).order_by(RagProviderReadiness.id)
        ).all()
    assert active == [
        ('query_embedding', 'text-embedding-3-small', True),
        ('answer_generation', 'gpt-5.4-mini-2026-03-17', False),
        ('answer_generation', 'reviewed-model-snapshot', True),
    ]


def test_cli_status_reads_no_stdin_and_mutations_consume_exact_stdin_once() -> None:
    seen: list[bytes] = []

    class FakeAdmin:
        def status(self):
            return ProviderSafetyAdminStatus(False, None, 0, 0, 0)

        def initialize(self, raw):
            seen.append(raw)
            return ProviderSafetyAdminResult('provider-safety-init', 0, 2)

    status = _run_cli(
        ['provider-safety-status'],
        stdin=io.BytesIO(b'must-not-be-read'),
        service_factory=lambda: FakeAdmin(),
    )
    assert status.exit_code == 0
    assert status.payload == {
        'authority_present': False,
        'blocked_family_count': 0,
        'global_safety_generation': None,
        'ok': True,
        'ready_family_count': 0,
        'rebind_required_family_count': 0,
    }
    assert seen == []

    raw = b'{"review":"private"}'
    initialized = _run_cli(
        ['provider-safety-init'],
        stdin=io.BytesIO(raw),
        service_factory=lambda: FakeAdmin(),
    )
    assert initialized.exit_code == 0
    assert initialized.payload == {
        'global_safety_generation': 0,
        'ok': True,
        'operation': 'provider-safety-init',
        'ready_family_count': 2,
    }
    assert seen == [raw]
    assert b'private' not in canonical_json_bytes(initialized.payload)


def test_cli_failures_are_fixed_sanitized_codes_without_traceback_or_raw_input() -> None:
    class FailingAdmin:
        def reset(self, raw):
            assert b'super-secret-reviewed-payload' in raw
            raise ProviderSafetyReviewError('leaky raw message super-secret')

    outcome = _run_cli(
        ['provider-safety-reset'],
        stdin=io.BytesIO(b'super-secret-reviewed-payload'),
        service_factory=lambda: FailingAdmin(),
    )
    assert outcome.exit_code == 2
    assert outcome.payload == {'code': 'review_refused', 'ok': False}
    rendered = canonical_json_bytes(outcome.payload)
    assert b'super-secret' not in rendered
    assert b'Traceback' not in rendered


def test_configured_production_and_live_validation_targets_are_disjoint(
    tmp_path: Path,
) -> None:
    production_path = tmp_path / 'production.json'
    live_path = tmp_path / 'live.json'
    common = {
        'paraworks_demo_mode': False,
        'paraworks_database_url': (
            f'postgresql+psycopg://app:secret@localhost/{tmp_path.name}_app'
        ),
        'paraworks_provider_safety_latch_path': str(production_path),
        'paraworks_rag_live_validation_database_url': (
            f'postgresql+psycopg://gate:secret@localhost/{tmp_path.name}_gate'
        ),
        'paraworks_rag_live_validation_provider_safety_latch_path': str(live_path),
        'paraworks_rag_live_validation_environment_id': 'live-validation-test',
    }
    production = _configured_target(
        Settings(**common, paraworks_provider_safety_admin_target='production'),
        review_secret=_REVIEW_KEY,
    )
    live = _configured_target(
        Settings(**common, paraworks_provider_safety_admin_target='live_validation'),
        review_secret=_REVIEW_KEY,
    )
    assert production.kind == 'production'
    assert live.kind == 'live_validation'
    assert production.database_url != live.database_url
    assert production.latch_path != live.latch_path
    assert production.review_identity != live.review_identity

    same_url = common['paraworks_database_url']
    same_path = common['paraworks_provider_safety_latch_path']
    with pytest.raises(ProviderSafetyReviewError, match='separate'):
        _configured_target(
            Settings(
                **{
                    **common,
                    'paraworks_rag_live_validation_database_url': same_url,
                    'paraworks_rag_live_validation_provider_safety_latch_path': same_path,
                },
                paraworks_provider_safety_admin_target='live_validation',
            ),
            review_secret=_REVIEW_KEY,
        )
    with pytest.raises(ProviderSafetyReviewError, match='separate'):
        _configured_target(
            Settings(
                **{
                    **common,
                    'paraworks_database_url': (
                        'postgresql+psycopg://app:a@localhost:5432/shared_db'
                    ),
                    'paraworks_rag_live_validation_database_url': (
                        'postgresql+psycopg://gate:b@127.0.0.1:5432/shared_db'
                    ),
                },
                paraworks_provider_safety_admin_target='live_validation',
            ),
            review_secret=_REVIEW_KEY,
        )


def test_all_admin_transitions_are_provider_and_network_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from backend.app.agent_runtime.rag_provider_transport import (
        _DirectOpenAIProviderClient,
    )
    from backend.app.rag.embeddings import OpenAIEmbeddingModel

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError('provider or network access is forbidden')

    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.setattr(_DirectOpenAIProviderClient, 'send', forbidden)
    monkeypatch.setattr(OpenAIEmbeddingModel, 'embed_many', forbidden)

    engine, target, runtime, admin = _service(tmp_path)
    admin.initialize(_review_bytes(target, 'provider-safety-init'))
    with admin.connection_factory() as connection:
        context = runtime.review_context(connection, 'answer_generation')
    admin.mark_rebind_required(
        _review_bytes(
            target,
            'provider-safety-mark-rebind-required',
            context=_context(context),
        )
    )
    with engine.connect() as connection:
        assert connection.scalar(
            select(func.count()).select_from(AgentRunCostComponent)
        ) == 0
