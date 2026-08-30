import base64
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.launch_confirmation import (
    LaunchConfirmationCodec,
    LaunchConfirmationError,
    build_v21_launch_snapshot,
)
from backend.app.agent_runtime.review_v2_drafting import ReviewDraftService
from backend.app.agent_runtime.review_v2_preflight import (
    PreparedReviewRequestV21,
    V21PreparedReviewConfig,
)
from backend.app.core.config import Settings
from backend.app.models import AgentWorkflowThread
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='task11-launch-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='task11-launch-v1',
        auto_review_mode='shadow',
        auto_review_enforce_percentage=0,
        auto_review_launch_confirmation_ttl_seconds=300,
    )


def _prepared() -> PreparedReviewRequestV21:
    route = {
        'agent_name': 'history_agent',
        'provider': 'openai',
        'model': 'gpt-5.4-mini-2026-03-17',
        'reasoning_effort': 'none',
        'route_version': 'auto-review-extraction-route:v1',
        'prompt_version': 'history-extraction:c5-v1',
        'output_contract_version': 'history-candidate:c5-v1',
        'cost_policy_version': 'auto-review-extraction-cost:v1',
        'token_estimator_version': 'openai-o200k-extraction:v1',
        'tokenizer_encoding': 'o200k_base',
        'reply_priming_tokens': 16,
        'framing_safety_tokens': 512,
        'max_input_chars': 24000,
        'max_input_tokens': 10000,
        'max_output_tokens': 2048,
        'max_candidates': 1,
        'max_provider_attempts': 1,
        'input_usd_per_1m': '0.750000',
        'output_usd_per_1m': '4.500000',
        'timing': [60, 5, 120, 30],
    }
    config = V21PreparedReviewConfig(
        configured_auto_review_mode='shadow',
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        validator_reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v1',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
        cost_policy_version='auto-review-cost:v1',
        fingerprint_key_version='task11-launch-v1',
        fingerprint_key_material_verifier='a' * 64,
        token_estimator_version='openai-o200k-chat:v1',
        tokenizer_encoding='o200k_base',
        max_input_tokens_per_batch=6000,
        max_output_tokens_per_batch=3072,
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        max_validation_batches_per_workflow=2,
        max_validation_candidates_per_batch=4,
        max_validation_candidates_per_workflow=5,
        max_provider_attempts=1,
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
        validator_input_usd_per_1m=Decimal('2.000000'),
        validator_output_usd_per_1m=Decimal('12.000000'),
        enforce_percentage=0,
        authorized_percentage_at_launch=0,
        rollout_authorization_generation=0,
        validation_provider_safety_state_version=7,
        rollout_control_epoch=0,
        extraction_plan_set_hmac='b' * 64,
        extraction_provider_safety_snapshot_set_hmac='c' * 64,
        confirmed_extraction_cost_ceiling_usd=Decimal('0.016716'),
        confirmed_validation_cost_ceiling_usd=Decimal('0.048864'),
        confirmed_total_cost_ceiling_usd=Decimal('0.065580'),
        total_budget_limit_usd=Decimal('0.200000'),
        extraction_plan_identities=(route,),
    )
    return PreparedReviewRequestV21(
        source_refs=(),
        agent_names=('history_agent',),
        input_hash='d' * 64,
        evidence_version_hash='e' * 64,
        selection_policy_version='company-memory-review-selection:v1',
        config=config,
    )


def _snapshot(prepared: PreparedReviewRequestV21 | None = None):
    return build_v21_launch_snapshot(
        prepared=prepared or _prepared(),
        security_scope_hmac='1' * 64,
        actor_subject_hmac='2' * 64,
        owner_permission_hmac='3' * 64,
    )


def test_launch_codec_round_trip_uses_exact_compact_key_set_and_no_raw_ids() -> None:
    now = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
    codec = LaunchConfirmationCodec(settings=_settings(), now=lambda: now)

    token = codec.issue(_snapshot())
    verified = codec.verify(token, expected=_snapshot())
    payload = json.loads(
        base64.urlsafe_b64decode(token.split('.')[0] + '==').decode()
    )

    assert verified['g'] == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    assert set(payload) == codec.exact_keys
    assert 'owner-raw' not in token
    assert len(token) <= 2048


def test_expired_changed_cross_actor_or_malformed_token_is_one_bounded_error() -> None:
    now = [datetime(2026, 8, 30, 10, 0, tzinfo=UTC)]
    codec = LaunchConfirmationCodec(settings=_settings(), now=lambda: now[0])
    snapshot = _snapshot()
    token = codec.issue(snapshot)

    changed = dict(snapshot)
    changed['ah'] = '9' * 64
    with pytest.raises(LaunchConfirmationError, match='cost_preview_changed'):
        codec.verify(token, expected=changed)
    now[0] += timedelta(minutes=6)
    with pytest.raises(LaunchConfirmationError, match='cost_preview_changed'):
        codec.verify(token, expected=snapshot)
    with pytest.raises(LaunchConfirmationError, match='cost_preview_changed'):
        codec.verify('not.a.valid.token', expected=snapshot)


def test_unknown_or_missing_payload_key_is_rejected() -> None:
    codec = LaunchConfirmationCodec(settings=_settings())
    snapshot = _snapshot()

    with pytest.raises(LaunchConfirmationError, match='cost_preview_changed'):
        codec.issue({**snapshot, 'unknown': 1})
    missing = dict(snapshot)
    missing.pop('cg')
    with pytest.raises(LaunchConfirmationError, match='cost_preview_changed'):
        codec.issue(missing)


def test_snapshot_binds_every_explicit_cost_cap_timing_and_rollout_identity() -> None:
    snapshot = _snapshot()

    assert snapshot['xi'] == 10000
    assert snapshot['xo'] == 2048
    assert snapshot['xk'] == 1
    assert snapshot['it'] == 6000
    assert snapshot['ot'] == 3072
    assert snapshot['mc'] == 4
    assert snapshot['mb'] == 2
    assert (snapshot['pt'], snapshot['sw'], snapshot['ls'], snapshot['cg']) == (
        60,
        5,
        120,
        30,
    )
    assert (snapshot['ip'], snapshot['op']) == ('2.000000', '12.000000')
    assert (snapshot['xip'], snapshot['xop']) == ('0.750000', '4.500000')
    assert (snapshot['rq'], snapshot['ap'], snapshot['rg']) == (0, 0, 0)


def test_any_prepared_identity_drift_requires_a_fresh_token() -> None:
    codec = LaunchConfirmationCodec(settings=_settings())
    prepared = _prepared()
    token = codec.issue(_snapshot(prepared))
    changed = replace(
        prepared,
        config=replace(prepared.config, validator_output_contract_version='changed'),
    )

    with pytest.raises(LaunchConfirmationError, match='cost_preview_changed'):
        codec.verify(token, expected=_snapshot(changed))


def test_v21_preview_is_zero_call_and_uses_conservative_combined_cost(
    db_session: Session,
) -> None:
    class NeverUsedCatalog:
        def get(self, _name: str):
            raise AssertionError('legacy adapter/provider must not run in V2.1 preview')

    preview = ReviewDraftService(
        session_factory=sessionmaker(
            bind=db_session.get_bind(), expire_on_commit=False
        ),
        catalog=NeverUsedCatalog(),  # type: ignore[arg-type]
        settings=_settings(),
    ).preview_prepared(
        prepared=_prepared(),
        actor_subject_id='owner-raw',
        allowed_permission_levels=('public', 'internal'),
    )

    assert preview.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    assert preview.estimated_cost_usd == pytest.approx(0.016716)
    assert preview.auto_review_estimated_cost_usd == pytest.approx(0.048864)
    assert preview.total_estimated_cost_usd == pytest.approx(0.065580)
    assert preview.launch_confirmation_token
    assert db_session.query(AgentWorkflowThread).count() == 0
