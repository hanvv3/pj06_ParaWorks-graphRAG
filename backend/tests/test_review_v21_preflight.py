from dataclasses import replace
from decimal import Decimal

import pytest

from backend.app.agent_runtime.review_v2_preflight import (
    PreparedReviewRequestV21,
    V21PreparedReviewConfig,
    build_prepared_review_identity_v21,
)
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21


def _config() -> V21PreparedReviewConfig:
    return V21PreparedReviewConfig(
        configured_auto_review_mode='shadow',
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        validator_reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v1',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
        cost_policy_version='auto-review-cost:v1',
        fingerprint_key_version='test-v1',
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
        enforce_percentage=10,
        authorized_percentage_at_launch=10,
        rollout_authorization_generation=3,
        validation_provider_safety_state_version=7,
        rollout_control_epoch=9,
        extraction_plan_set_hmac='b' * 64,
        extraction_provider_safety_snapshot_set_hmac='c' * 64,
        confirmed_extraction_cost_ceiling_usd=Decimal('0.016716'),
        confirmed_validation_cost_ceiling_usd=Decimal('0.048864'),
        confirmed_total_cost_ceiling_usd=Decimal('0.065580'),
        total_budget_limit_usd=Decimal('0.200000'),
    )


def _prepared(
    config: V21PreparedReviewConfig | None = None,
) -> PreparedReviewRequestV21:
    return build_prepared_review_identity_v21(
        source_refs=(),
        agent_names=('mail_document_agent',),
        evidence_version_hash='d' * 64,
        security_scope_id='scope-1',
        config=config or _config(),
        fingerprint_secret=b'test-secret-at-least-thirty-two-bytes',
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('configured_auto_review_mode', 'enforce'),
        ('policy_version', 'auto-review-policy:v2'),
        ('validator_model', 'gpt-5.6-sol'),
        ('validator_output_contract_version', 'candidate-validation-batch:v2'),
        ('enforce_percentage', 100),
        ('confirmed_total_cost_ceiling_usd', Decimal('0.065581')),
    ),
)
def test_v21_identity_changes_with_mode_policy_model_output_contract_percentage_or_ceiling(
    field, value
):
    baseline = _prepared()
    changed = _prepared(replace(_config(), **{field: value}))
    assert changed.input_hash != baseline.input_hash


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('fingerprint_key_material_verifier', 'e' * 64),
        ('cost_policy_version', 'auto-review-cost:v2'),
        ('max_provider_attempts', 2),
    ),
)
def test_v21_identity_changes_with_key_material_cost_policy_or_attempt_cap(
    field, value
):
    assert (
        _prepared(replace(_config(), **{field: value})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    'field',
    (
        'provider_timeout_seconds',
        'provider_send_start_window_seconds',
        'provider_attempt_lease_seconds',
        'provider_commit_grace_seconds',
    ),
)
def test_v21_identity_changes_with_any_shared_provider_timing_value(field):
    config = _config()
    assert (
        _prepared(replace(config, **{field: getattr(config, field) + 1})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    'field', ('validator_input_usd_per_1m', 'validator_output_usd_per_1m')
)
def test_v21_identity_changes_with_either_validator_price_snapshot(field):
    assert (
        _prepared(replace(_config(), **{field: Decimal('99.000000')})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (('authorized_percentage_at_launch', 100), ('rollout_authorization_generation', 4)),
)
def test_v21_identity_changes_with_authorized_percentage_or_rollout_generation(
    field, value
):
    assert (
        _prepared(replace(_config(), **{field: value})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (('validation_provider_safety_state_version', 8), ('rollout_control_epoch', 10)),
)
def test_v21_identity_changes_with_provider_safety_state_or_rollout_control_epoch(
    field, value
):
    assert (
        _prepared(replace(_config(), **{field: value})).input_hash
        != _prepared().input_hash
    )


@pytest.mark.parametrize('field', ('reply_priming_tokens', 'framing_safety_tokens'))
def test_v21_identity_changes_with_reply_priming_or_framing_safety_constant(field):
    config = _config()
    assert (
        _prepared(replace(config, **{field: getattr(config, field) + 1})).input_hash
        != _prepared().input_hash
    )


def test_v21_thread_replay_requires_every_stored_auto_review_field_to_match():
    prepared = _prepared()
    assert prepared.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    assert prepared.matches_stored_snapshot(prepared.stored_snapshot())
    for field, value in prepared.stored_snapshot().items():
        replacement = 'changed' if isinstance(value, str) else (value + 1)
        assert not prepared.matches_stored_snapshot(
            {**prepared.stored_snapshot(), field: replacement}
        )
