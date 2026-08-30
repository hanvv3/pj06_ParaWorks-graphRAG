from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.models import AutoReviewRolloutState
from backend.app.review.auto_review_rollout import (
    AutoReviewRolloutPolicyService,
    RolloutGateError,
    RolloutSelectionInput,
    resolve_effective_rollout,
    stable_rollout_selection,
)


def _settings(**updates) -> Settings:
    values = {
        '_env_file': None,
        'agent_runtime_fingerprint_secret': (
            'task10-rollout-secret-at-least-32-bytes'
        ),
        'agent_runtime_fingerprint_key_version': 'task10-key-v1',
        'auto_review_mode': 'enforce',
        'auto_review_enforce_percentage': 10,
    }
    values.update(updates)
    return Settings(**values)


def _selection_input(**updates) -> RolloutSelectionInput:
    values = {
        'security_scope_hmac': '1' * 64,
        'workflow_execution_hmac': '2' * 64,
        'candidate_key': '3' * 64,
        'policy_version': 'auto-review-policy:v1',
        'rollout_control_epoch': 4,
        'rollout_authorization_generation': 5,
        'requested_percentage': 10,
        'authorized_percentage': 10,
        'fingerprint_key_version': 'task10-key-v1',
        'fingerprint_key_material_verifier': '4' * 64,
    }
    values.update(updates)
    return RolloutSelectionInput(**values)


def test_fresh_scope_preview_uses_read_only_default_and_writes_no_row(
    db_session: Session,
) -> None:
    service = AutoReviewRolloutPolicyService(db_session, settings=_settings())

    snapshot = service.peek_or_default('scope-a', 'auto-review-policy:v1')

    assert snapshot.control_epoch == 0
    assert snapshot.max_authorized_percentage == 0
    assert snapshot.authorization_generation == 0
    assert snapshot.breaker_open is False
    assert snapshot.state_version == 0
    assert db_session.scalar(select(func.count(AutoReviewRolloutState.id))) == 0


def test_concurrent_first_use_converges_on_one_default_rollout_row(
    db_session: Session,
) -> None:
    service = AutoReviewRolloutPolicyService(db_session, settings=_settings())

    first = service.ensure_row('scope-a', 'auto-review-policy:v1')
    db_session.commit()
    second = service.ensure_row('scope-a', 'auto-review-policy:v1')
    db_session.commit()

    assert first.id == second.id
    assert db_session.scalar(select(func.count(AutoReviewRolloutState.id))) == 1
    assert service.peek_or_default(
        'scope-a', 'auto-review-policy:v1'
    ).max_authorized_percentage == 0


def test_disabled_always_demotes_to_zero_call() -> None:
    snapshot = AutoReviewRolloutPolicyService.default_snapshot(
        'scope-a', 'auto-review-policy:v1'
    )

    decision = resolve_effective_rollout(
        snapshot=replace(
            snapshot,
            shadow_completed_count=500,
            shadow_supported_count=500,
            max_authorized_percentage=10,
            authorization_generation=1,
        ),
        stored_mode='enforce',
        stored_percentage=10,
        stored_authorization_generation=1,
        settings=_settings(
            auto_review_mode='disabled',
            auto_review_enforce_percentage=0,
        ),
    )

    assert decision.effective_mode == 'disabled'
    assert decision.effective_percentage == 0
    assert decision.provider_allowed is False


def test_enforce_without_500_shadow_comparisons_stays_shadow() -> None:
    snapshot = replace(
        AutoReviewRolloutPolicyService.default_snapshot(
            'scope-a', 'auto-review-policy:v1'
        ),
        shadow_completed_count=499,
        shadow_supported_count=499,
        max_authorized_percentage=10,
        authorization_generation=1,
    )

    decision = resolve_effective_rollout(
        snapshot=snapshot,
        stored_mode='enforce',
        stored_percentage=10,
        stored_authorization_generation=1,
        settings=_settings(),
    )

    assert decision.effective_mode == 'shadow'
    assert decision.effective_percentage == 0
    assert decision.provider_allowed is True


def test_shadow_precision_is_supported_completed_over_completed_only() -> None:
    snapshot = replace(
        AutoReviewRolloutPolicyService.default_snapshot(
            'scope-a', 'auto-review-policy:v1'
        ),
        shadow_predicted_count=700,
        shadow_completed_count=500,
        shadow_supported_count=495,
    )

    assert snapshot.shadow_precision == Decimal('0.9900')
    assert snapshot.shadow_gate_passed is True
    assert replace(snapshot, shadow_supported_count=494).shadow_gate_passed is False


def test_shadow_gate_can_only_authorize_ten_percent_canary() -> None:
    snapshot = replace(
        AutoReviewRolloutPolicyService.default_snapshot(
            'scope-a', 'auto-review-policy:v1'
        ),
        shadow_completed_count=500,
        shadow_supported_count=500,
    )

    assert snapshot.require_authorization_transition(10) == 10
    with pytest.raises(RolloutGateError, match='jump'):
        snapshot.require_authorization_transition(100)


def test_canary_cannot_jump_to_hundred_percent_without_audit_gate() -> None:
    snapshot = replace(
        AutoReviewRolloutPolicyService.default_snapshot(
            'scope-a', 'auto-review-policy:v1'
        ),
        shadow_completed_count=500,
        shadow_supported_count=500,
        max_authorized_percentage=10,
        authorization_generation=1,
        enforce_promotion_ordinal=499,
        confirmed_mandatory_audit_count=50,
        post_audit_selected_count=50,
        post_audit_completed_count=50,
    )

    with pytest.raises(RolloutGateError, match='canary'):
        snapshot.require_authorization_transition(100)

    ready = replace(snapshot, enforce_promotion_ordinal=500)
    assert ready.require_authorization_transition(100) == 100


def test_stable_ten_percent_canary_selection_is_replay_safe() -> None:
    settings = _settings()
    value = _selection_input()

    first = stable_rollout_selection(
        value,
        percentage=10,
        domain='auto-review-enforce-selection:v1',
        settings=settings,
    )
    replay = stable_rollout_selection(
        value,
        percentage=10,
        domain='auto-review-enforce-selection:v1',
        settings=settings,
    )
    changed = stable_rollout_selection(
        replace(value, candidate_key='5' * 64),
        percentage=10,
        domain='auto-review-enforce-selection:v1',
        settings=settings,
    )

    assert first == replay
    assert first.bucket == int(first.fingerprint, 16) % 100
    assert first.selected is (first.bucket < 10)
    assert changed.fingerprint != first.fingerprint

