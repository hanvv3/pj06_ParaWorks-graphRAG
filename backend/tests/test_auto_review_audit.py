
import pytest
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AutoReviewPostAudit,
    AutoReviewRolloutControlEvent,
    AutoReviewRolloutState,
    ReviewItem,
)
from backend.app.review.actors import human_review_actor
from backend.app.review.auto_review_audit import (
    AutoReviewPostAuditTransitionService,
    AutoReviewPromotionReservationService,
    PromotionReservationInput,
)
from backend.app.review.auto_review_rollout import (
    RolloutGateError,
    RolloutSelectionInput,
    stable_rollout_selection,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret=(
            'task10-audit-secret-at-least-32-bytes-long'
        ),
        agent_runtime_fingerprint_key_version='task10-key-v1',
        auto_review_mode='enforce',
        auto_review_enforce_percentage=100,
    )


def _item(db: Session, item_id: int) -> ReviewItem:
    item = ReviewItem(
        id=item_id,
        item_type='timeline_event',
        payload={'title': f'후보 {item_id}'},
        source_links=[],
        source_snippets=[],
        confidence_score=0.99,
        permission_level='internal',
        status='pending_review',
    )
    db.add(item)
    db.flush()
    return item


def _rollout(db: Session, **updates) -> AutoReviewRolloutState:
    values = {
        'security_scope_id': 'scope-task10',
        'policy_version': 'auto-review-policy:v1',
        'state_version': 1,
        'control_epoch': 1,
        'shadow_completed_count': 500,
        'shadow_supported_count': 500,
        'max_authorized_percentage': 10,
        'authorization_generation': 1,
    }
    values.update(updates)
    row = AutoReviewRolloutState(**values)
    db.add(row)
    db.flush()
    return row


def _selection(candidate_key: str, *, percentage: int = 10):
    settings = _settings()
    return RolloutSelectionInput(
        security_scope_hmac='1' * 64,
        workflow_execution_hmac='2' * 64,
        candidate_key=candidate_key,
        policy_version='auto-review-policy:v1',
        rollout_control_epoch=1,
        rollout_authorization_generation=1,
        requested_percentage=percentage,  # type: ignore[arg-type]
        authorized_percentage=percentage,  # type: ignore[arg-type]
        fingerprint_key_version='task10-key-v1',
        fingerprint_key_material_verifier=fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ),
    )


def _selected_candidate(settings: Settings, *, percentage: int) -> str:
    for value in range(1000):
        candidate_key = f'{value:064x}'
        selected = stable_rollout_selection(
            _selection(candidate_key, percentage=percentage),
            percentage=percentage,  # type: ignore[arg-type]
            domain='auto-review-enforce-selection:v1',
            settings=settings,
        )
        if selected.selected:
            return candidate_key
    raise AssertionError('fixed selection vector was unavailable')


def _request(
    *,
    review_item_id: int,
    candidate_key: str,
    percentage: int = 10,
) -> PromotionReservationInput:
    return PromotionReservationInput(
        review_item_id=review_item_id,
        security_scope_id='scope-task10',
        policy_version='auto-review-policy:v1',
        selection=_selection(candidate_key, percentage=percentage),
        requested_percentage=percentage,  # type: ignore[arg-type]
        stored_percentage=percentage,  # type: ignore[arg-type]
        authorized_percentage=percentage,  # type: ignore[arg-type]
    )


def test_first_fifty_mandatory_slots_all_require_audit(db_session: Session) -> None:
    settings = _settings()
    rollout = _rollout(db_session)
    _item(db_session, 1)
    service = AutoReviewPromotionReservationService(
        db_session, settings=settings
    )
    candidate_key = _selected_candidate(settings, percentage=10)

    result = service.reserve_auto_promotion(
        _request(review_item_id=1, candidate_key=candidate_key)
    )

    assert result.decision.promotion_ordinal == 1
    assert result.decision.selection_result == 'first_50'
    assert result.audit is not None
    assert result.audit.sample_cohort == 'first_50'
    assert result.audit.status == 'pending'
    assert rollout.pending_mandatory_audit_count == 1
    assert rollout.post_audit_selected_count == 1


def test_full_mandatory_slots_block_unsampled_promotion_until_resolution(
    db_session: Session,
) -> None:
    settings = _settings()
    _rollout(
        db_session,
        enforce_promotion_ordinal=50,
        pending_mandatory_audit_count=50,
        post_audit_selected_count=50,
    )
    _item(db_session, 2)
    service = AutoReviewPromotionReservationService(
        db_session, settings=settings
    )

    with pytest.raises(RolloutGateError, match='mandatory audit slots'):
        service.reserve_auto_promotion(
            _request(
                review_item_id=2,
                candidate_key=_selected_candidate(settings, percentage=10),
            )
        )


def test_replay_returns_one_decision_and_does_not_increment_counters(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _rollout(db_session)
    _item(db_session, 3)
    service = AutoReviewPromotionReservationService(
        db_session, settings=settings
    )
    request = _request(
        review_item_id=3,
        candidate_key=_selected_candidate(settings, percentage=10),
    )

    first = service.reserve_auto_promotion(request)
    replay = service.reserve_auto_promotion(request)

    assert replay.decision.id == first.decision.id
    assert replay.audit is not None and first.audit is not None
    assert replay.audit.id == first.audit.id
    assert rollout.enforce_promotion_ordinal == 1
    assert rollout.pending_mandatory_audit_count == 1


def test_after_mandatory_gate_canary_uses_stable_ten_percent_audit(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _rollout(
        db_session,
        enforce_promotion_ordinal=50,
        confirmed_mandatory_audit_count=50,
        post_audit_selected_count=50,
        post_audit_completed_count=50,
    )
    _item(db_session, 4)
    service = AutoReviewPromotionReservationService(
        db_session, settings=settings
    )
    candidate_key = _selected_candidate(settings, percentage=10)
    request = _request(review_item_id=4, candidate_key=candidate_key)

    result = service.reserve_auto_promotion(request)
    replay = service.reserve_auto_promotion(request)

    assert result.decision.selection_result in {'sample_10', 'not_selected'}
    assert replay.decision.audit_selection_fingerprint == (
        result.decision.audit_selection_fingerprint
    )
    if result.decision.selection_result == 'sample_10':
        assert isinstance(result.audit, AutoReviewPostAudit)
    else:
        assert result.audit is None
    assert rollout.enforce_promotion_ordinal == 51


def test_existing_ten_percent_workflow_never_uses_sample_two(
    db_session: Session,
) -> None:
    settings = _settings()
    _rollout(
        db_session,
        enforce_promotion_ordinal=500,
        confirmed_mandatory_audit_count=50,
        post_audit_selected_count=50,
        post_audit_completed_count=50,
    )
    _item(db_session, 5)
    service = AutoReviewPromotionReservationService(
        db_session, settings=settings
    )

    result = service.reserve_auto_promotion(
        _request(
            review_item_id=5,
            candidate_key=_selected_candidate(settings, percentage=10),
        )
    )

    assert result.decision.selection_result != 'sample_2'


def test_completed_confirmed_audit_is_immutable_and_counts_once(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _rollout(db_session)
    _item(db_session, 6)
    reservation = AutoReviewPromotionReservationService(
        db_session, settings=settings
    ).reserve_auto_promotion(
        _request(
            review_item_id=6,
            candidate_key=_selected_candidate(settings, percentage=10),
        )
    )
    assert reservation.audit is not None
    actor = human_review_actor(USERS['admin'])
    service = AutoReviewPostAuditTransitionService(
        db_session, settings=settings
    )

    first = service.complete(
        audit_id=reservation.audit.id,
        actor=actor,
        outcome='confirmed',
        reason='  근거와 결과가 일치함  ',
    )
    replay = service.complete(
        audit_id=reservation.audit.id,
        actor=actor,
        outcome='confirmed',
        reason='근거와 결과가 일치함',
    )

    assert first.changed is True
    assert replay.replayed is True
    assert reservation.audit.status == 'completed'
    assert reservation.audit.audit_reason == '근거와 결과가 일치함'
    assert rollout.pending_mandatory_audit_count == 0
    assert rollout.confirmed_mandatory_audit_count == 1
    assert rollout.post_audit_completed_count == 1


def test_critical_audit_opens_breaker_and_quarantine_before_revoke(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _rollout(db_session)
    _item(db_session, 7)
    reservation = AutoReviewPromotionReservationService(
        db_session, settings=settings
    ).reserve_auto_promotion(
        _request(
            review_item_id=7,
            candidate_key=_selected_candidate(settings, percentage=10),
        )
    )
    assert reservation.audit is not None
    actor = human_review_actor(USERS['admin'])

    result = AutoReviewPostAuditTransitionService(
        db_session, settings=settings
    ).complete(
        audit_id=reservation.audit.id,
        actor=actor,
        outcome='incorrect',
        reason='내용이 실제 근거와 다름',
    )

    assert result.changed is True
    assert result.remediation_required is True
    assert reservation.audit.status == 'remediation_required'
    assert reservation.audit.system_resolution_code == 'revoke_pending'
    assert rollout.breaker_open is True
    assert rollout.max_authorized_percentage == 0
    assert rollout.authorization_generation == 2
    assert rollout.pending_mandatory_audit_count == 0
    assert rollout.post_audit_completed_count == 1
    assert rollout.post_audit_critical_count == 1
    events = db_session.query(AutoReviewRolloutControlEvent).all()
    assert len(events) == 1
    assert events[0].event_kind == 'breaker_opened'
