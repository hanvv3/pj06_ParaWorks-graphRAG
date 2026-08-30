from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRevocationAssessment,
    AutoReviewRolloutState,
    ReviewItem,
)
from backend.app.review.actors import human_review_actor
from backend.app.review.auto_review_quality_revoke import (
    AutoReviewQualityRevokeService,
    QualityRevokeRefused,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='task10-quality-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='task10-quality-v1',
    )


def _seed(
    db: Session,
    *,
    item_id: int,
    audit_status: str | None = None,
    audit_outcome: str | None = None,
) -> tuple[ReviewItem, AutoReviewRolloutState, AutoReviewPostAudit | None]:
    item = ReviewItem(
        id=item_id,
        item_type='timeline_event',
        payload={'title': f'approved {item_id}'},
        source_links=['https://example.test/source'],
        source_snippets=['evidence'],
        confidence_score=0.99,
        permission_level='internal',
        status='approved',
        resolution_source='auto_policy',
        resolution_policy_version='auto-review-policy:v1',
    )
    rollout = AutoReviewRolloutState(
        security_scope_id='scope-quality',
        policy_version='auto-review-policy:v1',
        state_version=1,
        control_epoch=1,
        max_authorized_percentage=10,
        authorization_generation=1,
        confirmed_mandatory_audit_count=50,
        post_audit_selected_count=50,
        post_audit_completed_count=50,
    )
    decision = AutoReviewPromotionDecision(
        review_item_id=item_id,
        security_scope_id='scope-quality',
        policy_version='auto-review-policy:v1',
        rollout_authorization_generation=1,
        promotion_ordinal=item_id,
        requested_percentage=10,
        stored_percentage=10,
        authorized_percentage=10,
        enforce_selection_fingerprint='1' * 64,
        audit_selection_fingerprint='2' * 64,
        selection_result='not_selected',
        fingerprint_key_version='task10-quality-v1',
        fingerprint_key_material_verifier='3' * 64,
    )
    db.add_all([item, rollout, decision])
    db.flush()
    audit = None
    if audit_status is not None:
        audit = AutoReviewPostAudit(
            review_item_id=item_id,
            promotion_decision_id=decision.id,
            sample_cohort='manual' if audit_status == 'completed' else 'sample_10',
            status=audit_status,
            outcome=audit_outcome,
            system_resolution_code=None,
            remediation_code=None,
            auditor_subject_hmac='a' * 64 if audit_outcome else None,
            auditor_fingerprint_key_version='task10-quality-v1' if audit_outcome else None,
            auditor_fingerprint_key_material_verifier='3' * 64 if audit_outcome else None,
            audit_reason='previous human confirmation' if audit_outcome else None,
            audited_at=datetime.now(UTC) if audit_outcome else None,
        )
        db.add(audit)
        db.flush()
    db.commit()
    return item, rollout, audit


def _service(
    db: Session,
    callback: Callable[..., object],
) -> AutoReviewQualityRevokeService:
    return AutoReviewQualityRevokeService(
        db,
        settings=_settings(),
        revoke_callback=callback,
    )


def test_quality_revoke_without_audit_creates_manual_critical_before_revoke(
    db_session: Session,
) -> None:
    item, rollout, _ = _seed(db_session, item_id=101)
    observed: list[tuple[bool, str, str]] = []

    def revoke(*, context, **_kwargs):
        db_session.expire_all()
        audit = db_session.query(AutoReviewPostAudit).filter_by(review_item_id=item.id).one()
        observed.append((rollout.breaker_open, audit.status, audit.system_resolution_code))
        return None

    result = _service(db_session, revoke).revoke_quality(
        review_item_id=item.id,
        actor=human_review_actor(USERS['admin']),
        reason_code='incorrect_content',
        reason='실제 근거와 다른 내용',
    )

    audit = db_session.query(AutoReviewPostAudit).filter_by(review_item_id=item.id).one()
    assert observed == [(True, 'remediation_required', 'revoke_pending')]
    assert result.replayed is False
    assert audit.sample_cohort == 'manual'
    assert audit.status == 'completed'
    assert audit.outcome == 'incorrect'


def test_quality_revoke_with_pending_audit_finalizes_critical_before_revoke(
    db_session: Session,
) -> None:
    item, _, audit = _seed(db_session, item_id=102, audit_status='pending')
    assert audit is not None
    observed: list[str] = []

    def revoke(*, context, **_kwargs):
        db_session.expire_all()
        observed.append(db_session.get(AutoReviewPostAudit, audit.id).status)
        return None

    _service(db_session, revoke).revoke_quality(
        review_item_id=item.id,
        actor=human_review_actor(USERS['admin']),
        reason_code='permission_violation',
        reason='권한 범위를 벗어난 승격',
    )

    assert observed == ['remediation_required']
    assert db_session.get(AutoReviewPostAudit, audit.id).outcome == 'permission_violation'


def test_confirmed_audit_gets_immutable_correction_and_replay_counts_once(
    db_session: Session,
) -> None:
    item, rollout, audit = _seed(
        db_session,
        item_id=103,
        audit_status='completed',
        audit_outcome='confirmed',
    )
    assert audit is not None
    calls: list[int] = []

    def revoke(*, context, **_kwargs):
        db_session.expire_all()
        correction = db_session.query(AutoReviewAuditCorrection).one()
        assert correction.status == 'remediation_required'
        calls.append(context.review_item_id)
        return None

    service = _service(db_session, revoke)
    kwargs = {
        'review_item_id': item.id,
        'actor': human_review_actor(USERS['admin']),
        'reason_code': 'wrong_source_version',
        'reason': '승인 후 원문 버전이 달라짐',
    }
    first = service.revoke_quality(**kwargs)
    second = service.revoke_quality(**kwargs)

    db_session.expire_all()
    assert first.replayed is False
    assert second.replayed is True
    assert audit.outcome == 'confirmed'
    assert db_session.query(AutoReviewAuditCorrection).count() == 1
    assert db_session.query(AutoReviewRevocationAssessment).count() == 1
    assert db_session.get(AutoReviewRolloutState, rollout.id).corrected_critical_count == 1
    assert calls == [item.id]


def test_revoke_failure_keeps_breaker_and_recovery_marker(
    db_session: Session,
) -> None:
    item, rollout, _ = _seed(db_session, item_id=104)

    def fail(**_kwargs):
        raise RuntimeError('bounded downstream failure')

    result = _service(db_session, fail).revoke_quality(
        review_item_id=item.id,
        actor=human_review_actor(USERS['admin']),
        reason_code='policy_violation',
        reason='정책 기준 위반',
    )

    db_session.expire_all()
    audit = db_session.query(AutoReviewPostAudit).filter_by(review_item_id=item.id).one()
    assert result.remediation_required is True
    assert audit.status == 'remediation_required'
    assert audit.system_resolution_code == 'revoke_failed'
    assert db_session.get(AutoReviewRolloutState, rollout.id).breaker_open is True


def test_restart_recovery_retries_pending_revoke_without_new_assessment(
    db_session: Session,
) -> None:
    item, _, _ = _seed(db_session, item_id=106)
    calls = 0

    def flaky(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('first attempt crashes')
        return None

    service = _service(db_session, flaky)
    service.revoke_quality(
        review_item_id=item.id,
        actor=human_review_actor(USERS['admin']),
        reason_code='incorrect_content',
        reason='재시작 복구 대상',
    )
    recovered = service.recover_pending_remediation(limit=100)

    audit = db_session.query(AutoReviewPostAudit).filter_by(review_item_id=item.id).one()
    assert recovered == 1
    assert calls == 2
    assert audit.status == 'completed'
    assert db_session.query(AutoReviewRevocationAssessment).count() == 1


def test_business_withdrawal_never_enters_quality_metrics(
    db_session: Session,
) -> None:
    item, rollout, _ = _seed(db_session, item_id=105)

    with pytest.raises(QualityRevokeRefused, match='business_withdrawal'):
        _service(db_session, lambda **_kwargs: None).revoke_quality(
            review_item_id=item.id,
            actor=human_review_actor(USERS['admin']),
            reason_code='business_withdrawal',  # type: ignore[arg-type]
            reason='사업상 철회',
        )

    assert db_session.query(AutoReviewRevocationAssessment).count() == 0
    assert rollout.corrected_critical_count == 0
