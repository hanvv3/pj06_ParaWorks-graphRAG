import pytest
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.admin.auto_review_rollout import (
    AutoReviewRolloutAdminService,
    RolloutAdminRefused,
)
from backend.app.core.config import Settings
from backend.app.models import (
    AuditLog,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewRolloutControlEvent,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    ReviewItem,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret=(
            'task10-admin-secret-at-least-32-bytes-long'
        ),
        agent_runtime_fingerprint_key_version='task10-admin-v1',
    )


def _seed(db: Session, settings: Settings) -> AutoReviewRolloutState:
    db.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=(
                settings.agent_runtime_fingerprint_key_version
            ),
            fingerprint_key_material_verifier=(
                fingerprint_key_material_verifier(
                    settings.agent_runtime_fingerprint_secret
                )
            ),
            generation=1,
            ready=True,
        )
    )
    rollout = AutoReviewRolloutState(
        security_scope_id='scope-admin',
        policy_version='auto-review-policy:v1',
        shadow_completed_count=500,
        shadow_supported_count=500,
    )
    db.add(rollout)
    db.flush()
    return rollout


def test_authorize_ten_appends_one_control_event_and_backpointer(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _seed(db_session, settings)
    service = AutoReviewRolloutAdminService(db_session, settings=settings)

    result = service.authorize_percentage(
        scope='scope-admin',
        policy='auto-review-policy:v1',
        percentage=10,
        expected_state_version=0,
        reason='shadow 검증 통과',
        gate_ref='golden:task10:v1',
    )

    assert result.max_authorized_percentage == 10
    assert result.authorization_generation == 1
    assert result.control_epoch == 1
    assert result.state_version == 1
    events = db_session.query(AutoReviewRolloutControlEvent).all()
    assert len(events) == 1
    assert rollout.last_event_id == events[0].id
    assert rollout.last_event_sequence == 1
    audit = db_session.query(AuditLog).one()
    assert audit.actor_id == 'system:local-auto-review-rollout-admin'
    assert audit.metadata_['reason'] == 'shadow 검증 통과'


def test_direct_zero_to_hundred_is_refused_without_event(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed(db_session, settings)
    service = AutoReviewRolloutAdminService(db_session, settings=settings)

    with pytest.raises(RolloutAdminRefused, match='jump'):
        service.authorize_percentage(
            scope='scope-admin',
            policy='auto-review-policy:v1',
            percentage=100,
            expected_state_version=0,
            reason='잘못된 단계 상승',
            gate_ref='golden:task10:v1',
        )

    assert db_session.query(AutoReviewRolloutControlEvent).count() == 0


def test_expected_version_cas_refusal_writes_no_event(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed(db_session, settings)
    service = AutoReviewRolloutAdminService(db_session, settings=settings)

    with pytest.raises(RolloutAdminRefused, match='version'):
        service.authorize_percentage(
            scope='scope-admin',
            policy='auto-review-policy:v1',
            percentage=10,
            expected_state_version=1,
            reason='stale operator',
            gate_ref='golden:task10:v1',
        )

    assert db_session.query(AutoReviewRolloutControlEvent).count() == 0


def test_rollout_admin_never_accepts_subject_override() -> None:
    parameters = AutoReviewRolloutAdminService.authorize_percentage.__annotations__
    assert 'actor' not in parameters
    assert 'subject_id' not in parameters


def test_close_breaker_keeps_zero_latch_and_invalidates_generation(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _seed(db_session, settings)
    rollout.breaker_open = True
    rollout.breaker_reason_code = 'incorrect'
    rollout.max_authorized_percentage = 0
    rollout.authorization_generation = 4
    db_session.flush()

    result = AutoReviewRolloutAdminService(
        db_session, settings=settings
    ).close_rollout_breaker(
        scope='scope-admin',
        policy='auto-review-policy:v1',
        expected_state_version=0,
        reason='모든 영향 항목 재검토 완료',
        gate_ref='regression:task10:close-v1',
    )

    assert result.breaker_open is False
    assert result.max_authorized_percentage == 0
    assert result.authorization_generation == 5
    event = db_session.query(AutoReviewRolloutControlEvent).one()
    assert event.event_kind == 'breaker_closed'


def test_close_breaker_refuses_unresolved_remediation_and_correction(
    db_session: Session,
) -> None:
    settings = _settings()
    rollout = _seed(db_session, settings)
    rollout.breaker_open = True
    rollout.corrected_critical_count = 1

    with pytest.raises(RolloutAdminRefused, match='correction'):
        AutoReviewRolloutAdminService(
            db_session, settings=settings
        ).close_rollout_breaker(
            scope='scope-admin',
            policy='auto-review-policy:v1',
            expected_state_version=0,
            reason='잘못된 close 시도',
            gate_ref='regression:task10:blocked',
        )

    rollout.corrected_critical_count = 0
    item = ReviewItem(
        id=991,
        item_type='timeline_event',
        payload={'title': 'quarantined'},
        source_links=[],
        source_snippets=[],
        confidence_score=0.99,
        permission_level='internal',
        status='approved',
    )
    decision = AutoReviewPromotionDecision(
        review_item_id=991,
        security_scope_id='scope-admin',
        policy_version='auto-review-policy:v1',
        rollout_authorization_generation=4,
        promotion_ordinal=1,
        requested_percentage=10,
        stored_percentage=10,
        authorized_percentage=10,
        enforce_selection_fingerprint='1' * 64,
        audit_selection_fingerprint='2' * 64,
        selection_result='sample_10',
        fingerprint_key_version='task10-admin-v1',
        fingerprint_key_material_verifier='3' * 64,
    )
    db_session.add_all([item, decision])
    db_session.flush()
    db_session.add(
        AutoReviewPostAudit(
            review_item_id=991,
            promotion_decision_id=decision.id,
            sample_cohort='sample_10',
            status='pending',
            outcome=None,
        )
    )
    db_session.flush()

    with pytest.raises(RolloutAdminRefused, match='remediation'):
        AutoReviewRolloutAdminService(
            db_session, settings=settings
        ).close_rollout_breaker(
            scope='scope-admin',
            policy='auto-review-policy:v1',
            expected_state_version=0,
            reason='미완료 감사 존재',
            gate_ref='regression:task10:blocked',
        )


def test_initial_provider_authorization_is_purpose_specific_and_append_only(
    db_session: Session,
) -> None:
    settings = _settings()
    _seed(db_session, settings)
    service = AutoReviewRolloutAdminService(db_session, settings=settings)

    state = service.authorize_initial_provider_policy(
        purpose='validation',
        provider='openai',
        model='gpt-5.6-terra',
        reasoning_effort='medium',
        expected_absent=True,
        new_cost_policy_version='auto-review-cost:v1',
        reason='고정 가격 및 golden gate 검토 완료',
        gate_ref='golden:terra:v1',
    )

    assert state.purpose == 'validation'
    assert state.breaker_open is False
    event = db_session.query(AutoReviewProviderSafetyEvent).one()
    assert event.event_kind == 'initial_authorized'
    assert event.purpose == 'validation'
    assert state.last_event_id == event.id

    with pytest.raises(RolloutAdminRefused, match='registry'):
        service.authorize_initial_provider_policy(
            purpose='extraction',
            provider='openai',
            model='gpt-5.6-terra',
            reasoning_effort='medium',
            expected_absent=True,
            new_cost_policy_version='auto-review-cost:v1',
            reason='잘못된 목적 재사용',
            gate_ref='golden:wrong:v1',
        )

    assert db_session.query(AutoReviewProviderSafetyState).count() == 1
