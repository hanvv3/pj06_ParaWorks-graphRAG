from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.app.core.demo_auth import USERS, DemoUser
from backend.app.models import AutoReviewPostAudit, ReviewItem
from backend.tests.test_auto_review_source_reconciliation import (
    _seed_explicit_history,
)


def test_review_evidence_visibility_returns_404_to_unauthorized_actor_and_never_leaks_source_fields(
    db_session: Session,
) -> None:
    from backend.app.review.evidence_visibility import (
        ReviewEvidenceNotFound,
        ReviewEvidenceVisibilityService,
    )

    _, item, source, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    source.permission_level = 'restricted'
    db_session.commit()
    public_reviewer = DemoUser(
        id='public-reviewer',
        email='public@example.com',
        role='reviewer',
        permission_levels={'public'},
        name='Public Reviewer',
        title='Reviewer',
        department='Review',
    )

    try:
        ReviewEvidenceVisibilityService(db_session).project(item.id, public_reviewer)
    except ReviewEvidenceNotFound as exc:
        assert str(exc) == 'Review item not found'
        assert not hasattr(exc, 'source_id')
        assert not hasattr(exc, 'source_url')
    else:
        raise AssertionError('unauthorized current evidence must be concealed')


def test_missing_source_keeps_bounded_review_action_for_authorized_scope_but_conceals_evidence(
    db_session: Session,
) -> None:
    from backend.app.review.evidence_visibility import (
        ReviewEvidenceVisibilityService,
    )

    _, item, source, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    db_session.delete(source)
    db_session.commit()

    projected = ReviewEvidenceVisibilityService(db_session).project(
        item.id, USERS['admin']
    )

    assert projected.visible is True
    assert projected.evidence_available is False
    assert projected.source_ids == ()
    assert projected.source_links == ()
    assert projected.source_snippets == ()
    assert projected.evidence_status == 'evidence_unavailable'


def test_evidence_shaped_legacy_item_without_current_source_is_concealed(
    db_session: Session,
) -> None:
    from backend.app.review.evidence_visibility import (
        ReviewEvidenceVisibilityService,
    )

    item = ReviewItem(
        item_type='history_event',
        payload={'source_ids': ['gmail:deleted-legacy']},
        source_links=['https://gmail.invalid/deleted-legacy'],
        source_snippets=['historical confidential bytes'],
        confidence_score=0.8,
        permission_level='internal',
        status='pending_review',
    )
    db_session.add(item)
    db_session.commit()

    projected = ReviewEvidenceVisibilityService(db_session).project(
        item.id, USERS['viewer']
    )

    assert projected.evidence_status == 'evidence_unavailable'
    assert projected.evidence_available is False
    assert projected.source_ids == ()
    assert projected.source_links == ()
    assert projected.source_snippets == ()


def test_critical_item_is_hidden_from_trusted_serving_but_visible_actionable_in_review_to_authorized_reviewer(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )
    from backend.app.review.evidence_visibility import (
        ReviewEvidenceVisibilityService,
    )

    history, item, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    db_session.add(
        AutoReviewPostAudit(
            review_item_id=item.id,
            promotion_decision_id=1,
            sample_cohort='manual',
            status='remediation_required',
            outcome='permission_violation',
            system_resolution_code='revoke_pending',
            remediation_code='exact_revoke_required',
            auditor_subject_hmac='a' * 64,
            auditor_fingerprint_key_version='v1',
            auditor_fingerprint_key_material_verifier='b' * 64,
            audit_reason='Permission mismatch',
            audited_at=datetime.now(UTC),
        )
    )
    db_session.commit()

    trusted = TrustedServingEligibilityService(db_session).for_knowledge(
        'history_event', history.id
    )
    review = ReviewEvidenceVisibilityService(db_session).project(
        item.id, USERS['admin']
    )

    assert trusted.eligible is False
    assert review.visible is True
    assert review.action_required is True
    assert review.source_links == ('https://gmail.mock/evidence',)
