from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import USERS, DemoUser
from backend.app.models import (
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    TimelineEvent,
    Todo,
)
from backend.app.review.transitions import (
    InvalidReviewTransition,
    ReviewTransitionService,
)


def _service() -> ReviewTransitionService:
    return ReviewTransitionService()


def _seed_item(
    db: Session,
    *,
    item_type: str = 'decision_record',
    status: str = 'pending_review',
    permission_level: str = 'internal',
    payload: dict | None = None,
    source_links: list[str] | None = None,
    source_snippets: list[str] | None = None,
    workflow_thread_id: str | None = None,
) -> ReviewItem:
    item = ReviewItem(
        item_type=item_type,
        payload=payload or _valid_payload(item_type),
        source_links=source_links if source_links is not None else ['https://mail.mock/thread/1'],
        source_snippets=source_snippets if source_snippets is not None else ['The source directly supports the claim.'],
        confidence_score=0.91,
        permission_level=permission_level,
        status=status,
        workflow_thread_id=workflow_thread_id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _valid_payload(item_type: str) -> dict:
    payloads = {
        'decision_record': {
            'title': 'Use PostgreSQL as the source of record',
            'decision_summary': 'Approved company memory remains in PostgreSQL.',
        },
        'history_event': {
            'title': 'Review boundary introduced',
            'reason': 'AI candidates require human review before promotion.',
        },
        'timeline_event': {
            'title': 'Review queue launched',
            'result_summary': 'The team began reviewing evidence-backed candidates.',
        },
        'todo': {
            'title': 'Confirm the rollout owner',
            'priority': 'high',
            'priority_reason': 'The reviewed plan requires an accountable owner.',
        },
        'project_assignment': {
            'title': 'Assign source to Project Alpha',
            'project_key': 'project-alpha',
            'project_name': 'Project Alpha',
            'source_id': 'gmail:message-1',
            'evidence_reason': 'The source explicitly names Project Alpha.',
        },
    }
    return payloads[item_type]


def test_pending_approve_promotes_with_source_review_item_provenance(db_session: Session) -> None:
    item = _seed_item(db_session)

    result = _service().transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    decision = db_session.scalars(select(DecisionRecord)).one()
    timeline = db_session.scalars(select(TimelineEvent)).one()
    assert result.status == 'approved'
    assert result.replayed is False
    assert result.promotion.created_record_ids == (decision.id,)
    assert result.promotion.created_timeline_event_ids == (timeline.id,)
    assert decision.source_review_item_id == item.id
    assert timeline.source_review_item_id == item.id


def test_approve_replay_returns_existing_canonical_effect_ids(db_session: Session) -> None:
    item = _seed_item(db_session)
    service = _service()
    first = service.transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    replay = service.transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    assert replay.replayed is True
    assert replay.promotion == first.promotion
    assert len(db_session.scalars(select(DecisionRecord)).all()) == 1
    assert len(db_session.scalars(select(TimelineEvent)).all()) == 1


@pytest.mark.parametrize('terminal_status', ['rejected', 'needs_more_evidence'])
def test_terminal_review_transition_is_rejected(db_session: Session, terminal_status: str) -> None:
    item = _seed_item(db_session, status=terminal_status)

    with pytest.raises(InvalidReviewTransition) as exc_info:
        _service().transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    assert exc_info.value.code == 'invalid_state_transition'
    assert db_session.get(ReviewItem, item.id).status == terminal_status


def test_needs_more_evidence_is_terminal_and_never_promotes(db_session: Session) -> None:
    item = _seed_item(db_session)
    service = _service()

    result = service.transition(
        db=db_session,
        item_id=item.id,
        action='needs_more_evidence',
        actor=USERS['admin'],
        note='Please add the owner statement.',
    )

    db_session.refresh(item)
    assert result.status == 'needs_more_evidence'
    assert result.promotion is None
    assert item.reviewer_id == USERS['admin'].id
    assert item.reviewed_at is not None
    assert item.payload['needs_more_evidence']['note'] == 'Please add the owner statement.'
    assert db_session.scalars(select(DecisionRecord)).all() == []
    with pytest.raises(InvalidReviewTransition):
        service.transition(db=db_session, item_id=item.id, action='reject', actor=USERS['admin'])


def test_reject_records_reviewer_and_reviewed_at(db_session: Session) -> None:
    item = _seed_item(db_session)

    result = _service().transition(db=db_session, item_id=item.id, action='reject', actor=USERS['viewer'])

    db_session.refresh(item)
    assert result.status == 'rejected'
    assert item.reviewer_id == USERS['viewer'].id
    assert item.reviewed_at is not None


def test_bulk_transition_processes_ids_in_ascending_order(db_session: Session) -> None:
    items = [_seed_item(db_session, item_type='timeline_event') for _ in range(3)]

    result = _service().transition_many(
        db=db_session,
        item_ids=[items[2].id, items[0].id, items[1].id],
        action='reject',
        actor=USERS['viewer'],
    )

    assert [row.item_id for row in result.results] == sorted(item.id for item in items)
    assert result.failed_items == ()
    assert result.skipped_items == ()


def test_bulk_and_single_approve_create_effects_exactly_once(db_session: Session) -> None:
    item = _seed_item(db_session, item_type='todo')
    service = _service()
    first = service.transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    batch = service.transition_many(
        db=db_session,
        item_ids=[item.id],
        action='approve',
        actor=USERS['admin'],
    )

    assert batch.results[0].replayed is True
    assert batch.results[0].promotion == first.promotion
    assert len(db_session.scalars(select(Todo)).all()) == 1
    assert len(db_session.scalars(select(TimelineEvent)).all()) == 1


@pytest.mark.parametrize(
    ('item_type', 'record_model'),
    [
        ('decision_record', DecisionRecord),
        ('history_event', HistoryEvent),
        ('todo', Todo),
    ],
)
def test_companion_timeline_provenance_is_exactly_once(
    db_session: Session,
    item_type: str,
    record_model: type[DecisionRecord] | type[HistoryEvent] | type[Todo],
) -> None:
    item = _seed_item(db_session, item_type=item_type)
    service = _service()

    service.transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])
    replay = service.transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    record = db_session.scalars(select(record_model)).one()
    timeline = db_session.scalars(select(TimelineEvent)).one()
    assert record.source_review_item_id == item.id
    assert timeline.source_review_item_id == item.id
    assert replay.promotion.created_record_ids == (record.id,)
    assert replay.promotion.created_timeline_event_ids == (timeline.id,)


def test_transition_rechecks_exact_actor_permission_levels(db_session: Session) -> None:
    item = _seed_item(db_session, permission_level='restricted')
    restricted_blind_admin = DemoUser(
        id='limited-admin',
        email='limited-admin@example.com',
        role='admin',
        permission_levels={'public', 'internal'},
        name='Limited Admin',
        title='Administrator',
        department='Platform',
    )

    with pytest.raises(HTTPException) as exc_info:
        _service().transition(
            db=db_session,
            item_id=item.id,
            action='approve',
            actor=restricted_blind_admin,
        )

    assert exc_info.value.status_code == 403
    assert db_session.get(ReviewItem, item.id).status == 'pending_review'
    assert db_session.scalars(select(DecisionRecord)).all() == []


@pytest.mark.parametrize(
    ('actor', 'permission_level', 'allowed'),
    [
        (USERS['hanvv-employee'], 'internal', False),
        (USERS['viewer'], 'internal', True),
        (USERS['admin'], 'restricted', True),
    ],
)
def test_transition_preserves_employee_reviewer_admin_rbac(
    db_session: Session,
    actor: DemoUser,
    permission_level: str,
    allowed: bool,
) -> None:
    item = _seed_item(db_session, permission_level=permission_level)

    if allowed:
        result = _service().transition(db=db_session, item_id=item.id, action='reject', actor=actor)
        assert result.status == 'rejected'
    else:
        with pytest.raises(HTTPException) as exc_info:
            _service().transition(db=db_session, item_id=item.id, action='reject', actor=actor)
        assert exc_info.value.status_code == 403
        assert db_session.get(ReviewItem, item.id).status == 'pending_review'


@pytest.mark.parametrize(
    ('source_links', 'source_snippets'),
    [([], ['Supported snippet']), (['https://mail.mock/thread/1'], ['   '])],
)
def test_transition_rechecks_evidence_before_approval(
    db_session: Session,
    source_links: list[str],
    source_snippets: list[str],
) -> None:
    item = _seed_item(db_session, source_links=source_links, source_snippets=source_snippets)

    with pytest.raises(ValueError, match='Review item requires source evidence'):
        _service().transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    assert db_session.get(ReviewItem, item.id).status == 'pending_review'


def test_project_assignment_approval_has_no_knowledge_effect(db_session: Session) -> None:
    item = _seed_item(db_session, item_type='project_assignment')

    result = _service().transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    assert result.status == 'approved'
    assert result.promotion.target_type is None
    assert result.promotion.created_record_ids == ()
    assert result.promotion.created_timeline_event_ids == ()
    for model in (DecisionRecord, HistoryEvent, TimelineEvent, Todo):
        assert db_session.scalars(select(model)).all() == []


def test_legacy_approved_item_without_provenance_replays_without_new_effect(db_session: Session) -> None:
    item = _seed_item(db_session, status='approved')

    result = _service().transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    assert result.replayed is True
    assert result.promotion is None
    assert db_session.scalars(select(DecisionRecord)).all() == []
    assert db_session.scalars(select(TimelineEvent)).all() == []


def test_transition_flushes_without_committing(db_session: Session) -> None:
    item = _seed_item(db_session, item_type='timeline_event')

    _service().transition(db=db_session, item_id=item.id, action='approve', actor=USERS['admin'])

    assert db_session.in_transaction() is True


def test_transition_many_reports_typed_terminal_failures(db_session: Session) -> None:
    item = _seed_item(db_session, status='rejected')

    result = _service().transition_many(
        db=db_session,
        item_ids=[item.id],
        action='approve',
        actor=USERS['admin'],
    )

    assert result.results == ()
    assert result.failed_items == (
        {
            'id': item.id,
            'code': 'invalid_state_transition',
            'detail': 'Cannot approve review item from rejected',
        },
    )
    assert result.skipped_items == ()
