from __future__ import annotations

import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.v1.review import (
    approve_review_item,
    reject_review_item,
    request_more_evidence_for_review_item,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AuditLog,
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    TimelineEvent,
    Todo,
)
from backend.app.review.transitions import ReviewTransitionService
from backend.app.schemas.review import ReviewEvidenceRequest

_ISOLATED_DATABASE_ERROR = 'isolated PostgreSQL test database required'
_NON_TEST_DATABASES = frozenset({
    'postgres',
    'template0',
    'template1',
    'paraworks',
    'paraworks_dev',
    'paraworks_prod',
})
_NON_TEST_USERS = frozenset({'postgres', 'paraworks'})


def _validate_database_identity(
    database_name: str | None,
    user_name: str | None,
) -> None:
    normalized_database = (database_name or '').lower()
    normalized_user = (user_name or '').lower()
    if (
        normalized_database in _NON_TEST_DATABASES
        or not normalized_database.endswith('_test')
        or normalized_user in _NON_TEST_USERS
        or not normalized_user.endswith('_test')
    ):
        raise ValueError(_ISOLATED_DATABASE_ERROR)


def _postgres_test_url() -> str:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip(
            'set PARAWORKS_TEST_POSTGRES_URL to a disposable PostgreSQL '
            'database whose database and user names end in _test'
        )
    try:
        parsed = make_url(database_url)
    except (ArgumentError, ValueError):
        raise ValueError(_ISOLATED_DATABASE_ERROR) from None
    if parsed.get_backend_name() != 'postgresql':
        raise ValueError(_ISOLATED_DATABASE_ERROR)
    _validate_database_identity(parsed.database, parsed.username)
    return parsed.set(drivername='postgresql+psycopg').render_as_string(
        hide_password=False
    )


def _run_migrations(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        get_settings.cache_clear()


@dataclass(frozen=True)
class _TransitionHarness:
    engine: Engine
    session_factory: sessionmaker[Session]
    settings: Settings
    workflow_thread_id: str


@pytest.fixture
def postgres_transition_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_TransitionHarness, None, None]:
    database_url = _postgres_test_url()
    expected = make_url(database_url)
    engine = create_engine(database_url)
    workflow_thread_id = f'postgres-review-transition:{uuid4().hex}'
    try:
        with engine.connect() as connection:
            database_name, user_name = connection.execute(
                text('SELECT current_database(), current_user')
            ).one()
        _validate_database_identity(database_name, user_name)
        if database_name != expected.database or user_name != expected.username:
            raise ValueError(_ISOLATED_DATABASE_ERROR)
        _run_migrations(database_url, monkeypatch)
        factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        harness = _TransitionHarness(
            engine=engine,
            session_factory=factory,
            settings=Settings(
                _env_file=None,
                paraworks_demo_mode=True,
                paraworks_database_url=database_url,
                database_url=database_url,
            ),
            workflow_thread_id=workflow_thread_id,
        )
        try:
            yield harness
        finally:
            _cleanup_exact_rows(harness)
    except Exception:
        engine.dispose()
        raise


def _cleanup_exact_rows(harness: _TransitionHarness) -> None:
    with harness.session_factory() as db:
        review_ids = tuple(
            db.scalars(
                select(ReviewItem.id).where(
                    ReviewItem.workflow_thread_id
                    == harness.workflow_thread_id
                )
            ).all()
        )
        if review_ids:
            for model in (
                DecisionRecord,
                HistoryEvent,
                TimelineEvent,
                Todo,
            ):
                db.execute(
                    delete(model).where(
                        model.source_review_item_id.in_(review_ids)
                    )
                )
            db.execute(
                delete(ReviewItem).where(ReviewItem.id.in_(review_ids))
            )
        db.execute(
            delete(AuditLog).where(
                AuditLog.target_id == harness.workflow_thread_id
            )
        )
        db.commit()
    harness.engine.dispose()


def _valid_payload(item_type: str) -> dict[str, str]:
    return {
        'decision_record': {
            'title': 'Choose durable review checkpoints',
            'decision_summary': 'PostgreSQL preserves review recovery.',
        },
        'history_event': {
            'title': 'Human review boundary added',
            'reason': 'Candidates require evidence and approval.',
        },
        'todo': {
            'title': 'Verify release concurrency',
            'priority': 'high',
            'priority_reason': 'Duplicate effects would corrupt memory.',
        },
    }[item_type]


def _seed_item(
    harness: _TransitionHarness,
    *,
    item_type: str,
) -> int:
    with harness.session_factory() as db:
        item = ReviewItem(
            item_type=item_type,
            payload=_valid_payload(item_type),
            source_links=['https://postgres-race.invalid/evidence'],
            source_snippets=['PostgreSQL race evidence.'],
            confidence_score=0.94,
            permission_level='internal',
            status='pending_review',
            workflow_thread_id=harness.workflow_thread_id,
            candidate_key=f'postgres-transition:{uuid4().hex}',
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item.id


def _promotion_tuple(value: object) -> tuple[object, tuple[int, ...], tuple[int, ...]]:
    if isinstance(value, dict):
        return (
            value['target_type'],
            tuple(value['created_record_ids']),
            tuple(value['created_timeline_event_ids']),
        )
    return (
        value.target_type,
        tuple(value.created_record_ids),
        tuple(value.created_timeline_event_ids),
    )


def _assert_promotion_matches_persisted(
    returned: tuple[object, tuple[int, ...], tuple[int, ...]],
    persisted: tuple[object, tuple[int, ...], tuple[int, ...]],
) -> None:
    assert returned == persisted


@pytest.mark.parametrize(
    ('item_type', 'record_model'),
    [
        ('decision_record', DecisionRecord),
        ('history_event', HistoryEvent),
        ('todo', Todo),
    ],
)
def test_concurrent_single_and_bulk_approval_create_one_target_and_companion(
    postgres_transition_harness: _TransitionHarness,
    item_type: str,
    record_model: type[DecisionRecord] | type[HistoryEvent] | type[Todo],
) -> None:
    harness = postgres_transition_harness
    item_id = _seed_item(harness, item_type=item_type)
    barrier = Barrier(2)

    def approve_single() -> tuple[bool, tuple[object, tuple[int, ...], tuple[int, ...]]]:
        with harness.session_factory() as db:
            barrier.wait(timeout=10)
            response = approve_review_item(
                item_id=item_id,
                db=db,
                user=USERS['admin'],
                settings=harness.settings,
            )
            assert response['promotion'] is not None
            return response['replayed'], _promotion_tuple(response['promotion'])

    def approve_bulk() -> tuple[bool, tuple[object, tuple[int, ...], tuple[int, ...]]]:
        with harness.session_factory() as db:
            barrier.wait(timeout=10)
            batch = ReviewTransitionService().transition_many(
                db=db,
                item_ids=[item_id],
                action='approve',
                actor=USERS['admin'],
            )
            assert batch.failed_items == ()
            assert batch.skipped_items == ()
            assert len(batch.results) == 1
            result = batch.results[0]
            assert result.promotion is not None
            db.commit()
            return result.replayed, _promotion_tuple(result.promotion)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(approve_single),
            executor.submit(approve_bulk),
        )
        outcomes = tuple(future.result(timeout=30) for future in futures)

    assert sorted(replayed for replayed, _ in outcomes) == [False, True]
    assert outcomes[0][1] == outcomes[1][1]
    with harness.session_factory() as db:
        records = tuple(
            db.scalars(
                select(record_model).where(
                    record_model.source_review_item_id == item_id
                )
            ).all()
        )
        timeline_events = tuple(
            db.scalars(
                select(TimelineEvent).where(
                    TimelineEvent.source_review_item_id == item_id
                )
            ).all()
        )
        assert len(records) == 1
        assert len(timeline_events) == 1
        record = records[0]
        timeline_event = timeline_events[0]
        assert record.source_review_item_id == item_id
        assert timeline_event.source_review_item_id == item_id
        persisted = (
            item_type,
            (record.id,),
            (timeline_event.id,),
        )
        for _replayed, returned in outcomes:
            _assert_promotion_matches_persisted(returned, persisted)
        record_count = db.scalar(
            select(func.count()).select_from(record_model).where(
                record_model.source_review_item_id == item_id
            )
        )
        timeline_count = db.scalar(
            select(func.count()).select_from(TimelineEvent).where(
                TimelineEvent.source_review_item_id == item_id
            )
        )
        item = db.get(ReviewItem, item_id)
        assert item is not None
        assert item.status == 'approved'
        db.rollback()
    assert record_count == 1
    assert timeline_count == 1


def test_terminal_review_race_returns_bounded_conflict_without_effect(
    postgres_transition_harness: _TransitionHarness,
) -> None:
    harness = postgres_transition_harness
    item_id = _seed_item(harness, item_type='decision_record')
    barrier = Barrier(2)

    def transition(action: str) -> tuple[str, object]:
        with harness.session_factory() as db:
            barrier.wait(timeout=10)
            try:
                if action == 'reject':
                    response = reject_review_item(
                        item_id=item_id,
                        db=db,
                        user=USERS['admin'],
                        settings=harness.settings,
                    )
                else:
                    response = request_more_evidence_for_review_item(
                        item_id=item_id,
                        db=db,
                        user=USERS['admin'],
                        settings=harness.settings,
                        request=ReviewEvidenceRequest(
                            note='Add one more source.'
                        ),
                    )
                return ('success', response['status'])
            except HTTPException as exc:
                db.rollback()
                return ('conflict', (exc.status_code, exc.detail))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(transition, 'reject'),
            executor.submit(transition, 'needs_more_evidence'),
        )
        outcomes = tuple(future.result(timeout=30) for future in futures)

    assert sorted(kind for kind, _ in outcomes) == ['conflict', 'success']
    assert [value for kind, value in outcomes if kind == 'conflict'] == [
        (409, {'code': 'invalid_state_transition'})
    ]
    with harness.session_factory() as db:
        item = db.get(ReviewItem, item_id)
        assert item is not None
        assert item.status in {'rejected', 'needs_more_evidence'}
        effect_counts = tuple(
            db.scalar(
                select(func.count()).select_from(model).where(
                    model.source_review_item_id == item_id
                )
            )
            or 0
            for model in (DecisionRecord, HistoryEvent, TimelineEvent, Todo)
        )
        audit_count = db.scalar(
            select(func.count()).select_from(AuditLog).where(
                AuditLog.target_id == harness.workflow_thread_id
            )
        )
        db.rollback()
    assert effect_counts == (0, 0, 0, 0)
    assert audit_count == 1


def test_promotion_match_helper_rejects_noncanonical_returned_ids() -> None:
    returned = ('decision_record', (101,), (201,))
    persisted = ('decision_record', (102,), (201,))

    with pytest.raises(AssertionError):
        _assert_promotion_matches_persisted(returned, persisted)
