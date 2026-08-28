from __future__ import annotations

import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from time import monotonic, sleep
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from sqlalchemy import create_engine, delete, func, select, text, update
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
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    AuditLog,
    DecisionRecord,
    Document,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TimelineEvent,
    Todo,
)
from backend.app.review.actors import human_review_actor
from backend.app.review.transitions import (
    CanonicalReviewEvidenceStalenessResolver,
    InternalReviewTransitionService,
    ReviewTransitionService,
)
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
    workflow_evidence_ref_id: int
    source_id: int
    document_id: int
    current_document_version_id: int
    next_document_version_id: int


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
        with factory() as db:
            source = Source(
                source_type='drive',
                source_id=f'drive:{workflow_thread_id}',
                source_url='https://postgres-race.invalid/evidence',
                title='PostgreSQL transition evidence',
                permission_level='internal',
                raw_metadata={},
                server_content_signature_schema='server-source-content:v1',
                server_content_signature='a' * 64,
                connector_content_signature='fixture-display-only',
            )
            thread = AgentWorkflowThread(
                thread_id=workflow_thread_id,
                workflow_name='company_memory_review',
                graph_version='company-memory-review-v2.1-auto-review',
                checkpoint_thread_id=f'checkpoint:{workflow_thread_id}',
                checkpoint_store='postgres',
                owner_subject_id=USERS['admin'].id,
                security_scope_id='default',
                input_hash='b' * 64,
                evidence_version_hash='c' * 64,
                status='awaiting_review',
            )
            db.add_all([source, thread])
            db.flush()
            document = Document(
                source_id=source.id,
                title='PostgreSQL transition evidence document',
                current_version='v1',
                current_document_version_id=None,
            )
            db.add(document)
            db.flush()
            current_version = DocumentVersion(
                document_id=document.id,
                version='v1',
                body='Current PostgreSQL transition evidence.',
            )
            next_version = DocumentVersion(
                document_id=document.id,
                version='v2',
                body='Changed PostgreSQL transition evidence.',
            )
            db.add_all([current_version, next_version])
            db.flush()
            document.current_document_version_id = current_version.id
            evidence_ref = AgentWorkflowEvidenceRef(
                workflow_thread_id=workflow_thread_id,
                ordinal=1,
                canonical_source_type='drive',
                canonical_table='sources',
                canonical_row_id=source.id,
                document_version_id=current_version.id,
                external_revision=None,
                content_signature=source.server_content_signature,
                permission_level_snapshot='internal',
                content_fingerprint='d' * 64,
            )
            db.add(evidence_ref)
            db.commit()
            evidence_ref_id = evidence_ref.id
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
            workflow_evidence_ref_id=evidence_ref_id,
            source_id=source.id,
            document_id=document.id,
            current_document_version_id=current_version.id,
            next_document_version_id=next_version.id,
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
                delete(ReviewItemEvidenceRef).where(
                    ReviewItemEvidenceRef.review_item_id.in_(review_ids)
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
        db.execute(
            delete(AgentRun).where(
                AgentRun.workflow_thread_id == harness.workflow_thread_id
            )
        )
        db.execute(
            delete(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == harness.workflow_thread_id
            )
        )
        db.execute(
            update(Document)
            .where(Document.id == harness.document_id)
            .values(current_document_version_id=None)
        )
        db.execute(
            delete(DocumentVersion).where(
                DocumentVersion.document_id == harness.document_id
            )
        )
        db.execute(delete(Document).where(Document.id == harness.document_id))
        db.execute(
            delete(Source).where(
                Source.source_id == f'drive:{harness.workflow_thread_id}'
            )
        )
        db.execute(
            delete(AgentWorkflowThread).where(
                AgentWorkflowThread.thread_id == harness.workflow_thread_id
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
        run_key = uuid4().hex
        run = AgentRun(
            agent_name=f'{item_type}_agent',
            prompt_version=f'{item_type}-extraction:c5-v1',
            status='complete',
            source_window='bounded-postgres-transition-fixture',
            cache_key=f'postgres-transition:{run_key}',
            model_name='gpt-5.4-mini-2026-03-17',
            generation_provider='openai',
            generation_reasoning_effort='none',
            generation_route_version='auto-review-extraction-route:v1',
            generation_output_contract_version=f'{item_type}-candidate:c5-v1',
            permission_level='internal',
            workflow_thread_id=harness.workflow_thread_id,
            effect_key=f'postgres-transition:{run_key}',
        )
        db.add(run)
        db.flush()
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
            agent_run_id=run.id,
            candidate_contract_version='c5-v1',
        )
        db.add(item)
        db.flush()
        db.add(
            ReviewItemEvidenceRef(
                review_item_id=item.id,
                workflow_thread_id=harness.workflow_thread_id,
                workflow_evidence_ref_id=harness.workflow_evidence_ref_id,
                candidate_slot_ordinal=1,
                message_content_fingerprint='e' * 64,
                fingerprint_key_version='task4-postgres-v1',
                fingerprint_key_material_verifier='f' * 64,
            )
        )
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


def _wait_until_session_is_blocked_on_lock(
    harness: _TransitionHarness,
    *,
    application_name: str,
) -> None:
    deadline = monotonic() + 10
    while monotonic() < deadline:
        with harness.engine.connect() as connection:
            blocked = connection.scalar(
                text(
                    'SELECT EXISTS ('
                    'SELECT 1 FROM pg_stat_activity '
                    'WHERE application_name=:application_name '
                    "AND wait_event_type='Lock')"
                ),
                {'application_name': application_name},
            )
            connection.rollback()
        if blocked:
            return
        sleep(0.05)
    raise AssertionError('stale-evidence resolver did not block on the leading Source lock')


@pytest.mark.parametrize('mutation_kind', ['source', 'document_version', 'binding'])
def test_stale_evidence_uses_source_first_complete_locked_snapshot_without_deadlock(
    postgres_transition_harness: _TransitionHarness,
    mutation_kind: str,
) -> None:
    harness = postgres_transition_harness
    item_id = _seed_item(harness, item_type='history_event')
    application_name = f'task4-stale-{mutation_kind}-{uuid4().hex}'

    def mark_stale() -> str:
        with harness.session_factory() as db:
            db.execute(
                text("SELECT set_config('application_name', :application_name, true)"),
                {'application_name': application_name},
            )
            result = InternalReviewTransitionService(
                evidence_staleness_resolver=(
                    CanonicalReviewEvidenceStalenessResolver()
                )
            ).mark_evidence_stale(db=db, item_id=item_id)
            db.commit()
            return result.status

    with harness.session_factory() as mutator, ThreadPoolExecutor(
        max_workers=1
    ) as executor:
        mutator.scalar(
            select(Source)
            .where(Source.id == harness.source_id)
            .with_for_update()
        )
        stale_future = executor.submit(mark_stale)
        _wait_until_session_is_blocked_on_lock(
            harness,
            application_name=application_name,
        )

        mutator.scalar(
            select(AgentWorkflowThread)
            .where(
                AgentWorkflowThread.thread_id == harness.workflow_thread_id
            )
            .with_for_update()
        )
        mutator.scalar(
            select(ReviewItem)
            .where(ReviewItem.id == item_id)
            .with_for_update()
        )
        if mutation_kind == 'source':
            source = mutator.get(Source, harness.source_id)
            assert source is not None
            source.server_content_signature = 'f' * 64
        elif mutation_kind == 'document_version':
            document = mutator.scalar(
                select(Document)
                .where(Document.id == harness.document_id)
                .with_for_update()
            )
            assert document is not None
            document.current_document_version_id = (
                harness.next_document_version_id
            )
        else:
            evidence_ref = mutator.scalar(
                select(AgentWorkflowEvidenceRef)
                .where(
                    AgentWorkflowEvidenceRef.id
                    == harness.workflow_evidence_ref_id
                )
                .with_for_update()
            )
            assert evidence_ref is not None
            evidence_ref.content_signature = 'f' * 64
        mutator.commit()
        assert stale_future.result(timeout=20) == 'needs_more_evidence'

    with harness.session_factory() as db:
        item = db.get(ReviewItem, item_id)
        assert item is not None
        assert item.status == 'needs_more_evidence'
        assert item.payload['needs_more_evidence']['reason_code'] == (
            'evidence_version_changed'
        )


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
                actor=human_review_actor(USERS['admin']),
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
