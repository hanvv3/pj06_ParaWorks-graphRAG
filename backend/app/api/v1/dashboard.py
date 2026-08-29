from datetime import datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.core.demo_filters import filter_review_items
from backend.app.db.session import get_db
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
)
from backend.app.models import (
    DecisionRecord,
    Project,
    ReviewItem,
    Source,
    SyncJob,
    TimelineEvent,
    Todo,
)
from backend.app.projects import build_project_memory
from backend.app.review.evidence_visibility import (
    ReviewEvidenceNotFound,
    ReviewEvidenceVisibilityService,
)
from backend.app.services.review_display import review_item_display_title

router = APIRouter(prefix='/dashboard', tags=['dashboard'])
DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]


@router.get('')
def get_dashboard(
    db: DbSession, settings: AppSettings, user: CurrentUser
) -> dict:
    source_counts = dict(
        db.execute(
            select(Source.source_type, func.count(Source.id))
            .where(Source.permission_level.in_(tuple(user.permission_levels)))
            .group_by(Source.source_type)
        ).all()
    )
    raw_pending_review_items = db.scalars(
        select(ReviewItem).where(ReviewItem.status == 'pending_review')
    ).all()
    environment_items = (
        raw_pending_review_items
        if settings.paraworks_demo_mode
        else filter_review_items(raw_pending_review_items)
    )
    evidence_service = ReviewEvidenceVisibilityService(db)
    visible_pending_review_items = []
    for item in environment_items:
        try:
            evidence_service.project(item.id, user)
        except ReviewEvidenceNotFound:
            continue
        visible_pending_review_items.append(item)
    sorted_pending_review_items = _sort_review_items_for_queue(visible_pending_review_items)
    pending_review_count = len(sorted_pending_review_items)
    recent_jobs = db.scalars(select(SyncJob).order_by(SyncJob.created_at.desc()).limit(5)).all()

    pending_items = _unique_dashboard_review_items(sorted_pending_review_items)[:3]

    today = _today_kst()
    todo_candidates = db.scalars(
        select(Todo)
        .where(Todo.review_status == 'approved')
        .where(Todo.completed_at.is_(None))
        .order_by(Todo.id.desc())
    ).all()
    eligibility = TrustedServingEligibilityService(db)
    todo_items = sorted(
        [
            item
            for item in todo_candidates
            if _is_due_from_today(item.due_date or '', today)
            and _trusted_for_actor(eligibility, 'todo', item, user)
        ],
        key=lambda item: (item.due_date or '', item.id),
    )[:5]
    calendar_events = _calendar_events(db, user)
    today_events = _today_calendar_events(db, calendar_events)
    project_names = _project_names_by_key(db)

    assigned_projects = build_project_memory(db, user)

    recent_decisions = db.scalars(
        select(DecisionRecord)
        .where(DecisionRecord.review_status == 'approved')
        .order_by(DecisionRecord.created_at.desc())
    ).all()
    recent_decisions = [
        item
        for item in recent_decisions
        if _trusted_for_actor(eligibility, 'decision_record', item, user)
    ][:3]

    recent_timeline = db.scalars(
        select(TimelineEvent)
        .where(TimelineEvent.review_status == 'approved')
        .order_by(TimelineEvent.created_at.desc())
    ).all()
    recent_timeline = [
        item
        for item in recent_timeline
        if _trusted_for_actor(eligibility, 'timeline_event', item, user)
    ][:3]

    return {
        'source_counts': source_counts,
        'pending_review_count': pending_review_count or 0,
        'recent_jobs': [
            {
                'job_id': job.job_id,
                'connector_type': job.connector_type,
                'status': job.status,
                'message': job.message,
                'progress_pct': job.progress_pct,
            }
            for job in recent_jobs
        ],
        'pending_items': [
            {
                'id': item.id,
                'title': review_item_display_title(item),
                'item_type': item.item_type,
                'category': item.payload.get('category', 'Ad-hoc'),
                'confidence_score': item.confidence_score,
                'review_url': f'/review?itemId={item.id}',
            }
            for item in pending_items
        ],
        'today_todos': [
            {
                'id': item.id,
                'title': item.title,
                'assignee': item.assignee or '미정',
                'due_date': item.due_date or '기한 없음',
                'category': project_names.get(item.project_key or '', '프로젝트 미지정'),
                'priority': item.priority or 'medium',
                'completed_at': item.completed_at.isoformat() if item.completed_at else None,
            }
            for item in todo_items
        ],
        'today_events': today_events,
        'calendar_events': calendar_events,
        'assigned_projects': [
            {
                'project_key': project.project_key,
                'name': project.name,
                'summary': project.summary,
                'evidence_count': project.evidence_count,
                'activity_count': len(project.activity_items),
                'pending_review_count': project.pending_review_count,
                'latest_timestamp': project.latest_timestamp,
                'permission_level': project.permission_level,
            }
            for project in assigned_projects
        ],
        'recent_decisions': [
            {
                'id': d.id,
                'title': d.title,
                'summary': d.decision_summary,
                'created_at': d.created_at.isoformat(),
            }
            for d in recent_decisions
        ],
        'recent_timeline': [
            {
                'id': t.id,
                'title': t.title,
                'summary': t.result_summary,
                'created_at': t.created_at.isoformat(),
                'confidence_score': t.confidence_score,
                'source_links': t.source_links,
            }
            for t in recent_timeline
        ],
    }


def _today_kst() -> str:
    return datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()


def _sort_review_items_for_queue(items: list[ReviewItem]) -> list[ReviewItem]:
    return sorted(items, key=_review_queue_sort_key)


def _unique_dashboard_review_items(items: list[ReviewItem]) -> list[ReviewItem]:
    seen_group_keys: set[str] = set()
    unique_items: list[ReviewItem] = []
    for item in items:
        group_key = f'{item.item_type}:{review_item_display_title(item)}'
        if group_key in seen_group_keys:
            continue
        seen_group_keys.add(group_key)
        unique_items.append(item)
    return unique_items


def _review_queue_sort_key(item: ReviewItem) -> tuple[int, int]:
    priority = {
        'decision_record': 0,
        'todo': 1,
        'history_event': 2,
        'timeline_event': 3,
        'project_assignment': 10,
    }.get(item.item_type, 5)
    return (priority, -item.id)


def _today_calendar_events(db: Session, calendar_events: list[dict] | None = None) -> list[dict]:
    kst = ZoneInfo('Asia/Seoul')
    today = datetime.now(kst).date()
    day_start = datetime.combine(today, datetime.min.time(), tzinfo=kst)
    day_end = day_start + timedelta(days=1)
    events = calendar_events if calendar_events is not None else _calendar_events(db)
    today_events: list[tuple[datetime, dict]] = []
    for event in events:
        starts_at = _parse_calendar_datetime(event.get('start'))
        if starts_at is None:
            continue
        starts_at_kst = starts_at.astimezone(kst)
        if day_start <= starts_at_kst < day_end:
            today_events.append((starts_at_kst, event))
    return [event for _, event in sorted(today_events, key=lambda item: (item[0], item[1]['id']))[:5]]


def _calendar_events(db: Session, user: DemoUser | None = None) -> list[dict]:
    kst = ZoneInfo('Asia/Seoul')
    calendar_sources = db.scalars(
        select(Source).where(Source.source_type == 'calendar')
    ).all()
    events: list[tuple[datetime, dict]] = []
    for source in calendar_sources:
        if user is not None and source.permission_level not in user.permission_levels:
            continue
        metadata = source.raw_metadata or {}
        starts_at = _parse_calendar_datetime(metadata.get('event_start') or metadata.get('start'))
        if starts_at is None:
            continue
        starts_at_kst = starts_at.astimezone(kst)
        events.append(
            (
                starts_at_kst,
                {
                    'id': source.id,
                    'title': source.title,
                    'start': _metadata_string(metadata, 'event_start') or _metadata_string(metadata, 'start'),
                    'end': _metadata_string(metadata, 'event_end') or _metadata_string(metadata, 'end'),
                    'location': _metadata_string(metadata, 'location'),
                    'organizer': _metadata_string(metadata, 'organizer_email'),
                    'attendee_summary': _calendar_attendee_summary(metadata),
                    'source_url': source.source_url or '',
                    'permission_level': source.permission_level,
                },
            )
        )
    return [event for _, event in sorted(events, key=lambda item: (item[0], item[1]['id']))[:200]]


def _trusted_for_actor(
    service: TrustedServingEligibilityService,
    knowledge_type: str,
    item: object,
    user: DemoUser,
) -> bool:
    result = service.for_knowledge(knowledge_type, item.id)
    return bool(
        result.eligible
        and result.effective_permission in user.permission_levels
    )


def _parse_calendar_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo('Asia/Seoul'))
    return parsed


def _metadata_string(metadata: dict, key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ''


def _calendar_attendee_summary(metadata: dict) -> str:
    explicit = _metadata_string(metadata, 'calendar_attendee_summary')
    if explicit:
        return explicit
    counts = metadata.get('attendee_response_statuses') or metadata.get('attendee_response_counts')
    if not isinstance(counts, dict):
        return ''
    parts = [
        f'{status} {count}'
        for status, count in sorted(counts.items())
        if isinstance(status, str) and isinstance(count, int)
    ]
    return ', '.join(parts)


def _project_names_by_key(db: Session) -> dict[str, str]:
    projects = db.scalars(select(Project)).all()
    return {project.project_key: project.name for project in projects}


def _is_due_from_today(due_date: str, today: str) -> bool:
    if len(due_date) != 10:
        return False
    return due_date >= today
