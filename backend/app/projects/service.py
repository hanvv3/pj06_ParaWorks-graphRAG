from dataclasses import dataclass, replace
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
)
from backend.app.models import (
    DecisionRecord,
    HistoryEvent,
    Project,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
)
from backend.app.review.evidence_visibility import (
    ReviewEvidenceNotFound,
    ReviewEvidenceVisibilityService,
)

PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}
SOURCE_TYPE_RANK = {'gmail': 0, 'gmail_attachment': 1, 'drive': 2, 'calendar': 3, 'slack': 4}


@dataclass(frozen=True)
class ProjectEvidence:
    id: str
    source_id: str
    source_type: str
    title: str
    source_url: str
    source_snippet: str
    permission_level: str
    timestamp: str
    task_summary: str
    evidence_reason: str


@dataclass(frozen=True)
class ProjectTimelineItem:
    id: str
    item_type: str
    title: str
    summary: str
    source_links: list[str]
    source_snippets: list[str]
    confidence_score: float
    permission_level: str
    review_status: str
    created_at: str
    occurred_at: str
    evidence_reason: str
    project_key: str | None = None
    completed_at: str | None = None
    completed_by: str | None = None


@dataclass(frozen=True)
class ProjectMemory:
    project_key: str
    name: str
    summary: str
    source_types: list[str]
    evidence_count: int
    permission_level: str
    latest_timestamp: str
    pending_review_count: int
    evidence: list[ProjectEvidence]
    timeline_items: list[ProjectTimelineItem]
    activity_items: list[ProjectTimelineItem]


@dataclass(frozen=True)
class _VisibleReviewItem:
    item: ReviewItem
    permission_level: str
    source_links: list[str]
    source_snippets: list[str]

    def __getattr__(self, name: str) -> object:
        return getattr(self.item, name)


def build_project_memory(
    db: Session, user: DemoUser | None = None
) -> list[ProjectMemory]:
    if user is None:
        return []
    approved_assignments = db.scalars(
        select(ReviewItem)
        .where(ReviewItem.item_type == 'project_assignment', ReviewItem.status == 'approved')
        .order_by(ReviewItem.created_at.desc(), ReviewItem.id.desc())
    ).all()
    approved_knowledge_items = db.scalars(
        select(ReviewItem)
        .where(ReviewItem.item_type.in_(['decision_record', 'history_event', 'timeline_event', 'todo']), ReviewItem.status == 'approved')
    ).all()
    evidence_service = ReviewEvidenceVisibilityService(db)
    approved_assignments = _visible_review_items(
        evidence_service, approved_assignments, user
    )
    approved_knowledge_items = _visible_review_items(
        evidence_service, approved_knowledge_items, user
    )
    pending_counts = _pending_assignment_counts(db, user)
    memory_records = _approved_memory_records(db, user)
    db_projects = db.scalars(select(Project).order_by(Project.created_at.desc(), Project.id.desc())).all()

    projects: list[ProjectMemory] = []
    all_approved_items = approved_assignments + approved_knowledge_items
    for db_project in db_projects:
        project_key = db_project.project_key
        project_link_items = [
            item
            for item in all_approved_items
            if item.payload.get('project_key') == project_key
        ]
        assignment_evidence_items = [
            item
            for item in approved_assignments
            if item.payload.get('project_key') == project_key
        ]
        assignment_evidence = _evidence_from_assignments(assignment_evidence_items)
        linked_records = _memory_records_for_project(project_key, project_link_items, memory_records)
        timeline_items = _timeline_items_from_records(linked_records)
        activity_items = _dedupe_activity_items(linked_records)
        evidence = _dedupe_project_evidence([*assignment_evidence, *_evidence_from_activity_items(activity_items)])

        permission_levels = [item.permission_level for item in evidence] + [
            item.permission_level for item in activity_items
        ]
        latest_candidates = [item.timestamp for item in evidence] + [item.occurred_at for item in activity_items]
        source_types = sorted(
            {item.source_type for item in evidence},
            key=lambda source_type: SOURCE_TYPE_RANK.get(source_type, 99),
        )
        projects.append(
            ProjectMemory(
                project_key=project_key,
                name=db_project.name,
                summary=_project_summary(db_project.summary, evidence, activity_items),
                source_types=source_types,
                evidence_count=len(evidence),
                permission_level=_strictest_permission(permission_levels),
                latest_timestamp=max(latest_candidates) if latest_candidates else '',
                pending_review_count=pending_counts.get(project_key, 0),
                evidence=evidence,
                timeline_items=timeline_items,
                activity_items=activity_items,
            )
        )
    return projects


def _pending_assignment_counts(
    db: Session, user: DemoUser | None
) -> dict[str, int]:
    if user is None:
        return {}
    pending = db.scalars(
        select(ReviewItem).where(ReviewItem.status == 'pending_review')
    ).all()
    counts: dict[str, int] = {}
    service = ReviewEvidenceVisibilityService(db)
    for item in pending:
        try:
            service.project(item.id, user)
        except ReviewEvidenceNotFound:
            continue
        project_key = item.payload.get('project_key')
        if isinstance(project_key, str) and project_key:
            counts[project_key] = counts.get(project_key, 0) + 1
    return counts


def _evidence_from_assignments(assignments: list[ReviewItem]) -> list[ProjectEvidence]:
    evidence: list[ProjectEvidence] = []
    seen: set[str] = set()
    for item in assignments:
        payload = item.payload or {}
        source_id = str(payload.get('source_id') or f'review-item-{item.id}')
        project_key = str(payload.get('project_key') or 'unknown')
        identity = f'{project_key}:{source_id}'
        if identity in seen:
            continue
        seen.add(identity)
        source_type = str(payload.get('source_type') or 'source')
        task_summary = str(payload.get('task_summary') or payload.get('summary') or payload.get('source_title') or '')
        evidence_reason = str(payload.get('evidence_reason') or '승인된 프로젝트 연결 후보입니다.')
        timestamp = str(payload.get('timestamp') or item.created_at.isoformat())
        evidence.append(
            ProjectEvidence(
                id=identity,
                source_id=source_id,
                source_type=source_type,
                title=_display_title(task_summary, source_type),
                source_url=item.source_links[0] if item.source_links else '',
                source_snippet=item.source_snippets[0] if item.source_snippets else '',
                permission_level=item.permission_level,
                timestamp=timestamp,
                task_summary=task_summary,
                evidence_reason=evidence_reason,
            )
        )
    return sorted(evidence, key=lambda item: (item.timestamp, item.id), reverse=True)


def _approved_memory_records(
    db: Session, user: DemoUser
) -> list[ProjectTimelineItem]:
    records: list[ProjectTimelineItem] = []
    source_by_url = _source_lookup_by_url(db)
    eligibility = TrustedServingEligibilityService(db)
    records.extend(
        ProjectTimelineItem(
            id=f'decision_record:{item.id}',
            item_type='decision_record',
            title=item.title,
            summary=item.decision_summary,
            source_links=item.source_links,
            source_snippets=item.source_snippets,
            confidence_score=item.confidence_score,
            permission_level=effective_permission,
            review_status=item.review_status,
            created_at=item.created_at.isoformat(),
            occurred_at=_occurred_at_from_source_links(item.source_links, source_by_url, item.created_at),
            evidence_reason='승인된 의사결정 기록이 이 프로젝트와 연결되어 있습니다.',
            project_key=item.project_key,
            completed_at=None,
            completed_by=None,
        )
        for item in db.scalars(select(DecisionRecord).where(DecisionRecord.review_status == 'approved')).all()
        if (
            effective_permission := _knowledge_permission(
                eligibility, 'decision_record', item, user
            )
        ) is not None
    )
    records.extend(
        ProjectTimelineItem(
            id=f'history_event:{item.id}',
            item_type='history_event',
            title=item.title,
            summary=item.reason,
            source_links=item.source_links,
            source_snippets=item.source_snippets,
            confidence_score=item.confidence_score,
            permission_level=effective_permission,
            review_status=item.review_status,
            created_at=item.created_at.isoformat(),
            occurred_at=_occurred_at_from_source_links(item.source_links, source_by_url, item.created_at),
            evidence_reason='승인된 히스토리 기록이 이 프로젝트와 연결되어 있습니다.',
            project_key=item.project_key,
            completed_at=None,
            completed_by=None,
        )
        for item in db.scalars(select(HistoryEvent).where(HistoryEvent.review_status == 'approved')).all()
        if (
            effective_permission := _knowledge_permission(
                eligibility, 'history_event', item, user
            )
        ) is not None
    )
    records.extend(
        ProjectTimelineItem(
            id=f'timeline_event:{item.id}',
            item_type='timeline_event',
            title=item.title,
            summary=item.result_summary,
            source_links=item.source_links,
            source_snippets=item.source_snippets,
            confidence_score=item.confidence_score,
            permission_level=effective_permission,
            review_status=item.review_status,
            created_at=item.created_at.isoformat(),
            occurred_at=_occurred_at_from_source_links(item.source_links, source_by_url, item.created_at),
            evidence_reason='승인된 타임라인 항목이 이 프로젝트와 연결되어 있습니다.',
            project_key=item.project_key,
            completed_at=None,
            completed_by=None,
        )
        for item in db.scalars(select(TimelineEvent).where(TimelineEvent.review_status == 'approved')).all()
        if (
            effective_permission := _knowledge_permission(
                eligibility, 'timeline_event', item, user
            )
        ) is not None
    )
    records.extend(
        ProjectTimelineItem(
            id=f'todo:{item.id}',
            item_type='todo',
            title=item.title,
            summary=item.priority_reason,
            source_links=item.source_links,
            source_snippets=item.source_snippets,
            confidence_score=item.confidence_score,
            permission_level=effective_permission,
            review_status=item.review_status,
            created_at=item.created_at.isoformat(),
            occurred_at=_occurred_at_from_source_links(item.source_links, source_by_url, item.created_at),
            evidence_reason='승인된 할 일이 이 프로젝트와 연결되어 있습니다.',
            project_key=item.project_key,
            completed_at=item.completed_at.isoformat() if item.completed_at else None,
            completed_by=item.completed_by,
        )
        for item in db.scalars(select(Todo).where(Todo.review_status == 'approved')).all()
        if (
            effective_permission := _knowledge_permission(
                eligibility, 'todo', item, user
            )
        ) is not None
    )
    return records


def _visible_review_items(
    service: ReviewEvidenceVisibilityService,
    items: list[ReviewItem],
    user: DemoUser,
) -> list[_VisibleReviewItem]:
    visible: list[_VisibleReviewItem] = []
    for item in items:
        try:
            projection = service.project(item.id, user)
        except ReviewEvidenceNotFound:
            continue
        visible.append(
            _VisibleReviewItem(
                item=item,
                permission_level=projection.effective_permission,
                source_links=list(projection.source_links),
                source_snippets=list(projection.source_snippets),
            )
        )
    return visible


def _knowledge_permission(
    service: TrustedServingEligibilityService,
    knowledge_type: str,
    item: object,
    user: DemoUser,
) -> str | None:
    result = service.for_knowledge(knowledge_type, item.id)
    if not result.eligible or result.effective_permission not in user.permission_levels:
        return None
    return result.effective_permission


def _memory_records_for_project(
    project_key: str,
    assignments: list[ReviewItem],
    memory_records: list[ProjectTimelineItem],
) -> list[ProjectTimelineItem]:
    project_links = {
        link
        for assignment in assignments
        for link in assignment.source_links
        if assignment.payload.get('project_key') == project_key
    }
    project_source_ids = {
        str(assignment.payload.get('source_id'))
        for assignment in assignments
        if assignment.payload.get('project_key') == project_key and assignment.payload.get('source_id')
    }

    items = []
    for item in memory_records:
        if item.project_key == project_key:
            items.append(item)
            continue

        links_text = ' '.join(item.source_links)
        if project_links.intersection(item.source_links) or any(
            source_id and source_id in links_text for source_id in project_source_ids
        ):
            items.append(item)

    return sorted(items, key=lambda item: (item.occurred_at, item.id), reverse=True)


def _timeline_items_from_records(records: list[ProjectTimelineItem]) -> list[ProjectTimelineItem]:
    completed_todos = [
        item for item in records if item.item_type == 'todo' and item.completed_at
    ]
    timeline_items: list[ProjectTimelineItem] = []
    for item in records:
        if item.item_type != 'timeline_event':
            continue
        completed_todo = _matching_completed_todo_for_timeline(item, completed_todos)
        if completed_todo:
            timeline_items.append(
                replace(
                    item,
                    completed_at=completed_todo.completed_at,
                    completed_by=completed_todo.completed_by,
                )
            )
        else:
            timeline_items.append(item)
    return timeline_items


def _dedupe_activity_items(records: list[ProjectTimelineItem]) -> list[ProjectTimelineItem]:
    non_timeline_signatures = {
        _activity_signature(item)
        for item in records
        if item.item_type != 'timeline_event'
    }
    activity_items: list[ProjectTimelineItem] = []
    seen: set[str] = set()
    for item in records:
        signature = _activity_signature(item)
        if item.item_type == 'timeline_event' and signature in non_timeline_signatures:
            continue
        identity = f'{item.item_type}:{signature}'
        if identity in seen:
            continue
        seen.add(identity)
        activity_items.append(item)
    return activity_items


def _evidence_from_activity_items(items: list[ProjectTimelineItem]) -> list[ProjectEvidence]:
    evidence: list[ProjectEvidence] = []
    seen: set[str] = set()
    for item in items:
        for index, link in enumerate(item.source_links):
            if not link.strip():
                continue
            identity = f'{item.project_key}:{link}'
            if identity in seen:
                continue
            seen.add(identity)
            evidence.append(
                ProjectEvidence(
                    id=identity,
                    source_id=link,
                    source_type=_source_type_from_link(link),
                    title=item.title,
                    source_url=link,
                    source_snippet=item.source_snippets[index] if index < len(item.source_snippets) else '',
                    permission_level=item.permission_level,
                    timestamp=item.occurred_at,
                    task_summary=item.summary,
                    evidence_reason=item.evidence_reason,
                )
            )
    return sorted(evidence, key=lambda item: (item.timestamp, item.id), reverse=True)


def _dedupe_project_evidence(items: list[ProjectEvidence]) -> list[ProjectEvidence]:
    deduped: list[ProjectEvidence] = []
    seen: set[str] = set()
    for item in items:
        identity = f'{item.source_url}:{item.source_snippet}'
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return sorted(deduped, key=lambda item: (item.timestamp, item.id), reverse=True)


def _activity_signature(item: ProjectTimelineItem) -> str:
    first_link = item.source_links[0] if item.source_links else ''
    first_snippet = item.source_snippets[0] if item.source_snippets else ''
    return '|'.join(
        [
            first_link.strip().lower(),
            ' '.join(first_snippet.split()).strip().lower(),
            ' '.join(item.summary.split()).strip().lower(),
        ]
    )


def _matching_completed_todo_for_timeline(
    timeline_item: ProjectTimelineItem,
    completed_todos: list[ProjectTimelineItem],
) -> ProjectTimelineItem | None:
    timeline_title = _normalized_todo_timeline_title(timeline_item.title)
    timeline_links = {link for link in timeline_item.source_links if link.strip()}
    for todo in completed_todos:
        if todo.project_key != timeline_item.project_key:
            continue
        if todo.title.strip() != timeline_title:
            continue
        todo_links = {link for link in todo.source_links if link.strip()}
        if timeline_links and todo_links and timeline_links.isdisjoint(todo_links):
            continue
        return todo
    return None


def _normalized_todo_timeline_title(title: str) -> str:
    cleaned = title.strip()
    for prefix in ('[할 일]', '[할일]', '할 일:', '할일:'):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


def _source_lookup_by_url(db: Session) -> dict[str, Source]:
    sources = db.scalars(select(Source)).all()
    return {source.source_url: source for source in sources if source.source_url}


def _occurred_at_from_source_links(
    source_links: list[str],
    source_by_url: dict[str, Source],
    fallback: datetime,
) -> str:
    for link in source_links:
        source = source_by_url.get(link)
        if source:
            raw_metadata = source.raw_metadata or {}
            if source.source_type == 'calendar':
                raw_event_start = raw_metadata.get('event_start') or raw_metadata.get('start')
                if isinstance(raw_event_start, str) and raw_event_start.strip():
                    try:
                        return datetime.fromisoformat(raw_event_start.replace('Z', '+00:00')).astimezone(UTC).isoformat()
                    except ValueError:
                        pass
            raw_ts = raw_metadata.get('ts')
            if isinstance(raw_ts, str):
                try:
                    return datetime.fromtimestamp(float(raw_ts), tz=UTC).isoformat()
                except ValueError:
                    pass

        parsed_ts = _slack_ts_from_permalink(link)
        if parsed_ts is not None:
            return datetime.fromtimestamp(parsed_ts, tz=UTC).isoformat()

        if source:
            return source.created_at.isoformat()

    return fallback.isoformat()


def _slack_ts_from_permalink(link: str) -> float | None:
    if '/p' not in link:
        return None
    raw = link.rsplit('/p', 1)[-1].split('?', 1)[0]
    if len(raw) < 11 or not raw.isdigit():
        return None
    seconds = raw[:10]
    micros = raw[10:].ljust(6, '0')[:6]
    return float(f'{seconds}.{micros}')


def _source_type_from_link(link: str) -> str:
    lowered = link.lower()
    if 'slack' in lowered:
        return 'slack'
    if 'mail.google' in lowered or 'gmail' in lowered:
        return 'gmail'
    if 'drive.google' in lowered or 'docs.google' in lowered:
        return 'drive'
    if 'calendar.google' in lowered or 'calendar' in lowered:
        return 'calendar'
    return 'source'


def _project_summary(base_summary: str, evidence: list[ProjectEvidence], activity_items: list[ProjectTimelineItem]) -> str:
    if not evidence and not activity_items:
        return f'{base_summary} 아직 승인된 프로젝트 근거가 없습니다.'
    return (
        f'{base_summary} 승인된 원본 근거 {len(evidence)}건과 '
        f'승인된 프로젝트 활동 {len(activity_items)}건이 연결되어 있습니다.'
    )


def _display_title(task_summary: str, source_type: str) -> str:
    cleaned = ' '.join(task_summary.split()).strip()
    if cleaned and not cleaned.lower().startswith(('slack message in ', 'slack thread reply in ')):
        return cleaned[:120]
    return f'{_source_type_label(source_type)} evidence'


def _source_type_label(source_type: str) -> str:
    return {
        'gmail': 'Gmail',
        'gmail_attachment': 'Gmail 첨부',
        'drive': 'Drive',
        'calendar': 'Calendar',
        'slack': 'Slack',
    }.get(source_type, 'Source')


def _strictest_permission(levels: list[str]) -> str:
    if not levels:
        return 'internal'
    return max(levels, key=lambda level: PERMISSION_RANK.get(level, 1))
