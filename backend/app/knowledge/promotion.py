from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import (
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    TimelineEvent,
    Todo,
)

PROMOTABLE_REVIEW_TYPES = {'decision_record', 'history_event', 'timeline_event', 'todo'}


class IncompleteReviewPromotionError(ValueError):
    pass


def find_review_item_promotion(db: Session, item: ReviewItem) -> dict | None:
    if item.item_type not in PROMOTABLE_REVIEW_TYPES:
        if item.workflow_thread_id is None:
            return None
        return {
            'target_type': None,
            'created_record_ids': [],
            'created_timeline_event_ids': [],
            'effect': 'created',
        }

    record_model = {
        'decision_record': DecisionRecord,
        'history_event': HistoryEvent,
        'todo': Todo,
    }.get(item.item_type)
    record_ids = (
        list(
            db.scalars(
                select(record_model.id)
                .where(record_model.source_review_item_id == item.id)
                .order_by(record_model.id)
            ).all()
        )
        if record_model is not None
        else []
    )
    timeline_ids = list(
        db.scalars(
            select(TimelineEvent.id)
            .where(TimelineEvent.source_review_item_id == item.id)
            .order_by(TimelineEvent.id)
        ).all()
    )
    if not record_ids and not timeline_ids:
        return None
    expected_record_count = 0 if item.item_type == 'timeline_event' else 1
    if len(record_ids) != expected_record_count or len(timeline_ids) != 1:
        raise IncompleteReviewPromotionError('Review promotion provenance is incomplete')
    return {
        'target_type': item.item_type,
        'created_record_ids': record_ids,
        'created_timeline_event_ids': timeline_ids,
        'effect': 'created',
    }


def build_promotion_response(
    item: ReviewItem,
    *,
    target_type: str | None,
    created_record_ids: list[int],
    created_timeline_event_ids: list[int],
) -> dict:
    return {
        'target_type': target_type or 'review_item',
        'created_record_ids': created_record_ids,
        'created_timeline_event_ids': created_timeline_event_ids,
        'project_key': item.payload.get('project_key'),
        'next_routes': _next_routes_for_item(item.item_type),
    }


def build_promotion_preview(item: ReviewItem) -> dict:
    normalized_payload = _normalized_payload_for_item(item)
    if _is_project_routed_memory_item(item):
        normalized_payload['project_key'] = _string_payload(item, 'project_key')
    if _requires_project_key(item):
        normalized_payload = {
            **normalized_payload,
            'project_key': _string_payload(item, 'project_key'),
        }
    required_fields = _required_fields_for_item(item)
    missing_required_fields = [
        field
        for field, value in normalized_payload.items()
        if field in required_fields and not str(value).strip()
    ]
    if _requires_project_selection(item) and 'project_key' not in missing_required_fields:
        missing_required_fields.append('project_key')

    return {
        'target_type': item.item_type if item.item_type in PROMOTABLE_REVIEW_TYPES else 'review_item',
        'can_approve': not missing_required_fields,
        'missing_required_fields': missing_required_fields,
        'normalized_payload': normalized_payload,
    }


def validate_review_item_for_approval(item: ReviewItem) -> None:
    preview = build_promotion_preview(item)
    if not preview['can_approve']:
        if preview['missing_required_fields'] == ['project_key']:
            raise ValueError('프로젝트를 선택해야 승인할 수 있습니다.')
        raise ValueError('Review item is missing required fields')


def promote_review_item(db: Session, item: ReviewItem) -> dict:
    validate_review_item_for_approval(item)
    normalized = _normalized_payload_for_item(item)
    base_fields = {
        'project_key': item.payload.get('project_key'),
        'source_links': item.source_links,
        'source_snippets': item.source_snippets,
        'confidence_score': item.confidence_score,
        'permission_level': item.permission_level,
        'review_status': 'approved',
        'source_review_item_id': item.id,
    }
    result = {
        'target_type': item.item_type if item.item_type in PROMOTABLE_REVIEW_TYPES else None,
        'created_record_ids': [],
        'created_timeline_event_ids': [],
        'effect': 'created',
        'project_key': item.payload.get('project_key'),
        'next_routes': _next_routes_for_item(item.item_type),
    }

    if item.item_type == 'decision_record':
        decision = DecisionRecord(
            title=normalized['title'],
            decision_summary=normalized['decision_summary'],
            **base_fields,
        )
        timeline = TimelineEvent(
            title=f"[결정] {normalized['title']}",
            result_summary=normalized['decision_summary'],
            **base_fields,
        )
        db.add_all([decision, timeline])
        db.flush()
        result['created_record_ids'] = [decision.id]
        result['created_timeline_event_ids'] = [timeline.id]
        return result

    if item.item_type == 'history_event':
        history = HistoryEvent(
            title=normalized['title'],
            reason=normalized['reason'],
            **base_fields,
        )
        timeline = TimelineEvent(
            title=normalized['title'],
            result_summary=normalized['reason'],
            **base_fields,
        )
        db.add_all([history, timeline])
        db.flush()
        result['created_record_ids'] = [history.id]
        result['created_timeline_event_ids'] = [timeline.id]
        return result

    if item.item_type == 'timeline_event':
        timeline = TimelineEvent(
            title=normalized['title'],
            result_summary=normalized['result_summary'],
            **base_fields,
        )
        db.add(timeline)
        db.flush()
        result['created_timeline_event_ids'] = [timeline.id]
        return result

    if item.item_type == 'todo':
        todo = Todo(
            title=normalized['title'],
            assignee=_string_payload(item, 'assignee') or None,
            due_date=_string_payload(item, 'due_date') or None,
            priority=normalized['priority'],
            priority_reason=normalized['priority_reason'],
            **base_fields,
        )
        timeline = TimelineEvent(
            title=f"[할 일] {normalized['title']}",
            result_summary=(
                f"담당자: {item.payload.get('assignee') or '미정'}, "
                f"기한: {item.payload.get('due_date') or '기한 없음'}"
            ),
            **base_fields,
        )
        db.add_all([todo, timeline])
        db.flush()
        result['created_record_ids'] = [todo.id]
        result['created_timeline_event_ids'] = [timeline.id]
        return result

    return result


def _normalized_payload_for_item(item: ReviewItem) -> dict[str, str]:
    if item.item_type == 'decision_record':
        return {
            'title': _string_payload(item, 'title'),
            'decision_summary': _string_payload(item, 'decision_summary') or _string_payload(item, 'summary'),
        }

    if item.item_type == 'history_event':
        return {
            'title': _string_payload(item, 'title'),
            'reason': _string_payload(item, 'reason') or _string_payload(item, 'summary'),
        }

    if item.item_type == 'timeline_event':
        return {
            'title': _string_payload(item, 'title'),
            'result_summary': (
                _string_payload(item, 'result_summary')
                or _string_payload(item, 'summary')
                or _string_payload(item, 'reason')
            ),
        }

    if item.item_type == 'todo':
        return {
            'title': _string_payload(item, 'title'),
            'priority': _string_payload(item, 'priority') or 'medium',
            'priority_reason': (
                _string_payload(item, 'priority_reason')
                or _string_payload(item, 'recommended_next_step')
                or _string_payload(item, 'task_summary')
                or _string_payload(item, 'summary')
            ),
        }

    if item.item_type == 'project_assignment':
        return {
            'title': _string_payload(item, 'title'),
            'summary': _string_payload(item, 'summary'),
            'project_key': _string_payload(item, 'project_key'),
            'project_name': _string_payload(item, 'project_name'),
            'source_id': _string_payload(item, 'source_id'),
            'evidence_reason': _string_payload(item, 'evidence_reason'),
        }

    return {
        'title': _string_payload(item, 'title'),
        'summary': _string_payload(item, 'summary'),
    }


def _required_fields_for_type(item_type: str) -> tuple[str, ...]:
    if item_type == 'decision_record':
        return ('title', 'decision_summary')
    if item_type == 'history_event':
        return ('title', 'reason')
    if item_type == 'timeline_event':
        return ('title', 'result_summary')
    if item_type == 'todo':
        return ('title', 'priority', 'priority_reason')
    if item_type == 'project_assignment':
        return ('title', 'project_key', 'project_name', 'source_id', 'evidence_reason')
    return ()


def _required_fields_for_item(item: ReviewItem) -> tuple[str, ...]:
    required = _required_fields_for_type(item.item_type)
    if _is_project_routed_memory_item(item):
        return (*required, 'project_key')
    return required


def _is_project_routed_memory_item(item: ReviewItem) -> bool:
    return (
        item.item_type in PROMOTABLE_REVIEW_TYPES
        and item.payload.get('project_assignment_method') == 'llm_tool'
    )


def _requires_project_key(item: ReviewItem) -> bool:
    return (
        item.item_type in PROMOTABLE_REVIEW_TYPES
        and item.payload.get('agent_name') == 'slack_agent'
        and item.payload.get('project_assignment_method') == 'llm_tool'
    )


def _requires_project_selection(item: ReviewItem) -> bool:
    if not _is_project_routed_memory_item(item):
        return False
    return not _string_payload(item, 'project_key') or item.payload.get('project_needs_user_selection') is True


def _string_payload(item: ReviewItem, key: str) -> str:
    value = item.payload.get(key)
    return value if isinstance(value, str) else ''


def _next_routes_for_item(item_type: str) -> list[str]:
    if item_type == 'todo':
        return ['/projects', '/timeline']
    if item_type in {'history_event', 'timeline_event'}:
        return ['/timeline']
    if item_type == 'decision_record':
        return ['/knowledge', '/timeline']
    return []
