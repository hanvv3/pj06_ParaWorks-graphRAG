from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.db.session import get_db
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
)
from backend.app.models import DecisionRecord, HistoryEvent, TimelineEvent, Todo

router = APIRouter(prefix='/knowledge', tags=['knowledge'])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]
PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}


@dataclass(frozen=True, slots=True)
class _EligibleRecord:
    record: object
    permission_level: str

    def __getattr__(self, name: str) -> object:
        return getattr(self.record, name)


def _strictest_permission(levels: list[str]) -> str:
    if not levels:
        return 'internal'
    return max(levels, key=lambda level: PERMISSION_RANK.get(level, 1))


def _source_label(source_url: str) -> str:
    parsed = urlparse(source_url)
    if not parsed.netloc:
        return source_url[:80]
    return f'{parsed.netloc}{parsed.path}'[:80]


def _memory_node(
    *,
    item_id: int,
    item_type: str,
    title: str,
    summary: str,
    permission_level: str,
    confidence_score: float,
    review_status: str,
    created_at: str,
    source_count: int,
) -> dict:
    return {
        'id': f'{item_type}:{item_id}',
        'type': item_type,
        'label': title,
        'summary': summary,
        'permission_level': permission_level,
        'confidence_score': confidence_score,
        'review_status': review_status,
        'created_at': created_at,
        'source_count': source_count,
        'href': {
            'decision': '/decisions',
            'history_event': '/history',
            'timeline_event': '/timeline',
            'todo': '/review',
        }.get(item_type, '/knowledge'),
    }


def _approved_memory_queries(
    db: Session, user: DemoUser
) -> tuple[list[DecisionRecord], list[HistoryEvent], list[TimelineEvent], list[Todo]]:
    decisions = db.scalars(
        select(DecisionRecord)
        .where(DecisionRecord.review_status == 'approved')
        .order_by(DecisionRecord.created_at.desc(), DecisionRecord.id.desc())
    ).all()
    history_events = db.scalars(
        select(HistoryEvent)
        .where(HistoryEvent.review_status == 'approved')
        .order_by(HistoryEvent.created_at.desc(), HistoryEvent.id.desc())
    ).all()
    timeline_events = db.scalars(
        select(TimelineEvent)
        .where(TimelineEvent.review_status == 'approved')
        .order_by(TimelineEvent.created_at.desc(), TimelineEvent.id.desc())
    ).all()
    todos = db.scalars(
        select(Todo).where(Todo.review_status == 'approved').order_by(Todo.created_at.desc(), Todo.id.desc())
    ).all()
    service = TrustedServingEligibilityService(db)
    return (
        _eligible_for_actor(service, 'decision_record', decisions, user),
        _eligible_for_actor(service, 'history_event', history_events, user),
        _eligible_for_actor(service, 'timeline_event', timeline_events, user),
        _eligible_for_actor(service, 'todo', todos, user),
    )


@router.get('/map')
def knowledge_map(db: DbSession, user: CurrentUser) -> dict:
    decisions, history_events, timeline_events, todos = _approved_memory_queries(
        db, user
    )
    memory_nodes: list[dict] = []
    edges: list[dict] = []
    source_permissions: dict[str, list[str]] = defaultdict(list)
    source_snippet_counts: dict[str, int] = defaultdict(int)
    source_connected_memory_ids: dict[str, set[str]] = defaultdict(set)
    permission_counts: Counter[str] = Counter()

    memory_records = [
        (
            'decision',
            item.id,
            item.title,
            item.decision_summary,
            item.permission_level,
            item.confidence_score,
            item.review_status,
            item.created_at.isoformat(),
            item.source_links,
            item.source_snippets,
        )
        for item in decisions
    ]
    memory_records.extend(
        (
            'history_event',
            item.id,
            item.title,
            item.reason,
            item.permission_level,
            item.confidence_score,
            item.review_status,
            item.created_at.isoformat(),
            item.source_links,
            item.source_snippets,
        )
        for item in history_events
    )
    memory_records.extend(
        (
            'timeline_event',
            item.id,
            item.title,
            item.result_summary,
            item.permission_level,
            item.confidence_score,
            item.review_status,
            item.created_at.isoformat(),
            item.source_links,
            item.source_snippets,
        )
        for item in timeline_events
    )
    memory_records.extend(
        (
            'todo',
            item.id,
            item.title,
            item.priority_reason,
            item.permission_level,
            item.confidence_score,
            item.review_status,
            item.created_at.isoformat(),
            item.source_links,
            item.source_snippets,
        )
        for item in todos
    )

    for (
        item_type,
        item_id,
        title,
        summary,
        permission_level,
        confidence_score,
        review_status,
        created_at,
        source_links,
        source_snippets,
    ) in memory_records:
        memory_id = f'{item_type}:{item_id}'
        permission_counts[permission_level] += 1
        memory_nodes.append(
            _memory_node(
                item_id=item_id,
                item_type=item_type,
                title=title,
                summary=summary,
                permission_level=permission_level,
                confidence_score=confidence_score,
                review_status=review_status,
                created_at=created_at,
                source_count=len(source_links),
            )
        )

        for index, source_url in enumerate(source_links):
            evidence_id = f'evidence_source:{source_url}'
            source_permissions[source_url].append(permission_level)
            source_connected_memory_ids[source_url].add(memory_id)
            if index < len(source_snippets) and source_snippets[index]:
                source_snippet_counts[source_url] += 1
            edges.append(
                {
                    'source': memory_id,
                    'target': evidence_id,
                    'relationship': 'supported_by',
                    'permission_level': permission_level,
                }
            )

    evidence_nodes = [
        {
            'id': f'evidence_source:{source_url}',
            'type': 'evidence_source',
            'label': _source_label(source_url),
            'source_url': source_url,
            'permission_level': _strictest_permission(source_permissions[source_url]),
            'connected_memory_count': len(source_connected_memory_ids[source_url]),
            'snippet_count': source_snippet_counts[source_url],
        }
        for source_url in sorted(source_permissions)
    ]

    return {
        'counts': {
            'memory_nodes': len(memory_nodes),
            'evidence_nodes': len(evidence_nodes),
            'edges': len(edges),
            'permission_levels': dict(sorted(permission_counts.items())),
        },
        'nodes': memory_nodes + evidence_nodes,
        'edges': sorted(edges, key=lambda edge: (edge['source'], edge['target'])),
        'cost_policy': {
            'paid_llm_calls': False,
            'embedding_calls': False,
            'sync_jobs_triggered': False,
            'strategy': 'approved_memory_source_link_graph',
        },
    }


@router.get('')
def list_knowledge(db: DbSession, user: CurrentUser) -> dict:
    decisions, history_events, timeline_events, todos = _approved_memory_queries(
        db, user
    )

    return {
        'counts': {
            'decisions': len(decisions),
            'history_events': len(history_events),
            'timeline_events': len(timeline_events),
            'todos': len(todos),
        },
        'decisions': [
            {
                'id': item.id,
                'title': item.title,
                'summary': item.decision_summary,
                'source_links': item.source_links,
                'source_snippets': item.source_snippets,
                'confidence_score': item.confidence_score,
                'permission_level': item.permission_level,
                'review_status': item.review_status,
                'created_at': item.created_at.isoformat(),
            }
            for item in decisions
        ],
        'history_events': [
            {
                'id': item.id,
                'title': item.title,
                'summary': item.reason,
                'source_links': item.source_links,
                'source_snippets': item.source_snippets,
                'confidence_score': item.confidence_score,
                'permission_level': item.permission_level,
                'review_status': item.review_status,
                'created_at': item.created_at.isoformat(),
            }
            for item in history_events
        ],
        'timeline_events': [
            {
                'id': item.id,
                'title': item.title,
                'summary': item.result_summary,
                'source_links': item.source_links,
                'source_snippets': item.source_snippets,
                'confidence_score': item.confidence_score,
                'permission_level': item.permission_level,
                'review_status': item.review_status,
                'created_at': item.created_at.isoformat(),
            }
            for item in timeline_events
        ],
        'todos': [
            {
                'id': item.id,
                'title': item.title,
                'summary': item.priority_reason,
                'priority': item.priority,
                'source_links': item.source_links,
                'source_snippets': item.source_snippets,
                'confidence_score': item.confidence_score,
                'permission_level': item.permission_level,
                'review_status': item.review_status,
                'created_at': item.created_at.isoformat(),
            }
            for item in todos
        ],
    }


def _eligible_for_actor(
    service: TrustedServingEligibilityService,
    knowledge_type: str,
    records: list,
    user: DemoUser,
) -> list:
    visible = []
    for record in records:
        result = service.for_knowledge(knowledge_type, record.id)
        if (
            result.eligible
            and result.effective_permission in user.permission_levels
        ):
            visible.append(_EligibleRecord(record, result.effective_permission))
    return visible
