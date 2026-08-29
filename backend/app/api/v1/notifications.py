from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.core.demo_filters import filter_review_items
from backend.app.db.session import get_db
from backend.app.models import AgentRun, AgentWorkflowThread, ReviewItem
from backend.app.review.evidence_visibility import (
    ReviewEvidenceNotFound,
    ReviewEvidenceVisibilityService,
)

router = APIRouter(prefix='/notifications', tags=['notifications'])
DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]


@router.get('')
def list_notifications(
    db: DbSession, settings: AppSettings, user: CurrentUser
) -> dict:
    review_notifications = _review_notifications(db, settings, user)
    agent_run_notifications = _agent_run_notifications(db, settings, user)
    notifications = review_notifications + agent_run_notifications
    return {
        'counts': {
            'total': len(notifications),
            'review': len(review_notifications),
            'agent_runs': len(agent_run_notifications),
        },
        'notifications': notifications,
    }


def _review_notifications(
    db: Session, settings: Settings, user: DemoUser
) -> list[dict]:
    pending_count = _review_count(db, settings, user, 'pending_review')
    needs_more_evidence_count = _review_count(
        db, settings, user, 'needs_more_evidence'
    )
    notifications = []
    if pending_count:
        notifications.append(
            {
                'id': 'review:pending_review',
                'category': 'review',
                'severity': 'info',
                'title': '검토 대기 항목',
                'message': f'{pending_count}개 항목이 승인 또는 반려를 기다립니다.',
                'action_href': '/review',
                'source_count': pending_count,
                'created_at': None,
            }
        )
    if needs_more_evidence_count:
        notifications.append(
            {
                'id': 'review:needs_more_evidence',
                'category': 'review',
                'severity': 'warning',
                'title': '근거 추가 필요',
                'message': f'{needs_more_evidence_count}개 항목에 추가 근거가 필요합니다.',
                'action_href': '/review',
                'source_count': needs_more_evidence_count,
                'created_at': None,
            }
        )
    return notifications


def _agent_run_notifications(
    db: Session,
    settings: Settings,
    user: DemoUser | None = None,
) -> list[dict]:
    if user is None:
        return []
    failed_runs = db.scalars(
        select(AgentRun)
        .join(
            AgentWorkflowThread,
            AgentWorkflowThread.thread_id == AgentRun.workflow_thread_id,
        )
        .where(AgentRun.status != 'complete')
        .where(AgentRun.permission_level.in_(tuple(user.permission_levels)))
        .where(AgentWorkflowThread.owner_subject_id == user.id)
        .where(
            AgentWorkflowThread.security_scope_id
            == settings.agent_runtime_security_scope_id
        )
        .order_by(AgentRun.started_at.desc(), AgentRun.id.desc())
        .limit(5)
    ).all()
    return [
        {
            'id': f'agent_run:{run.id}',
            'category': 'agent_run',
            'severity': 'error',
            'title': f'{run.agent_name} 실행 확인 필요',
            'message': _bounded_agent_run_failure(run.status),
            'action_href': f'/agent-runs/{run.id}',
            'source_count': 1,
            'created_at': run.started_at.isoformat(),
        }
        for run in failed_runs
    ]


def _bounded_agent_run_failure(status: str) -> str:
    return {
        'failed': 'Agent run failed',
        'cancelled': 'Agent run cancelled',
        'timed_out': 'Agent run timed out',
    }.get(status, 'Agent run requires review')


def _review_count(
    db: Session,
    settings: Settings,
    user: DemoUser | None,
    status: str,
) -> int:
    if user is None:
        return 0
    items = db.scalars(select(ReviewItem).where(ReviewItem.status == status)).all()
    environment_items = items if settings.paraworks_demo_mode else filter_review_items(items)
    service = ReviewEvidenceVisibilityService(db)
    visible = 0
    for item in environment_items:
        try:
            service.project(item.id, user)
        except ReviewEvidenceNotFound:
            continue
        visible += 1
    return visible
