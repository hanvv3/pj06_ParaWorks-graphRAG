from dataclasses import asdict, replace
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser, get_demo_user, require_admin_user
from backend.app.db.session import get_db
from backend.app.models import Project
from backend.app.permissions.service import can_access_permission
from backend.app.projects import build_project_memory
from backend.app.projects.classifier import (
    build_project_assignment_candidates,
    create_project_assignment_review_items,
)

router = APIRouter(prefix='/projects', tags=['projects'])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]
AdminUser = Annotated[DemoUser, Depends(require_admin_user)]

class ProjectCreate(BaseModel):
    name: str
    summary: str

@router.get('/defined')
def list_defined_projects(db: DbSession, user: CurrentUser) -> dict:
    db_projects = db.scalars(select(Project).order_by(Project.created_at.desc(), Project.id.desc())).all()
    return {
        'projects': [
            {
                'project_key': project.project_key,
                'name': project.name,
                'summary': project.summary,
            }
            for project in db_projects
        ]
    }

@router.post('/define')
def define_project(
    data: ProjectCreate,
    db: DbSession,
    user: AdminUser,
) -> dict:
    import re as regex
    import uuid
    # ASCII kebab-case slug ?앹꽦 ?쒕룄
    slug = regex.sub(r'[^a-z0-9]+', '-', data.name.lower()).strip('-')
    if not slug:
        slug = str(uuid.uuid4())[:8]
    project_key = f"project-{slug}"

    existing = db.scalars(select(Project).where(Project.project_key == project_key)).first()
    if existing:
        # 以묐났 諛⑹?瑜??꾪빐 ?쒕뜡 ?묐???異붽?
        project_key = f"{project_key}-{str(uuid.uuid4())[:4]}"

    project = Project(
        project_key=project_key,
        name=data.name,
        summary=data.summary,
    )
    db.add(project)
    db.flush()
    created = create_project_assignment_review_items(db)
    db.commit()
    db.refresh(project)

    return {
        'status': 'success',
        'created_review_items': len(created),
        'project': {
            'project_key': project.project_key,
            'name': project.name,
            'summary': project.summary,
        }
    }

@router.get('')
def list_projects(db: DbSession, user: CurrentUser) -> dict:
    projects = build_project_memory(db, user)
    visible_projects = [_visible_project(project, user) for project in projects]
    hidden_evidence_count = sum(project.evidence_count - len(visible.evidence) for project, visible in zip(projects, visible_projects, strict=True))
    return {
        'project_count': len(visible_projects),
        'hidden_project_count': 0,
        'hidden_evidence_count': hidden_evidence_count,
        'projects': [asdict(project) for project in visible_projects],
    }


@router.post('/reclassify')
def reclassify_projects(db: DbSession, user: AdminUser, dry_run: bool = True) -> dict:
    candidates = build_project_assignment_candidates(db)
    if dry_run:
        created_count = 0
    else:
        created = create_project_assignment_review_items(db)
        created_count = len(created)
        db.commit()
    counts_by_project: dict[str, int] = {}
    for candidate in candidates:
        counts_by_project[candidate.project_key] = counts_by_project.get(candidate.project_key, 0) + 1
    return {
        'dry_run': dry_run,
        'candidate_count': len(candidates),
        'created_review_items': created_count,
        'counts_by_project': counts_by_project,
        'cost_policy': {
            'paid_llm_calls': False,
            'estimated_input_tokens': 0,
            'estimated_cost_usd': 0,
            'strategy': 'deterministic_project_classifier',
            'requires_human_review_state': True,
        },
    }


def _visible_project(project, user: DemoUser):
    visible_evidence = [
        evidence for evidence in project.evidence if can_access_permission(user, evidence.permission_level)
    ]
    visible_timeline = [
        item for item in project.timeline_items if can_access_permission(user, item.permission_level)
    ]
    visible_activity = [
        item for item in project.activity_items if can_access_permission(user, item.permission_level)
    ]
    visible_levels = [item.permission_level for item in visible_evidence] + [item.permission_level for item in visible_activity]
    return replace(
        project,
        evidence=visible_evidence,
        timeline_items=visible_timeline,
        activity_items=visible_activity,
        evidence_count=len(visible_evidence),
        permission_level=project.permission_level if visible_levels else 'internal',
    )
