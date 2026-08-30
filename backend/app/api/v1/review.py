from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.core.demo_filters import filter_review_items
from backend.app.core.rbac import ensure_can_review_permission
from backend.app.db.session import get_db
from backend.app.knowledge.promotion import (
    build_promotion_preview,
    build_promotion_response,
)
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRolloutState,
    AutoReviewValidation,
    Project,
    ReviewItem,
    Source,
)
from backend.app.review.actors import ReviewResolutionActor, human_review_actor
from backend.app.review.auto_review_audit import (
    AutoReviewPostAuditTransitionService,
)
from backend.app.review.auto_review_quality_revoke import (
    AutoReviewQualityRevokeService,
    QualityRevokeRefused,
)
from backend.app.review.auto_review_revoke import (
    AutoReviewRevokeRefused,
    AutoReviewRevokeService,
)
from backend.app.review.auto_review_rollout import RolloutGateError
from backend.app.review.evidence_visibility import (
    ReviewEvidenceNotFound,
    ReviewEvidenceProjection,
    ReviewEvidenceVisibilityService,
)
from backend.app.review.transitions import (
    InvalidReviewTransition,
    ReviewAction,
    ReviewTransitionResult,
    ReviewTransitionService,
)
from backend.app.schemas.auto_review import (
    AUTO_REVIEW_POLICY_VERSION,
    AUTO_REVIEW_REASONING_EFFORT,
    AUTO_REVIEW_VALIDATOR_MODEL,
    AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
    AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
)
from backend.app.schemas.review import (
    AutoReviewAuditRequest,
    AutoReviewAuditResponse,
    ReviewBulkActionRequest,
    ReviewEvidenceRequest,
    ReviewItemUpdate,
    RevokeAutoApprovalRequest,
    RevokeAutoApprovalResponse,
)
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
)
from backend.app.services.audit import (
    record_audit_log,
    record_auto_review_control_audit,
    record_review_resolution_audit,
)
from backend.app.services.review_display import review_item_display_title

router = APIRouter(prefix='/review', tags=['review'])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def _review_item_response(
    item: ReviewItem,
    agent_run: AgentRun | None = None,
    evidence: ReviewEvidenceProjection | None = None,
    *,
    auto_review_summary: dict[str, object] | None = None,
    auto_review_audit: dict[str, object] | None = None,
) -> dict:
    agent_run_id = _agent_run_id(item)

    # 에이전트 실행 상세 정보 추출
    agent_details = {
        'model_name': None,
        'prompt_version': None,
        'estimated_cost_usd': None,
        'total_tokens': 0,
    }

    evidence_available = evidence is None or evidence.evidence_available
    if not evidence_available:
        agent_run_id = None
        agent_run = None

    if agent_run:
        agent_details.update({
            'model_name': agent_run.model_name or 'gpt-4o-mini',
            'prompt_version': agent_run.prompt_version or 'v1',
            'estimated_cost_usd': agent_run.estimated_cost_usd or 0.0,
            'total_tokens': agent_run.total_tokens or 0,
        })

    return {
        'id': item.id,
        'item_type': item.item_type,
        'payload': (
            item.payload
            if evidence_available
            else _payload_without_source_fields(item.payload)
        ),
        'source_links': item.source_links if evidence_available else [],
        'source_snippets': item.source_snippets if evidence_available else [],
        'source_evidence': (
            _source_evidence_response(item, agent_run)
            if evidence_available
            else []
        ),
        'evidence_status': (
            evidence.evidence_status if evidence is not None else 'available'
        ),
        'action_required': evidence.action_required if evidence else False,
        'agent_run_id': agent_run_id,
        'agent_run_details': agent_details,  # 상세 정보 추가
        'confidence_score': item.confidence_score,
        'permission_level': (
            evidence.effective_permission
            if evidence is not None
            else item.permission_level
        ),
        'status': item.status,
        'reviewer_id': item.reviewer_id,
        'resolution_source': item.resolution_source,
        'resolution_policy_version': item.resolution_policy_version,
        'auto_review_summary': auto_review_summary,
        'auto_review_audit': auto_review_audit,
    }


_SAFE_UNAVAILABLE_PAYLOAD_FIELDS = frozenset({
    'agent_name',
    'assignee',
    'decision_summary',
    'due_date',
    'priority',
    'priority_reason',
    'project_assignment_method',
    'project_assignment_reason',
    'project_assignment_summary',
    'project_key',
    'project_name',
    'project_needs_user_selection',
    'reason',
    'result_summary',
    'summary',
    'task_summary',
    'title',
})


def _payload_without_source_fields(payload: dict) -> dict:
    projected = {
        key: value
        for key, value in (payload or {}).items()
        if key in _SAFE_UNAVAILABLE_PAYLOAD_FIELDS
        and _is_safe_unavailable_scalar(value)
    }
    request = (payload or {}).get('needs_more_evidence')
    if isinstance(request, dict):
        projected['needs_more_evidence'] = {
            key: request[key]
            for key in ('requested_at', 'requested_by', 'note', 'previous_status')
            if key in request
            and _is_safe_unavailable_scalar(request[key])
        }
    return projected


def _is_safe_unavailable_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, float, int, str))


@router.get('')
def list_review_items(
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
    status: str = 'pending_review',
    resolution_source: Literal['human', 'auto_policy'] | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_previews: bool = False,
    workflow_thread_id: str | None = Query(default=None, min_length=1, max_length=64),
) -> dict:
    statement = select(ReviewItem).where(ReviewItem.status == status)
    if resolution_source is not None:
        statement = statement.where(
            ReviewItem.resolution_source == resolution_source
        )
    if workflow_thread_id is not None:
        statement = statement.where(
            ReviewItem.workflow_thread_id == workflow_thread_id
        )
    items = db.scalars(statement.order_by(
        ReviewItem.created_at.desc(), ReviewItem.id.desc()
    )).all()
    all_visible_items = _visible_review_items(db, items, user, settings)
    if workflow_thread_id is not None:
        all_visible_items = _visible_workflow_items(
            db,
            items=all_visible_items,
            user=user,
            settings=settings,
            workflow_thread_id=workflow_thread_id,
        )
    all_visible_items = _sort_review_items_for_queue(all_visible_items)
    total_count = len(all_visible_items)
    visible_items = all_visible_items[offset : offset + limit]
    agent_runs = _agent_runs_by_id(db, visible_items)
    evidence_by_id = {
        item.id: ReviewEvidenceVisibilityService(db).project(item.id, user)
        for item in visible_items
    }

    groups: dict[str, dict] = {}
    for item in visible_items:
        agent_run = agent_runs.get(_agent_run_id(item) or -1)
        response_item = _review_item_response(
            item,
            agent_run,
            evidence_by_id[item.id],
            auto_review_summary=_auto_review_summary(db, item),
            auto_review_audit=_auto_review_audit(db, item),
        )
        title = review_item_display_title(item)
        group_key = f'{item.item_type}:{title}'

        if group_key not in groups:
            groups[group_key] = {
                'group_id': group_key,
                'title': title,
                'item_type': item.item_type,
                'status': item.status,
                'permission_level': item.permission_level,
                'items': [],
                'total_count': 0,
                'avg_confidence': 0.0,
            }

        groups[group_key]['items'].append(response_item)
        groups[group_key]['total_count'] += 1
        groups[group_key]['avg_confidence'] += item.confidence_score

    result_groups = []
    for group in groups.values():
        if group['total_count'] > 0:
            group['avg_confidence'] /= group['total_count']
        result_groups.append(group)

    return {
        'groups': result_groups,
        'items': [
            _review_item_response(
                item,
                agent_runs.get(_agent_run_id(item) or -1),
                evidence_by_id[item.id],
                auto_review_summary=_auto_review_summary(db, item),
                auto_review_audit=_auto_review_audit(db, item),
            )
            for item in visible_items
        ],
        'total_count': total_count,
        'limit': limit,
        'offset': offset,
        'has_more': offset + limit < total_count,
        'include_previews': include_previews,
    }


@router.post('/approve-agent-candidates')
def approve_agent_review_candidates(
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    all_items = db.scalars(
        select(ReviewItem)
        .where(ReviewItem.status.in_(['pending_review', 'approved']))
        .order_by(ReviewItem.id)
    ).all()
    visible_items = _visible_review_items(db, all_items, user, settings)
    candidate_items = [item for item in visible_items if _is_agent_candidate(item)]
    actor = _human_actor_for_items(user, candidate_items)
    batch = ReviewTransitionService().transition_many(
        db=db,
        item_ids=[item.id for item in candidate_items],
        action='approve',
        actor=actor,
    )
    approved_item_ids = [result.item_id for result in batch.results]
    skipped_count = (
        len(visible_items) - len(candidate_items) + len(batch.failed_items) + len(batch.skipped_items)
    )
    if any(not result.replayed for result in batch.results) or batch.failed_items:
        record_audit_log(
            db=db,
            actor=user,
            action='review.approve_agent_candidates',
            target_type='review_queue',
            target_id='agent_candidates',
            metadata={
                'action': 'approve',
                'approved_count': len(approved_item_ids),
                'replayed_count': sum(result.replayed for result in batch.results),
                'failed_count': len(batch.failed_items),
                'skipped_count': skipped_count,
            },
        )
    db.commit()

    return {
        'approved_count': len(approved_item_ids),
        'skipped_count': skipped_count,
        'approved_item_ids': approved_item_ids,
        'replayed_items': [
            _transition_result_response(result)
            for result in batch.results
            if result.replayed
        ],
        'cost_policy': {
            'paid_llm_calls': False,
            'embedding_calls': False,
            'requires_human_review_state': True,
        },
    }


@router.post('/bulk')
def bulk_review_items(
    request: ReviewBulkActionRequest,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    items = _bulk_action_items(db, request, user, settings)
    actor = _human_actor_for_items(user, items)
    batch = ReviewTransitionService().transition_many(
        db=db,
        item_ids=[item.id for item in items],
        action=request.action,
        actor=actor,
    )
    approved_item_ids = [result.item_id for result in batch.results if result.status == 'approved']
    rejected_item_ids = [result.item_id for result in batch.results if result.status == 'rejected']
    if any(not result.replayed for result in batch.results) or batch.failed_items:
        record_audit_log(
            db=db,
            actor=user,
            action=f'review.bulk_{request.action}',
            target_type='review_queue',
            target_id='bulk',
            metadata={
                'action': request.action,
                'approved_count': len(approved_item_ids),
                'rejected_count': len(rejected_item_ids),
                'replayed_count': sum(result.replayed for result in batch.results),
                'failed_count': len(batch.failed_items),
                'skipped_count': len(batch.skipped_items),
            },
        )
    db.commit()

    return {
        'action': request.action,
        'approved_count': len(approved_item_ids),
        'rejected_count': len(rejected_item_ids),
        'failed_items': list(batch.failed_items),
        'skipped_items': list(batch.skipped_items),
        'approved_item_ids': approved_item_ids,
        'rejected_item_ids': rejected_item_ids,
        'replayed_items': [
            _transition_result_response(result)
            for result in batch.results
            if result.replayed
        ],
        'cost_policy': {
            'paid_llm_calls': False,
            'embedding_calls': False,
            'requires_human_review_state': True,
        },
    }


@router.patch('/{item_id}')
def update_review_item(
    item_id: int,
    update: ReviewItemUpdate,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    item = _get_review_item_for_user(db, item_id, user, settings)
    ensure_can_review_permission(user, item.permission_level)
    if item.workflow_thread_id is not None and item.status != 'pending_review':
        _raise_invalid_transition_http()

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == 'payload' and isinstance(value, dict):
            # 병합(merge)하여 기존 payload의 다른 필드 유실 방지
            new_payload = dict(item.payload or {})
            new_payload.update(value)
            if 'project_key' in value:
                _validate_payload_project_key(db, new_payload)
            item.payload = new_payload
        else:
            setattr(item, field, value)

    db.commit()
    db.refresh(item)
    return _projected_review_item_response(db, item, user)


def _is_agent_candidate(item: ReviewItem) -> bool:
    return isinstance(item.payload.get('agent_name'), str)


@router.get('/{item_id}/promotion-preview')
def preview_review_item_promotion(
    item_id: int,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    item = _get_review_item_for_user(db, item_id, user, settings)
    return build_promotion_preview(item)


@router.post('/{item_id}/approve')
def approve_review_item(
    item_id: int,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    item = _get_review_item_for_user(db, item_id, user, settings)
    actor = _human_actor_for_item(user, item)
    result = _transition_or_http(
        db=db,
        item_id=item.id,
        action='approve',
        actor=actor,
    )
    if not result.replayed:
        _record_transition_audit(
            db=db,
            actor=actor,
            human_user=user,
            item=item,
            action='approve',
            result=result,
        )
    db.commit()
    db.refresh(item)
    response = _projected_review_item_response(db, item, user)
    response['replayed'] = result.replayed
    response['promotion'] = _promotion_response(result)
    response['promotion_result'] = _legacy_promotion_response(item, result)
    return response


@router.post('/{item_id}/request-more-evidence')
def request_more_evidence_for_review_item(
    item_id: int,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
    request: ReviewEvidenceRequest | None = None,
) -> dict:
    item = _get_review_item_for_user(db, item_id, user, settings)
    actor = _human_actor_for_item(user, item)
    note = (request.note or '').strip() if request else ''
    result = _transition_or_http(
        db=db,
        item_id=item.id,
        action='needs_more_evidence',
        actor=actor,
        note=note,
    )
    _record_transition_audit(
        db=db,
        actor=actor,
        human_user=user,
        item=item,
        action='needs_more_evidence',
        result=result,
        note=note,
    )
    db.commit()
    db.refresh(item)
    response = _projected_review_item_response(db, item, user)
    response['replayed'] = result.replayed
    response['promotion'] = _promotion_response(result)
    return response


@router.post('/{item_id}/reject')
def reject_review_item(
    item_id: int,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    item = _get_review_item_for_user(db, item_id, user, settings)
    actor = _human_actor_for_item(user, item)
    result = _transition_or_http(
        db=db,
        item_id=item.id,
        action='reject',
        actor=actor,
    )
    _record_transition_audit(
        db=db,
        actor=actor,
        human_user=user,
        item=item,
        action='reject',
        result=result,
    )
    db.commit()
    db.refresh(item)
    response = _projected_review_item_response(db, item, user)
    response['replayed'] = result.replayed
    response['promotion'] = _promotion_response(result)
    return response


@router.post(
    '/{item_id}/revoke-auto-approval',
    response_model=RevokeAutoApprovalResponse,
)
def revoke_auto_approval(
    item_id: int,
    request: RevokeAutoApprovalRequest,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> RevokeAutoApprovalResponse:
    item = _get_review_item_for_user(db, item_id, user, settings)
    actor = _human_actor_for_item(user, item)
    try:
        if request.reason_code == 'business_withdrawal':
            result = AutoReviewRevokeService(
                db, settings=settings
            ).revoke(
                review_item_id=item.id,
                actor=actor,
                reason_code=request.reason_code,
            )
        else:
            quality = AutoReviewQualityRevokeService(
                db, settings=settings
            ).revoke_quality(
                review_item_id=item.id,
                actor=actor,
                reason_code=request.reason_code,
                reason=f'operator_reason:{request.reason_code}',
            )
            if quality.remediation_required:
                raise HTTPException(
                    status_code=409,
                    detail={'code': 'remediation_required'},
                )
            item = _get_review_item_for_user(db, item_id, user, settings)
            result = RevokeAutoApprovalResponse(
                review_item_id=item.id,
                status='revoked',
                replayed=quality.replayed,
                knowledge_remains_trusted=bool(
                    item.revoke_knowledge_remained_trusted
                ),
                revoked_document_count=item.revoke_document_count or 0,
            )
    except AutoReviewRevokeRefused as exc:
        _raise_auto_review_conflict(exc.code)
    except QualityRevokeRefused as exc:
        _raise_auto_review_conflict(exc.code)
    record_auto_review_control_audit(
        db=db,
        actor=user,
        action='review.auto_revoke',
        review_item_id=item.id,
        outcome='revoked',
        replayed=result.replayed,
    )
    db.commit()
    return RevokeAutoApprovalResponse(
        review_item_id=result.review_item_id,
        status='revoked',
        replayed=result.replayed,
        knowledge_remains_trusted=result.knowledge_remains_trusted,
        revoked_document_count=result.revoked_document_count,
    )


@router.post(
    '/{item_id}/auto-review-audit',
    response_model=AutoReviewAuditResponse,
)
def submit_auto_review_audit(
    item_id: int,
    request: AutoReviewAuditRequest,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> AutoReviewAuditResponse:
    item = _get_review_item_for_user(db, item_id, user, settings)
    actor = _human_actor_for_item(user, item)
    try:
        if request.outcome == 'confirmed':
            audit_service = AutoReviewPostAuditTransitionService(
                db, settings=settings
            )
            audit = audit_service.ensure_manual_audit(
                review_item_id=item.id,
                actor=actor,
            )
            transition = audit_service.complete(
                audit_id=audit.id,
                actor=actor,
                outcome='confirmed',
                reason=request.reason,
            )
            db.commit()
            response = AutoReviewAuditResponse(
                audit_status='completed',
                breaker_open=_auto_review_breaker_open(db, item.id),
                revoke_status='not_required',
            )
            replayed = transition.replayed
        else:
            reason_code = {
                'incorrect': 'incorrect_content',
                'permission_violation': 'permission_violation',
                'source_version_violation': 'wrong_source_version',
                'policy_violation': 'policy_violation',
            }[request.outcome]
            quality = AutoReviewQualityRevokeService(
                db, settings=settings
            ).revoke_quality(
                review_item_id=item.id,
                actor=actor,
                reason_code=reason_code,
                reason=request.reason,
            )
            if quality.remediation_required:
                raise HTTPException(
                    status_code=409,
                    detail={'code': 'remediation_required'},
                )
            response = AutoReviewAuditResponse(
                audit_status='completed',
                breaker_open=_auto_review_breaker_open(db, item.id),
                revoke_status='revoked',
            )
            replayed = quality.replayed
    except (QualityRevokeRefused, AutoReviewRevokeRefused) as exc:
        _raise_auto_review_conflict(exc.code)
    except RolloutGateError as exc:
        code = (
            'audit_conflict'
            if 'immutable' in str(exc)
            else 'audit_unavailable'
        )
        raise HTTPException(status_code=409, detail={'code': code}) from exc
    record_auto_review_control_audit(
        db=db,
        actor=user,
        action='review.auto_audit',
        review_item_id=item.id,
        outcome=response.audit_status,
        replayed=replayed,
    )
    db.commit()
    return response


def _raise_auto_review_conflict(code: str) -> None:
    if code == 'not_found':
        raise HTTPException(status_code=404, detail='Review item not found')
    allowed = {
        'audit_required',
        'quality_audit_required',
        'remediation_required',
        'revoke_reason_conflict',
        'unsupported_transition',
    }
    public_code = code if code in allowed else 'invalid_auto_review_action'
    raise HTTPException(status_code=409, detail={'code': public_code})


def _transition_or_http(
    *,
    db: Session,
    item_id: int,
    action: ReviewAction,
    actor: ReviewResolutionActor,
    note: str | None = None,
) -> ReviewTransitionResult:
    try:
        return ReviewTransitionService().transition(
            db=db,
            item_id=item_id,
            action=action,
            actor=actor,
            note=note,
        )
    except InvalidReviewTransition as exc:
        raise HTTPException(status_code=409, detail={'code': exc.code}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _raise_invalid_transition_http() -> None:
    raise HTTPException(status_code=409, detail={'code': InvalidReviewTransition.code})


def _transition_result_response(result: ReviewTransitionResult) -> dict[str, object]:
    return {
        'item_id': result.item_id,
        'status': result.status,
        'replayed': result.replayed,
        'promotion': _promotion_response(result),
    }


def _promotion_response(result: ReviewTransitionResult) -> dict[str, object] | None:
    if result.promotion is None:
        return None
    return {
        'target_type': result.promotion.target_type,
        'created_record_ids': list(result.promotion.created_record_ids),
        'created_timeline_event_ids': list(result.promotion.created_timeline_event_ids),
    }


def _legacy_promotion_response(item: ReviewItem, result: ReviewTransitionResult) -> dict | None:
    if result.promotion is None:
        return None
    return build_promotion_response(
        item,
        target_type=result.promotion.target_type,
        created_record_ids=list(result.promotion.created_record_ids),
        created_timeline_event_ids=list(result.promotion.created_timeline_event_ids),
    )


def _record_transition_audit(
    *,
    db: Session,
    actor: ReviewResolutionActor,
    human_user: DemoUser,
    item: ReviewItem,
    action: ReviewAction,
    result: ReviewTransitionResult,
    note: str | None = None,
) -> None:
    audit_action = {
        'approve': 'review.approve',
        'reject': 'review.reject',
        'needs_more_evidence': 'review.request_more_evidence',
    }[action]
    if item.workflow_thread_id is not None:
        effect_count = 0
        if result.promotion is not None:
            effect_count = (
                len(result.promotion.created_record_ids)
                + len(result.promotion.created_timeline_event_ids)
            )
        record_review_resolution_audit(
            db=db,
            actor=actor,
            human_user=human_user,
            action=audit_action,
            review_item_id=item.id,
            outcome=result.status,
            target_type='review_workflow',
            target_id=item.workflow_thread_id,
            metadata={
                'replayed': result.replayed,
                'effect_count': effect_count,
            },
        )
        return

    if action == 'approve':
        metadata: dict[str, object] = {
            'item_type': item.item_type,
            'permission_level': item.permission_level,
        }
    elif action == 'reject':
        metadata = {
            'item_type': item.item_type,
        }
    else:
        metadata = {
            'item_type': item.item_type,
            'note_present': bool(note),
            'source_count': len(item.source_snippets or []),
        }
    record_review_resolution_audit(
        db=db,
        actor=actor,
        human_user=human_user,
        action=audit_action,
        review_item_id=item.id,
        outcome=result.status,
        target_type='review_item',
        target_id=item.id,
        metadata=metadata,
    )


def _human_actor_for_item(
    user: DemoUser,
    item: ReviewItem,
) -> ReviewResolutionActor:
    ensure_can_review_permission(user, item.permission_level)
    if item.permission_level not in user.permission_levels:
        raise HTTPException(
            status_code=403,
            detail='Review approval permission required.',
        )
    return human_review_actor(user)


def _human_actor_for_items(
    user: DemoUser,
    items: list[ReviewItem],
) -> ReviewResolutionActor:
    del items
    return human_review_actor(user)


def _visible_review_items(
    db: Session,
    items: list[ReviewItem],
    user: DemoUser,
    settings: Settings,
) -> list[ReviewItem]:
    environment_items = items if settings.paraworks_demo_mode else filter_review_items(items)
    service = ReviewEvidenceVisibilityService(db)
    visible: list[ReviewItem] = []
    for item in environment_items:
        if not _user_can_see_review_item(user, item):
            continue
        try:
            service.project(item.id, user)
        except ReviewEvidenceNotFound:
            continue
        visible.append(item)
    return visible


def _visible_workflow_items(
    db: Session,
    *,
    items: list[ReviewItem],
    user: DemoUser,
    settings: Settings,
    workflow_thread_id: str,
) -> list[ReviewItem]:
    thread = db.get(AgentWorkflowThread, workflow_thread_id)
    if (
        thread is None
        or thread.security_scope_id != settings.agent_runtime_security_scope_id
        or thread.workflow_name != COMPANY_MEMORY_REVIEW_WORKFLOW
        or thread.graph_version not in {
            COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
        }
    ):
        return []
    refs = tuple(
        db.scalars(
            select(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id == workflow_thread_id
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).all()
    )
    if not refs or any(ref.canonical_table != 'sources' for ref in refs):
        return []
    source_ids = tuple(ref.canonical_row_id for ref in refs)
    sources = tuple(
        db.scalars(select(Source).where(Source.id.in_(source_ids))).all()
    )
    sources_by_id = {source.id: source for source in sources}
    if len(sources_by_id) != len(source_ids) or any(
        sources_by_id[ref.canonical_row_id].source_type
        != ref.canonical_source_type
        or sources_by_id[ref.canonical_row_id].permission_level
        not in user.permission_levels
        for ref in refs
    ):
        return []
    return [item for item in items if item.workflow_thread_id == workflow_thread_id]


def _sort_review_items_for_queue(items: list[ReviewItem]) -> list[ReviewItem]:
    return sorted(items, key=_review_queue_sort_key)


def _review_queue_sort_key(item: ReviewItem) -> tuple[int, int]:
    priority = {
        'decision_record': 0,
        'todo': 1,
        'history_event': 2,
        'timeline_event': 3,
        'project_assignment': 10,
    }.get(item.item_type, 5)
    return (priority, -item.id)


def _bulk_action_items(
    db: Session,
    request: ReviewBulkActionRequest,
    user: DemoUser,
    settings: Settings,
) -> list[ReviewItem]:
    if request.item_ids is not None:
        if not request.item_ids:
            return []
        items = db.scalars(
            select(ReviewItem)
            .where(ReviewItem.id.in_(request.item_ids))
            .order_by(ReviewItem.created_at.desc(), ReviewItem.id.desc())
        ).all()
        by_id = {item.id: item for item in items}
        items = [by_id[item_id] for item_id in request.item_ids if item_id in by_id]
    else:
        items = db.scalars(
            select(ReviewItem)
            .where(ReviewItem.status == 'pending_review')
            .order_by(ReviewItem.created_at.desc(), ReviewItem.id.desc())
        ).all()
    return _visible_review_items(db, items, user, settings)


def _validate_payload_project_key(db: Session, payload: dict) -> None:
    raw_key = payload.get('project_key')
    if raw_key in (None, ''):
        payload.pop('project_name', None)
        return
    if not isinstance(raw_key, str):
        raise HTTPException(status_code=400, detail='Project key must be a string')
    project = db.scalar(select(Project).where(Project.project_key == raw_key))
    if project is None:
        raise HTTPException(status_code=400, detail='Project key is not registered')
    payload['project_name'] = project.name
    if payload.get('project_assignment_method') == 'llm_tool':
        payload['project_needs_user_selection'] = False


def _get_review_item_for_user(db: Session, item_id: int, user: DemoUser, settings: Settings) -> ReviewItem:
    item = db.get(ReviewItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail='Review item not found')
    if item not in _visible_review_items(db, [item], user, settings):
        raise HTTPException(status_code=404, detail='Review item not found')
    return item


def _user_can_see_review_item(user: DemoUser, item: ReviewItem) -> bool:
    return item.permission_level in user.permission_levels


def _projected_review_item_response(
    db: Session, item: ReviewItem, user: DemoUser
) -> dict:
    try:
        evidence = ReviewEvidenceVisibilityService(db).project(item.id, user)
    except ReviewEvidenceNotFound:
        raise HTTPException(
            status_code=404, detail='Review item not found'
        ) from None
    return _review_item_response(
        item,
        _agent_run_for_item(db, item),
        evidence,
        auto_review_summary=_auto_review_summary(db, item),
        auto_review_audit=_auto_review_audit(db, item),
    )


_PUBLIC_AUTO_REVIEW_REASON_CODES = frozenset({
    'direct_fact_supported',
    'trusted_exact_reaffirmation',
})
_PUBLIC_AUTO_REVIEW_OUTCOMES = frozenset({
    'confirmed',
    'incorrect',
    'permission_violation',
    'source_version_violation',
    'policy_violation',
})


def _auto_review_summary(
    db: Session,
    item: ReviewItem,
) -> dict[str, object] | None:
    if (
        item.resolution_source != 'auto_policy'
        or item.auto_validation_id is None
    ):
        return None
    validation = db.get(AutoReviewValidation, item.auto_validation_id)
    if (
        validation is None
        or validation.review_item_id != item.id
        or validation.workflow_thread_id != item.workflow_thread_id
        or validation.status != 'completed'
        or validation.validator_model != AUTO_REVIEW_VALIDATOR_MODEL
        or validation.reasoning_effort != AUTO_REVIEW_REASONING_EFFORT
        or validation.validator_prompt_version
        != AUTO_REVIEW_VALIDATOR_PROMPT_VERSION
        or validation.validator_output_contract_version
        != AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION
        or validation.policy_version != AUTO_REVIEW_POLICY_VERSION
        or validation.minimum_entailment_score is None
        or validation.completed_at is None
    ):
        return None
    reason_codes = validation.policy_reason_codes
    if (
        not isinstance(reason_codes, list)
        or not 1 <= len(reason_codes) <= 2
        or len(set(reason_codes)) != len(reason_codes)
        or any(
            not isinstance(code, str)
            or code not in _PUBLIC_AUTO_REVIEW_REASON_CODES
            for code in reason_codes
        )
    ):
        return None
    claim_results = validation.claim_results
    if not isinstance(claim_results, list):
        return None
    supported_count = sum(
        1
        for claim in claim_results
        if isinstance(claim, dict) and claim.get('verdict') == 'supported'
    )
    if supported_count > 2:
        return None
    score = float(validation.minimum_entailment_score)
    if not 0.0 <= score <= 1.0:
        return None
    return {
        'validator_model': validation.validator_model,
        'reasoning_effort': validation.reasoning_effort,
        'validator_prompt_version': validation.validator_prompt_version,
        'validator_output_contract_version': (
            validation.validator_output_contract_version
        ),
        'policy_version': validation.policy_version,
        'supported_substantive_field_count': supported_count,
        'minimum_entailment_score': score,
        'policy_reason_codes': list(reason_codes),
        'validated_at': validation.completed_at,
    }


def _auto_review_audit(
    db: Session,
    item: ReviewItem,
) -> dict[str, object] | None:
    if item.resolution_source != 'auto_policy':
        return None
    audit = db.scalar(
        select(AutoReviewPostAudit).where(
            AutoReviewPostAudit.review_item_id == item.id
        )
    )
    if audit is None:
        return None
    correction = db.scalar(
        select(AutoReviewAuditCorrection).where(
            AutoReviewAuditCorrection.review_item_id == item.id
        )
    )
    if correction is not None:
        if (
            correction.status not in {'completed', 'remediation_required'}
            or correction.effective_outcome not in _PUBLIC_AUTO_REVIEW_OUTCOMES
            or correction.effective_outcome == 'confirmed'
        ):
            return None
        return {
            'status': correction.status,
            'outcome': correction.effective_outcome,
            'action_required': correction.status == 'remediation_required',
        }
    if (
        audit.status not in {'pending', 'completed', 'remediation_required'}
        or audit.outcome not in _PUBLIC_AUTO_REVIEW_OUTCOMES | {None}
        or (audit.status == 'pending' and audit.outcome is not None)
    ):
        return None
    return {
        'status': audit.status,
        'outcome': audit.outcome,
        'action_required': audit.status in {'pending', 'remediation_required'},
    }


def _auto_review_breaker_open(db: Session, review_item_id: int) -> bool:
    decision = db.scalar(
        select(AutoReviewPromotionDecision).where(
            AutoReviewPromotionDecision.review_item_id == review_item_id
        )
    )
    if decision is None:
        return False
    rollout = db.scalar(
        select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id
            == decision.security_scope_id,
            AutoReviewRolloutState.policy_version == decision.policy_version,
        )
    )
    return bool(rollout and rollout.breaker_open)


def _agent_run_id(item: ReviewItem) -> int | None:
    raw_id = item.payload.get('agent_run_id')
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str) and raw_id.isdecimal():
        return int(raw_id)
    return None


def _agent_run_for_item(db: Session, item: ReviewItem) -> AgentRun | None:
    agent_run_id = _agent_run_id(item)
    if agent_run_id is None:
        return None
    return db.get(AgentRun, agent_run_id)


def _agent_runs_by_id(db: Session, items: list[ReviewItem]) -> dict[int, AgentRun]:
    agent_run_ids = sorted({agent_run_id for item in items if (agent_run_id := _agent_run_id(item)) is not None})
    if not agent_run_ids:
        return {}
    runs = db.scalars(select(AgentRun).where(AgentRun.id.in_(agent_run_ids))).all()
    return {run.id: run for run in runs}


def _source_evidence_response(item: ReviewItem, agent_run: AgentRun | None) -> list[dict]:
    links = item.source_links or []
    snippets = item.source_snippets or []
    # 베이킹된 정보 로드
    source_authors = item.payload.get('source_authors', [])
    source_ids = item.payload.get('source_ids', [])
    source_types = item.payload.get('source_types', [])
    
    evidence_count = max(len(links), len(snippets))
    agent_run_id = _agent_run_id(item)
    summaries_by_url, summaries_by_id = _agent_evidence_summary_lookup(agent_run)
    rows: list[dict] = []

    for index in range(evidence_count):
        source_url = links[index] if index < len(links) else None
        payload_source_id = _indexed_string(source_ids, index)
        summary = summaries_by_id.get(payload_source_id) or summaries_by_url.get(source_url) or {}
        
        # ID 및 작성자 정보 폴백 로직
        source_id = (
            payload_source_id or summary.get('source_id')
        )
        author = (
            _indexed_string(source_authors, index) or summary.get('author') or "Unknown"
        )
        source_type = (
            summary.get('source_type')
            or _indexed_string(source_types, index)
            or item.payload.get('source_type')
            or _source_type_from_url(source_url)
        )
        
        if index < len(snippets):
            source_snippet = snippets[index]
        else:
            source_snippet = snippets[-1] if snippets else '원문 발췌 내용이 없습니다.'
            
        row = {
            'index': index + 1,
            'rank': index + 1,
            'source_id': source_id,
            'source_url': source_url,
            'source_type': source_type,
            'source_snippet': source_snippet,
            'permission_level': item.permission_level,
            'confidence_score': item.confidence_score,
            'importance_score': summary.get('importance_score', 0),
            'timestamp': summary.get('timestamp'),
            'author': author,
            'agent_run_id': agent_run_id,
            'parser_status': summary.get('parser_status'),
            'section_path': summary.get('section_path'),
            'evidence_reason': summary.get('evidence_reason'),
        }
        calendar_fields = {
            'calendar_id': summary.get('calendar_id') or item.payload.get('calendar_id'),
            'calendar_name': summary.get('calendar_summary') or item.payload.get('calendar_name'),
            'calendar_start': summary.get('event_start') or item.payload.get('calendar_start'),
            'calendar_end': summary.get('event_end') or item.payload.get('calendar_end'),
            'calendar_location': summary.get('location') or item.payload.get('calendar_location'),
            'calendar_organizer': summary.get('organizer_email') or item.payload.get('calendar_organizer'),
            'calendar_attendee_summary': _calendar_attendee_summary(summary, item.payload),
            'event_context_key': summary.get('event_context_key') or item.payload.get('event_context_key'),
        }
        row.update({key: value for key, value in calendar_fields.items() if value})
        rows.append(row)

    return rows


def _calendar_attendee_summary(summary: dict, payload: dict) -> str | None:
    value = summary.get('attendee_domains') or payload.get('calendar_attendee_summary')
    if isinstance(value, list):
        return ', '.join(str(item) for item in value if str(item).strip()) or None
    if isinstance(value, str):
        return value
    return None


def _indexed_string(value: object, index: int) -> str | None:
    if isinstance(value, list) and index < len(value):
        item = value[index]
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _source_type_from_url(url: str | None) -> str | None:
    if not url:
        return None
    lowered = url.lower()
    if 'mail.google.com' in lowered or 'gmail.' in lowered:
        return 'gmail'
    if 'drive.google.com' in lowered or 'drive.' in lowered:
        return 'drive'
    if 'calendar.google.com' in lowered or 'calendar.' in lowered:
        return 'calendar'
    if 'slack.com' in lowered or 'slack.' in lowered:
        return 'slack'
    return None


def _normalize_slack_url(url: str | None) -> str | None:
    """슬랙 URL에서 타임스탬프 부분을 추출하여 정규화합니다."""
    if not url or '/p' not in url:
        return None
    
    # p 뒤의 숫자만 추출
    ts_part = url.split('/p')[-1].split('?')[0]
    
    # 만약 16자리 숫자라면 (표준 규격), 이를 . 포맷으로 변환하여 매칭 확률을 극대화
    if len(ts_part) == 16 and ts_part.isdigit():
        return f"{ts_part[:10]}.{ts_part[10:]}".rstrip('0').rstrip('.')
        
    return ts_part.rstrip('0')


def _agent_evidence_summary_lookup(agent_run: AgentRun | None) -> tuple[dict[str, dict], dict[str, dict]]:
    if agent_run is None:
        return {}, {}
    raw_summary = (agent_run.metadata_ or {}).get('evidence_summary')
    if not isinstance(raw_summary, list):
        return {}, {}
    by_url: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    for row in raw_summary:
        if not isinstance(row, dict):
            continue
        source_url = row.get('source_url')
        if isinstance(source_url, str):
            by_url[source_url] = row
        source_id = row.get('source_id')
        if isinstance(source_id, str):
            by_id[source_id] = row
    return by_url, by_id


def _int_or_default(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return default
