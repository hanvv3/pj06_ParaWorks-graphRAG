from __future__ import annotations

from collections.abc import Mapping

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.review_v21_state import (
    REVIEW_STATUS_ORDER_V21,
    ReviewGraphInputV21,
    ReviewGraphOutputV21,
    ReviewGraphStateV21,
    ReviewRuntimeContextV21,
    ReviewRuntimeV21,
)
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
from backend.app.schemas.review_workflow import COMPANY_MEMORY_REVIEW_WORKFLOW


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(REVIEW_STATUS_ORDER_V21, 0)


def _load_thread(db: Session, thread_id: str) -> AgentWorkflowThread:
    thread = db.get(AgentWorkflowThread, thread_id)
    if thread is None:
        raise ValueError('not_found')
    if (
        thread.workflow_name != COMPANY_MEMORY_REVIEW_WORKFLOW
        or thread.graph_version != COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    ):
        raise ValueError('runtime_version_unavailable')
    return thread


def _owner_permissions(
    runtime: ReviewRuntimeV21, thread_id: str
) -> tuple[str, tuple[str, ...]]:
    with runtime.context['session_factory']() as db:
        owner = _load_thread(db, thread_id).owner_subject_id
        db.rollback()
    levels = tuple(sorted(
        set(runtime.context['permission_resolver'](owner))
        & {'public', 'internal'}
    ))
    if not levels:
        raise PermissionError('permission_denied')
    return owner, levels


def _validate_input(
    state: ReviewGraphInputV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    thread_id = state.get('workflow_thread_id')
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError('invalid_input')
    with runtime.context['session_factory']() as db:
        thread = _load_thread(db, thread_id)
        result = {
            'graph_version': thread.graph_version,
            'input_hash': thread.input_hash,
            'evidence_version_hash': thread.evidence_version_hash,
        }
        db.rollback()
    return {
        **result,
        'review_status_counts': _empty_counts(),
        'phase': 'input_validated',
        'completed_nodes': ['validate_input'],
        'error_codes': [],
    }


def _collect_evidence_refs(
    state: ReviewGraphStateV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    _, allowed = _owner_permissions(runtime, state['workflow_thread_id'])
    with runtime.context['session_factory']() as db:
        permissions = tuple(db.scalars(
            select(AgentWorkflowEvidenceRef.permission_level_snapshot)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == state['workflow_thread_id']
            )
            .order_by(AgentWorkflowEvidenceRef.ordinal)
        ).all())
        db.rollback()
    if not permissions or any(value not in allowed for value in permissions):
        raise PermissionError('permission_denied')
    return {
        'phase': 'evidence_refs_collected',
        'completed_nodes': ['collect_evidence_refs'],
    }


def _plan_agent_runs(
    state: ReviewGraphStateV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    with runtime.context['session_factory']() as db:
        request = db.get(AgentWorkflowRequest, state['workflow_thread_id'])
        if request is None or not request.agent_names:
            raise ValueError('invalid_input')
        names = tuple(request.agent_names)
        db.rollback()
    for name in names:
        try:
            runtime.context['agent_registry'].get(name)
        except KeyError:
            raise ValueError('invalid_input') from None
    return {
        'phase': 'agent_runs_planned',
        'completed_nodes': ['plan_agent_runs'],
    }


def _counts(runtime: ReviewRuntimeV21, thread_id: str) -> dict[str, int]:
    _, allowed = _owner_permissions(runtime, thread_id)
    with runtime.context['session_factory']() as db:
        rows = tuple(db.execute(
            select(ReviewItem.status, ReviewItem.permission_level)
            .where(ReviewItem.workflow_thread_id == thread_id)
            .order_by(ReviewItem.id)
        ).all())
        db.rollback()
    counts = _empty_counts()
    for status, permission in rows:
        if permission not in allowed:
            raise PermissionError('permission_denied')
        if status not in counts:
            raise ValueError('invalid_state_transition')
        counts[status] += 1
    return counts


def _draft(
    state: ReviewGraphStateV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    owner, allowed = _owner_permissions(runtime, state['workflow_thread_id'])
    runtime.context['extraction_coordinator'].draft(
        workflow_thread_id=state['workflow_thread_id'],
        actor_subject_id=owner,
        allowed_permission_levels=allowed,
    )
    return {
        'review_status_counts': _counts(runtime, state['workflow_thread_id']),
        'phase': 'candidates_drafted',
        'completed_nodes': ['draft_review_candidates_transaction'],
    }


def _run_auto_review(
    state: ReviewGraphStateV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    owner, allowed = _owner_permissions(runtime, state['workflow_thread_id'])
    runtime.context['validation_coordinator'].run_auto_review(
        workflow_thread_id=state['workflow_thread_id'],
        actor_subject_id=owner,
        allowed_permission_levels=allowed,
    )
    return {
        'phase': 'auto_review_run',
        'completed_nodes': ['run_auto_review'],
    }


def _refresh(
    state: ReviewGraphStateV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    return {
        'review_status_counts': _counts(runtime, state['workflow_thread_id']),
        'phase': 'review_resolution_refreshed',
        'completed_nodes': ['refresh_review_resolution'],
    }


def _route(_state: ReviewGraphStateV21) -> dict[str, object]:
    return {
        'phase': 'review_boundary_routed',
        'completed_nodes': ['route_review_boundary'],
    }


def _branch(state: ReviewGraphStateV21) -> str:
    counts = state['review_status_counts']
    total = sum(counts.values())
    if total == 0:
        return 'no_candidates'
    if counts['pending_review']:
        return 'human'
    if counts['needs_more_evidence']:
        return 'needs_more'
    if counts['approved'] + counts['rejected'] + counts['revoked'] == total:
        return 'resolved'
    raise ValueError('invalid_state_transition')


def _await(
    state: ReviewGraphStateV21, runtime: ReviewRuntimeV21
) -> dict[str, object]:
    with runtime.context['session_factory']() as db:
        version = _load_thread(db, state['workflow_thread_id']).state_version
        db.rollback()
    acknowledgement = interrupt({
        'event': 'review_resolution_required',
        'state_version': version,
    })
    if not isinstance(acknowledgement, Mapping) or (
        acknowledgement.get('event') != 'review_resolution_checked'
        or acknowledgement.get('state_version') != version
    ):
        raise ValueError('invalid_state_transition')
    return {
        'phase': 'review_resolution_acknowledged',
        'completed_nodes': ['await_human_review'],
    }


def _output(
    state: ReviewGraphStateV21, *, status: str, node: str
) -> dict[str, object]:
    return {
        'workflow_thread_id': state['workflow_thread_id'],
        'status': status,
        'review_item_count': sum(state['review_status_counts'].values()),
        'review_status_counts': dict(state['review_status_counts']),
        'error_codes': list(state['error_codes']),
        'phase': status,
        'completed_nodes': [node],
    }


def _finalize_no_candidates(state: ReviewGraphStateV21) -> dict[str, object]:
    return _output(state, status='completed', node='finalize_no_candidates')


def _finalize_needs_more(state: ReviewGraphStateV21) -> dict[str, object]:
    return _output(
        state,
        status='needs_more_evidence',
        node='finalize_needs_more_evidence',
    )


def _finalize_resolved(state: ReviewGraphStateV21) -> dict[str, object]:
    return _output(state, status='completed', node='finalize_auto_resolved')


def build_company_memory_review_v21_graph(
    saver: BaseCheckpointSaver,
) -> object:
    builder = StateGraph(
        ReviewGraphStateV21,
        input_schema=ReviewGraphInputV21,
        output_schema=ReviewGraphOutputV21,
        context_schema=ReviewRuntimeContextV21,
    )
    builder.add_node('validate_input', _validate_input)
    builder.add_node('collect_evidence_refs', _collect_evidence_refs)
    builder.add_node('plan_agent_runs', _plan_agent_runs)
    builder.add_node('draft_review_candidates_transaction', _draft)
    builder.add_node('run_auto_review', _run_auto_review)
    builder.add_node('refresh_review_resolution', _refresh)
    builder.add_node('route_review_boundary', _route)
    builder.add_node('await_human_review', _await)
    builder.add_node('finalize_no_candidates', _finalize_no_candidates)
    builder.add_node('finalize_needs_more_evidence', _finalize_needs_more)
    builder.add_node('finalize_auto_resolved', _finalize_resolved)
    builder.add_edge(START, 'validate_input')
    builder.add_edge('validate_input', 'collect_evidence_refs')
    builder.add_edge('collect_evidence_refs', 'plan_agent_runs')
    builder.add_edge('plan_agent_runs', 'draft_review_candidates_transaction')
    builder.add_edge('draft_review_candidates_transaction', 'run_auto_review')
    builder.add_edge('run_auto_review', 'refresh_review_resolution')
    builder.add_edge('refresh_review_resolution', 'route_review_boundary')
    builder.add_conditional_edges(
        'route_review_boundary',
        _branch,
        {
            'no_candidates': 'finalize_no_candidates',
            'human': 'await_human_review',
            'needs_more': 'finalize_needs_more_evidence',
            'resolved': 'finalize_auto_resolved',
        },
    )
    builder.add_edge('await_human_review', 'refresh_review_resolution')
    builder.add_edge('finalize_no_candidates', END)
    builder.add_edge('finalize_needs_more_evidence', END)
    builder.add_edge('finalize_auto_resolved', END)
    return builder.compile(checkpointer=saver)
