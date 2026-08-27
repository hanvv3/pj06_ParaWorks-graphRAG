from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.state import (
    REVIEW_STATUS_ORDER,
    ReviewGraphInput,
    ReviewGraphOutput,
    ReviewGraphState,
    ReviewRuntime,
    ReviewRuntimeContext,
)
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    ReviewItemResolutionStatus,
)

_NO_CANDIDATES = 'no_candidates'
_CANDIDATES_CREATED = 'candidates_created'
_UNRESOLVED = 'unresolved'
_NEEDS_MORE_EVIDENCE = 'needs_more_evidence'
_APPROVED_OR_REJECTED = 'approved_or_rejected'


def _empty_review_status_counts() -> dict[str, int]:
    return dict.fromkeys(REVIEW_STATUS_ORDER, 0)


def _load_thread(db: Session, workflow_thread_id: str) -> AgentWorkflowThread:
    thread = db.get(AgentWorkflowThread, workflow_thread_id)
    if thread is None:
        raise ValueError('not_found')
    if (
        thread.workflow_name != COMPANY_MEMORY_REVIEW_WORKFLOW
        or thread.graph_version != COMPANY_MEMORY_REVIEW_GRAPH_VERSION
    ):
        raise ValueError('runtime_version_unavailable')
    return thread


def _current_permissions(runtime: ReviewRuntime) -> tuple[str, ...]:
    context = runtime.context
    permissions = tuple(sorted({
        value.strip()
        for value in context['permission_resolver'](
            context['actor_subject_id']
        )
        if value.strip()
    }))
    if not permissions:
        raise PermissionError('permission_denied')
    return permissions


def _validate_input(
    state: ReviewGraphInput,
    runtime: ReviewRuntime,
) -> dict[str, object]:
    workflow_thread_id = state.get('workflow_thread_id')
    if not isinstance(workflow_thread_id, str) or not workflow_thread_id.strip():
        raise ValueError('invalid_input')
    with runtime.context['session_factory']() as db:
        thread = _load_thread(db, workflow_thread_id)
        values = {
            'graph_version': thread.graph_version,
            'input_hash': thread.input_hash,
            'evidence_version_hash': thread.evidence_version_hash,
        }
        db.rollback()
    return {
        **values,
        'review_status_counts': _empty_review_status_counts(),
        'phase': 'input_validated',
        'completed_nodes': ['validate_input'],
        'error_codes': [],
    }


def _collect_evidence_refs(
    state: ReviewGraphState,
    runtime: ReviewRuntime,
) -> dict[str, object]:
    allowed_permissions = set(_current_permissions(runtime))
    with runtime.context['session_factory']() as db:
        permission_snapshots = tuple(
            db.scalars(
                select(AgentWorkflowEvidenceRef.permission_level_snapshot)
                .where(
                    AgentWorkflowEvidenceRef.workflow_thread_id
                    == state['workflow_thread_id']
                )
                .order_by(AgentWorkflowEvidenceRef.ordinal)
            ).all()
        )
        db.rollback()
    if not permission_snapshots:
        raise ValueError('invalid_input')
    if any(
        permission not in allowed_permissions
        for permission in permission_snapshots
    ):
        raise PermissionError('permission_denied')
    return {
        'phase': 'evidence_refs_collected',
        'completed_nodes': ['collect_evidence_refs'],
    }


def _plan_agent_runs(
    state: ReviewGraphState,
    runtime: ReviewRuntime,
) -> dict[str, object]:
    with runtime.context['session_factory']() as db:
        request = db.get(AgentWorkflowRequest, state['workflow_thread_id'])
        if request is None or not request.agent_names:
            raise ValueError('invalid_input')
        agent_names = tuple(request.agent_names)
        db.rollback()
    try:
        for agent_name in agent_names:
            runtime.context['agent_registry'].get(agent_name)
    except KeyError:
        raise ValueError('invalid_input') from None
    return {
        'phase': 'agent_runs_planned',
        'completed_nodes': ['plan_agent_runs'],
    }


def _draft_review_candidates_transaction(
    state: ReviewGraphState,
    runtime: ReviewRuntime,
) -> dict[str, object]:
    context = runtime.context
    result = context['draft_service'].draft(
        workflow_thread_id=state['workflow_thread_id'],
        actor_subject_id=context['actor_subject_id'],
        allowed_permission_levels=_current_permissions(runtime),
    )
    counts = {
        status: result.review_status_counts.get(
            cast(ReviewItemResolutionStatus, status),
            0,
        )
        for status in REVIEW_STATUS_ORDER
    }
    if len(result.review_item_ids) != sum(counts.values()):
        raise ValueError('invalid_state_transition')
    return {
        'review_status_counts': counts,
        'phase': 'checkpoint_pending',
        'completed_nodes': ['draft_review_candidates_transaction'],
    }


def _route_review_boundary(
    _state: ReviewGraphState,
) -> dict[str, object]:
    return {
        'phase': 'review_boundary_routed',
        'completed_nodes': ['route_review_boundary'],
    }


def _review_boundary_branch(state: ReviewGraphState) -> str:
    if sum(state['review_status_counts'].values()) == 0:
        return _NO_CANDIDATES
    return _CANDIDATES_CREATED


def _output(
    state: ReviewGraphState,
    *,
    status: str,
    completed_node: str,
) -> dict[str, object]:
    return {
        'workflow_thread_id': state['workflow_thread_id'],
        'status': status,
        'review_item_count': sum(state['review_status_counts'].values()),
        'review_status_counts': dict(state['review_status_counts']),
        'error_codes': list(state['error_codes']),
        'phase': status,
        'completed_nodes': [completed_node],
    }


def _validate_review_resolution_acknowledgement(
    acknowledgement: object,
    *,
    state_version: int,
) -> None:
    if not isinstance(acknowledgement, Mapping):
        raise ValueError('invalid_state_transition')
    acknowledgement_state_version = acknowledgement.get('state_version')
    if (
        acknowledgement.get('event') != 'review_resolution_checked'
        or type(acknowledgement_state_version) is not int
        or acknowledgement_state_version != state_version
    ):
        raise ValueError('invalid_state_transition')


def _finalize_no_candidates(
    state: ReviewGraphState,
) -> dict[str, object]:
    return _output(
        state,
        status='completed',
        completed_node='finalize_no_candidates',
    )


def _await_human_review(
    state: ReviewGraphState,
    runtime: ReviewRuntime,
) -> dict[str, object]:
    with runtime.context['session_factory']() as db:
        state_version = _load_thread(
            db,
            state['workflow_thread_id'],
        ).state_version
        db.rollback()
    acknowledgement = interrupt({
        'event': 'review_resolution_required',
        'state_version': state_version,
    })
    _validate_review_resolution_acknowledgement(
        acknowledgement,
        state_version=state_version,
    )
    return {
        'phase': 'review_resolution_acknowledged',
        'completed_nodes': ['await_human_review'],
    }


def _verify_review_resolution_from_postgres(
    state: ReviewGraphState,
    runtime: ReviewRuntime,
) -> dict[str, object]:
    allowed_permissions = set(_current_permissions(runtime))
    with runtime.context['session_factory']() as db:
        _load_thread(db, state['workflow_thread_id'])
        review_items = tuple(
            db.execute(
                select(ReviewItem.status, ReviewItem.permission_level)
                .where(
                    ReviewItem.workflow_thread_id
                    == state['workflow_thread_id']
                )
                .order_by(ReviewItem.id)
            ).all()
        )
        db.rollback()
    if not review_items:
        raise ValueError('invalid_state_transition')
    if any(
        permission not in allowed_permissions
        for _, permission in review_items
    ):
        raise PermissionError('permission_denied')
    counts = _empty_review_status_counts()
    for status, _ in review_items:
        if status not in counts:
            raise ValueError('invalid_state_transition')
        counts[status] += 1
    return {
        'review_status_counts': counts,
        'phase': 'review_resolution_verified',
        'completed_nodes': ['verify_review_resolution_from_postgres'],
    }


def _review_resolution_branch(state: ReviewGraphState) -> str:
    counts = state['review_status_counts']
    if counts['needs_more_evidence']:
        return _NEEDS_MORE_EVIDENCE
    if counts['pending_review']:
        return _UNRESOLVED
    if counts['approved'] + counts['rejected'] == sum(counts.values()):
        return _APPROVED_OR_REJECTED
    raise ValueError('invalid_state_transition')


def _finalize_needs_more_evidence(
    state: ReviewGraphState,
) -> dict[str, object]:
    return _output(
        state,
        status='needs_more_evidence',
        completed_node='finalize_needs_more_evidence',
    )


def _finalize_review_trace(
    state: ReviewGraphState,
) -> dict[str, object]:
    return _output(
        state,
        status='completed',
        completed_node='finalize_review_trace',
    )


def build_company_memory_review_v2_graph(
    saver: BaseCheckpointSaver,
) -> object:
    builder = StateGraph(
        ReviewGraphState,
        input_schema=ReviewGraphInput,
        output_schema=ReviewGraphOutput,
        context_schema=ReviewRuntimeContext,
    )
    builder.add_node('validate_input', _validate_input)
    builder.add_node('collect_evidence_refs', _collect_evidence_refs)
    builder.add_node('plan_agent_runs', _plan_agent_runs)
    builder.add_node(
        'draft_review_candidates_transaction',
        _draft_review_candidates_transaction,
    )
    builder.add_node('route_review_boundary', _route_review_boundary)
    builder.add_node('finalize_no_candidates', _finalize_no_candidates)
    builder.add_node('await_human_review', _await_human_review)
    builder.add_node(
        'verify_review_resolution_from_postgres',
        _verify_review_resolution_from_postgres,
    )
    builder.add_node(
        'finalize_needs_more_evidence',
        _finalize_needs_more_evidence,
    )
    builder.add_node('finalize_review_trace', _finalize_review_trace)

    builder.add_edge(START, 'validate_input')
    builder.add_edge('validate_input', 'collect_evidence_refs')
    builder.add_edge('collect_evidence_refs', 'plan_agent_runs')
    builder.add_edge('plan_agent_runs', 'draft_review_candidates_transaction')
    builder.add_edge(
        'draft_review_candidates_transaction',
        'route_review_boundary',
    )
    builder.add_conditional_edges(
        'route_review_boundary',
        _review_boundary_branch,
        {
            _NO_CANDIDATES: 'finalize_no_candidates',
            _CANDIDATES_CREATED: 'await_human_review',
        },
    )
    builder.add_edge('finalize_no_candidates', END)
    builder.add_edge(
        'await_human_review',
        'verify_review_resolution_from_postgres',
    )
    builder.add_conditional_edges(
        'verify_review_resolution_from_postgres',
        _review_resolution_branch,
        {
            _UNRESOLVED: 'await_human_review',
            _NEEDS_MORE_EVIDENCE: 'finalize_needs_more_evidence',
            _APPROVED_OR_REJECTED: 'finalize_review_trace',
        },
    )
    builder.add_edge('finalize_needs_more_evidence', END)
    builder.add_edge('finalize_review_trace', END)
    return builder.compile(checkpointer=saver)
