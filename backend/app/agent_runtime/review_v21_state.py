from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Protocol

from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from backend.app.agent_runtime.state import (
    MAX_COMPLETED_NODES,
    MAX_ERROR_CODES,
    _append_unique_bounded,
    _validate_completed_nodes,
    _validate_error_codes,
    _validate_hash_field,
    _validate_json_value,
    _validate_string_field,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from backend.app.agent_runtime.registry import AgentRegistry

REVIEW_STATUS_ORDER_V21 = (
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
    'revoked',
)
ALLOWED_REVIEW_STATUSES_V21 = frozenset(REVIEW_STATUS_ORDER_V21)
_CHECKPOINT_STATE_KEYS_V21 = frozenset({
    'workflow_thread_id',
    'graph_version',
    'input_hash',
    'evidence_version_hash',
    'review_status_counts',
    'phase',
    'completed_nodes',
    'error_codes',
})


def merge_completed_nodes_v21(
    current: Sequence[str], updates: Sequence[str]
) -> list[str]:
    return _append_unique_bounded(
        current, updates, limit=MAX_COMPLETED_NODES
    )


def merge_error_codes_v21(
    current: Sequence[str], updates: Sequence[str]
) -> list[str]:
    _validate_error_codes(list(updates))
    return _append_unique_bounded(current, updates, limit=MAX_ERROR_CODES)


def _validate_review_status_counts_v21(value: object) -> None:
    if type(value) is not dict:
        raise ValueError('checkpoint review_status_counts must be a dictionary')
    if set(value) != ALLOWED_REVIEW_STATUSES_V21:
        raise ValueError('checkpoint review status keys are incomplete')
    for status, count in value.items():
        if type(status) is not str or status not in ALLOWED_REVIEW_STATUSES_V21:
            raise ValueError('checkpoint review status is unsupported')
        if type(count) is not int or count < 0:
            raise ValueError('checkpoint review status count must be non-negative')


def validate_v21_checkpoint_state(state: Mapping[str, object]) -> None:
    if set(state) != _CHECKPOINT_STATE_KEYS_V21:
        raise ValueError('checkpoint state keys do not match ReviewGraphStateV21')
    try:
        _validate_json_value(dict(state))
    except RecursionError:
        raise ValueError(
            'checkpoint state nesting is too deep or cyclic'
        ) from None
    _validate_string_field(state, 'workflow_thread_id')
    _validate_string_field(state, 'graph_version')
    _validate_hash_field(state, 'input_hash')
    _validate_hash_field(state, 'evidence_version_hash')
    _validate_review_status_counts_v21(state['review_status_counts'])
    _validate_string_field(state, 'phase')
    _validate_completed_nodes(state['completed_nodes'])
    _validate_error_codes(state['error_codes'])


class ReviewGraphInputV21(TypedDict):
    workflow_thread_id: str


class ReviewGraphStateV21(TypedDict):
    workflow_thread_id: str
    graph_version: str
    input_hash: str
    evidence_version_hash: str
    review_status_counts: dict[str, int]
    phase: str
    completed_nodes: Annotated[list[str], merge_completed_nodes_v21]
    error_codes: Annotated[list[str], merge_error_codes_v21]


class ReviewGraphOutputV21(TypedDict):
    workflow_thread_id: str
    status: str
    review_item_count: int
    review_status_counts: dict[str, int]
    error_codes: list[str]


class PermissionResolverV21(Protocol):
    def __call__(self, owner_subject_id: str) -> Sequence[str]: ...


class V21ExtractionCoordinator(Protocol):
    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> object: ...


class V21ValidationCoordinator(Protocol):
    def run_auto_review(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> object: ...


class ReviewRuntimeContextV21(TypedDict):
    session_factory: Callable[[], Session]
    permission_resolver: PermissionResolverV21
    agent_registry: AgentRegistry
    extraction_coordinator: V21ExtractionCoordinator
    validation_coordinator: V21ValidationCoordinator


ReviewRuntimeV21 = Runtime[ReviewRuntimeContextV21]
