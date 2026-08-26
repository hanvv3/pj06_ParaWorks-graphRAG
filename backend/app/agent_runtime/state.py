from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Protocol

from langgraph.runtime import Runtime
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from backend.app.agent_runtime.registry import AgentRegistry
    from backend.app.agent_runtime.review_v2_drafting import ReviewDraftResult

MAX_COMPLETED_NODES = 64
MAX_ERROR_CODES = 16
ALLOWED_REVIEW_ERROR_CODES = frozenset({
    'invalid_input',
    'idempotency_key_reused',
    'evidence_changed',
    'permission_denied',
    'checkpoint_unavailable',
    'checkpoint_failed',
    'review_unresolved',
    'runtime_version_unavailable',
    'model_unavailable',
    'budget_exceeded',
    'concurrent_resume',
    'invalid_state_transition',
})
ALLOWED_REVIEW_STATUSES = frozenset({
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
})

_CHECKPOINT_STATE_KEYS = frozenset({
    'workflow_thread_id',
    'graph_version',
    'input_hash',
    'evidence_version_hash',
    'review_item_ids',
    'review_status_counts',
    'phase',
    'completed_nodes',
    'error_codes',
})


def _append_unique_bounded(
    current: Sequence[str],
    updates: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    merged: list[str] = []
    for value in (*current, *updates):
        if value not in merged:
            merged.append(value)
    return merged[-limit:]


def merge_completed_nodes(
    current: Sequence[str],
    updates: Sequence[str],
) -> list[str]:
    return _append_unique_bounded(current, updates, limit=MAX_COMPLETED_NODES)


def merge_error_codes(
    current: Sequence[str],
    updates: Sequence[str],
) -> list[str]:
    values = (*current, *updates)
    for value in values:
        if value not in ALLOWED_REVIEW_ERROR_CODES:
            raise ValueError(f'unsupported checkpoint error code: {value}')
    return _append_unique_bounded(current, updates, limit=MAX_ERROR_CODES)


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('checkpoint state contains a non-finite float')
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('checkpoint state keys must be strings')
            _validate_json_value(item)
        return
    raise ValueError('checkpoint state must contain JSON-safe primitives')


def _validate_string_field(state: Mapping[str, object], field: str) -> None:
    if type(state[field]) is not str:
        raise ValueError(f'checkpoint state field {field} must be a string')


def _validate_hash_field(state: Mapping[str, object], field: str) -> None:
    value = state[field]
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise ValueError(f'checkpoint state field {field} must be a keyed HMAC')


def _validate_review_item_ids(value: object) -> None:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ValueError('checkpoint review_item_ids must be a list of integers')


def _validate_review_status_counts(value: object) -> None:
    if type(value) is not dict:
        raise ValueError('checkpoint review_status_counts must be a dictionary')
    for status, count in value.items():
        if type(status) is not str or status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError('checkpoint review status is unsupported')
        if type(count) is not int or count < 0:
            raise ValueError('checkpoint review status count must be non-negative')


def _validate_completed_nodes(value: object) -> None:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError('checkpoint completed_nodes must be a list of strings')
    if len(value) > MAX_COMPLETED_NODES or len(set(value)) != len(value):
        raise ValueError('checkpoint completed_nodes violates reducer bounds')


def _validate_error_codes(value: object) -> None:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError('checkpoint error_codes must be a list of strings')
    if len(value) > MAX_ERROR_CODES or len(set(value)) != len(value):
        raise ValueError('checkpoint error_codes violates reducer bounds')
    for value_item in value:
        if value_item not in ALLOWED_REVIEW_ERROR_CODES:
            raise ValueError(f'unsupported checkpoint error code: {value_item}')


def validate_checkpoint_state(state: Mapping[str, object]) -> None:
    if set(state) != _CHECKPOINT_STATE_KEYS:
        raise ValueError('checkpoint state keys do not match ReviewGraphState')
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
    _validate_review_item_ids(state['review_item_ids'])
    _validate_review_status_counts(state['review_status_counts'])
    _validate_string_field(state, 'phase')
    _validate_completed_nodes(state['completed_nodes'])
    _validate_error_codes(state['error_codes'])


class ReviewGraphInput(TypedDict):
    workflow_thread_id: str


class ReviewGraphState(TypedDict):
    workflow_thread_id: str
    graph_version: str
    input_hash: str
    evidence_version_hash: str
    review_item_ids: list[int]
    review_status_counts: dict[str, int]
    phase: str
    completed_nodes: Annotated[list[str], merge_completed_nodes]
    error_codes: Annotated[list[str], merge_error_codes]


class ReviewGraphOutput(TypedDict):
    workflow_thread_id: str
    status: str
    review_item_count: int
    review_status_counts: dict[str, int]
    error_codes: list[str]


class PermissionResolver(Protocol):
    def __call__(self, actor_subject_id: str) -> Sequence[str]:
        raise NotImplementedError


class ReviewDraftService(Protocol):
    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewDraftResult:
        raise NotImplementedError


class ReviewLeaseService(Protocol):
    def acquire(self, *, workflow_thread_id: str) -> str:
        raise NotImplementedError


class ReviewRuntimeContext(TypedDict):
    session_factory: Callable[[], Session]
    actor_subject_id: str
    permission_resolver: PermissionResolver
    agent_registry: AgentRegistry
    draft_service: ReviewDraftService
    lease_service: ReviewLeaseService


ReviewRuntime = Runtime[ReviewRuntimeContext]
