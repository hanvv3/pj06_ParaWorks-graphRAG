import math
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Protocol

from langgraph.runtime import Runtime
from typing_extensions import TypedDict

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
    'concurrent_resume',
    'invalid_state_transition',
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


def validate_checkpoint_state(state: Mapping[str, object]) -> None:
    if set(state) != _CHECKPOINT_STATE_KEYS:
        raise ValueError('checkpoint state keys do not match ReviewGraphState')
    _validate_json_value(dict(state))


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


class ReviewRuntimeContext(TypedDict):
    session_factory: Callable[[], object]
    actor_subject_id: str
    permission_resolver: PermissionResolver
    agent_registry: object
    draft_service: object
    lease_service: object


ReviewRuntime = Runtime[ReviewRuntimeContext]
