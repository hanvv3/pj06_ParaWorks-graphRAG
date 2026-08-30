import pytest

from backend.app.agent_runtime.review_v21_state import (
    REVIEW_STATUS_ORDER_V21,
    validate_v21_checkpoint_state,
)


def _state() -> dict[str, object]:
    return {
        'workflow_thread_id': 'thread-v21',
        'graph_version': 'company-memory-review-v2.1-auto-review',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_status_counts': dict.fromkeys(REVIEW_STATUS_ORDER_V21, 0),
        'phase': 'input_validated',
        'completed_nodes': ['validate_input'],
        'error_codes': [],
    }


def test_v21_checkpoint_has_exact_eight_safe_keys_and_five_counts() -> None:
    state = _state()

    validate_v21_checkpoint_state(state)

    assert set(state) == {
        'workflow_thread_id',
        'graph_version',
        'input_hash',
        'evidence_version_hash',
        'review_status_counts',
        'phase',
        'completed_nodes',
        'error_codes',
    }
    assert tuple(state['review_status_counts']) == REVIEW_STATUS_ORDER_V21
    assert REVIEW_STATUS_ORDER_V21 == (
        'pending_review',
        'approved',
        'rejected',
        'needs_more_evidence',
        'revoked',
    )


@pytest.mark.parametrize(
    'forbidden_key',
    ('review_item_ids', 'validation_output', 'source_snippets', 'model_output'),
)
def test_v21_checkpoint_rejects_sensitive_or_unbounded_fields(
    forbidden_key: str,
) -> None:
    state = _state()
    state[forbidden_key] = ['secret']

    with pytest.raises(ValueError, match='checkpoint state keys'):
        validate_v21_checkpoint_state(state)


def test_v21_checkpoint_rejects_the_v20_four_status_shape() -> None:
    state = _state()
    state['review_status_counts'] = {
        'pending_review': 0,
        'approved': 0,
        'rejected': 0,
        'needs_more_evidence': 0,
    }

    with pytest.raises(ValueError, match='status keys'):
        validate_v21_checkpoint_state(state)
