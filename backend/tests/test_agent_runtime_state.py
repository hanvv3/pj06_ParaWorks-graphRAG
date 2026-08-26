import pytest

from backend.app.agent_runtime.state import (
    merge_completed_nodes,
    merge_error_codes,
    validate_checkpoint_state,
)


def test_completed_nodes_are_append_unique_and_bounded() -> None:
    current = [f'node-{index}' for index in range(64)]
    assert merge_completed_nodes(current, ['node-1', 'node-64']) == [
        *current[1:],
        'node-64',
    ]


def test_error_codes_reject_unknown_values_and_stay_bounded() -> None:
    with pytest.raises(ValueError, match='unsupported checkpoint error code'):
        merge_error_codes([], ['raw-provider-exception'])

    merged = merge_error_codes(
        ['invalid_input'] * 2,
        ['checkpoint_failed', 'checkpoint_failed'],
    )
    assert merged == ['invalid_input', 'checkpoint_failed']


@pytest.mark.parametrize(
    'unsafe_key,unsafe_value',
    [
        ('question', 'secret question'),
        ('source_url', 'https://restricted.example'),
        ('model_output', {'answer': 'secret'}),
        ('session', object()),
    ],
)
def test_checkpoint_state_rejects_sensitive_or_non_json_values(
    unsafe_key: str,
    unsafe_value: object,
) -> None:
    state = {
        'workflow_thread_id': 'thread-1',
        'graph_version': 'company-memory-review-v2.0',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_item_ids': [],
        'review_status_counts': {},
        'phase': 'created',
        'completed_nodes': [],
        'error_codes': [],
        unsafe_key: unsafe_value,
    }

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)
