import pytest

from backend.app.agent_runtime.state import (
    merge_completed_nodes,
    merge_error_codes,
    validate_checkpoint_state,
)


def _valid_checkpoint_state() -> dict[str, object]:
    return {
        'workflow_thread_id': 'thread-1',
        'graph_version': 'company-memory-review-v2.0',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_status_counts': {
            'pending_review': 2,
            'approved': 0,
            'rejected': 0,
            'needs_more_evidence': 0,
        },
        'phase': 'created',
        'completed_nodes': ['validate_input'],
        'error_codes': [],
    }


def test_checkpoint_state_accepts_the_declared_state_contract() -> None:
    validate_checkpoint_state(_valid_checkpoint_state())


def test_v20_checkpoint_keeps_eight_keys_and_four_statuses() -> None:
    state = _valid_checkpoint_state()

    assert set(state) == {
        'workflow_thread_id', 'graph_version', 'input_hash',
        'evidence_version_hash', 'review_status_counts', 'phase',
        'completed_nodes', 'error_codes',
    }
    assert set(state['review_status_counts']) == {
        'pending_review', 'approved', 'rejected', 'needs_more_evidence'
    }


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
        ('review_item_ids', []),
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
        'review_status_counts': {},
        'phase': 'created',
        'completed_nodes': [],
        'error_codes': [],
        unsafe_key: unsafe_value,
    }

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)


@pytest.mark.parametrize(
    'field,value',
    [
        ('workflow_thread_id', 1),
        ('graph_version', False),
        ('input_hash', 'not-a-keyed-hmac'),
        ('evidence_version_hash', 'g' * 64),
        ('review_status_counts', {'pending_review': True}),
        ('review_status_counts', {'pending_review': -1}),
        ('review_status_counts', {'unreviewed': 1}),
        ('phase', 1),
        ('completed_nodes', ['validate_input', 1]),
        ('error_codes', ['raw-provider-exception']),
    ],
)
def test_checkpoint_state_rejects_malformed_declared_fields(
    field: str,
    value: object,
) -> None:
    state = _valid_checkpoint_state()
    state[field] = value

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)


@pytest.mark.parametrize(
    'field,value',
    [
        ('completed_nodes', [f'node-{index}' for index in range(65)]),
        ('completed_nodes', ['validate_input', 'validate_input']),
        ('error_codes', ['invalid_input'] * 17),
        ('error_codes', ['invalid_input', 'invalid_input']),
    ],
)
def test_checkpoint_state_rejects_broken_reducer_invariants(
    field: str,
    value: object,
) -> None:
    state = _valid_checkpoint_state()
    state[field] = value

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)


def test_checkpoint_state_rejects_recursive_values_with_value_error() -> None:
    recursive_counts: dict[str, object] = {}
    recursive_counts['pending_review'] = recursive_counts
    state = _valid_checkpoint_state()
    state['review_status_counts'] = recursive_counts

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)


def test_checkpoint_state_rejects_excessive_nesting_with_value_error() -> None:
    nested: object = 1
    for _ in range(2_000):
        nested = [nested]
    state = _valid_checkpoint_state()
    state['review_status_counts'] = nested

    with pytest.raises(ValueError):
        validate_checkpoint_state(state)
