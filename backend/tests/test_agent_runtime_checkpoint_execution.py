import math
from typing import NotRequired

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Interrupt, interrupt
from typing_extensions import TypedDict

from backend.app.agent_runtime import (
    CheckpointConfirmationError,
    checkpoint_config,
    invoke_and_confirm_checkpoint,
    require_resumable_checkpoint,
)
from backend.app.agent_runtime.checkpointing import (
    CheckpointUnavailableError,
    build_strict_checkpoint_serializer,
)
from backend.app.agent_runtime.state import ReviewGraphState

FORBIDDEN_CHECKPOINT_KEYS = {
    'question',
    'objective',
    'source_url',
    'source_snippet',
    'message_body',
    'model_output',
    'provider_error',
    'api_key',
    'oauth_token',
}


class RuntimeTestContext(TypedDict):
    pause: bool
    event_log: NotRequired[list[str]]


class ObservedGraph:
    def __init__(self, graph: object, event_log: list[str]) -> None:
        self._graph = graph
        self._event_log = event_log
        self.invoke_calls: list[dict[str, object]] = []

    def invoke(
        self,
        command_or_input: object,
        config: dict[str, dict[str, str]],
        **kwargs: object,
    ) -> object:
        self.invoke_calls.append({
            'command_or_input': command_or_input,
            'config': config,
            **kwargs,
        })
        self._event_log.append('invoke_called')
        result = self._graph.invoke(command_or_input, config, **kwargs)
        self._event_log.append('invoke_returned')
        return result

    def get_state(self, config: dict[str, dict[str, str]]) -> object:
        self._event_log.append('get_state_called')
        return self._graph.get_state(config)


class MissingReturnedInterruptGraph(ObservedGraph):
    def invoke(
        self,
        command_or_input: object,
        config: dict[str, dict[str, str]],
        **kwargs: object,
    ) -> object:
        result = super().invoke(command_or_input, config, **kwargs)
        assert isinstance(result, dict)
        return {
            key: value
            for key, value in result.items()
            if key != '__interrupt__'
        }


class ObservedInMemorySaver(InMemorySaver):
    def __init__(self, event_log: list[str]) -> None:
        super().__init__(serde=build_strict_checkpoint_serializer())
        self._event_log = event_log

    def get_tuple(self, config: object) -> object:
        self._event_log.append('get_tuple_called')
        return super().get_tuple(config)  # type: ignore[arg-type]


_MISSING = object()


class SavedTupleConfigProxy:
    def __init__(
        self,
        saver: InMemorySaver,
        **configurable_overrides: object,
    ) -> None:
        self._saver = saver
        self._configurable_overrides = configurable_overrides

    def get_tuple(self, config: object) -> object:
        saved = self._saver.get_tuple(config)  # type: ignore[arg-type]
        if saved is None:
            return None
        configurable = dict(saved.config.get('configurable', {}))
        for key, value in self._configurable_overrides.items():
            if value is _MISSING:
                configurable.pop(key, None)
            else:
                configurable[key] = value
        return saved._replace(config={
            **saved.config,
            'configurable': configurable,
        })


class FrozenSavedTupleProxy:
    def __init__(self, saved: object) -> None:
        self._saved = saved

    def get_tuple(self, _config: object) -> object:
        return self._saved


class SnapshotConfigProxy:
    def __init__(
        self,
        graph: object,
        **configurable_overrides: object,
    ) -> None:
        self._graph = graph
        self._configurable_overrides = configurable_overrides

    def invoke(
        self,
        command_or_input: object,
        config: dict[str, dict[str, str]],
        **kwargs: object,
    ) -> object:
        return self._graph.invoke(command_or_input, config, **kwargs)

    def get_state(self, config: dict[str, dict[str, str]]) -> object:
        snapshot = self._graph.get_state(config)
        configurable = dict(snapshot.config.get('configurable', {}))
        configurable.update(self._configurable_overrides)
        return snapshot._replace(config={
            **snapshot.config,
            'configurable': configurable,
        })


class ExplodingSaver:
    def get_tuple(self, _config: object) -> object:
        raise RuntimeError('credential-marker')


class ExplodingSnapshotGraph:
    def __init__(self, graph: object) -> None:
        self._graph = graph

    def invoke(
        self,
        command_or_input: object,
        config: dict[str, dict[str, str]],
        **kwargs: object,
    ) -> object:
        return self._graph.invoke(command_or_input, config, **kwargs)

    def get_state(self, _config: dict[str, dict[str, str]]) -> object:
        raise RuntimeError('credential-marker')


def _checkpoint_state() -> ReviewGraphState:
    return {
        'workflow_thread_id': 'workflow-thread-1',
        'graph_version': 'company-memory-review-v2.0',
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_item_ids': [11, 12],
        'review_status_counts': {'pending_review': 2},
        'phase': 'checkpoint_pending',
        'completed_nodes': ['draft_candidates'],
        'error_codes': ['checkpoint_failed'],
    }


def _build_test_graph(saver: InMemorySaver) -> object:
    def await_review(
        _state: ReviewGraphState,
        runtime: Runtime[RuntimeTestContext],
    ) -> dict[str, object]:
        if runtime.context['pause']:
            interrupt({'kind': 'review_resolution'})
        return {
            'phase': 'awaiting_human_review',
            'completed_nodes': ['await_review'],
            'error_codes': ['checkpoint_failed'],
        }

    builder = StateGraph(
        ReviewGraphState,
        context_schema=RuntimeTestContext,
    )
    builder.add_node('await_review', await_review)
    builder.add_edge(START, 'await_review')
    builder.add_edge('await_review', END)
    return builder.compile(checkpointer=saver)


def _assert_json_safe_checkpoint_value(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        assert math.isfinite(value)
        return
    if type(value) is list:
        for item in value:
            _assert_json_safe_checkpoint_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            assert key not in FORBIDDEN_CHECKPOINT_KEYS
            _assert_json_safe_checkpoint_value(item)
        return
    if type(value) is Interrupt:
        assert type(value.id) is str
        _assert_json_safe_checkpoint_value(value.value)
        return
    if type(value) is tuple and value:
        assert all(type(item) is Interrupt for item in value)
        for item in value:
            assert type(item.id) is str
            _assert_json_safe_checkpoint_value(item.value)
        return
    pytest.fail(f'non-JSON-safe checkpoint value: {type(value).__name__}')


def test_checkpoint_config_uses_only_the_server_issued_root_thread_id() -> None:
    assert checkpoint_config('checkpoint-thread-1') == {
        'configurable': {'thread_id': 'checkpoint-thread-1'}
    }


@pytest.mark.parametrize('checkpoint_thread_id', ['', '   '])
def test_checkpoint_config_rejects_a_blank_thread_id(
    checkpoint_thread_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match='^checkpoint_thread_id is required$',
    ):
        checkpoint_config(checkpoint_thread_id)


def test_interrupt_confirmation_observes_sync_root_checkpoint_persistence() -> None:
    event_log: list[str] = []
    saver = ObservedInMemorySaver(event_log)
    graph = ObservedGraph(_build_test_graph(saver), event_log)

    confirmation = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True, 'event_log': event_log},
        expect_interrupt=True,
    )

    assert graph.invoke_calls == [
        {
            'command_or_input': _checkpoint_state(),
            'config': {'configurable': {'thread_id': 'checkpoint-thread-1'}},
            'context': {'pause': True, 'event_log': event_log},
            'durability': 'sync',
        }
    ]
    pre_read_at = event_log.index('get_tuple_called')
    invoke_called_at = event_log.index('invoke_called')
    invoke_returned_at = event_log.index('invoke_returned')
    confirmation_read_at = event_log.index(
        'get_tuple_called',
        invoke_returned_at + 1,
    )
    state_read_at = event_log.index('get_state_called', confirmation_read_at + 1)
    assert pre_read_at < invoke_called_at < invoke_returned_at
    assert invoke_returned_at < confirmation_read_at < state_read_at
    assert confirmation.checkpoint_thread_id == 'checkpoint-thread-1'
    assert confirmation.checkpoint_id
    assert confirmation.checkpoint_ns == ''
    assert confirmation.interrupted is True

    returned_interrupts = confirmation.result['__interrupt__']
    snapshot = graph.get_state(checkpoint_config('checkpoint-thread-1'))
    pending_interrupts = tuple(
        pending
        for task in snapshot.tasks
        for pending in task.interrupts
    )
    assert tuple(returned_interrupts) == pending_interrupts

    saved = saver.get_tuple(checkpoint_config('checkpoint-thread-1'))
    assert saved is not None
    saved_config = saved.config['configurable']
    assert saved_config['thread_id'] == 'checkpoint-thread-1'
    assert saved_config.get('checkpoint_ns', '') == ''


def test_confirmation_fails_when_the_supplied_saver_has_no_persisted_tuple() -> None:
    graph_saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(graph_saver)
    unrelated_saver = InMemorySaver(
        serde=build_strict_checkpoint_serializer()
    )

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint was not persisted$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=unrelated_saver,
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


def test_confirmation_fails_when_returned_and_pending_interrupts_disagree() -> None:
    event_log: list[str] = []
    saver = ObservedInMemorySaver(event_log)
    graph = MissingReturnedInterruptGraph(_build_test_graph(saver), event_log)

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint interrupt state mismatch$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=saver,
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


def test_confirmation_rejects_expected_interrupt_when_graph_is_terminal() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint interrupt state mismatch$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=saver,
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': False},
            expect_interrupt=True,
        )


def test_confirmation_rejects_a_stale_saver_for_the_same_thread() -> None:
    saver_a = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph_a = _build_test_graph(saver_a)
    invoke_and_confirm_checkpoint(
        graph=graph_a,
        saver=saver_a,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True},
        expect_interrupt=True,
    )
    saver_b = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph_b = _build_test_graph(saver_b)

    with pytest.raises(CheckpointConfirmationError):
        invoke_and_confirm_checkpoint(
            graph=graph_b,
            saver=saver_a,
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


def test_confirmation_rejects_a_checkpoint_that_did_not_advance() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True},
        expect_interrupt=True,
    )
    frozen = saver.get_tuple(checkpoint_config('checkpoint-thread-1'))
    assert frozen is not None

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint did not advance$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=FrozenSavedTupleProxy(frozen),  # type: ignore[arg-type]
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('thread_id', 'wrong-thread'),
        ('checkpoint_id', 'wrong-checkpoint'),
        ('checkpoint_ns', 'nested'),
    ],
)
def test_confirmation_rejects_snapshot_identity_mismatch(
    field: str,
    value: str,
) -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    proxy = SnapshotConfigProxy(graph, **{field: value})

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint snapshot identity mismatch$',
    ):
        invoke_and_confirm_checkpoint(
            graph=proxy,
            saver=saver,
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


def test_resume_confirmation_requires_checkpoint_progression() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    paused = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True},
        expect_interrupt=True,
    )

    resumed = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=Command(resume='approved'),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True},
        expect_interrupt=False,
    )

    assert resumed.checkpoint_id != paused.checkpoint_id


def test_confirmation_rejects_a_wrong_saved_thread_id() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    proxy = SavedTupleConfigProxy(saver, thread_id='wrong-thread')

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint identity mismatch$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=proxy,  # type: ignore[arg-type]
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


@pytest.mark.parametrize('checkpoint_id', [_MISSING, '', False])
def test_confirmation_rejects_a_missing_or_false_saved_checkpoint_id(
    checkpoint_id: object,
) -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    proxy = SavedTupleConfigProxy(saver, checkpoint_id=checkpoint_id)

    with pytest.raises(
        CheckpointConfirmationError,
        match='^checkpoint identity mismatch$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=proxy,  # type: ignore[arg-type]
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


def test_confirmation_rejects_a_non_root_saved_namespace() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    proxy = SavedTupleConfigProxy(saver, checkpoint_ns='nested')

    with pytest.raises(
        CheckpointConfirmationError,
        match='^top-level checkpoint namespace must be root$',
    ):
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=proxy,  # type: ignore[arg-type]
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )


def test_confirmation_sanitizes_saver_read_failures() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)

    with pytest.raises(CheckpointConfirmationError) as exc_info:
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=ExplodingSaver(),  # type: ignore[arg-type]
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )

    assert str(exc_info.value) == 'checkpoint confirmation failed'
    assert 'credential-marker' not in str(exc_info.value)


def test_confirmation_sanitizes_snapshot_read_failures() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = ExplodingSnapshotGraph(_build_test_graph(saver))

    with pytest.raises(CheckpointConfirmationError) as exc_info:
        invoke_and_confirm_checkpoint(
            graph=graph,
            saver=saver,
            command_or_input=_checkpoint_state(),
            checkpoint_thread_id='checkpoint-thread-1',
            runtime_context={'pause': True},
            expect_interrupt=True,
        )

    assert str(exc_info.value) == 'checkpoint confirmation failed'
    assert 'credential-marker' not in str(exc_info.value)


def test_restart_with_a_new_memory_saver_reports_checkpoint_unavailable() -> None:
    saver_a = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver_a)
    invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver_a,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True},
        expect_interrupt=True,
    )
    saver_b = InMemorySaver(serde=build_strict_checkpoint_serializer())

    with pytest.raises(
        CheckpointUnavailableError,
        match='^checkpoint_unavailable$',
    ):
        require_resumable_checkpoint(saver_b, 'checkpoint-thread-1')


def test_checkpoint_tuple_contains_only_safe_opaque_state() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)
    invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': True},
        expect_interrupt=True,
    )

    saved = saver.get_tuple(checkpoint_config('checkpoint-thread-1'))
    assert saved is not None
    _assert_json_safe_checkpoint_value(saved.checkpoint)
    for task_id, channel, value in saved.pending_writes:
        assert type(task_id) is str
        assert type(channel) is str
        assert channel not in FORBIDDEN_CHECKPOINT_KEYS
        _assert_json_safe_checkpoint_value(value)


def test_replayed_graph_updates_do_not_duplicate_reducer_entries() -> None:
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = _build_test_graph(saver)

    first = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': False},
        expect_interrupt=False,
    )
    second = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=_checkpoint_state(),
        checkpoint_thread_id='checkpoint-thread-1',
        runtime_context={'pause': False},
        expect_interrupt=False,
    )

    assert first.result['completed_nodes'] == [
        'draft_candidates',
        'await_review',
    ]
    assert first.result['error_codes'] == ['checkpoint_failed']
    assert second.result['completed_nodes'] == [
        'draft_candidates',
        'await_review',
    ]
    assert second.result['error_codes'] == ['checkpoint_failed']
    assert second.checkpoint_id != first.checkpoint_id
