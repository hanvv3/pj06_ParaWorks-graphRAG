import math
from collections.abc import Mapping
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.types import Interrupt

from backend.app.agent_runtime.checkpointing import CheckpointUnavailableError


class CheckpointConfirmationError(RuntimeError):
    pass


_CHECKPOINT_CONFIRMATION_FAILED = 'checkpoint confirmation failed'
_CHECKPOINT_IDENTITY_MISMATCH = 'checkpoint identity mismatch'
_CHECKPOINT_SNAPSHOT_IDENTITY_MISMATCH = (
    'checkpoint snapshot identity mismatch'
)


@dataclass(frozen=True)
class CheckpointConfirmation:
    result: dict[str, object]
    checkpoint_id: str
    checkpoint_thread_id: str
    checkpoint_ns: str
    interrupted: bool


def checkpoint_config(
    checkpoint_thread_id: str,
) -> dict[str, dict[str, str]]:
    if not checkpoint_thread_id.strip():
        raise ValueError('checkpoint_thread_id is required')
    return {'configurable': {'thread_id': checkpoint_thread_id}}


def require_resumable_checkpoint(
    saver: BaseCheckpointSaver,
    checkpoint_thread_id: str,
) -> dict[str, dict[str, str]]:
    config = checkpoint_config(checkpoint_thread_id)
    try:
        saved = saver.get_tuple(config)
    except Exception:
        raise CheckpointUnavailableError('checkpoint_unavailable') from None
    if saved is None:
        raise CheckpointUnavailableError('checkpoint_unavailable')
    return config


def _read_checkpoint_tuple(
    saver: BaseCheckpointSaver,
    config: dict[str, dict[str, str]],
) -> CheckpointTuple | None:
    try:
        return saver.get_tuple(config)
    except Exception:
        raise CheckpointConfirmationError(
            _CHECKPOINT_CONFIRMATION_FAILED
        ) from None


def _configurable_values(
    config: object,
    *,
    error_message: str,
) -> Mapping[str, object]:
    if not isinstance(config, Mapping):
        raise CheckpointConfirmationError(error_message)
    configurable = config.get('configurable')
    if not isinstance(configurable, Mapping):
        raise CheckpointConfirmationError(error_message)
    return configurable


def _tuple_configurable_values(
    saved: object,
    *,
    error_message: str,
) -> Mapping[str, object]:
    try:
        config = saved.config
    except Exception:
        raise CheckpointConfirmationError(error_message) from None
    return _configurable_values(config, error_message=error_message)


def _previous_checkpoint_id(saved: CheckpointTuple | None) -> str | None:
    if saved is None:
        return None
    configurable = _tuple_configurable_values(
        saved,
        error_message=_CHECKPOINT_IDENTITY_MISMATCH,
    )
    checkpoint_id = configurable.get('checkpoint_id')
    if type(checkpoint_id) is str and checkpoint_id:
        return checkpoint_id
    return None


def _read_graph_snapshot(
    graph: object,
    config: dict[str, dict[str, str]],
) -> object:
    try:
        return graph.get_state(config)  # type: ignore[attr-defined]
    except Exception:
        raise CheckpointConfirmationError(
            _CHECKPOINT_CONFIRMATION_FAILED
        ) from None


def _normalize_interrupt_value(value: object) -> object:
    if value is None:
        return ('none',)
    if type(value) is bool:
        return ('bool', value)
    if type(value) is int:
        return ('int', value)
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointConfirmationError(
                'checkpoint interrupt state mismatch'
            )
        return ('float', value)
    if type(value) is str:
        return ('str', value)
    if type(value) is list:
        return (
            'list',
            tuple(_normalize_interrupt_value(item) for item in value),
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise CheckpointConfirmationError(
                'checkpoint interrupt state mismatch'
            )
        return (
            'dict',
            tuple(
                (key, _normalize_interrupt_value(value[key]))
                for key in sorted(value)
            ),
        )
    raise CheckpointConfirmationError(
        'checkpoint interrupt state mismatch'
    )


def _normalize_interrupt_sequence(
    interrupts: object,
) -> tuple[tuple[str, object], ...]:
    if type(interrupts) not in {list, tuple}:
        raise CheckpointConfirmationError(
            'checkpoint interrupt state mismatch'
        )
    normalized: list[tuple[str, object]] = []
    for pending in interrupts:
        if type(pending) is not Interrupt or type(pending.id) is not str:
            raise CheckpointConfirmationError(
                'checkpoint interrupt state mismatch'
            )
        normalized.append((
            pending.id,
            _normalize_interrupt_value(pending.value),
        ))
    return tuple(normalized)


def invoke_and_confirm_checkpoint(
    *,
    graph: object,
    saver: BaseCheckpointSaver,
    command_or_input: object,
    checkpoint_thread_id: str,
    runtime_context: object,
    expect_interrupt: bool,
) -> CheckpointConfirmation:
    config = checkpoint_config(checkpoint_thread_id)
    before = _read_checkpoint_tuple(saver, config)
    previous_checkpoint_id = _previous_checkpoint_id(before)
    result = graph.invoke(  # type: ignore[attr-defined]
        command_or_input,
        config,
        context=runtime_context,
        durability='sync',
    )
    if not isinstance(result, dict):
        raise CheckpointConfirmationError('graph result must be a mapping')
    saved = _read_checkpoint_tuple(saver, config)
    if saved is None:
        raise CheckpointConfirmationError('checkpoint was not persisted')
    snapshot = _read_graph_snapshot(graph, config)
    saved_config = _tuple_configurable_values(
        saved,
        error_message=_CHECKPOINT_IDENTITY_MISMATCH,
    )
    saved_thread_id = saved_config.get('thread_id')
    checkpoint_id = saved_config.get('checkpoint_id')
    checkpoint_ns = saved_config.get('checkpoint_ns', '')
    if (
        type(saved_thread_id) is not str
        or saved_thread_id != checkpoint_thread_id
        or type(checkpoint_id) is not str
        or not checkpoint_id
    ):
        raise CheckpointConfirmationError(_CHECKPOINT_IDENTITY_MISMATCH)
    if type(checkpoint_ns) is not str or checkpoint_ns != '':
        raise CheckpointConfirmationError(
            'top-level checkpoint namespace must be root'
        )
    if checkpoint_id == previous_checkpoint_id:
        raise CheckpointConfirmationError('checkpoint did not advance')
    try:
        snapshot_config = snapshot.config
    except Exception:
        raise CheckpointConfirmationError(
            _CHECKPOINT_SNAPSHOT_IDENTITY_MISMATCH
        ) from None
    snapshot_values = _configurable_values(
        snapshot_config,
        error_message=_CHECKPOINT_SNAPSHOT_IDENTITY_MISMATCH,
    )
    snapshot_identity = (
        snapshot_values.get('thread_id'),
        snapshot_values.get('checkpoint_id'),
        snapshot_values.get('checkpoint_ns', ''),
    )
    if snapshot_identity != (
        saved_thread_id,
        checkpoint_id,
        checkpoint_ns,
    ):
        raise CheckpointConfirmationError(
            _CHECKPOINT_SNAPSHOT_IDENTITY_MISMATCH
        )
    try:
        pending_interrupts = tuple(
            pending
            for task in snapshot.tasks
            for pending in getattr(task, 'interrupts', ())
        )
    except Exception:
        raise CheckpointConfirmationError(
            'checkpoint interrupt state mismatch'
        ) from None
    returned_interrupts = _normalize_interrupt_sequence(
        result.get('__interrupt__', ())
    )
    normalized_pending_interrupts = _normalize_interrupt_sequence(
        pending_interrupts
    )
    if returned_interrupts != normalized_pending_interrupts:
        raise CheckpointConfirmationError(
            'checkpoint interrupt state mismatch'
        )
    returned_interrupt = bool(returned_interrupts)
    if returned_interrupt != expect_interrupt:
        raise CheckpointConfirmationError(
            'checkpoint interrupt state mismatch'
        )
    return CheckpointConfirmation(
        result=result,
        checkpoint_id=checkpoint_id,
        checkpoint_thread_id=checkpoint_thread_id,
        checkpoint_ns=checkpoint_ns,
        interrupted=returned_interrupt,
    )
