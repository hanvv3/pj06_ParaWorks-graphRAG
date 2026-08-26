from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.app.agent_runtime.checkpointing import CheckpointUnavailableError


class CheckpointConfirmationError(RuntimeError):
    pass


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
    if saver.get_tuple(config) is None:
        raise CheckpointUnavailableError('checkpoint_unavailable')
    return config


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
    result = graph.invoke(  # type: ignore[attr-defined]
        command_or_input,
        config,
        context=runtime_context,
        durability='sync',
    )
    if not isinstance(result, dict):
        raise CheckpointConfirmationError('graph result must be a mapping')
    saved = saver.get_tuple(config)
    if saved is None:
        raise CheckpointConfirmationError('checkpoint was not persisted')
    saved_config = saved.config.get('configurable', {})
    saved_thread_id = saved_config.get('thread_id')
    checkpoint_id = saved_config.get('checkpoint_id')
    checkpoint_ns = saved_config.get('checkpoint_ns', '')
    if saved_thread_id != checkpoint_thread_id or not checkpoint_id:
        raise CheckpointConfirmationError('checkpoint identity mismatch')
    if checkpoint_ns != '':
        raise CheckpointConfirmationError(
            'top-level checkpoint namespace must be root'
        )
    snapshot = graph.get_state(config)  # type: ignore[attr-defined]
    returned_interrupt = bool(result.get('__interrupt__'))
    pending_interrupt = any(
        bool(getattr(task, 'interrupts', ())) for task in snapshot.tasks
    )
    if returned_interrupt != pending_interrupt or returned_interrupt != expect_interrupt:
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
