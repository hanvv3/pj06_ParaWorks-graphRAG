import pytest
from fastapi.testclient import TestClient

from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    CheckpointUnavailableError,
)
from backend.app.core.config import Settings
from backend.app.main import create_app


class _FakeCheckpointRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(self) -> None:
        self.events.append('start')

    def close(self) -> None:
        self.events.append('close')


def test_create_app_does_not_start_checkpoint_runtime_before_lifespan() -> None:
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)

    def runtime_factory(settings: Settings) -> _FakeCheckpointRuntime:
        assert isinstance(settings, Settings)
        events.append('factory')
        return runtime

    app = create_app(checkpoint_runtime_factory=runtime_factory)  # type: ignore[arg-type]

    assert events == ['factory']
    assert not hasattr(app.state, 'agent_checkpoint_runtime')


def test_app_lifespan_starts_exposes_and_closes_checkpoint_runtime_once() -> None:
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)

    def runtime_factory(_settings: Settings) -> _FakeCheckpointRuntime:
        events.append('factory')
        return runtime

    app = create_app(checkpoint_runtime_factory=runtime_factory)  # type: ignore[arg-type]

    with TestClient(app) as client:
        assert events == ['factory', 'start']
        assert app.state.agent_checkpoint_runtime is runtime
        assert client.get('/health').status_code == 200

    assert events == ['factory', 'start', 'close']


def test_same_app_lifespan_reentry_rejects_closed_checkpoint_runtime() -> None:
    runtime = CheckpointRuntime(
        Settings(
            _env_file=None,
            paraworks_demo_mode=True,
            langgraph_review_v2_enabled=True,
        )
    )
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
    )

    with TestClient(app):
        assert runtime.readiness.ready is True

    assert runtime.readiness.ready is False
    assert runtime.readiness.durable is False
    with pytest.raises(
        CheckpointUnavailableError,
        match='^checkpoint runtime is closed$',
    ), TestClient(app):
        pass

    assert runtime.saver is None
    assert runtime.readiness.error_code == 'checkpoint_unavailable'
