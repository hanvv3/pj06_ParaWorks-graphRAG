import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    CheckpointUnavailableError,
)
from backend.app.agent_runtime.model_router import ReviewModelUnavailableError
from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app
from backend.app.models.agent_workflows import AgentWorkflowThread
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
)


class _FakeCheckpointRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(self, preserve_existing_review_threads: bool = False) -> None:
        self.events.append(f'start:{preserve_existing_review_threads}')

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


def test_app_lifespan_starts_exposes_and_closes_checkpoint_runtime_once(
    db_session: Session,
) -> None:
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)

    def runtime_factory(_settings: Settings) -> _FakeCheckpointRuntime:
        events.append('factory')
        return runtime

    app = create_app(
        checkpoint_runtime_factory=runtime_factory,  # type: ignore[arg-type]
        workflow_session_factory=sessionmaker(
            bind=db_session.get_bind(), expire_on_commit=False
        ),
    )

    with TestClient(app) as client:
        assert events == ['factory', 'start:False']
        assert app.state.agent_checkpoint_runtime is runtime
        assert app.state.review_model_readiness.ready is True
        assert client.get('/health').status_code == 200

    assert events == ['factory', 'start:False', 'close']


def test_app_lifespan_registers_immutable_v2_graph_even_when_new_runs_disabled(
    db_session: Session,
) -> None:
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,  # type: ignore[arg-type]
        workflow_session_factory=sessionmaker(
            bind=db_session.get_bind(), expire_on_commit=False
        ),
    )

    with TestClient(app):
        builder = app.state.agent_graph_registry.resolve(
            COMPANY_MEMORY_REVIEW_WORKFLOW,
            COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        )

    assert callable(builder)
    assert events == ['start:False', 'close']


def test_disabled_start_preserves_nonterminal_existing_review_threads(
    db_session: Session,
) -> None:
    db_session.add(
        AgentWorkflowThread(
            thread_id='existing-review-thread',
            workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
            graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            checkpoint_thread_id='review-v2:existing-checkpoint',
            checkpoint_store='memory',
            owner_subject_id='owner-1',
            security_scope_id='default',
            input_hash='a' * 64,
            evidence_version_hash='b' * 64,
            status='awaiting_human_review',
        )
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,  # type: ignore[arg-type]
        workflow_session_factory=factory,
    )

    with TestClient(app):
        pass

    assert events == ['start:True', 'close']


def test_disabled_start_ignores_terminal_or_non_v2_threads(
    db_session: Session,
) -> None:
    db_session.add(
        AgentWorkflowThread(
            thread_id='terminal-review-thread',
            workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
            graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            checkpoint_thread_id='review-v2:terminal-checkpoint',
            checkpoint_store='memory',
            owner_subject_id='owner-1',
            security_scope_id='default',
            input_hash='a' * 64,
            evidence_version_hash='b' * 64,
            status='completed',
        )
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,  # type: ignore[arg-type]
        workflow_session_factory=factory,
    )

    with TestClient(app):
        pass

    assert events == ['start:False', 'close']


def test_same_app_lifespan_reentry_rejects_closed_checkpoint_runtime(
    db_session: Session,
) -> None:
    runtime = CheckpointRuntime(
        Settings(
            _env_file=None,
            paraworks_demo_mode=True,
            langgraph_review_v2_enabled=True,
        )
    )
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,
        workflow_session_factory=sessionmaker(
            bind=db_session.get_bind(), expire_on_commit=False
        ),
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


def test_disabled_runtime_can_start_only_to_preserve_existing_memory_threads() -> None:
    runtime = CheckpointRuntime(
        Settings(
            _env_file=None,
            paraworks_demo_mode=True,
            langgraph_review_v2_enabled=False,
        )
    )

    runtime.start(preserve_existing_review_threads=True)

    assert runtime.readiness.ready is True
    assert runtime.readiness.mode == 'memory'
    assert runtime.readiness.checkpoint_store == 'memory'
    runtime.close()


def test_missing_production_model_credentials_do_not_block_app_lifespan(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('AGENT_LLM_ENABLED', 'false')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    get_settings.cache_clear()
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,  # type: ignore[arg-type]
        workflow_session_factory=sessionmaker(
            bind=db_session.get_bind(), expire_on_commit=False
        ),
    )

    try:
        with TestClient(app) as client:
            assert client.get('/health').status_code == 200
            assert app.state.review_workflow_service is not None
            assert app.state.review_model_readiness.ready is False
            assert (
                app.state.review_model_readiness.error_code
                == 'model_unavailable'
            )
    finally:
        get_settings.cache_clear()

    assert events == ['start:False', 'close']


def test_lifespan_bootstraps_c5_keys_before_building_drafting_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)

    def ensure_initialized(_self):
        events.append('key_bootstrap')
        return type(
            'Result',
            (),
            {
                'schema_available': True,
                'initialized': False,
                'ready': False,
            },
        )()

    def build_catalog(_settings):
        assert 'key_bootstrap' in events
        events.append('catalog')
        raise ReviewModelUnavailableError('unavailable')

    monkeypatch.setattr(
        'backend.app.main.AutoReviewKeyBootstrapService.ensure_initialized',
        ensure_initialized,
    )
    monkeypatch.setattr('backend.app.main.build_review_agent_catalog', build_catalog)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,  # type: ignore[arg-type]
    )

    with TestClient(app):
        pass

    assert events == ['start:False', 'key_bootstrap', 'catalog', 'close']


def test_lifespan_fails_closed_when_c5_key_bootstrap_database_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = _FakeCheckpointRuntime(events)
    database_error = OperationalError(
        'SELECT auto_review_runtime_key_states',
        {},
        RuntimeError('database unavailable'),
    )

    def ensure_initialized(_self):
        events.append('key_bootstrap')
        raise database_error

    def build_catalog(_settings):
        events.append('catalog')
        raise AssertionError('catalog must not be constructed after bootstrap failure')

    monkeypatch.setattr(
        'backend.app.main.AutoReviewKeyBootstrapService.ensure_initialized',
        ensure_initialized,
    )
    monkeypatch.setattr('backend.app.main.build_review_agent_catalog', build_catalog)
    app = create_app(
        checkpoint_runtime_factory=lambda _settings: runtime,  # type: ignore[arg-type]
    )

    with pytest.raises(OperationalError) as exc_info, TestClient(app):
        pass

    assert exc_info.value is database_error
    assert events == ['start:False', 'key_bootstrap', 'close']
