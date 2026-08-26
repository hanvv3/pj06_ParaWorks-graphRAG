import json
import os
from collections.abc import Callable, Mapping, Sequence
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import sessionmaker
from typing_extensions import TypedDict

from backend.app.agent_runtime import (
    GraphVersionRegistry,
    bootstrap_langgraph_checkpointer,
    checkpoint_config,
    invoke_and_confirm_checkpoint,
    require_resumable_checkpoint,
)
from backend.app.agent_runtime.checkpointing import (
    build_postgres_pool,
    build_postgres_saver,
    build_strict_checkpoint_serializer,
    sqlalchemy_url_to_psycopg_dsn,
)
from backend.app.agent_runtime.state import ReviewGraphState
from backend.app.core.config import Settings, get_settings

WORKFLOW_NAME = 'company-memory-review'
GRAPH_VERSION = 'company-memory-review-v2.0'
EXPECTED_RESUME = {
    'event': 'review_resolution_checked',
    'state_version': 1,
}
FORBIDDEN_CHECKPOINT_KEYS = {
    'question',
    'objective',
    'source_url',
    'source_snippet',
    'message_body',
    'relationship_path',
    'prompt',
    'llm_prompt',
    'model_output',
    'provider_error',
    'api_key',
    'oauth_token',
    'raw_connector_payload',
}
SENSITIVE_RUNTIME_CONTEXT = {
    'question': 'private integration question marker',
    'source_url': 'https://sensitive.invalid/checkpoint-source',
    'source_snippet': 'private integration source snippet marker',
    'relationship_path': 'private relationship path marker',
    'llm_prompt': 'private llm prompt marker',
    'model_output': 'private integration model output marker',
    'provider_error': 'private integration provider error marker',
    'api_key': 'private-integration-api-key-marker',
    'oauth_token': 'private-integration-oauth-token-marker',
    'raw_connector_payload': 'private raw connector payload marker',
}
_ISOLATED_TEST_DATABASE_ERROR = 'isolated PostgreSQL test database required'
_KNOWN_NON_TEST_DATABASES = frozenset({
    'postgres',
    'template0',
    'template1',
    'paraworks',
    'paraworks_dev',
    'paraworks_prod',
})
_KNOWN_NON_TEST_USERS = frozenset({'postgres', 'paraworks'})


class SensitiveRuntimeContext(TypedDict):
    question: str
    source_url: str
    source_snippet: str
    relationship_path: str
    llm_prompt: str
    model_output: str
    provider_error: str
    api_key: str
    oauth_token: str
    raw_connector_payload: str


def _checkpoint_state(thread_id: str) -> ReviewGraphState:
    return {
        'workflow_thread_id': f'workflow:{thread_id}',
        'graph_version': GRAPH_VERSION,
        'input_hash': 'a' * 64,
        'evidence_version_hash': 'b' * 64,
        'review_item_ids': [101],
        'review_status_counts': {'pending_review': 1},
        'phase': 'checkpoint_pending',
        'completed_nodes': ['draft_candidates'],
        'error_codes': [],
    }


def _build_test_graph(saver: BaseCheckpointSaver) -> object:
    def await_review(
        _state: ReviewGraphState,
        runtime: Runtime[SensitiveRuntimeContext],
    ) -> dict[str, object]:
        if dict(runtime.context) != SENSITIVE_RUNTIME_CONTEXT:
            return {
                'phase': 'failed',
                'error_codes': ['invalid_input'],
            }
        resolution = interrupt({
            'event': 'review_resolution_required',
            'state_version': 1,
        })
        if resolution != EXPECTED_RESUME:
            return {
                'phase': 'failed',
                'error_codes': ['review_unresolved'],
            }
        return {
            'phase': 'completed',
            'review_status_counts': {'approved': 1},
            'completed_nodes': ['review_resolution_checked'],
        }

    builder = StateGraph(
        ReviewGraphState,
        context_schema=SensitiveRuntimeContext,
    )
    builder.add_node('await_review', await_review)
    builder.add_edge(START, 'await_review')
    builder.add_edge('await_review', END)
    return builder.compile(checkpointer=saver)


def _isolated_postgres_test_url(database_url: str) -> str:
    try:
        parsed_url = make_url(database_url)
    except (ArgumentError, ValueError):
        raise ValueError(_ISOLATED_TEST_DATABASE_ERROR) from None
    if parsed_url.get_backend_name() != 'postgresql':
        raise ValueError(_ISOLATED_TEST_DATABASE_ERROR)
    _validate_test_database_identity(
        parsed_url.database,
        parsed_url.username,
    )
    return parsed_url.set(drivername='postgresql+psycopg').render_as_string(
        hide_password=False
    )


def _validate_test_database_identity(
    database_name: str | None,
    user_name: str | None,
) -> None:
    normalized_database = (database_name or '').lower()
    normalized_user = (user_name or '').lower()
    if (
        normalized_database in _KNOWN_NON_TEST_DATABASES
        or not normalized_database.endswith('_test')
        or normalized_user in _KNOWN_NON_TEST_USERS
        or not normalized_user.endswith('_test')
    ):
        raise ValueError(_ISOLATED_TEST_DATABASE_ERROR)


def _postgres_test_url() -> str:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if database_url is None:
        pytest.skip(
            'set PARAWORKS_TEST_POSTGRES_URL to a disposable PostgreSQL test database'
        )
    return _isolated_postgres_test_url(database_url)


def _settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=database_url,
        database_url=database_url,
        langgraph_review_v2_enabled=True,
        langgraph_strict_msgpack=True,
    )


def _run_alembic_to_head(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        get_settings.cache_clear()


def _prepare_postgres_test_database(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine_factory: Callable[[str], object] = create_engine,
    alembic_runner: Callable[[str, pytest.MonkeyPatch], None] = (
        _run_alembic_to_head
    ),
    bootstrapper: Callable[..., object] = bootstrap_langgraph_checkpointer,
) -> object:
    isolated_url = _isolated_postgres_test_url(database_url)
    expected_url = make_url(isolated_url)
    engine = engine_factory(isolated_url)
    try:
        with engine.connect() as connection:  # type: ignore[attr-defined]
            database_name, user_name = connection.execute(
                text('SELECT current_database(), current_user')
            ).one()
        _validate_test_database_identity(database_name, user_name)
        if (
            database_name != expected_url.database
            or user_name != expected_url.username
        ):
            raise ValueError(_ISOLATED_TEST_DATABASE_ERROR)

        alembic_runner(isolated_url, monkeypatch)
        session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
        )
        bootstrapper(
            _settings(isolated_url),
            backup_confirmed=True,
            session_factory=session_factory,
        )
        return engine
    except Exception:
        engine.dispose()  # type: ignore[attr-defined]
        raise


def _record_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytes):
        return value
    return json.dumps(value, sort_keys=True, default=str).encode()


def _assert_checkpoint_payloads_are_opaque(
    payloads: Sequence[object],
) -> None:
    stored_bytes = b'\n'.join(_record_bytes(value) for value in payloads).lower()
    forbidden_markers = {
        *FORBIDDEN_CHECKPOINT_KEYS,
        *SENSITIVE_RUNTIME_CONTEXT.values(),
    }
    for marker in forbidden_markers:
        assert marker.encode().lower() not in stored_bytes


def _assert_checkpoint_records_are_opaque(
    pool: object,
    saver: BaseCheckpointSaver,
    thread_id: str,
) -> None:
    records: list[Mapping[str, object]] = []
    with pool.connection() as connection:  # type: ignore[attr-defined]
        records.extend(
            connection.execute(
                'SELECT * FROM checkpoints WHERE thread_id = %s',
                (thread_id,),
            ).fetchall()
        )
        records.extend(
            connection.execute(
                'SELECT * FROM checkpoint_blobs WHERE thread_id = %s',
                (thread_id,),
            ).fetchall()
        )
        records.extend(
            connection.execute(
                'SELECT * FROM checkpoint_writes WHERE thread_id = %s',
                (thread_id,),
            ).fetchall()
        )

    assert records
    saved_tuples = list(saver.list(checkpoint_config(thread_id)))
    assert saved_tuples
    payloads: list[object] = [*records]
    for saved in saved_tuples:
        payloads.extend([
            saved.config,
            saved.checkpoint,
            saved.metadata,
            saved.parent_config,
            saved.pending_writes,
        ])
    _assert_checkpoint_payloads_are_opaque(payloads)


def _delete_exact_checkpoint_thread(engine: object, thread_id: str) -> None:
    with engine.begin() as connection:  # type: ignore[attr-defined]
        for table_name in (
            'checkpoint_writes',
            'checkpoint_blobs',
            'checkpoints',
        ):
            table_exists = connection.scalar(
                text('SELECT to_regclass(:table_name)'),
                {'table_name': table_name},
            )
            if table_exists is not None:
                connection.execute(
                    text(
                        f'DELETE FROM {table_name} '
                        'WHERE thread_id = :thread_id'
                    ),
                    {'thread_id': thread_id},
                )


def _cleanup_checkpoint_test_resources(
    *,
    engine: object | None,
    thread_id: str,
    pool_b: object | None,
    pool_a: object | None,
    delete_thread: Callable[[object, str], None] = (
        _delete_exact_checkpoint_thread
    ),
    clear_settings: Callable[[], None] = get_settings.cache_clear,
) -> None:
    try:
        if engine is not None:
            delete_thread(engine, thread_id)
    finally:
        try:
            if pool_b is not None:
                pool_b.close()  # type: ignore[attr-defined]
        finally:
            try:
                if pool_a is not None:
                    pool_a.close()  # type: ignore[attr-defined]
            finally:
                try:
                    if engine is not None:
                        engine.dispose()  # type: ignore[attr-defined]
                finally:
                    clear_settings()


@pytest.mark.parametrize(
    'sensitive_value',
    [
        'relationship_path',
        'private relationship path marker',
        'llm_prompt',
        'private llm prompt marker',
        'raw_connector_payload',
        'private raw connector payload marker',
    ],
)
@pytest.mark.parametrize(
    'payload_factory',
    [
        lambda value: {'payload': value},
        lambda value: value.encode(),
        lambda value: ('task', 'channel', [value]),
    ],
)
def test_checkpoint_payload_inspection_rejects_every_sensitive_category(
    sensitive_value: str,
    payload_factory,
) -> None:
    with pytest.raises(AssertionError):
        _assert_checkpoint_payloads_are_opaque([
            payload_factory(sensitive_value),
        ])


@pytest.mark.parametrize(
    'database_url',
    [
        'postgresql+psycopg://runtime_test:unused@localhost/postgres',
        'postgresql+psycopg://runtime_test:unused@localhost/paraworks',
        'postgresql+psycopg://runtime_test:unused@localhost/paraworks_dev',
        'postgresql+psycopg://runtime_test:unused@localhost/paraworks_prod',
        'postgresql+psycopg://postgres:unused@localhost/paraworks_runtime_test',
        'postgresql+psycopg://paraworks:unused@localhost/paraworks_runtime_test',
    ],
)
def test_unsafe_database_identity_is_rejected_before_any_mutation(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    with pytest.raises(
        ValueError,
        match='^isolated PostgreSQL test database required$',
    ):
        _prepare_postgres_test_database(
            database_url,
            monkeypatch,
            engine_factory=lambda _url: events.append('engine'),
            alembic_runner=lambda *_args: events.append('alembic'),
            bootstrapper=lambda *_args, **_kwargs: events.append('bootstrap'),
        )

    assert events == []


class _IdentityResult:
    def __init__(self, database_name: str, user_name: str) -> None:
        self._identity = (database_name, user_name)

    def one(self) -> tuple[str, str]:
        return self._identity


class _IdentityConnection:
    def __init__(
        self,
        events: list[str],
        database_name: str,
        user_name: str,
    ) -> None:
        self._events = events
        self._database_name = database_name
        self._user_name = user_name

    def __enter__(self) -> '_IdentityConnection':
        self._events.append('connection_enter')
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append('connection_exit')

    def execute(self, _statement: object) -> _IdentityResult:
        self._events.append('identity_query')
        return _IdentityResult(self._database_name, self._user_name)


class _IdentityEngine:
    def __init__(
        self,
        events: list[str],
        database_name: str,
        user_name: str,
    ) -> None:
        self._events = events
        self._database_name = database_name
        self._user_name = user_name

    def connect(self) -> _IdentityConnection:
        self._events.append('connect')
        return _IdentityConnection(
            self._events,
            self._database_name,
            self._user_name,
        )

    def dispose(self) -> None:
        self._events.append('dispose')


def test_connected_identity_mismatch_is_rejected_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    database_url = (
        'postgresql+psycopg://runtime_test:unused@localhost/'
        'paraworks_runtime_test'
    )

    with pytest.raises(
        ValueError,
        match='^isolated PostgreSQL test database required$',
    ):
        _prepare_postgres_test_database(
            database_url,
            monkeypatch,
            engine_factory=lambda _url: _IdentityEngine(
                events,
                'paraworks',
                'runtime_test',
            ),
            alembic_runner=lambda *_args: events.append('alembic'),
            bootstrapper=lambda *_args, **_kwargs: events.append('bootstrap'),
        )

    assert events == [
        'connect',
        'connection_enter',
        'identity_query',
        'connection_exit',
        'dispose',
    ]


class _CleanupResource:
    def __init__(
        self,
        name: str,
        events: list[str],
        failing_step: str,
    ) -> None:
        self._name = name
        self._events = events
        self._failing_step = failing_step

    def close(self) -> None:
        self._run()

    def dispose(self) -> None:
        self._run()

    def _run(self) -> None:
        self._events.append(self._name)
        if self._name == self._failing_step:
            raise RuntimeError(self._name)


@pytest.mark.parametrize(
    'failing_step',
    ['delete', 'pool_b', 'pool_a', 'engine'],
)
def test_cleanup_attempts_every_stage_after_an_earlier_failure(
    failing_step: str,
) -> None:
    events: list[str] = []

    def delete_thread(_engine: object, _thread_id: str) -> None:
        events.append('delete')
        if failing_step == 'delete':
            raise RuntimeError('delete')

    with pytest.raises(RuntimeError, match=f'^{failing_step}$'):
        _cleanup_checkpoint_test_resources(
            engine=_CleanupResource('engine', events, failing_step),
            thread_id='postgres-restart:test-only',
            pool_b=_CleanupResource('pool_b', events, failing_step),
            pool_a=_CleanupResource('pool_a', events, failing_step),
            delete_thread=delete_thread,
            clear_settings=lambda: events.append('settings'),
        )

    assert events == ['delete', 'pool_b', 'pool_a', 'engine', 'settings']


def test_postgres_checkpoint_resumes_after_pool_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _postgres_test_url()
    thread_id = f'postgres-restart:{uuid4()}'
    engine = None
    pool_a = None
    pool_b = None

    try:
        engine = _prepare_postgres_test_database(
            database_url,
            monkeypatch,
        )

        dsn = sqlalchemy_url_to_psycopg_dsn(database_url)
        serializer = build_strict_checkpoint_serializer()
        registry = GraphVersionRegistry()
        registry.register(WORKFLOW_NAME, GRAPH_VERSION, _build_test_graph)

        pool_a = build_postgres_pool(dsn)
        pool_a.open(wait=True)
        saver_a = build_postgres_saver(pool_a, serializer)
        graph_a = registry.resolve(WORKFLOW_NAME, GRAPH_VERSION)(saver_a)
        paused = invoke_and_confirm_checkpoint(
            graph=graph_a,
            saver=saver_a,
            command_or_input=_checkpoint_state(thread_id),
            checkpoint_thread_id=thread_id,
            runtime_context=dict(SENSITIVE_RUNTIME_CONTEXT),
            expect_interrupt=True,
        )
        assert paused.interrupted is True
        assert paused.result['__interrupt__']
        assert paused.checkpoint_ns == ''

        pool_a.close()
        pool_a = None

        pool_b = build_postgres_pool(dsn)
        pool_b.open(wait=True)
        saver_b = build_postgres_saver(pool_b, serializer)
        resumable_config = require_resumable_checkpoint(saver_b, thread_id)
        restarted_tuple = saver_b.get_tuple(resumable_config)
        assert restarted_tuple is not None
        restarted_values = restarted_tuple.config['configurable']
        assert restarted_values['checkpoint_id'] == paused.checkpoint_id
        assert restarted_values.get('checkpoint_ns', '') == ''

        graph_b = registry.resolve(WORKFLOW_NAME, GRAPH_VERSION)(saver_b)
        resumed = invoke_and_confirm_checkpoint(
            graph=graph_b,
            saver=saver_b,
            command_or_input=Command(resume=EXPECTED_RESUME),
            checkpoint_thread_id=thread_id,
            runtime_context=dict(SENSITIVE_RUNTIME_CONTEXT),
            expect_interrupt=False,
        )

        assert resumed.interrupted is False
        assert resumed.checkpoint_id != paused.checkpoint_id
        assert resumed.checkpoint_ns == ''
        assert resumed.result['phase'] == 'completed'
        assert resumed.result['review_status_counts'] == {'approved': 1}
        assert resumed.result['completed_nodes'] == [
            'draft_candidates',
            'review_resolution_checked',
        ]
        assert '__interrupt__' not in resumed.result

        saved = saver_b.get_tuple(checkpoint_config(thread_id))
        assert saved is not None
        saved_values = saved.config['configurable']
        assert saved_values['checkpoint_id'] == resumed.checkpoint_id
        assert saved_values.get('checkpoint_ns', '') == ''
        _assert_checkpoint_records_are_opaque(pool_b, saver_b, thread_id)
    finally:
        _cleanup_checkpoint_test_resources(
            engine=engine,
            thread_id=thread_id,
            pool_b=pool_b,
            pool_a=pool_a,
        )
