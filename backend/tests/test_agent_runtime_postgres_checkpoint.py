import json
import os
from collections.abc import Mapping
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
    'model_output',
    'provider_error',
    'api_key',
    'oauth_token',
}
SENSITIVE_RUNTIME_CONTEXT = {
    'question': 'private integration question marker',
    'source_url': 'https://sensitive.invalid/checkpoint-source',
    'source_snippet': 'private integration source snippet marker',
    'model_output': 'private integration model output marker',
    'provider_error': 'private integration provider error marker',
    'api_key': 'private-integration-api-key-marker',
    'oauth_token': 'private-integration-oauth-token-marker',
}


class SensitiveRuntimeContext(TypedDict):
    question: str
    source_url: str
    source_snippet: str
    model_output: str
    provider_error: str
    api_key: str
    oauth_token: str


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
        if set(runtime.context) != set(SENSITIVE_RUNTIME_CONTEXT):
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


def _postgres_test_url() -> str:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if database_url is None:
        pytest.skip(
            'set PARAWORKS_TEST_POSTGRES_URL to a disposable PostgreSQL test database'
        )
    try:
        parsed_url = make_url(database_url)
    except (ArgumentError, ValueError):
        pytest.fail('PARAWORKS_TEST_POSTGRES_URL must be a valid PostgreSQL URL')
    if parsed_url.get_backend_name() != 'postgresql':
        pytest.fail('PARAWORKS_TEST_POSTGRES_URL must use PostgreSQL')
    return parsed_url.set(drivername='postgresql+psycopg').render_as_string(
        hide_password=False
    )


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


def _record_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytes):
        return value
    return json.dumps(value, sort_keys=True, default=str).encode()


def _assert_checkpoint_records_are_opaque(
    pool: object,
    thread_id: str,
) -> None:
    records: list[Mapping[str, object]] = []
    with pool.connection() as connection:  # type: ignore[attr-defined]
        records.extend(
            connection.execute(
                'SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = %s',
                (thread_id,),
            ).fetchall()
        )
        records.extend(
            connection.execute(
                'SELECT channel, type, blob FROM checkpoint_blobs '
                'WHERE thread_id = %s',
                (thread_id,),
            ).fetchall()
        )
        records.extend(
            connection.execute(
                'SELECT channel, type, blob FROM checkpoint_writes '
                'WHERE thread_id = %s',
                (thread_id,),
            ).fetchall()
        )

    assert records
    stored_bytes = b'\n'.join(
        _record_bytes(value)
        for record in records
        for value in record.values()
    ).lower()
    forbidden_markers = {
        *FORBIDDEN_CHECKPOINT_KEYS,
        *SENSITIVE_RUNTIME_CONTEXT.values(),
    }
    for marker in forbidden_markers:
        assert marker.encode().lower() not in stored_bytes


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


def test_postgres_checkpoint_resumes_after_pool_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _postgres_test_url()
    thread_id = f'postgres-restart:{uuid4()}'
    engine = create_engine(database_url)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    pool_a = None
    pool_b = None

    try:
        _run_alembic_to_head(database_url, monkeypatch)
        settings = _settings(database_url)
        bootstrap_langgraph_checkpointer(
            settings,
            backup_confirmed=True,
            session_factory=session_factory,
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
        _assert_checkpoint_records_are_opaque(pool_b, thread_id)
    finally:
        try:
            _delete_exact_checkpoint_thread(engine, thread_id)
        finally:
            if pool_b is not None:
                pool_b.close()
            if pool_a is not None:
                pool_a.close()
            engine.dispose()
            get_settings.cache_clear()
