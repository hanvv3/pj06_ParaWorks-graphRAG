from __future__ import annotations

import os
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from threading import Barrier, Lock
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from langgraph.types import Interrupt
from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime import bootstrap_langgraph_checkpointer
from backend.app.agent_runtime.checkpoint_execution import checkpoint_config
from backend.app.agent_runtime.checkpointing import CheckpointRuntime
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    register_company_memory_review_v2,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_drafting import ReviewDraftResult
from backend.app.agent_runtime.review_v2_preflight import PreparedReviewRequest
from backend.app.agent_runtime.review_v2_service import (
    ReviewWorkflowService,
    ReviewWorkflowServiceError,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import USERS, DemoUser
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AuditLog,
    DecisionRecord,
    HistoryEvent,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
)
from backend.app.review.transitions import ReviewTransitionService
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    ReviewWorkflowDryRunResponse,
    ReviewWorkflowRunRequest,
    ReviewWorkflowSourceRef,
)

_ISOLATED_DATABASE_ERROR = 'isolated PostgreSQL test database required'
_NON_TEST_DATABASES = frozenset({
    'postgres',
    'template0',
    'template1',
    'paraworks',
    'paraworks_dev',
    'paraworks_prod',
})
_NON_TEST_USERS = frozenset({'postgres', 'paraworks'})
_SENSITIVE_MARKERS = frozenset({
    'gmail:postgres-sensitive-source',
    'https://postgres-sensitive.invalid/source',
    'postgres-sensitive-source-snippet',
    'postgres-sensitive-provider-prompt',
    'postgres-sensitive-model-output',
    'postgres-sensitive-provider-error',
    'postgres-sensitive-api-key',
})
_FORBIDDEN_CHECKPOINT_KEYS = frozenset({
    'source_id',
    'source_ids',
    'source_url',
    'source_urls',
    'source_link',
    'source_links',
    'source_snippet',
    'source_snippets',
    'source_ref',
    'source_refs',
    'review_item_id',
    'review_item_ids',
    'prompt',
    'llm_prompt',
    'model_input',
    'model_output',
    'provider_error',
    'raw_provider_error',
    'raw_error',
    'api_key',
    'oauth_token',
    'credential',
    'credentials',
    'raw_connector_payload',
})


def _validate_database_identity(
    database_name: str | None,
    user_name: str | None,
) -> None:
    normalized_database = (database_name or '').lower()
    normalized_user = (user_name or '').lower()
    if (
        normalized_database in _NON_TEST_DATABASES
        or not normalized_database.endswith('_test')
        or normalized_user in _NON_TEST_USERS
        or not normalized_user.endswith('_test')
    ):
        raise ValueError(_ISOLATED_DATABASE_ERROR)


def _isolated_postgres_url(database_url: str) -> str:
    try:
        parsed = make_url(database_url)
    except (ArgumentError, ValueError):
        raise ValueError(_ISOLATED_DATABASE_ERROR) from None
    if parsed.get_backend_name() != 'postgresql':
        raise ValueError(_ISOLATED_DATABASE_ERROR)
    _validate_database_identity(parsed.database, parsed.username)
    return parsed.set(drivername='postgresql+psycopg').render_as_string(
        hide_password=False
    )


def _postgres_test_url() -> str:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip(
            'set PARAWORKS_TEST_POSTGRES_URL to a disposable PostgreSQL '
            'database whose database and user names end in _test'
        )
    return _isolated_postgres_url(database_url)


def _run_migrations(
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


def _settings(database_url: str, scope_id: str) -> Settings:
    return Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=database_url,
        database_url=database_url,
        langgraph_review_v2_enabled=True,
        langgraph_strict_msgpack=True,
        agent_runtime_security_scope_id=scope_id,
        agent_runtime_fingerprint_secret='postgres-review-v2-test-secret',
        agent_runtime_fingerprint_key_version='postgres-test-v1',
    )


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            name='history_agent',
            owner='developer-c',
            input_contract='EvidencePacket',
            output_contract='ReviewCandidate',
            prompt_versions=('postgres-deterministic-v1',),
            supported_permissions=('public', 'internal', 'restricted'),
            capabilities=('history_candidate',),
        )
    )
    return registry


def _shared_launch_actors() -> tuple[DemoUser, DemoUser]:
    return (
        DemoUser(
            id=f'postgres-creator-{uuid4().hex}',
            email=f'creator-{uuid4().hex}@postgres-test.invalid',
            role='admin',
            permission_levels={'public', 'internal', 'restricted'},
            name='PostgreSQL Creator',
            title='Release verifier',
            department='Platform',
            aliases=(f'creator-alias-{uuid4().hex}',),
        ),
        DemoUser(
            id=f'postgres-reuser-{uuid4().hex}',
            email=f'reuser-{uuid4().hex}@postgres-test.invalid',
            role='reviewer',
            permission_levels={'public', 'internal'},
            name='PostgreSQL Reuser',
            title='Review verifier',
            department='Product',
            aliases=(f'reuser-alias-{uuid4().hex}',),
        ),
    )


@dataclass
class _DeterministicDraftService:
    session_factory: Callable[[], Session]
    draft_calls: int = 0
    _lock: Lock = field(default_factory=Lock)

    def preview_prepared(
        self,
        *,
        prepared: PreparedReviewRequest,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewWorkflowDryRunResponse:
        del actor_subject_id, allowed_permission_levels
        return ReviewWorkflowDryRunResponse(
            workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
            graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            source_count=len(prepared.source_refs),
            agent_names=list(prepared.agent_names),
            selection_policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
            estimated_input_tokens=64,
            estimated_output_tokens=32,
            estimated_cost_usd=0.0,
            budget_limit_usd=0.001,
            budget_status='within_budget',
            cache_hit=False,
            requires_explicit_run=True,
        )

    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewDraftResult:
        del actor_subject_id, allowed_permission_levels
        with self._lock:
            self.draft_calls += 1
        with self.session_factory() as db:
            items = tuple(
                db.scalars(
                    select(ReviewItem)
                    .where(
                        ReviewItem.workflow_thread_id
                        == workflow_thread_id
                    )
                    .order_by(ReviewItem.id)
                ).all()
            )
            if not items:
                run = AgentRun(
                    agent_name='history_agent',
                    prompt_version='postgres-deterministic-v1',
                    status='complete',
                    source_window='postgres-release-window',
                    cache_key=f'postgres-cache:{workflow_thread_id}',
                    model_name='deterministic-test-adapter',
                    input_tokens=64,
                    output_tokens=32,
                    total_tokens=96,
                    estimated_cost_usd=0.0,
                    permission_level='internal',
                    metadata_={'adapter': 'deterministic'},
                    workflow_thread_id=workflow_thread_id,
                    effect_key=f'postgres-effect:{workflow_thread_id}',
                    completed_at=datetime.now(UTC),
                )
                db.add(run)
                db.flush()
                item = ReviewItem(
                    id=int(uuid4().hex[:7], 16) + 100_000_000,
                    item_type='history_event',
                    payload={
                        'title': 'PostgreSQL recovery candidate',
                        'reason': 'Durable review recovery must be proven.',
                        'agent_name': 'history_agent',
                        'agent_run_id': run.id,
                        'source_ids': [
                            'gmail:postgres-sensitive-source'
                        ],
                        'prompt': 'postgres-sensitive-provider-prompt',
                        'model_output': 'postgres-sensitive-model-output',
                        'provider_error': 'postgres-sensitive-provider-error',
                        'api_key': 'postgres-sensitive-api-key',
                    },
                    source_links=[
                        'https://postgres-sensitive.invalid/source'
                    ],
                    source_snippets=[
                        'postgres-sensitive-source-snippet'
                    ],
                    confidence_score=0.93,
                    permission_level='internal',
                    status='pending_review',
                    workflow_thread_id=workflow_thread_id,
                    candidate_key=(
                        f'postgres-candidate:{workflow_thread_id}'
                    ),
                )
                db.add(item)
                db.commit()
                db.refresh(item)
                items = (item,)
            counts = {
                'pending_review': 0,
                'approved': 0,
                'rejected': 0,
                'needs_more_evidence': 0,
            }
            for item in items:
                counts[item.status] += 1
            ids = tuple(item.id for item in items)
            db.rollback()
        return ReviewDraftResult(
            review_item_ids=ids,
            review_status_counts=counts,
        )


class _ExplodingGraph:
    def invoke(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError('deterministic checkpoint failure')


@dataclass
class _AppResources:
    application_engine: Engine
    session_factory: sessionmaker[Session]
    runtime: CheckpointRuntime
    service: ReviewWorkflowService
    draft_service: _DeterministicDraftService
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.runtime.close()
        finally:
            self.application_engine.dispose()
            self.closed = True


@dataclass
class _PostgresHarness:
    database_url: str
    engine: Engine
    session_factory: sessionmaker[Session]
    settings: Settings
    scope_id: str
    source_ids: set[int] = field(default_factory=set)
    checkpoint_thread_ids: set[str] = field(default_factory=set)
    apps: list[_AppResources] = field(default_factory=list)

    def new_app(self, *, exploding_graph: bool = False) -> _AppResources:
        application_engine = create_engine(self.database_url)
        application_session_factory = sessionmaker(
            bind=application_engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        runtime = CheckpointRuntime(self.settings)
        try:
            runtime.start()
            assert runtime.readiness.ready is True
            assert runtime.readiness.mode == 'postgres'
            assert runtime.readiness.durable is True
            registry = GraphVersionRegistry()
            if exploding_graph:
                registry.register(
                    COMPANY_MEMORY_REVIEW_WORKFLOW,
                    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
                    lambda _saver: _ExplodingGraph(),
                )
            else:
                register_company_memory_review_v2(registry)
            draft_service = _DeterministicDraftService(
                application_session_factory
            )
            service = ReviewWorkflowService(
                session_factory=application_session_factory,
                settings=self.settings,
                checkpoint_runtime=runtime,
                graph_registry=registry,
                agent_registry=_registry(),
                draft_service=draft_service,
            )
        except Exception:
            try:
                runtime.close()
            finally:
                application_engine.dispose()
            raise
        resources = _AppResources(
            application_engine=application_engine,
            session_factory=application_session_factory,
            runtime=runtime,
            service=service,
            draft_service=draft_service,
        )
        self.apps.append(resources)
        return resources

    def seed_source_request(
        self,
        *,
        client_request_id: str | None = None,
    ) -> ReviewWorkflowRunRequest:
        suffix = uuid4().hex
        source_id = f'gmail:postgres-sensitive-source:{suffix}'
        signature = f'postgres-signature-{suffix}'
        with self.session_factory() as db:
            source = Source(
                source_type='gmail',
                source_id=source_id,
                source_url=(
                    f'https://postgres-sensitive.invalid/source/{suffix}'
                ),
                title='PostgreSQL recovery source',
                author='deterministic-fixture',
                permission_level='internal',
                raw_metadata={
                    'content_signature': signature,
                    'review_batch_mode': 'v2_explicit',
                    'review_batch_signature': signature,
                    'external_revision': f'revision-{suffix}',
                },
            )
            db.add(source)
            db.commit()
            db.refresh(source)
            self.source_ids.add(source.id)
        return ReviewWorkflowRunRequest(
            source_refs=[
                ReviewWorkflowSourceRef(
                    source_type='gmail',
                    source_id=source_id,
                    version_or_signature=signature,
                )
            ],
            agent_names=['history_agent'],
            client_request_id=client_request_id,
        )

    def remember_thread(self, workflow_thread_id: str) -> None:
        with self.session_factory() as db:
            thread = db.get(AgentWorkflowThread, workflow_thread_id)
            assert thread is not None
            self.checkpoint_thread_ids.add(thread.checkpoint_thread_id)
            db.rollback()

    def cleanup(self) -> None:
        try:
            try:
                for app in reversed(self.apps):
                    app.close()
            finally:
                with self.session_factory() as db:
                    threads = tuple(
                        db.scalars(
                            select(AgentWorkflowThread).where(
                                AgentWorkflowThread.security_scope_id
                                == self.scope_id
                            )
                        ).all()
                    )
                    thread_ids = tuple(
                        thread.thread_id for thread in threads
                    )
                    self.checkpoint_thread_ids.update(
                        thread.checkpoint_thread_id for thread in threads
                    )
                    review_ids = (
                        tuple(
                            db.scalars(
                                select(ReviewItem.id).where(
                                    ReviewItem.workflow_thread_id.in_(
                                        thread_ids
                                    )
                                )
                            ).all()
                        )
                        if thread_ids
                        else ()
                    )
                    if review_ids:
                        for model in (
                            DecisionRecord,
                            HistoryEvent,
                            TimelineEvent,
                            Todo,
                        ):
                            db.execute(
                                delete(model).where(
                                    model.source_review_item_id.in_(
                                        review_ids
                                    )
                                )
                            )
                        db.execute(
                            delete(ReviewItem).where(
                                ReviewItem.id.in_(review_ids)
                            )
                        )
                    if thread_ids:
                        db.execute(
                            delete(AuditLog).where(
                                AuditLog.target_id.in_(thread_ids)
                            )
                        )
                        db.execute(
                            delete(AgentRun).where(
                                AgentRun.workflow_thread_id.in_(thread_ids)
                            )
                        )
                        db.execute(
                            delete(AgentWorkflowEvidenceRef).where(
                                AgentWorkflowEvidenceRef.workflow_thread_id.in_(
                                    thread_ids
                                )
                            )
                        )
                        db.execute(
                            delete(AgentWorkflowRequest).where(
                                AgentWorkflowRequest.workflow_thread_id.in_(
                                    thread_ids
                                )
                            )
                        )
                        db.execute(
                            delete(AgentWorkflowThread).where(
                                AgentWorkflowThread.thread_id.in_(thread_ids)
                            )
                        )
                    if self.source_ids:
                        db.execute(
                            delete(Source).where(
                                Source.id.in_(self.source_ids)
                            )
                        )
                    db.commit()
        finally:
            try:
                with self.engine.begin() as connection:
                    for checkpoint_thread_id in self.checkpoint_thread_ids:
                        for table_name in (
                            'checkpoint_writes',
                            'checkpoint_blobs',
                            'checkpoints',
                        ):
                            if connection.scalar(
                                text('SELECT to_regclass(:table_name)'),
                                {'table_name': table_name},
                            ) is not None:
                                connection.execute(
                                    text(
                                        f'DELETE FROM {table_name} '
                                        'WHERE thread_id = :thread_id'
                                    ),
                                    {'thread_id': checkpoint_thread_id},
                                )
            finally:
                self.engine.dispose()


@pytest.fixture
def postgres_review_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_PostgresHarness, None, None]:
    database_url = _postgres_test_url()
    expected = make_url(database_url)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            database_name, user_name = connection.execute(
                text('SELECT current_database(), current_user')
            ).one()
        _validate_database_identity(database_name, user_name)
        if database_name != expected.database or user_name != expected.username:
            raise ValueError(_ISOLATED_DATABASE_ERROR)
        _run_migrations(database_url, monkeypatch)
        factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        scope_id = f'postgres-review-v2:{uuid4().hex}'
        settings = _settings(database_url, scope_id)
        bootstrap_langgraph_checkpointer(
            settings,
            backup_confirmed=True,
            session_factory=factory,
        )
        harness = _PostgresHarness(
            database_url=database_url,
            engine=engine,
            session_factory=factory,
            settings=settings,
            scope_id=scope_id,
        )
        try:
            yield harness
        finally:
            harness.cleanup()
    except Exception:
        engine.dispose()
        raise


def _approve_only_item(
    harness: _PostgresHarness,
    workflow_thread_id: str,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    with harness.session_factory() as db:
        item = db.scalars(
            select(ReviewItem).where(
                ReviewItem.workflow_thread_id == workflow_thread_id
            )
        ).one()
        result = ReviewTransitionService().transition(
            db=db,
            item_id=item.id,
            action='approve',
            actor=USERS['admin'],
        )
        assert result.replayed is False
        assert result.promotion is not None
        db.commit()
        return (
            item.id,
            result.promotion.created_record_ids,
            result.promotion.created_timeline_event_ids,
        )


def _business_counts(
    harness: _PostgresHarness,
    workflow_thread_id: str,
) -> tuple[int, int, int, int]:
    with harness.session_factory() as db:
        item_ids = tuple(
            db.scalars(
                select(ReviewItem.id).where(
                    ReviewItem.workflow_thread_id == workflow_thread_id
                )
            ).all()
        )
        run_count = db.scalar(
            select(func.count()).select_from(AgentRun).where(
                AgentRun.workflow_thread_id == workflow_thread_id
            )
        )
        history_count = (
            db.scalar(
                select(func.count()).select_from(HistoryEvent).where(
                    HistoryEvent.source_review_item_id.in_(item_ids)
                )
            )
            if item_ids
            else 0
        )
        timeline_count = (
            db.scalar(
                select(func.count()).select_from(TimelineEvent).where(
                    TimelineEvent.source_review_item_id.in_(item_ids)
                )
            )
            if item_ids
            else 0
        )
        db.rollback()
    return (
        int(run_count or 0),
        len(item_ids),
        int(history_count or 0),
        int(timeline_count or 0),
    )


def _walk_values(value: object) -> Generator[object, None, None]:
    yield value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_values(key)
            yield from _walk_values(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            yield from _walk_values(getattr(value, field_info.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    elif isinstance(value, Interrupt):
        yield from _walk_values(value.value)


def _binary_value_markers(value: object) -> tuple[bytes, ...]:
    if isinstance(value, memoryview):
        return (value.tobytes().lower(),)
    if isinstance(value, bytes):
        return (value.lower(),)
    if isinstance(value, str):
        return (value.encode().lower(),)
    if type(value) is int and value >= 0:
        markers: list[bytes] = []
        for width in (4, 8):
            try:
                markers.extend(
                    value.to_bytes(width, byteorder=byteorder)
                    for byteorder in ('big', 'little')
                )
            except OverflowError:
                continue
        return tuple(markers)
    return ()


def _assert_checkpoint_payloads_are_private(
    payloads: Sequence[object],
    *,
    forbidden_values: Sequence[object],
) -> None:
    forbidden_binary_values = tuple(
        marker
        for value in forbidden_values
        for marker in _binary_value_markers(value)
        if marker
    )
    forbidden_key_markers = tuple(
        key.encode() for key in _FORBIDDEN_CHECKPOINT_KEYS
    )

    def inspect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str):
                    assert key.lower() not in _FORBIDDEN_CHECKPOINT_KEYS
                elif isinstance(key, (bytes, memoryview)):
                    key_bytes = (
                        key.tobytes()
                        if isinstance(key, memoryview)
                        else key
                    ).lower()
                    assert key_bytes not in forbidden_key_markers
                inspect(key)
                inspect(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field_info in fields(value):
                assert field_info.name.lower() not in _FORBIDDEN_CHECKPOINT_KEYS
                inspect(getattr(value, field_info.name))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
            return
        if isinstance(value, Interrupt):
            inspect(value.value)
            return

        assert all(value != forbidden for forbidden in forbidden_values)
        if isinstance(value, (bytes, memoryview)):
            payload_bytes = (
                value.tobytes()
                if isinstance(value, memoryview)
                else value
            ).lower()
            for key_marker in forbidden_key_markers:
                assert key_marker not in payload_bytes
            for forbidden_marker in forbidden_binary_values:
                assert forbidden_marker not in payload_bytes
        elif isinstance(value, str):
            normalized = value.lower()
            for forbidden in forbidden_values:
                if isinstance(forbidden, str):
                    assert forbidden.lower() not in normalized

    for payload in payloads:
        inspect(payload)


def _checkpoint_payloads(
    harness: _PostgresHarness,
    app: _AppResources,
    checkpoint_thread_id: str,
) -> list[object]:
    saver = app.runtime.saver
    assert saver is not None
    saved_tuples = list(saver.list(checkpoint_config(checkpoint_thread_id)))
    assert saved_tuples
    payloads: list[object] = list(saved_tuples)
    with harness.engine.connect() as connection:
        for table_name in (
            'checkpoints',
            'checkpoint_blobs',
            'checkpoint_writes',
        ):
            payloads.extend(
                connection.execute(
                    text(
                        f'SELECT * FROM {table_name} '
                        'WHERE thread_id = :thread_id'
                    ),
                    {'thread_id': checkpoint_thread_id},
                ).mappings().all()
            )
    return payloads


def test_postgres_interrupt_survives_pool_and_app_restart_then_resumes_same_thread(
    postgres_review_harness: _PostgresHarness,
) -> None:
    harness = postgres_review_harness
    request = harness.seed_source_request(client_request_id=uuid4().hex)
    app_a = harness.new_app()
    paused = app_a.service.start(actor=USERS['admin'], request=request)
    harness.remember_thread(paused.thread_id)

    assert paused.status == 'awaiting_human_review'
    assert paused.checkpoint_resumable is True
    assert paused.durable is True
    before_counts = _business_counts(harness, paused.thread_id)
    _approve_only_item(harness, paused.thread_id)
    ready = app_a.service.status(
        actor=USERS['admin'],
        thread_id=paused.thread_id,
    )
    assert ready.status == 'awaiting_human_review'
    assert ready.checkpoint_resumable is True
    assert ready.review_resolution_ready is True
    assert ready.resume_allowed is True
    with harness.session_factory() as db:
        before_thread = db.get(AgentWorkflowThread, paused.thread_id)
        assert before_thread is not None
        checkpoint_thread_id = before_thread.checkpoint_thread_id
        state_version = before_thread.state_version
        db.rollback()
    app_a_engine = app_a.application_engine
    app_a_pool = app_a_engine.pool
    app_a.close()
    assert app_a.closed is True
    assert app_a_engine.pool is not app_a_pool

    app_b = harness.new_app()
    assert app_b.application_engine is not app_a_engine
    assert app_b.application_engine.pool is not app_a_pool

    def forbidden_path(*_args: object, **_kwargs: object) -> None:
        pytest.fail('normal approval must not rotate, repair, or fail checkpoint')

    app_b.service._mark_checkpoint_failed = forbidden_path  # type: ignore[method-assign]
    app_b.service._repair_checkpoint = forbidden_path  # type: ignore[method-assign]
    app_b.service._rotate_checkpoint_attempt = forbidden_path  # type: ignore[method-assign]
    resumed = app_b.service.resume(
        actor=USERS['admin'],
        thread_id=paused.thread_id,
    )

    assert resumed.thread_id == paused.thread_id
    assert resumed.status == 'completed'
    assert resumed.review_status_counts == {'approved': 1}
    assert _business_counts(harness, paused.thread_id) == (
        before_counts[0],
        before_counts[1],
        1,
        1,
    )
    with harness.session_factory() as db:
        thread = db.get(AgentWorkflowThread, paused.thread_id)
        assert thread is not None
        assert thread.checkpoint_thread_id == checkpoint_thread_id
        assert thread.state_version == state_version + 2
        saver = app_b.runtime.saver
        assert saver is not None
        saved = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
        assert saved is not None
        assert saved.config['configurable'].get('checkpoint_ns', '') == ''
        db.rollback()


def test_postgres_saved_tuple_contains_no_source_or_model_content(
    postgres_review_harness: _PostgresHarness,
) -> None:
    harness = postgres_review_harness
    request = harness.seed_source_request(client_request_id=uuid4().hex)
    app = harness.new_app()
    paused = app.service.start(actor=USERS['admin'], request=request)
    harness.remember_thread(paused.thread_id)
    with harness.session_factory() as db:
        thread = db.get(AgentWorkflowThread, paused.thread_id)
        item = db.scalars(
            select(ReviewItem).where(
                ReviewItem.workflow_thread_id == paused.thread_id
            )
        ).one()
        assert thread is not None
        evidence_ref = db.scalars(
            select(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == paused.thread_id
            )
        ).one()
        source = db.get(Source, evidence_ref.canonical_row_id)
        assert source is not None
        checkpoint_thread_id = thread.checkpoint_thread_id
        assert item.payload['model_output'] in _SENSITIVE_MARKERS
        forbidden_values = (
            item.id,
            str(item.id),
            source.source_id,
            source.source_url,
            source.title,
            source.raw_metadata['content_signature'],
            source.raw_metadata['external_revision'],
            *item.source_links,
            *item.source_snippets,
            *item.payload['source_ids'],
            item.payload['prompt'],
            item.payload['model_output'],
            item.payload['provider_error'],
            item.payload['api_key'],
        )
        db.rollback()

    _assert_checkpoint_payloads_are_private(
        _checkpoint_payloads(harness, app, checkpoint_thread_id),
        forbidden_values=forbidden_values,
    )


def test_checkpoint_commit_then_status_failure_reconciles_without_duplicate_effects(
    postgres_review_harness: _PostgresHarness,
) -> None:
    harness = postgres_review_harness
    request = harness.seed_source_request(client_request_id=uuid4().hex)
    app_a = harness.new_app()
    app_a.service._finish_initial_checkpoint = (  # type: ignore[method-assign]
        lambda **_kwargs: False
    )

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        app_a.service.start(actor=USERS['admin'], request=request)
    assert exc_info.value.code == 'checkpoint_failed'
    with harness.session_factory() as db:
        thread = db.scalars(
            select(AgentWorkflowThread).where(
                AgentWorkflowThread.security_scope_id == harness.scope_id
            )
        ).one()
        workflow_thread_id = thread.thread_id
        checkpoint_thread_id = thread.checkpoint_thread_id
        assert thread.status == 'checkpoint_pending'
        db.rollback()
    harness.checkpoint_thread_ids.add(checkpoint_thread_id)
    before_counts = _business_counts(harness, workflow_thread_id)
    app_a.close()

    app_b = harness.new_app()
    reconciled = app_b.service.status(
        actor=USERS['admin'],
        thread_id=workflow_thread_id,
    )

    assert reconciled.status == 'awaiting_human_review'
    assert reconciled.checkpoint_resumable is True
    assert _business_counts(harness, workflow_thread_id) == before_counts
    assert app_b.draft_service.draft_calls == 0


def test_draft_commit_then_checkpoint_failure_repairs_without_duplicate_effects(
    postgres_review_harness: _PostgresHarness,
) -> None:
    harness = postgres_review_harness
    request = harness.seed_source_request(client_request_id=uuid4().hex)
    app_a = harness.new_app(exploding_graph=True)

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        app_a.service.start(actor=USERS['admin'], request=request)
    assert exc_info.value.code == 'checkpoint_failed'
    with harness.session_factory() as db:
        thread = db.scalars(
            select(AgentWorkflowThread).where(
                AgentWorkflowThread.security_scope_id == harness.scope_id
            )
        ).one()
        workflow_thread_id = thread.thread_id
        first_checkpoint_thread_id = thread.checkpoint_thread_id
        assert thread.status == 'checkpoint_failed'
        db.rollback()
    harness.checkpoint_thread_ids.add(first_checkpoint_thread_id)
    before_counts = _business_counts(harness, workflow_thread_id)
    app_a.close()

    app_b = harness.new_app()
    repaired = app_b.service.resume(
        actor=USERS['admin'],
        thread_id=workflow_thread_id,
    )

    assert repaired.status == 'awaiting_human_review'
    assert repaired.checkpoint_resumable is True
    assert _business_counts(harness, workflow_thread_id) == before_counts
    assert app_b.draft_service.draft_calls == 0
    with harness.session_factory() as db:
        thread = db.get(AgentWorkflowThread, workflow_thread_id)
        assert thread is not None
        assert thread.checkpoint_thread_id != first_checkpoint_thread_id
        harness.checkpoint_thread_ids.add(thread.checkpoint_thread_id)
        db.rollback()


def test_concurrent_exact_batch_launches_create_one_thread(
    postgres_review_harness: _PostgresHarness,
) -> None:
    harness = postgres_review_harness
    base_request = harness.seed_source_request()
    requests = (
        base_request.model_copy(
            update={'client_request_id': f'creator-transport:{uuid4().hex}'}
        ),
        base_request.model_copy(
            update={'client_request_id': f'reuser-transport:{uuid4().hex}'}
        ),
    )
    actors = _shared_launch_actors()
    app_a = harness.new_app()
    app_b = harness.new_app()
    barrier = Barrier(2)

    def launch(
        app: _AppResources,
        request: ReviewWorkflowRunRequest,
        actor: DemoUser,
    ) -> str:
        barrier.wait(timeout=10)
        return app.service.start(
            actor=actor,
            request=request,
        ).thread_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(launch, app_a, requests[0], actors[0]),
            executor.submit(launch, app_b, requests[1], actors[1]),
        )
        thread_ids = tuple(future.result(timeout=30) for future in futures)

    assert thread_ids[0] == thread_ids[1]
    harness.remember_thread(thread_ids[0])
    with harness.session_factory() as db:
        threads = tuple(
            db.scalars(
                select(AgentWorkflowThread).where(
                    AgentWorkflowThread.security_scope_id == harness.scope_id
                )
            ).all()
        )
        assert len(threads) == 1
        thread = threads[0]
        stored_request = db.get(AgentWorkflowRequest, thread.thread_id)
        evidence_refs = tuple(
            db.scalars(
                select(AgentWorkflowEvidenceRef).where(
                    AgentWorkflowEvidenceRef.workflow_thread_id
                    == thread.thread_id
                )
            ).all()
        )
        assert stored_request is not None
        assert len(evidence_refs) == 1
        creator_index = next(
            index
            for index, actor in enumerate(actors)
            if actor.id == thread.owner_subject_id
        )
        reuser_index = 1 - creator_index
        assert thread.client_request_id == requests[creator_index].client_request_id
        persisted_values = tuple(
            getattr(row, column.name)
            for row in (thread, stored_request, *evidence_refs)
            for column in row.__table__.columns
        )
        persisted_text = repr(persisted_values).lower()
        assert actors[reuser_index].id not in persisted_values
        assert requests[reuser_index].client_request_id not in persisted_values
        for actor in actors:
            for caller_alias in (
                actor.email,
                actor.name,
                actor.title,
                actor.department,
                *actor.aliases,
            ):
                assert caller_alias.lower() not in persisted_text
        persisted_owner = thread.owner_subject_id
        persisted_client_request_id = thread.client_request_id
        db.rollback()
    for app, actor in zip((app_a, app_b), actors, strict=True):
        status = app.service.status(actor=actor, thread_id=thread_ids[0])
        assert status.thread_id == thread_ids[0]
    with harness.session_factory() as db:
        stable_thread = db.get(AgentWorkflowThread, thread_ids[0])
        assert stable_thread is not None
        assert (
            stable_thread.owner_subject_id,
            stable_thread.client_request_id,
        ) == (
            persisted_owner,
            persisted_client_request_id,
        )
        thread_count = db.scalar(
            select(func.count()).select_from(AgentWorkflowThread).where(
                AgentWorkflowThread.security_scope_id == harness.scope_id
            )
        )
        db.rollback()
    assert thread_count == 1
    assert _business_counts(harness, thread_ids[0])[:2] == (1, 1)
    assert app_a.draft_service.draft_calls + app_b.draft_service.draft_calls == 1


def test_concurrent_resume_has_one_state_version_winner(
    postgres_review_harness: _PostgresHarness,
) -> None:
    harness = postgres_review_harness
    request = harness.seed_source_request(client_request_id=uuid4().hex)
    app_a = harness.new_app()
    paused = app_a.service.start(actor=USERS['admin'], request=request)
    harness.remember_thread(paused.thread_id)
    _approve_only_item(harness, paused.thread_id)
    app_b = harness.new_app()
    barrier = Barrier(2)

    def gate_first_resume_projection(service: ReviewWorkflowService) -> None:
        original = service._read_projection
        waited = False

        def wrapped(
            *,
            actor: DemoUser,
            thread_id: str,
            action: str | None = None,
            require_current_versions: bool = False,
        ):
            nonlocal waited
            projection = original(
                actor=actor,
                thread_id=thread_id,
                action=action,
                require_current_versions=require_current_versions,
            )
            if (
                not waited
                and action == 'resume'
                and projection.status == 'awaiting_human_review'
            ):
                waited = True
                barrier.wait(timeout=10)
            return projection

        service._read_projection = wrapped  # type: ignore[method-assign]

    gate_first_resume_projection(app_a.service)
    gate_first_resume_projection(app_b.service)
    with harness.session_factory() as db:
        before = db.get(AgentWorkflowThread, paused.thread_id)
        assert before is not None
        before_version = before.state_version
        checkpoint_thread_id = before.checkpoint_thread_id
        db.rollback()

    def resume(app: _AppResources) -> tuple[str, object]:
        try:
            result = app.service.resume(
                actor=USERS['admin'],
                thread_id=paused.thread_id,
            )
            return ('success', result)
        except ReviewWorkflowServiceError as exc:
            return ('error', exc.code)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(resume, app_a),
            executor.submit(resume, app_b),
        )
        outcomes = tuple(future.result(timeout=30) for future in futures)

    assert sorted(kind for kind, _ in outcomes) == ['error', 'success']
    assert [value for kind, value in outcomes if kind == 'error'] == [
        'concurrent_resume'
    ]
    completed = [value for kind, value in outcomes if kind == 'success'][0]
    assert completed.status == 'completed'
    with harness.session_factory() as db:
        thread = db.get(AgentWorkflowThread, paused.thread_id)
        assert thread is not None
        assert thread.status == 'completed'
        assert thread.state_version == before_version + 2
        assert thread.checkpoint_thread_id == checkpoint_thread_id
        db.rollback()
    assert _business_counts(harness, paused.thread_id) == (1, 1, 1, 1)


def test_checkpoint_privacy_scan_rejects_forbidden_keys_and_binary_values() -> None:
    forbidden_value = b'private-binary-source-marker'
    payloads = [
        {'source_snippet': ''},
        {'opaque': memoryview(forbidden_value)},
    ]

    with pytest.raises(AssertionError):
        _assert_checkpoint_payloads_are_private(
            payloads,
            forbidden_values=(),
        )
    with pytest.raises(AssertionError):
        _assert_checkpoint_payloads_are_private(
            [{'opaque': memoryview(forbidden_value)}],
            forbidden_values=(forbidden_value,),
        )


def test_shared_launch_actors_are_distinct_authorized_callers() -> None:
    creator, reuser = _shared_launch_actors()

    assert creator.id != reuser.id
    assert creator.email != reuser.email
    assert 'internal' in creator.permission_levels
    assert 'internal' in reuser.permission_levels
    assert creator.id not in reuser.aliases
    assert reuser.id not in creator.aliases


def test_cleanup_runs_checkpoint_and_dispose_after_application_cleanup_failure() -> None:
    events: list[str] = []

    class FailingSessionContext:
        def __enter__(self):
            events.append('application_cleanup')
            raise RuntimeError('application cleanup failed')

        def __exit__(self, *_args: object) -> None:
            events.append('application_exit')

    class CheckpointConnection:
        def scalar(self, *_args: object, **_kwargs: object) -> None:
            return None

    class CheckpointContext:
        def __enter__(self) -> CheckpointConnection:
            events.append('checkpoint_cleanup')
            return CheckpointConnection()

        def __exit__(self, *_args: object) -> None:
            events.append('checkpoint_exit')

    class FakeEngine:
        def begin(self) -> CheckpointContext:
            return CheckpointContext()

        def dispose(self) -> None:
            events.append('engine_dispose')

    harness = _PostgresHarness(
        database_url='postgresql+psycopg://unused-test-target',
        engine=FakeEngine(),  # type: ignore[arg-type]
        session_factory=lambda: FailingSessionContext(),  # type: ignore[arg-type]
        settings=object(),  # type: ignore[arg-type]
        scope_id='cleanup-test-scope',
    )

    with pytest.raises(RuntimeError, match='^application cleanup failed$'):
        harness.cleanup()

    assert events == [
        'application_cleanup',
        'checkpoint_cleanup',
        'checkpoint_exit',
        'engine_dispose',
    ]
