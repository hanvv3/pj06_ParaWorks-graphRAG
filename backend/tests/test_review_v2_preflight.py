from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

import backend.app.models  # noqa: F401
from backend.app.agent_runtime.canonical_sources import ReviewWorkflowPreflightError
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_preflight import (
    create_or_reuse_review_thread,
    prepare_review_request,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.db.base import Base
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.source import Source
from backend.app.schemas.review_workflow import ReviewWorkflowRunRequest


def _actor(subject_id: str) -> DemoUser:
    return DemoUser(
        id=subject_id,
        email=f'{subject_id}@example.test',
        role='employee',
        permission_levels={'public', 'internal'},
        name=subject_id,
        title='Tester',
        department='Quality',
    )


def _registry(*names: str) -> AgentRegistry:
    registry = AgentRegistry()
    for name in names:
        registry.register(
            AgentManifest(
                name=name,
                owner='Test Owner',
                input_contract='PreparedReviewRequest',
                output_contract='AgentRunResult',
                prompt_versions=('test:v1',),
                supported_permissions=('public', 'internal', 'restricted'),
                capabilities=('review_draft',),
            )
        )
    return registry


def _settings(scope: str = 'scope-1') -> Settings:
    return Settings(
        agent_runtime_security_scope_id=scope,
        agent_runtime_fingerprint_secret='test-review-preflight-secret',
        agent_runtime_fingerprint_key_version='test-v1',
    )


def _seed_source(
    db: Session,
    sequence: int,
    *,
    permission_level: str = 'internal',
) -> Source:
    source = Source(
        source_type='gmail',
        source_id=f'gmail:message-{sequence}',
        source_url=f'https://sensitive.example/message-{sequence}',
        title=f'Sensitive message {sequence}',
        permission_level=permission_level,
        raw_metadata={
            'content_signature': f'gmail-signature-{sequence}',
            'source_snippet': f'sensitive snippet {sequence}',
        },
    )
    db.add(source)
    db.flush()
    return source


def _request(
    source: Source,
    *,
    client_request_id: str | None,
    agent_names: list[str] | None = None,
) -> ReviewWorkflowRunRequest:
    return ReviewWorkflowRunRequest(
        source_refs=[
            {
                'source_type': source.source_type,
                'source_id': source.source_id,
                'version_or_signature': source.raw_metadata['content_signature'],
            }
        ],
        agent_names=agent_names or ['mail_document_agent'],
        client_request_id=client_request_id,
    )


def _launch(
    db,
    *,
    request: ReviewWorkflowRunRequest,
    actor: DemoUser,
    registry: AgentRegistry,
    settings: Settings,
):
    prepared = prepare_review_request(
        db,
        request=request,
        actor=actor,
        registry=registry,
        settings=settings,
    )
    return prepared, create_or_reuse_review_thread(
        db,
        prepared=prepared,
        request=request,
        actor=actor,
        settings=settings,
    )


def test_same_scope_exact_batch_reuses_thread_across_authorized_owners(
    db_session,
) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    request_1 = _request(source, client_request_id='owner-1-retry')
    prepared_1, first = _launch(
        db_session,
        request=request_1,
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    request_2 = _request(source, client_request_id='owner-2-retry')
    prepared_2, second = _launch(
        db_session,
        request=request_2,
        actor=_actor('owner-2'),
        registry=registry,
        settings=settings,
    )

    assert prepared_1.input_hash == prepared_2.input_hash
    assert first.created is True
    assert first.shared_reuse is False
    assert second.created is False
    assert second.shared_reuse is True
    assert second.thread.thread_id == first.thread.thread_id
    assert second.thread.owner_subject_id == 'owner-1'
    assert db_session.scalar(select(func.count()).select_from(AgentWorkflowThread)) == 1

    stored_ref = db_session.scalar(select(AgentWorkflowEvidenceRef))
    assert stored_ref is not None
    assert 'source_url' not in stored_ref.__dict__
    assert 'source_snippet' not in stored_ref.__dict__
    assert source.source_id not in stored_ref.__dict__.values()


def test_cross_owner_shared_reuse_does_not_bind_caller_client_key(
    db_session,
) -> None:
    source_1 = _seed_source(db_session, 1)
    source_2 = _seed_source(db_session, 2)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()

    _, first = _launch(
        db_session,
        request=_request(source_1, client_request_id='creator-key'),
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    _, shared = _launch(
        db_session,
        request=_request(source_1, client_request_id='caller-key'),
        actor=_actor('owner-2'),
        registry=registry,
        settings=settings,
    )

    assert shared.thread.thread_id == first.thread.thread_id
    assert db_session.scalar(
        select(func.count())
        .select_from(AgentWorkflowThread)
        .where(
            AgentWorkflowThread.owner_subject_id == 'owner-2',
            AgentWorkflowThread.client_request_id == 'caller-key',
        )
    ) == 0

    _, independent = _launch(
        db_session,
        request=_request(source_2, client_request_id='caller-key'),
        actor=_actor('owner-2'),
        registry=registry,
        settings=settings,
    )
    assert independent.created is True
    assert independent.thread.thread_id != first.thread.thread_id


def test_same_owner_client_key_mismatch_wins_before_shared_lookup(
    db_session,
) -> None:
    source_1 = _seed_source(db_session, 1)
    source_2 = _seed_source(db_session, 2)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    _launch(
        db_session,
        request=_request(source_1, client_request_id='stable-key'),
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    _, existing_shared = _launch(
        db_session,
        request=_request(source_2, client_request_id='owner-2-key'),
        actor=_actor('owner-2'),
        registry=registry,
        settings=settings,
    )

    mismatched_request = _request(source_2, client_request_id='stable-key')
    mismatched_prepared = prepare_review_request(
        db_session,
        request=mismatched_request,
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        create_or_reuse_review_thread(
            db_session,
            prepared=mismatched_prepared,
            request=mismatched_request,
            actor=_actor('owner-1'),
            settings=settings,
        )

    assert exc_info.value.code == 'idempotency_key_reused'
    assert existing_shared.thread.owner_subject_id == 'owner-2'
    assert db_session.scalar(select(func.count()).select_from(AgentWorkflowThread)) == 2


def test_shared_lookup_compares_the_stored_evidence_ordinal(db_session) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    _, first = _launch(
        db_session,
        request=_request(source, client_request_id='creator-key'),
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    stored_ref = db_session.scalar(
        select(AgentWorkflowEvidenceRef).where(
            AgentWorkflowEvidenceRef.workflow_thread_id == first.thread.thread_id
        )
    )
    assert stored_ref is not None
    stored_ref.ordinal = 1
    db_session.commit()

    _, second = _launch(
        db_session,
        request=_request(source, client_request_id='owner-2-key'),
        actor=_actor('owner-2'),
        registry=registry,
        settings=settings,
    )

    assert second.created is True
    assert second.thread.thread_id != first.thread.thread_id


def test_different_scope_creates_independent_thread(db_session) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    actor = _actor('owner-1')
    request = _request(source, client_request_id='same-transport-key')

    prepared_1, first = _launch(
        db_session,
        request=request,
        actor=actor,
        registry=registry,
        settings=_settings('scope-1'),
    )
    prepared_2, second = _launch(
        db_session,
        request=request,
        actor=actor,
        registry=registry,
        settings=_settings('scope-2'),
    )

    assert first.created is True
    assert second.created is True
    assert first.thread.thread_id != second.thread.thread_id
    assert prepared_1.input_hash != prepared_2.input_hash


@pytest.mark.parametrize('terminal_status', ['cancelled', 'failed'])
def test_cancelled_and_failed_zero_effect_threads_remain_stable_batch_owners(
    db_session,
    terminal_status: str,
) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    request = _request(source, client_request_id='creator-key')
    _, first = _launch(
        db_session,
        request=request,
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    first.thread.status = terminal_status
    db_session.commit()

    _, replay = _launch(
        db_session,
        request=_request(source, client_request_id='other-owner-key'),
        actor=_actor('owner-2'),
        registry=registry,
        settings=settings,
    )

    assert replay.created is False
    assert replay.shared_reuse is True
    assert replay.thread.thread_id == first.thread.thread_id
    assert replay.thread.status == terminal_status
    assert db_session.scalar(select(func.count()).select_from(AgentWorkflowThread)) == 1


def test_group_alias_and_unregistered_manifest_are_rejected(db_session) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()

    with pytest.raises(ValidationError):
        _request(
            source,
            client_request_id=None,
            agent_names=['memory_extraction_agent'],
        )

    request = _request(
        source,
        client_request_id=None,
        agent_names=['timeline_agent'],
    )
    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        prepare_review_request(
            db_session,
            request=request,
            actor=_actor('owner-1'),
            registry=_registry('mail_document_agent'),
            settings=_settings(),
        )
    assert exc_info.value.code == 'invalid_input'


class _PostgresDialect:
    name = 'postgresql'


class _PostgresBind:
    dialect = _PostgresDialect()


class FakePostgresSession:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.events: list[tuple[str, object]] = []
        self.advisory_lock_seen = False

    def get_bind(self):
        return _PostgresBind()

    def execute(self, statement, params=None):
        sql = str(statement)
        if sql == 'SELECT pg_advisory_xact_lock(:key)':
            self.events.append(('advisory_lock', (sql, params)))
            self.advisory_lock_seen = True
            return None
        self.events.append(('execute', self.advisory_lock_seen))
        return self.session.execute(statement, params or {})

    def scalars(self, statement):
        self.events.append(('lookup', self.advisory_lock_seen))
        return self.session.scalars(statement)

    def scalar(self, statement):
        self.events.append(('lookup', self.advisory_lock_seen))
        return self.session.scalar(statement)

    def add_all(self, instances) -> None:
        self.events.append(('create', self.advisory_lock_seen))
        self.session.add_all(instances)

    def flush(self) -> None:
        self.events.append(('flush', self.advisory_lock_seen))
        self.session.flush()

    def commit(self) -> None:
        self.events.append(('commit', self.advisory_lock_seen))
        self.session.commit()

    def rollback(self) -> None:
        self.events.append(('rollback', self.advisory_lock_seen))
        self.session.rollback()


@pytest.fixture
def fake_postgres_session(db_session) -> FakePostgresSession:
    return FakePostgresSession(db_session)


def test_postgres_preflight_uses_transaction_scoped_advisory_lock(
    fake_postgres_session,
) -> None:
    source = _seed_source(fake_postgres_session.session, 1)
    fake_postgres_session.session.commit()
    request = _request(source, client_request_id='postgres-key')
    prepared = prepare_review_request(
        fake_postgres_session.session,
        request=request,
        actor=_actor('owner-1'),
        registry=_registry('mail_document_agent'),
        settings=_settings(),
    )

    result = create_or_reuse_review_thread(
        fake_postgres_session,
        prepared=prepared,
        request=request,
        actor=_actor('owner-1'),
        settings=_settings(),
    )

    assert result.created is True
    lock_events = [event for event in fake_postgres_session.events if event[0] == 'advisory_lock']
    assert len(lock_events) == 1
    sql, params = lock_events[0][1]
    assert sql == 'SELECT pg_advisory_xact_lock(:key)'
    assert params == {'key': int(params['key'])}
    assert any(event == ('lookup', False) for event in fake_postgres_session.events)
    assert any(event == ('lookup', True) for event in fake_postgres_session.events)
    assert ('create', True) in fake_postgres_session.events
    assert ('commit', True) in fake_postgres_session.events


def test_sqlite_preflight_process_lock_serializes_exact_batch(tmp_path) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'preflight.sqlite3'}",
        connect_args={'check_same_thread': False, 'timeout': 10},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_local() as seed_session:
        source = _seed_source(seed_session, 1)
        source_id = source.source_id
        signature = source.raw_metadata['content_signature']
        seed_session.commit()

    barrier = Barrier(2)
    registry = _registry('mail_document_agent')
    settings = _settings()

    def launch(owner: str) -> tuple[str, bool]:
        with session_local() as db:
            request = ReviewWorkflowRunRequest(
                source_refs=[
                    {
                        'source_type': 'gmail',
                        'source_id': source_id,
                        'version_or_signature': signature,
                    }
                ],
                agent_names=['mail_document_agent'],
                client_request_id=f'{owner}-key',
            )
            prepared = prepare_review_request(
                db,
                request=request,
                actor=_actor(owner),
                registry=registry,
                settings=settings,
            )
            barrier.wait()
            result = create_or_reuse_review_thread(
                db,
                prepared=prepared,
                request=request,
                actor=_actor(owner),
                settings=settings,
            )
            return result.thread.thread_id, result.created

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(launch, ('owner-1', 'owner-2')))

    with session_local() as db:
        assert db.scalar(select(func.count()).select_from(AgentWorkflowThread)) == 1
        assert db.scalar(select(func.count()).select_from(AgentWorkflowRequest)) == 1
        assert db.scalar(select(func.count()).select_from(AgentWorkflowEvidenceRef)) == 1
    assert len({thread_id for thread_id, _ in results}) == 1
    assert sorted(created for _, created in results) == [False, True]
