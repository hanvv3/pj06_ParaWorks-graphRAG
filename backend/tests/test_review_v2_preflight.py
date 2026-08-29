from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

import backend.app.models  # noqa: F401
from backend.app.agent_runtime.canonical_sources import ReviewWorkflowPreflightError
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_preflight import (
    INPUT_HASH_POLICY,
    advisory_key_from_hmac,
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
from backend.app.models.source import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    ReviewWorkflowRunRequest,
)


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
    server_signature = f'{sequence:064x}'
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
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=server_signature,
    )
    db.add(source)
    db.flush()
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v1',
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body=f'sensitive body {sequence}',
    )
    db.add(version)
    db.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id=f'revision-{sequence}',
        content_signature=server_signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=server_signature,
        parser_policy_version='server-source-parser-policy:v1',
        parser_version='source-event-paragraph-parser:v1',
        chunk_policy_version='paragraph-chunks:1200:v1',
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text=f'sensitive body {sequence}',
            source_snippet=f'sensitive snippet {sequence}',
            permission_level=permission_level,
            metadata_={},
        )
    )
    document.current_document_version_id = version.id
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
                'version_or_signature': source.server_content_signature,
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


def test_same_owner_exact_client_key_replays_and_closes_transaction(
    db_session,
) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    actor = _actor('owner-1')
    request = _request(source, client_request_id='same-owner-key')
    _, first = _launch(
        db_session,
        request=request,
        actor=actor,
        registry=registry,
        settings=settings,
    )

    prepared = prepare_review_request(
        db_session,
        request=request,
        actor=actor,
        registry=registry,
        settings=settings,
    )
    replay = create_or_reuse_review_thread(
        db_session,
        prepared=prepared,
        request=request,
        actor=actor,
        settings=settings,
    )

    assert db_session.in_transaction() is False
    assert replay.created is False
    assert replay.shared_reuse is False
    assert replay.thread.thread_id == first.thread.thread_id


def test_v20_prepared_identity_is_unchanged(
    db_session,
) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    request_a = _request(source, client_request_id='client-a')
    request_b = _request(source, client_request_id='client-b')

    prepared_a = prepare_review_request(
        db_session,
        request=request_a,
        actor=_actor('owner-a'),
        registry=registry,
        settings=settings,
    )
    prepared_other_client = prepare_review_request(
        db_session,
        request=request_b,
        actor=_actor('owner-a'),
        registry=registry,
        settings=settings,
    )
    prepared_other_owner = prepare_review_request(
        db_session,
        request=request_a,
        actor=_actor('owner-b'),
        registry=registry,
        settings=settings,
    )
    secret, _ = fingerprint_secret_bytes(settings)
    expected = keyed_fingerprint(
        {
            'security_scope_id': 'scope-1',
            'workflow_name': COMPANY_MEMORY_REVIEW_WORKFLOW,
            'graph_version': COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            'evidence_version_hash': prepared_a.evidence_version_hash,
            'agent_names': ['mail_document_agent'],
            'selection_policy_version': COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        },
        secret=secret,
        schema_version=COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
        policy_version=INPUT_HASH_POLICY,
    )

    assert prepared_a.input_hash == expected
    assert prepared_other_client.input_hash == expected
    assert prepared_other_owner.input_hash == expected


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
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(AgentWorkflowThread)
            .where(
                AgentWorkflowThread.owner_subject_id == 'owner-2',
                AgentWorkflowThread.client_request_id == 'caller-key',
            )
        )
        == 0
    )

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


@pytest.mark.parametrize(
    ('record_type', 'field', 'changed_value'),
    [
        ('request', 'input_schema_version', 'changed-schema-v1'),
        ('request', 'request_kind', 'changed_request_kind'),
        ('request', 'agent_names', ['timeline_agent']),
        ('request', 'selection_policy_version', 'changed-policy-v1'),
        ('request', 'input_hash', '1' * 64),
        ('request', 'fingerprint_key_version', 'changed-key-v1'),
        ('evidence', 'ordinal', 1),
        ('evidence', 'canonical_source_type', 'drive'),
        ('evidence', 'canonical_table', 'document_versions'),
        ('evidence', 'canonical_row_id', 999_999),
        ('evidence', 'document_version_id', 999_999),
        ('evidence', 'external_revision', 'changed-revision'),
        ('evidence', 'content_signature', 'changed-signature'),
        ('evidence', 'permission_level_snapshot', 'public'),
        ('evidence', 'content_fingerprint', '2' * 64),
    ],
)
def test_shared_lookup_compares_every_stored_request_and_evidence_identity_field(
    db_session,
    record_type: str,
    field: str,
    changed_value: object,
) -> None:
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
    model = (
        AgentWorkflowRequest if record_type == 'request' else AgentWorkflowEvidenceRef
    )
    stored_record = db_session.scalar(
        select(model).where(
            (
                AgentWorkflowRequest.workflow_thread_id
                if record_type == 'request'
                else AgentWorkflowEvidenceRef.workflow_thread_id
            )
            == first.thread.thread_id
        )
    )
    assert stored_record is not None
    setattr(stored_record, field, changed_value)
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


def test_permission_loss_between_prepare_and_shared_reuse_is_hidden(
    db_session,
) -> None:
    source = _seed_source(db_session, 1)
    db_session.commit()
    registry = _registry('mail_document_agent')
    settings = _settings()
    _launch(
        db_session,
        request=_request(source, client_request_id='creator-key'),
        actor=_actor('owner-1'),
        registry=registry,
        settings=settings,
    )
    request = _request(source, client_request_id='owner-2-key')
    actor = _actor('owner-2')
    prepared = prepare_review_request(
        db_session,
        request=request,
        actor=actor,
        registry=registry,
        settings=settings,
    )
    source.permission_level = 'restricted'
    db_session.commit()

    with pytest.raises(ReviewWorkflowPreflightError) as exc_info:
        create_or_reuse_review_thread(
            db_session,
            prepared=prepared,
            request=request,
            actor=actor,
            settings=settings,
        )

    assert exc_info.value.code == 'not_found'
    assert str(exc_info.value) == 'source reference was not found'
    assert db_session.in_transaction() is False


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


class FakeAdvisoryLockManager:
    def __init__(self) -> None:
        self._guard = Lock()
        self._locks: dict[int, Lock] = {}

    def acquire(self, key: int) -> Lock:
        with self._guard:
            lock = self._locks.setdefault(key, Lock())
        lock.acquire()
        return lock


class FakePostgresSession:
    def __init__(
        self,
        session: Session,
        *,
        lock_manager: FakeAdvisoryLockManager | None = None,
        prelock_barrier: Barrier | None = None,
    ) -> None:
        self.session = session
        self.events: list[tuple[int, str, int, object | None]] = []
        self.epoch = 0
        self._lock_manager = lock_manager or FakeAdvisoryLockManager()
        self._prelock_barrier = prelock_barrier
        self._locks: list[Lock] = []

    def get_bind(self):
        return _PostgresBind()

    def execute(self, statement, params=None):
        sql = str(statement)
        if sql == 'SELECT pg_advisory_xact_lock(:key)':
            key = params['key']
            self._locks.append(self._lock_manager.acquire(key))
            self.events.append(
                (self.epoch, 'advisory_lock', len(self._locks), (sql, params))
            )
            return None
        self.events.append((self.epoch, 'execute', len(self._locks), None))
        return self.session.execute(statement, params or {})

    def scalars(self, statement):
        self.events.append((self.epoch, 'lookup', len(self._locks), None))
        return self.session.scalars(statement)

    def scalar(self, statement):
        self.events.append((self.epoch, 'lookup', len(self._locks), None))
        return self.session.scalar(statement)

    def add_all(self, instances) -> None:
        self.events.append((self.epoch, 'create', len(self._locks), None))
        self.session.add_all(instances)

    def flush(self) -> None:
        self.events.append((self.epoch, 'flush', len(self._locks), None))
        self.session.flush()

    def commit(self) -> None:
        self.events.append((self.epoch, 'commit', len(self._locks), None))
        try:
            self.session.commit()
        finally:
            self._finish_transaction()

    def rollback(self) -> None:
        starting_epoch = self.epoch
        had_locks = bool(self._locks)
        self.events.append((self.epoch, 'rollback', len(self._locks), None))
        try:
            self.session.rollback()
        finally:
            self._finish_transaction()
        if starting_epoch == 0 and not had_locks and self._prelock_barrier is not None:
            self._prelock_barrier.wait()

    @property
    def active_lock_count(self) -> int:
        return len(self._locks)

    def _finish_transaction(self) -> None:
        while self._locks:
            self._locks.pop().release()
        self.epoch += 1


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
    assert [
        (epoch, event, lock_count)
        for epoch, event, lock_count, _ in fake_postgres_session.events
    ] == [
        (0, 'lookup', 0),
        (0, 'rollback', 0),
        (1, 'advisory_lock', 1),
        (1, 'advisory_lock', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'lookup', 2),
        (1, 'create', 2),
        (1, 'flush', 2),
        (1, 'commit', 2),
    ]
    lock_events = [
        detail
        for _, event, _, detail in fake_postgres_session.events
        if event == 'advisory_lock'
    ]
    assert all(
        sql == 'SELECT pg_advisory_xact_lock(:key)'
        and params == {'key': int(params['key'])}
        for sql, params in lock_events
    )
    assert [params['key'] for _, params in lock_events] == sorted(
        params['key'] for _, params in lock_events
    )
    assert advisory_key_from_hmac(prepared.input_hash) in {
        params['key'] for _, params in lock_events
    }
    assert fake_postgres_session.active_lock_count == 0
    assert fake_postgres_session.epoch == 2


def test_postgres_concurrent_same_creator_key_different_batches_is_typed_conflict(
    tmp_path,
) -> None:
    engine = create_engine(
        f'sqlite:///{tmp_path / "postgres-contract.sqlite3"}',
        connect_args={'check_same_thread': False, 'timeout': 10},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_local() as seed_session:
        sources = [_seed_source(seed_session, sequence) for sequence in (1, 2)]
        refs = [
            (
                source.source_type,
                source.source_id,
                source.server_content_signature,
            )
            for source in sources
        ]
        seed_session.commit()

    barrier = Barrier(2)
    lock_manager = FakeAdvisoryLockManager()
    registry = _registry('mail_document_agent')
    settings = _settings()
    actor = _actor('same-owner')

    def launch(ref: tuple[str, str, str]) -> str:
        with session_local() as session:
            request = ReviewWorkflowRunRequest(
                source_refs=[
                    {
                        'source_type': ref[0],
                        'source_id': ref[1],
                        'version_or_signature': ref[2],
                    }
                ],
                agent_names=['mail_document_agent'],
                client_request_id='same-client-key',
            )
            prepared = prepare_review_request(
                session,
                request=request,
                actor=actor,
                registry=registry,
                settings=settings,
            )
            fake = FakePostgresSession(
                session,
                lock_manager=lock_manager,
                prelock_barrier=barrier,
            )
            try:
                result = create_or_reuse_review_thread(
                    fake,
                    prepared=prepared,
                    request=request,
                    actor=actor,
                    settings=settings,
                )
            except ReviewWorkflowPreflightError as exc:
                return exc.code
            return 'created' if result.created else 'replayed'

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(launch, refs))

    assert sorted(outcomes) == ['created', 'idempotency_key_reused']
    with session_local() as db:
        assert db.scalar(select(func.count()).select_from(AgentWorkflowThread)) == 1


def test_sqlite_preflight_process_lock_serializes_exact_batch(tmp_path) -> None:
    engine = create_engine(
        f'sqlite:///{tmp_path / "preflight.sqlite3"}',
        connect_args={'check_same_thread': False, 'timeout': 10},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_local() as seed_session:
        source = _seed_source(seed_session, 1)
        source_id = source.source_id
        signature = source.server_content_signature
        assert signature is not None
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
        assert (
            db.scalar(select(func.count()).select_from(AgentWorkflowEvidenceRef)) == 1
        )
    assert len({thread_id for thread_id, _ in results}) == 1
    assert sorted(created for _, created in results) == [False, True]
