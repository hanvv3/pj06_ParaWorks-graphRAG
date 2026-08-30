from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.checkpointing import CheckpointRuntime
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    register_company_memory_review_versions,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_service import ReviewWorkflowServiceError
from backend.app.agent_runtime.review_v21_service import ReviewV21Service
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AuditLog,
    ReviewItem,
)


def _actor() -> DemoUser:
    return DemoUser(
        id='owner-1',
        email='owner@example.test',
        role='admin',
        permission_levels={'public', 'internal'},
        name='Owner',
        title='Admin',
        department='Engineering',
    )


@dataclass
class _Launch:
    thread_id: str

    def start(self, **_kwargs):
        return SimpleNamespace(
            thread_id=self.thread_id,
            graph_version='company-memory-review-v2.1-auto-review',
            status='created',
        )

    def dry_run(self, **_kwargs):
        return 'preview'

    def diagnostic(self):
        return SimpleNamespace(model_copy=lambda **_kwargs: 'diagnostic')


@dataclass
class _Extraction:
    factory: sessionmaker

    def draft(self, *, workflow_thread_id: str, **_kwargs) -> None:
        with self.factory() as db, db.begin():
            db.add(ReviewItem(
                item_type='history_event',
                payload={'title': 'candidate'},
                source_links=['https://example.test/source'],
                source_snippets=['evidence'],
                confidence_score=0.99,
                permission_level='internal',
                status='pending_review',
                workflow_thread_id=workflow_thread_id,
                candidate_key='candidate-1',
            ))


@dataclass
class _Validation:
    factory: sessionmaker
    transition: str | None

    def run_auto_review(self, *, workflow_thread_id: str, **_kwargs) -> None:
        if self.transition is None:
            return
        with self.factory() as db, db.begin():
            item = db.scalar(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == workflow_thread_id
            ))
            assert item is not None
            item.status = self.transition
            if self.transition == 'approved':
                item.resolution_source = 'auto_policy'


def _service(
    db_session: Session, *, transition: str | None
) -> tuple[ReviewV21Service, sessionmaker, str, CheckpointRuntime]:
    db_session.commit()
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    thread_id = uuid4().hex
    with factory() as db, db.begin():
        db.add(AgentWorkflowThread(
            thread_id=thread_id,
            workflow_name='company-memory-review',
            graph_version='company-memory-review-v2.1-auto-review',
            checkpoint_thread_id=f'v21:{uuid4().hex}',
            checkpoint_store='memory',
            owner_subject_id='owner-1',
            security_scope_id='default',
            input_hash='a' * 64,
            evidence_version_hash='b' * 64,
            status='created',
            state_version=7,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ))
        db.add(AgentWorkflowRequest(
            workflow_thread_id=thread_id,
            input_schema_version='review-source-versions:v1',
            agent_names=['history_agent'],
            selection_policy_version='company-memory-review-selection:v1',
            input_hash='a' * 64,
            fingerprint_key_version='test-v1',
            auto_review_mode='enforce',
            auto_review_policy_version='auto-review-policy:v1',
            auto_review_enforce_percentage=10,
        ))
        db.add(AgentWorkflowEvidenceRef(
            workflow_thread_id=thread_id,
            ordinal=0,
            canonical_source_type='gmail',
            canonical_table='sources',
            canonical_row_id=1,
            external_revision='rev-1',
            content_signature='sig-1',
            permission_level_snapshot='internal',
            content_fingerprint='c' * 64,
        ))
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=True,
        database_url='sqlite+pysqlite:///:memory:',
        langgraph_review_v2_enabled=True,
        auto_review_mode='disabled',
        agent_runtime_fingerprint_secret='v21-service-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='v21-test-v1',
    )
    runtime = CheckpointRuntime(settings)
    runtime.start()
    registry = GraphVersionRegistry()
    register_company_memory_review_versions(registry)
    agents = AgentRegistry()
    agents.register(AgentManifest(
        name='history_agent',
        owner='developer-c',
        input_contract='EvidencePacket',
        output_contract='ReviewCandidate',
        prompt_versions=('history-v1',),
        supported_permissions=('public', 'internal'),
        capabilities=('history_candidate',),
    ))
    service = ReviewV21Service(
        launch_service=_Launch(thread_id),
        session_factory=factory,
        settings=settings,
        checkpoint_runtime=runtime,
        graph_registry=registry,
        agent_registry=agents,
        extraction_coordinator=_Extraction(factory),
        validation_coordinator=_Validation(factory, transition),
    )
    return service, factory, thread_id, runtime


def test_v21_service_all_auto_resolved_completes_without_interrupt(
    db_session: Session,
) -> None:
    service, _, thread_id, runtime = _service(
        db_session, transition='approved'
    )
    try:
        status = service.start(actor=_actor(), request=object())
    finally:
        runtime.close()

    assert status.thread_id == thread_id
    assert status.status == 'completed'
    assert status.auto_approved_count == 1
    assert status.review_status_counts['revoked'] == 0


def test_stored_v21_global_disabled_is_zero_call_human_interrupt(
    db_session: Session,
) -> None:
    service, _, _, runtime = _service(db_session, transition=None)
    try:
        status = service.start(actor=_actor(), request=object())
    finally:
        runtime.close()

    assert status.status == 'awaiting_human_review'
    assert status.human_review_required_count == 1
    assert status.checkpoint_resumable is True


def test_v21_resume_accepts_human_pending_to_terminal_resolution(
    db_session: Session,
) -> None:
    service, factory, thread_id, runtime = _service(
        db_session, transition=None
    )
    try:
        paused = service.start(actor=_actor(), request=object())
        assert paused.status == 'awaiting_human_review'
        with factory() as db, db.begin():
            item = db.scalar(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread_id
            ))
            assert item is not None
            item.status = 'rejected'
            item.resolution_source = 'human'
            item.reviewer_id = 'owner-1'
            item.reviewed_at = datetime.now(UTC)
            db.add(AuditLog(
                actor_id='owner-1',
                actor_email='owner@example.test',
                actor_role='admin',
                action='review.reject',
                target_type='review_item',
                target_id=str(item.id),
                status='success',
                metadata_={'outcome': 'rejected'},
            ))

        completed = service.resume(actor=_actor(), thread_id=thread_id)
    finally:
        runtime.close()

    assert completed.status == 'completed'
    assert completed.review_status_counts['rejected'] == 1


def test_v21_resume_rejects_unverified_live_count_drift(
    db_session: Session,
) -> None:
    service, factory, thread_id, runtime = _service(
        db_session, transition=None
    )
    try:
        service.start(actor=_actor(), request=object())
        with factory() as db, db.begin():
            item = db.scalar(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread_id
            ))
            assert item is not None
            item.status = 'approved'
            item.resolution_source = 'human'

        try:
            service.resume(actor=_actor(), thread_id=thread_id)
        except ReviewWorkflowServiceError as exc:
            assert exc.code == 'invalid_state_transition'
        else:
            raise AssertionError('unverified drift must fail closed')
    finally:
        runtime.close()
