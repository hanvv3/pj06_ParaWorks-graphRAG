from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime import review_v2_service as lifecycle_module
from backend.app.agent_runtime.checkpoint_execution import checkpoint_config
from backend.app.agent_runtime.checkpointing import (
    CheckpointReadiness,
    CheckpointRuntime,
)
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    register_company_memory_review_v2,
)
from backend.app.agent_runtime.review_v2_agents import build_review_agent_catalog
from backend.app.agent_runtime.review_v2_drafting import (
    ReviewDraftError,
    ReviewDraftResult,
)
from backend.app.agent_runtime.review_v2_service import (
    ReviewWorkflowService,
    ReviewWorkflowServiceError,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models.agent_runs import AgentRun
from backend.app.models.agent_workflows import (
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.models.source import Source
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    ReviewWorkflowDryRunResponse,
    ReviewWorkflowRunRequest,
)


@pytest.fixture
def application_session_factory(
    db_session: Session,
) -> sessionmaker[Session]:
    db_session.commit()
    return sessionmaker(
        bind=db_session.get_bind(),
        autoflush=False,
        expire_on_commit=False,
    )


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        '_env_file': None,
        'paraworks_demo_mode': True,
        'langgraph_review_v2_enabled': True,
        'agent_runtime_security_scope_id': 'scope-review-v2-service',
        'agent_runtime_fingerprint_secret': 'test-review-v2-service-secret',
        'agent_runtime_fingerprint_key_version': 'test-v1',
        'agent_llm_max_estimated_cost_usd': 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def _actor(
    subject_id: str = 'owner-1',
    *,
    role: str = 'employee',
    permission_levels: set[str] | None = None,
) -> DemoUser:
    return DemoUser(
        id=subject_id,
        email=f'{subject_id}@example.test',
        role=role,
        permission_levels=permission_levels or {'public', 'internal'},
        name=subject_id,
        title='Tester',
        department='Quality',
    )


def _seed_source(
    db: Session,
    sequence: int = 1,
    *,
    permission_level: str = 'internal',
) -> Source:
    signature = f'gmail-signature-{sequence}'
    source = Source(
        source_type='gmail',
        source_id=f'gmail:message-{sequence}',
        source_url=f'https://sensitive.example.test/message-{sequence}',
        title=f'Sensitive message {sequence}',
        permission_level=permission_level,
        raw_metadata={
            'content_signature': signature,
            'review_batch_mode': 'v2_explicit',
            'review_batch_signature': signature,
        },
    )
    db.add(source)
    db.flush()
    return source


def _request(
    source: Source,
    *,
    client_request_id: str = 'client-request-1',
) -> ReviewWorkflowRunRequest:
    return ReviewWorkflowRunRequest(
        source_refs=[
            {
                'source_type': source.source_type,
                'source_id': source.source_id,
                'version_or_signature': source.raw_metadata['content_signature'],
            }
        ],
        agent_names=['mail_document_agent'],
        client_request_id=client_request_id,
    )


@dataclass
class _FakeDraftService:
    session_factory: Callable[[], Session]
    candidate_count: int = 1
    budget_status: str = 'within_budget'
    model_failures_remaining: int = 0
    draft_calls: int = 0

    def preview_prepared(
        self,
        *,
        prepared,
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
            estimated_input_tokens=128,
            estimated_output_tokens=64,
            estimated_cost_usd=2.0 if self.budget_status == 'over_budget' else 0.001,
            budget_limit_usd=1.0,
            budget_status=self.budget_status,
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
        self.draft_calls += 1
        if self.model_failures_remaining:
            self.model_failures_remaining -= 1
            raise ReviewDraftError(
                'model_unavailable',
                'sensitive provider failure must stay bounded',
            )
        with self.session_factory() as db:
            thread = db.get(AgentWorkflowThread, workflow_thread_id)
            assert thread is not None
            run = db.scalar(
                select(AgentRun).where(
                    AgentRun.workflow_thread_id == workflow_thread_id,
                    AgentRun.effect_key == 'fake-effect-key',
                )
            )
            if run is None and self.candidate_count:
                run = AgentRun(
                    agent_name='mail_document_agent',
                    prompt_version='test:v1',
                    status='complete',
                    source_window='canonical-source-version-batch',
                    cache_key='fake-cache-key',
                    model_name='fake-model',
                    input_tokens=128,
                    output_tokens=64,
                    total_tokens=192,
                    estimated_cost_usd=0.001,
                    permission_level='internal',
                    metadata_={'source_count': 1},
                    workflow_thread_id=workflow_thread_id,
                    effect_key='fake-effect-key',
                )
                db.add(run)
                db.flush()
            existing = list(
                db.scalars(
                    select(ReviewItem)
                    .where(ReviewItem.workflow_thread_id == workflow_thread_id)
                    .order_by(ReviewItem.id)
                ).all()
            )
            if not existing:
                for index in range(self.candidate_count):
                    item = ReviewItem(
                        item_type='history_event',
                        payload={
                            'title': f'Candidate {index + 1}',
                            'agent_run_id': run.id if run is not None else None,
                        },
                        source_links=['https://sensitive.example.test/source'],
                        source_snippets=['sensitive source snippet'],
                        confidence_score=0.9,
                        permission_level='internal',
                        status='pending_review',
                        workflow_thread_id=workflow_thread_id,
                        candidate_key=f'candidate-{index + 1}',
                    )
                    db.add(item)
                db.flush()
                existing = list(
                    db.scalars(
                        select(ReviewItem)
                        .where(ReviewItem.workflow_thread_id == workflow_thread_id)
                        .order_by(ReviewItem.id)
                    ).all()
                )
            if thread.status != 'checkpoint_pending':
                thread.status = 'checkpoint_pending'
                thread.state_version += 1
            counts = {
                'pending_review': 0,
                'approved': 0,
                'rejected': 0,
                'needs_more_evidence': 0,
            }
            for item in existing:
                counts[item.status] += 1
            ids = tuple(item.id for item in existing)
            db.commit()
        return ReviewDraftResult(
            review_item_ids=ids,
            review_status_counts=counts,  # type: ignore[arg-type]
        )


class _CorruptTupleProxy(InMemorySaver):
    def __init__(
        self,
        delegate: InMemorySaver,
        corrupt_thread_id: str,
    ) -> None:
        self.__dict__.update(delegate.__dict__)
        self._corrupt_thread_id = corrupt_thread_id

    def get_tuple(self, config: dict[str, dict[str, str]]) -> object:
        if config['configurable']['thread_id'] == self._corrupt_thread_id:
            return SimpleNamespace(
                config={
                    'configurable': {
                        'thread_id': 'wrong-checkpoint-thread',
                        'checkpoint_ns': '',
                        'checkpoint_id': 'corrupt-checkpoint',
                    }
                }
            )
        return super().get_tuple(config)


@pytest.fixture
def service_parts(
    application_session_factory: sessionmaker[Session],
):
    settings = _settings()
    runtime = CheckpointRuntime(settings)
    runtime.start()
    registry = GraphVersionRegistry()
    register_company_memory_review_v2(registry)
    catalog = build_review_agent_catalog(settings)
    draft = _FakeDraftService(application_session_factory)
    try:
        yield settings, runtime, registry, catalog.registry, draft
    finally:
        runtime.close()


def _service(
    application_session_factory: Callable[[], Session],
    service_parts,
    *,
    draft: _FakeDraftService | None = None,
    runtime: object | None = None,
    settings: Settings | None = None,
) -> ReviewWorkflowService:
    base_settings, base_runtime, registry, agent_registry, base_draft = service_parts
    return ReviewWorkflowService(
        session_factory=application_session_factory,
        settings=settings or base_settings,
        checkpoint_runtime=runtime or base_runtime,
        graph_registry=registry,
        agent_registry=agent_registry,
        draft_service=draft or base_draft,
    )


def _start(
    db: Session,
    service: ReviewWorkflowService,
    *,
    actor: DemoUser | None = None,
    sequence: int = 1,
):
    source = _seed_source(db, sequence)
    db.commit()
    status = service.start(
        actor=actor or _actor(),
        request=_request(source, client_request_id=f'client-request-{sequence}'),
    )
    db.expire_all()
    return source, status


def _set_item_statuses(
    session_factory: Callable[[], Session],
    thread_id: str,
    statuses: Sequence[str],
) -> None:
    with session_factory() as db:
        items = list(
            db.scalars(
                select(ReviewItem)
                .where(ReviewItem.workflow_thread_id == thread_id)
                .order_by(ReviewItem.id)
            ).all()
        )
        assert len(items) == len(statuses)
        for item, status in zip(items, statuses, strict=True):
            item.status = status
        db.commit()


def test_initial_candidate_run_returns_only_after_confirmed_interrupt(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)

    _, status = _start(db_session, service)

    assert status.status == 'awaiting_human_review'
    assert status.review_status_counts == {'pending_review': 1}
    assert status.checkpoint_resumable is True
    thread = db_session.get(AgentWorkflowThread, status.thread_id)
    assert thread is not None
    assert thread.checkpoint_confirmed_at is not None
    assert service_parts[1].saver.get_tuple(  # type: ignore[union-attr]
        checkpoint_config(thread.checkpoint_thread_id)
    ) is not None


def test_initial_no_candidate_run_is_terminal(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    draft = _FakeDraftService(application_session_factory, candidate_count=0)
    service = _service(application_session_factory, service_parts, draft=draft)

    _, status = _start(db_session, service)

    assert status.status == 'completed'
    assert status.review_item_count == 0
    assert status.resume_allowed is False
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == 0


def test_actual_start_rejects_over_budget_before_thread_write(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    source = _seed_source(db_session)
    db_session.commit()
    draft = _FakeDraftService(
        application_session_factory,
        budget_status='over_budget',
    )
    service = _service(application_session_factory, service_parts, draft=draft)

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.start(actor=_actor(), request=_request(source))

    assert exc_info.value.code == 'budget_exceeded'
    assert db_session.scalar(
        select(func.count()).select_from(AgentWorkflowThread)
    ) == 0


def test_status_projects_current_review_rows_not_checkpoint_counts(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])

    status = service.status(actor=_actor(), thread_id=started.thread_id)

    assert status.review_status_counts == {'approved': 1}
    assert status.review_resolution_ready is True
    assert status.resume_allowed is True


def test_pending_resume_returns_review_unresolved_before_command_or_saver_write(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    saver = service_parts[1].saver
    before = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)

    assert exc_info.value.code == 'review_unresolved'
    after = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
    assert after is not None and before is not None
    assert after.config['configurable']['checkpoint_id'] == before.config['configurable']['checkpoint_id']


def test_same_thread_resume_completes_after_all_items_resolve(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    checkpoint_thread_id = thread.checkpoint_thread_id
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])

    resumed = service.resume(actor=_actor(), thread_id=started.thread_id)

    assert resumed.status == 'completed'
    db_session.expire_all()
    assert db_session.get(AgentWorkflowThread, started.thread_id).checkpoint_thread_id == checkpoint_thread_id


def test_checkpoint_failed_repairs_same_application_thread_and_reuses_effects(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    original_thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert original_thread is not None
    original_checkpoint_thread_id = original_thread.checkpoint_thread_id
    original_thread.status = 'checkpoint_failed'
    original_thread.checkpoint_thread_id = 'review-v2:missing-attempt'
    original_thread.state_version += 1
    db_session.commit()
    run_count = db_session.scalar(select(func.count()).select_from(AgentRun))
    item_count = db_session.scalar(select(func.count()).select_from(ReviewItem))

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    assert repaired.thread_id == started.thread_id
    assert repaired.status == 'awaiting_human_review'
    db_session.expire_all()
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread.checkpoint_thread_id not in {
        original_checkpoint_thread_id,
        'review-v2:missing-attempt',
    }
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == run_count
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == item_count


def test_valid_saved_tuple_reconciles_status_without_rewriting_checkpoint(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()
    saver = service_parts[1].saver
    before = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))

    reconciled = service.status(actor=_actor(), thread_id=started.thread_id)

    after = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
    assert reconciled.status == 'awaiting_human_review'
    assert before is not None and after is not None
    assert before.config['configurable']['checkpoint_id'] == after.config['configurable']['checkpoint_id']


def test_no_candidate_confirmed_checkpoint_reconciles_after_application_cas_failure(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = _FakeDraftService(application_session_factory, candidate_count=0)
    service = _service(application_session_factory, service_parts, draft=draft)
    source = _seed_source(db_session)
    db_session.commit()
    monkeypatch.setattr(
        service,
        '_finish_initial_checkpoint',
        lambda **_kwargs: False,
    )

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.start(actor=_actor(), request=_request(source))
    thread = db_session.scalar(select(AgentWorkflowThread))
    assert thread is not None and thread.status == 'checkpoint_pending'

    reconciled = service.status(actor=_actor(), thread_id=thread.thread_id)

    assert exc_info.value.code == 'checkpoint_failed'
    assert reconciled.status == 'completed'
    assert reconciled.review_item_count == 0
    assert draft.draft_calls == 1


def test_completed_resume_checkpoint_reconciles_after_application_cas_failure(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    cas_status = service._cas_status

    def fail_completed_cas(**kwargs: object) -> bool:
        if kwargs['target_status'] == 'completed':
            return False
        return cas_status(**kwargs)

    monkeypatch.setattr(service, '_cas_status', fail_completed_cas)
    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)
    monkeypatch.setattr(service, '_cas_status', cas_status)

    reconciled = service.status(actor=_actor(), thread_id=started.thread_id)

    assert exc_info.value.code == 'concurrent_resume'
    assert reconciled.status == 'completed'
    assert reconciled.review_item_count == 1


def test_missing_tuple_rotates_only_checkpoint_attempt_id_and_reuses_business_rows(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    original_values = (thread.input_hash, thread.evidence_version_hash, thread.owner_subject_id)
    thread.status = 'checkpoint_failed'
    thread.checkpoint_thread_id = 'review-v2:lost-attempt'
    thread.state_version += 1
    db_session.commit()

    service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    repaired = db_session.get(AgentWorkflowThread, started.thread_id)
    assert repaired is not None
    assert repaired.thread_id == started.thread_id
    assert repaired.checkpoint_thread_id != 'review-v2:lost-attempt'
    assert (repaired.input_hash, repaired.evidence_version_hash, repaired.owner_subject_id) == original_values


def test_corrupt_tuple_rotates_only_checkpoint_attempt_and_reuses_business_rows(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _, runtime, _, _, _ = service_parts
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    original_checkpoint_thread_id = thread.checkpoint_thread_id
    run_count = db_session.scalar(select(func.count()).select_from(AgentRun))
    item_count = db_session.scalar(select(func.count()).select_from(ReviewItem))
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()
    original_saver = runtime.saver
    assert isinstance(original_saver, InMemorySaver)
    runtime._saver = _CorruptTupleProxy(
        original_saver,
        original_checkpoint_thread_id,
    )

    try:
        repaired = service.resume(actor=_actor(), thread_id=started.thread_id)
    finally:
        runtime._saver = original_saver

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert repaired.status == 'awaiting_human_review'
    assert current.checkpoint_thread_id != original_checkpoint_thread_id
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == run_count
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == item_count


def test_checkpoint_write_failure_is_bounded_and_preserves_business_rows(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    source = _seed_source(db_session)
    db_session.commit()

    def fail_checkpoint_write(**_kwargs: object) -> None:
        raise RuntimeError('sensitive checkpoint backend failure')

    monkeypatch.setattr(
        'backend.app.agent_runtime.review_v2_service.invoke_and_confirm_checkpoint',
        fail_checkpoint_write,
    )

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.start(actor=_actor(), request=_request(source))

    db_session.expire_all()
    thread = db_session.scalar(select(AgentWorkflowThread))
    assert exc_info.value.code == 'checkpoint_failed'
    assert str(exc_info.value) == 'checkpoint_failed'
    assert thread is not None and thread.status == 'checkpoint_failed'
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == 1
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == 1


@pytest.mark.parametrize('terminal_status', ['failed', 'cancelled'])
def test_failed_and_cancelled_threads_are_terminal_without_retry(
    db_session: Session,
    application_session_factory,
    service_parts,
    terminal_status: str,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.status = terminal_status
    thread.state_version += 1
    db_session.commit()

    status = service.status(actor=_actor(), thread_id=started.thread_id)
    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)

    assert status.retry_allowed is False
    assert exc_info.value.code == 'invalid_state_transition'


def test_draft_retry_budget_exhaustion_marks_failed_without_replacement(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    source = _seed_source(db_session)
    db_session.commit()
    draft = _FakeDraftService(
        application_session_factory,
        model_failures_remaining=2,
    )
    service = _service(application_session_factory, service_parts, draft=draft)
    request = _request(source)

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.start(actor=_actor(), request=request)
    replay = service.start(actor=_actor(), request=request)

    assert exc_info.value.code == 'model_unavailable'
    assert replay.status == 'failed'
    assert replay.retry_allowed is False
    assert draft.draft_calls == 2
    assert db_session.scalar(
        select(func.count()).select_from(AgentWorkflowThread)
    ) == 1


def test_success_after_one_model_failure_stays_within_two_draft_attempts(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    source = _seed_source(db_session)
    db_session.commit()
    draft = _FakeDraftService(
        application_session_factory,
        model_failures_remaining=1,
    )
    service = _service(application_session_factory, service_parts, draft=draft)

    status = service.start(actor=_actor(), request=_request(source))

    assert status.status == 'awaiting_human_review'
    assert draft.draft_calls == 2


def test_concurrent_resume_and_cancel_allow_one_cas_winner(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    competing_service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])

    invoke = lifecycle_module.invoke_and_confirm_checkpoint

    def invoke_then_cancel(**kwargs: object):
        confirmation = invoke(**kwargs)
        competing_service.cancel(actor=_actor(), thread_id=started.thread_id)
        return confirmation

    monkeypatch.setattr(
        lifecycle_module,
        'invoke_and_confirm_checkpoint',
        invoke_then_cancel,
    )

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert exc_info.value.code == 'concurrent_resume'
    assert current is not None and current.status == 'cancelled'


def test_owner_reviewer_admin_and_cross_owner_authorization_matrix(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    other_employee = _actor('employee-2')
    reviewer = _actor('reviewer-1', role='reviewer')
    admin = _actor('admin-1', role='admin', permission_levels={'public', 'internal', 'restricted'})

    assert service.status(actor=other_employee, thread_id=started.thread_id).thread_id == started.thread_id
    with pytest.raises(ReviewWorkflowServiceError) as resume_error:
        service.resume(actor=other_employee, thread_id=started.thread_id)
    with pytest.raises(ReviewWorkflowServiceError) as cancel_error:
        service.cancel(actor=other_employee, thread_id=started.thread_id)
    assert resume_error.value.code == 'not_found'
    assert cancel_error.value.code == 'not_found'

    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    assert service.resume(actor=reviewer, thread_id=started.thread_id).status == 'completed'

    _, second = _start(db_session, service, actor=_actor(), sequence=2)
    assert service.cancel(actor=admin, thread_id=second.thread_id).status == 'cancelled'


def test_permission_revocation_fails_closed_without_deleting_knowledge(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    source, started = _start(db_session, service)
    source.permission_level = 'restricted'
    db_session.commit()
    item_count = db_session.scalar(select(func.count()).select_from(ReviewItem))

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.status(actor=_actor(), thread_id=started.thread_id)

    assert exc_info.value.code == 'not_found'
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == item_count


def test_unsupported_graph_version_is_non_mutating(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.graph_version = 'company-memory-review-v999.0'
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    before = (thread.status, thread.state_version, thread.checkpoint_thread_id)
    db_session.commit()

    status = service.status(actor=_actor(), thread_id=started.thread_id)
    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    after_thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert status.error_code == 'runtime_version_unavailable'
    assert status.retry_allowed is False
    assert exc_info.value.code == 'runtime_version_unavailable'
    assert (after_thread.status, after_thread.state_version, after_thread.checkpoint_thread_id) == before


class _ExplodingSaver:
    def __init__(self) -> None:
        self.probe_count = 0

    def get_tuple(self, _config):
        self.probe_count += 1
        raise AssertionError('mismatched saver must never be probed')


def test_checkpoint_store_mismatch_fails_closed_without_tuple_probe(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.checkpoint_store = 'postgres'
    db_session.commit()
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    saver = _ExplodingSaver()
    mismatch_runtime = SimpleNamespace(
        readiness=CheckpointReadiness(
            enabled=True,
            mode='memory',
            ready=True,
            durable=False,
            checkpoint_store='memory',
        ),
        saver=saver,
    )
    mismatch_service = _service(
        application_session_factory,
        service_parts,
        runtime=mismatch_runtime,
    )

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        mismatch_service.resume(actor=_actor(), thread_id=started.thread_id)

    assert exc_info.value.code == 'checkpoint_unavailable'
    assert saver.probe_count == 0
