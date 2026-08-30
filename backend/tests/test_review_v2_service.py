from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime import review_v2_service as lifecycle_module
from backend.app.agent_runtime.checkpoint_execution import (
    checkpoint_config,
    invoke_and_confirm_checkpoint,
)
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
    ReviewDraftService,
)
from backend.app.agent_runtime.review_v2_preflight import (
    PreparedReviewRequestV21,
    V21PreparedReviewConfig,
)
from backend.app.agent_runtime.review_v2_service import (
    ReviewModelReadiness,
    ReviewWorkflowService,
    ReviewWorkflowServiceError,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models.agent_runs import AgentRun
from backend.app.models.agent_workflows import (
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.models.source import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    ReviewWorkflowDryRunResponse,
    ReviewWorkflowRunRequest,
    ReviewWorkflowRunRequestV21,
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
    connector_signature = f'gmail-signature-{sequence}'
    server_signature = f'{sequence:064x}'
    source = Source(
        source_type='gmail',
        source_id=f'gmail:message-{sequence}',
        source_url=f'https://sensitive.example.test/message-{sequence}',
        title=f'Sensitive message {sequence}',
        permission_level=permission_level,
        raw_metadata={
            'content_signature': connector_signature,
            'review_batch_mode': 'v2_explicit',
            'review_batch_signature': server_signature,
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
    document_version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body='body for v1',
    )
    db.add(document_version)
    db.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=document_version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id='revision-1',
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
            version_id=document_version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text='body for v1',
            source_snippet='body for v1',
            permission_level=permission_level,
            metadata_={},
        )
    )
    document.current_document_version_id = document_version.id
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
                'version_or_signature': source.server_content_signature,
            }
        ],
        agent_names=['mail_document_agent'],
        client_request_id=client_request_id,
    )


def _attach_document_version(
    db: Session,
    source: Source,
    *,
    version: str = 'v1',
    revision_id: str = 'revision-1',
) -> tuple[Document, DocumentVersion, DocumentParserRun]:
    document = db.scalar(select(Document).where(Document.source_id == source.id))
    assert document is not None
    assert document.current_version == version
    document_version = db.get(DocumentVersion, document.current_document_version_id)
    assert document_version is not None
    parser_run = db.scalar(
        select(DocumentParserRun).where(
            DocumentParserRun.source_id == source.id,
            DocumentParserRun.document_id == document.id,
            DocumentParserRun.document_version_id == document_version.id,
        )
    )
    assert parser_run is not None
    assert parser_run.revision_id == revision_id
    return document, document_version, parser_run


@dataclass
class _FakeDraftService:
    session_factory: Callable[[], Session]
    candidate_count: int = 1
    budget_status: str = 'within_budget'
    model_failures_remaining: int = 0
    draft_calls: int = 0
    preview_calls: int = 0
    last_prepared: object | None = None

    def preview_prepared(
        self,
        *,
        prepared,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewWorkflowDryRunResponse:
        del actor_subject_id, allowed_permission_levels
        self.preview_calls += 1
        self.last_prepared = prepared
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


@dataclass
class _UnavailableDraftService:
    draft_calls: int = 0

    def draft(self, **_kwargs: object) -> ReviewDraftResult:
        self.draft_calls += 1
        raise ReviewDraftError(
            'model_unavailable',
            'provider credentials or model route changed',
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


class _SnapshotMutatingGraph:
    def __init__(
        self,
        delegate: object,
        *,
        checkpoint_thread_id: str,
        identity_field: str | None = None,
        partial: bool = False,
        terminal_status: str | None = None,
        shape_corruption: str | None = None,
    ) -> None:
        self._delegate = delegate
        self._checkpoint_thread_id = checkpoint_thread_id
        self._identity_field = identity_field
        self._partial = partial
        self._terminal_status = terminal_status
        self._shape_corruption = shape_corruption

    def get_state(self, config: dict[str, dict[str, str]]):
        snapshot = self._delegate.get_state(config)
        if config['configurable']['thread_id'] != self._checkpoint_thread_id:
            return snapshot
        if self._partial:
            return snapshot._replace(
                tasks=(),
                next=('verify_review_resolution_from_postgres',),
            )
        values = dict(snapshot.values)
        if self._shape_corruption is not None:
            corruption = self._shape_corruption
            tasks = tuple(snapshot.tasks)
            interrupts = tuple(snapshot.interrupts)
            if corruption == 'nonempty_review_item_ids':
                values['review_item_ids'] = [987_654_321]
            elif corruption == 'extra_state_key':
                values['source_content'] = 'must-not-be-checkpointed'
            elif corruption in {
                'swapped_equal_total_counts',
                'wrong_total_counts',
                'missing_count_status',
                'extra_count_status',
                'pending_approved_terminal_counts',
            }:
                counts = dict(values['review_status_counts'])
                if corruption == 'swapped_equal_total_counts':
                    counts['pending_review'] = 0
                    counts['approved'] = 1
                elif corruption == 'wrong_total_counts':
                    counts['pending_review'] += 1
                elif corruption == 'missing_count_status':
                    counts.pop('rejected')
                elif corruption == 'pending_approved_terminal_counts':
                    counts = {
                        'pending_review': 1,
                        'approved': 1,
                        'rejected': 0,
                        'needs_more_evidence': 0,
                    }
                else:
                    counts['unknown_status'] = 0
                values['review_status_counts'] = counts
            elif corruption in {
                'extra_interrupt_key',
                'future_interrupt_version',
                'stale_interrupt_version',
            }:
                task = tasks[0]
                original = task.interrupts[0]
                payload = dict(original.value)
                if corruption == 'extra_interrupt_key':
                    payload['source_id'] = 'must-not-be-checkpointed'
                elif corruption == 'future_interrupt_version':
                    payload['state_version'] += 10_000
                else:
                    payload['state_version'] -= 1
                mutated_interrupt = Interrupt(payload, id=original.id)
                tasks = (task._replace(interrupts=(mutated_interrupt,)),)
                interrupts = (mutated_interrupt,)
            elif corruption == 'wrong_interrupt_phase':
                values['phase'] = 'checkpoint_pending'
            elif corruption == 'wrong_terminal_phase':
                values['phase'] = 'needs_more_evidence'
            elif corruption == 'wrong_terminal_node':
                completed_nodes = list(values['completed_nodes'])
                completed_nodes[-1] = 'finalize_needs_more_evidence'
                values['completed_nodes'] = completed_nodes
            elif corruption == 'wrong_terminal_status':
                values['status'] = 'needs_more_evidence'
            elif corruption == 'wrong_terminal_review_count':
                values['review_item_count'] += 1
            else:
                raise AssertionError(f'unknown corruption: {corruption}')
            return snapshot._replace(
                values=values,
                tasks=tasks,
                interrupts=interrupts,
            )
        if self._terminal_status is not None:
            values['status'] = self._terminal_status
            return snapshot._replace(tasks=(), next=(), values=values)
        assert self._identity_field is not None
        values[self._identity_field] = 'wrong-snapshot-identity'
        return snapshot._replace(values=values)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


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
    draft: object | None = None,
    runtime: object | None = None,
    settings: Settings | None = None,
    model_readiness: object | None = None,
    v21_launch_authority: object | None = None,
) -> ReviewWorkflowService:
    base_settings, base_runtime, registry, agent_registry, base_draft = service_parts
    extra = (
        {'model_readiness': model_readiness}
        if model_readiness is not None
        else {}
    )
    return ReviewWorkflowService(
        session_factory=application_session_factory,
        settings=settings or base_settings,
        checkpoint_runtime=runtime or base_runtime,
        graph_registry=registry,
        agent_registry=agent_registry,
        draft_service=draft or base_draft,
        v21_launch_authority=v21_launch_authority,
        **extra,
    )


def _v21_config() -> V21PreparedReviewConfig:
    plan = {
        'agent_name': 'mail_document_agent',
        'provider': 'openai',
        'model': 'gpt-5.4-mini-2026-03-17',
        'reasoning_effort': 'none',
        'route_version': 'auto-review-extraction-route:v1',
        'prompt_version': 'mail-document-extraction:v1',
        'output_contract_version': 'mail-document-extraction:v1',
        'extraction_registry_version': 'auto-review-extraction-registry:v1',
        'cost_policy_version': 'auto-review-extraction-cost:v1',
        'token_estimator_version': 'openai-o200k-extraction:v1',
        'tokenizer_encoding': 'o200k_base',
        'reply_priming_tokens': 16,
        'framing_safety_tokens': 512,
        'max_input_chars': 24000,
        'max_input_tokens': 10000,
        'max_output_tokens': 2048,
        'max_candidates': 1,
        'max_provider_attempts': 1,
        'input_usd_per_1m': '0.750000',
        'output_usd_per_1m': '4.500000',
        'provider_safety_state_version': 1,
        'timing': [60, 5, 120, 30],
        'prepared_content_hmac': 'e' * 64,
        'prepared_character_count': 800,
        'framed_input_tokens': 900,
        'reserved_cost_usd': '0.016716',
    }
    return V21PreparedReviewConfig(
        configured_auto_review_mode='shadow',
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        validator_reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v2',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
        cost_policy_version='auto-review-cost:v1',
        fingerprint_key_version='test-v1',
        fingerprint_key_material_verifier='a' * 64,
        token_estimator_version='openai-o200k-chat:v1',
        tokenizer_encoding='o200k_base',
        max_input_tokens_per_batch=6000,
        max_output_tokens_per_batch=3072,
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        max_validation_batches_per_workflow=2,
        max_validation_candidates_per_batch=4,
        max_validation_candidates_per_workflow=5,
        max_provider_attempts=1,
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
        validator_input_usd_per_1m=Decimal('2.000000'),
        validator_output_usd_per_1m=Decimal('12.000000'),
        enforce_percentage=0,
        authorized_percentage_at_launch=0,
        rollout_authorization_generation=1,
        validation_provider_safety_state_version=1,
        rollout_control_epoch=1,
        extraction_plan_set_hmac='b' * 64,
        extraction_provider_safety_snapshot_set_hmac='c' * 64,
        confirmed_extraction_cost_ceiling_usd=Decimal('0.016716'),
        confirmed_validation_cost_ceiling_usd=Decimal('0.048864'),
        confirmed_total_cost_ceiling_usd=Decimal('0.065580'),
        total_budget_limit_usd=Decimal('0.200000'),
        extraction_plan_identities=(plan,),
    )


@dataclass(frozen=True)
class _V21Authority:
    config: V21PreparedReviewConfig

    def resolve_v21_config(self, **_kwargs) -> V21PreparedReviewConfig:
        return self.config


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


def test_shadow_service_dispatches_dry_run_through_supplied_v21_authority(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    draft = _FakeDraftService(application_session_factory)
    service = _service(
        application_session_factory,
        service_parts,
        draft=draft,
        settings=_settings(auto_review_mode='shadow'),
        v21_launch_authority=_V21Authority(_v21_config()),
    )
    source = _seed_source(db_session)
    db_session.commit()

    service.dry_run(actor=_actor(), request=_request(source))

    assert draft.preview_calls == 1
    assert isinstance(draft.last_prepared, PreparedReviewRequestV21)
    assert draft.last_prepared.config == _v21_config()


def test_shadow_service_without_launch_authority_is_zero_call_fail_closed(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    draft = _FakeDraftService(application_session_factory)
    service = _service(
        application_session_factory,
        service_parts,
        draft=draft,
        settings=_settings(auto_review_mode='shadow'),
    )
    source = _seed_source(db_session)
    db_session.commit()

    with pytest.raises(ReviewWorkflowServiceError) as captured:
        service.dry_run(actor=_actor(), request=_request(source))

    assert captured.value.code == 'cost_preview_changed'
    assert draft.preview_calls == 0
    assert db_session.scalar(
        select(func.count()).select_from(AgentWorkflowThread)
    ) == 0


def test_shadow_service_start_persists_v21_snapshot_without_running_future_graph(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    settings = _settings(auto_review_mode='shadow')
    actual_draft = ReviewDraftService(
        session_factory=application_session_factory,
        catalog=build_review_agent_catalog(settings),
        settings=settings,
    )
    service = _service(
        application_session_factory,
        service_parts,
        draft=actual_draft,
        settings=settings,
        v21_launch_authority=_V21Authority(_v21_config()),
    )
    source = _seed_source(db_session)
    db_session.commit()

    request = _request(source)
    preview = service.dry_run(actor=_actor(), request=request)
    assert hasattr(preview, 'launch_confirmation_token')
    status = service.start(
        actor=_actor(),
        request=ReviewWorkflowRunRequestV21(
            **request.model_dump(),
            launch_confirmation_token=preview.launch_confirmation_token,
        ),
    )

    db_session.expire_all()
    thread = db_session.get(AgentWorkflowThread, status.thread_id)
    request_row = db_session.get(AgentWorkflowRequest, status.thread_id)
    assert status.status == 'created'
    assert thread is not None and request_row is not None
    assert thread.graph_version == COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
    assert request_row.auto_review_mode == 'shadow'
    assert request_row.auto_review_extraction_provider == 'openai'
    assert request_row.selected_extraction_agent_count == 1
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == 0


def test_model_readiness_blocks_only_new_work_and_preserves_checkpoint_mode(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    unavailable_draft = _UnavailableDraftService()
    runtime = SimpleNamespace(
        readiness=CheckpointReadiness(
            enabled=True,
            mode='postgres',
            ready=True,
            durable=True,
            checkpoint_store='postgres',
        ),
        saver=object(),
    )
    service = _service(
        application_session_factory,
        service_parts,
        runtime=runtime,
        draft=unavailable_draft,
        model_readiness=ReviewModelReadiness(
            ready=False,
            error_code='model_unavailable',
        ),
    )
    source = _seed_source(db_session)
    db_session.commit()

    diagnostic = service.diagnostic()
    with pytest.raises(ReviewWorkflowServiceError) as dry_error:
        service.dry_run(actor=_actor(), request=_request(source))
    with pytest.raises(ReviewWorkflowServiceError) as start_error:
        service.start(actor=_actor(), request=_request(source))

    assert diagnostic.enabled is True
    assert diagnostic.available is False
    assert diagnostic.checkpoint_mode == 'postgres'
    assert diagnostic.durable is True
    assert diagnostic.error_code == 'model_unavailable'
    assert dry_error.value.code == start_error.value.code == 'model_unavailable'
    assert unavailable_draft.draft_calls == 0
    assert db_session.scalar(
        select(func.count()).select_from(AgentWorkflowThread)
    ) == 0


def test_model_unavailable_keeps_existing_status_resume_and_cancel_serviceable(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    initial_service = _service(application_session_factory, service_parts)
    _, resumable = _start(db_session, initial_service)
    _, cancellable = _start(db_session, initial_service, sequence=2)
    unavailable_draft = _UnavailableDraftService()
    existing_service = _service(
        application_session_factory,
        service_parts,
        draft=unavailable_draft,
        model_readiness=ReviewModelReadiness(
            ready=False,
            error_code='model_unavailable',
        ),
    )

    projected = existing_service.status(
        actor=_actor(),
        thread_id=resumable.thread_id,
    )
    _set_item_statuses(
        application_session_factory,
        resumable.thread_id,
        ['approved'],
    )
    resumed = existing_service.resume(
        actor=_actor(),
        thread_id=resumable.thread_id,
    )
    cancelled = existing_service.cancel(
        actor=_actor(),
        thread_id=cancellable.thread_id,
    )

    assert projected.status == 'awaiting_human_review'
    assert resumed.status == 'completed'
    assert cancelled.status == 'cancelled'
    assert unavailable_draft.draft_calls == 0


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
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    checkpoint_thread_id = thread.checkpoint_thread_id
    state_version = thread.state_version
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])

    status = service.status(actor=_actor(), thread_id=started.thread_id)

    assert status.status == 'awaiting_human_review'
    assert status.review_status_counts == {'approved': 1}
    assert status.review_resolution_ready is True
    assert status.checkpoint_resumable is True
    assert status.resume_allowed is True
    assert status.resume_error_code is None
    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert current.status == 'awaiting_human_review'
    assert current.checkpoint_thread_id == checkpoint_thread_id
    assert current.state_version == state_version


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
    state_version = thread.state_version
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])

    resumed = service.resume(actor=_actor(), thread_id=started.thread_id)

    assert resumed.status == 'completed'
    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert current.thread_id == started.thread_id
    assert current.checkpoint_thread_id == checkpoint_thread_id
    assert current.state_version == state_version + 2


def test_live_review_count_change_uses_direct_saved_resume_without_repair(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    old_state_version = thread.state_version
    run_count = db_session.scalar(select(func.count()).select_from(AgentRun))
    item_count = db_session.scalar(select(func.count()).select_from(ReviewItem))
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])

    def forbidden_path(*_args: object, **_kwargs: object) -> None:
        pytest.fail('normal approval must not rotate, repair, or fail checkpoint')

    monkeypatch.setattr(service, '_mark_checkpoint_failed', forbidden_path)
    monkeypatch.setattr(service, '_repair_checkpoint', forbidden_path)
    monkeypatch.setattr(service, '_rotate_checkpoint_attempt', forbidden_path)

    ready = service.status(actor=_actor(), thread_id=started.thread_id)
    assert ready.status == 'awaiting_human_review'
    assert ready.checkpoint_resumable is True
    assert ready.review_resolution_ready is True
    assert ready.resume_allowed is True

    resumed = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert resumed.status == 'completed'
    assert current.checkpoint_thread_id == old_checkpoint_thread_id
    assert current.state_version == old_state_version + 2
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == run_count
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == item_count
    saver = service_parts[1].saver
    assert saver is not None
    graph = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )(saver)
    snapshot = graph.get_state(checkpoint_config(current.checkpoint_thread_id))
    assert 'review_item_ids' not in snapshot.values
    assert set(snapshot.values) == {
        'workflow_thread_id',
        'graph_version',
        'input_hash',
        'evidence_version_hash',
        'review_status_counts',
        'phase',
        'completed_nodes',
        'error_codes',
        'status',
        'review_item_count',
    }


def test_resume_rechecks_exact_runtime_after_claim_before_invoke(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    runtime = service_parts[1]
    saver = runtime.saver
    assert saver is not None
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    checkpoint_thread_id = thread.checkpoint_thread_id
    before = saver.get_tuple(checkpoint_config(checkpoint_thread_id))
    assert before is not None
    original_readiness = runtime.readiness
    original_cas = service._cas_status

    def switch_runtime_after_claim(**kwargs: object) -> bool:
        result = original_cas(**kwargs)
        if result and kwargs.get('target_status') == 'resuming':
            runtime._readiness = CheckpointReadiness(
                enabled=True,
                mode='postgres',
                ready=True,
                durable=True,
                checkpoint_store='postgres',
            )
        return result

    monkeypatch.setattr(service, '_cas_status', switch_runtime_after_claim)
    try:
        with pytest.raises(ReviewWorkflowServiceError) as exc_info:
            service.resume(actor=_actor(), thread_id=started.thread_id)
    finally:
        runtime._readiness = original_readiness

    assert exc_info.value.code == 'checkpoint_unavailable'
    after = saver.get_tuple(checkpoint_config(checkpoint_thread_id))
    assert after is not None
    assert (
        after.config['configurable']['checkpoint_id']
        == before.config['configurable']['checkpoint_id']
    )
    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None and current.status == 'awaiting_human_review'


def test_evidence_drift_after_resume_claim_restores_awaiting_without_saver_write(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    source, started = _start(db_session, service)
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    saver = service_parts[1].saver
    assert saver is not None
    before = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
    assert before is not None
    original_cas = service._cas_status

    def drift_after_claim(**kwargs: object) -> bool:
        result = original_cas(**kwargs)
        if result and kwargs.get('target_status') == 'resuming':
            with application_session_factory() as db:
                current_source = db.get(Source, source.id)
                assert current_source is not None
                current_source.server_content_signature = 'e' * 64
                db.commit()
        return result

    monkeypatch.setattr(service, '_cas_status', drift_after_claim)

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)

    assert exc_info.value.code == 'evidence_changed'
    after = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
    assert after is not None
    assert (
        after.config['configurable']['checkpoint_id']
        == before.config['configurable']['checkpoint_id']
    )
    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None and current.status == 'awaiting_human_review'


def test_repair_rechecks_exact_runtime_after_rotation_before_invoke(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    runtime = service_parts[1]
    saver = runtime.saver
    assert saver is not None
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.status = 'checkpoint_failed'
    thread.checkpoint_thread_id = 'review-v2:missing-before-runtime-change'
    thread.state_version += 1
    db_session.commit()
    original_readiness = runtime.readiness
    original_rotate = service._rotate_checkpoint_attempt

    def switch_runtime_after_rotation(projection):
        rotated = original_rotate(projection)
        if rotated is not None:
            runtime._readiness = CheckpointReadiness(
                enabled=True,
                mode='postgres',
                ready=True,
                durable=True,
                checkpoint_store='postgres',
            )
        return rotated

    monkeypatch.setattr(
        service,
        '_rotate_checkpoint_attempt',
        switch_runtime_after_rotation,
    )
    try:
        with pytest.raises(ReviewWorkflowServiceError) as exc_info:
            service.resume(actor=_actor(), thread_id=started.thread_id)
    finally:
        runtime._readiness = original_readiness

    assert exc_info.value.code == 'checkpoint_unavailable'
    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None and current.status == 'checkpoint_failed'
    assert saver.get_tuple(
        checkpoint_config(current.checkpoint_thread_id)
    ) is None


@pytest.mark.parametrize(
    'drift',
    [
        'parser_status',
        'parser_signature',
        'external_revision',
        'document_version',
        'source_signature',
        'permission_level',
    ],
)
def test_resume_revalidates_full_canonical_evidence_before_saver_write(
    db_session: Session,
    application_session_factory,
    service_parts,
    drift: str,
) -> None:
    service = _service(application_session_factory, service_parts)
    source = _seed_source(db_session)
    document, _, parser_run = _attach_document_version(db_session, source)
    db_session.commit()
    started = service.start(actor=_actor(), request=_request(source))
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    saver = service_parts[1].saver
    assert saver is not None
    before = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
    assert before is not None

    if drift == 'parser_status':
        parser_run.parser_status = 'failed'
    elif drift == 'parser_signature':
        parser_run.content_signature = 'd' * 64
    elif drift == 'external_revision':
        parser_run.revision_id = 'revision-2'
    elif drift == 'source_signature':
        source.server_content_signature = 'f' * 64
    elif drift == 'permission_level':
        source.permission_level = 'public'
    else:
        document.current_version = 'v2'
        v2 = DocumentVersion(
            document_id=document.id,
            version='v2',
            body='changed document body',
        )
        db_session.add(v2)
        db_session.flush()
        document.current_document_version_id = v2.id
    db_session.commit()

    with pytest.raises(ReviewWorkflowServiceError) as exc_info:
        service.resume(actor=_actor(), thread_id=started.thread_id)

    assert exc_info.value.code == 'evidence_changed'
    after = saver.get_tuple(checkpoint_config(thread.checkpoint_thread_id))
    assert after is not None
    assert (
        after.config['configurable']['checkpoint_id']
        == before.config['configurable']['checkpoint_id']
    )
    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert current.status == 'awaiting_human_review'


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


def test_checkpoint_repair_replays_bound_items_without_draft_or_provider_calls(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    initial_service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, initial_service)
    item_ids = tuple(
        db_session.scalars(
            select(ReviewItem.id).where(
                ReviewItem.workflow_thread_id == started.thread_id
            )
        ).all()
    )
    run_ids = tuple(
        db_session.scalars(
            select(AgentRun.id).where(
                AgentRun.workflow_thread_id == started.thread_id
            )
        ).all()
    )
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.status = 'checkpoint_failed'
    thread.checkpoint_thread_id = 'review-v2:missing-after-model-route-change'
    thread.state_version += 1
    db_session.commit()
    unavailable_draft = _UnavailableDraftService()
    repair_service = _service(
        application_session_factory,
        service_parts,
        draft=unavailable_draft,
        model_readiness=ReviewModelReadiness(
            ready=False,
            error_code='model_unavailable',
        ),
    )

    before = repair_service.status(actor=_actor(), thread_id=started.thread_id)
    repaired = repair_service.resume(
        actor=_actor(permission_levels={'public', 'internal', 'restricted'}),
        thread_id=started.thread_id,
    )

    assert before.retry_allowed is True
    assert repaired.status == 'awaiting_human_review'
    assert unavailable_draft.draft_calls == 0
    assert tuple(
        db_session.scalars(
            select(ReviewItem.id).where(
                ReviewItem.workflow_thread_id == started.thread_id
            )
        ).all()
    ) == item_ids
    assert tuple(
        db_session.scalars(
            select(AgentRun.id).where(
                AgentRun.workflow_thread_id == started.thread_id
            )
        ).all()
    ) == run_ids


def test_zero_item_checkpoint_failed_repairs_to_terminal_completed(
    db_session: Session,
    application_session_factory,
    service_parts,
) -> None:
    initial_draft = _FakeDraftService(
        application_session_factory,
        candidate_count=0,
    )
    initial_service = _service(
        application_session_factory,
        service_parts,
        draft=initial_draft,
    )
    _, started = _start(db_session, initial_service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    thread.status = 'checkpoint_failed'
    thread.checkpoint_thread_id = 'review-v2:missing-zero-item-attempt'
    thread.state_version += 1
    db_session.commit()
    unavailable_draft = _UnavailableDraftService()
    repair_service = _service(
        application_session_factory,
        service_parts,
        draft=unavailable_draft,
        model_readiness=ReviewModelReadiness(
            ready=False,
            error_code='model_unavailable',
        ),
    )

    before = repair_service.status(actor=_actor(), thread_id=started.thread_id)
    repaired = repair_service.resume(actor=_actor(), thread_id=started.thread_id)

    assert before.retry_allowed is True
    assert repaired.status == 'completed'
    assert repaired.review_item_count == 0
    assert repaired.retry_allowed is False
    assert unavailable_draft.draft_calls == 0
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == 0
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == 0


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


@pytest.mark.parametrize(
    'identity_field',
    [
        'workflow_thread_id',
        'graph_version',
        'input_hash',
        'evidence_version_hash',
    ],
)
def test_checkpoint_snapshot_identity_mismatch_rotates_before_repair(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()
    real_builder = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )

    def corrupting_builder(saver: object) -> _SnapshotMutatingGraph:
        return _SnapshotMutatingGraph(
            real_builder(saver),
            checkpoint_thread_id=old_checkpoint_thread_id,
            identity_field=identity_field,
        )

    monkeypatch.setattr(
        service,
        '_resolve_builder',
        lambda _projection: corrupting_builder,
    )

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert repaired.status == 'awaiting_human_review'
    assert current is not None
    assert current.checkpoint_thread_id != old_checkpoint_thread_id


def test_nonterminal_checkpoint_without_interrupt_rotates_before_repair(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()
    real_builder = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )

    def partial_builder(saver: object) -> _SnapshotMutatingGraph:
        return _SnapshotMutatingGraph(
            real_builder(saver),
            checkpoint_thread_id=old_checkpoint_thread_id,
            partial=True,
        )

    monkeypatch.setattr(
        service,
        '_resolve_builder',
        lambda _projection: partial_builder,
    )

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert repaired.status == 'awaiting_human_review'
    assert current is not None
    assert current.checkpoint_thread_id != old_checkpoint_thread_id


@pytest.mark.parametrize(
    'corruption',
    [
        'nonempty_review_item_ids',
        'extra_state_key',
        'extra_interrupt_key',
        'future_interrupt_version',
        'stale_interrupt_version',
        'wrong_total_counts',
        'missing_count_status',
        'extra_count_status',
        'wrong_interrupt_phase',
    ],
)
def test_checkpoint_interrupt_requires_exact_minimized_task6_shape(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()
    run_count = db_session.scalar(select(func.count()).select_from(AgentRun))
    item_count = db_session.scalar(select(func.count()).select_from(ReviewItem))
    saver = service_parts[1].saver
    assert saver is not None
    real_builder = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )

    def corrupting_builder(checkpointer: object) -> _SnapshotMutatingGraph:
        return _SnapshotMutatingGraph(
            real_builder(checkpointer),
            checkpoint_thread_id=old_checkpoint_thread_id,
            shape_corruption=corruption,
        )

    monkeypatch.setattr(
        service,
        '_resolve_builder',
        lambda _projection: corrupting_builder,
    )

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert repaired.status == 'awaiting_human_review'
    assert current.checkpoint_thread_id != old_checkpoint_thread_id
    snapshot = real_builder(saver).get_state(
        checkpoint_config(current.checkpoint_thread_id)
    )
    assert set(snapshot.values) == {
        'workflow_thread_id',
        'graph_version',
        'input_hash',
        'evidence_version_hash',
        'review_status_counts',
        'phase',
        'completed_nodes',
        'error_codes',
    }
    assert 'review_item_ids' not in snapshot.values
    assert snapshot.values['review_status_counts'] == {
        'pending_review': 1,
        'approved': 0,
        'rejected': 0,
        'needs_more_evidence': 0,
    }
    assert db_session.scalar(select(func.count()).select_from(AgentRun)) == run_count
    assert db_session.scalar(select(func.count()).select_from(ReviewItem)) == item_count


@pytest.mark.parametrize(
    'corruption',
    [
        'wrong_terminal_phase',
        'wrong_terminal_node',
        'wrong_terminal_status',
        'wrong_terminal_review_count',
    ],
)
def test_checkpoint_terminal_requires_exact_node_status_and_count_shape(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _set_item_statuses(application_session_factory, started.thread_id, ['approved'])
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    saver = service_parts[1].saver
    assert saver is not None
    real_builder = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )
    invoke_and_confirm_checkpoint(
        graph=real_builder(saver),
        saver=saver,
        command_or_input=Command(resume={
            'event': 'review_resolution_checked',
            'state_version': thread.state_version,
        }),
        checkpoint_thread_id=old_checkpoint_thread_id,
        runtime_context=service._runtime_context(_actor()),
        expect_interrupt=False,
    )
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()

    def corrupting_builder(checkpointer: object) -> _SnapshotMutatingGraph:
        return _SnapshotMutatingGraph(
            real_builder(checkpointer),
            checkpoint_thread_id=old_checkpoint_thread_id,
            shape_corruption=corruption,
        )

    monkeypatch.setattr(
        service,
        '_resolve_builder',
        lambda _projection: corrupting_builder,
    )

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert repaired.status == 'awaiting_human_review'
    assert current.checkpoint_thread_id != old_checkpoint_thread_id
    snapshot = real_builder(saver).get_state(
        checkpoint_config(current.checkpoint_thread_id)
    )
    assert 'review_item_ids' not in snapshot.values
    assert snapshot.values['review_status_counts'] == {
        'pending_review': 0,
        'approved': 1,
        'rejected': 0,
        'needs_more_evidence': 0,
    }


@pytest.mark.parametrize(
    ('statuses', 'expected_counts'),
    [
        (
            ['pending_review', 'needs_more_evidence'],
            {
                'pending_review': 1,
                'approved': 0,
                'rejected': 0,
                'needs_more_evidence': 1,
            },
        ),
        (
            ['needs_more_evidence'],
            {
                'pending_review': 0,
                'approved': 0,
                'rejected': 0,
                'needs_more_evidence': 1,
            },
        ),
    ],
)
def test_needs_more_terminal_reconciles_with_task6_precedence(
    db_session: Session,
    application_session_factory,
    service_parts,
    statuses: list[str],
    expected_counts: dict[str, int],
) -> None:
    draft = _FakeDraftService(
        application_session_factory,
        candidate_count=len(statuses),
    )
    service = _service(application_session_factory, service_parts, draft=draft)
    _, started = _start(db_session, service)
    _set_item_statuses(
        application_session_factory,
        started.thread_id,
        statuses,
    )
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    checkpoint_thread_id = thread.checkpoint_thread_id
    saver = service_parts[1].saver
    assert saver is not None
    graph = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )(saver)
    invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input=Command(resume={
            'event': 'review_resolution_checked',
            'state_version': thread.state_version,
        }),
        checkpoint_thread_id=checkpoint_thread_id,
        runtime_context=service._runtime_context(_actor()),
        expect_interrupt=False,
    )
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()

    reconciled = service.status(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert reconciled.status == 'needs_more_evidence'
    assert current.checkpoint_thread_id == checkpoint_thread_id
    snapshot = graph.get_state(checkpoint_config(checkpoint_thread_id))
    assert set(snapshot.values) == {
        'workflow_thread_id',
        'graph_version',
        'input_hash',
        'evidence_version_hash',
        'review_status_counts',
        'phase',
        'completed_nodes',
        'error_codes',
        'status',
        'review_item_count',
    }
    assert 'review_item_ids' not in snapshot.values
    assert snapshot.values['review_status_counts'] == expected_counts
    assert snapshot.values['status'] == 'needs_more_evidence'
    assert snapshot.values['phase'] == 'needs_more_evidence'
    assert snapshot.values['completed_nodes'] == [
        'validate_input',
        'collect_evidence_refs',
        'plan_agent_runs',
        'draft_review_candidates_transaction',
        'route_review_boundary',
        'await_human_review',
        'verify_review_resolution_from_postgres',
        'finalize_needs_more_evidence',
    ]


def test_pending_approved_terminal_is_corrupt_and_rotates(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = _FakeDraftService(application_session_factory, candidate_count=2)
    service = _service(application_session_factory, service_parts, draft=draft)
    _, started = _start(db_session, service)
    _set_item_statuses(
        application_session_factory,
        started.thread_id,
        ['approved', 'approved'],
    )
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    saver = service_parts[1].saver
    assert saver is not None
    real_builder = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )
    invoke_and_confirm_checkpoint(
        graph=real_builder(saver),
        saver=saver,
        command_or_input=Command(resume={
            'event': 'review_resolution_checked',
            'state_version': thread.state_version,
        }),
        checkpoint_thread_id=old_checkpoint_thread_id,
        runtime_context=service._runtime_context(_actor()),
        expect_interrupt=False,
    )
    _set_item_statuses(
        application_session_factory,
        started.thread_id,
        ['pending_review', 'approved'],
    )
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()

    def terminal_builder(checkpointer: object) -> _SnapshotMutatingGraph:
        return _SnapshotMutatingGraph(
            real_builder(checkpointer),
            checkpoint_thread_id=old_checkpoint_thread_id,
            shape_corruption='pending_approved_terminal_counts',
        )

    monkeypatch.setattr(
        service,
        '_resolve_builder',
        lambda _projection: terminal_builder,
    )

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert current is not None
    assert repaired.status == 'awaiting_human_review'
    assert current.checkpoint_thread_id != old_checkpoint_thread_id
    snapshot = real_builder(saver).get_state(
        checkpoint_config(current.checkpoint_thread_id)
    )
    assert set(snapshot.values) == {
        'workflow_thread_id',
        'graph_version',
        'input_hash',
        'evidence_version_hash',
        'review_status_counts',
        'phase',
        'completed_nodes',
        'error_codes',
    }
    assert 'review_item_ids' not in snapshot.values
    assert snapshot.values['review_status_counts'] == {
        'pending_review': 1,
        'approved': 1,
        'rejected': 0,
        'needs_more_evidence': 0,
    }
    assert snapshot.values['phase'] == 'review_boundary_routed'
    assert snapshot.values['completed_nodes'] == [
        'validate_input',
        'collect_evidence_refs',
        'plan_agent_runs',
        'draft_review_candidates_transaction',
        'route_review_boundary',
    ]


@pytest.mark.parametrize(
    ('live_status', 'forged_terminal_status'),
    [
        ('pending_review', 'completed'),
        ('needs_more_evidence', 'completed'),
        ('approved', 'needs_more_evidence'),
    ],
)
def test_checkpoint_terminal_shape_must_match_live_review_rows(
    db_session: Session,
    application_session_factory,
    service_parts,
    monkeypatch: pytest.MonkeyPatch,
    live_status: str,
    forged_terminal_status: str,
) -> None:
    service = _service(application_session_factory, service_parts)
    _, started = _start(db_session, service)
    _set_item_statuses(
        application_session_factory,
        started.thread_id,
        [live_status],
    )
    thread = db_session.get(AgentWorkflowThread, started.thread_id)
    assert thread is not None
    old_checkpoint_thread_id = thread.checkpoint_thread_id
    thread.status = 'checkpoint_failed'
    thread.state_version += 1
    db_session.commit()
    real_builder = service_parts[2].resolve(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    )

    def terminal_builder(saver: object) -> _SnapshotMutatingGraph:
        return _SnapshotMutatingGraph(
            real_builder(saver),
            checkpoint_thread_id=old_checkpoint_thread_id,
            terminal_status=forged_terminal_status,
        )

    monkeypatch.setattr(
        service,
        '_resolve_builder',
        lambda _projection: terminal_builder,
    )

    repaired = service.resume(actor=_actor(), thread_id=started.thread_id)

    db_session.expire_all()
    current = db_session.get(AgentWorkflowThread, started.thread_id)
    assert repaired.status == 'awaiting_human_review'
    assert current is not None
    assert current.checkpoint_thread_id != old_checkpoint_thread_id


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
