from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.checkpoint_execution import (
    checkpoint_config,
    invoke_and_confirm_checkpoint,
    require_resumable_checkpoint,
)
from backend.app.agent_runtime.checkpointing import (
    build_strict_checkpoint_serializer,
)
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v2_drafting import ReviewDraftResult
from backend.app.agent_runtime.review_v2_graph import (
    build_company_memory_review_v2_graph,
)
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.review import ReviewItem
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
)

_STATE_VERSION = 73
_STATUS_NAMES = (
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
)
_SENSITIVE_MARKERS = frozenset({
    'gmail:message-sensitive-78451',
    'https://restricted.example.test/review-source-marker-78451',
    'restricted-source-snippet-marker-78451',
    'sensitive-external-revision-78451',
    'sensitive-content-signature-78451',
    'provider-prompt-marker-78451',
    'model-output-marker-78451',
    'credential-marker-78451',
    'raw-provider-error-marker-78451',
})


@pytest.fixture
def in_memory_saver() -> InMemorySaver:
    return InMemorySaver(serde=build_strict_checkpoint_serializer())


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


@dataclass
class _PermissionResolver:
    levels: tuple[str, ...] = ('public', 'internal')
    calls: list[str] = field(default_factory=list)

    def __call__(self, actor_subject_id: str) -> Sequence[str]:
        self.calls.append(actor_subject_id)
        return self.levels


class _UnusedLeaseService:
    def acquire(self, *, workflow_thread_id: str) -> str:
        raise AssertionError(
            f'graph must not acquire a second draft lease: {workflow_thread_id}'
        )


@dataclass
class _FakeDraftService:
    session_factory: Callable[[], Session]
    candidate_count: int
    draft_calls: int = 0
    call_arguments: list[dict[str, object]] = field(default_factory=list)

    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewDraftResult:
        self.draft_calls += 1
        self.call_arguments.append({
            'workflow_thread_id': workflow_thread_id,
            'actor_subject_id': actor_subject_id,
            'allowed_permission_levels': tuple(allowed_permission_levels),
        })
        with self.session_factory() as db:
            existing = tuple(
                db.scalars(
                    select(ReviewItem)
                    .where(ReviewItem.workflow_thread_id == workflow_thread_id)
                    .order_by(ReviewItem.id)
                ).all()
            )
            if not existing:
                for index in range(self.candidate_count):
                    db.add(
                        ReviewItem(
                            item_type='history_event',
                            payload={
                                'title': f'검토 후보 {index + 1}',
                                'summary': '사람의 검토가 필요한 후보입니다.',
                                'source_ids': [
                                    'gmail:message-sensitive-78451'
                                ],
                                'prompt': 'provider-prompt-marker-78451',
                                'model_output': 'model-output-marker-78451',
                                'api_key': 'credential-marker-78451',
                                'provider_error': (
                                    'raw-provider-error-marker-78451'
                                ),
                            },
                            source_links=[
                                'https://restricted.example.test/'
                                'review-source-marker-78451'
                            ],
                            source_snippets=[
                                'restricted-source-snippet-marker-78451'
                            ],
                            confidence_score=0.91,
                            permission_level='internal',
                            status='pending_review',
                            workflow_thread_id=workflow_thread_id,
                            candidate_key=f'candidate-{index + 1}',
                        )
                    )
                db.flush()
                existing = tuple(
                    db.scalars(
                        select(ReviewItem)
                        .where(
                            ReviewItem.workflow_thread_id
                            == workflow_thread_id
                        )
                        .order_by(ReviewItem.id)
                    ).all()
                )
            db.commit()
            ids = tuple(item.id for item in existing)
            counts = dict.fromkeys(_STATUS_NAMES, 0)
            for item in existing:
                counts[item.status] += 1
            return ReviewDraftResult(
                review_item_ids=ids,
                review_status_counts=counts,  # type: ignore[arg-type]
            )


@dataclass(frozen=True)
class _WorkflowFixture:
    workflow_thread_id: str
    checkpoint_thread_id: str


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentManifest(
            name='history_agent',
            owner='developer-c',
            input_contract='EvidencePacket',
            output_contract='ReviewCandidate',
            prompt_versions=('history-v1',),
            supported_permissions=('public', 'internal', 'restricted'),
            capabilities=('history_candidate',),
        )
    )
    return registry


def _seed_workflow(
    session_factory: Callable[[], Session],
) -> _WorkflowFixture:
    workflow_thread_id = uuid4().hex
    checkpoint_thread_id = f'review-v2:{uuid4().hex}'
    with session_factory() as db:
        db.add(
            AgentWorkflowThread(
                thread_id=workflow_thread_id,
                workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
                graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
                checkpoint_thread_id=checkpoint_thread_id,
                checkpoint_store='memory',
                owner_subject_id='actor-1',
                security_scope_id='scope-review-v2-graph',
                input_hash='a' * 64,
                evidence_version_hash='b' * 64,
                status='created',
                state_version=_STATE_VERSION,
            )
        )
        db.add(
            AgentWorkflowRequest(
                workflow_thread_id=workflow_thread_id,
                input_schema_version=COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
                request_kind='review_source_versions',
                agent_names=['history_agent'],
                selection_policy_version=(
                    COMPANY_MEMORY_SELECTION_POLICY_VERSION
                ),
                input_hash='a' * 64,
                fingerprint_key_version='test-v1',
            )
        )
        db.add(
            AgentWorkflowEvidenceRef(
                workflow_thread_id=workflow_thread_id,
                ordinal=0,
                canonical_source_type='gmail',
                canonical_table='sources',
                canonical_row_id=991,
                document_version_id=882,
                external_revision='sensitive-external-revision-78451',
                content_signature='sensitive-content-signature-78451',
                permission_level_snapshot='internal',
                content_fingerprint='c' * 64,
            )
        )
        db.commit()
    return _WorkflowFixture(
        workflow_thread_id=workflow_thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
    )


def _runtime_context(
    session_factory: Callable[[], Session],
    draft_service: _FakeDraftService,
    permission_resolver: _PermissionResolver | None = None,
) -> tuple[dict[str, object], _PermissionResolver]:
    resolver = permission_resolver or _PermissionResolver()
    return (
        {
            'session_factory': session_factory,
            'actor_subject_id': 'actor-1',
            'permission_resolver': resolver,
            'agent_registry': _registry(),
            'draft_service': draft_service,
            'lease_service': _UnusedLeaseService(),
        },
        resolver,
    )


def _pause_candidate_run(
    *,
    saver: InMemorySaver,
    session_factory: Callable[[], Session],
    candidate_count: int = 1,
) -> tuple[object, _WorkflowFixture, _FakeDraftService, dict[str, object]]:
    workflow = _seed_workflow(session_factory)
    draft_service = _FakeDraftService(
        session_factory=session_factory,
        candidate_count=candidate_count,
    )
    runtime_context, _ = _runtime_context(session_factory, draft_service)
    graph = build_company_memory_review_v2_graph(saver)
    paused = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input={
            'workflow_thread_id': workflow.workflow_thread_id,
        },
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=True,
    )
    assert paused.interrupted is True
    assert tuple(paused.result['__interrupt__'])[0].value == {
        'event': 'review_resolution_required',
        'state_version': _STATE_VERSION,
    }
    return graph, workflow, draft_service, runtime_context


def _set_review_statuses(
    session_factory: Callable[[], Session],
    workflow_thread_id: str,
    statuses: Sequence[str],
) -> list[int]:
    with session_factory() as db:
        items = list(
            db.scalars(
                select(ReviewItem)
                .where(ReviewItem.workflow_thread_id == workflow_thread_id)
                .order_by(ReviewItem.id)
            ).all()
        )
        assert len(items) == len(statuses)
        for item, status in zip(items, statuses, strict=True):
            item.status = status
        ids = [item.id for item in items]
        db.commit()
        return ids


def _resume_acknowledgement() -> dict[str, object]:
    return {
        'event': 'review_resolution_checked',
        'state_version': _STATE_VERSION,
    }


def _walk_values(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_values(key)
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    elif isinstance(value, Interrupt):
        yield from _walk_values(value.value)


def test_no_candidate_run_finishes_without_interrupt(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    workflow = _seed_workflow(application_session_factory)
    draft_service = _FakeDraftService(
        session_factory=application_session_factory,
        candidate_count=0,
    )
    runtime_context, _ = _runtime_context(
        application_session_factory,
        draft_service,
    )
    graph = build_company_memory_review_v2_graph(in_memory_saver)

    confirmation = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input={
            'workflow_thread_id': workflow.workflow_thread_id,
        },
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=False,
    )

    assert confirmation.result == {
        'workflow_thread_id': workflow.workflow_thread_id,
        'status': 'completed',
        'review_item_count': 0,
        'review_status_counts': dict.fromkeys(_STATUS_NAMES, 0),
        'error_codes': [],
    }
    assert draft_service.draft_calls == 1


def test_candidate_run_interrupts_with_exact_safe_payload(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    _, workflow, draft_service, _ = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
    )

    assert draft_service.draft_calls == 1
    with application_session_factory() as db:
        thread = db.get(AgentWorkflowThread, workflow.workflow_thread_id)
        assert thread is not None
        assert thread.state_version == _STATE_VERSION
        assert thread.status == 'created'
        assert thread.checkpoint_confirmed_at is None
        assert db.scalar(select(func.count(ReviewItem.id))) == 1


def test_resume_uses_same_thread_and_acknowledgement_only(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    graph, workflow, draft_service, runtime_context = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
    )
    _set_review_statuses(
        application_session_factory,
        workflow.workflow_thread_id,
        ['approved'],
    )

    assert require_resumable_checkpoint(
        in_memory_saver,
        workflow.checkpoint_thread_id,
    ) == checkpoint_config(workflow.checkpoint_thread_id)
    resumed = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input=Command(resume=_resume_acknowledgement()),
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=False,
    )

    assert resumed.checkpoint_thread_id == workflow.checkpoint_thread_id
    assert resumed.checkpoint_ns == ''
    assert resumed.result['status'] == 'completed'
    assert draft_service.draft_calls == 1


def test_pending_database_items_reinterrupt_defensively(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    graph, workflow, draft_service, runtime_context = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
    )

    resumed = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input=Command(resume=_resume_acknowledgement()),
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=True,
    )

    pending_interrupt = tuple(resumed.result['__interrupt__'])[0]
    assert pending_interrupt.value == {
        'event': 'review_resolution_required',
        'state_version': _STATE_VERSION,
    }
    assert draft_service.draft_calls == 1


def test_needs_more_evidence_finishes_in_terminal_branch(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    graph, workflow, _, runtime_context = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
    )
    _set_review_statuses(
        application_session_factory,
        workflow.workflow_thread_id,
        ['needs_more_evidence'],
    )

    resumed = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input=Command(resume=_resume_acknowledgement()),
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=False,
    )

    assert resumed.result == {
        'workflow_thread_id': workflow.workflow_thread_id,
        'status': 'needs_more_evidence',
        'review_item_count': 1,
        'review_status_counts': {
            **dict.fromkeys(_STATUS_NAMES, 0),
            'needs_more_evidence': 1,
        },
        'error_codes': [],
    }


def test_approved_and_rejected_items_finish_completed(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    graph, workflow, _, runtime_context = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
        candidate_count=2,
    )
    _set_review_statuses(
        application_session_factory,
        workflow.workflow_thread_id,
        ['approved', 'rejected'],
    )

    resumed = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input=Command(resume=_resume_acknowledgement()),
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=False,
    )

    assert resumed.result['status'] == 'completed'
    assert resumed.result['review_item_count'] == 2
    assert resumed.result['review_status_counts'] == {
        'pending_review': 0,
        'approved': 1,
        'rejected': 1,
        'needs_more_evidence': 0,
    }


def test_database_resolution_ignores_resume_payload_claims(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    resolver = _PermissionResolver()
    workflow = _seed_workflow(application_session_factory)
    draft_service = _FakeDraftService(
        session_factory=application_session_factory,
        candidate_count=1,
    )
    runtime_context, _ = _runtime_context(
        application_session_factory,
        draft_service,
        permission_resolver=resolver,
    )
    graph = build_company_memory_review_v2_graph(in_memory_saver)
    invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input={
            'workflow_thread_id': workflow.workflow_thread_id,
        },
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=True,
    )
    _set_review_statuses(
        application_session_factory,
        workflow.workflow_thread_id,
        ['rejected'],
    )
    permission_reads_before_resume = len(resolver.calls)

    resumed = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input=Command(resume={
            **_resume_acknowledgement(),
            'status': 'approved',
            'approved': True,
        }),
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=False,
    )

    assert resumed.result['review_status_counts']['approved'] == 0
    assert resumed.result['review_status_counts']['rejected'] == 1
    assert len(resolver.calls) > permission_reads_before_resume


def test_checkpoint_contains_no_evidence_prompt_or_model_output(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    _, workflow, _, _ = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
    )

    saved = in_memory_saver.get_tuple(
        checkpoint_config(workflow.checkpoint_thread_id)
    )
    assert saved is not None
    values = tuple(_walk_values((saved.checkpoint, saved.pending_writes)))
    checkpoint_strings = {value for value in values if isinstance(value, str)}
    assert checkpoint_strings.isdisjoint(_SENSITIVE_MARKERS)
    assert checkpoint_strings.isdisjoint({
        'source_id',
        'source_ids',
        'source_url',
        'source_snippet',
        'prompt',
        'model_output',
        'api_key',
        'oauth_token',
        'provider_error',
        'raw_error',
    })
    assert saved.checkpoint['channel_values']['review_item_ids'] == []


def test_graph_has_no_rag_slack_neo4j_or_trusted_knowledge_node(
    in_memory_saver: InMemorySaver,
) -> None:
    graph = build_company_memory_review_v2_graph(in_memory_saver)

    node_names = set(graph.get_graph().nodes) - {'__start__', '__end__'}
    assert node_names == {
        'validate_input',
        'collect_evidence_refs',
        'plan_agent_runs',
        'draft_review_candidates_transaction',
        'route_review_boundary',
        'finalize_no_candidates',
        'await_human_review',
        'verify_review_resolution_from_postgres',
        'finalize_needs_more_evidence',
        'finalize_review_trace',
    }
    assert not any(
        forbidden in node_name.lower()
        for node_name in node_names
        for forbidden in ('rag', 'slack', 'neo4j', 'trusted_knowledge')
    )
    assert {
        (edge.source, edge.target, edge.conditional)
        for edge in graph.get_graph().edges
    } == {
        ('__start__', 'validate_input', False),
        ('validate_input', 'collect_evidence_refs', False),
        ('collect_evidence_refs', 'plan_agent_runs', False),
        ('plan_agent_runs', 'draft_review_candidates_transaction', False),
        (
            'draft_review_candidates_transaction',
            'route_review_boundary',
            False,
        ),
        ('route_review_boundary', 'finalize_no_candidates', True),
        ('route_review_boundary', 'await_human_review', True),
        (
            'await_human_review',
            'verify_review_resolution_from_postgres',
            False,
        ),
        (
            'verify_review_resolution_from_postgres',
            'await_human_review',
            True,
        ),
        (
            'verify_review_resolution_from_postgres',
            'finalize_needs_more_evidence',
            True,
        ),
        (
            'verify_review_resolution_from_postgres',
            'finalize_review_trace',
            True,
        ),
        ('finalize_no_candidates', '__end__', False),
        ('finalize_needs_more_evidence', '__end__', False),
        ('finalize_review_trace', '__end__', False),
    }


def test_candidate_set_is_immutable_after_confirmed_pause(
    in_memory_saver: InMemorySaver,
    application_session_factory: sessionmaker[Session],
) -> None:
    graph, workflow, draft_service, runtime_context = _pause_candidate_run(
        saver=in_memory_saver,
        session_factory=application_session_factory,
    )
    draft_service.candidate_count = 5

    invoke_and_confirm_checkpoint(
        graph=graph,
        saver=in_memory_saver,
        command_or_input=Command(resume=_resume_acknowledgement()),
        checkpoint_thread_id=workflow.checkpoint_thread_id,
        runtime_context=runtime_context,
        expect_interrupt=True,
    )

    with application_session_factory() as db:
        assert db.scalar(
            select(func.count(ReviewItem.id)).where(
                ReviewItem.workflow_thread_id == workflow.workflow_thread_id
            )
        ) == 1
    assert draft_service.draft_calls == 1
