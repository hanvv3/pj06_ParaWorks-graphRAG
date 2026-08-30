from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.agent_runtime.checkpoint_execution import (
    checkpoint_config,
    invoke_and_confirm_checkpoint,
)
from backend.app.agent_runtime.checkpointing import (
    build_strict_checkpoint_serializer,
)
from backend.app.agent_runtime.contracts import AgentManifest
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.review_v21_graph import (
    _branch,
    build_company_memory_review_v21_graph,
)
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    ReviewItem,
)


@dataclass
class _Extraction:
    session_factory: Callable[[], Session]
    statuses: tuple[str, ...]
    calls: int = 0

    def draft(self, *, workflow_thread_id: str, **_kwargs) -> None:
        self.calls += 1
        with self.session_factory() as db, db.begin():
            if db.scalar(select(ReviewItem.id).where(
                ReviewItem.workflow_thread_id == workflow_thread_id
            )) is not None:
                return
            for index, status in enumerate(self.statuses):
                db.add(ReviewItem(
                    item_type='history_event',
                    payload={'title': f'candidate-{index}'},
                    source_links=['https://example.test/source'],
                    source_snippets=['bounded evidence'],
                    confidence_score=0.99,
                    permission_level='internal',
                    status=status,
                    workflow_thread_id=workflow_thread_id,
                    candidate_key=f'candidate-{index}',
                ))


@dataclass
class _Validation:
    session_factory: Callable[[], Session]
    transitions: tuple[str, ...] = ()
    calls: int = 0

    def run_auto_review(self, *, workflow_thread_id: str, **_kwargs) -> None:
        self.calls += 1
        if not self.transitions:
            return
        with self.session_factory() as db, db.begin():
            items = tuple(db.scalars(select(ReviewItem).where(
                ReviewItem.workflow_thread_id == workflow_thread_id
            ).order_by(ReviewItem.id)).all())
            for item, status in zip(items, self.transitions, strict=True):
                item.status = status


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(AgentManifest(
        name='history_agent',
        owner='developer-c',
        input_contract='EvidencePacket',
        output_contract='ReviewCandidate',
        prompt_versions=('history-v1',),
        supported_permissions=('public', 'internal'),
        capabilities=('history_candidate',),
    ))
    return registry


def _seed(factory: Callable[[], Session]) -> tuple[str, str]:
    thread_id = uuid4().hex
    checkpoint_id = f'review-v21:{uuid4().hex}'
    with factory() as db, db.begin():
        db.add(AgentWorkflowThread(
            thread_id=thread_id,
            workflow_name='company-memory-review',
            graph_version='company-memory-review-v2.1-auto-review',
            checkpoint_thread_id=checkpoint_id,
            checkpoint_store='memory',
            owner_subject_id='owner-1',
            security_scope_id='default',
            input_hash='a' * 64,
            evidence_version_hash='b' * 64,
            status='created',
            state_version=7,
        ))
        db.add(AgentWorkflowRequest(
            workflow_thread_id=thread_id,
            input_schema_version='review-source-versions:v1',
            agent_names=['history_agent'],
            selection_policy_version='company-memory-review-selection:v1',
            input_hash='a' * 64,
            fingerprint_key_version='test-v1',
        ))
        db.add(AgentWorkflowEvidenceRef(
            workflow_thread_id=thread_id,
            ordinal=0,
            canonical_source_type='gmail',
            canonical_table='sources',
            canonical_row_id=1,
            document_version_id=None,
            external_revision='rev-1',
            content_signature='sig-1',
            permission_level_snapshot='internal',
            content_fingerprint='c' * 64,
        ))
    return thread_id, checkpoint_id


def _run(
    db_session: Session,
    *,
    statuses: Sequence[str],
    transitions: Sequence[str] = (),
):
    db_session.commit()
    factory = sessionmaker(bind=db_session.get_bind(), expire_on_commit=False)
    thread_id, checkpoint_id = _seed(factory)
    extraction = _Extraction(factory, tuple(statuses))
    validation = _Validation(factory, tuple(transitions))
    saver = InMemorySaver(serde=build_strict_checkpoint_serializer())
    graph = build_company_memory_review_v21_graph(saver)
    result = invoke_and_confirm_checkpoint(
        graph=graph,
        saver=saver,
        command_or_input={'workflow_thread_id': thread_id},
        checkpoint_thread_id=checkpoint_id,
        runtime_context={
            'session_factory': factory,
            'permission_resolver': (
                lambda owner: ('public', 'internal') if owner == 'owner-1' else ()
            ),
            'agent_registry': _registry(),
            'extraction_coordinator': extraction,
            'validation_coordinator': validation,
        },
        expect_interrupt=bool(
            (transitions or statuses)
            and 'pending_review' in (transitions or statuses)
        ),
    )
    return graph, saver, checkpoint_id, result, extraction, validation


def test_v21_graph_uses_real_stategraph_auto_review_refresh_and_interrupt(
    db_session: Session,
) -> None:
    graph, _, _, result, extraction, validation = _run(
        db_session, statuses=('pending_review',)
    )

    nodes = set(graph.get_graph().nodes)
    assert {
        'draft_review_candidates_transaction',
        'run_auto_review',
        'refresh_review_resolution',
        'await_human_review',
    } <= nodes
    assert result.interrupted is True
    assert extraction.calls == validation.calls == 1


def test_all_auto_resolved_completes_without_interrupt(db_session: Session) -> None:
    _, _, _, result, _, _ = _run(
        db_session,
        statuses=('pending_review',),
        transitions=('approved',),
    )

    assert result.interrupted is False
    assert result.result['status'] == 'completed'
    assert result.result['review_status_counts']['approved'] == 1


def test_zero_candidates_uses_distinct_finalize_node_and_status(
    db_session: Session,
) -> None:
    graph, saver, checkpoint_id, result, _, _ = _run(db_session, statuses=())

    snapshot = graph.get_state(checkpoint_config(checkpoint_id))
    assert result.result['status'] == 'completed'
    assert 'finalize_no_candidates' in snapshot.values['completed_nodes']


def test_pending_takes_priority_over_needs_more_evidence() -> None:
    assert _branch({'review_status_counts': {
        'pending_review': 1,
        'approved': 0,
        'rejected': 0,
        'needs_more_evidence': 1,
        'revoked': 0,
    }}) == 'human'


def test_revoked_is_resolved_but_not_approved(db_session: Session) -> None:
    _, _, _, result, _, _ = _run(db_session, statuses=('revoked',))

    assert result.interrupted is False
    assert result.result['review_status_counts'] == {
        'pending_review': 0,
        'approved': 0,
        'rejected': 0,
        'needs_more_evidence': 0,
        'revoked': 1,
    }
