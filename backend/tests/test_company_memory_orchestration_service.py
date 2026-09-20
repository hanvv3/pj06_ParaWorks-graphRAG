from sqlalchemy.orm import Session

from backend.app.agent_runtime.company_memory import (
    run_company_memory_agent_orchestration,
)
from backend.app.core.demo_auth import USERS
from backend.app.models import (
    AgentRun,
    Document,
    DocumentChunk,
    DocumentVersion,
    ReviewItem,
    Source,
)


def seed_chunk(db: Session, source_type: str, source_id: str, text: str, permission_level: str = 'internal') -> None:
    source = Source(
        source_type=source_type,
        source_id=source_id,
        source_url=f'https://{source_type}.mock/{source_id}',
        title=f'{source_type} evidence',
        author='owner@example.com',
        permission_level=permission_level,
        raw_metadata={'ts': '2026-05-02T09:00:00+09:00'},
    )
    db.add(source)
    db.flush()

    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()

    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()

    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text=text,
            source_snippet=text[:240],
            permission_level=permission_level,
            metadata_={'source_url': source.source_url, 'source_type': source_type},
        )
    )
    db.commit()


def test_company_memory_orchestration_runs_real_agent_services(db_session: Session) -> None:
    seed_chunk(
        db_session,
        'slack',
        'slack-redis',
        'Use Redis to support queue and job progress workflows.',
    )
    seed_chunk(
        db_session,
        'gmail',
        'gmail-redis',
        'PostgreSQL remains the durable source of record while Redis handles transient job state.',
    )

    result = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )

    assert result.backend == 'langgraph'
    assert result.completed_nodes == [
        'collect_evidence',
        'draft_review_candidates',
        'retrieve_company_memory',
        'answer_with_rag',
    ]
    assert result.outputs['slack_review_items_created'] == 1
    assert result.outputs['mail_document_review_items_created'] == 1
    assert result.outputs['memory_review_items_created'] == 4
    assert result.outputs['rag_agent_run_created'] is True
    assert result.outputs['review_boundary'] == 'metadata_only'
    assert result.outputs['hitl_checkpoint'] == {
        'checkpoint_type': 'review_queue_metadata',
        'node_name': 'draft_review_candidates',
        'status': 'metadata_only',
        'review_item_ids': [1, 2, 3, 4, 5, 6],
        'resume_from_node': None,
        'resume_policy': 'not_resumable',
        'required_review_statuses': ['approved', 'rejected', 'needs_more_evidence'],
        'trusted_knowledge_requires_approval': True,
        'paid_llm_calls': False,
    }
    assert result.outputs['token_budget_policy'] == 'delta_sync_hash_skip_evidence_budget'
    cost_plan = result.outputs['cost_plan']
    assert cost_plan['slack_agent']['action'] == 'run'
    assert cost_plan['slack_agent']['reason'] == 'slack_evidence_available'
    assert cost_plan['slack_agent']['source_window'] == 'orchestrated-slack:ranked:12'
    assert cost_plan['slack_agent']['selection_strategy'] == 'ranked'
    assert cost_plan['slack_agent']['evidence_message_count'] == 1
    assert cost_plan['slack_agent']['estimated_input_tokens'] == 13
    assert cost_plan['slack_agent']['budget_status'] == 'within_budget'
    assert cost_plan['mail_document_agent']['action'] == 'run'
    assert cost_plan['mail_document_agent']['reason'] == 'mail_document_evidence_available'
    assert cost_plan['mail_document_agent']['estimated_input_tokens'] == 22
    assert cost_plan['mail_document_agent']['budget_status'] == 'within_budget'
    assert cost_plan['rag_orchestrator_agent']['action'] == 'run'
    assert cost_plan['rag_orchestrator_agent']['reason'] == 'question_provided'
    assert cost_plan['rag_orchestrator_agent']['estimated_input_tokens'] == 3
    assert cost_plan['rag_orchestrator_agent']['budget_status'] == 'within_budget'

    agent_names = [run.agent_name for run in db_session.query(AgentRun).order_by(AgentRun.id).all()]
    assert agent_names == [
        'slack_agent',
        'mail_document_agent',
        'timeline_agent',
        'history_agent',
        'decision_record_agent',
        'todo_agent',
        'rag_orchestrator_agent',
    ]
    slack_run = db_session.query(AgentRun).filter(AgentRun.agent_name == 'slack_agent').one()
    assert slack_run.source_window == 'orchestrated-slack:ranked:12'
    assert slack_run.metadata_['selection_strategy'] == 'ranked'
    assert slack_run.metadata_['evidence_summary'][0]['rank'] == 1

    review_items = db_session.query(ReviewItem).order_by(ReviewItem.id).all()
    assert len(review_items) == 6
    assert {item.payload['agent_name'] for item in review_items} == {
        'slack_agent',
        'mail_document_agent',
        'timeline_agent',
        'history_agent',
        'decision_record_agent',
        'todo_agent',
    }


def test_company_memory_orchestration_marks_missing_evidence_as_cost_skips(db_session: Session) -> None:
    result = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='',
    )

    assert result.outputs['slack_review_items_created'] == 0
    assert result.outputs['mail_document_review_items_created'] == 0
    assert result.outputs['memory_review_items_created'] == 0
    assert result.outputs['rag_agent_run_created'] is False
    assert result.outputs['hitl_checkpoint']['status'] == 'metadata_only'
    assert result.outputs['hitl_checkpoint']['review_item_ids'] == []
    assert result.outputs['cost_plan']['slack_agent']['action'] == 'skip'
    assert result.outputs['cost_plan']['slack_agent']['reason'] == 'no_slack_evidence'
    assert result.outputs['cost_plan']['mail_document_agent']['action'] == 'skip'
    assert result.outputs['cost_plan']['mail_document_agent']['reason'] == 'no_mail_document_evidence'
    assert result.outputs['cost_plan']['rag_orchestrator_agent']['action'] == 'skip'
    assert result.outputs['cost_plan']['rag_orchestrator_agent']['reason'] == 'empty_question'
    assert db_session.query(AgentRun).count() == 0


def test_company_memory_orchestration_skips_agents_that_exceed_cost_budget(db_session: Session) -> None:
    seed_chunk(
        db_session,
        'slack',
        'slack-large-channel',
        'Use Redis with selective summarization to limit budget. ' * 3_000,
    )

    result = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )

    slack_plan = result.outputs['cost_plan']['slack_agent']
    assert slack_plan['action'] == 'skip'
    assert slack_plan['reason'] == 'budget_exceeded'
    assert slack_plan['budget_status'] == 'over_budget'
    assert slack_plan['estimated_cost_usd'] > slack_plan['budget_limit_usd']
    assert result.outputs['slack_review_items_created'] == 0
    assert result.outputs['rag_agent_run_created'] is True

    agent_names = [run.agent_name for run in db_session.query(AgentRun).order_by(AgentRun.id).all()]
    assert agent_names == ['rag_orchestrator_agent']


def test_company_memory_orchestration_uses_cache_when_evidence_is_unchanged(db_session: Session) -> None:
    seed_chunk(
        db_session,
        'slack',
        'slack-cache-redis',
        'Use Redis to support queue and job progress workflows.',
    )
    seed_chunk(
        db_session,
        'gmail',
        'gmail-cache-redis',
        'PostgreSQL remains durable while Redis handles transient job state.',
    )

    first = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )
    second = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='Redis job state',
    )

    assert first.outputs['cost_plan']['slack_agent']['action'] == 'run'
    assert first.outputs['cost_plan']['mail_document_agent']['action'] == 'run'
    assert first.outputs['cost_plan']['rag_orchestrator_agent']['action'] == 'run'
    assert second.outputs['cost_plan']['slack_agent']['action'] == 'use_cache'
    assert second.outputs['cost_plan']['slack_agent']['reason'] == 'cache_hit'
    assert second.outputs['cost_plan']['mail_document_agent']['action'] == 'use_cache'
    assert second.outputs['cost_plan']['mail_document_agent']['reason'] == 'cache_hit'
    assert second.outputs['cost_plan']['rag_orchestrator_agent']['action'] == 'use_cache'
    assert second.outputs['cost_plan']['rag_orchestrator_agent']['reason'] == 'cache_hit'
    assert second.outputs['slack_review_items_created'] == 0
    assert second.outputs['mail_document_review_items_created'] == 0
    assert second.outputs['memory_review_items_created'] == 0
    assert second.outputs['rag_agent_run_created'] is False
    assert db_session.query(AgentRun).count() == 7
    assert db_session.query(ReviewItem).count() == 6
