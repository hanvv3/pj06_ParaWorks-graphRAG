from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.agent_runtime import AgentWorkflowState, build_company_memory_workflow
from backend.app.agent_runtime.company_memory import (
    DEFAULT_AGENT_RUN_BUDGET_USD,
    run_company_memory_agent_orchestration,
)
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.db.session import get_db
from backend.app.services.audit import record_audit_log

router = APIRouter(prefix='/orchestration', tags=['orchestration'])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]


class CompanyMemoryDryRunRequest(BaseModel):
    objective: str = 'answer_from_company_memory'
    question: str = ''


class CompanyMemoryRunRequest(BaseModel):
    question: str = ''


def _cost_policy_response() -> dict:
    return {
        'delta_sync': True,
        'source_hash_skip': True,
        'evidence_cache_reuse': True,
        'evidence_token_budget': True,
        'per_run_budget_usd': DEFAULT_AGENT_RUN_BUDGET_USD,
        'budget_actions': ['run', 'skip', 'use_cache'],
        'paid_llm_calls_in_status_api': False,
        'requires_explicit_run': True,
        'hitl_checkpointing': False,
        'checkpoint_store': 'none',
        'review_boundary': 'metadata_only',
        'trusted_knowledge_requires_approval': True,
    }


@router.get('/company-memory')
def get_company_memory_orchestration() -> dict:
    workflow = build_company_memory_workflow()

    return {
        'workflow_name': 'company_memory',
        'backend': workflow.backend,
        'node_names': workflow.node_names,
        'graph_mermaid': workflow.graph_mermaid,
        'cost_policy': _cost_policy_response(),
    }


@router.post('/company-memory/dry-run')
def dry_run_company_memory_orchestration(request: CompanyMemoryDryRunRequest) -> dict:
    workflow = build_company_memory_workflow()
    result = workflow.run(
        AgentWorkflowState(
            objective=request.objective,
            inputs={'question': request.question},
        )
    )

    return {
        'workflow_name': 'company_memory',
        'backend': workflow.backend,
        'objective': result.objective,
        'inputs': result.inputs,
        'completed_nodes': result.completed_nodes,
        'outputs': result.outputs,
        'token_cost_usd': 0,
        'cost_policy': _cost_policy_response(),
    }


@router.post('/company-memory/run')
def run_company_memory_orchestration(
    request: CompanyMemoryRunRequest,
    db: DbSession,
    user: CurrentUser,
) -> dict:
    result = run_company_memory_agent_orchestration(
        db=db,
        user=user,
        question=request.question,
    )
    record_audit_log(
        db=db,
        actor=user,
        action='orchestration.company_memory.run',
        target_type='workflow',
        target_id='company_memory',
        metadata={
            'backend': result.backend,
            'completed_nodes': result.completed_nodes,
            'slack_review_items_created': result.outputs.get('slack_review_items_created', 0),
            'mail_document_review_items_created': result.outputs.get('mail_document_review_items_created', 0),
            'rag_agent_run_created': result.outputs.get('rag_agent_run_created', False),
            'hitl_checkpoint_status': (result.outputs.get('hitl_checkpoint') or {}).get('status'),
            'hitl_checkpoint_review_item_count': len(
                (result.outputs.get('hitl_checkpoint') or {}).get('review_item_ids') or []
            ),
        },
    )
    db.commit()

    return {
        'workflow_name': 'company_memory',
        'backend': result.backend,
        'completed_nodes': result.completed_nodes,
        'outputs': result.outputs,
        'cost_policy': _cost_policy_response(),
    }
