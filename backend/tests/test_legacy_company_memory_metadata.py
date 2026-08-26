from sqlalchemy.orm import Session

from backend.app.agent_runtime.company_memory import (
    run_company_memory_agent_orchestration,
)
from backend.app.core.demo_auth import USERS


def test_legacy_run_describes_review_queue_metadata_without_pause_or_resume(db_session: Session) -> None:
    result = run_company_memory_agent_orchestration(
        db=db_session,
        user=USERS['admin'],
        question='',
    )

    checkpoint = result.outputs['hitl_checkpoint']
    assert result.outputs['review_boundary'] == 'metadata_only'
    assert checkpoint == {
        'checkpoint_type': 'review_queue_metadata',
        'node_name': 'draft_review_candidates',
        'status': 'metadata_only',
        'review_item_ids': [],
        'resume_from_node': None,
        'resume_policy': 'not_resumable',
        'required_review_statuses': ['approved', 'rejected', 'needs_more_evidence'],
        'trusted_knowledge_requires_approval': True,
        'paid_llm_calls': False,
    }
