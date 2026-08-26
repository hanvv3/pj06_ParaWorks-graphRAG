from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime import (
    AgentWorkflowState,
    EvidencePacket,
    PermissionContext,
    TokenUsage,
    build_agent_workflow,
    build_evidence_cache_key,
    evaluate_agent_cost_budget,
)
from backend.app.agents.mail_document_agent import (
    MAIL_DOCUMENT_AGENT_NAME,
    MAIL_DOCUMENT_AGENT_PROMPT_VERSION,
    MAIL_DOCUMENT_SOURCE_TYPES,
    DeterministicMailDocumentAgentModel,
    MailDocumentAgent,
    build_mail_document_evidence_packet,
    create_mail_document_agent_review_items_for_changed_sources,
)
from backend.app.agents.memory_extraction_agent import (
    build_memory_extraction_evidence_packet,
    create_memory_extraction_review_items,
)
from backend.app.agents.rag_orchestrator_agent import (
    RAG_ORCHESTRATOR_AGENT_NAME,
    RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION,
    answer_question_with_rag,
    build_rag_evidence_packet,
    retrieve_matching_evidence_candidates,
)
from backend.app.agents.slack_agent import (
    SLACK_AGENT_NAME,
    SLACK_AGENT_PROMPT_VERSION,
    DeterministicSlackAgentModel,
    SlackAgent,
    build_slack_evidence_packet,
    create_slack_agent_review_items,
)
from backend.app.core.demo_auth import DemoUser
from backend.app.models import AgentRun, DocumentChunk, Source
from backend.app.permissions.service import can_access_permission

DEFAULT_AGENT_RUN_BUDGET_USD = 0.001
DEFAULT_INPUT_COST_PER_1M = 0.15
DEFAULT_OUTPUT_COST_PER_1M = 0.60
DEFAULT_ESTIMATED_OUTPUT_TOKENS = 32
ORCHESTRATED_SLACK_MAX_EVIDENCE_MESSAGES = 12
ORCHESTRATED_SLACK_SOURCE_WINDOW = f'orchestrated-slack:ranked:{ORCHESTRATED_SLACK_MAX_EVIDENCE_MESSAGES}'
ORCHESTRATED_MAIL_DOCUMENT_SOURCE_WINDOW = 'orchestrated-mail-docs:grouped'


@dataclass(frozen=True)
class CompanyMemoryOrchestrationResult:
    backend: str
    completed_nodes: list[str]
    outputs: dict


def run_company_memory_agent_orchestration(
    *,
    db: Session,
    user: DemoUser,
    question: str,
) -> CompanyMemoryOrchestrationResult:
    permission_context = _permission_context(user)
    cost_plan = build_company_memory_cost_plan(db=db, question=question, user=user)
    workflow = build_agent_workflow(
        (
            ('collect_evidence', _collect_evidence_node(db, cost_plan)),
            ('draft_review_candidates', _draft_review_candidates_node(db, permission_context, cost_plan)),
            ('retrieve_company_memory', _retrieve_company_memory_node()),
            ('answer_with_rag', _answer_with_rag_node(db, user, question, cost_plan)),
        )
    )
    result = workflow.run(
        AgentWorkflowState(
            objective='company_memory_agent_orchestration',
            inputs={'question': question, 'user_id': user.id},
        )
    )

    return CompanyMemoryOrchestrationResult(
        backend=workflow.backend,
        completed_nodes=result.completed_nodes,
        outputs=result.outputs,
    )


def build_company_memory_cost_plan(
    *,
    db: Session,
    question: str,
    user: DemoUser,
) -> dict[str, dict[str, float | int | str | None]]:
    permission_context = _permission_context(user)
    slack_packet = build_slack_evidence_packet(
        db=db,
        permission_context=permission_context,
        source_window=ORCHESTRATED_SLACK_SOURCE_WINDOW,
        max_messages=ORCHESTRATED_SLACK_MAX_EVIDENCE_MESSAGES,
        selection_strategy='ranked',
    )
    mail_document_packet = build_mail_document_evidence_packet(
        db=db,
        permission_context=permission_context,
        source_window=ORCHESTRATED_MAIL_DOCUMENT_SOURCE_WINDOW,
    )
    rag_packet = _build_planning_rag_packet(db=db, user=user, question=question, permission_context=permission_context)
    slack_token_estimate = _estimate_tokens_for_packet(slack_packet)
    mail_document_token_estimate = _estimate_tokens_for_packet(mail_document_packet)
    question_token_estimate = _estimate_tokens(question)

    return {
        'slack_agent': _agent_cost_plan(
            agent_name=SLACK_AGENT_NAME,
            prompt_version=SLACK_AGENT_PROMPT_VERSION,
            packet=slack_packet,
            db=db,
            has_input=bool(slack_packet.messages),
            run_reason='slack_evidence_available',
            skip_reason='no_slack_evidence',
            estimated_input_tokens=slack_token_estimate,
        ),
        'mail_document_agent': _agent_cost_plan(
            agent_name=MAIL_DOCUMENT_AGENT_NAME,
            prompt_version=MAIL_DOCUMENT_AGENT_PROMPT_VERSION,
            packet=mail_document_packet,
            db=db,
            has_input=bool(mail_document_packet.messages),
            run_reason='mail_document_evidence_available',
            skip_reason='no_mail_document_evidence',
            estimated_input_tokens=mail_document_token_estimate,
        ),
        'rag_orchestrator_agent': _agent_cost_plan(
            agent_name=RAG_ORCHESTRATOR_AGENT_NAME,
            prompt_version=RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION,
            packet=rag_packet,
            db=db,
            has_input=question_token_estimate > 0,
            run_reason='question_provided',
            skip_reason='empty_question',
            estimated_input_tokens=question_token_estimate,
        ),
    }


def _collect_evidence_node(db: Session, cost_plan: dict[str, dict[str, float | int | str | None]]):
    def collect_evidence(state: AgentWorkflowState) -> AgentWorkflowState:
        slack_count = _count_chunks(db=db, source_types=('slack',))
        mail_document_count = _count_chunks(db=db, source_types=('gmail', 'gmail_attachment', 'drive', 'calendar'))
        return state.complete_node(
            'collect_evidence',
            evidence_sources='slack,gmail,drive,approved_knowledge',
            slack_evidence_count=slack_count,
            mail_document_evidence_count=mail_document_count,
            token_budget_policy='delta_sync_hash_skip_evidence_budget',
            cost_plan=cost_plan,
        )

    return collect_evidence


def _draft_review_candidates_node(
    db: Session,
    permission_context: PermissionContext,
    cost_plan: dict[str, dict[str, float | int | str | None]],
):
    def draft_review_candidates(state: AgentWorkflowState) -> AgentWorkflowState:
        slack_items = []
        if cost_plan['slack_agent']['action'] == 'run':
            slack_items = create_slack_agent_review_items(
                db=db,
                agent=SlackAgent(model=DeterministicSlackAgentModel()),
                permission_context=permission_context,
                source_window=ORCHESTRATED_SLACK_SOURCE_WINDOW,
                max_messages=ORCHESTRATED_SLACK_MAX_EVIDENCE_MESSAGES,
                selection_strategy='ranked',
            )
        mail_document_items = []
        if cost_plan['mail_document_agent']['action'] == 'run':
            mail_document_items = create_mail_document_agent_review_items_for_changed_sources(
                db=db,
                agent=MailDocumentAgent(model=DeterministicMailDocumentAgentModel()),
                permission_context=permission_context,
                source_window=ORCHESTRATED_MAIL_DOCUMENT_SOURCE_WINDOW,
                source_ids=_mail_document_source_ids(db=db, permission_context=permission_context),
            )
        memory_items = []
        if slack_items or mail_document_items:
            memory_packet = build_memory_extraction_evidence_packet(
                db=db,
                permission_context=permission_context,
                source_window='orchestrated-memory:source-agent-output',
            )
            memory_items = create_memory_extraction_review_items(db=db, packet=memory_packet)
        return state.complete_node(
            'draft_review_candidates',
            review_boundary='metadata_only',
            slack_review_items_created=len(slack_items),
            mail_document_review_items_created=len(mail_document_items),
            memory_review_items_created=len(memory_items),
            hitl_checkpoint=_build_review_queue_hitl_checkpoint(
                review_items=[*slack_items, *mail_document_items, *memory_items],
            ),
        )

    return draft_review_candidates


def _retrieve_company_memory_node():
    def retrieve_company_memory(state: AgentWorkflowState) -> AgentWorkflowState:
        return state.complete_node(
            'retrieve_company_memory',
            retrieval_mode='vector_ready',
        )

    return retrieve_company_memory


def _answer_with_rag_node(
    db: Session,
    user: DemoUser,
    question: str,
    cost_plan: dict[str, dict[str, float | int | str | None]],
):
    def answer_with_rag(state: AgentWorkflowState) -> AgentWorkflowState:
        if cost_plan['rag_orchestrator_agent']['action'] != 'run':
            return state.complete_node(
                'answer_with_rag',
                orchestrator='rag_orchestrator_agent',
                rag_agent_run_created=False,
            )
        answer_question_with_rag(db=db, user=user, question=question)
        return state.complete_node(
            'answer_with_rag',
            orchestrator='rag_orchestrator_agent',
            rag_agent_run_created=True,
        )

    return answer_with_rag


def _count_chunks(*, db: Session, source_types: tuple[str, ...]) -> int:
    return (
        db.scalar(
            select(func.count(DocumentChunk.id))
            .join(Source, DocumentChunk.source_id == Source.id)
            .where(Source.source_type.in_(source_types))
        )
        or 0
    )


def _permission_context(user: DemoUser) -> PermissionContext:
    return PermissionContext(
        user_id=user.id,
        role=user.role,
        allowed_permission_levels=tuple(user.permission_levels),
    )


def _mail_document_source_ids(*, db: Session, permission_context: PermissionContext) -> list[str]:
    rows = db.scalars(
        select(Source.source_id)
        .join(DocumentChunk, DocumentChunk.source_id == Source.id)
        .where(Source.source_type.in_(MAIL_DOCUMENT_SOURCE_TYPES))
        .where(DocumentChunk.permission_level.in_(permission_context.allowed_permission_levels))
        .order_by(Source.id)
    ).all()
    seen: set[str] = set()
    source_ids: list[str] = []
    for source_id in rows:
        if source_id not in seen:
            source_ids.append(source_id)
            seen.add(source_id)
    return source_ids


def _estimate_tokens_for_sources(*, db: Session, source_types: tuple[str, ...]) -> int:
    rows = db.scalars(
        select(DocumentChunk.text)
        .join(Source, DocumentChunk.source_id == Source.id)
        .where(Source.source_type.in_(source_types))
    ).all()
    return sum(_estimate_tokens(text) for text in rows)


def _estimate_tokens_for_packet(packet: EvidencePacket) -> int:
    return sum(_estimate_tokens(message.text) for message in packet.messages)


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def _agent_cost_plan(
    *,
    agent_name: str,
    prompt_version: str,
    packet: EvidencePacket,
    db: Session,
    has_input: bool,
    run_reason: str,
    skip_reason: str,
    estimated_input_tokens: int,
) -> dict[str, float | int | str | None]:
    if not has_input:
        return {
            'action': 'skip',
            'reason': skip_reason,
            'source_window': packet.source_window,
            'selection_strategy': _selection_strategy(packet),
            'evidence_message_count': len(packet.messages),
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_output_tokens': 0,
            'estimated_cost_usd': 0.0,
            'budget_limit_usd': DEFAULT_AGENT_RUN_BUDGET_USD,
            'budget_status': 'no_input',
            'cache_hit': False,
            'cache_key': None,
        }

    cache_key = build_evidence_cache_key(packet, prompt_version)
    if _has_completed_cache_hit(db=db, agent_name=agent_name, prompt_version=prompt_version, cache_key=cache_key):
        return {
            'action': 'use_cache',
            'reason': 'cache_hit',
            'source_window': packet.source_window,
            'selection_strategy': _selection_strategy(packet),
            'evidence_message_count': len(packet.messages),
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_output_tokens': 0,
            'estimated_cost_usd': 0.0,
            'budget_limit_usd': DEFAULT_AGENT_RUN_BUDGET_USD,
            'budget_status': 'cached',
            'cache_hit': True,
            'cache_key': cache_key,
        }

    decision = evaluate_agent_cost_budget(
        model_name='planning-estimate',
        token_usage=TokenUsage(
            input_tokens=estimated_input_tokens,
            output_tokens=DEFAULT_ESTIMATED_OUTPUT_TOKENS,
        ),
        input_cost_per_1m=DEFAULT_INPUT_COST_PER_1M,
        output_cost_per_1m=DEFAULT_OUTPUT_COST_PER_1M,
        max_cost_usd=DEFAULT_AGENT_RUN_BUDGET_USD,
        cache_hit=False,
    )

    return {
        'action': decision.action,
        'reason': run_reason if decision.action == 'run' else decision.reason,
        'source_window': packet.source_window,
        'selection_strategy': _selection_strategy(packet),
        'evidence_message_count': len(packet.messages),
        'estimated_input_tokens': estimated_input_tokens,
        'estimated_output_tokens': DEFAULT_ESTIMATED_OUTPUT_TOKENS,
        'estimated_cost_usd': round(decision.estimated_cost_usd, 6),
        'budget_limit_usd': decision.budget_limit_usd,
        'budget_status': decision.budget_status,
        'cache_hit': decision.cache_hit,
        'cache_key': cache_key,
    }


def _selection_strategy(packet: EvidencePacket) -> str:
    if packet.messages:
        return str(packet.messages[0].metadata.get('selection_strategy') or 'standard')
    if ':ranked:' in packet.source_window:
        return 'ranked'
    return 'standard'


def _has_completed_cache_hit(*, db: Session, agent_name: str, prompt_version: str, cache_key: str) -> bool:
    return (
        db.scalar(
            select(func.count(AgentRun.id)).where(
                AgentRun.agent_name == agent_name,
                AgentRun.prompt_version == prompt_version,
                AgentRun.cache_key == cache_key,
                AgentRun.status == 'complete',
            )
        )
        or 0
    ) > 0


def _build_planning_rag_packet(
    *,
    db: Session,
    user: DemoUser,
    question: str,
    permission_context: PermissionContext,
) -> EvidencePacket:
    matching_candidates = retrieve_matching_evidence_candidates(db=db, question=question)
    visible_candidates = [
        candidate for candidate in matching_candidates if can_access_permission(user, candidate.permission_level)
    ]
    return build_rag_evidence_packet(
        candidates=visible_candidates,
        question=question,
        permission_context=permission_context,
    )


def _build_review_queue_hitl_checkpoint(*, review_items: list) -> dict:
    review_item_ids = [item.id for item in review_items if item.id is not None]
    return {
        'checkpoint_type': 'review_queue_metadata',
        'node_name': 'draft_review_candidates',
        'status': 'metadata_only',
        'review_item_ids': review_item_ids,
        'resume_from_node': None,
        'resume_policy': 'not_resumable',
        'required_review_statuses': ['approved', 'rejected', 'needs_more_evidence'],
        'trusted_knowledge_requires_approval': True,
        'paid_llm_calls': False,
    }
