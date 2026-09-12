from __future__ import annotations

from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy
from backend.app.agent_runtime.rag_v2_contracts import (
    resolved_rag_mode,
    resolved_rag_stage,
)
from backend.app.assistant.service import (
    MAX_SUMMARY_LINES,
    _compact_context_text,
    assistant_message_evidence_is_live,
    eligible_context_messages,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import AgentRun, AssistantConversation, AssistantMessage

RAG_RENDER_CAPABILITY_HEADER = b'x-paraworks-rag-render-capability'
RAG_RENDER_CAPABILITY_VALUE = b'rag-v2-plain-text-citations:v1'
RAG_RENDER_CAPABILITY_VARY = 'X-ParaWorks-Rag-Render-Capability'
RAG_RENDER_CAPABILITY_RESPONSE_HEADERS = {
    'Cache-Control': 'private, no-store',
    'Vary': RAG_RENDER_CAPABILITY_VARY,
}


def require_rag_render_capability(
    raw_headers: Sequence[tuple[bytes, bytes]],
) -> None:
    declarations = [
        value
        for name, value in raw_headers
        if type(name) is bytes
        and type(value) is bytes
        and name.lower() == RAG_RENDER_CAPABILITY_HEADER
    ]
    if declarations != [RAG_RENDER_CAPABILITY_VALUE]:
        raise HTTPException(
            status_code=409,
            detail={'code': 'client_upgrade_required'},
        )


def assistant_post_requires_render_capability(settings: Settings) -> bool:
    decision = RagRolloutPolicy().decide(
        mode=resolved_rag_mode(settings),
        stage=resolved_rag_stage(settings),
        surface='assistant',
    )
    return decision.public_owner == 'v2'


def assistant_message_is_live_v2(
    db: Session,
    *,
    user: DemoUser,
    message: AssistantMessage,
) -> bool:
    if (
        message.role != 'assistant'
        or message.content_write_mode != 'rag_v2_exact'
        or type(message.linked_agent_run_id) is not int
        or message.linked_agent_run_id <= 0
        or message.agent_run_id != message.linked_agent_run_id
        or message.content_hmac_schema_version
        != 'assistant-message-content-hmac:v1'
        or not _is_lower_hmac(message.assistant_message_content_hmac)
        or not _is_lower_hmac(message.content_hmac_key_material_verifier)
        or not _is_lower_hmac(message.content_origin_hmac)
        or not _is_lower_hmac(message.rag_result_hmac)
    ):
        return False
    parent = db.get(AgentRun, message.linked_agent_run_id)
    if (
        parent is None
        or parent.run_contract_version != 'rag-run:v2'
        or parent.run_record_phase != 'final'
        or parent.status not in {'complete', 'failed'}
        or parent.completed_at is None
        or parent.agent_name != 'rag_orchestrator_agent'
        or parent.prompt_version != 'rag-answer:v2'
        or (parent.metadata_ or {}).get('rag_result_hmac')
        != message.rag_result_hmac
    ):
        return False
    if message.content_origin == 'rag_canned':
        return bool(
            message.evidence_contract_version == 'none-v1'
            and message.serving_dependency_count == 0
            and not message.citations
            and not message.source_ids
            and not message.source_links
            and not message.source_snippets
            and (message.metadata_ or {}).get('evidence_derived') is not True
            and message.dependency_set_hmac_schema_version is None
            and message.dependency_set_hmac is None
            and message.parent_selected_evidence_projection_hmac is None
            and message.model_influence_set_hmac is None
        )
    if message.content_origin != 'rag_assembled':
        return False
    return assistant_message_evidence_is_live(db, user=user, message=message)


def messages_projection_contains_live_v2(
    db: Session,
    *,
    user: DemoUser,
    messages: Sequence[AssistantMessage],
) -> bool:
    return any(
        assistant_message_is_live_v2(db, user=user, message=message)
        for message in messages
    )


def conversation_summary_contains_live_v2(
    db: Session,
    *,
    user: DemoUser,
    conversation: AssistantConversation,
) -> bool:
    seen: set[str] = set()
    contributors: list[AssistantMessage] = []
    for message in eligible_context_messages(
        db, user, list(conversation.messages)
    ):
        if message.role != 'assistant':
            continue
        compacted = _compact_context_text(message.content)
        if not compacted or compacted in seen:
            continue
        seen.add(compacted)
        contributors.append(message)
    return messages_projection_contains_live_v2(
        db,
        user=user,
        messages=contributors[-MAX_SUMMARY_LINES:],
    )


def _is_lower_hmac(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )
