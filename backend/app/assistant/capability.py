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
    assistant_message_projection_is_live,
    eligible_context_messages,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import AssistantConversation, AssistantMessage

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
    return bool(
        message.content_write_mode == 'rag_v2_exact'
        and assistant_message_projection_is_live(db, user=user, message=message)
    )


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
    for message in eligible_context_messages(db, user, list(conversation.messages)):
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
