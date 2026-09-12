from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.core.demo_auth import DemoUser
from backend.app.knowledge.serving_text import canonical_knowledge_text
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
    knowledge_model_for_type,
)
from backend.app.models import (
    AgentRun,
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
    AutoReviewRuntimeKeyState,
    Document,
    DocumentChunk,
    DocumentParserRun,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)

if TYPE_CHECKING:
    from backend.app.assistant.evidence_reader import AssistantMessageView

RECENT_CONTEXT_MESSAGE_LIMIT = 6
DEFAULT_CONVERSATION_TITLE = '새 대화'
MAX_CONVERSATION_TITLE_LENGTH = 32
MAX_CONTEXT_MESSAGE_CHARS = 500
MAX_SUMMARY_LINES = 4


def create_conversation(
    db: Session, user: DemoUser, *, title: str | None = None
) -> AssistantConversation:
    conversation = AssistantConversation(
        user_id=user.id,
        title=_conversation_title(title),
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def list_conversations(db: Session, user: DemoUser) -> list[AssistantConversation]:
    return list(
        db.scalars(
            select(AssistantConversation)
            .where(AssistantConversation.user_id == user.id)
            .order_by(
                AssistantConversation.updated_at.desc(), AssistantConversation.id.desc()
            )
        )
    )


def find_reusable_empty_conversation(
    db: Session, user: DemoUser
) -> AssistantConversation | None:
    return db.scalar(
        select(AssistantConversation)
        .where(
            AssistantConversation.user_id == user.id,
            AssistantConversation.title == DEFAULT_CONVERSATION_TITLE,
            ~AssistantConversation.messages.any(),
        )
        .order_by(
            AssistantConversation.updated_at.desc(), AssistantConversation.id.desc()
        )
    )


def get_owned_conversation(
    db: Session, user: DemoUser, conversation_id: int
) -> AssistantConversation:
    conversation = db.scalar(
        select(AssistantConversation)
        .options(selectinload(AssistantConversation.messages))
        .where(
            AssistantConversation.id == conversation_id,
            AssistantConversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise ValueError('assistant conversation not found')
    return conversation


def list_messages(
    db: Session, user: DemoUser, conversation_id: int
) -> list[AssistantMessage]:
    conversation = get_owned_conversation(db, user, conversation_id)
    return list(conversation.messages)


def get_owned_message(db: Session, user: DemoUser, message_id: int) -> AssistantMessage:
    message = db.scalar(
        select(AssistantMessage)
        .join(AssistantConversation)
        .where(
            AssistantMessage.id == message_id,
            AssistantConversation.user_id == user.id,
        )
    )
    if message is None:
        raise ValueError('assistant message not found')
    return message


def append_user_message(
    db: Session,
    user: DemoUser,
    conversation: AssistantConversation,
    content: str,
) -> AssistantMessage:
    _ensure_owned_conversation(user, conversation)
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError('assistant message content is required')

    message = AssistantMessage(
        conversation_id=conversation.id,
        role='user',
        content=normalized_content,
    )
    conversation.updated_at = datetime.now(UTC)
    if conversation.title == DEFAULT_CONVERSATION_TITLE:
        conversation.title = summarize_conversation_title(normalized_content)
    db.add(message)
    db.commit()
    db.refresh(message)
    db.refresh(conversation)
    return message


def update_message_metadata(
    db: Session,
    user: DemoUser,
    message: AssistantMessage,
    metadata: dict,
) -> AssistantMessage:
    conversation = get_owned_conversation(db, user, message.conversation_id)
    _ensure_owned_conversation(user, conversation)
    # MutableDict 변경 감지를 확실하게 만들기 위해 새 dict 인스턴스로 교체한다.
    message.metadata_ = dict(metadata)
    conversation.updated_at = datetime.now(UTC)
    db.add(message)
    db.commit()
    db.refresh(message)
    db.refresh(conversation)
    return message


def append_assistant_message(
    db: Session,
    user: DemoUser,
    conversation: AssistantConversation,
    *,
    content: str,
    citations: list,
    source_ids: list,
    source_links: list,
    source_snippets: list,
    permission_level: str | None,
    hidden_match_count: int,
    permission_notice: str | None,
    agent_run_id: int | None,
    metadata: dict,
    serving_dependencies: tuple[object, ...] = (),
    evidence_derived: bool = False,
) -> AssistantMessage:
    _ensure_owned_conversation(user, conversation)
    if set(metadata).intersection(
        {
            'content_write_mode',
            'rag_result_hmac',
            'assistant_message_content_hmac',
            'dependency_set_hmac',
        }
    ):
        raise ValueError('RAG V2 persistence fields are server-owned')
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError('assistant message content is required')
    if evidence_derived and not serving_dependencies:
        raise ValueError(
            'evidence-derived message requires complete serving dependencies'
        )

    stored_metadata = dict(metadata)
    if evidence_derived:
        stored_metadata['evidence_derived'] = True

    message = AssistantMessage(
        conversation_id=conversation.id,
        role='assistant',
        content=normalized_content,
        citations=citations,
        source_ids=source_ids,
        source_links=source_links,
        source_snippets=source_snippets,
        permission_level=permission_level,
        hidden_match_count=hidden_match_count,
        permission_notice=permission_notice,
        agent_run_id=agent_run_id,
        evidence_contract_version=(
            'assistant-evidence:v1' if serving_dependencies else 'none-v1'
        ),
        serving_dependency_count=len(serving_dependencies),
        metadata_=stored_metadata,
    )
    conversation.updated_at = datetime.now(UTC)
    conversation.summary = update_summary(conversation.summary, message.content)
    conversation.summary_updated_at = datetime.now(UTC)
    db.add(message)
    try:
        db.flush([message])
        _persist_serving_dependencies(
            db, message=message, dependencies=serving_dependencies
        )
        if serving_dependencies and not _message_evidence_is_live(
            db, user=user, message=message
        ):
            raise ValueError('assistant serving dependency changed before commit')
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(message)
    db.refresh(conversation)
    return message


def build_contextual_question(
    *,
    conversation: AssistantConversation,
    messages: list[AssistantMessage],
    new_message: str,
    db: Session | None = None,
    user: DemoUser | None = None,
) -> str:
    if db is not None and user is not None:
        messages = eligible_context_messages(db, user, messages)
    parts: list[str] = []
    seen_context: set[str] = set()
    if conversation.summary and (db is None or user is None):
        summary_lines = _dedupe_lines(conversation.summary.splitlines())
        if summary_lines:
            seen_context.update(f'assistant:{line}' for line in summary_lines)
            parts.append(f'대화 요약: {" ".join(summary_lines)}')

    # 전체 대화를 보내지 않고 최근 메시지만 사용해 토큰 사용량을 제한한다.
    context_messages = _exclude_current_user_message(messages, new_message)
    recent_messages = context_messages[-RECENT_CONTEXT_MESSAGE_LIMIT:]
    if recent_messages:
        parts.append('최근 대화:')
        for message in recent_messages:
            content = _compact_context_text(message.content)
            dedupe_key = f'{message.role}:{content}'
            if not content or dedupe_key in seen_context:
                continue
            seen_context.add(dedupe_key)
            parts.append(f'{message.role}: {content}')

    parts.append(f'현재 질문: {new_message.strip()}')
    return '\n'.join(parts)


def update_summary(existing_summary: str | None, latest_answer: str) -> str:
    lines = [*(existing_summary or '').splitlines(), latest_answer]
    return '\n'.join(_dedupe_lines(lines)[-MAX_SUMMARY_LINES:])[:1000]


def serialize_conversation(
    conversation: AssistantConversation,
    *,
    db: Session | None = None,
    user: DemoUser | None = None,
) -> dict:
    summary = None
    if db is not None and user is not None:
        live_messages = eligible_context_messages(db, user, list(conversation.messages))
        summary = (
            '\n'.join(
                _dedupe_lines(
                    [
                        message.content
                        for message in live_messages
                        if message.role == 'assistant'
                    ]
                )[-MAX_SUMMARY_LINES:]
            )
            or None
        )
    return {
        'id': conversation.id,
        'title': conversation.title,
        'summary': summary,
        'created_at': conversation.created_at.isoformat(),
        'updated_at': conversation.updated_at.isoformat(),
    }


def serialize_message(
    message: AssistantMessage,
    *,
    db: Session | None = None,
    user: DemoUser | None = None,
) -> dict:
    from backend.app.assistant.evidence_reader import AssistantEvidenceReader

    return (
        AssistantEvidenceReader()
        .project_message(db=db, actor=user, message=message)
        .to_response()
    )


def eligible_context_messages(
    db: Session,
    user: DemoUser,
    messages: list[AssistantMessage],
) -> list['AssistantMessageView']:
    from backend.app.assistant.evidence_reader import AssistantEvidenceReader

    reader = AssistantEvidenceReader()
    return [
        view
        for message in messages
        for view in (reader.project_message(db=db, actor=user, message=message),)
        if view.evidence_available
    ]


def _persist_serving_dependencies(
    db: Session,
    *,
    message: AssistantMessage,
    dependencies: tuple[object, ...],
) -> None:
    if not dependencies:
        return
    runtime = db.scalar(
        select(AutoReviewRuntimeKeyState).where(
            AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
        )
    )
    if runtime is None or not runtime.ready:
        raise ValueError('assistant serving dependency key runtime unavailable')
    for ordinal, snapshot in enumerate(dependencies):
        values = (
            vars(snapshot)
            if hasattr(snapshot, '__dict__')
            else {
                name: getattr(snapshot, name)
                for name in getattr(snapshot, '__slots__', ())
            }
        )
        evidence_link_ids = tuple(values.pop('evidence_link_ids', ()))
        identity = '|'.join(f'{key}={values[key]!r}' for key in sorted(values))
        dependency = AssistantMessageEvidenceDependency(
            assistant_message_id=message.id,
            candidate_ordinal=ordinal,
            dependency_set_hmac=sha256(identity.encode('utf-8')).hexdigest(),
            fingerprint_key_version=runtime.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                runtime.fingerprint_key_material_verifier
            ),
            **values,
        )
        db.add(dependency)
        db.flush([dependency])
        if dependency.approval_link_id is not None:
            refs = [
                AssistantMessageKnowledgeEvidenceRef(
                    dependency_id=dependency.id,
                    assistant_message_id=message.id,
                    approval_link_id=dependency.approval_link_id,
                    trusted_knowledge_evidence_link_id=evidence_link_id,
                )
                for evidence_link_id in evidence_link_ids
            ]
            db.add_all(refs)
            db.flush(refs)


def _message_evidence_is_live(
    db: Session, *, user: DemoUser, message: AssistantMessage
) -> bool:
    evidence_shaped = bool(
        message.citations
        or message.source_ids
        or message.source_links
        or message.source_snippets
        or (message.metadata_ or {}).get('evidence_derived') is True
    )
    dependencies = tuple(
        db.scalars(
            select(AssistantMessageEvidenceDependency)
            .where(
                AssistantMessageEvidenceDependency.assistant_message_id == message.id
            )
            .order_by(AssistantMessageEvidenceDependency.candidate_ordinal)
        ).all()
    )
    if not evidence_shaped and not dependencies:
        return message.evidence_contract_version in {None, 'none-v1'}
    if (
        message.evidence_contract_version != 'assistant-evidence:v1'
        or message.serving_dependency_count != len(dependencies)
        or not dependencies
    ):
        return False
    eligibility = TrustedServingEligibilityService(db)
    return all(
        dependency.dependency_serving_scope is None
        and dependency.dependency_role is None
        and dependency.permission_level in user.permission_levels
        and _dependency_is_live(db, eligibility, message, dependency)
        for dependency in dependencies
    )


def assistant_message_evidence_is_live(
    db: Session, *, user: DemoUser, message: AssistantMessage
) -> bool:
    """Public fail-closed gate for operations that reuse stored answer bytes."""
    return assistant_message_projection_is_live(db, user=user, message=message)


def assistant_message_projection_is_live(
    db: Session, *, user: DemoUser, message: AssistantMessage
) -> bool:
    """Return whether the current serializer may expose stored message bytes."""
    from backend.app.assistant.evidence_reader import AssistantEvidenceReader

    view = AssistantEvidenceReader().project_message(db=db, actor=user, message=message)
    return view.evidence_available


def _assistant_message_has_valid_v2_structure(
    db: Session, *, message: AssistantMessage
) -> bool:
    if (
        message.role != 'assistant'
        or type(message.linked_agent_run_id) is not int
        or message.linked_agent_run_id <= 0
        or message.agent_run_id != message.linked_agent_run_id
        or message.content_hmac_schema_version != 'assistant-message-content-hmac:v1'
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
        or (parent.metadata_ or {}).get('rag_result_hmac') != message.rag_result_hmac
    ):
        return False
    if message.content_origin == 'rag_canned':
        return bool(
            message.evidence_contract_version == 'none-v1'
            and message.serving_dependency_count == 0
            and message.permission_level is None
            and not message.citations
            and not message.source_ids
            and not message.source_links
            and not message.source_snippets
            and _canned_hidden_projection_is_valid(message)
            and (message.metadata_ or {}).get('evidence_derived') is not True
            and message.dependency_set_hmac_schema_version is None
            and message.dependency_set_hmac is None
            and message.parent_selected_evidence_projection_hmac is None
            and message.model_influence_set_hmac is None
        )
    return message.content_origin == 'rag_assembled'


def _canned_hidden_projection_is_valid(message: AssistantMessage) -> bool:
    hidden_count = message.hidden_match_count
    if type(hidden_count) is not int or not 0 <= hidden_count <= 20:
        return False
    if hidden_count > 0:
        return message.permission_notice == 'Some sources may be hidden by permissions.'
    return message.permission_notice in {None, 'evidence_unavailable'}


def _is_lower_hmac(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _dependency_is_live(
    db: Session,
    eligibility: TrustedServingEligibilityService,
    message: AssistantMessage,
    dependency: AssistantMessageEvidenceDependency,
) -> bool:
    serving = eligibility.for_document(dependency.serving_document_id)
    if (
        not serving.eligible
        or serving.effective_permission != dependency.permission_level
    ):
        return False
    if dependency.dependency_kind == 'raw_chunk':
        chunk = db.get(DocumentChunk, dependency.document_chunk_id)
        source = db.get(Source, dependency.source_id)
        parser_run = db.get(DocumentParserRun, dependency.parser_run_id)
        document = (
            db.get(Document, parser_run.document_id) if parser_run is not None else None
        )
        identity_is_current = bool(
            chunk is not None
            and source is not None
            and parser_run is not None
            and document is not None
            and chunk.version_id == dependency.document_version_id
            and chunk.parser_run_id == dependency.parser_run_id
            and document.current_document_version_id
            == dependency.current_document_version_id
            and source.server_content_signature == dependency.server_content_signature
            and parser_run.parser_policy_version == dependency.parser_policy_version
            and parser_run.parser_version == dependency.parser_version
            and parser_run.chunk_policy_version == dependency.chunk_policy_version
        )
        return bool(
            identity_is_current
            and _raw_dependency_content_hash(
                chunk=chunk,
                source=source,
                permission_level=dependency.permission_level,
            )
            == dependency.serving_content_hash
        )
    if dependency.legacy_human_base:
        if dependency.knowledge_type is None or dependency.knowledge_id is None:
            return False
        target = db.get(
            knowledge_model_for_type(dependency.knowledge_type),
            dependency.knowledge_id,
        )
        item = (
            db.get(ReviewItem, dependency.legacy_source_review_item_id)
            if dependency.legacy_source_review_item_id is not None
            else None
        )
        legacy_base_is_live = bool(
            target is not None
            and (
                (
                    dependency.legacy_source_review_item_id is None
                    and target.source_review_item_id is None
                )
                or (
                    item is not None
                    and target.source_review_item_id == item.id
                    and item.status == 'approved'
                    and item.resolution_source in {None, 'human'}
                    and item.candidate_contract_version != 'c5-v1'
                )
            )
        )
        return bool(
            legacy_base_is_live
            and _knowledge_dependency_content_hash(db, dependency)
            == dependency.serving_content_hash
        )
    link = db.get(TrustedKnowledgeApprovalLink, dependency.approval_link_id)
    if (
        link is None
        or not link.active
        or link.knowledge_type != dependency.knowledge_type
        or link.knowledge_id != dependency.knowledge_id
        or not eligibility.approval_link_is_live(link.id)
    ):
        return False
    current_children = set(
        db.scalars(
            select(TrustedKnowledgeEvidenceLink.id).where(
                TrustedKnowledgeEvidenceLink.approval_link_id == link.id
            )
        ).all()
    )
    snapshot_children = set(
        db.scalars(
            select(
                AssistantMessageKnowledgeEvidenceRef.trusted_knowledge_evidence_link_id
            ).where(
                AssistantMessageKnowledgeEvidenceRef.dependency_id == dependency.id,
                AssistantMessageKnowledgeEvidenceRef.assistant_message_id == message.id,
            )
        ).all()
    )
    return bool(
        current_children
        and current_children == snapshot_children
        and _knowledge_dependency_content_hash(db, dependency)
        == dependency.serving_content_hash
    )


def _raw_dependency_content_hash(
    *,
    chunk: DocumentChunk,
    source: Source,
    permission_level: str,
) -> str:
    return _serving_content_hash(
        source_id=source.source_id,
        text=chunk.text,
        source_url=source.source_url,
        source_snippet=chunk.source_snippet,
        permission_level=permission_level,
    )


def _knowledge_dependency_content_hash(
    db: Session,
    dependency: AssistantMessageEvidenceDependency,
) -> str | None:
    if dependency.knowledge_type is None or dependency.knowledge_id is None:
        return None
    try:
        target = db.get(
            knowledge_model_for_type(dependency.knowledge_type),
            dependency.knowledge_id,
        )
    except (LookupError, ValueError):
        return None
    if target is None:
        return None
    try:
        text = canonical_knowledge_text(dependency.knowledge_type, target)
    except (AttributeError, ValueError):
        return None
    source_links = list(target.source_links or [])
    source_snippets = list(target.source_snippets or [])
    return _serving_content_hash(
        source_id=dependency.serving_document_id,
        text=text,
        source_url=(
            source_links[0]
            if source_links
            else f'knowledge://{dependency.serving_document_id}'
        ),
        source_snippet=(source_snippets[0] if source_snippets else text[:240]),
        permission_level=dependency.permission_level,
    )


def _serving_content_hash(
    *,
    source_id: str,
    text: str,
    source_url: str,
    source_snippet: str,
    permission_level: str,
) -> str:
    value = '\n'.join((source_id, text, source_url, source_snippet, permission_level))
    return sha256(value.encode('utf-8')).hexdigest()


def _conversation_title(value: str | None) -> str:
    normalized = (
        value or DEFAULT_CONVERSATION_TITLE
    ).strip() or DEFAULT_CONVERSATION_TITLE
    return normalized[:80]


def summarize_conversation_title(value: str) -> str:
    normalized = ' '.join(value.strip().split())
    if not normalized:
        return DEFAULT_CONVERSATION_TITLE
    if len(normalized) <= MAX_CONVERSATION_TITLE_LENGTH:
        return normalized
    return f'{normalized[: MAX_CONVERSATION_TITLE_LENGTH - 1].rstrip()}…'


def _ensure_owned_conversation(
    user: DemoUser, conversation: AssistantConversation
) -> None:
    if conversation.user_id != user.id:
        raise ValueError('assistant conversation not found')


def _exclude_current_user_message(
    messages: list[AssistantMessage],
    new_message: str,
) -> list[AssistantMessage]:
    if not messages:
        return messages
    latest_message = messages[-1]
    if (
        latest_message.role == 'user'
        and latest_message.content.strip() == new_message.strip()
    ):
        return messages[:-1]
    return messages


def _compact_context_text(value: str) -> str:
    compacted = ' '.join(value.strip().split())
    if len(compacted) <= MAX_CONTEXT_MESSAGE_CHARS:
        return compacted
    return f'{compacted[: MAX_CONTEXT_MESSAGE_CHARS - 1].rstrip()}…'


def _dedupe_lines(lines: list[str]) -> list[str]:
    unique_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        compacted = _compact_context_text(line)
        if not compacted or compacted in seen:
            continue
        seen.add(compacted)
        unique_lines.append(compacted)
    return unique_lines
