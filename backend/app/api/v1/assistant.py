import json
import unicodedata
from contextlib import suppress
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.dependencies.utils import get_dependant, solve_dependencies
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from sqlalchemy.orm import Session
from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse, Response

from backend.app.agent_runtime.rag_application import (
    AssistantIngressError,
    RagApplicationFacade,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    RagInputSafetyError,
    RagInputScannerUnavailableError,
    _scan_value,
)
from backend.app.assistant.capability import (
    RAG_RENDER_CAPABILITY_RESPONSE_HEADERS,
    assistant_post_requires_render_capability,
    conversation_summary_contains_live_v2,
    messages_projection_contains_live_v2,
    require_rag_render_capability,
)
from backend.app.assistant.contact_lookup import (
    contact_lookup_response_content,
    detect_contact_lookup_request,
)
from backend.app.assistant.delivery import AssistantDeliveryResult
from backend.app.assistant.email_actions import (
    EMAIL_ACTION_PROMPT_VERSION,
    assistant_email_draft_content,
    email_draft_metadata,
)
from backend.app.assistant.email_agent import (
    EmailActionDecision,
    EmailIntentDecision,
    build_email_draft_composer,
    build_email_intent_gate,
    render_email_action_context,
    render_recent_assistant_context_for_email,
)
from backend.app.assistant.email_draft_context import (
    EmailSourceContext,
    build_email_source_context,
    build_generated_email_source_request,
    ensure_draft_contains_source,
    fallback_draft_from_source,
    is_pending_draft_recipient_problem,
    merge_resolved_recipients,
    render_email_source_context,
)
from backend.app.assistant.gmail_sender import GmailDraftSender, GmailSendError
from backend.app.assistant.recipient_resolver import resolve_email_recipients
from backend.app.assistant.service import (
    append_assistant_message,
    append_user_message,
    assistant_message_evidence_is_live,
    build_contextual_question,
    create_conversation,
    eligible_context_messages,
    find_reusable_empty_conversation,
    get_owned_conversation,
    get_owned_message,
    list_conversations,
    list_messages,
    serialize_conversation,
    serialize_message,
    update_message_metadata,
)
from backend.app.assistant.tool_logging import AssistantToolLogger
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.db.session import get_db
from backend.app.rag.search_store import build_pgvector_search_store
from backend.app.schemas.assistant import (
    AssistantConversationCreatedResponse,
    AssistantConversationCreateRequest,
    AssistantConversationsResponse,
    AssistantEmailSendResponse,
    AssistantMessageCreateRequest,
    AssistantMessagesResponse,
    AssistantTurnResponse,
)

_CAPABILITY_DEPENDENT_STATE = 'assistant_rag_render_capability_dependent'


def answer_question_with_rag(**kwargs):
    return RagApplicationFacade.invoke_legacy_assistant_context(**kwargs)


def _map_assistant_delivery(result: AssistantDeliveryResult):
    """Map only the algebra's public status/body; never append or infer outcomes."""
    if type(result) is not AssistantDeliveryResult:
        raise HTTPException(status_code=500, detail='Internal Server Error')
    if result.public_status == 200:
        return result.assistant_message_id
    detail = {
        'budget_error': {'code': 'budget_exceeded'},
        'generation_error': 'assistant answer generation failed',
        'permission_error': 'Permission denied.',
        'owner_not_found': 'assistant conversation not found',
        'validation_error': 'assistant message content is invalid',
        'persistence_error': 'Internal Server Error',
        'reconciliation_required': 'Internal Server Error',
    }[result.body_kind]
    raise HTTPException(status_code=result.public_status, detail=detail)


class _AssistantCapabilityHeadersRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()
        is_capability_post = self.endpoint.__name__ in {
            'create_assistant_conversation',
            'create_assistant_message',
        }
        pre_body_dependant = (
            get_dependant(
                path=self.path_format,
                call=_prepare_assistant_post_context,
            )
            if is_capability_post
            else None
        )

        async def capability_headers_handler(request: Request):
            if pre_body_dependant is not None:
                solved = await solve_dependencies(
                    request=request,
                    dependant=pre_body_dependant,
                    body=None,
                    dependency_overrides_provider=self.dependency_overrides_provider,
                    async_exit_stack=request.scope['fastapi_inner_astack'],
                    embed_body_fields=False,
                )
                if solved.errors:
                    raise RequestValidationError(solved.errors)
                _prepare_assistant_post_context(**solved.values)
            return await original_handler(request)

        return capability_headers_handler

    async def handle(self, scope, receive, send) -> None:
        response_started = False

        async def send_with_capability_headers(message) -> None:
            nonlocal response_started
            if message['type'] == 'http.response.start':
                if _scope_is_capability_dependent(scope):
                    headers = MutableHeaders(scope=message)
                    for name, value in RAG_RENDER_CAPABILITY_RESPONSE_HEADERS.items():
                        headers[name] = value
                response_started = True
            await send(message)

        try:
            await super().handle(scope, receive, send_with_capability_headers)
        except Exception as exc:
            validation_error = _surrogate_validation_error(exc)
            if validation_error is not None and not response_started:
                response = _ascii_safe_json_response(
                    status_code=422,
                    content={'detail': validation_error.errors()},
                    capability_dependent=_scope_is_capability_dependent(scope),
                )
                await response(scope, receive, send_with_capability_headers)
                return
            if not _scope_is_capability_dependent(scope) or response_started:
                raise
            response = PlainTextResponse(
                'Internal Server Error',
                status_code=500,
                headers=RAG_RENDER_CAPABILITY_RESPONSE_HEADERS,
            )
            await response(scope, receive, send_with_capability_headers)
            raise


router = APIRouter(
    prefix='/assistant',
    tags=['assistant'],
    route_class=_AssistantCapabilityHeadersRoute,
)
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]
AppSettings = Annotated[Settings, Depends(get_settings)]
ASSISTANT_FAILURE_CONTENT = (
    '답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.'
)


def _ascii_safe_json_response(
    *,
    status_code: int,
    content: object,
    capability_dependent: bool,
) -> Response:
    response_headers: dict[str, str] = {}
    if capability_dependent:
        response_headers.update(RAG_RENDER_CAPABILITY_RESPONSE_HEADERS)
    return Response(
        status_code=status_code,
        media_type='application/json',
        headers=response_headers,
        content=json.dumps(
            jsonable_encoder(content),
            ensure_ascii=True,
            allow_nan=False,
            separators=(',', ':'),
        ),
    )


def _mark_capability_dependent(request: Request) -> None:
    setattr(request.state, _CAPABILITY_DEPENDENT_STATE, True)


def _is_capability_dependent(request: Request) -> bool:
    return getattr(request.state, _CAPABILITY_DEPENDENT_STATE, False) is True


def _scope_is_capability_dependent(scope: dict) -> bool:
    state = scope.get('state')
    return bool(
        isinstance(state, dict)
        and state.get(_CAPABILITY_DEPENDENT_STATE) is True
    )


def _surrogate_validation_error(
    exc: Exception,
) -> RequestValidationError | None:
    if not isinstance(exc, UnicodeEncodeError):
        return None
    pending: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, RequestValidationError):
            return current
        pending.extend((current.__cause__, current.__context__))
    return None


def _mark_capability_dependent_post(
    request: Request,
    settings: AppSettings,
) -> None:
    request.state.assistant_post_settings = settings
    if assistant_post_requires_render_capability(settings):
        _mark_capability_dependent(request)


CapabilityDependentPost = Annotated[
    None,
    Depends(_mark_capability_dependent_post),
]


def _prepare_assistant_post_context(
    request: Request,
    _capability_context: CapabilityDependentPost,
    db: DbSession,
    user: CurrentUser,
) -> None:
    request.state.assistant_post_db = db
    request.state.assistant_post_user = user


def _prepared_assistant_post_db(request: Request) -> Session:
    return request.state.assistant_post_db


def _prepared_assistant_post_user(request: Request) -> DemoUser:
    return request.state.assistant_post_user


def _prepared_assistant_post_settings(request: Request) -> Settings:
    return request.state.assistant_post_settings


AssistantPostDbSession = Annotated[Session, Depends(_prepared_assistant_post_db)]
AssistantPostCurrentUser = Annotated[DemoUser, Depends(_prepared_assistant_post_user)]
AssistantPostSettings = Annotated[Settings, Depends(_prepared_assistant_post_settings)]


def _require_post_capability(request: Request, settings: Settings) -> None:
    if assistant_post_requires_render_capability(settings):
        require_rag_render_capability(request.scope.get('headers', ()))


def _require_assistant_ingress_scan(value: str) -> None:
    normalized = ' '.join(unicodedata.normalize('NFC', value).split())
    try:
        _scan_value(value)
        _scan_value(normalized)
    except RagInputSafetyError as exc:
        raise HTTPException(
            status_code=422,
            detail={'code': 'input_safety_blocked'},
        ) from exc
    except RagInputScannerUnavailableError as exc:
        raise HTTPException(
            status_code=500,
            detail='assistant request failed',
        ) from exc


def require_conversation(db: Session, user: DemoUser, conversation_id: int):
    try:
        return get_owned_conversation(db, user, conversation_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail='assistant conversation not found'
        ) from exc


def append_failed_assistant_message(
    db: Session,
    user: DemoUser,
    conversation,
    *,
    reason: str,
    failure_class: str,
    agent_run_id: int | None = None,
):
    return append_assistant_message(
        db,
        user,
        conversation,
        content=ASSISTANT_FAILURE_CONTENT,
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=agent_run_id,
        metadata={
            'status': 'failed',
            'failure_reason': reason,
            'failure_class': failure_class,
        },
    )


def _answer_question_or_raise(
    *,
    db: Session,
    user: DemoUser,
    conversation,
    question: str,
    settings: Settings,
    vector_store,
    tool_logger: AssistantToolLogger,
    before_failure_write=None,
):
    try:
        return answer_question_with_rag(
            db=db,
            user=user,
            question=question,
            settings=settings,
            vector_store=vector_store,
            tool_logger=tool_logger,
            commit_agent_run=False,
        )
    except Exception as exc:
        if before_failure_write is not None:
            before_failure_write()
        append_failed_assistant_message(
            db,
            user,
            conversation,
            reason='rag_exception',
            failure_class=exc.__class__.__name__,
        )
        raise HTTPException(
            status_code=502,
            detail='assistant answer generation failed',
        ) from exc


def _render_rag_answer_for_email(answer) -> str:
    source_lines = [f'- {link}' for link in answer.source_links[:5]]
    snippet_lines = [f'- {snippet}' for snippet in answer.source_snippets[:5]]
    return '\n'.join(
        [
            'RAG answer:',
            answer.answer,
            '',
            'Source links:',
            *source_lines,
            '',
            'Source snippets:',
            *snippet_lines,
        ]
    )


def _recipient_clarification_content(
    recipient_resolution, *, correction: bool = False
) -> str:
    if correction:
        return '수신자를 누구로 수정할까요? 이름이나 이메일 주소를 알려주세요.'
    if recipient_resolution.status == 'ambiguous' and recipient_resolution.candidates:
        lines = [
            f'- {candidate.display_name}: {candidate.email}'
            for candidate in recipient_resolution.candidates[:5]
        ]
        return (
            '수신자를 하나로 확정하기 어렵습니다. 누구에게 보낼지 선택해 주세요.\n'
            + '\n'.join(lines)
        )
    return '수신자를 확정할 수 없습니다. 받을 사람의 정확한 이름이나 이메일 주소를 알려주세요.'


def _append_recipient_clarification_message(
    *,
    db: Session,
    user: DemoUser,
    conversation,
    recipient_resolution=None,
    correction: bool = False,
    source_context=None,
):
    return append_assistant_message(
        db,
        user,
        conversation,
        content=_recipient_clarification_content(
            recipient_resolution, correction=correction
        ),
        citations=[],
        source_ids=[],
        source_links=[],
        source_snippets=[],
        permission_level=None,
        hidden_match_count=0,
        permission_notice=None,
        agent_run_id=None,
        metadata={
            'action_type': 'email_clarification',
            'status': 'needs_input',
            'prompt_version': EMAIL_ACTION_PROMPT_VERSION,
            'agent_name': 'recipient_resolver',
            'reason': 'recipient_correction_requested'
            if correction
            else 'recipient_not_resolved',
            'recipient_status': getattr(recipient_resolution, 'status', 'not_found'),
            'source_context': source_context.metadata if source_context else None,
        },
    )


def _append_source_email_draft_message(
    *,
    db: Session,
    user: DemoUser,
    conversation,
    user_message,
    settings: Settings,
    email_context: str,
    source_context,
    resolved_recipients: list[dict[str, object]],
    before_write=None,
    before_provider=None,
):
    from backend.app.assistant.email_agent import NoopEmailDraftComposer

    source_rag_context = render_email_source_context(
        source_context,
        max_chars=settings.assistant_email_agent_max_input_chars,
    )
    source_intent = EmailIntentDecision(
        email_intent=True,
        intent_type='send',
        confidence_score=1.0,
        requires_rag_result=False,
        reason=source_context.reason,
        model_name='deterministic',
    )
    composer = build_email_draft_composer(settings, before_provider=before_provider)
    # An injected implementation is not proof of provider-free behavior. Only
    # the exact built-in Noop can bypass preflight for deterministic fallback.
    if type(composer) is not NoopEmailDraftComposer and before_provider is not None:
        before_provider()
    email_decision: EmailActionDecision = composer.compose(
        conversation_context=email_context,
        latest_message=user_message.content,
        intent=source_intent,
        rag_context=source_rag_context,
        resolved_recipients=resolved_recipients,
    )
    email_draft = email_decision.to_draft()
    if email_draft is not None:
        email_draft = ensure_draft_contains_source(email_draft, source_context)
    else:
        email_draft = fallback_draft_from_source(
            source_context=source_context,
            resolved_recipients=resolved_recipients,
        )

    if email_draft is not None:
        if before_write is not None:
            before_write()
        return append_assistant_message(
            db,
            user,
            conversation,
            content=assistant_email_draft_content(email_draft),
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level=None,
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={
                **email_draft_metadata(email_draft),
                'agent_name': 'email_draft_composer',
                'model_name': email_decision.model_name,
                'confidence_score': email_decision.confidence_score,
                'requires_rag_result': False,
                'source_context': source_context.metadata,
            },
        )

    if (
        email_decision.action_type == 'needs_clarification'
        and email_decision.clarification_question
    ):
        if before_write is not None:
            before_write()
        return append_assistant_message(
            db,
            user,
            conversation,
            content=email_decision.clarification_question,
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level=None,
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={
                'action_type': 'email_clarification',
                'status': 'needs_input',
                'prompt_version': EMAIL_ACTION_PROMPT_VERSION,
                'agent_name': 'email_draft_composer',
                'model_name': email_decision.model_name,
                'confidence_score': email_decision.confidence_score,
                'requires_rag_result': False,
                'source_context': source_context.metadata,
            },
        )

    return None


@router.get('/conversations', response_model=AssistantConversationsResponse)
def list_assistant_conversations(
    raw_request: Request,
    db: DbSession,
    user: CurrentUser,
) -> dict:
    conversations = list_conversations(db, user)
    if any(
        conversation_summary_contains_live_v2(db, user=user, conversation=conversation)
        for conversation in conversations
    ):
        _mark_capability_dependent(raw_request)
        require_rag_render_capability(raw_request.scope.get('headers', ()))
    return {
        'conversations': [
            serialize_conversation(conversation, db=db, user=user)
            for conversation in conversations
        ]
    }


@router.post('/conversations', response_model=AssistantConversationCreatedResponse)
def create_assistant_conversation(
    request: AssistantConversationCreateRequest,
    raw_request: Request,
    db: AssistantPostDbSession,
    user: AssistantPostCurrentUser,
    settings: AssistantPostSettings,
) -> dict:
    _require_post_capability(raw_request, settings)
    _require_assistant_ingress_scan(request.title or '새 대화')
    facade = raw_request.app.state.rag_application_facade
    if facade.execution_owner('assistant') == 'v2':
        try:
            facade.prepare_assistant_ingress(actor=user, conversation_id=None,
                caller_text=request.title or '새 대화')
        except AssistantIngressError as exc:
            _map_assistant_delivery(exc.result)
    if request.title is None or request.title.strip() == '새 대화':
        reusable_conversation = find_reusable_empty_conversation(db, user)
        if reusable_conversation is not None:
            return {
                'conversation': serialize_conversation(
                    reusable_conversation, db=db, user=user
                )
            }

    conversation = create_conversation(db, user, title=request.title)
    return {'conversation': serialize_conversation(conversation, db=db, user=user)}


@router.get(
    '/conversations/{conversation_id}/messages',
    response_model=AssistantMessagesResponse,
)
def list_assistant_messages(
    conversation_id: int,
    raw_request: Request,
    db: DbSession,
    user: CurrentUser,
) -> dict:
    conversation = require_conversation(db, user, conversation_id)
    messages = list_messages(db, user, conversation.id)
    if messages_projection_contains_live_v2(db, user=user, messages=messages):
        _mark_capability_dependent(raw_request)
        require_rag_render_capability(raw_request.scope.get('headers', ()))
    return {
        'conversation': serialize_conversation(conversation, db=db, user=user),
        'messages': [
            serialize_message(message, db=db, user=user) for message in messages
        ],
    }


@router.post(
    '/conversations/{conversation_id}/messages',
    response_model=AssistantTurnResponse,
)
def create_assistant_message(
    conversation_id: int,
    request: AssistantMessageCreateRequest,
    raw_request: Request,
    db: AssistantPostDbSession,
    user: AssistantPostCurrentUser,
    settings: AssistantPostSettings,
) -> dict:
    if not request.content.strip():
        raise HTTPException(status_code=422, detail='assistant message content is required')
    conversation = require_conversation(db, user, conversation_id)
    _require_post_capability(raw_request, settings)
    _require_assistant_ingress_scan(request.content)
    facade = raw_request.app.state.rag_application_facade
    prepared_ingress = None
    assistant_rag_owner = facade.execution_owner('assistant')
    v2_rag_owner = assistant_rag_owner == 'v2'
    # Non-RAG planning sees only eligible immutable prior snapshots and current
    # input. Defer the V2 user INSERT until an action actually writes a product,
    # or until provider/RAG ingress has passed. A tentative email decision cannot
    # downgrade a later RAG fallback or leave an invalid-prior user row behind.
    user_message = SimpleNamespace(id=None, content=request.content.strip())

    def ensure_v2_preflight():
        nonlocal prepared_ingress
        if v2_rag_owner and prepared_ingress is None:
            try:
                prepared_ingress = facade.prepare_assistant_ingress(
                    actor=user, conversation_id=conversation.id, caller_text=request.content,
                )
            except AssistantIngressError as exc:
                _map_assistant_delivery(exc.result)

    def persist_user_message():
        nonlocal user_message
        if user_message.id is None:
            try:
                user_message = append_user_message(db, user, conversation, request.content)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return user_message

    if not v2_rag_owner:
        persist_user_message()

    messages = list_messages(db, user, conversation.id)
    messages = eligible_context_messages(db, user, messages)
    prior_messages = messages if v2_rag_owner else messages[:-1]
    assistant_context_shadow = bool(
        assistant_rag_owner == 'shadow'
        and any(message.role == 'assistant' for message in prior_messages)
    )
    tool_logger = AssistantToolLogger()
    email_context = render_email_action_context(
        messages=prior_messages,
        max_chars=settings.assistant_email_agent_max_input_chars,
    )
    recent_assistant_context = render_recent_assistant_context_for_email(
        messages=prior_messages,
        max_chars=settings.assistant_email_agent_max_input_chars,
    )
    if is_pending_draft_recipient_problem(
        messages=prior_messages,
        latest_message=user_message.content,
    ):
        persist_user_message()
        assistant_message = _append_recipient_clarification_message(
            db=db,
            user=user,
            conversation=conversation,
            correction=True,
        )
        return {
            'conversation': serialize_conversation(conversation, db=db, user=user),
            'user_message': serialize_message(user_message, db=db, user=user),
            'assistant_message': serialize_message(assistant_message, db=db, user=user),
        }

    contact_lookup = detect_contact_lookup_request(
        latest_message=user_message.content,
        conversation_context=email_context,
    )
    if contact_lookup.is_lookup:
        recipient_resolution = resolve_email_recipients(
            db=db,
            latest_message=contact_lookup.lookup_query,
            conversation_context=email_context,
        )
        tool_logger.log(
            'contact_lookup',
            (
                f'result status={recipient_resolution.status} '
                f'candidate_count={len(recipient_resolution.candidates)} '
                f'reason={recipient_resolution.reason} '
                f'lookup_reason={contact_lookup.reason}'
            ),
        )
        persist_user_message()
        assistant_message = append_assistant_message(
            db,
            user,
            conversation,
            content=contact_lookup_response_content(recipient_resolution),
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level=None,
            hidden_match_count=0,
            permission_notice=None,
            agent_run_id=None,
            metadata={
                'action_type': 'contact_lookup',
                'status': recipient_resolution.status,
                'agent_name': 'recipient_resolver',
                'lookup_reason': contact_lookup.reason,
                'lookup_query': contact_lookup.lookup_query,
                'resolved_recipients': recipient_resolution.resolved_recipients,
                'candidate_count': len(recipient_resolution.candidates),
                'reason': recipient_resolution.reason,
            },
        )
        return {
            'conversation': serialize_conversation(conversation, db=db, user=user),
            'user_message': serialize_message(user_message, db=db, user=user),
            'assistant_message': serialize_message(assistant_message, db=db, user=user),
        }

    generated_source_request = build_generated_email_source_request(
        latest_message=user_message.content,
    )
    if generated_source_request.should_route:
        recipient_resolution = resolve_email_recipients(
            db=db,
            latest_message=user_message.content,
            conversation_context=email_context,
        )
        resolved_recipients = recipient_resolution.resolved_recipients
        if recipient_resolution.status != 'resolved':
            persist_user_message()
            assistant_message = _append_recipient_clarification_message(
                db=db,
                user=user,
                conversation=conversation,
                recipient_resolution=recipient_resolution,
            )
            return {
                'conversation': serialize_conversation(conversation, db=db, user=user),
                'user_message': serialize_message(user_message, db=db, user=user),
                'assistant_message': serialize_message(
                    assistant_message, db=db, user=user
                ),
            }

        tool_logger.log(
            'email_generated_source',
            (
                f'start reason={generated_source_request.reason} '
                f'recipient_count={len(resolved_recipients)}'
            ),
        )
        ensure_v2_preflight()
        answer = _answer_question_or_raise(
            db=db,
            user=user,
            conversation=conversation,
            question=generated_source_request.question,
            settings=settings,
            vector_store=build_pgvector_search_store(db=db, settings=settings),
            tool_logger=tool_logger,
            before_failure_write=persist_user_message,
        )
        source_context = EmailSourceContext(
            should_route=True,
            kind='generated_rag_answer',
            content=answer.answer,
            reason=generated_source_request.reason,
        )
        assistant_message = _append_source_email_draft_message(
            db=db,
            user=user,
            conversation=conversation,
            user_message=user_message,
            settings=settings,
            email_context=email_context,
            source_context=source_context,
            resolved_recipients=resolved_recipients,
            before_write=persist_user_message,
            before_provider=ensure_v2_preflight,
        )
        if assistant_message is not None:
            return {
                'conversation': serialize_conversation(conversation, db=db, user=user),
                'user_message': serialize_message(user_message, db=db, user=user),
                'assistant_message': serialize_message(
                    assistant_message, db=db, user=user
                ),
            }

    source_context = build_email_source_context(
        messages=prior_messages,
        latest_message=user_message.content,
    )
    if source_context.should_route:
        recipient_resolution = resolve_email_recipients(
            db=db,
            latest_message=user_message.content,
            conversation_context=email_context,
        )
        resolved_recipients = merge_resolved_recipients(
            recipient_resolution.resolved_recipients,
            source_context,
        )
        if not resolved_recipients:
            persist_user_message()
            assistant_message = _append_recipient_clarification_message(
                db=db,
                user=user,
                conversation=conversation,
                recipient_resolution=recipient_resolution,
                source_context=source_context,
            )
            return {
                'conversation': serialize_conversation(conversation, db=db, user=user),
                'user_message': serialize_message(user_message, db=db, user=user),
                'assistant_message': serialize_message(
                    assistant_message, db=db, user=user
                ),
            }

        tool_logger.log(
            'email_source_context',
            (
                f'result kind={source_context.kind} '
                f'reason={source_context.reason} '
                f'recipient_count={len(resolved_recipients)}'
            ),
        )
        assistant_message = _append_source_email_draft_message(
            db=db,
            user=user,
            conversation=conversation,
            user_message=user_message,
            settings=settings,
            email_context=email_context,
            source_context=source_context,
            resolved_recipients=resolved_recipients,
            before_write=persist_user_message,
            before_provider=ensure_v2_preflight,
        )
        if assistant_message is not None:
            return {
                'conversation': serialize_conversation(conversation, db=db, user=user),
                'user_message': serialize_message(user_message, db=db, user=user),
                'assistant_message': serialize_message(
                    assistant_message, db=db, user=user
                ),
            }

    # Ambiguous routing can dispatch a classifier even when it returns non-email.
    # Validate once before that first provider-capable boundary and reuse it if
    # composition later falls through to the V2 graph.
    ensure_v2_preflight()
    tool_logger.log(
        'email_intent_gate',
        f'start conversation_id={conversation.id} message_id={user_message.id}',
    )
    email_intent: EmailIntentDecision = build_email_intent_gate(settings).decide(
        conversation_context=email_context,
        latest_message=user_message.content,
    )
    tool_logger.log(
        'email_intent_gate',
        (
            f'result email_intent={email_intent.email_intent} '
            f'confidence={email_intent.confidence_score:g} '
            f'requires_rag_result={email_intent.requires_rag_result} '
            f'model={email_intent.model_name or "deterministic"}'
        ),
    )

    contextual_question = build_contextual_question(
        conversation=conversation,
        messages=messages,
        new_message=user_message.content,
        db=db,
        user=user,
    )
    pgvector_shared_shadow = bool(
        assistant_rag_owner == 'shadow'
        and settings.rag_retrieval_backend == 'pgvector'
        and not assistant_context_shadow
    )
    assistant_keyword_shadow = bool(
        assistant_rag_owner == 'shadow'
        and settings.rag_retrieval_backend == 'keyword'
        and not assistant_context_shadow
    )
    vector_store = (
        build_pgvector_search_store(db=db, settings=settings)
        if not v2_rag_owner and not pgvector_shared_shadow
        else None
    )
    confident_email_intent = (
        email_intent.email_intent
        and email_intent.confidence_score
        >= settings.assistant_email_agent_min_confidence
    )
    if confident_email_intent:
        if v2_rag_owner:
            vector_store = build_pgvector_search_store(db=db, settings=settings)
        recipient_resolution = resolve_email_recipients(
            db=db,
            latest_message=user_message.content,
            conversation_context=email_context,
        )
        tool_logger.log(
            'recipient_resolver',
            (
                f'result status={recipient_resolution.status} '
                f'candidate_count={len(recipient_resolution.candidates)} '
                f'reason={recipient_resolution.reason}'
            ),
        )
        rag_context = recent_assistant_context
        email_serving_dependencies: tuple[object, ...] = ()
        email_evidence_derived = False
        if email_intent.requires_rag_result:
            answer = _answer_question_or_raise(
                db=db,
                user=user,
                conversation=conversation,
                question=contextual_question,
                settings=settings,
                vector_store=vector_store,
                tool_logger=tool_logger,
                before_failure_write=persist_user_message,
            )
            rag_context = _render_rag_answer_for_email(answer)
            email_serving_dependencies = answer.serving_dependencies
            email_evidence_derived = True

        tool_logger.log(
            'email_draft_composer',
            f'start intent_type={email_intent.intent_type} requires_rag_result={email_intent.requires_rag_result}',
        )
        email_decision: EmailActionDecision = build_email_draft_composer(
            settings
        ).compose(
            conversation_context=email_context,
            latest_message=user_message.content,
            intent=email_intent,
            rag_context=rag_context,
            resolved_recipients=recipient_resolution.resolved_recipients,
        )
        tool_logger.log(
            'email_draft_composer',
            (
                f'result action={email_decision.action_type} '
                f'confidence={email_decision.confidence_score:g} '
                f'model={email_decision.model_name or "deterministic"}'
            ),
        )

        email_draft = email_decision.to_draft()
        if email_draft is not None:
            persist_user_message()
            assistant_message = append_assistant_message(
                db,
                user,
                conversation,
                content=assistant_email_draft_content(email_draft),
                citations=[],
                source_ids=[],
                source_links=[],
                source_snippets=[],
                permission_level=None,
                hidden_match_count=0,
                permission_notice=None,
                agent_run_id=None,
                metadata={
                    **email_draft_metadata(email_draft),
                    'agent_name': 'email_draft_composer',
                    'model_name': email_decision.model_name,
                    'confidence_score': email_decision.confidence_score,
                    'requires_rag_result': email_intent.requires_rag_result,
                },
                serving_dependencies=email_serving_dependencies,
                evidence_derived=email_evidence_derived,
            )
            return {
                'conversation': serialize_conversation(conversation, db=db, user=user),
                'user_message': serialize_message(user_message, db=db, user=user),
                'assistant_message': serialize_message(
                    assistant_message, db=db, user=user
                ),
            }

        if (
            email_decision.action_type == 'needs_clarification'
            and email_decision.clarification_question
        ):
            persist_user_message()
            assistant_message = append_assistant_message(
                db,
                user,
                conversation,
                content=email_decision.clarification_question,
                citations=[],
                source_ids=[],
                source_links=[],
                source_snippets=[],
                permission_level=None,
                hidden_match_count=0,
                permission_notice=None,
                agent_run_id=None,
                metadata={
                    'action_type': 'email_clarification',
                    'status': 'needs_input',
                    'prompt_version': EMAIL_ACTION_PROMPT_VERSION,
                    'agent_name': 'email_draft_composer',
                    'model_name': email_decision.model_name,
                    'confidence_score': email_decision.confidence_score,
                    'requires_rag_result': email_intent.requires_rag_result,
                },
            )
            return {
                'conversation': serialize_conversation(conversation, db=db, user=user),
                'user_message': serialize_message(user_message, db=db, user=user),
                'assistant_message': serialize_message(
                    assistant_message, db=db, user=user
                ),
            }

    if v2_rag_owner:
        ensure_v2_preflight()
        persist_user_message()
        # Close route reads before the facade opens its finalizer transaction.
        db.rollback()
        result = facade.invoke_assistant(
            actor=user, conversation_id=conversation_id, user_message_id=user_message.id,
            prepared_ingress=prepared_ingress,
        )
        message_id = _map_assistant_delivery(result)
        db.expire_all()
        assistant_message = get_owned_message(db, user, message_id)
        if (assistant_message.conversation_id != conversation_id
            or assistant_message.linked_agent_run_id != result.parent_agent_run_id):
            raise HTTPException(status_code=500, detail='Internal Server Error')
        return {
            'conversation': serialize_conversation(conversation, db=db, user=user),
            'user_message': serialize_message(user_message, db=db, user=user),
            'assistant_message': serialize_message(assistant_message, db=db, user=user),
        }

    shadow_prepared = None
    try:
        if pgvector_shared_shadow:
            shadow_prepared = facade.prepare_assistant_shadow_ingress(
                actor=user,
                conversation_id=conversation_id,
                user_message_id=user_message.id,
                caller_text=user_message.content,
            )
            if shadow_prepared.retrieval_query_text != contextual_question:
                raise ValueError('assistant shadow query bytes changed')
            answer = facade.invoke_assistant_pgvector_shadow_legacy(
                actor=user,
                prepared_text=shadow_prepared,
                legacy_invoke=lambda shared: answer_question_with_rag(
                    db=db,
                    user=user,
                    question=contextual_question,
                    settings=settings,
                    vector_store=build_pgvector_search_store(
                        db=db,
                        settings=settings,
                        shared_query_embedding=shared,
                    ),
                    tool_logger=tool_logger,
                    commit_agent_run=False,
                ),
            )
        else:
            if assistant_keyword_shadow:
                shadow_prepared = facade.prepare_assistant_shadow_ingress(
                    actor=user,
                    conversation_id=conversation_id,
                    user_message_id=user_message.id,
                    caller_text=user_message.content,
                )
                if shadow_prepared.retrieval_query_text != contextual_question:
                    raise ValueError('assistant shadow query bytes changed')
            answer = answer_question_with_rag(
                db=db,
                user=user,
                question=contextual_question,
                settings=settings,
                vector_store=vector_store,
                tool_logger=tool_logger,
                commit_agent_run=False,
            )
    except Exception as exc:
        append_failed_assistant_message(
            db,
            user,
            conversation,
            reason='rag_exception',
            failure_class=exc.__class__.__name__,
        )
        raise HTTPException(
            status_code=502,
            detail='assistant answer generation failed',
        ) from exc

    if assistant_context_shadow:
        with suppress(Exception):
            facade.observe_assistant_context_shadow(
                actor=user,
                contextual_query=contextual_question,
                public_agent_run_id=answer.agent_run_id,
            )
    elif assistant_keyword_shadow:
        with suppress(Exception):
            facade.observe_assistant_keyword_shadow(
                actor=user,
                prepared_text=shadow_prepared,
                legacy_delivery=answer,
            )

    try:
        assistant_message = append_assistant_message(
            db,
            user,
            conversation,
            content=answer.answer,
            citations=answer.citations,
            source_ids=answer.source_ids,
            source_links=answer.source_links,
            source_snippets=answer.source_snippets,
            permission_level=answer.permission_level,
            hidden_match_count=answer.hidden_match_count,
            permission_notice=answer.permission_notice,
            agent_run_id=answer.agent_run_id,
            metadata={
                'agent_name': answer.agent_name,
                'prompt_version': answer.prompt_version,
                'question': answer.question,
                'effective_backend': 'pgvector' if vector_store is not None else 'deterministic_lexical',
            },
            serving_dependencies=getattr(answer, 'serving_dependencies', ()),
        )
    except ValueError as exc:
        append_failed_assistant_message(
            db,
            user,
            conversation,
            reason='blank_answer',
            failure_class=exc.__class__.__name__,
            agent_run_id=answer.agent_run_id,
        )
        raise HTTPException(
            status_code=502,
            detail='assistant answer generation failed',
        ) from exc
    return {
        'conversation': serialize_conversation(conversation, db=db, user=user),
        'user_message': serialize_message(user_message, db=db, user=user),
        'assistant_message': serialize_message(assistant_message, db=db, user=user),
    }


@router.post(
    '/messages/{message_id}/email/send', response_model=AssistantEmailSendResponse
)
def send_assistant_email_draft(
    message_id: int,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    try:
        message = get_owned_message(db, user, message_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail='assistant message not found'
        ) from exc

    metadata = dict(message.metadata_ or {})
    # Persisted authority, not a mutable email flag, gates evidence-backed sends.
    if (message.content_write_mode is not None or message.serving_dependency_count
        or message.evidence_contract_version == 'assistant-evidence:v1'
        or metadata.get('evidence_derived') is True) and not (
            assistant_message_evidence_is_live(db, user=user, message=message)):
        raise HTTPException(status_code=409, detail='email draft evidence is unavailable')
    draft = metadata.get('email_draft')
    if metadata.get('action_type') != 'email_draft' or not isinstance(draft, dict):
        raise HTTPException(
            status_code=422, detail='assistant message is not an email draft'
        )
    if metadata.get('status') == 'sent':
        return {
            'message': serialize_message(message, db=db, user=user),
            'status': 'sent',
        }
    if metadata.get('status') != 'pending_approval':
        raise HTTPException(
            status_code=409, detail='email draft is not pending approval'
        )
    if metadata.get('evidence_derived') is True and not (
        assistant_message_evidence_is_live(db, user=user, message=message)
    ):
        raise HTTPException(
            status_code=409, detail='email draft evidence is unavailable'
        )

    try:
        result = GmailDraftSender(settings=settings).send(
            db=db,
            to=[str(item) for item in draft.get('to', [])],
            subject=str(draft.get('subject') or ''),
            body=str(draft.get('body') or ''),
        )
    except GmailSendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # 전송 결과는 원본 초안 메시지 metadata에 기록해 승인 이력을 남긴다.
    next_metadata = {
        **metadata,
        'status': 'sent',
        'sent_at': datetime.now(UTC).isoformat(),
        'gmail_message_id': result.message_id,
    }
    updated_message = update_message_metadata(db, user, message, next_metadata)
    return {
        'message': serialize_message(updated_message, db=db, user=user),
        'status': 'sent',
        'gmail_message_id': result.message_id,
    }
