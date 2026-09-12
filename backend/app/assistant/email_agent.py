import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.assistant.email_actions import EmailDraft
from backend.app.core.config import Settings

EMAIL_ACTION_TYPES = {'not_email', 'email_draft', 'needs_clarification'}
EMAIL_INTENT_PROMPT_VERSION = 'assistant-email-intent-gate:v1'
EMAIL_DRAFT_PROMPT_VERSION = 'assistant-email-draft-composer:v1'
EMAIL_INTENT_TYPES = {'none', 'compose', 'send', 'forward', 'reply'}


class EmailActionAgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmailIntentDecision:
    email_intent: bool
    intent_type: str = 'none'
    confidence_score: float = 1.0
    missing_fields: list[str] = field(default_factory=list)
    requires_rag_result: bool = False
    reason: str = ''
    model_name: str | None = None


@dataclass(frozen=True)
class EmailActionDecision:
    action_type: str
    to: list[str] = field(default_factory=list)
    subject: str = ''
    body: str = ''
    clarification_question: str = ''
    reply: str = ''
    confidence_score: float = 1.0
    model_name: str | None = None

    def to_draft(self) -> EmailDraft | None:
        if self.action_type != 'email_draft':
            return None
        if not self.to or not self.subject.strip() or not self.body.strip():
            return None
        return EmailDraft(
            to=[item.strip() for item in self.to if item.strip()],
            subject=self.subject.strip(),
            body=self.body.strip(),
        )


class NoopEmailIntentGate:
    def decide(self, **kwargs) -> EmailIntentDecision:
        return EmailIntentDecision(email_intent=False, confidence_score=0.0)


class NoopEmailDraftComposer:
    def compose(self, **kwargs) -> EmailActionDecision:
        return EmailActionDecision(action_type='not_email', confidence_score=0.0)


class EmailIntentGate:
    def __init__(self, *, model) -> None:
        self.model = model

    def decide(self, *, conversation_context: str, latest_message: str) -> EmailIntentDecision:
        try:
            return self.model.decide(
                conversation_context=conversation_context,
                latest_message=latest_message,
            )
        except EmailActionAgentError:
            return EmailIntentDecision(email_intent=False, confidence_score=0.0)


class EmailDraftComposer:
    def __init__(self, *, model) -> None:
        self.model = model

    def compose(
        self,
        *,
        conversation_context: str,
        latest_message: str,
        intent: EmailIntentDecision,
        rag_context: str = '',
        resolved_recipients: list[dict[str, object]] | None = None,
    ) -> EmailActionDecision:
        try:
            return self.model.compose(
                conversation_context=conversation_context,
                latest_message=latest_message,
                intent=intent,
                rag_context=rag_context,
                resolved_recipients=resolved_recipients or [],
            )
        except EmailActionAgentError:
            return EmailActionDecision(action_type='not_email', confidence_score=0.0)


class LangChainEmailIntentGateModel:
    def __init__(
        self,
        *,
        chat_model: Any,
        model_name: str,
        max_input_chars: int,
    ) -> None:
        self.chat_model = chat_model
        self.model_name = model_name
        self.max_input_chars = max_input_chars

    def decide(self, *, conversation_context: str, latest_message: str) -> EmailIntentDecision:
        messages = [
            (
                'system',
                (
                    'You are ParaWorks Email Intent Gate. '
                    'Your only job is to decide whether the latest user message requests an email action. '
                    'Email action means the user wants to write, draft, send, forward, or reply to an email/message to a person or group. '
                    'Use recent conversation only to resolve omitted recipients, groups, or content when the latest message is a direct continuation of an email request. '
                    'Do not answer the user. Do not write the email body. Do not call RAG. Do not send email. Return strict JSON only.'
                ),
            ),
            (
                'user',
                render_email_intent_prompt(
                    conversation_context=conversation_context,
                    latest_message=latest_message,
                    max_input_chars=self.max_input_chars,
                ),
            ),
        ]
        try:
            response = self.chat_model.invoke(messages)
        except Exception as exc:  # pragma: no cover - 외부 LLM provider 장애 경로
            raise EmailActionAgentError(f'{self.model_name} email intent model failed: {exc}') from exc

        payload = _parse_json_object(_response_content(response))
        decision = _intent_decision_from_payload(payload)
        return EmailIntentDecision(
            email_intent=decision.email_intent,
            intent_type=decision.intent_type,
            missing_fields=decision.missing_fields,
            requires_rag_result=decision.requires_rag_result,
            reason=decision.reason,
            confidence_score=decision.confidence_score,
            model_name=self.model_name,
        )


class LangChainEmailDraftComposerModel:
    def __init__(
        self,
        *,
        chat_model: Any,
        model_name: str,
        max_input_chars: int,
    ) -> None:
        self.chat_model = chat_model
        self.model_name = model_name
        self.max_input_chars = max_input_chars

    def compose(
        self,
        *,
        conversation_context: str,
        latest_message: str,
        intent: EmailIntentDecision,
        rag_context: str = '',
        resolved_recipients: list[dict[str, object]] | None = None,
    ) -> EmailActionDecision:
        messages = [
            (
                'system',
                (
                    'You are ParaWorks Email Draft Agent. '
                    'The user has requested an email action. '
                    'Create a concise Korean business email draft. '
                    'Never send email directly. '
                    'If recipient or content is missing, ask one concise Korean clarification question. '
                    'Return strict JSON only.'
                ),
            ),
            (
                'user',
                render_email_draft_prompt(
                    conversation_context=conversation_context,
                    latest_message=latest_message,
                    intent=intent,
                    rag_context=rag_context,
                    resolved_recipients=resolved_recipients or [],
                    max_input_chars=self.max_input_chars,
                ),
            ),
        ]
        try:
            response = self.chat_model.invoke(messages)
        except Exception as exc:  # pragma: no cover - 외부 LLM provider 장애 경로
            raise EmailActionAgentError(f'{self.model_name} email draft model failed: {exc}') from exc

        payload = _parse_json_object(_response_content(response))
        decision = _decision_from_payload(payload)
        return EmailActionDecision(
            action_type=decision.action_type,
            to=decision.to,
            subject=decision.subject,
            body=decision.body,
            clarification_question=decision.clarification_question,
            reply=decision.reply,
            confidence_score=decision.confidence_score,
            model_name=self.model_name,
        )


def build_email_intent_gate(settings: Settings) -> EmailIntentGate | NoopEmailIntentGate:
    if settings.paraworks_demo_mode or not settings.assistant_email_agent_enabled or not settings.openai_api_key:
        return NoopEmailIntentGate()

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:  # pragma: no cover - 선택 의존성 누락 경로
        return NoopEmailIntentGate()

    return EmailIntentGate(
        model=LangChainEmailIntentGateModel(
            model_name=settings.assistant_email_agent_model,
            max_input_chars=settings.assistant_email_agent_max_input_chars,
            chat_model=ChatOpenAI(
                model=settings.assistant_email_agent_model,
                api_key=settings.openai_api_key,
                temperature=settings.assistant_email_agent_temperature,
                timeout=settings.assistant_email_agent_timeout_seconds,
                max_tokens=settings.assistant_email_agent_max_output_tokens,
                max_retries=1,
            ),
        )
    )


def build_email_draft_composer(
    settings: Settings, *, before_provider: Callable[[], None] | None = None,
) -> EmailDraftComposer | NoopEmailDraftComposer:
    if settings.paraworks_demo_mode or not settings.assistant_email_agent_enabled or not settings.openai_api_key:
        return NoopEmailDraftComposer()

    if before_provider is not None:
        before_provider()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:  # pragma: no cover - 선택 의존성 누락 경로
        return NoopEmailDraftComposer()

    return EmailDraftComposer(
        model=LangChainEmailDraftComposerModel(
            model_name=settings.assistant_email_draft_agent_model,
            max_input_chars=settings.assistant_email_agent_max_input_chars,
            chat_model=ChatOpenAI(
                model=settings.assistant_email_draft_agent_model,
                api_key=settings.openai_api_key,
                temperature=settings.assistant_email_agent_temperature,
                timeout=settings.assistant_email_agent_timeout_seconds,
                max_tokens=settings.assistant_email_agent_max_output_tokens,
                max_retries=1,
            ),
        )
    )


def render_email_action_context(*, messages: list[Any], max_chars: int) -> str:
    # 최근 대화를 완성된 JSON row 단위로 담아 모델이 잘린 JSON을 해석하지 않게 한다.
    selected_rows: list[dict[str, str]] = []
    for message in reversed(messages[-12:]):
        row = {
            'role': str(getattr(message, 'role', '')),
            'content': str(getattr(message, 'content', '')),
        }
        candidate = [row, *selected_rows]
        rendered = json.dumps(candidate, ensure_ascii=False)
        if len(rendered) > max_chars and selected_rows:
            break
        if len(rendered) > max_chars:
            row['content'] = row['content'][-max(80, max_chars // 2):]
            candidate = [row]
        selected_rows = candidate
    return json.dumps(selected_rows, ensure_ascii=False)


def render_recent_assistant_context_for_email(*, messages: list[Any], max_chars: int) -> str:
    # "이 내용으로" 같은 지시가 이전 AI 답변을 이메일 본문 재료로 참조할 수 있게 보존한다.
    chunks = []
    for message in reversed(messages):
        if getattr(message, 'role', '') != 'assistant':
            continue
        content = str(getattr(message, 'content', '')).strip()
        if not content:
            continue
        chunks.append(f'Previous assistant answer:\n{content}')
        rendered = '\n\n'.join(reversed(chunks))
        if len(rendered) >= max_chars:
            return rendered[-max_chars:]
        if len(chunks) >= 3:
            break
    return '\n\n'.join(reversed(chunks))[-max_chars:]


def render_email_intent_prompt(
    *,
    conversation_context: str,
    latest_message: str,
    max_input_chars: int,
) -> str:
    payload = {
        'prompt_version': EMAIL_INTENT_PROMPT_VERSION,
        'task': 'Decide only whether the latest user message requests an email action.',
        'conversation_context': conversation_context[-max_input_chars:],
        'latest_message': latest_message,
        'email_action_definition': (
            'The user asks to write, draft, send, forward, or reply to an email/message '
            'to a person or group.'
        ),
        'rules': [
            'Return email_intent=true only when the latest message clearly requests an email-related action.',
            'Return email_intent=false for company knowledge questions, search requests, summaries, greetings, explanations, or ordinary chat.',
            'Set requires_rag_result=true only when the user asks to find/search/answer company-memory information and email that result.',
            'Use recent conversation only when the latest message is a direct continuation such as "send that to them".',
            'Do not write the email body. Do not answer the user. Do not call RAG.',
            'Return confidence_score between 0 and 1. Use low confidence when intent is ambiguous.',
        ],
        'json_schema': {
            'email_intent': True,
            'intent_type': 'compose | send | forward | reply | none',
            'confidence_score': 0.0,
            'missing_fields': ['recipient', 'content'],
            'requires_rag_result': False,
            'reason': 'short English reason',
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def render_email_draft_prompt(
    *,
    conversation_context: str,
    latest_message: str,
    intent: EmailIntentDecision,
    rag_context: str,
    resolved_recipients: list[dict[str, object]] | None = None,
    max_input_chars: int,
) -> str:
    payload = {
        'prompt_version': EMAIL_DRAFT_PROMPT_VERSION,
        'task': 'Create an approval-only email draft after Email Intent Gate already accepted the request.',
        'conversation_context': conversation_context[-max_input_chars:],
        'latest_message': latest_message,
        'intent': {
            'intent_type': intent.intent_type,
            'missing_fields': intent.missing_fields,
            'requires_rag_result': intent.requires_rag_result,
            'reason': intent.reason,
        },
        'rag_context': rag_context[-max_input_chars:],
        'resolved_recipients': resolved_recipients or [],
        'rules': [
            'For a complete email request, return action_type=email_draft.',
            'If recipient or content is still missing, return action_type=needs_clarification.',
            'If resolved_recipients is not empty, use those email addresses as the to list.',
            'Use rag_context as the factual content source when it is provided.',
            'If the latest message only provides a recipient or address, use conversation_context and rag_context as the email body source.',
            'If the user says "this content", "that summary", or similar continuation, use the latest relevant assistant answer from rag_context.',
            'Never invent a recipient that is not in latest_message or conversation_context.',
            'Write a concise Korean business subject and body.',
            'Never send email directly; this draft always requires user approval.',
        ],
        'json_schema': {
            'action_type': 'email_draft | needs_clarification | not_email',
            'to': ['recipient@example.com'],
            'subject': 'Korean business email subject',
            'body': 'Korean business email body',
            'clarification_question': 'Korean question when required information is missing',
            'reply': '',
            'confidence_score': 0.0,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _intent_decision_from_payload(payload: dict[str, Any]) -> EmailIntentDecision:
    intent_type = str(payload.get('intent_type') or 'none').strip()
    if intent_type not in EMAIL_INTENT_TYPES:
        intent_type = 'none'

    missing_fields = payload.get('missing_fields') or []
    if not isinstance(missing_fields, list):
        missing_fields = []

    email_intent = _bool(payload.get('email_intent'))
    if not email_intent:
        intent_type = 'none'

    return EmailIntentDecision(
        email_intent=email_intent,
        intent_type=intent_type,
        confidence_score=_confidence_score(payload.get('confidence_score')),
        missing_fields=[str(item).strip() for item in missing_fields if str(item).strip()],
        requires_rag_result=_bool(payload.get('requires_rag_result')),
        reason=str(payload.get('reason') or '').strip(),
    )


def _decision_from_payload(payload: dict[str, Any]) -> EmailActionDecision:
    action_type = str(payload.get('action_type') or 'not_email').strip()
    if action_type not in EMAIL_ACTION_TYPES:
        action_type = 'not_email'

    to = payload.get('to') or []
    if not isinstance(to, list):
        to = []
    recipients = [str(item).strip() for item in to if str(item).strip()]

    return EmailActionDecision(
        action_type=action_type,
        to=recipients,
        subject=str(payload.get('subject') or '').strip(),
        body=str(payload.get('body') or '').strip(),
        clarification_question=str(payload.get('clarification_question') or '').strip(),
        reply=str(payload.get('reply') or '').strip(),
        confidence_score=_confidence_score(payload.get('confidence_score')),
    )


def _confidence_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(parsed, 1.0))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`').strip()
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EmailActionAgentError('email action model returned invalid JSON') from exc
    if not isinstance(payload, dict):
        raise EmailActionAgentError('email action model returned non-object JSON')
    return payload


def _response_content(response: Any) -> str:
    content = getattr(response, 'content', response)
    if isinstance(content, list):
        return ''.join(str(part.get('text', part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)
