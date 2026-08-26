import json
from dataclasses import dataclass
from typing import Any

from backend.app.agent_runtime import (
    EvidencePacket,
    LangChainInvocationPayload,
    TokenUsage,
    evaluate_agent_cost_budget,
)
from backend.app.agent_runtime.langchain_evidence import (
    render_bounded_evidence_rows,
)
from backend.app.agents.mail_document_agent.agent import MailDocumentAgentModelResponse

DEFAULT_OPENAI_MODEL = 'gpt-5.4-mini'
DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash'
DEFAULT_INPUT_COST_PER_1M = 0.15
DEFAULT_OUTPUT_COST_PER_1M = 0.60
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_MAX_INPUT_CHARS = 12_000
DEFAULT_PROMPT_OVERHEAD_CHARS = 3_200
OPENAI_COMPATIBLE_PROVIDERS = {'openai', 'azure_openai'}
MAIL_DOCUMENT_SYSTEM_PROMPT = (
    'You extract one reviewable company history candidate, timeline event, decision, or todo from Gmail, '
    'Drive, Calendar, or internal document evidence. Think like a Korean business operator: '
    'the review queue must show what happened, what needs a decision, or what the user should '
    'do next. The title and summary must be written in Korean. Do not copy raw email headers '
    'or long body excerpts into the summary. If action is required, use item_type=todo and '
    'populate structured_data with action_required, task_summary, recommended_next_step, '
    'assignee, due_date, counterparty, source_subject, business_context, evidence_sentence, '
    'summary_quality, and reviewability_decision where available. Determine if the evidence is '
    'For Calendar evidence, use item_type=timeline_event for confirmed meetings, milestones, '
    'and schedule confirmations; use item_type=todo for preparation, deadline, or follow-up work; '
    'copy calendar_id, calendar_name, calendar_start, calendar_end, calendar_location, '
    'calendar_organizer, calendar_attendee_summary, and event_context_key into structured_data. '
    'reviewable business evidence. If it is purely personal, private, promotional, a newsletter, '
    'or a low-signal notification without a concrete company work object, set '
    'is_business_related=false. Return ONLY JSON with fields: '
    'title, summary, item_type, confidence_score, is_business_related, project_tag, '
    'structured_data, and optional uncertainty_reason.'
)


class MailDocumentLlmProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class MailDocumentLlmSettings:
    enabled: bool = False
    provider_order: tuple[str, ...] = ('openai', 'gemini')
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M
    max_estimated_cost_usd: float | None = 0.001
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_evidence_messages: int = 12
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.2
    timeout_seconds: float = 30.0


class LangChainMailDocumentAgentModel:
    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        chat_model: Any,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.chat_model = chat_model
        self.max_input_chars = max_input_chars

    def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
        invocation = render_mail_document_langchain_invocation(
            packet,
            max_input_chars=self.max_input_chars,
        )
        messages = list(invocation.messages)
        try:
            response = self.chat_model.invoke(messages)
        except Exception as exc:  # pragma: no cover - provider-specific branch
            raise MailDocumentLlmProviderError(f'{self.provider} provider failed: {exc}') from exc

        payload = _parse_json_content(_response_content(response))
        input_tokens, output_tokens = _usage_tokens(response, messages, payload)

        return MailDocumentAgentModelResponse(
            title=str(payload.get('title') or 'Mail/Docs LLM candidate')[:200],
            summary=str(payload.get('summary') or 'Evidence was summarized by the LLM.')[:1200],
            item_type=_safe_item_type(payload.get('item_type')),
            confidence_score=_safe_confidence(payload.get('confidence_score')),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=self.model_name,
            uncertainty_reason=payload.get('uncertainty_reason'),
            is_business_related=_safe_bool(payload.get('is_business_related', True), default=True),
            project_tag=payload.get('project_tag'),
            structured_data=_safe_structured_data(payload.get('structured_data')),
        )


class FallbackMailDocumentAgentModel:
    def __init__(self, providers: list[Any]) -> None:
        self.providers = providers
        self.max_input_chars = (
            providers[0].max_input_chars if providers else DEFAULT_MAX_INPUT_CHARS
        )

    def extract(self, packet: EvidencePacket) -> MailDocumentAgentModelResponse:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.extract(packet)
            except MailDocumentLlmProviderError as exc:
                provider_name = getattr(provider, 'provider', 'unknown')
                errors.append(f'{provider_name}: {exc}')
        raise MailDocumentLlmProviderError('; '.join(errors) or 'no LLM providers configured')


def build_mail_document_llm_preflight(
    *,
    packet: EvidencePacket,
    settings: MailDocumentLlmSettings,
) -> dict[str, Any]:
    provider_order = _clean_provider_order(settings.provider_order)
    available_providers = _available_providers(provider_order, settings)
    if not settings.enabled:
        return _preflight_response(
            action='blocked',
            reason='llm_disabled',
            budget_status='disabled',
            provider_order=provider_order,
            available_providers=available_providers,
            settings=settings,
            packet=packet,
        )
    if not packet.messages:
        return _preflight_response(
            action='skip',
            reason='no_input',
            budget_status='no_input',
            provider_order=provider_order,
            available_providers=available_providers,
            settings=settings,
            packet=packet,
        )
    if not available_providers:
        return _preflight_response(
            action='blocked',
            reason='missing_credentials',
            budget_status='missing_credentials',
            provider_order=provider_order,
            available_providers=available_providers,
            settings=settings,
            packet=packet,
        )

    model_name = _model_for_provider(available_providers[0], settings)
    decision = evaluate_agent_cost_budget(
        model_name=model_name,
        token_usage=_estimated_token_usage(packet, settings),
        input_cost_per_1m=settings.input_cost_per_1m,
        output_cost_per_1m=settings.output_cost_per_1m,
        max_cost_usd=settings.max_estimated_cost_usd,
        cache_hit=False,
    )
    return {
        'action': decision.action,
        'reason': decision.reason,
        'budget_status': decision.budget_status,
        'model_name': decision.model_name,
        'provider_order': list(provider_order),
        'available_providers': available_providers,
        'estimated_input_tokens': decision.token_usage.input_tokens,
        'estimated_output_tokens': decision.token_usage.output_tokens,
        'estimated_total_tokens': decision.token_usage.total_tokens,
        'estimated_cost_usd': round(decision.estimated_cost_usd, 6),
        'budget_limit_usd': decision.budget_limit_usd,
        'evidence_message_count': len(packet.messages),
        'max_evidence_messages': settings.max_evidence_messages,
        'requires_paid_confirmation': True,
    }


def build_langchain_mail_document_agent_model(
    settings: MailDocumentLlmSettings,
) -> FallbackMailDocumentAgentModel:
    providers = []
    for provider in _available_providers(_clean_provider_order(settings.provider_order), settings):
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            providers.append(_build_openai_model(settings, provider=provider))
        elif provider == 'gemini':
            providers.append(_build_gemini_model(settings))
    if not providers:
        raise MailDocumentLlmProviderError('No LLM providers have credentials')
    return FallbackMailDocumentAgentModel(providers)


def render_mail_docs_llm_prompt(
    packet: EvidencePacket,
    *,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> str:
    evidence_rows = render_bounded_evidence_rows(
        packet,
        max_input_chars=max_input_chars,
    )
    return _render_mail_docs_prompt(packet, evidence_rows=evidence_rows)


def _render_mail_docs_prompt(
    packet: EvidencePacket,
    *,
    evidence_rows: list[dict[str, Any]],
) -> str:

    return json.dumps(
        {
            'task': 'Extract structured output and project tags from mail/document evidence for human review.',
            'allowed_item_types': ['history_event', 'decision_record', 'timeline_event', 'todo'],
            'requirements': [
                'Use only the provided evidence.',
                'The title and summary must be written in Korean.',
                'Identify whether the evidence is business related.',
                'Before summarizing, write structured_data.reviewability_decision as "reviewable" or "not_reviewable".',
                'Only set is_business_related=true when the evidence contains a concrete company work object: a decision, request, deadline, deliverable, meeting, contract, project, customer issue, budget, revenue, policy, or document change that should be reviewed.',
                'For personal mail, newsletters, promotions, automated alerts with no action, receipts, greetings, or contextless notifications, set is_business_related=false and do not invent a business summary.',
                'Assign a project_tag if a specific project is mentioned.',
                'Populate structured_data with concrete fields for the source type.',
                'For todos, include action_required, task_summary, recommended_next_step, assignee, due_date, counterparty, and business_context when evidence supports them.',
                'For Calendar timeline events, include result_summary plus calendar_id, calendar_name, calendar_start, calendar_end, calendar_location, calendar_organizer, calendar_attendee_summary, and event_context_key.',
                'For personal or low-signal Calendar events, set is_business_related=false.',
                'Keep raw email headers and body excerpts only as evidence, not as title or summary.',
                'Summaries must explain the business meaning in one or two Korean sentences, not merely restate that an email or file exists.',
                'Preserve uncertainty_reason when evidence is weak or incomplete.',
            ],
            'source_window': packet.source_window,
            'evidence': evidence_rows,
        },
        ensure_ascii=False,
        default=str,
    )


def render_mail_document_langchain_invocation(
    packet: EvidencePacket,
    *,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> LangChainInvocationPayload:
    return LangChainInvocationPayload(
        messages=(
            ('system', MAIL_DOCUMENT_SYSTEM_PROMPT),
            (
                'user',
                render_mail_docs_llm_prompt(
                    packet,
                    max_input_chars=max_input_chars,
                ),
            ),
        ),
    )


def _build_openai_model(
    settings: MailDocumentLlmSettings,
    *,
    provider: str = 'openai',
) -> LangChainMailDocumentAgentModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MailDocumentLlmProviderError('langchain-openai is not installed') from exc
    return LangChainMailDocumentAgentModel(
        provider=provider,
        model_name=settings.openai_model,
        max_input_chars=_effective_max_input_chars(settings),
        chat_model=ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=1,
            max_completion_tokens=settings.max_output_tokens,
        ),
    )


def _build_gemini_model(settings: MailDocumentLlmSettings) -> LangChainMailDocumentAgentModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MailDocumentLlmProviderError('langchain-google-genai is not installed') from exc
    return LangChainMailDocumentAgentModel(
        provider='gemini',
        model_name=settings.gemini_model,
        max_input_chars=_effective_max_input_chars(settings),
        chat_model=ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=1,
            max_tokens=settings.max_output_tokens,
        ),
    )


def _preflight_response(
    *,
    action: str,
    reason: str,
    budget_status: str,
    provider_order: tuple[str, ...],
    available_providers: list[str],
    settings: MailDocumentLlmSettings,
    packet: EvidencePacket,
) -> dict[str, Any]:
    token_usage = _estimated_token_usage(packet, settings)
    return {
        'action': action,
        'reason': reason,
        'budget_status': budget_status,
        'model_name': _model_for_provider(available_providers[0], settings) if available_providers else None,
        'provider_order': list(provider_order),
        'available_providers': available_providers,
        'estimated_input_tokens': token_usage.input_tokens,
        'estimated_output_tokens': token_usage.output_tokens,
        'estimated_total_tokens': token_usage.total_tokens,
        'estimated_cost_usd': 0.0,
        'budget_limit_usd': settings.max_estimated_cost_usd,
        'evidence_message_count': len(packet.messages),
        'max_evidence_messages': settings.max_evidence_messages,
        'requires_paid_confirmation': True,
    }


def _estimated_token_usage(packet: EvidencePacket, settings: MailDocumentLlmSettings) -> TokenUsage:
    max_input_chars = _effective_max_input_chars(settings)
    prompt = _render_legacy_mail_budget_prompt(
        packet,
        max_input_chars=max_input_chars,
    )
    affordable_prompt_chars = _affordable_prompt_chars(settings)
    for _ in range(4):
        if affordable_prompt_chars is None or len(prompt) <= affordable_prompt_chars or max_input_chars <= 0:
            break
        overage = len(prompt) - affordable_prompt_chars
        max_input_chars = max(0, max_input_chars - overage - 128)
        prompt = _render_legacy_mail_budget_prompt(
            packet,
            max_input_chars=max_input_chars,
        )
    return TokenUsage(
        input_tokens=max(1, len(prompt)),
        output_tokens=settings.max_output_tokens,
    )


def _render_legacy_mail_budget_prompt(
    packet: EvidencePacket,
    *,
    max_input_chars: int,
) -> str:
    """Reproduce the pre-round-2 estimate shape; never send this to a provider."""

    evidence_rows: list[dict[str, Any]] = []
    remaining_chars = max_input_chars
    for message in packet.messages:
        text = message.text[: max(0, min(len(message.text), remaining_chars))]
        remaining_chars -= len(text)
        evidence_rows.append(
            {
                'source_type': message.metadata.get('source_type', 'unknown'),
                'source_id': message.source_id,
                'source_url': message.source_url,
                'timestamp': message.timestamp,
                'author': message.author,
                'permission_level': message.permission_level,
                'parser_status': message.metadata.get('parser_status'),
                'section_path': message.metadata.get('section_path'),
                'calendar_id': message.metadata.get('calendar_id'),
                'calendar_name': message.metadata.get('calendar_summary'),
                'calendar_start': message.metadata.get('event_start')
                or message.metadata.get('start'),
                'calendar_end': message.metadata.get('event_end')
                or message.metadata.get('end'),
                'calendar_location': message.metadata.get('location'),
                'calendar_organizer': message.metadata.get('organizer_email'),
                'calendar_attendee_domains': message.metadata.get('attendee_domains'),
                'event_context_key': message.metadata.get('event_context_key'),
                'text': text,
            }
        )
        if remaining_chars <= 0:
            break
    return _render_mail_docs_prompt(packet, evidence_rows=evidence_rows)


def _effective_max_input_chars(settings: MailDocumentLlmSettings) -> int:
    affordable_prompt_chars = _affordable_prompt_chars(settings)
    if affordable_prompt_chars is None:
        return settings.max_input_chars
    affordable_evidence_chars = affordable_prompt_chars - DEFAULT_PROMPT_OVERHEAD_CHARS
    return max(0, min(settings.max_input_chars, affordable_evidence_chars))


def _affordable_prompt_chars(settings: MailDocumentLlmSettings) -> int | None:
    if settings.max_estimated_cost_usd is None:
        return None
    total_budget_units = settings.max_estimated_cost_usd * 1_000_000
    reserved_output_units = settings.max_output_tokens * settings.output_cost_per_1m
    remaining_input_units = total_budget_units - reserved_output_units
    if remaining_input_units <= 0:
        return 0
    return int(remaining_input_units // settings.input_cost_per_1m)


def _available_providers(
    provider_order: tuple[str, ...],
    settings: MailDocumentLlmSettings,
) -> list[str]:
    available = []
    for provider in provider_order:
        if provider in OPENAI_COMPATIBLE_PROVIDERS and settings.openai_api_key:
            available.append(provider)
        if provider == 'gemini' and settings.gemini_api_key:
            available.append(provider)
    return available


def _clean_provider_order(provider_order: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    cleaned = []
    for provider in provider_order:
        normalized = provider.strip().lower()
        if normalized in {*OPENAI_COMPATIBLE_PROVIDERS, 'gemini'} and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return tuple(cleaned) or ('openai', 'gemini')


def _model_for_provider(provider: str, settings: MailDocumentLlmSettings) -> str:
    return settings.gemini_model if provider == 'gemini' else settings.openai_model


def _response_content(response: Any) -> str:
    content = getattr(response, 'content', response)
    if isinstance(content, list):
        return ''.join(str(part.get('text', part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _parse_json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        cleaned = cleaned.removeprefix('json').strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise MailDocumentLlmProviderError('LLM response was not valid JSON') from exc
    if not isinstance(payload, dict):
        raise MailDocumentLlmProviderError('LLM response JSON must be an object')
    return payload


def _usage_tokens(response: Any, messages: list[tuple[str, str]], payload: dict[str, Any]) -> tuple[int, int]:
    usage = getattr(response, 'usage_metadata', None) or getattr(response, 'response_metadata', {}).get('token_usage', {})
    input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens')
    output_tokens = usage.get('output_tokens') or usage.get('completion_tokens')
    if input_tokens is None:
        input_tokens = max(1, sum(len(message[1]) for message in messages) // 4)
    if output_tokens is None:
        input_tokens = int(input_tokens)
        output_tokens = max(1, len(json.dumps(payload, ensure_ascii=False)) // 4)
    return int(input_tokens), int(output_tokens)


def _safe_item_type(raw_item_type: Any) -> str:
    item_type = str(raw_item_type or 'history_event')
    return item_type if item_type in {'history_event', 'decision_record', 'timeline_event', 'todo'} else 'history_event'


def _safe_confidence(raw_confidence: Any) -> float:
    try:
        return min(max(float(raw_confidence), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.7


def _safe_bool(raw_value: Any, *, default: bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {'true', '1', 'yes', 'y'}:
            return True
        if normalized in {
            'false',
            '0',
            'no',
            'n',
            'personal',
            'private',
            'not_business',
            'not business',
            'non-business',
            'nonbusiness',
            'not_reviewable',
            'not reviewable',
            '비업무',
            '개인',
            '개인 메일',
            '업무 관련 없음',
        }:
            return False
    if raw_value is None:
        return default
    return bool(raw_value)


def _safe_structured_data(raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        return {}
    return {str(key): value for key, value in raw_value.items()}
