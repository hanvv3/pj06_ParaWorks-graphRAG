import json
from dataclasses import dataclass
from typing import Any

from backend.app.agent_runtime import EvidencePacket
from backend.app.agents.rag_orchestrator_agent.agent import RagModelResponse
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    PreparedAnswerInvocation as PreparedAnswerInvocation,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    ProviderAnswerEnvelope as ProviderAnswerEnvelope,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    StructuredRagAnswerModel as StructuredRagAnswerModel,
)

DEFAULT_RAG_OPENAI_MODEL = 'gpt-5.4'
DEFAULT_RAG_GEMINI_MODEL = 'gemini-2.5-flash'
DEFAULT_MAX_INPUT_CHARS = 12_000
DEFAULT_MAX_OUTPUT_TOKENS = 512
OPENAI_COMPATIBLE_PROVIDERS = {'openai', 'azure_openai'}


class RagLlmProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RagLlmSettings:
    enabled: bool = False
    provider_order: tuple[str, ...] = ('openai', 'gemini')
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_primary_model: str = DEFAULT_RAG_OPENAI_MODEL
    openai_fallback_model: str | None = None
    gemini_model: str = DEFAULT_RAG_GEMINI_MODEL
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.2
    timeout_seconds: float = 30.0


class LangChainRagOrchestratorModel:
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

    def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
        messages = [
            (
                'system',
                'You are the ParaWorks AI assistant. Answer in Korean using only the provided evidence. '
                'If evidence is insufficient, say what cannot be confirmed. Do not reveal hidden or unauthorized sources.',
            ),
            ('user', render_rag_llm_prompt(question=question, packet=packet, max_input_chars=self.max_input_chars)),
        ]
        try:
            response = self.chat_model.invoke(messages)
        except Exception as exc:  # pragma: no cover - provider별 예외 경로
            raise RagLlmProviderError(f'{self.provider}:{self.model_name} provider failed: {exc}') from exc

        answer = _response_content(response).strip()
        if not answer:
            raise RagLlmProviderError(f'{self.provider}:{self.model_name} returned an empty answer')

        input_tokens, output_tokens = _usage_tokens(response, messages, answer)
        return RagModelResponse(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=self.model_name,
        )


class FallbackRagOrchestratorModel:
    def __init__(self, providers: list[Any]) -> None:
        self.providers = providers

    def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.answer(question, packet)
            except RagLlmProviderError as exc:
                provider_name = getattr(provider, 'provider', 'unknown')
                model_name = getattr(provider, 'model_name', 'unknown')
                errors.append(f'{provider_name}:{model_name}: {exc}')
        raise RagLlmProviderError('; '.join(errors) or 'no RAG LLM providers configured')


def build_langchain_rag_orchestrator_model(settings: RagLlmSettings) -> FallbackRagOrchestratorModel:
    if not settings.enabled:
        raise RagLlmProviderError('RAG LLM is disabled')

    providers: list[Any] = []
    for provider in _available_providers(_clean_provider_order(settings.provider_order), settings):
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            for model_name in _openai_model_order(settings):
                providers.append(_build_openai_model(settings, provider=provider, model_name=model_name))
        elif provider == 'gemini':
            providers.append(_build_gemini_model(settings))
    if not providers:
        raise RagLlmProviderError('No RAG LLM providers have credentials')
    return FallbackRagOrchestratorModel(providers)


def render_rag_llm_prompt(
    *,
    question: str,
    packet: EvidencePacket,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> str:
    evidence_rows = []
    remaining_chars = max_input_chars
    for message in packet.messages:
        text = message.text[: max(0, min(len(message.text), remaining_chars))]
        remaining_chars -= len(text)
        evidence_rows.append(
            {
                'source_id': message.source_id,
                'source_url': message.source_url,
                'timestamp': message.timestamp,
                'author': message.author,
                'channel_name': message.metadata.get('channel_name'),
                'category': message.metadata.get('category'),
                'topic_tag': message.metadata.get('topic_tag'),
                'permission_level': message.permission_level,
                'text': text,
            }
        )
        if remaining_chars <= 0:
            break

    return json.dumps(
        {
            'task': 'Answer the user question with permission-safe company memory evidence.',
            'question': question,
            'requirements': [
                'Use only the provided evidence.',
                'Answer in natural Korean.',
                'Keep the answer concise and business-ready.',
                'Do not invent facts when evidence is missing.',
            ],
            'source_window': packet.source_window,
            'evidence': evidence_rows,
        },
        ensure_ascii=False,
    )


def _build_openai_model(
    settings: RagLlmSettings,
    *,
    provider: str,
    model_name: str,
) -> LangChainRagOrchestratorModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - 선택 설치 패키지 경로
        raise RagLlmProviderError('langchain-openai is not installed') from exc

    return LangChainRagOrchestratorModel(
        provider=provider,
        model_name=model_name,
        max_input_chars=settings.max_input_chars,
        chat_model=ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=1,
        ),
    )


def _build_gemini_model(settings: RagLlmSettings) -> LangChainRagOrchestratorModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover - 선택 설치 패키지 경로
        raise RagLlmProviderError('langchain-google-genai is not installed') from exc

    return LangChainRagOrchestratorModel(
        provider='gemini',
        model_name=settings.gemini_model,
        max_input_chars=settings.max_input_chars,
        chat_model=ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=1,
        ),
    )


def _available_providers(provider_order: tuple[str, ...], settings: RagLlmSettings) -> list[str]:
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


def _openai_model_order(settings: RagLlmSettings) -> list[str]:
    models = [settings.openai_primary_model, settings.openai_fallback_model]
    cleaned = []
    for model_name in models:
        normalized = (model_name or '').strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned or [DEFAULT_RAG_OPENAI_MODEL]


def _response_content(response: Any) -> str:
    content = getattr(response, 'content', response)
    if isinstance(content, list):
        return ''.join(str(part.get('text', part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _usage_tokens(response: Any, messages: list[tuple[str, str]], answer: str) -> tuple[int, int]:
    usage = getattr(response, 'usage_metadata', None) or getattr(response, 'response_metadata', {}).get('token_usage', {})
    input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens')
    output_tokens = usage.get('output_tokens') or usage.get('completion_tokens')
    if input_tokens is None:
        input_tokens = max(1, sum(len(message[1]) for message in messages) // 4)
    if output_tokens is None:
        output_tokens = max(1, len(answer) // 4)
    return int(input_tokens), int(output_tokens)
