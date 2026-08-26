from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.app.agents.mail_document_agent import (
    DeterministicMailDocumentAgentModel,
    MailDocumentLlmProviderError,
    MailDocumentLlmSettings,
    build_langchain_mail_document_agent_model,
)
from backend.app.core.config import Settings

MODEL_ROUTE_VERSION = 'review-model-route:v1'
OPENAI_COMPATIBLE_PROVIDERS = frozenset({'openai', 'azure_openai'})


class ReviewModelUnavailableError(RuntimeError):
    code = 'model_unavailable'


@dataclass(frozen=True)
class RoutedReviewModel:
    model: Any
    model_name: str
    route_version: str
    deterministic: bool


def build_mail_document_model_route(
    settings: Settings,
    *,
    model_builder: Callable[[MailDocumentLlmSettings], Any] | None = None,
) -> RoutedReviewModel:
    if settings.paraworks_demo_mode:
        return RoutedReviewModel(
            model=DeterministicMailDocumentAgentModel(),
            model_name='deterministic-mail-document-agent-model',
            route_version=f'{MODEL_ROUTE_VERSION}:demo:mail-document',
            deterministic=True,
        )

    llm_settings = build_mail_document_llm_settings(settings)
    available = _available_provider_routes(settings)
    if not settings.agent_llm_enabled or not available:
        raise ReviewModelUnavailableError('review model is unavailable')
    builder = model_builder or build_langchain_mail_document_agent_model
    try:
        model = builder(llm_settings)
    except (MailDocumentLlmProviderError, ImportError, ValueError):
        raise ReviewModelUnavailableError('review model is unavailable') from None
    return RoutedReviewModel(
        model=model,
        model_name=available[0][1],
        route_version=_route_version('mail-document', available),
        deterministic=False,
    )


def build_memory_model_route(
    settings: Settings,
    *,
    max_output_tokens: int,
    chat_model_builder: Callable[..., Any] | None = None,
) -> RoutedReviewModel:
    if settings.paraworks_demo_mode:
        return RoutedReviewModel(
            model=None,
            model_name='deterministic-memory-extraction-model',
            route_version=f'{MODEL_ROUTE_VERSION}:demo:memory-extraction',
            deterministic=True,
        )

    available = _available_provider_routes(settings)
    if not settings.agent_llm_enabled or not available:
        raise ReviewModelUnavailableError('review model is unavailable')
    builder = chat_model_builder or build_langchain_review_chat_model
    try:
        model = builder(settings, max_output_tokens=max_output_tokens)
    except (ImportError, ValueError, TypeError):
        raise ReviewModelUnavailableError('review model is unavailable') from None
    return RoutedReviewModel(
        model=model,
        model_name=available[0][1],
        route_version=_route_version('memory-extraction', available),
        deterministic=False,
    )


def build_mail_document_llm_settings(settings: Settings) -> MailDocumentLlmSettings:
    return MailDocumentLlmSettings(
        enabled=settings.agent_llm_enabled,
        provider_order=_provider_order(settings),
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key or settings.google_api_key,
        openai_model=settings.agent_llm_openai_model,
        gemini_model=settings.agent_llm_gemini_model,
        input_cost_per_1m=settings.agent_llm_input_cost_per_1m_tokens,
        output_cost_per_1m=settings.agent_llm_output_cost_per_1m_tokens,
        max_estimated_cost_usd=settings.agent_llm_max_estimated_cost_usd,
        max_input_chars=settings.agent_llm_max_input_chars,
        max_evidence_messages=settings.agent_llm_max_evidence_messages,
        max_output_tokens=settings.agent_llm_max_output_tokens,
        temperature=settings.agent_llm_temperature,
        timeout_seconds=settings.agent_llm_timeout_seconds,
    )


def build_langchain_review_chat_model(
    settings: Settings,
    *,
    max_output_tokens: int,
) -> Any:
    if max_output_tokens <= 0:
        raise ValueError('max output tokens must be positive')
    available = _available_provider_routes(settings)
    if not settings.agent_llm_enabled or not available:
        raise ReviewModelUnavailableError('review model is unavailable')
    provider, model_name = available[0]
    try:
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                api_key=settings.openai_api_key,
                temperature=settings.agent_llm_temperature,
                timeout=settings.agent_llm_timeout_seconds,
                max_retries=1,
                max_completion_tokens=max_output_tokens,
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key or settings.google_api_key,
            temperature=settings.agent_llm_temperature,
            timeout=settings.agent_llm_timeout_seconds,
            max_retries=1,
            max_tokens=max_output_tokens,
        )
    except Exception:
        raise ReviewModelUnavailableError('review model is unavailable') from None


def _provider_order(settings: Settings) -> tuple[str, ...]:
    supported = {*OPENAI_COMPATIBLE_PROVIDERS, 'gemini'}
    result: list[str] = []
    for raw_provider in settings.agent_llm_provider_order.split(','):
        provider = raw_provider.strip().lower()
        if provider in supported and provider not in result:
            result.append(provider)
    return tuple(result) or ('openai', 'gemini')


def _available_provider_routes(settings: Settings) -> tuple[tuple[str, str], ...]:
    routes: list[tuple[str, str]] = []
    for provider in _provider_order(settings):
        if provider in OPENAI_COMPATIBLE_PROVIDERS and settings.openai_api_key:
            routes.append((provider, settings.agent_llm_openai_model))
        elif provider == 'gemini' and (settings.gemini_api_key or settings.google_api_key):
            routes.append((provider, settings.agent_llm_gemini_model))
    return tuple(routes)


def _route_version(kind: str, routes: tuple[tuple[str, str], ...]) -> str:
    route_names = ','.join(f'{provider}/{model}' for provider, model in routes)
    return f'{MODEL_ROUTE_VERSION}:langchain:{kind}:{route_names}'
