from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from backend.app.agent_runtime.auto_review_cost_policy import (
    AUTO_REVIEW_MAX_OUTPUT_TOKENS,
    AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
    AUTO_REVIEW_REASONING_EFFORT,
    AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
    AUTO_REVIEW_VALIDATOR_MODEL,
    AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
    is_server_owned_fenced_send_hook,
)
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agents.mail_document_agent import (
    DeterministicMailDocumentAgentModel,
    MailDocumentLlmProviderError,
    MailDocumentLlmSettings,
    build_langchain_mail_document_agent_model,
)
from backend.app.core.config import Settings
from backend.app.rag.retrieval import is_lower_hex_64
from backend.app.schemas.auto_review import AUTO_REVIEW_VALIDATOR_PROMPT_VERSION

MODEL_ROUTE_VERSION = 'review-model-route:v1'
OPENAI_COMPATIBLE_PROVIDERS = frozenset({'openai', 'azure_openai'})
RAG_ANSWER_MODEL = 'gpt-5.4-mini-2026-03-17'
RAG_ANSWER_MODEL_CONFIG_VERSION = 'rag-answer-model-config:v1'
RAG_OPENAI_API_BASE_URL = 'https://api.openai.com/v1'
RAG_ANSWER_ENDPOINT_IDENTITY = 'openai-direct-standard-global:v1'
RAG_ANSWER_SERVICE_TIER = 'default'


class ReviewModelUnavailableError(RuntimeError):
    code = 'model_unavailable'


@dataclass(frozen=True)
class RoutedReviewModel:
    model: Any
    model_name: str
    route_version: str
    deterministic: bool


@dataclass(frozen=True, slots=True)
class RoutedRagAnswerModel:
    model: Any
    provider: Literal['openai']
    model_name: Literal['gpt-5.4-mini-2026-03-17']
    model_config_snapshot_hmac: str


def build_rag_answer_model_route(*, settings: Settings) -> RoutedRagAnswerModel:
    from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
        ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT,
        assert_answer_contract_registry_ready,
        build_answer_output_schema_hmac,
        build_answer_prompt_renderer_hmac,
    )

    if not settings.openai_api_key:
        raise ReviewModelUnavailableError('review model is unavailable')
    try:
        assert_answer_contract_registry_ready()
        output_schema_hmac = build_answer_output_schema_hmac(settings)
        renderer_hmac = build_answer_prompt_renderer_hmac(settings)
        model_config_hmac = build_rag_answer_model_config_snapshot_hmac(
            settings,
            output_schema_hmac=output_schema_hmac,
            prompt_renderer_hmac=renderer_hmac,
        )
        import httpx
        from langchain_openai import ChatOpenAI

        raw_model = ChatOpenAI(
            model=RAG_ANSWER_MODEL,
            api_key=settings.openai_api_key,
            base_url=RAG_OPENAI_API_BASE_URL,
            reasoning_effort='none',
            use_responses_api=True,
            service_tier=RAG_ANSWER_SERVICE_TIER,
            timeout=30,
            max_retries=0,
            max_completion_tokens=512,
            store=False,
            streaming=False,
            verbose=False,
            cache=False,
            callbacks=[],
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )
        model = raw_model.with_structured_output(
            ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT,
            method='json_schema',
            strict=True,
            include_raw=True,
        )
    except Exception:
        raise ReviewModelUnavailableError('review model is unavailable') from None
    return RoutedRagAnswerModel(
        model=model,
        provider='openai',
        model_name=RAG_ANSWER_MODEL,
        model_config_snapshot_hmac=model_config_hmac,
    )


def rag_answer_model_config_snapshot(
    *,
    output_schema_hmac: str,
    prompt_renderer_hmac: str,
) -> dict[str, object]:
    if not is_lower_hex_64(output_schema_hmac) or not is_lower_hex_64(
        prompt_renderer_hmac
    ):
        raise ValueError('RAG answer model identity is invalid')
    return {
        'api_base_url': RAG_OPENAI_API_BASE_URL,
        'answer_block_joiner_version': 'rag-answer-block-joiner:v1',
        'cache_enabled': False,
        'callbacks_enabled': False,
        'endpoint_identity': RAG_ANSWER_ENDPOINT_IDENTITY,
        'max_output_tokens': 512,
        'max_provider_attempts': 1,
        'max_retries': 0,
        'model': RAG_ANSWER_MODEL,
        'output_schema_hmac': output_schema_hmac,
        'output_schema_name': 'rag_answer_blocks_v1',
        'prompt_renderer_hmac': prompt_renderer_hmac,
        'prompt_renderer_version': 'rag-answer-renderer:v1',
        'provider': 'openai',
        'provider_fallback': 'none',
        'provider_send_start_window_seconds': 5,
        'reasoning_effort': 'none',
        'regional_processing': False,
        'seed_state': 'omitted',
        'service_tier': RAG_ANSWER_SERVICE_TIER,
        'store': False,
        'streaming': False,
        'structured_output_identity': (
            'langchain-json-schema-strict-include-raw:v1'
        ),
        'temperature_state': 'omitted',
        'timeout_seconds': 30,
        'tool_binding': 'none',
        'top_p_state': 'omitted',
        'tracing_enabled': False,
        'use_responses_api': True,
    }


def build_rag_answer_model_config_snapshot_hmac(
    settings: Settings,
    *,
    output_schema_hmac: str,
    prompt_renderer_hmac: str,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        rag_answer_model_config_snapshot(
            output_schema_hmac=output_schema_hmac,
            prompt_renderer_hmac=prompt_renderer_hmac,
        ),
        secret=secret,
        schema_version='rag-answer-model-config-snapshot:v1',
        policy_version=RAG_ANSWER_MODEL_CONFIG_VERSION,
    )


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


def build_auto_review_validator_model_route(
    settings: Settings,
    *,
    timeout_seconds: int,
    http_hook: object,
    chat_model_builder: Callable[..., Any] | None = None,
) -> RoutedReviewModel:
    """Build the one exact Terra route after a committed attempt is admitted."""
    if (
        not settings.openai_api_key
        or timeout_seconds != AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS
        or settings.auto_review_validator_input_cost_per_1m_tokens
        != AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M
        or settings.auto_review_validator_output_cost_per_1m_tokens
        != AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M
        or not is_server_owned_fenced_send_hook(http_hook)
    ):
        raise ReviewModelUnavailableError('review model is unavailable')
    try:
        if chat_model_builder is None:
            from langchain_openai import ChatOpenAI

            chat_model_builder = ChatOpenAI
        model = chat_model_builder(
            model=AUTO_REVIEW_VALIDATOR_MODEL,
            api_key=settings.openai_api_key,
            reasoning_effort=AUTO_REVIEW_REASONING_EFFORT,
            use_responses_api=True,
            timeout=timeout_seconds,
            max_retries=0,
            max_completion_tokens=AUTO_REVIEW_MAX_OUTPUT_TOKENS,
            verbose=False,
            cache=False,
        )
        if bool(getattr(model, 'verbose', False)) or bool(
            getattr(model, 'cache', False)
        ):
            raise ValueError('unsafe model controls')
    except Exception:
        raise ReviewModelUnavailableError('review model is unavailable') from None
    return RoutedReviewModel(
        model=model,
        model_name=AUTO_REVIEW_VALIDATOR_MODEL,
        route_version=AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
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
