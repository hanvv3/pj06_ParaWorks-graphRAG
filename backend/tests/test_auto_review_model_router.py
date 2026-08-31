from decimal import Decimal

import pytest

from backend.app.agent_runtime.auto_review_cost_policy import (
    _SERVER_OWNED_FENCED_SEND_HOOK,
)
from backend.app.agent_runtime.model_router import (
    ReviewModelUnavailableError,
    build_auto_review_validator_model_route,
    build_rag_answer_model_config_snapshot_hmac,
    rag_answer_model_config_snapshot,
)
from backend.app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    values = {
        'openai_api_key': 'test-openai-key-not-live',
        'auto_review_validator_input_cost_per_1m_tokens': Decimal('2.000000'),
        'auto_review_validator_output_cost_per_1m_tokens': Decimal('12.000000'),
    }
    values.update(overrides)
    return Settings(
        _env_file=None,
        **values,
    )


def test_live_route_constructs_chat_openai_terra_medium_without_fallback() -> None:
    calls: list[dict[str, object]] = []
    model = object()

    def builder(**kwargs):
        calls.append(kwargs)
        return model

    route = build_auto_review_validator_model_route(
        _settings(
            agent_llm_provider_order='gemini,azure_openai,openai',
            gemini_api_key='must-not-be-used',
        ),
        timeout_seconds=60,
        http_hook=_SERVER_OWNED_FENCED_SEND_HOOK,
        chat_model_builder=builder,
    )

    assert route.model is model
    assert route.model_name == 'gpt-5.6-terra'
    assert route.route_version == 'auto-review-validation:v2'
    assert route.deterministic is False
    assert calls == [
        {
            'model': 'gpt-5.6-terra',
            'api_key': 'test-openai-key-not-live',
            'reasoning_effort': 'medium',
            'use_responses_api': True,
            'timeout': 60,
            'max_retries': 0,
            'max_completion_tokens': 3072,
            'verbose': False,
            'cache': False,
        }
    ]


def test_validator_route_requires_key_price_timeout_and_server_hook() -> None:
    invalid_settings = (
        _settings(openai_api_key=None),
        _settings(auto_review_validator_input_cost_per_1m_tokens=Decimal('2.1')),
    )
    for settings in invalid_settings:
        with pytest.raises(ReviewModelUnavailableError, match='unavailable'):
            build_auto_review_validator_model_route(
                settings,
                timeout_seconds=60,
                http_hook=_SERVER_OWNED_FENCED_SEND_HOOK,
                chat_model_builder=lambda **kwargs: kwargs,
            )

    for timeout, hook in ((0, _SERVER_OWNED_FENCED_SEND_HOOK), (60, object())):
        with pytest.raises(ReviewModelUnavailableError, match='unavailable'):
            build_auto_review_validator_model_route(
                _settings(),
                timeout_seconds=timeout,
                http_hook=hook,
                chat_model_builder=lambda **kwargs: kwargs,
            )


def test_server_owned_http_hook_is_body_blind_and_has_no_mutable_state() -> None:
    hook = _SERVER_OWNED_FENCED_SEND_HOOK

    assert not hasattr(hook, '__dict__')
    assert hook.__slots__ == ()
    assert not any(
        name in dir(hook)
        for name in ('request', 'response', 'body', 'content', 'headers')
    )

def test_route_sanitizes_constructor_failure() -> None:
    def fail(**kwargs):
        del kwargs
        raise RuntimeError('raw provider constructor details')

    with pytest.raises(ReviewModelUnavailableError) as exc_info:
        build_auto_review_validator_model_route(
            _settings(),
            timeout_seconds=60,
            http_hook=_SERVER_OWNED_FENCED_SEND_HOOK,
            chat_model_builder=fail,
        )

    assert str(exc_info.value) == 'review model is unavailable'
    assert exc_info.value.__cause__ is None


def test_rag_answer_model_config_snapshot_is_exact_direct_standard_identity() -> None:
    snapshot = rag_answer_model_config_snapshot(
        output_schema_hmac='a' * 64,
        prompt_renderer_hmac='b' * 64,
    )

    assert snapshot == {
        'api_base_url': 'https://api.openai.com/v1',
        'answer_block_joiner_version': 'rag-answer-block-joiner:v1',
        'cache_enabled': False,
        'callbacks_enabled': False,
        'endpoint_identity': 'openai-direct-standard-global:v1',
        'max_output_tokens': 512,
        'max_provider_attempts': 1,
        'max_retries': 0,
        'model': 'gpt-5.4-mini-2026-03-17',
        'output_schema_hmac': 'a' * 64,
        'output_schema_name': 'rag_answer_blocks_v1',
        'prompt_renderer_hmac': 'b' * 64,
        'prompt_renderer_version': 'rag-answer-renderer:v1',
        'provider': 'openai',
        'provider_fallback': 'none',
        'provider_send_start_window_seconds': 5,
        'reasoning_effort': 'none',
        'regional_processing': False,
        'seed_state': 'omitted',
        'service_tier': 'default',
        'store': False,
        'streaming': False,
        'structured_output_identity': 'langchain-json-schema-strict-include-raw:v1',
        'temperature_state': 'omitted',
        'timeout_seconds': 30,
        'tool_binding': 'none',
        'top_p_state': 'omitted',
        'tracing_enabled': False,
        'use_responses_api': True,
    }
    first = build_rag_answer_model_config_snapshot_hmac(
        _settings(),
        output_schema_hmac='a' * 64,
        prompt_renderer_hmac='b' * 64,
    )
    second = build_rag_answer_model_config_snapshot_hmac(
        _settings(),
        output_schema_hmac='c' * 64,
        prompt_renderer_hmac='b' * 64,
    )
    assert first != second
