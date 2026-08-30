from decimal import Decimal

import pytest

from backend.app.agent_runtime.auto_review_cost_policy import (
    _SERVER_OWNED_FENCED_SEND_HOOK,
)
from backend.app.agent_runtime.model_router import (
    ReviewModelUnavailableError,
    build_auto_review_validator_model_route,
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
