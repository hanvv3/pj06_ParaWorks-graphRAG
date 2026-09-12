import copy
import sys
from types import SimpleNamespace

import httpx
import pytest

from backend.app.agent_runtime.model_router import build_rag_answer_model_route
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT,
    ANSWER_PROMPT_RENDERER_STATIC,
)
from backend.app.core.config import Settings


def test_rag_answer_route_binds_exact_openai_model_and_strict_schema(monkeypatch) -> None:
    constructor_calls: list[dict[str, object]] = []
    structured_calls: list[tuple[object, dict[str, object]]] = []
    bound_model = object()

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)

        def with_structured_output(self, schema: object, **kwargs: object) -> object:
            structured_calls.append((schema, kwargs))
            return bound_model

    monkeypatch.setitem(
        sys.modules,
        'langchain_openai',
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    settings = Settings(_env_file=None, openai_api_key='test-key-not-live')

    route = build_rag_answer_model_route(settings=settings)

    assert route.model is bound_model
    assert route.provider == 'openai'
    assert route.model_name == 'gpt-5.4-mini-2026-03-17'
    assert len(route.model_config_snapshot_hmac) == 64
    sync_client = constructor_calls[0].pop('http_client')
    async_client = constructor_calls[0].pop('http_async_client')
    assert type(sync_client) is httpx.Client
    assert type(async_client) is httpx.AsyncClient
    assert sync_client._trust_env is False
    assert async_client._trust_env is False
    assert constructor_calls == [
        {
            'model': 'gpt-5.4-mini-2026-03-17',
            'api_key': 'test-key-not-live',
            'base_url': 'https://api.openai.com/v1',
            'reasoning_effort': 'none',
            'use_responses_api': True,
            'service_tier': 'default',
            'timeout': 30,
            'max_retries': 0,
            'max_completion_tokens': 512,
            'store': False,
            'streaming': False,
            'verbose': False,
            'cache': False,
            'callbacks': [],
        }
    ]
    assert structured_calls == [
        (
            ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT,
            {'method': 'json_schema', 'strict': True, 'include_raw': True},
        )
    ]


def test_route_fails_closed_on_renderer_registry_drift(monkeypatch) -> None:
    original = copy.deepcopy(ANSWER_PROMPT_RENDERER_STATIC)
    monkeypatch.setitem(ANSWER_PROMPT_RENDERER_STATIC, 'renderer_version', 'drift')
    settings = Settings(_env_file=None, openai_api_key='test-key-not-live')

    from backend.app.agent_runtime.model_router import ReviewModelUnavailableError

    with pytest.raises(ReviewModelUnavailableError, match='unavailable'):
        build_rag_answer_model_route(settings=settings)

    ANSWER_PROMPT_RENDERER_STATIC.clear()
    ANSWER_PROMPT_RENDERER_STATIC.update(original)


@pytest.mark.parametrize('failure', (None, 'constructor', 'schema'))
@pytest.mark.parametrize('async_host', (False, True))
def test_answer_route_closes_only_owned_http_clients_even_on_build_failure(monkeypatch, failure, async_host):
    import asyncio

    from backend.app.agent_runtime.model_router import ReviewModelUnavailableError
    events = []
    class SyncClient:
        def __init__(self, **kwargs):
            pass
        def close(self):
            events.append('sync_closed')
    class AsyncClient:
        def __init__(self, **kwargs):
            pass
        async def aclose(self):
            events.append('async_closed')
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            if failure == 'constructor':
                raise RuntimeError('private constructor error')
        def with_structured_output(self, *args, **kwargs):
            if failure == 'schema':
                raise RuntimeError('private schema error')
            return object()
    monkeypatch.setattr(httpx, 'Client', SyncClient)
    monkeypatch.setattr(httpx, 'AsyncClient', AsyncClient)
    monkeypatch.setitem(sys.modules, 'langchain_openai', SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    settings = Settings(_env_file=None, openai_api_key='test-key-not-live')
    if failure is not None:
        with pytest.raises(ReviewModelUnavailableError):
            build_rag_answer_model_route(settings=settings)
    else:
        route = build_rag_answer_model_route(settings=settings)
        assert events == []
        def close_twice():
            route.close_owned_clients()
            route.close_owned_clients()
        if async_host:
            async def run():
                close_twice()
            asyncio.run(run())
        else:
            close_twice()
    assert events == ['sync_closed', 'async_closed']
