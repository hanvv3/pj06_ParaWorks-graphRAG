from __future__ import annotations

from contextlib import contextmanager

import pytest

from backend.app.agents.rag_orchestrator_agent.v2_input import (
    _prepared,
    prepare_direct_request_text,
)
from backend.tests.test_rag_v2_graph import _context


@pytest.mark.parametrize('mode', ('disabled', 'shadow', 'enforce'))
@pytest.mark.parametrize('stage', ('none', 'ask', 'search', 'assistant'))
@pytest.mark.parametrize('surface', ('ask', 'search', 'assistant'))
def test_facade_owner_matrix_never_constructs_request_services(
    tmp_path, mode, stage, surface
):
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )

    context = _context(tmp_path)
    settings = context.settings.model_copy(
        update={'langgraph_rag_v2_mode': mode, 'langgraph_rag_v2_stage': stage}
    )

    def no_request(**kwargs):
        raise AssertionError('owner selection cannot open request services')

    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=no_request,
        dependencies=RagRuntimeDependencies(request_factory=no_request),
    )
    included = ('none', 'ask', 'search', 'assistant').index(surface) <= (
        'none',
        'ask',
        'search',
        'assistant',
    ).index(stage)
    expected = (
        'v2'
        if included and mode == 'enforce'
        else 'shadow'
        if included and mode == 'shadow'
        else 'legacy'
    )
    assert bundle.facade.execution_owner(surface) == expected


def test_facade_owns_request_factory_lifetime_and_committed_graph_output(tmp_path):
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )

    context = _context(tmp_path)
    exits = []

    @contextmanager
    def request_factory(**kwargs):
        assert kwargs['actor'] == context.actor
        try:
            yield context.services
        finally:
            exits.append('closed')

    bundle = build_rag_v2_runtime(
        settings=context.settings,
        session_factory=lambda: None,
        dependencies=RagRuntimeDependencies(request_factory=request_factory),
    )
    text = prepare_direct_request_text(
        '근거', key=context.settings.agent_runtime_fingerprint_secret.encode()
    )
    result = bundle.facade.invoke_graph(
        actor=context.actor, surface='search', prepared_text=text
    )
    assert result['outcome'] == 'search_projected'
    assert exits == ['closed']
    with pytest.raises((TypeError, ValueError)):
        bundle.facade.invoke_graph(
            actor=context.actor,
            surface='search',
            prepared_text=text,
            backend='pgvector',
        )


def test_assistant_context_1001_terms_refused_before_request_factory(tmp_path):
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )

    context = _context(tmp_path)
    settings = context.settings.model_copy(
        update={'langgraph_rag_v2_stage': 'assistant'}
    )
    query = ' '.join(f'term{index}' for index in range(1001))
    text = _prepared(
        caller_text='현재',
        normalized_current_user_text='현재',
        retrieval_query_text=query,
        answer_question_text='현재',
        query_context_version='assistant-context:v1',
        key=settings.agent_runtime_fingerprint_secret.encode(),
    )

    def no_request(**kwargs):
        raise AssertionError('budget refusal must precede DB/factory/provider work')

    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=no_request,
        dependencies=RagRuntimeDependencies(request_factory=no_request),
    )
    result = bundle.facade.invoke_graph(
        actor=context.actor, surface='assistant', prepared_text=text
    )
    assert result['outcome'] == 'budget_exceeded'
    assert result['charged_cost_usd'] == 0


def test_static_invalid_frame_blocks_composition_before_any_request_work(
    tmp_path, monkeypatch
):
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.app.agents.rag_orchestrator_agent import v2_answer_schema

    context = _context(tmp_path)
    monkeypatch.setitem(
        v2_answer_schema.ANSWER_PROMPT_RENDERER_STATIC, 'injected', '\x00'
    )
    with pytest.raises(Exception, match='runtime_version_unavailable'):
        build_rag_v2_runtime(
            settings=context.settings,
            session_factory=lambda: None,
            dependencies=RagRuntimeDependencies(request_factory=lambda **kwargs: None),
        )


@pytest.mark.parametrize('kind', ('unsafe_context', 'missing_target'))
def test_assistant_invalid_context_or_missing_persistence_target_precedes_factory(
    tmp_path, kind
):
    from backend.app.agent_runtime.rag_finalization import AssistantProjectionTarget
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )

    context = _context(tmp_path)
    settings = context.settings.model_copy(
        update={'langgraph_rag_v2_stage': 'assistant'}
    )
    query = (
        'Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456'
        if kind == 'unsafe_context'
        else 'current'
    )
    text = _prepared(
        caller_text='current',
        normalized_current_user_text='current',
        retrieval_query_text=query,
        answer_question_text='current',
        query_context_version='assistant-context:v1',
        key=settings.agent_runtime_fingerprint_secret.encode(),
    )

    def no_request(**kwargs):
        raise AssertionError('invalid ingress cannot open request dependencies')

    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=no_request,
        dependencies=RagRuntimeDependencies(request_factory=no_request),
    )
    with pytest.raises(ValueError):
        bundle.facade.invoke_graph(
            actor=context.actor,
            surface='assistant',
            prepared_text=text,
            assistant_target=AssistantProjectionTarget(1, 2, context.actor.id)
            if kind == 'unsafe_context'
            else None,
        )
