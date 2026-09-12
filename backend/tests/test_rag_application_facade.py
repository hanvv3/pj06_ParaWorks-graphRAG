from __future__ import annotations

from contextlib import contextmanager

import pytest

from backend.app.agents.rag_orchestrator_agent.v2_input import (
    _prepared,
    prepare_direct_request_text,
)
from backend.tests.test_rag_v2_graph import _context


def test_keyword_shadow_returns_exact_legacy_delivery_and_runs_observer_once(
    tmp_path, monkeypatch,
):
    """Catches shadow replacing public bytes/run or invoking V2 generation."""
    from backend.app.agent_runtime.rag_application import DirectRagDeliveryResult
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.app.schemas.rag import SearchV1Projection

    context = _context(tmp_path)
    settings = context.settings.model_copy(update={
        'langgraph_rag_v2_mode': 'shadow',
        'langgraph_rag_v2_stage': 'search',
        'rag_retrieval_backend': 'keyword',
    })
    legacy = DirectRagDeliveryResult(
        200,
        'success',
        'search_projected',
        SearchV1Projection(
            retrieval_backend='deterministic_lexical',
            cost_policy={
                'embedding_query_call': False,
                'paid_llm_call': False,
                'requires_pgvector_flag': True,
            },
            hidden_match_count=0,
            permission_notice=None,
            results=[],
        ),
        None,
    )
    calls = []

    def shadow_runner(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('comparison failure must not replace legacy')

    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=lambda: None,
        dependencies=RagRuntimeDependencies(
            request_factory=lambda **kwargs: None,
            shadow_runner=shadow_runner,
        ),
    )
    monkeypatch.setattr(type(bundle.facade), '_invoke_legacy', lambda *args, **kwargs: legacy)

    delivered = bundle.facade.invoke_search(actor=context.actor, caller_text='exact bytes')

    assert delivered is legacy
    assert len(calls) == 1
    assert calls[0]['surface'] == 'search'
    assert calls[0]['legacy_delivery'] is legacy
    assert calls[0]['prepared_text'].caller_text == 'exact bytes'


@pytest.mark.parametrize(
    ('mode', 'stage', 'calls'),
    (('disabled', 'assistant', 0), ('shadow', 'none', 0), ('enforce', 'search', 0)),
)
def test_shadow_observer_has_no_disabled_noncutover_or_enforce_background_work(
    tmp_path, monkeypatch, mode, stage, calls,
):
    """Catches a shadow callback surviving rollback or enforce cutover."""
    from backend.app.agent_runtime.rag_application import DirectRagDeliveryResult
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.app.schemas.rag import SearchV1Projection

    context = _context(tmp_path)
    settings = context.settings.model_copy(update={
        'langgraph_rag_v2_mode': mode,
        'langgraph_rag_v2_stage': stage,
        'rag_retrieval_backend': 'keyword',
    })
    legacy = DirectRagDeliveryResult(
        200, 'success', 'search_projected', SearchV1Projection(
            retrieval_backend='deterministic_lexical',
            cost_policy={'embedding_query_call': False, 'paid_llm_call': False,
                         'requires_pgvector_flag': True},
            hidden_match_count=0, permission_notice=None, results=[],
        ), None,
    )
    observed = []
    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=lambda: None,
        dependencies=RagRuntimeDependencies(
            request_factory=lambda **kwargs: None,
            shadow_runner=lambda **kwargs: observed.append(kwargs),
        ),
    )
    monkeypatch.setattr(type(bundle.facade), '_invoke_legacy', lambda *args, **kwargs: legacy)
    if mode == 'enforce' and stage == 'search':
        monkeypatch.setattr(type(bundle.facade), 'invoke_graph', lambda *args, **kwargs: {
            'outcome': 'unexpected_internal_error',
        })

    bundle.facade.invoke_search(actor=context.actor, caller_text='rollback')
    assert len(observed) == calls


def test_pgvector_shadow_runner_owns_single_shared_embedding_before_legacy(
    tmp_path, monkeypatch,
):
    """Catches legacy dispatching first or the internal run leaking into public delivery."""
    from backend.app.agent_runtime.rag_application import DirectRagDeliveryResult
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.app.schemas.rag import SearchV1Projection

    context = _context(tmp_path)
    settings = context.settings.model_copy(update={
        'langgraph_rag_v2_mode': 'shadow',
        'langgraph_rag_v2_stage': 'search',
        'rag_retrieval_backend': 'pgvector',
    })
    legacy = DirectRagDeliveryResult(
        200, 'success', 'search_projected', SearchV1Projection(
            retrieval_backend='pgvector',
            cost_policy={'embedding_query_call': True, 'paid_llm_call': False,
                         'requires_pgvector_flag': True},
            hidden_match_count=0, permission_notice=None, results=[],
        ), None,
    )
    shared = object()
    calls = []

    def shadow_runner(**kwargs):
        calls.append(kwargs)
        return kwargs['legacy_invoke'](shared)

    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=lambda: None,
        dependencies=RagRuntimeDependencies(
            request_factory=lambda **kwargs: None,
            shadow_runner=shadow_runner,
        ),
    )

    def legacy_invoke(_facade, **kwargs):
        assert kwargs['query_embedding_result'] is shared
        return legacy

    monkeypatch.setattr(type(bundle.facade), '_invoke_legacy', legacy_invoke)

    delivered = bundle.facade.invoke_search(actor=context.actor, caller_text='same')

    assert delivered is legacy
    assert len(calls) == 1
    assert 'legacy_delivery' not in calls[0]
    assert callable(calls[0]['legacy_invoke'])


def test_pgvector_shadow_embedding_safety_failure_overrides_legacy_with_typed_503(
    tmp_path, monkeypatch,
):
    """Catches invalid paid output reaching legacy or being hidden by parity mode."""
    from backend.app.agent_runtime.rag_v2_composition import (
        RagRuntimeDependencies,
        build_rag_v2_runtime,
    )
    from backend.app.rag.shadow import RagShadowProviderOverrideError

    context = _context(tmp_path)
    settings = context.settings.model_copy(update={
        'langgraph_rag_v2_mode': 'shadow',
        'langgraph_rag_v2_stage': 'search',
        'rag_retrieval_backend': 'pgvector',
    })

    def shadow_runner(**_kwargs):
        raise RagShadowProviderOverrideError(
            'provider_usage_overrun', component='query_embedding'
        )

    bundle = build_rag_v2_runtime(
        settings=settings,
        session_factory=lambda: None,
        dependencies=RagRuntimeDependencies(
            request_factory=lambda **kwargs: None,
            shadow_runner=shadow_runner,
        ),
    )
    monkeypatch.setattr(
        type(bundle.facade),
        '_invoke_legacy',
        lambda *args, **kwargs: pytest.fail('invalid carrier must not reach legacy'),
    )

    delivered = bundle.facade.invoke_search(actor=context.actor, caller_text='same')

    assert delivered.public_status == 503
    assert delivered.application_outcome == 'provider_usage_overrun'
    assert delivered.error is not None
    assert delivered.error.code == 'provider_usage_overrun'


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
