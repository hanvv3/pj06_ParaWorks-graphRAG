from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize('mode', ('disabled', 'shadow', 'enforce'))
@pytest.mark.parametrize(('stage', 'surfaces'), (
    ('none', ()), ('ask', ('ask',)), ('search', ('ask', 'search')),
    ('assistant', ('ask', 'search', 'assistant')),
))
@pytest.mark.parametrize('surface', ('ask', 'search', 'assistant'))
def test_cumulative_rollout_preserves_legacy_and_never_shadows_enforce(
    mode, stage, surfaces, surface,
):
    module = importlib.util.find_spec('backend.app.agent_runtime.rag_rollout')
    assert module is not None, 'Task 14 rollout policy is missing'
    policy = importlib.import_module(module.name).RagRolloutPolicy()
    result = policy.decide(mode=mode, stage=stage, surface=surface)
    included = surface in surfaces
    assert result.public_owner == ('v2' if mode == 'enforce' and included else 'legacy')
    assert result.shadow_retrieval is (mode == 'shadow' and included)
    assert result.cutover_surface is included


@pytest.mark.parametrize('field', ('mode', 'stage', 'surface'))
def test_unknown_rollout_values_fail_closed(field):
    module = importlib.util.find_spec('backend.app.agent_runtime.rag_rollout')
    assert module is not None, 'Task 14 rollout policy is missing'
    policy = importlib.import_module(module.name).RagRolloutPolicy()
    values = {'mode': 'enforce', 'stage': 'assistant', 'surface': 'ask'}
    values[field] = 'latest'
    with pytest.raises(ValueError):
        policy.decide(**values)


@pytest.mark.parametrize('mode', ('disabled', 'shadow', 'enforce'))
@pytest.mark.parametrize('stage', ('none', 'ask', 'search', 'assistant'))
@pytest.mark.parametrize('surface', ('ask', 'search', 'assistant'))
@pytest.mark.parametrize('backend', ('keyword', 'pgvector'))
def test_execution_plan_exhaustively_owns_only_authorized_v2_work(
    mode, stage, surface, backend,
):
    """Catches accidental background shadow/provider work after rollback or enforce."""
    from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy

    plan = RagRolloutPolicy().plan(
        mode=mode, stage=stage, surface=surface, backend=backend,
    )
    rank = {'none': 0, 'ask': 1, 'search': 2, 'assistant': 3}
    included = rank[stage] >= rank[surface]
    shadow = mode == 'shadow' and included
    enforce = mode == 'enforce' and included

    assert plan.public_owner == ('v2' if enforce else 'legacy')
    assert plan.v2_retrieval_count == int(shadow or enforce)
    assert plan.v2_generation_count == int(enforce and surface != 'search')
    assert plan.v2_embedding_count == int((shadow or enforce) and backend == 'pgvector')
    assert plan.agent_run_owner == (
        'v2_public' if enforce else 'shadow_internal'
        if shadow and backend == 'pgvector' else 'none'
    )
    assert plan.audit_owner == ('v2_public' if enforce else 'shadow_aggregate' if shadow else 'none')
    assert plan.shadow_retrieval is shadow
    assert plan.background_shadow is False


def test_execution_plan_rejects_request_owned_override_shape():
    """Catches adding a request/header/query override to deployment-static routing."""
    from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy

    with pytest.raises(TypeError):
        RagRolloutPolicy().plan(
            mode='shadow', stage='ask', surface='ask', backend='keyword',
            request_mode='enforce',
        )
