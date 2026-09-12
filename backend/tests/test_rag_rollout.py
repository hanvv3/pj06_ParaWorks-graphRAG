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
