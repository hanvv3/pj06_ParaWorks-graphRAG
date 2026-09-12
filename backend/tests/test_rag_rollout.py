from __future__ import annotations

import importlib
from decimal import Decimal

import pytest
from sqlalchemy import create_engine

from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyService,
)
from backend.app.db.base import Base
from backend.tests.test_rag_v2_costs import _snapshot


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


@pytest.mark.parametrize('blocker', ('blocked_overrun', 'blocked_remediation'))
@pytest.mark.parametrize('mode', ('disabled', 'shadow', 'enforce'))
@pytest.mark.parametrize('stage', ('none', 'ask', 'search', 'assistant'))
@pytest.mark.parametrize('surface', ('ask', 'search', 'assistant'))
def test_retained_query_blocker_matrix_preserves_legacy_or_refuses_d_work(
    tmp_path, blocker, mode, stage, surface,
):
    """Exercise every rollout cell against the real retained refusal boundary."""
    from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy

    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    path = tmp_path / f'{blocker}-{mode}-{stage}-{surface}.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'retained-blocker-task21-key',
        designated_environment_id='test',
    )
    snapshots = (_snapshot('query_embedding'), _snapshot('answer_generation'))
    with engine.connect() as connection:
        service.bootstrap(
            connection,
            snapshots,
            reviewed_transition_reference_hmac='9' * 64,
        )
        if blocker == 'blocked_overrun':
            service.block_overrun(
                connection,
                'query_embedding',
                agent_run_id=21,
                input_tokens=20,
                output_tokens=0,
                cost_usd=Decimal('0.000011'),
            )
        else:
            service.block_remediation(
                connection,
                'query_embedding',
                category='provider_safety_unavailable',
                agent_run_id=21,
                input_tokens=20,
                output_tokens=0,
                cost_usd=Decimal('0.000010'),
            )

        plan = RagRolloutPolicy().plan(
            mode=mode,
            stage=stage,
            surface=surface,
            backend='pgvector',
        )
        legacy_delivery = b'exact legacy public bytes'
        public_delivery = legacy_delivery
        safety_reads = 0
        advancement_allowed = None
        public_http = None
        if plan.v2_retrieval_count:
            safety_reads += 1
            with pytest.raises(RagProviderSafetyError, match='blocked'):
                service.require_ready(
                    connection,
                    'query_embedding',
                    snapshots[0],
                )
            advancement_allowed = False
            if plan.public_owner == 'v2':
                public_delivery = None
                if surface == 'assistant':
                    from fastapi import HTTPException
                    from fastapi.responses import JSONResponse

                    from backend.app.api.v1.assistant import _map_assistant_delivery
                    from backend.app.assistant.delivery import AssistantDeliveryResult

                    failed = AssistantDeliveryResult(
                        'provider_safety_unavailable', 1, 'generation_error',
                        'committed_safe_failure', 1, 502,
                    )
                    with pytest.raises(HTTPException) as caught:
                        _map_assistant_delivery(failed)
                    response = JSONResponse(
                        status_code=caught.value.status_code,
                        content={'detail': caught.value.detail},
                    )
                else:
                    from backend.app.agent_runtime.rag_application import (
                        direct_rag_error,
                    )
                    from backend.app.api.v1.rag_delivery import deliver_direct_rag

                    response = deliver_direct_rag(
                        direct_rag_error('provider_safety_unavailable')
                    )
                public_http = (response.status_code, bytes(response.body))

        rank = {'none': 0, 'ask': 1, 'search': 2, 'assistant': 3}
        active = mode in {'shadow', 'enforce'} and rank[stage] >= rank[surface]
        assert safety_reads == int(active)
        assert plan.v2_embedding_count == int(active)
        assert public_delivery is (
            None if mode == 'enforce' and active else legacy_delivery
        )
        assert advancement_allowed is (False if active else None)
        assert public_http == (
            (502, b'{"detail":"assistant answer generation failed"}')
            if mode == 'enforce' and active and surface == 'assistant'
            else (503, b'{"detail":{"code":"provider_safety_unavailable"}}')
            if mode == 'enforce' and active
            else None
        )

        restarted = RagProviderSafetyService(
            latch_path=path,
            identity_secret=b'retained-blocker-task21-key',
            designated_environment_id='test',
        )
        with pytest.raises(RagProviderSafetyError, match='blocked'):
            restarted.require_ready(
                connection,
                'query_embedding',
                snapshots[0],
            )


def test_rollout_rollback_never_clears_retained_query_blocker(tmp_path):
    """Deployment-static rollback changes routing, never safety state."""
    from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy

    engine = create_engine('sqlite+pysqlite:///:memory:')
    Base.metadata.create_all(engine)
    path = tmp_path / 'rollback-retained.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'retained-blocker-task21-key',
        designated_environment_id='test',
    )
    snapshot = _snapshot('query_embedding')
    with engine.connect() as connection:
        service.bootstrap(
            connection,
            (snapshot, _snapshot('answer_generation')),
            reviewed_transition_reference_hmac='9' * 64,
        )
        service.block_overrun(
            connection,
            'query_embedding',
            agent_run_id=22,
            input_tokens=20,
            output_tokens=0,
            cost_usd=Decimal('0.000011'),
        )
        for mode, stage, expected_work in (
            ('enforce', 'assistant', 1),
            ('disabled', 'none', 0),
            ('shadow', 'assistant', 1),
        ):
            plan = RagRolloutPolicy().plan(
                mode=mode,
                stage=stage,
                surface='search',
                backend='pgvector',
            )
            assert plan.v2_retrieval_count == expected_work
            if expected_work:
                with pytest.raises(RagProviderSafetyError, match='blocked'):
                    service.require_ready(connection, 'query_embedding', snapshot)

        restarted = RagProviderSafetyService(
            latch_path=path,
            identity_secret=b'retained-blocker-task21-key',
            designated_environment_id='test',
        )
        with pytest.raises(RagProviderSafetyError, match='blocked'):
            restarted.require_ready(connection, 'query_embedding', snapshot)
