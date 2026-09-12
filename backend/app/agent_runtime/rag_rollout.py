from __future__ import annotations

# Deployment-owned, cumulative RAG cutover; no request supplies this policy.
from dataclasses import dataclass
from typing import Literal

from backend.app.agent_runtime.rag_v2_contracts import (
    RagCutoverStage,
    RagMode,
    RagRetrievalBackend,
    RagSurface,
)


@dataclass(frozen=True, slots=True)
class RagRouteDecision:
    public_owner: Literal['legacy', 'v2']
    shadow_retrieval: bool
    cutover_surface: bool


@dataclass(frozen=True, slots=True)
class RagExecutionPlan:
    """Complete deployment-owned execution authority for one surface."""

    public_owner: Literal['legacy', 'v2']
    shadow_retrieval: bool
    cutover_surface: bool
    v2_retrieval_count: Literal[0, 1]
    v2_generation_count: Literal[0, 1]
    v2_embedding_count: Literal[0, 1]
    agent_run_owner: Literal['none', 'v2_public', 'shadow_internal']
    audit_owner: Literal['none', 'v2_public', 'shadow_aggregate']
    background_shadow: Literal[False] = False


@dataclass(frozen=True, slots=True)
class RagRolloutPolicy:
    def decide(
        self, *, mode: RagMode, stage: RagCutoverStage, surface: RagSurface,
    ) -> RagRouteDecision:
        stages = ('none', 'ask', 'search', 'assistant')
        if mode not in ('disabled', 'shadow', 'enforce'):
            raise ValueError('unknown RAG rollout mode')
        if stage not in stages or surface not in stages[1:]:
            raise ValueError('unknown RAG rollout stage or surface')
        included = stages.index(surface) <= stages.index(stage)
        return RagRouteDecision(
            public_owner='v2' if included and mode == 'enforce' else 'legacy',
            shadow_retrieval=included and mode == 'shadow',
            cutover_surface=included,
        )

    def plan(
        self,
        *,
        mode: RagMode,
        stage: RagCutoverStage,
        surface: RagSurface,
        backend: RagRetrievalBackend,
    ) -> RagExecutionPlan:
        if backend not in {'keyword', 'pgvector'}:
            raise ValueError('unknown RAG retrieval backend')
        route = self.decide(mode=mode, stage=stage, surface=surface)
        shadow = route.shadow_retrieval
        enforce = route.public_owner == 'v2'
        has_v2_work = shadow or enforce
        return RagExecutionPlan(
            public_owner=route.public_owner,
            shadow_retrieval=shadow,
            cutover_surface=route.cutover_surface,
            v2_retrieval_count=int(has_v2_work),  # type: ignore[arg-type]
            v2_generation_count=int(enforce and surface != 'search'),  # type: ignore[arg-type]
            v2_embedding_count=int(has_v2_work and backend == 'pgvector'),  # type: ignore[arg-type]
            agent_run_owner=(
                'v2_public'
                if enforce
                else 'shadow_internal'
                if shadow and backend == 'pgvector'
                else 'none'
            ),
            audit_owner=(
                'v2_public'
                if enforce
                else 'shadow_aggregate'
                if shadow
                else 'none'
            ),
        )
