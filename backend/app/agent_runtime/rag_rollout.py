from __future__ import annotations

# Deployment-owned, cumulative RAG cutover; no request supplies this policy.
from dataclasses import dataclass
from typing import Literal

from backend.app.agent_runtime.rag_v2_contracts import (
    RagCutoverStage,
    RagMode,
    RagSurface,
)


@dataclass(frozen=True, slots=True)
class RagRouteDecision:
    public_owner: Literal['legacy', 'v2']
    shadow_retrieval: bool
    cutover_surface: bool


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
