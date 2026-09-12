from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Literal

from langsmith import tracing_context
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.rag_finalization import (
    AssistantProjectionTarget,
    _empty_answer_projection,
)
from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy
from backend.app.agent_runtime.rag_v2_contracts import (
    COMPANY_MEMORY_RAG_GRAPH_VERSION,
    COMPANY_MEMORY_RAG_WORKFLOW,
    RagSurface,
    resolved_rag_mode,
    resolved_rag_stage,
)
from backend.app.agent_runtime.rag_v2_identity import StrictUnicodeScalarValidator
from backend.app.agent_runtime.rag_v2_registry import RagGraphRegistry
from backend.app.agent_runtime.rag_v2_state import (
    RagGraphOutput,
    RagRequestServices,
    RagRuntimeContext,
    RagRunTrace,
)
from backend.app.agents.rag_orchestrator_agent.v2_input import (
    PreparedRagRequestText,
    _prepared,
    _scan_value,
    prepare_direct_request_text,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.rag.lexical_projection import tokenize_rag_lexical_query


class RagApplicationError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def validate_prepared_ingress(
    text: PreparedRagRequestText, *, surface: RagSurface, settings: Settings
) -> bool:
    """Authenticate server preparation before opening any request dependency."""
    if type(text) is not PreparedRagRequestText:
        raise ValueError('prepared request text is required')
    for value in (
        text.caller_text,
        text.normalized_current_user_text,
        text.retrieval_query_text,
        text.answer_question_text,
    ):
        StrictUnicodeScalarValidator.validate(value)
    key = fingerprint_secret_bytes(settings)[0]
    direct = prepare_direct_request_text(text.caller_text, key=key)
    if surface in {'ask', 'search'}:
        if direct != text:
            raise ValueError('direct request preparation changed')
        return True
    if surface != 'assistant' or text.query_context_version != 'assistant-context:v1':
        raise ValueError('request context does not match surface')
    if (
        len(text.caller_text) > 4000
        or len(text.retrieval_query_text) > 8000
        or not text.caller_text.strip()
    ):
        raise ValueError('assistant request bounds exceeded')
    expected = _prepared(
        caller_text=text.caller_text,
        normalized_current_user_text=direct.normalized_current_user_text,
        retrieval_query_text=text.retrieval_query_text,
        answer_question_text=text.caller_text.strip(),
        query_context_version='assistant-context:v1',
        key=key,
    )
    if expected != text:
        raise ValueError('assistant request preparation changed')
    _scan_value(text.retrieval_query_text)
    return len(tokenize_rag_lexical_query(text.retrieval_query_text)) <= 1000


def ingress_budget_refusal(settings: Settings) -> RagGraphOutput:
    return {
        'outcome': 'budget_exceeded',
        'answer_blocks': None,
        'selected_slot_ids': (),
        'evidence_projection': _empty_answer_projection(settings),
        'model_influence': (),
        'hidden_match_count': 0,
        'effective_backend': 'deterministic_lexical',
        'fallback_category': None,
        'charged_cost_usd': Decimal('0.000000'),
        'sanitized_trace': RagRunTrace(
            {'validate_input': 1},
            {'validate_input': 0},
            {'query_embedding': 0, 'answer_generation': 0},
            {},
            {},
        ),
    }


@dataclass(frozen=True, slots=True)
class RagApplicationFacade:
    _settings: Settings = field(repr=False)
    _session_factory: Callable[[], Session] = field(repr=False)
    _request_factory: Callable[..., AbstractContextManager[RagRequestServices]] = field(
        repr=False
    )
    _graph_registry: RagGraphRegistry = field(repr=False)
    _rollout: RagRolloutPolicy = field(default_factory=RagRolloutPolicy)

    def execution_owner(self, surface: RagSurface) -> Literal['legacy', 'shadow', 'v2']:
        decision = self._rollout.decide(
            mode=resolved_rag_mode(self._settings),
            stage=resolved_rag_stage(self._settings),
            surface=surface,
        )
        return 'shadow' if decision.shadow_retrieval else decision.public_owner

    def invoke_graph(
        self,
        *,
        actor: DemoUser,
        surface: RagSurface,
        prepared_text: PreparedRagRequestText,
        assistant_target: AssistantProjectionTarget | None = None,
    ) -> RagGraphOutput:
        if self.execution_owner(surface) != 'v2':
            # The later shadow bridge owns comparison; this never invokes the
            # product graph or silently substitutes for the legacy service.
            raise RagApplicationError('runtime_version_unavailable')
        settings = self._settings.model_copy(deep=True)
        if not validate_prepared_ingress(
            prepared_text, surface=surface, settings=settings
        ):
            return ingress_budget_refusal(settings)
        if surface == 'assistant' and assistant_target is None:
            raise ValueError('assistant persistence target is required')
        if assistant_target is not None and (
            surface != 'assistant' or assistant_target.owner_user_id != actor.id
        ):
            raise ValueError('assistant target does not match request owner')
        graph = self._graph_registry.resolve(
            COMPANY_MEMORY_RAG_WORKFLOW, COMPANY_MEMORY_RAG_GRAPH_VERSION
        )
        request_actor = replace(
            actor, permission_levels=frozenset(actor.permission_levels)
        )
        with (
            tracing_context(enabled=False),
            self._request_factory(
                session_factory=self._session_factory,
                settings=settings,
                actor=request_actor,
                surface=surface,
                assistant_target=assistant_target,
            ) as services,
        ):
            if type(services) is not RagRequestServices:
                raise RagApplicationError('runtime_version_unavailable')
            context = RagRuntimeContext(
                actor=request_actor,
                surface=surface,
                settings=settings,
                services=services,
                assistant_target=assistant_target,
            )
            return graph.invoke(
                {'prepared_text': prepared_text},
                context=context,
                config={'callbacks': [], 'metadata': {}, 'recursion_limit': 40},
            )
