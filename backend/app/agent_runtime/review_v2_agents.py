from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from backend.app.agent_runtime.contracts import (
    AgentCostBudgetDecision,
    AgentManifest,
    AgentRunResult,
    EvidencePacket,
    TokenUsage,
)
from backend.app.agent_runtime.cost_policy import (
    estimate_agent_run_cost,
    evaluate_agent_cost_budget,
)
from backend.app.agent_runtime.model_router import (
    RoutedReviewModel,
    build_mail_document_model_route,
    build_memory_model_route,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agents.mail_document_agent import (
    MAIL_DOCUMENT_AGENT_MANIFEST,
    MailDocumentAgent,
)
from backend.app.agents.memory_extraction_agent import (
    DECISION_RECORD_AGENT_MANIFEST,
    HISTORY_AGENT_MANIFEST,
    TIMELINE_AGENT_MANIFEST,
    TODO_AGENT_MANIFEST,
    DecisionRecordAgent,
    DeterministicDecisionRecordModel,
    DeterministicHistoryModel,
    DeterministicTimelineModel,
    DeterministicTodoModel,
    HistoryAgent,
    LangChainMemoryExtractionModel,
    TimelineAgent,
    TodoAgent,
)
from backend.app.core.config import Settings
from backend.app.schemas.review_workflow import DEFAULT_REVIEW_AGENT_NAMES

MEMORY_ESTIMATED_OUTPUT_TOKENS = 128


class ReviewAgentAdapter(Protocol):
    manifest: AgentManifest
    estimated_output_tokens: int
    model_name: str
    model_route_version: str

    def preflight(self, packet: EvidencePacket) -> AgentCostBudgetDecision:
        raise NotImplementedError

    def run(self, packet: EvidencePacket) -> AgentRunResult:
        raise NotImplementedError


@dataclass(frozen=True)
class _ReviewAgentAdapter:
    manifest: AgentManifest
    agent: object
    estimated_output_tokens: int
    model_name: str
    model_route_version: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    max_cost_usd: float | None
    deterministic: bool
    preserve_result_model_name: bool = False

    def preflight(self, packet: EvidencePacket) -> AgentCostBudgetDecision:
        if not packet.messages:
            return AgentCostBudgetDecision(
                action='skip',
                reason='no_input',
                budget_status='no_input',
                model_name=self.model_name,
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
                estimated_cost_usd=0.0,
                budget_limit_usd=self.max_cost_usd,
                cache_hit=False,
            )
        token_usage = TokenUsage(
            input_tokens=sum(
                max(1, len(message.text.strip()) // 4)
                for message in packet.messages
                if message.text.strip()
            ),
            output_tokens=self.estimated_output_tokens,
        )
        return evaluate_agent_cost_budget(
            model_name=self.model_name,
            token_usage=token_usage,
            input_cost_per_1m=0.0 if self.deterministic else self.input_cost_per_1m,
            output_cost_per_1m=0.0 if self.deterministic else self.output_cost_per_1m,
            max_cost_usd=self.max_cost_usd,
            cache_hit=False,
        )

    def run(self, packet: EvidencePacket) -> AgentRunResult:
        result = self.agent.run(packet)  # type: ignore[attr-defined]
        cost = estimate_agent_run_cost(
            model_name=(
                result.cost.model_name
                if self.preserve_result_model_name
                else self.model_name
            ),
            token_usage=result.cost.token_usage,
            input_cost_per_1m=0.0 if self.deterministic else self.input_cost_per_1m,
            output_cost_per_1m=0.0 if self.deterministic else self.output_cost_per_1m,
            cache_hit=False,
        )
        return replace(result, cost=cost)


class ReviewAgentCatalog:
    def __init__(self, adapters: Sequence[ReviewAgentAdapter]) -> None:
        by_name = {adapter.manifest.name: adapter for adapter in adapters}
        if len(by_name) != len(adapters) or set(by_name) != set(DEFAULT_REVIEW_AGENT_NAMES):
            raise ValueError('review agent catalog must contain the exact public manifests')
        self._registry = _FixedAgentRegistry()
        self._adapters: dict[str, ReviewAgentAdapter] = {}
        for name in DEFAULT_REVIEW_AGENT_NAMES:
            adapter = by_name[name]
            if adapter.manifest.name != name:
                raise ValueError('review agent manifest does not match its catalog key')
            self._registry.register(adapter.manifest)
            self._adapters[name] = adapter
        self._registry.seal()

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def get(self, name: str) -> ReviewAgentAdapter:
        self._registry.get(name)
        return self._adapters[name]


class _FixedAgentRegistry(AgentRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._sealed = False

    def register(self, manifest: AgentManifest) -> None:
        if self._sealed:
            raise ValueError('review agent registry is immutable')
        super().register(manifest)

    def seal(self) -> None:
        self._sealed = True


def build_review_agent_catalog(
    settings: Settings,
    *,
    mail_model_builder: Callable | None = None,
    chat_model_builder: Callable | None = None,
) -> ReviewAgentCatalog:
    mail_route = build_mail_document_model_route(
        settings,
        model_builder=mail_model_builder,
    )
    memory_route = build_memory_model_route(
        settings,
        chat_model_builder=chat_model_builder,
    )
    common = {
        'input_cost_per_1m': settings.agent_llm_input_cost_per_1m_tokens,
        'output_cost_per_1m': settings.agent_llm_output_cost_per_1m_tokens,
        'max_cost_usd': settings.agent_llm_max_estimated_cost_usd,
    }
    adapters: list[ReviewAgentAdapter] = [
        _ReviewAgentAdapter(
            manifest=MAIL_DOCUMENT_AGENT_MANIFEST,
            agent=MailDocumentAgent(
                model=mail_route.model,
                input_cost_per_1m=settings.agent_llm_input_cost_per_1m_tokens,
                output_cost_per_1m=settings.agent_llm_output_cost_per_1m_tokens,
            ),
            estimated_output_tokens=settings.agent_llm_max_output_tokens,
            model_name=mail_route.model_name,
            model_route_version=mail_route.route_version,
            deterministic=mail_route.deterministic,
            preserve_result_model_name=True,
            **common,
        ),
    ]
    adapters.extend(
        _memory_adapters(settings=settings, route=memory_route, common=common)
    )
    return ReviewAgentCatalog(adapters)


def _memory_adapters(
    *,
    settings: Settings,
    route: RoutedReviewModel,
    common: dict[str, float | None],
) -> list[ReviewAgentAdapter]:
    definitions = (
        (
            TIMELINE_AGENT_MANIFEST,
            TimelineAgent,
            DeterministicTimelineModel,
            'timeline_event',
            'timeline extraction',
        ),
        (
            HISTORY_AGENT_MANIFEST,
            HistoryAgent,
            DeterministicHistoryModel,
            'history_event',
            'history extraction',
        ),
        (
            DECISION_RECORD_AGENT_MANIFEST,
            DecisionRecordAgent,
            DeterministicDecisionRecordModel,
            'decision_record',
            'decision record extraction',
        ),
        (
            TODO_AGENT_MANIFEST,
            TodoAgent,
            DeterministicTodoModel,
            'todo',
            'todo extraction',
        ),
    )
    adapters: list[ReviewAgentAdapter] = []
    for manifest, agent_type, deterministic_type, item_type, task_name in definitions:
        model = (
            deterministic_type()
            if route.deterministic
            else LangChainMemoryExtractionModel(
                chat_model=route.model,
                expected_item_type=item_type,
                task_name=task_name,
                model_name=route.model_name,
                max_input_chars=settings.agent_llm_max_input_chars,
            )
        )
        adapters.append(
            _ReviewAgentAdapter(
                manifest=manifest,
                agent=agent_type(model=model),
                estimated_output_tokens=MEMORY_ESTIMATED_OUTPUT_TOKENS,
                model_name=route.model_name,
                model_route_version=route.route_version,
                deterministic=route.deterministic,
                **common,
            )
        )
    return adapters
