from importlib import import_module
from typing import Any

from backend.app.agent_runtime.bootstrap import (
    BootstrapResult,
    CheckpointBootstrapError,
    bootstrap_langgraph_checkpointer,
)
from backend.app.agent_runtime.checkpoint_execution import (
    CheckpointConfirmation,
    CheckpointConfirmationError,
    checkpoint_config,
    invoke_and_confirm_checkpoint,
    require_resumable_checkpoint,
)
from backend.app.agent_runtime.contracts import (
    AgentCostBudgetDecision,
    AgentManifest,
    AgentRunCost,
    AgentRunResult,
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
    ReviewCandidate,
    TokenUsage,
)
from backend.app.agent_runtime.cost_policy import (
    build_evidence_cache_key,
    estimate_agent_run_cost,
    evaluate_agent_cost_budget,
)
from backend.app.agent_runtime.evidence_summary import build_evidence_summary
from backend.app.agent_runtime.fingerprints import (
    canonical_json_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.graph_versions import (
    GraphVersionRegistry,
    RuntimeVersionUnavailable,
)
from backend.app.agent_runtime.orchestration import (
    AgentWorkflow,
    AgentWorkflowState,
    build_agent_workflow,
    build_company_memory_workflow,
)
from backend.app.agent_runtime.project_routing import (
    LangChainProjectRouterModel,
    ProjectOption,
    ProjectRouterModel,
    ProjectRoutingCandidate,
    ProjectRoutingDecision,
    ProjectRoutingResult,
    apply_project_routing_to_payload,
    build_project_tools,
    route_projects_for_candidates,
    score_project_aliases,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.agent_runtime.retention import (
    CheckpointPruneResult,
    prune_expired_checkpoints,
)
from backend.app.agent_runtime.state import (
    ReviewGraphInput,
    ReviewGraphOutput,
    ReviewGraphState,
    ReviewRuntimeContext,
)

__all__ = [
    'AgentManifest',
    'AgentCostBudgetDecision',
    'AgentRegistry',
    'AgentWorkflow',
    'AgentWorkflowState',
    'AgentRunCost',
    'AgentRunResult',
    'BootstrapResult',
    'CheckpointBootstrapError',
    'CheckpointConfirmation',
    'CheckpointConfirmationError',
    'CheckpointPruneResult',
    'EvidenceMessage',
    'EvidencePacket',
    'GraphVersionRegistry',
    'PermissionContext',
    'LangChainProjectRouterModel',
    'ProjectOption',
    'ProjectRouterModel',
    'ProjectRoutingCandidate',
    'ProjectRoutingDecision',
    'ProjectRoutingResult',
    'ReviewCandidate',
    'ReviewAgentAdapter',
    'ReviewAgentCatalog',
    'ReviewDraftError',
    'ReviewDraftResult',
    'ReviewDraftService',
    'ReviewGraphInput',
    'ReviewGraphOutput',
    'ReviewGraphState',
    'ReviewRuntimeContext',
    'RuntimeVersionUnavailable',
    'ReviewModelUnavailableError',
    'RoutedReviewModel',
    'TokenUsage',
    'apply_project_routing_to_payload',
    'build_project_tools',
    'build_evidence_cache_key',
    'build_evidence_summary',
    'build_agent_workflow',
    'build_company_memory_workflow',
    'build_langchain_review_chat_model',
    'build_permission_fingerprint',
    'build_review_agent_catalog',
    'build_review_effect_key',
    'bootstrap_langgraph_checkpointer',
    'canonical_json_bytes',
    'checkpoint_config',
    'evaluate_agent_cost_budget',
    'estimate_agent_run_cost',
    'keyed_fingerprint',
    'invoke_and_confirm_checkpoint',
    'prune_expired_checkpoints',
    'route_projects_for_candidates',
    'require_resumable_checkpoint',
    'score_project_aliases',
]

_LAZY_EXPORTS = {
    'ReviewModelUnavailableError': (
        'backend.app.agent_runtime.model_router',
        'ReviewModelUnavailableError',
    ),
    'RoutedReviewModel': (
        'backend.app.agent_runtime.model_router',
        'RoutedReviewModel',
    ),
    'build_langchain_review_chat_model': (
        'backend.app.agent_runtime.model_router',
        'build_langchain_review_chat_model',
    ),
    'ReviewAgentAdapter': (
        'backend.app.agent_runtime.review_v2_agents',
        'ReviewAgentAdapter',
    ),
    'ReviewAgentCatalog': (
        'backend.app.agent_runtime.review_v2_agents',
        'ReviewAgentCatalog',
    ),
    'build_review_agent_catalog': (
        'backend.app.agent_runtime.review_v2_agents',
        'build_review_agent_catalog',
    ),
    'ReviewDraftError': (
        'backend.app.agent_runtime.review_v2_drafting',
        'ReviewDraftError',
    ),
    'ReviewDraftResult': (
        'backend.app.agent_runtime.review_v2_drafting',
        'ReviewDraftResult',
    ),
    'ReviewDraftService': (
        'backend.app.agent_runtime.review_v2_drafting',
        'ReviewDraftService',
    ),
    'build_permission_fingerprint': (
        'backend.app.agent_runtime.review_v2_drafting',
        'build_permission_fingerprint',
    ),
    'build_review_effect_key': (
        'backend.app.agent_runtime.review_v2_drafting',
        'build_review_effect_key',
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
