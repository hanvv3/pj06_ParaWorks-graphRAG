from __future__ import annotations

import inspect
import typing

from backend.app.agent_runtime.rag_cost_ledger import RagCostLedger
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderDispatchAuthority,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    PreparedAnswerInvocation,
)
from backend.app.rag.retrieval import PreparedQueryEmbedding


def test_public_dispatch_boundary_has_no_raw_client_bytes_or_factory_hooks():
    import backend.app.agent_runtime.rag_provider_transport as transport_module
    import backend.app.agent_runtime.rag_runtime_contracts as contracts_module

    assert tuple(inspect.signature(RagCostLedger.__init__).parameters) == (
        'self',
        'authority',
    )
    assert tuple(
        inspect.signature(RagProviderDispatchAuthority.__init__).parameters
    ) == ('self', 'authority')
    production_assembler = transport_module._assemble_direct_openai_rag_provider_dispatch_authority
    assert 'provider_client' not in inspect.signature(
        production_assembler
    ).parameters
    assert not hasattr(transport_module, 'RagProviderRequest')
    assert not hasattr(contracts_module, 'StrictProviderOutcome')
    assert not hasattr(RagCostLedger, '_transport_provider_client')
    assert tuple(
        inspect.signature(RagProviderDispatchAuthority.prepare).parameters
    ) == ('self', 'grant', 'prepared')


def test_provider_prepare_accepts_only_frozen_domain_invocations():
    annotation = typing.get_type_hints(
        RagProviderDispatchAuthority.prepare
    )['prepared']
    assert annotation == PreparedQueryEmbedding | PreparedAnswerInvocation


def test_dispatch_authority_exposes_no_caller_send_or_raw_serialization_hooks():
    import backend.app.agent_runtime.provider_send_fence as fence_module

    assert not hasattr(fence_module, 'run_shared_advisory_send_fence')
    for method_name in ('prepare', 'dispatch', 'finalize'):
        method = getattr(RagProviderDispatchAuthority, method_name)
        parameters = set(inspect.signature(method).parameters)
        assert not parameters & {
            'provider_client',
            'send',
            'request_bytes',
            'rendered_input_utf8',
            'response_schema_json',
            'connection_factory',
            'freshness',
            'operation',
        }
