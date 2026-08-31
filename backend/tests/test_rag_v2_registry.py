from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from backend.app.agent_runtime.rag_v2_contracts import (
    COMPANY_MEMORY_RAG_GRAPH_VERSION,
    COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION,
    COMPANY_MEMORY_RAG_WORKFLOW,
)
from backend.app.agent_runtime.rag_v2_registry import (
    RagGraphRegistration,
    RagGraphRegistry,
    RagRuntimeVersionUnavailableError,
    build_rag_manifest_registry,
)


class _State(TypedDict):
    value: str


def _compiled_test_graph():
    builder = StateGraph(_State)
    builder.add_node('pass_through', lambda state: state)
    builder.add_edge(START, 'pass_through')
    builder.add_edge('pass_through', END)
    return builder.compile()


def _registration():
    return RagGraphRegistration(
        workflow_name=COMPANY_MEMORY_RAG_WORKFLOW,
        graph_version=COMPANY_MEMORY_RAG_GRAPH_VERSION,
        state_schema_version=COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION,
        graph=_compiled_test_graph(),
    )


def test_rag_graph_registry_resolves_only_the_exact_registered_version() -> None:
    registration = _registration()
    registry = RagGraphRegistry()
    registry.register(registration)

    assert registry.resolve(
        COMPANY_MEMORY_RAG_WORKFLOW,
        COMPANY_MEMORY_RAG_GRAPH_VERSION,
    ) is registration.graph


def test_rag_graph_registry_rejects_duplicate_registration() -> None:
    registry = RagGraphRegistry()
    registry.register(_registration())

    with pytest.raises(ValueError, match='already registered'):
        registry.register(_registration())


@pytest.mark.parametrize('graph_version', ('', 'latest', 'company-memory-rag-answer-v1'))
def test_rag_graph_registry_fails_closed_for_missing_or_unknown_versions(
    graph_version: str,
) -> None:
    registry = RagGraphRegistry()

    with pytest.raises(RagRuntimeVersionUnavailableError) as error:
        registry.resolve(COMPANY_MEMORY_RAG_WORKFLOW, graph_version)

    assert error.value.code == 'runtime_version_unavailable'


def test_rag_graph_registration_is_frozen_and_carries_no_request_data() -> None:
    registration = _registration()

    assert tuple(field.name for field in fields(registration)) == (
        'workflow_name',
        'graph_version',
        'state_schema_version',
        'graph',
    )
    with pytest.raises(FrozenInstanceError):
        registration.workflow_name = 'other'  # type: ignore[misc]


def test_rag_manifest_registry_contains_only_the_rag_manifest() -> None:
    registry = build_rag_manifest_registry()

    assert registry.names == ('rag_orchestrator_agent',)
    assert registry.get('rag_orchestrator_agent').output_contract == 'RagGraphOutput'
