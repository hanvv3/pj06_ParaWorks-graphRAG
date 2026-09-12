from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from langgraph.graph.state import CompiledStateGraph

from backend.app.agent_runtime.registry import AgentRegistry


class RagRuntimeVersionUnavailableError(RuntimeError):
    code = 'runtime_version_unavailable'

    def __init__(self, graph_version: str) -> None:
        super().__init__(f'unsupported RAG graph version: {graph_version}')


@dataclass(frozen=True, slots=True)
class RagGraphRegistration:
    workflow_name: str
    graph_version: str
    state_schema_version: str
    graph: CompiledStateGraph


class RagGraphRegistry:
    def __init__(self) -> None:
        self._graphs: dict[tuple[str, str], CompiledStateGraph] = {}
        self._sealed = False

    def seal(self) -> None:
        self._graphs = MappingProxyType(dict(self._graphs))
        self._sealed = True

    def register(self, registration: RagGraphRegistration) -> None:
        if self._sealed:
            raise ValueError('RAG graph registry is sealed')
        if not (
            registration.workflow_name.strip()
            and registration.graph_version.strip()
            and registration.state_schema_version.strip()
        ):
            raise ValueError('RAG graph registration requires non-empty identities')
        key = (registration.workflow_name, registration.graph_version)
        if key in self._graphs:
            raise ValueError('RAG graph version is already registered')
        self._graphs = {**self._graphs, key: registration.graph}

    def resolve(self, workflow_name: str, graph_version: str) -> CompiledStateGraph:
        try:
            return self._graphs[(workflow_name, graph_version)]
        except KeyError:
            raise RagRuntimeVersionUnavailableError(graph_version) from None


class _SealedRagManifestRegistry(AgentRegistry):
    def __init__(self, manifest) -> None:
        super().__init__()
        self._manifests = MappingProxyType({manifest.name: manifest})

    def register(self, manifest) -> None:
        raise ValueError('RAG manifest registry is sealed')


def build_rag_manifest_registry() -> AgentRegistry:
    from backend.app.agents.rag_orchestrator_agent.agent import (
        RAG_ORCHESTRATOR_AGENT_MANIFEST,
    )

    return _SealedRagManifestRegistry(RAG_ORCHESTRATOR_AGENT_MANIFEST)
