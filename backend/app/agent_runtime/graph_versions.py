from collections.abc import Callable
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver

GraphBuilder = Callable[[BaseCheckpointSaver], object]


class RuntimeVersionUnavailable(RuntimeError):  # noqa: N818
    def __init__(self, graph_version: str) -> None:
        super().__init__(f'unsupported graph version: {graph_version}')


@dataclass(frozen=True)
class GraphVersionKey:
    workflow_name: str
    graph_version: str


class GraphVersionRegistry:
    def __init__(self) -> None:
        self._builders: dict[GraphVersionKey, GraphBuilder] = {}

    def register(
        self,
        workflow_name: str,
        graph_version: str,
        builder: GraphBuilder,
    ) -> None:
        if not workflow_name.strip() or not graph_version.strip():
            raise ValueError('workflow_name and graph_version are required')
        key = GraphVersionKey(workflow_name, graph_version)
        if key in self._builders:
            raise ValueError('graph version is already registered')
        self._builders = {**self._builders, key: builder}

    def resolve(self, workflow_name: str, graph_version: str) -> GraphBuilder:
        key = GraphVersionKey(workflow_name, graph_version)
        try:
            return self._builders[key]
        except KeyError:
            raise RuntimeVersionUnavailable(graph_version) from None


def register_company_memory_review_v2(
    registry: GraphVersionRegistry,
) -> None:
    from backend.app.agent_runtime.review_v2_graph import (
        build_company_memory_review_v2_graph,
    )
    from backend.app.schemas.review_workflow import (
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        COMPANY_MEMORY_REVIEW_WORKFLOW,
    )

    registry.register(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        build_company_memory_review_v2_graph,
    )


def register_company_memory_review_versions(
    registry: GraphVersionRegistry,
) -> None:
    from backend.app.agent_runtime.review_v21_graph import (
        build_company_memory_review_v21_graph,
    )
    from backend.app.schemas.auto_review import (
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
    )
    from backend.app.schemas.review_workflow import (
        COMPANY_MEMORY_REVIEW_WORKFLOW,
    )

    register_company_memory_review_v2(registry)
    registry.register(
        COMPANY_MEMORY_REVIEW_WORKFLOW,
        COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
        build_company_memory_review_v21_graph,
    )
