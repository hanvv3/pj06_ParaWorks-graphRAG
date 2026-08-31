from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from backend.app.core.config import Settings

RagSurface = Literal['search', 'ask', 'assistant']
RagMode = Literal['disabled', 'shadow', 'enforce']
RagCutoverStage = Literal['none', 'ask', 'search', 'assistant']
RagRetrievalBackend = Literal['keyword', 'pgvector']
RagEffectiveBackend = Literal['deterministic_lexical', 'pgvector']

COMPANY_MEMORY_RAG_WORKFLOW = 'company-memory-rag-answer'
COMPANY_MEMORY_RAG_GRAPH_VERSION = 'company-memory-rag-answer-v2.0'
COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION = 'rag-graph-state:v2'
RAG_ANSWER_PROMPT_VERSION = 'rag-answer:v2'


class ProviderDispatchPermit(Protocol):
    """One-use runtime dispatch capability; live-gate composition is separate."""

    def consume_at_dispatch(self) -> None: ...


def resolved_rag_mode(settings: Settings) -> RagMode:
    if 'langgraph_rag_v2_mode' in settings.model_fields_set:
        return settings.langgraph_rag_v2_mode
    if settings.langgraph_rag_v2_enabled:
        return 'shadow'
    return settings.langgraph_rag_v2_mode


def resolved_rag_stage(settings: Settings) -> RagCutoverStage:
    return settings.langgraph_rag_v2_stage


def resolved_rag_backend(settings: Settings) -> RagRetrievalBackend:
    if 'rag_retrieval_backend' in settings.model_fields_set:
        return settings.rag_retrieval_backend
    if settings.rag_use_pgvector_search:
        return 'pgvector'
    return settings.rag_retrieval_backend
