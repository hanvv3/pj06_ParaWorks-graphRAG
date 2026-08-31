from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from backend.app.agent_runtime.rag_v2_contracts import (
    COMPANY_MEMORY_RAG_GRAPH_VERSION,
    COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION,
    COMPANY_MEMORY_RAG_WORKFLOW,
    RAG_ANSWER_PROMPT_VERSION,
    RagCutoverStage,
    RagEffectiveBackend,
    RagMode,
    RagRetrievalBackend,
    RagSurface,
    resolved_rag_backend,
    resolved_rag_mode,
    resolved_rag_stage,
)
from backend.app.agents.rag_orchestrator_agent.agent import (
    RAG_ORCHESTRATOR_AGENT_MANIFEST,
)
from backend.app.core.config import Settings


def test_rag_v2_contract_literals_and_identities_are_frozen() -> None:
    assert get_args(RagSurface) == ('search', 'ask', 'assistant')
    assert get_args(RagMode) == ('disabled', 'shadow', 'enforce')
    assert get_args(RagCutoverStage) == ('none', 'ask', 'search', 'assistant')
    assert get_args(RagRetrievalBackend) == ('keyword', 'pgvector')
    assert get_args(RagEffectiveBackend) == ('deterministic_lexical', 'pgvector')
    assert COMPANY_MEMORY_RAG_WORKFLOW == 'company-memory-rag-answer'
    assert COMPANY_MEMORY_RAG_GRAPH_VERSION == 'company-memory-rag-answer-v2.0'
    assert COMPANY_MEMORY_RAG_STATE_SCHEMA_VERSION == 'rag-graph-state:v2'
    assert RAG_ANSWER_PROMPT_VERSION == 'rag-answer:v2'


def test_rag_v2_manifest_preserves_the_public_agent_name() -> None:
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.name == 'rag_orchestrator_agent'
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.owner == 'Developer C'
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.input_contract == 'RagGraphInput'
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.output_contract == 'RagGraphOutput'
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.prompt_versions == (
        'rag-answer:v1',
        'rag-answer:v2',
    )
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.supported_permissions == (
        'public',
        'internal',
        'restricted',
    )
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.capabilities == (
        'question_answering',
        'rag_answering',
        'orchestration',
    )


@pytest.mark.parametrize(
    ('setting_name', 'invalid_value'),
    (
        ('langgraph_rag_v2_mode', 'enabled'),
        ('langgraph_rag_v2_stage', 'all'),
        ('rag_retrieval_backend', 'semantic'),
    ),
)
def test_rag_v2_invalid_canonical_configuration_is_rejected(
    setting_name: str,
    invalid_value: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{setting_name: invalid_value})


def test_legacy_rag_v2_enabled_maps_only_to_shadow_without_canonical_mode() -> None:
    settings = Settings(_env_file=None, langgraph_rag_v2_enabled=True)

    assert resolved_rag_mode(settings) == 'shadow'
    assert resolved_rag_stage(settings) == 'none'


def test_canonical_rag_mode_takes_precedence_over_legacy_enabled_alias() -> None:
    settings = Settings(
        _env_file=None,
        langgraph_rag_v2_enabled=True,
        langgraph_rag_v2_mode='enforce',
    )

    assert resolved_rag_mode(settings) == 'enforce'


def test_canonical_backend_takes_precedence_over_legacy_pgvector_alias() -> None:
    settings = Settings(
        _env_file=None,
        rag_use_pgvector_search=True,
        rag_retrieval_backend='keyword',
    )

    assert resolved_rag_backend(settings) == 'keyword'


def test_legacy_pgvector_alias_is_used_without_canonical_backend() -> None:
    settings = Settings(_env_file=None, rag_use_pgvector_search=True)

    assert resolved_rag_backend(settings) == 'pgvector'
