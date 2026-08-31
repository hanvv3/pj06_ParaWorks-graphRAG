from backend.app.agent_runtime import EvidenceMessage, EvidencePacket, PermissionContext
from backend.app.agents.rag_orchestrator_agent import (
    RAG_ORCHESTRATOR_AGENT_MANIFEST,
    RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION,
    DeterministicRagOrchestratorModel,
    RagModelResponse,
    RagOrchestratorAgent,
)


def test_rag_orchestrator_manifest_declares_shared_contracts() -> None:
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.name == 'rag_orchestrator_agent'
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.input_contract == 'RagGraphInput'
    assert RAG_ORCHESTRATOR_AGENT_MANIFEST.output_contract == 'RagGraphOutput'
    assert RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION in RAG_ORCHESTRATOR_AGENT_MANIFEST.prompt_versions
    assert 'question_answering' in RAG_ORCHESTRATOR_AGENT_MANIFEST.capabilities


def test_rag_orchestrator_returns_evidence_backed_answer_with_cost() -> None:
    packet = EvidencePacket(
        source_type='rag',
        source_window='ask:redis',
        messages=[
            EvidenceMessage(
                source_id='gmail-redis',
                source_url='https://gmail.mock/project-alpha/redis-summary',
                text='Redis should be used for transient job state.',
                author='noah@example.com',
                timestamp='2026-04-30T10:15:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            ),
            EvidenceMessage(
                source_id='drive-redis',
                source_url='https://drive.mock/project-alpha/architecture-note',
                text='PostgreSQL remains the durable source of record.',
                author='lee@example.com',
                timestamp='2026-04-30T11:00:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'drive'},
            ),
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    answer = RagOrchestratorAgent(model=DeterministicRagOrchestratorModel()).answer(
        question='Redis는 무엇에 쓰이나요?',
        packet=packet,
        hidden_match_count=0,
    )

    assert answer.agent_name == 'rag_orchestrator_agent'
    assert answer.prompt_version == 'rag-answer:v1'
    assert 'Redis' in answer.answer
    assert answer.source_links == [
        'https://gmail.mock/project-alpha/redis-summary',
        'https://drive.mock/project-alpha/architecture-note',
    ]
    assert answer.cost.token_usage.total_tokens > 0
    assert answer.cost.estimated_cost_usd > 0
    assert answer.permission_notice is None


def test_rag_orchestrator_records_actual_model_name_from_llm_response() -> None:
    class FakeLlmModel:
        def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
            return RagModelResponse(
                answer='근거에 따르면 Redis는 임시 작업 상태에 사용됩니다.',
                input_tokens=120,
                output_tokens=30,
                model_name='gpt-5.4-mini',
            )

    packet = EvidencePacket(
        source_type='rag',
        source_window='ask:redis',
        messages=[
            EvidenceMessage(
                source_id='gmail-redis',
                source_url='https://gmail.mock/redis',
                text='Redis should be used for transient job state.',
                author='noah@example.com',
                timestamp='2026-04-30T10:15:00+00:00',
                permission_level='internal',
                metadata={'source_type': 'gmail'},
            ),
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )

    answer = RagOrchestratorAgent(model=FakeLlmModel()).answer(
        question='Redis는 무엇에 쓰이나요?',
        packet=packet,
        hidden_match_count=0,
    )

    assert answer.cost.model_name == 'gpt-5.4-mini'
