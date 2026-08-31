from backend.app.agent_runtime import EvidenceMessage, EvidencePacket, PermissionContext
from backend.app.agents.rag_orchestrator_agent.agent import RagModelResponse
from backend.app.agents.rag_orchestrator_agent.llm import (
    FallbackRagOrchestratorModel,
    RagLlmProviderError,
    render_rag_llm_prompt,
)


def test_v2_structured_answer_boundary_is_exported_without_changing_v1() -> None:
    from backend.app.agents.rag_orchestrator_agent import llm

    assert llm.StructuredRagAnswerModel.__name__ == 'StructuredRagAnswerModel'
    assert llm.PreparedAnswerInvocation.__name__ == 'PreparedAnswerInvocation'
    assert llm.ProviderAnswerEnvelope.__name__ == 'ProviderAnswerEnvelope'


def build_packet() -> EvidencePacket:
    return EvidencePacket(
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
            )
        ],
        permission_context=PermissionContext(user_id='viewer', role='employee'),
    )


def test_fallback_rag_model_uses_env_fallback_model_when_primary_openai_fails() -> None:
    class FailingPrimary:
        provider = 'openai'
        model_name = 'gpt-5.4-mini'

        def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
            raise RagLlmProviderError('primary unavailable')

    class EnvFallback:
        provider = 'openai'
        model_name = 'gpt-4.1-mini'

        def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
            return RagModelResponse(
                answer='fallback answer',
                input_tokens=11,
                output_tokens=3,
                model_name=self.model_name,
            )

    model = FallbackRagOrchestratorModel([FailingPrimary(), EnvFallback()])

    response = model.answer('Redis는 어디에 쓰이나요?', build_packet())

    assert response.answer == 'fallback answer'
    assert response.model_name == 'gpt-4.1-mini'


def test_render_rag_llm_prompt_bounds_evidence_text() -> None:
    prompt = render_rag_llm_prompt(
        question='Redis는 어디에 쓰이나요?',
        packet=build_packet(),
        max_input_chars=12,
    )

    assert 'Redis는 어디에 쓰이나요?' in prompt
    assert 'Redis should' in prompt
    assert 'transient job state' not in prompt
