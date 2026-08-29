from dataclasses import dataclass
from typing import Protocol

from backend.app.agent_runtime import (
    AgentManifest,
    AgentRunCost,
    EvidencePacket,
    TokenUsage,
    build_evidence_cache_key,
    estimate_agent_run_cost,
)

RAG_ORCHESTRATOR_AGENT_NAME = 'rag_orchestrator_agent'
RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION = 'rag-answer:v1'
RAG_ORCHESTRATOR_AGENT_MODEL_NAME = 'fake-rag-orchestrator-model'

RAG_ORCHESTRATOR_AGENT_MANIFEST = AgentManifest(
    name=RAG_ORCHESTRATOR_AGENT_NAME,
    owner='Developer C',
    input_contract='EvidencePacket',
    output_contract='RagAnswer',
    prompt_versions=(RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION,),
    supported_permissions=('internal', 'restricted'),
    capabilities=('question_answering', 'rag_answering', 'orchestration'),
)


@dataclass(frozen=True)
class RagModelResponse:
    answer: str
    input_tokens: int
    output_tokens: int
    model_name: str | None = None


@dataclass(frozen=True)
class RagAnswer:
    agent_name: str
    prompt_version: str
    question: str
    answer: str
    source_ids: list[str]
    source_links: list[str]
    source_snippets: list[str]
    citations: list[dict[str, object]]
    permission_level: str
    hidden_match_count: int
    permission_notice: str | None
    cost: AgentRunCost
    cache_key: str
    agent_run_id: int | None = None
    serving_dependencies: tuple[object, ...] = ()


class RagOrchestratorModel(Protocol):
    def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
        raise NotImplementedError


class DeterministicRagOrchestratorModel:
    def answer(self, question: str, packet: EvidencePacket) -> RagModelResponse:
        combined_text = '\n'.join(message.text for message in packet.messages)
        if not packet.messages:
            answer = '권한 내에서 확인 가능한 근거를 찾지 못했습니다.'
        elif 'redis' in combined_text.lower():
            answer = (
                'Redis는 일시적인 작업 상태와 큐 진행 상황을 빠르게 공유하는 데 사용되며, '
                'PostgreSQL은 오래 보존해야 하는 기록의 기준 저장소로 남습니다.'
            )
        else:
            answer = packet.messages[0].source_snippet

        token_basis = f'{question}\n{combined_text}'
        input_tokens = max(1, len(token_basis) // 4)
        output_tokens = max(32, len(answer) // 4)

        return RagModelResponse(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@dataclass(frozen=True)
class RagOrchestratorAgent:
    model: RagOrchestratorModel
    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60

    def answer(
        self,
        *,
        question: str,
        packet: EvidencePacket,
        hidden_match_count: int,
    ) -> RagAnswer:
        model_response = self.model.answer(question, packet)
        model_name = model_response.model_name or RAG_ORCHESTRATOR_AGENT_MODEL_NAME
        token_usage = TokenUsage(
            input_tokens=model_response.input_tokens,
            output_tokens=model_response.output_tokens,
        )
        cost = estimate_agent_run_cost(
            model_name=model_name,
            token_usage=token_usage,
            input_cost_per_1m=self.input_cost_per_1m,
            output_cost_per_1m=self.output_cost_per_1m,
            cache_hit=False,
        )
        cache_key = build_evidence_cache_key(packet, RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION)

        return RagAnswer(
            agent_name=RAG_ORCHESTRATOR_AGENT_NAME,
            prompt_version=RAG_ORCHESTRATOR_AGENT_PROMPT_VERSION,
            question=question,
            answer=model_response.answer,
            source_ids=packet.source_ids,
            source_links=packet.source_links,
            source_snippets=packet.source_snippets,
            citations=[
                {
                    'source_id': message.source_id,
                    'source_url': message.source_url,
                    'source_type': message.metadata.get('source_type'),
                    'permission_level': message.permission_level,
                    'source_snippet': message.source_snippet,
                    'relevance_score': message.metadata.get('relevance_score', 0.0),
                    'matched_terms': message.metadata.get('matched_terms', []),
                }
                for message in packet.messages
            ],
            permission_level=packet.strictest_permission,
            hidden_match_count=hidden_match_count,
            permission_notice='Some sources may be hidden by permissions.' if hidden_match_count else None,
            cost=cost,
            cache_key=cache_key,
        )
