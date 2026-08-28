import json

from backend.app.agent_runtime import EvidenceMessage, EvidencePacket, PermissionContext
from backend.app.agents.memory_extraction_agent.langchain_adapter import (
    LangChainMemoryExtractionModel,
    StructuredMemoryExtractionOutput,
    render_memory_extraction_prompt,
)


class FakeStructuredChatModel:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return StructuredMemoryExtractionOutput(
            title='Redis decision',
            summary='Redis will handle queue progress while PostgreSQL remains durable storage.',
            item_type='decision_record',
            confidence_score=0.86,
            payload_fields={'decision_summary': 'Redis handles queue progress.'},
            uncertainty_reason=None,
        )


class FakeChatModel:
    def __init__(self) -> None:
        self.schema = None
        self.structured_model = FakeStructuredChatModel()

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_model


def test_langchain_memory_adapter_uses_structured_output_contract() -> None:
    chat_model = FakeChatModel()
    packet = build_packet()
    model = LangChainMemoryExtractionModel(
        chat_model=chat_model,
        expected_item_type='decision_record',
        task_name='decision record extraction',
        model_name='fake-structured-model',
    )

    response = model.extract(packet)

    assert chat_model.schema is StructuredMemoryExtractionOutput
    assert response.title == 'Redis decision'
    assert response.item_type == 'decision_record'
    assert response.confidence_score == 0.86
    assert response.payload_fields == {
        'decision_summary': 'Redis handles queue progress.'
    }
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert chat_model.structured_model.messages[0][0] == 'system'
    assert (
        'Use only the provided evidence' in chat_model.structured_model.messages[0][1]
    )


def test_memory_langchain_result_records_exact_provider_and_model() -> None:
    response = LangChainMemoryExtractionModel(
        chat_model=FakeChatModel(),
        expected_item_type='decision_record',
        task_name='decision record extraction',
        model_name='gpt-5.4-mini-2026-03-17',
        model_provider='openai',
        reasoning_effort='none',
        route_version='auto-review-extraction-route:v1',
        output_contract_version='decision-record-extraction:v1',
    ).extract(build_packet())

    assert response.model_provider == 'openai'
    assert response.model_name == 'gpt-5.4-mini-2026-03-17'
    assert response.model_reasoning_effort == 'none'
    assert response.route_version == 'auto-review-extraction-route:v1'
    assert response.output_contract_version == 'decision-record-extraction:v1'


def test_render_memory_extraction_prompt_omits_non_atomic_mandatory_envelope() -> None:
    packet = build_packet(text='A' * 120)

    prompt = render_memory_extraction_prompt(
        packet,
        expected_item_type='timeline_event',
        task_name='timeline extraction',
        max_input_chars=32,
    )

    payload = json.loads(prompt)

    assert payload['expected_item_type'] == 'timeline_event'
    assert payload['evidence'] == []


def build_packet(
    text: str = 'Decision: Redis queue progress moves into company memory.',
) -> EvidencePacket:
    return EvidencePacket(
        source_type='company_memory',
        source_window='test:structured-output',
        messages=[
            EvidenceMessage(
                source_id='source-1',
                source_url='https://example.test/source-1',
                text=text,
                author='owner@example.com',
                timestamp='2026-05-02T09:00:00+09:00',
                permission_level='internal',
            )
        ],
        permission_context=PermissionContext(user_id='demo-admin', role='admin'),
    )
