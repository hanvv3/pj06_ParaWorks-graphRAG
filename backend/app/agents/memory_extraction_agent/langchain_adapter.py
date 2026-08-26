import json
from typing import Any

from pydantic import BaseModel, Field

from backend.app.agent_runtime import EvidencePacket, LangChainInvocationPayload
from backend.app.agent_runtime.langchain_evidence import (
    render_bounded_evidence_rows,
)
from backend.app.agents.memory_extraction_agent.agent import (
    MemoryExtractionModelResponse,
)

DEFAULT_MAX_INPUT_CHARS = 12_000


class StructuredMemoryExtractionOutput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=1200)
    item_type: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    payload_fields: dict[str, str] = Field(default_factory=dict)
    uncertainty_reason: str | None = None


class LangChainMemoryExtractionModel:
    def __init__(
        self,
        *,
        chat_model: Any,
        expected_item_type: str,
        task_name: str,
        model_name: str,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        self.chat_model = chat_model
        self.expected_item_type = expected_item_type
        self.task_name = task_name
        self.model_name = model_name
        self.max_input_chars = max_input_chars

    def extract(self, packet: EvidencePacket) -> MemoryExtractionModelResponse:
        invocation = render_memory_extraction_langchain_invocation(
            packet,
            expected_item_type=self.expected_item_type,
            task_name=self.task_name,
            max_input_chars=self.max_input_chars,
        )
        structured_model = self.chat_model.with_structured_output(
            invocation.structured_output_schema
        )
        output = _coerce_structured_output(
            structured_model.invoke(list(invocation.messages))
        )
        summary = output.summary
        return MemoryExtractionModelResponse(
            title=output.title,
            summary=summary,
            item_type=output.item_type or self.expected_item_type,
            confidence_score=output.confidence_score,
            input_tokens=max(1, len(invocation.canonical_description()) // 4),
            output_tokens=max(1, (len(output.title) + len(summary)) // 4),
            payload_fields=output.payload_fields,
            uncertainty_reason=output.uncertainty_reason,
        )


def render_memory_extraction_prompt(
    packet: EvidencePacket,
    *,
    expected_item_type: str,
    task_name: str,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> str:
    evidence_rows = render_bounded_evidence_rows(
        packet,
        max_input_chars=max_input_chars,
    )

    return json.dumps(
        {
            'task': task_name,
            'expected_item_type': expected_item_type,
            'source_window': packet.source_window,
            'requirements': [
                'Use only the provided evidence.',
                'The title and summary must be written in Korean.',
                'Preserve uncertainty when evidence is weak.',
                'Keep payload_fields aligned with the expected item type.',
            ],
            'evidence': evidence_rows,
        },
        ensure_ascii=False,
        default=str,
    )


def render_memory_extraction_langchain_invocation(
    packet: EvidencePacket,
    *,
    expected_item_type: str,
    task_name: str,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> LangChainInvocationPayload:
    system_prompt = (
        f'You are performing {task_name} for ParaWorks. '
        'The title and summary must be written in Korean. '
        'Use only the provided evidence. '
        'Return one reviewable candidate through the structured schema. '
        f'The item_type must be {expected_item_type}.'
    )
    return LangChainInvocationPayload(
        messages=(
            ('system', system_prompt),
            (
                'user',
                render_memory_extraction_prompt(
                    packet,
                    expected_item_type=expected_item_type,
                    task_name=task_name,
                    max_input_chars=max_input_chars,
                ),
            ),
        ),
        structured_output_schema=StructuredMemoryExtractionOutput,
    )


def _coerce_structured_output(output: Any) -> StructuredMemoryExtractionOutput:
    if isinstance(output, StructuredMemoryExtractionOutput):
        return output
    if isinstance(output, dict):
        return StructuredMemoryExtractionOutput.model_validate(output)
    if hasattr(output, 'model_dump'):
        return StructuredMemoryExtractionOutput.model_validate(output.model_dump())
    return StructuredMemoryExtractionOutput.model_validate(output)
