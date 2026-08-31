from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.app.agent_runtime.auto_review_input_safety import (
    scan_auto_review_plaintext,
)
from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
from backend.app.agent_runtime.provider_usage import StrictChatUsageParser
from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
from backend.app.agent_runtime.rag_v2_contracts import ProviderDispatchPermit
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
    ANSWER_PROMPT_RENDERER_STATIC,
    RagAnswerOutputValidator,
    ValidatedAnswerBlocks,
    assert_answer_contract_registry_ready,
)
from backend.app.rag.evidence_projection import PreparedModelInfluenceObservation
from backend.app.rag.retrieval import (
    AnswerGenerationCostInput,
    EvidenceSlot,
    PreparedPaidCallBudget,
    StrictProviderUsage,
    is_lower_hex_64,
)


@dataclass(frozen=True, slots=True)
class PreparedAnswerInvocation:
    messages: tuple[tuple[Literal['system', 'user'], str], ...]
    evidence_slots: tuple[EvidenceSlot, ...]
    model_influence: tuple[PreparedModelInfluenceObservation, ...]
    answer_question_hmac: str
    retrieval_query_hmac: str
    rendered_input_hmac: str
    generation_estimator_input_hmac: str
    encoded_input_tokens: int
    framed_input_tokens: int
    reserved_cost_usd: Decimal
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str
    budget: PreparedPaidCallBudget


@dataclass(frozen=True, slots=True)
class ProviderAnswerEnvelope:
    raw_message: object
    returned_model: str
    returned_object: str
    returned_service_tier: str
    usage: StrictProviderUsage
    latency_ms: int


class RagAnswerModelBoundaryError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__('RAG answer model is unavailable')


def render_answer_messages(
    *,
    question: str,
    slots: tuple[EvidenceSlot, ...],
) -> tuple[tuple[str, str], ...]:
    assert_answer_contract_registry_ready()
    if type(question) is not str or type(slots) is not tuple:
        raise ValueError('RAG answer input is invalid')
    question_json = _compact_json({'question': question})
    evidence_json = _compact_json([
        {
            'slot_id': slot.slot_id,
            'support_tier': slot.support_mode,
            'content': slot.evidence.model_content,
        }
        for slot in slots
    ])
    user_content = f'QUESTION_JSON\n{question_json}\nEVIDENCE_JSON\n{evidence_json}'
    system_content = ANSWER_PROMPT_RENDERER_STATIC['system_content']
    if type(system_content) is not str:
        raise ValueError('RAG answer renderer is unavailable')
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        MessagesPlaceholder('untrusted_data_message'),
    ])
    rendered = prompt.invoke({
        'untrusted_data_message': [HumanMessage(content=user_content)],
    }).to_messages()
    if (
        len(rendered) != 2
        or type(rendered[0]) is not SystemMessage
        or type(rendered[1]) is not HumanMessage
        or type(rendered[0].content) is not str
        or type(rendered[1].content) is not str
    ):
        raise ValueError('RAG answer renderer is unavailable')
    return (('system', rendered[0].content), ('user', rendered[1].content))


class StructuredRagAnswerModel:
    def __init__(self, *, routed_model: RoutedRagAnswerModel, cost_policy: object) -> None:
        if (
            type(routed_model) is not RoutedRagAnswerModel
            or routed_model.provider != 'openai'
            or routed_model.model_name != 'gpt-5.4-mini-2026-03-17'
            or not is_lower_hex_64(routed_model.model_config_snapshot_hmac)
        ):
            raise RagAnswerModelBoundaryError('model_unavailable')
        try:
            if (
                cost_policy.answer_model_config_snapshot_hmac
                != routed_model.model_config_snapshot_hmac
            ):
                raise ValueError
            policy_hmac = cost_policy.authorized_policy_snapshot_hmac(
                'answer_generation'
            )
            if not is_lower_hex_64(policy_hmac):
                raise ValueError
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None
        self._route = routed_model
        self._cost_policy = cost_policy
        self._policy_hmac = policy_hmac
        self._validator = RagAnswerOutputValidator(
            signer=cost_policy.sign_answer_artifact
        )

    def prepare(
        self,
        *,
        question: str,
        slots: tuple[EvidenceSlot, ...],
        model_influence: tuple[PreparedModelInfluenceObservation, ...],
        answer_question_hmac: str,
        retrieval_query_hmac: str,
    ) -> PreparedAnswerInvocation:
        try:
            if (
                type(question) is not str
                or type(slots) is not tuple
                or not slots
                or len(slots) > 8
                or type(model_influence) is not tuple
                or not is_lower_hex_64(answer_question_hmac)
                or not is_lower_hex_64(retrieval_query_hmac)
            ):
                raise ValueError
            self._require_safe(question)
            working_slots = slots
            while working_slots:
                prepared_slots = _renumber_slots(working_slots)
                messages = render_answer_messages(
                    question=question,
                    slots=prepared_slots,
                )
                self._scan_rendered(messages, prepared_slots)
                messages_json = _canonical_json_bytes([
                    {'content': content, 'ordinal': index, 'role': role}
                    for index, (role, content) in enumerate(messages, start=1)
                ])
                try:
                    budget = self._cost_policy.prepare_answer_generation(
                        AnswerGenerationCostInput(
                            exact_messages_json=messages_json,
                            exact_response_schema_json=(
                                ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES
                            ),
                            model_config_snapshot_hmac=(
                                self._route.model_config_snapshot_hmac
                            ),
                        )
                    )
                except RagBudgetExceededError:
                    working_slots = working_slots[:-1]
                    continue
                if (
                    type(budget) is not PreparedPaidCallBudget
                    or budget.component != 'answer_generation'
                    or budget.cost_policy_snapshot_hmac != self._policy_hmac
                ):
                    raise ValueError
                rendered_hmac = self._cost_policy.sign_answer_artifact(
                    'rendered_input',
                    {
                        'answer_question_hmac': answer_question_hmac,
                        'messages': [list(message) for message in messages],
                        'retrieval_query_hmac': retrieval_query_hmac,
                    },
                )
                selected_influence = model_influence[: len(prepared_slots)]
                return PreparedAnswerInvocation(
                    messages=messages,
                    evidence_slots=prepared_slots,
                    model_influence=selected_influence,
                    answer_question_hmac=answer_question_hmac,
                    retrieval_query_hmac=retrieval_query_hmac,
                    rendered_input_hmac=rendered_hmac,
                    generation_estimator_input_hmac=budget.estimator_input_hmac,
                    encoded_input_tokens=max(0, budget.estimated_input_tokens - 528),
                    framed_input_tokens=budget.estimated_input_tokens,
                    reserved_cost_usd=budget.reserved_cost_usd,
                    model_config_snapshot_hmac=(
                        self._route.model_config_snapshot_hmac
                    ),
                    provider_policy_snapshot_hmac=self._policy_hmac,
                    budget=budget,
                )
            raise RagBudgetExceededError
        except RagBudgetExceededError:
            raise
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None

    def invoke_once(
        self,
        prepared: PreparedAnswerInvocation,
        permit: ProviderDispatchPermit,
    ) -> ProviderAnswerEnvelope:
        if type(prepared) is not PreparedAnswerInvocation:
            raise RagAnswerModelBoundaryError('model_unavailable')
        if (
            prepared.model_config_snapshot_hmac
            != self._route.model_config_snapshot_hmac
            or prepared.provider_policy_snapshot_hmac != self._policy_hmac
        ):
            raise RagAnswerModelBoundaryError('model_unavailable')
        provider_messages = [
            SystemMessage(content=content)
            if role == 'system'
            else HumanMessage(content=content)
            for role, content in prepared.messages
        ]
        started = time.monotonic_ns()
        try:
            permit.consume_at_dispatch()
            result = self._route.model.invoke(provider_messages)
        except Exception:
            raise RagAnswerModelBoundaryError('model_provider_failed') from None
        latency_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        try:
            if type(result) is not dict or set(result) != {
                'raw',
                'parsed',
                'parsing_error',
            }:
                raise ValueError
            raw = result['raw']
            usage = StrictChatUsageParser().parse_message(raw)
            actual = self._cost_policy.charge_actual('answer_generation', usage)
            metadata = raw.response_metadata
            if type(metadata) is not dict:
                raise ValueError
            returned_model = metadata.get('model')
            returned_object = metadata.get('object')
            returned_tier = metadata.get('service_tier')
            if (
                type(returned_model) is not str
                or returned_model != self._route.model_name
                or type(returned_object) is not str
                or returned_object != 'response'
                or type(returned_tier) is not str
                or returned_tier != 'default'
            ):
                raise RagAnswerModelBoundaryError(
                    'provider_response_identity_invalid'
                )
            if (
                usage.input_tokens > prepared.budget.estimated_input_tokens
                or usage.output_tokens > prepared.budget.maximum_output_tokens
                or actual > prepared.budget.reserved_cost_usd
                or actual > Decimal('0.012000')
            ):
                raise RagAnswerModelBoundaryError('provider_usage_overrun')
            return ProviderAnswerEnvelope(
                raw_message=result,
                returned_model=returned_model,
                returned_object=returned_object,
                returned_service_tier=returned_tier,
                usage=usage,
                latency_ms=latency_ms,
            )
        except RagAnswerModelBoundaryError:
            raise
        except Exception:
            raise RagAnswerModelBoundaryError('model_provider_failed') from None

    def validate(
        self,
        envelope: ProviderAnswerEnvelope,
        prepared: PreparedAnswerInvocation,
    ) -> ValidatedAnswerBlocks:
        try:
            if (
                type(envelope) is not ProviderAnswerEnvelope
                or type(prepared) is not PreparedAnswerInvocation
                or envelope.returned_model != self._route.model_name
                or envelope.returned_object != 'response'
                or envelope.returned_service_tier != 'default'
                or type(envelope.raw_message) is not dict
            ):
                raise ValueError
            result = envelope.raw_message
            if result['parsing_error'] is not None:
                raise ValueError
            return self._validator.validate(
                result['parsed'],
                slots=prepared.evidence_slots,
            )
        except Exception:
            raise RagAnswerModelBoundaryError('structured_output_invalid') from None

    @staticmethod
    def _require_safe(value: str) -> None:
        if not scan_auto_review_plaintext(value).allowed:
            raise ValueError

    def _scan_rendered(
        self,
        messages: tuple[tuple[str, str], ...],
        slots: tuple[EvidenceSlot, ...],
    ) -> None:
        for slot in slots:
            self._require_safe(slot.evidence.model_content)
        for _, content in messages:
            self._require_safe(content)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
        separators=(',', ':'),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8', errors='strict')


def _renumber_slots(slots: tuple[EvidenceSlot, ...]) -> tuple[EvidenceSlot, ...]:
    result = []
    for index, slot in enumerate(slots, start=1):
        slot_id = f'E{index}'
        try:
            result.append(replace(slot, slot_id=slot_id))
        except TypeError:
            if getattr(slot, 'slot_id', None) != slot_id:
                raise ValueError('RAG answer evidence slots are invalid') from None
            result.append(slot)
    return tuple(result)
