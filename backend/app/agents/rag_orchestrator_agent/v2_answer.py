from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import tracing_context

from backend.app.agent_runtime.auto_review_input_safety import (
    provider_logging_is_safe,
    scan_auto_review_plaintext,
)
from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
from backend.app.agent_runtime.provider_usage import StrictChatUsageParser
from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError
from backend.app.agent_runtime.rag_v2_contracts import ProviderDispatchPermit
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
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
from backend.app.rag.serving_contracts import ServingEvidence, ServingEvidenceIdentity

_MODEL_CONFIG_IDENTITY = 'rag-answer-model-config:v1'
_OUTPUT_SCHEMA_IDENTITY = 'rag-answer-blocks:v1'
_ANSWER_FRAME_TOKEN_OVERHEAD = 528


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
    prepared_input_hmac: str
    prepared_invocation_hmac: str
    budget: PreparedPaidCallBudget


@dataclass(frozen=True, slots=True)
class PreparedAnswerInput:
    messages: tuple[tuple[Literal['system', 'user'], str], ...]
    evidence_slots: tuple[EvidenceSlot, ...]
    answer_question_hmac: str
    retrieval_query_hmac: str
    rendered_input_hmac: str
    generation_estimator_input_hmac: str
    encoded_input_tokens: int
    framed_input_tokens: int
    reserved_cost_usd: Decimal
    model_config_snapshot_hmac: str
    provider_policy_snapshot_hmac: str
    prepared_input_hmac: str
    budget: PreparedPaidCallBudget


@dataclass(frozen=True, slots=True)
class AnswerPreparationNoGeneration:
    outcome: Literal['safety_filter_empty'] = 'safety_filter_empty'


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
        try:
            static_messages = render_answer_messages(question='', slots=())
            self._scan_messages(static_messages)
            self._prepare_budget(static_messages)
        except Exception:
            raise RagAnswerModelBoundaryError(
                'runtime_version_unavailable'
            ) from None

    def prepare_input(
        self,
        *,
        question: str,
        slots: tuple[EvidenceSlot, ...],
        answer_question_hmac: str,
        retrieval_query_hmac: str,
    ) -> PreparedAnswerInput | AnswerPreparationNoGeneration:
        try:
            if (
                type(question) is not str
                or type(slots) is not tuple
                or not 1 <= len(slots) <= 8
                or not is_lower_hex_64(answer_question_hmac)
                or not is_lower_hex_64(retrieval_query_hmac)
            ):
                raise ValueError
            self._require_safe(question)
            _validate_slots(slots)
            safe_slots = tuple(
                slot
                for slot in slots
                if scan_auto_review_plaintext(slot.evidence.model_content).allowed
            )
            if not safe_slots:
                return AnswerPreparationNoGeneration()
            working_slots = safe_slots
            while working_slots:
                prepared_slots = _renumber_slots(working_slots)
                messages = render_answer_messages(
                    question=question,
                    slots=prepared_slots,
                )
                self._scan_messages(messages)
                try:
                    budget = self._prepare_budget(messages)
                except RagBudgetExceededError:
                    working_slots = working_slots[:-1]
                    continue
                rendered_hmac = self._rendered_input_hmac(messages)
                draft = PreparedAnswerInput(
                    messages=messages,
                    evidence_slots=prepared_slots,
                    answer_question_hmac=answer_question_hmac,
                    retrieval_query_hmac=retrieval_query_hmac,
                    rendered_input_hmac=rendered_hmac,
                    generation_estimator_input_hmac=budget.estimator_input_hmac,
                    encoded_input_tokens=(
                        budget.estimated_input_tokens
                        - _ANSWER_FRAME_TOKEN_OVERHEAD
                    ),
                    framed_input_tokens=budget.estimated_input_tokens,
                    reserved_cost_usd=budget.reserved_cost_usd,
                    model_config_snapshot_hmac=(
                        self._route.model_config_snapshot_hmac
                    ),
                    provider_policy_snapshot_hmac=self._policy_hmac,
                    prepared_input_hmac='0' * 64,
                    budget=budget,
                )
                return replace(
                    draft,
                    prepared_input_hmac=self._cost_policy.sign_answer_artifact(
                        'prepared_input',
                        _prepared_input_authority_payload(draft),
                    ),
                )
            raise RagBudgetExceededError
        except RagBudgetExceededError:
            raise
        except RagAnswerModelBoundaryError:
            raise
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None

    def bind_influence(
        self,
        prepared_input: PreparedAnswerInput,
        *,
        model_influence: tuple[PreparedModelInfluenceObservation, ...],
    ) -> PreparedAnswerInvocation:
        try:
            self._validate_prepared_input(prepared_input)
            _validate_influence_alignment(
                prepared_input.evidence_slots,
                model_influence,
            )
            self._cost_policy.verify_answer_model_influence(
                prepared_input.evidence_slots,
                model_influence,
            )
            draft = PreparedAnswerInvocation(
                messages=prepared_input.messages,
                evidence_slots=prepared_input.evidence_slots,
                model_influence=model_influence,
                answer_question_hmac=prepared_input.answer_question_hmac,
                retrieval_query_hmac=prepared_input.retrieval_query_hmac,
                rendered_input_hmac=prepared_input.rendered_input_hmac,
                generation_estimator_input_hmac=(
                    prepared_input.generation_estimator_input_hmac
                ),
                encoded_input_tokens=prepared_input.encoded_input_tokens,
                framed_input_tokens=prepared_input.framed_input_tokens,
                reserved_cost_usd=prepared_input.reserved_cost_usd,
                model_config_snapshot_hmac=(
                    prepared_input.model_config_snapshot_hmac
                ),
                provider_policy_snapshot_hmac=(
                    prepared_input.provider_policy_snapshot_hmac
                ),
                prepared_input_hmac=prepared_input.prepared_input_hmac,
                prepared_invocation_hmac='0' * 64,
                budget=prepared_input.budget,
            )
            return replace(
                draft,
                prepared_invocation_hmac=self._cost_policy.sign_answer_artifact(
                    'prepared_invocation',
                    _prepared_invocation_authority_payload(draft),
                ),
            )
        except RagAnswerModelBoundaryError:
            raise
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None

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
            prepared_input = self.prepare_input(
                question=question,
                slots=slots,
                answer_question_hmac=answer_question_hmac,
                retrieval_query_hmac=retrieval_query_hmac,
            )
            if type(prepared_input) is not PreparedAnswerInput:
                raise ValueError
            if not _same_slot_sequence(slots, prepared_input.evidence_slots):
                raise ValueError
            return self.bind_influence(
                prepared_input,
                model_influence=model_influence,
            )
        except RagBudgetExceededError:
            raise
        except RagAnswerModelBoundaryError:
            raise
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None

    def invoke_once(
        self,
        prepared: PreparedAnswerInvocation,
        permit: ProviderDispatchPermit,
    ) -> ProviderAnswerEnvelope:
        try:
            self._validate_prepared(prepared)
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None
        if not provider_logging_is_safe():
            raise RagAnswerModelBoundaryError('provider_safety_unavailable')
        try:
            provider_messages = [
                SystemMessage(content=content)
                if role == 'system'
                else HumanMessage(content=content)
                for role, content in prepared.messages
            ]
        except Exception:
            raise RagAnswerModelBoundaryError('model_unavailable') from None
        started = time.monotonic_ns()
        try:
            permit.consume_at_dispatch()
            with tracing_context(enabled=False):
                result = self._route.model.invoke(
                    provider_messages,
                    config={'callbacks': []},
                )
        except Exception:
            raise RagAnswerModelBoundaryError('model_provider_failed') from None
        latency_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        if not provider_logging_is_safe():
            raise RagAnswerModelBoundaryError('provider_safety_unavailable')
        try:
            if type(result) is not dict or not _has_exact_keys(
                result,
                ('raw', 'parsed', 'parsing_error'),
            ):
                raise ValueError
            raw = result['raw']
            usage = StrictChatUsageParser().parse_message(raw)
            actual = self._cost_policy.charge_actual('answer_generation', usage)
            if (
                usage.input_tokens > prepared.budget.estimated_input_tokens
                or usage.output_tokens > prepared.budget.maximum_output_tokens
                or actual > prepared.budget.reserved_cost_usd
                or actual > Decimal('0.012000')
            ):
                raise RagAnswerModelBoundaryError('provider_usage_overrun')
            metadata = raw.response_metadata
            if type(metadata) is not dict or any(
                type(key) is not str for key in metadata
            ):
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
        if type(value) is not str or not scan_auto_review_plaintext(value).allowed:
            raise ValueError

    def _scan_messages(
        self,
        messages: tuple[tuple[str, str], ...],
    ) -> None:
        for _, content in messages:
            self._require_safe(content)

    def _prepare_budget(
        self,
        messages: tuple[tuple[Literal['system', 'user'], str], ...],
    ) -> PreparedPaidCallBudget:
        messages_json = _canonical_json_bytes([
            {'content': content, 'ordinal': index, 'role': role}
            for index, (role, content) in enumerate(messages, start=1)
        ])
        budget = self._cost_policy.prepare_answer_generation(
            AnswerGenerationCostInput(
                exact_messages_json=messages_json,
                exact_response_schema_json=ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
                model_config_snapshot_hmac=(
                    self._route.model_config_snapshot_hmac
                ),
            )
        )
        _validate_budget(budget, policy_hmac=self._policy_hmac)
        return budget

    def _rendered_input_hmac(
        self,
        messages: tuple[tuple[Literal['system', 'user'], str], ...],
    ) -> str:
        return self._cost_policy.sign_answer_artifact(
            'rendered_input',
            {
                'messages': [
                    {
                        'content_bytes': exact_utf8_bytes(content),
                        'ordinal': ordinal,
                        'role': role,
                    }
                    for ordinal, (role, content) in enumerate(
                        messages,
                        start=1,
                    )
                ],
                'model_config_identity': _MODEL_CONFIG_IDENTITY,
                'output_schema_identity': _OUTPUT_SCHEMA_IDENTITY,
            },
        )

    def _validate_prepared_input(self, value: PreparedAnswerInput) -> None:
        if type(value) is not PreparedAnswerInput:
            raise ValueError
        _validate_messages(value.messages)
        _validate_slots(value.evidence_slots)
        if (
            not is_lower_hex_64(value.answer_question_hmac)
            or not is_lower_hex_64(value.retrieval_query_hmac)
            or not is_lower_hex_64(value.rendered_input_hmac)
            or not is_lower_hex_64(value.generation_estimator_input_hmac)
            or type(value.encoded_input_tokens) is not int
            or value.encoded_input_tokens < 0
            or type(value.framed_input_tokens) is not int
            or value.framed_input_tokens < 0
            or type(value.reserved_cost_usd) is not Decimal
            or not value.reserved_cost_usd.is_finite()
            or value.reserved_cost_usd < 0
            or not is_lower_hex_64(value.model_config_snapshot_hmac)
            or value.model_config_snapshot_hmac
            != self._route.model_config_snapshot_hmac
            or not is_lower_hex_64(value.provider_policy_snapshot_hmac)
            or value.provider_policy_snapshot_hmac != self._policy_hmac
            or not is_lower_hex_64(value.prepared_input_hmac)
            or self._cost_policy.authorized_policy_snapshot_hmac(
                'answer_generation'
            )
            != self._policy_hmac
        ):
            raise ValueError
        question = _question_from_messages(value.messages)
        expected_messages = render_answer_messages(
            question=question,
            slots=value.evidence_slots,
        )
        if not _same_messages(value.messages, expected_messages):
            raise ValueError
        self._require_safe(question)
        for slot in value.evidence_slots:
            self._require_safe(slot.evidence.model_content)
        self._scan_messages(value.messages)
        if value.rendered_input_hmac != self._rendered_input_hmac(value.messages):
            raise ValueError
        recomputed = self._prepare_budget(value.messages)
        if not _same_budget(value.budget, recomputed):
            raise ValueError
        if (
            value.generation_estimator_input_hmac
            != recomputed.estimator_input_hmac
            or value.encoded_input_tokens
            != recomputed.estimated_input_tokens - _ANSWER_FRAME_TOKEN_OVERHEAD
            or value.framed_input_tokens != recomputed.estimated_input_tokens
            or value.reserved_cost_usd != recomputed.reserved_cost_usd
            or value.prepared_input_hmac
            != self._cost_policy.sign_answer_artifact(
                'prepared_input',
                _prepared_input_authority_payload(value),
            )
        ):
            raise ValueError

    def _validate_prepared(self, value: PreparedAnswerInvocation) -> None:
        if type(value) is not PreparedAnswerInvocation:
            raise ValueError
        self._validate_prepared_input(PreparedAnswerInput(
            messages=value.messages,
            evidence_slots=value.evidence_slots,
            answer_question_hmac=value.answer_question_hmac,
            retrieval_query_hmac=value.retrieval_query_hmac,
            rendered_input_hmac=value.rendered_input_hmac,
            generation_estimator_input_hmac=(
                value.generation_estimator_input_hmac
            ),
            encoded_input_tokens=value.encoded_input_tokens,
            framed_input_tokens=value.framed_input_tokens,
            reserved_cost_usd=value.reserved_cost_usd,
            model_config_snapshot_hmac=value.model_config_snapshot_hmac,
            provider_policy_snapshot_hmac=value.provider_policy_snapshot_hmac,
            prepared_input_hmac=value.prepared_input_hmac,
            budget=value.budget,
        ))
        _validate_influence_alignment(value.evidence_slots, value.model_influence)
        self._cost_policy.verify_answer_model_influence(
            value.evidence_slots,
            value.model_influence,
        )
        if (
            not is_lower_hex_64(value.prepared_invocation_hmac)
            or value.prepared_invocation_hmac
            != self._cost_policy.sign_answer_artifact(
                'prepared_invocation',
                _prepared_invocation_authority_payload(value),
            )
        ):
            raise ValueError


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


def _prepared_input_authority_payload(
    value: PreparedAnswerInput | PreparedAnswerInvocation,
) -> dict[str, object]:
    return {
        'answer_question_hmac': value.answer_question_hmac,
        'encoded_input_tokens': value.encoded_input_tokens,
        'evidence_slots': [
            {
                'canonical_citation_projection_hmac': (
                    slot.evidence.canonical_citation_projection_hmac
                ),
                'effective_permission': slot.evidence.effective_permission,
                'model_content_hmac': slot.evidence.model_content_hmac,
                'public_source_id': slot.evidence.public_source_id,
                'public_source_type': slot.evidence.public_source_type,
                'serving_document_id': slot.evidence.serving_document_id,
                'serving_identity_hmac': slot.evidence.serving_identity_hmac,
                'serving_kind': slot.evidence.serving_kind,
                'serving_version_fingerprint': (
                    slot.evidence.serving_version_fingerprint
                ),
                'slot_id': slot.slot_id,
                'support_mode': slot.support_mode,
            }
            for slot in value.evidence_slots
        ],
        'framed_input_tokens': value.framed_input_tokens,
        'generation_estimator_input_hmac': (
            value.generation_estimator_input_hmac
        ),
        'maximum_output_tokens': value.budget.maximum_output_tokens,
        'model_config_snapshot_hmac': value.model_config_snapshot_hmac,
        'provider_policy_snapshot_hmac': value.provider_policy_snapshot_hmac,
        'rendered_input_hmac': value.rendered_input_hmac,
        'reserved_cost_usd': format(value.reserved_cost_usd, 'f'),
        'retrieval_query_hmac': value.retrieval_query_hmac,
    }


def _prepared_invocation_authority_payload(
    value: PreparedAnswerInvocation,
) -> dict[str, object]:
    return {
        **_prepared_input_authority_payload(value),
        'model_influence': [
            {
                'observation_hmac': observation.observation_hmac,
                'ordinal': observation.ordinal,
                'slot_id': observation.slot_id,
            }
            for observation in value.model_influence
        ],
        'prepared_input_hmac': value.prepared_input_hmac,
    }


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


def _validate_budget(value: object, *, policy_hmac: str) -> None:
    if (
        type(value) is not PreparedPaidCallBudget
        or type(value.component) is not str
        or value.component != 'answer_generation'
        or type(value.estimated_input_tokens) is not int
        or value.estimated_input_tokens < _ANSWER_FRAME_TOKEN_OVERHEAD
        or type(value.maximum_output_tokens) is not int
        or value.maximum_output_tokens != 512
        or type(value.reserved_cost_usd) is not Decimal
        or not value.reserved_cost_usd.is_finite()
        or value.reserved_cost_usd < 0
        or value.cost_policy_snapshot_hmac != policy_hmac
        or not is_lower_hex_64(value.estimator_input_hmac)
    ):
        raise ValueError('RAG answer budget is invalid')


def _validate_messages(value: object) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError('RAG answer messages are invalid')
    for expected_role, message in zip(('system', 'user'), value, strict=True):
        if (
            type(message) is not tuple
            or len(message) != 2
            or type(message[0]) is not str
            or message[0] != expected_role
            or type(message[1]) is not str
        ):
            raise ValueError('RAG answer messages are invalid')


def _validate_slots(value: object) -> None:
    if type(value) is not tuple or not 1 <= len(value) <= 8:
        raise ValueError('RAG answer slots are invalid')
    for ordinal, slot in enumerate(value):
        if (
            type(slot) is not EvidenceSlot
            or type(slot.slot_id) is not str
            or slot.slot_id != f'E{ordinal + 1}'
            or type(slot.support_mode) is not str
            or slot.support_mode not in {'trusted_fact', 'source_observation'}
            or type(slot.evidence) is not ServingEvidence
            or type(slot.evidence.serving_document_id) is not str
            or type(slot.evidence.serving_kind) is not str
            or slot.evidence.serving_kind
            not in {'raw_chunk', 'trusted_knowledge'}
            or type(slot.evidence.public_source_id) is not str
            or type(slot.evidence.public_source_type) not in {str, type(None)}
            or type(slot.evidence.support_mode) is not str
            or slot.evidence.support_mode != slot.support_mode
            or type(slot.evidence.model_content) is not str
            or type(slot.evidence.title) is not str
            or type(slot.evidence.effective_permission) is not str
            or slot.evidence.effective_permission
            not in {'public', 'internal', 'restricted'}
            or type(slot.relevance_score) is not float
            or not math.isfinite(slot.relevance_score)
            or type(slot.matched_terms) is not tuple
            or any(type(term) is not str for term in slot.matched_terms)
        ):
            raise ValueError('RAG answer slots are invalid')
        for observed in (
            slot.evidence.serving_identity_hmac,
            slot.evidence.serving_version_fingerprint,
            slot.evidence.model_content_hmac,
            slot.evidence.canonical_citation_projection_hmac,
        ):
            if not is_lower_hex_64(observed):
                raise ValueError('RAG answer slot identity is invalid')


def _validate_influence_alignment(
    slots: tuple[EvidenceSlot, ...],
    observations: object,
) -> None:
    _validate_slots(slots)
    if (
        type(observations) is not tuple
        or len(observations) != len(slots)
        or not observations
    ):
        raise ValueError('RAG answer influence is invalid')
    for ordinal, (slot, observation) in enumerate(
        zip(slots, observations, strict=True)
    ):
        evidence = slot.evidence
        if (
            type(observation) is not PreparedModelInfluenceObservation
            or type(observation.ordinal) is not int
            or observation.ordinal != ordinal
            or type(observation.slot_id) is not str
            or observation.slot_id != slot.slot_id
            or type(observation.support_mode) is not str
            or observation.support_mode != slot.support_mode
            or type(observation.lookup_identity) is not ServingEvidenceIdentity
            or type(observation.effective_permission) is not str
            or observation.effective_permission != evidence.effective_permission
            or observation.serving_identity_hmac != evidence.serving_identity_hmac
            or observation.serving_version_fingerprint
            != evidence.serving_version_fingerprint
            or observation.model_content_hmac != evidence.model_content_hmac
            or observation.canonical_citation_projection_hmac
            != evidence.canonical_citation_projection_hmac
            or not is_lower_hex_64(observation.observation_hmac)
        ):
            raise ValueError('RAG answer influence is invalid')
        identity = observation.lookup_identity
        if (
            type(identity.serving_document_id) is not str
            or identity.serving_document_id != evidence.serving_document_id
            or type(identity.serving_kind) is not str
            or identity.serving_kind != evidence.serving_kind
            or type(identity.public_source_id) is not str
            or identity.public_source_id != evidence.public_source_id
            or type(identity.public_source_type) not in {str, type(None)}
            or identity.public_source_type != evidence.public_source_type
            or type(identity.effective_permission) is not str
            or identity.effective_permission != evidence.effective_permission
            or not is_lower_hex_64(identity.model_content_hmac)
            or identity.model_content_hmac != evidence.model_content_hmac
            or not is_lower_hex_64(
                identity.canonical_citation_projection_hmac
            )
            or identity.canonical_citation_projection_hmac
            != evidence.canonical_citation_projection_hmac
            or not is_lower_hex_64(identity.serving_version_fingerprint)
            or identity.serving_version_fingerprint
            != evidence.serving_version_fingerprint
            or type(observation.approval_provenance_hmac) not in {str, type(None)}
            or type(observation.evidence_link_set_hmac) not in {str, type(None)}
        ):
            raise ValueError('RAG answer influence identity is invalid')
        for optional_hmac in (
            observation.approval_provenance_hmac,
            observation.evidence_link_set_hmac,
        ):
            if optional_hmac is not None and not is_lower_hex_64(optional_hmac):
                raise ValueError('RAG answer influence identity is invalid')
        if evidence.serving_kind == 'raw_chunk' and (
            observation.approval_provenance_hmac is not None
            or observation.evidence_link_set_hmac is not None
        ):
            raise ValueError('RAG answer influence identity is invalid')


def _same_slot_sequence(
    left: tuple[EvidenceSlot, ...],
    right: tuple[EvidenceSlot, ...],
) -> bool:
    if type(left) is not tuple or type(right) is not tuple or len(left) != len(right):
        return False
    return all(
        type(first) is EvidenceSlot
        and type(second) is EvidenceSlot
        and first.slot_id == second.slot_id
        and first.support_mode == second.support_mode
        and first.evidence is second.evidence
        and first.relevance_score == second.relevance_score
        and first.matched_terms == second.matched_terms
        for first, second in zip(left, right, strict=True)
    )


def _same_messages(left: object, right: object) -> bool:
    _validate_messages(left)
    _validate_messages(right)
    return all(
        first[0] == second[0] and first[1] == second[1]
        for first, second in zip(left, right, strict=True)
    )


def _same_budget(left: object, right: object) -> bool:
    if type(left) is not PreparedPaidCallBudget or type(right) is not PreparedPaidCallBudget:
        return False
    return (
        left.component == right.component
        and left.estimated_input_tokens == right.estimated_input_tokens
        and left.maximum_output_tokens == right.maximum_output_tokens
        and left.reserved_cost_usd == right.reserved_cost_usd
        and left.cost_policy_snapshot_hmac == right.cost_policy_snapshot_hmac
        and left.estimator_input_hmac == right.estimator_input_hmac
    )


def _question_from_messages(
    messages: tuple[tuple[Literal['system', 'user'], str], ...],
) -> str:
    _validate_messages(messages)
    user_content = messages[1][1]
    prefix = 'QUESTION_JSON\n'
    separator = '\nEVIDENCE_JSON\n'
    if not user_content.startswith(prefix):
        raise ValueError('RAG answer user frame is invalid')
    question_json, found, _ = user_content.removeprefix(prefix).partition(separator)
    if not found:
        raise ValueError('RAG answer user frame is invalid')
    try:
        value = json.loads(
            question_json,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except Exception:
        raise ValueError('RAG answer question frame is invalid') from None
    if (
        type(value) is not dict
        or not _has_exact_keys(value, ('question',))
        or type(value['question']) is not str
        or _compact_json(value) != question_json
    ):
        raise ValueError('RAG answer question frame is invalid')
    return value['question']


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError
        result[key] = value
    return result


def _has_exact_keys(value: dict[object, object], expected: tuple[str, ...]) -> bool:
    keys = tuple(value.keys())
    if any(type(key) is not str for key in keys):
        return False
    return len(keys) == len(expected) and set(keys) == set(expected)
