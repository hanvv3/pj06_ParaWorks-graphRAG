from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    StructuredRagAnswerModel,
    render_answer_messages,
)
from backend.app.rag.retrieval import PreparedPaidCallBudget


def test_renderer_uses_exact_langchain_two_message_boundary_and_json_escaping() -> None:
    slots = (
        SimpleNamespace(
            slot_id='E1',
            support_mode='trusted_fact',
            evidence=SimpleNamespace(model_content='내용 "인용"\n둘째 줄'),
        ),
        SimpleNamespace(
            slot_id='E2',
            support_mode='source_observation',
            evidence=SimpleNamespace(model_content='관찰'),
        ),
    )

    messages = render_answer_messages(question='질문 {그대로}', slots=slots)

    assert tuple(role for role, _ in messages) == ('system', 'user')
    assert messages[1][1] == (
        'QUESTION_JSON\n{"question":"질문 {그대로}"}\nEVIDENCE_JSON\n'
        '[{"slot_id":"E1","support_tier":"trusted_fact","content":"내용 '
        '\\"인용\\"\\n둘째 줄"},{"slot_id":"E2","support_tier":'
        '"source_observation","content":"관찰"}]'
    )


class _Permit:
    def __init__(self) -> None:
        self.calls = 0

    def consume_at_dispatch(self) -> None:
        self.calls += 1


class _CostPolicy:
    answer_model_config_snapshot_hmac = 'a' * 64

    def __init__(self) -> None:
        self.prepared = []
        self.charged = []

    def authorized_policy_snapshot_hmac(self, component: str) -> str:
        assert component == 'answer_generation'
        return 'b' * 64

    def prepare_answer_generation(self, value):
        self.prepared.append(value)
        return PreparedPaidCallBudget(
            component='answer_generation',
            estimated_input_tokens=600,
            maximum_output_tokens=512,
            reserved_cost_usd=Decimal('0.002800'),
            cost_policy_snapshot_hmac='b' * 64,
            estimator_input_hmac='c' * 64,
        )

    def charge_actual(self, component, usage):
        self.charged.append((component, usage))
        return Decimal('0.000010')

    def sign_answer_artifact(self, kind, payload):
        from backend.app.agent_runtime.fingerprints import keyed_fingerprint

        domains = {
            'rendered_input': ('rag-rendered-answer-input:v1', 'rag-answer:v2'),
            'block_text': ('rag-answer-block-text-bytes:v1', 'rag-answer:v2'),
            'block_result': ('rag-answer-block-result:v1', 'rag-answer-block-confidence:v1'),
            'audit_set': ('rag-answer-block-audit-set:v1', 'rag-answer-block-confidence:v1'),
            'assembled': ('rag-assembled-answer-bytes:v1', 'rag-answer-block-joiner:v1'),
        }
        schema, policy = domains[kind]
        return keyed_fingerprint(
            payload, secret=b'test-secret',
            schema_version=schema, policy_version=policy,
        )


class _Model:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def invoke(self, messages: object) -> object:
        self.calls += 1
        assert len(messages) == 2
        return self.result


class _FailingModel(_Model):
    def invoke(self, messages: object) -> object:
        self.calls += 1
        raise RuntimeError('provider-secret')


def _answer_model(result: object):
    provider = _Model(result)
    route = RoutedRagAnswerModel(
        model=provider,
        provider='openai',
        model_name='gpt-5.4-mini-2026-03-17',
        model_config_snapshot_hmac='a' * 64,
    )
    cost = _CostPolicy()
    return StructuredRagAnswerModel(routed_model=route, cost_policy=cost), provider, cost


def _single_slot():
    return (
        SimpleNamespace(
            slot_id='E1',
            support_mode='trusted_fact',
            evidence=SimpleNamespace(model_content='확정 근거'),
        ),
    )


def test_prepare_invoke_once_and_validate_strict_envelope() -> None:
    raw = AIMessage(
        content='',
        usage_metadata={'input_tokens': 500, 'output_tokens': 10, 'total_tokens': 510},
        response_metadata={
            'model': 'gpt-5.4-mini-2026-03-17',
            'object': 'response',
            'service_tier': 'default',
        },
    )
    result = {
        'raw': raw,
        'parsed': {
            'answer_blocks': [
                {
                    'text': '답변',
                    'evidence_slot_ids': ['E1'],
                    'support_mode': 'trusted_fact',
                }
            ],
            'insufficient_evidence_reason': None,
        },
        'parsing_error': None,
    }
    answer_model, provider, cost = _answer_model(result)
    prepared = answer_model.prepare(
        question='질문',
        slots=_single_slot(),
        model_influence=(),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='e' * 64,
    )
    permit = _Permit()

    envelope = answer_model.invoke_once(prepared, permit)
    validated = answer_model.validate(envelope, prepared)

    assert permit.calls == 1
    assert provider.calls == 1
    assert len(cost.prepared) == 1
    assert len(cost.charged) == 1
    assert envelope.returned_object == 'response'
    assert validated.assembled_answer == '답변'


def test_invoke_charges_valid_usage_before_rejecting_identity() -> None:
    raw = AIMessage(
        content='',
        usage_metadata={'input_tokens': 500, 'output_tokens': 10, 'total_tokens': 510},
        response_metadata={
            'model': 'wrong-model',
            'object': 'response',
            'service_tier': 'default',
        },
    )
    answer_model, provider, cost = _answer_model(
        {'raw': raw, 'parsed': {}, 'parsing_error': None}
    )
    prepared = answer_model.prepare(
        question='질문',
        slots=_single_slot(),
        model_influence=(),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='e' * 64,
    )

    with pytest.raises(ValueError, match='unavailable'):
        answer_model.invoke_once(prepared, _Permit())

    assert provider.calls == 1
    assert len(cost.charged) == 1


def test_prepare_rejects_credential_before_provider_and_tail_drops_whole_slot() -> None:
    class TailDropCost(_CostPolicy):
        def prepare_answer_generation(self, value):
            self.prepared.append(value)
            if len(self.prepared) == 1:
                from backend.app.agent_runtime.rag_cost_policy import (
                    RagBudgetExceededError,
                )

                raise RagBudgetExceededError
            return super().prepare_answer_generation(value)

    route = RoutedRagAnswerModel(
        model=_Model(None),
        provider='openai',
        model_name='gpt-5.4-mini-2026-03-17',
        model_config_snapshot_hmac='a' * 64,
    )
    cost = TailDropCost()
    answer_model = StructuredRagAnswerModel(routed_model=route, cost_policy=cost)
    slots = (
        SimpleNamespace(slot_id='E1', support_mode='trusted_fact', evidence=SimpleNamespace(model_content='one')),
        SimpleNamespace(slot_id='E2', support_mode='trusted_fact', evidence=SimpleNamespace(model_content='two')),
    )

    prepared = answer_model.prepare(
        question='질문', slots=slots, model_influence=(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    assert tuple(slot.slot_id for slot in prepared.evidence_slots) == ('E1',)
    assert '"content":"two"' not in prepared.messages[1][1]

    with pytest.raises(ValueError, match='unavailable'):
        answer_model.prepare(
            question='OPENAI_API_KEY=abcdefghijklmnopqrstuvwxyz123456',
            slots=slots,
            model_influence=(),
            answer_question_hmac='d' * 64,
            retrieval_query_hmac='e' * 64,
        )


def test_response_less_and_malformed_usage_are_sanitized_without_actual_charge() -> None:
    route = RoutedRagAnswerModel(
        model=_FailingModel(None), provider='openai',
        model_name='gpt-5.4-mini-2026-03-17',
        model_config_snapshot_hmac='a' * 64,
    )
    cost = _CostPolicy()
    answer_model = StructuredRagAnswerModel(routed_model=route, cost_policy=cost)
    prepared = answer_model.prepare(
        question='질문', slots=_single_slot(), model_influence=(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    permit = _Permit()
    with pytest.raises(ValueError, match='unavailable') as exc_info:
        answer_model.invoke_once(prepared, permit)
    assert permit.calls == 1
    assert cost.charged == []
    assert exc_info.value.__cause__ is None

    malformed = SimpleNamespace(
        usage_metadata={'input_tokens': True, 'output_tokens': 1, 'total_tokens': 2},
        response_metadata={'model': 'gpt-5.4-mini-2026-03-17', 'object': 'response', 'service_tier': 'default'},
    )
    answer_model, _, cost = _answer_model(
        {'raw': malformed, 'parsed': {}, 'parsing_error': None}
    )
    prepared = answer_model.prepare(
        question='질문', slots=_single_slot(), model_influence=(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    with pytest.raises(ValueError, match='unavailable'):
        answer_model.invoke_once(prepared, _Permit())
    assert cost.charged == []


def test_static_frame_overflow_refuses_after_whole_tail_exhaustion() -> None:
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError

    class AlwaysOverBudget(_CostPolicy):
        def prepare_answer_generation(self, value):
            self.prepared.append(value)
            raise RagBudgetExceededError

    cost = AlwaysOverBudget()
    route = RoutedRagAnswerModel(
        model=_Model(None), provider='openai',
        model_name='gpt-5.4-mini-2026-03-17',
        model_config_snapshot_hmac='a' * 64,
    )
    answer_model = StructuredRagAnswerModel(routed_model=route, cost_policy=cost)

    with pytest.raises(RagBudgetExceededError):
        answer_model.prepare(
            question='질문', slots=_single_slot(), model_influence=(),
            answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
        )
    assert len(cost.prepared) == 1
