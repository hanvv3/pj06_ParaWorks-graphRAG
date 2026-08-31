from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.agents.rag_orchestrator_agent.v2_answer import (
    RagAnswerModelBoundaryError,
    StructuredRagAnswerModel,
    render_answer_messages,
)
from backend.app.rag.evidence_projection import PreparedModelInfluenceObservation
from backend.app.rag.retrieval import EvidenceSlot, PreparedPaidCallBudget
from backend.app.rag.serving_contracts import (
    RawChunkProvenance,
    RawServingVersionEnvelope,
    ServingEvidence,
    ServingEvidenceIdentity,
)


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

    def verify_answer_model_influence(self, slots, observations) -> None:
        assert len(slots) == len(observations)
        for ordinal, observation in enumerate(observations):
            if observation.observation_hmac != f'{ordinal + 81:x}'.zfill(64):
                raise ValueError('invalid test observation authority')

    def sign_answer_artifact(self, kind, payload):
        from backend.app.agent_runtime.fingerprints import keyed_fingerprint

        domains = {
            'rendered_input': ('rag-rendered-model-input-bytes:v1', 'rag-answer:v2'),
            'prepared_input': ('rag-prepared-answer-input:v1', 'rag-answer:v2'),
            'prepared_invocation': ('rag-prepared-answer-invocation:v1', 'rag-answer:v2'),
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

        self.configs: list[object] = []

    def invoke(self, messages: object, config: object = None) -> object:
        self.calls += 1
        self.configs.append(config)
        assert len(messages) == 2
        return self.result


class _FailingModel(_Model):
    def invoke(self, messages: object, config: object = None) -> object:
        self.calls += 1
        self.configs.append(config)
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


def _slot_and_observation(
    ordinal: int = 0,
    *,
    content: str = '확정 근거',
) -> tuple[EvidenceSlot, PreparedModelInfluenceObservation]:
    slot_id = f'E{ordinal + 1}'
    model_hmac = f'{ordinal + 1:x}'.zfill(64)
    citation_hmac = f'{ordinal + 17:x}'.zfill(64)
    version_hmac = f'{ordinal + 33:x}'.zfill(64)
    serving_hmac = f'{ordinal + 49:x}'.zfill(64)
    envelope = RawServingVersionEnvelope(
        serving_document_id=f'chunk:{ordinal + 1}',
        source_row_id=ordinal + 1,
        public_source_id=f'gmail:{ordinal + 1}',
        document_id=ordinal + 1,
        document_version_id=ordinal + 1,
        current_document_version_id=ordinal + 1,
        document_chunk_id=ordinal + 1,
        parser_run_id=ordinal + 1,
        external_revision=f'rev-{ordinal + 1}',
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=f'{ordinal + 65:x}'.zfill(64),
        parser_policy_version='parser-policy:v1',
        parser_version='parser:v1',
        chunk_policy_version='chunk:v1',
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        effective_permission='internal',
    )
    identity = ServingEvidenceIdentity(
        serving_document_id=f'chunk:{ordinal + 1}',
        serving_kind='raw_chunk',
        public_source_id=f'gmail:{ordinal + 1}',
        public_source_type='gmail',
        effective_permission='internal',
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        serving_version_fingerprint=version_hmac,
        version_envelope=envelope,
    )
    evidence = ServingEvidence(
        serving_document_id=identity.serving_document_id,
        serving_kind='raw_chunk',
        public_source_id=identity.public_source_id,
        public_source_type=identity.public_source_type,
        support_mode='source_observation',
        model_content=content,
        title=f'title-{ordinal + 1}',
        effective_permission='internal',
        serving_identity_hmac=serving_hmac,
        serving_version_fingerprint=version_hmac,
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        version_envelope=envelope,
        provenance=RawChunkProvenance(branch='raw_chunk', raw_version=envelope),
    )
    slot = EvidenceSlot(
        slot_id=slot_id,
        support_mode='source_observation',
        evidence=evidence,
        relevance_score=0.9,
        matched_terms=(),
    )
    observation = PreparedModelInfluenceObservation(
        ordinal=ordinal,
        slot_id=slot_id,
        support_mode='source_observation',
        lookup_identity=identity,
        effective_permission='internal',
        serving_identity_hmac=serving_hmac,
        serving_version_fingerprint=version_hmac,
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        approval_provenance_hmac=None,
        evidence_link_set_hmac=None,
        observation_hmac=f'{ordinal + 81:x}'.zfill(64),
    )
    return slot, observation


def _single_slot() -> tuple[EvidenceSlot, ...]:
    return (_slot_and_observation()[0],)


def _single_influence() -> tuple[PreparedModelInfluenceObservation, ...]:
    return (_slot_and_observation()[1],)


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
                    'support_mode': 'source_observation',
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
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='e' * 64,
    )
    permit = _Permit()

    envelope = answer_model.invoke_once(prepared, permit)
    validated = answer_model.validate(envelope, prepared)

    assert permit.calls == 1
    assert provider.calls == 1
    assert provider.configs == [{'callbacks': []}]
    assert len(cost.prepared) >= 1
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
        model_influence=_single_influence(),
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
            if not self.prepared:
                return super().prepare_answer_generation(value)
            if len(self.prepared) == 1:
                self.prepared.append(value)
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
    pairs = (_slot_and_observation(0, content='one'), _slot_and_observation(1, content='two'))
    slots = tuple(pair[0] for pair in pairs)
    influence = tuple(pair[1] for pair in pairs)

    prepared = answer_model.prepare_input(
        question='질문', slots=slots,
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    assert tuple(slot.slot_id for slot in prepared.evidence_slots) == ('E1',)
    assert '"content":"two"' not in prepared.messages[1][1]

    with pytest.raises(ValueError, match='unavailable'):
        answer_model.prepare(
            question='OPENAI_API_KEY=abcdefghijklmnopqrstuvwxyz123456',
            slots=slots,
            model_influence=influence,
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
        question='질문', slots=_single_slot(), model_influence=_single_influence(),
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
        question='질문', slots=_single_slot(), model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    with pytest.raises(ValueError, match='unavailable'):
        answer_model.invoke_once(prepared, _Permit())
    assert cost.charged == []


def test_static_frame_overflow_refuses_after_whole_tail_exhaustion() -> None:
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError

    class AlwaysOverBudget(_CostPolicy):
        def prepare_answer_generation(self, value):
            if not self.prepared:
                return super().prepare_answer_generation(value)
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
            question='질문', slots=_single_slot(), model_influence=_single_influence(),
            answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
        )
    assert len(cost.prepared) >= 2


def _valid_result(*, model: str = 'gpt-5.4-mini-2026-03-17', output: int = 10):
    raw = AIMessage(
        content='',
        usage_metadata={
            'input_tokens': 500,
            'output_tokens': output,
            'total_tokens': 500 + output,
        },
        response_metadata={
            'model': model,
            'object': 'response',
            'service_tier': 'default',
        },
    )
    return {
        'raw': raw,
        'parsed': {
            'answer_blocks': [
                {
                    'text': '답변',
                    'evidence_slot_ids': ['E1'],
                    'support_mode': 'source_observation',
                }
            ],
            'insufficient_evidence_reason': None,
        },
        'parsing_error': None,
    }


@pytest.mark.parametrize(
    'influence',
    (
        (),
        (replace(_single_influence()[0], ordinal=1),),
        (replace(_single_influence()[0], slot_id='E2'),),
        (replace(_single_influence()[0], support_mode='trusted_fact'),),
        (replace(_single_influence()[0], model_content_hmac='f' * 64),),
        (replace(_single_influence()[0], effective_permission='public'),),
    ),
)
def test_prepare_requires_exact_one_to_one_model_influence_authority(
    influence: tuple[PreparedModelInfluenceObservation, ...],
) -> None:
    answer_model, provider, _ = _answer_model(_valid_result())

    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.prepare(
            question='질문',
            slots=_single_slot(),
            model_influence=influence,
            answer_question_hmac='d' * 64,
            retrieval_query_hmac='e' * 64,
        )

    assert provider.calls == 0


def test_rendered_input_hmac_uses_exact_frozen_payload_and_domain() -> None:
    answer_model, _, cost = _answer_model(_valid_result())
    prepared = answer_model.prepare(
        question='질문',
        slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='e' * 64,
    )
    expected = cost.sign_answer_artifact(
        'rendered_input',
        {
            'messages': [
                {
                    'content_bytes': exact_utf8_bytes(content),
                    'ordinal': ordinal,
                    'role': role,
                }
                for ordinal, (role, content) in enumerate(
                    prepared.messages,
                    start=1,
                )
            ],
            'model_config_identity': 'rag-answer-model-config:v1',
            'output_schema_identity': 'rag-answer-blocks:v1',
        },
    )

    assert prepared.rendered_input_hmac == expected


@pytest.mark.parametrize(
    'mutate',
    (
        lambda value: replace(
            value,
            messages=(value.messages[0], ('user', value.messages[1][1] + 'x')),
        ),
        lambda value: replace(value, rendered_input_hmac='0' * 64),
        lambda value: replace(value, encoded_input_tokens=value.encoded_input_tokens + 1),
        lambda value: replace(
            value,
            budget=replace(value.budget, estimator_input_hmac='0' * 64),
        ),
    ),
)
def test_forged_prepared_invocation_never_reaches_dispatch(mutate) -> None:
    answer_model, provider, _ = _answer_model(_valid_result())
    valid = answer_model.prepare(
        question='질문', slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    permit = _Permit()

    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.invoke_once(mutate(valid), permit)

    assert permit.calls == 0
    assert provider.calls == 0


def test_forged_observation_hmac_never_reaches_dispatch() -> None:
    answer_model, provider, _ = _answer_model(_valid_result())
    valid = answer_model.prepare(
        question='질문', slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    forged = replace(
        valid,
        model_influence=(
            replace(valid.model_influence[0], observation_hmac='0' * 64),
        ),
    )
    permit = _Permit()

    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.invoke_once(forged, permit)

    assert permit.calls == 0
    assert provider.calls == 0


@pytest.mark.parametrize(
    'field',
    ('answer_question_hmac', 'retrieval_query_hmac'),
)
def test_prepared_question_and_retrieval_hmac_mutation_never_dispatches(
    field: str,
) -> None:
    answer_model, provider, _ = _answer_model(_valid_result())
    valid = answer_model.prepare(
        question='질문', slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    forged = replace(valid, **{field: 'f' * 64})
    permit = _Permit()

    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.invoke_once(forged, permit)

    assert permit.calls == 0
    assert provider.calls == 0


def test_usage_overrun_precedes_returned_identity_and_keeps_unclamped_actual() -> None:
    answer_model, _, cost = _answer_model(_valid_result(model='wrong', output=513))
    prepared = answer_model.prepare(
        question='질문', slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )

    with pytest.raises(RagAnswerModelBoundaryError) as exc_info:
        answer_model.invoke_once(prepared, _Permit())

    assert exc_info.value.code == 'provider_usage_overrun'
    assert len(cost.charged) == 1


def test_invoke_disables_callbacks_tracing_and_checks_provider_logging(
    monkeypatch,
) -> None:
    from backend.app.agents.rag_orchestrator_agent import v2_answer

    answer_model, provider, _ = _answer_model(_valid_result())
    prepared = answer_model.prepare(
        question='질문', slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    monkeypatch.setattr(
        v2_answer,
        'provider_logging_is_safe',
        lambda: False,
        raising=False,
    )
    permit = _Permit()

    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.invoke_once(prepared, permit)

    assert permit.calls == 0
    assert provider.calls == 0


def test_invoke_rechecks_provider_logging_after_dispatch(monkeypatch) -> None:
    from backend.app.agents.rag_orchestrator_agent import v2_answer

    answer_model, provider, cost = _answer_model(_valid_result())
    prepared = answer_model.prepare(
        question='질문', slots=_single_slot(),
        model_influence=_single_influence(),
        answer_question_hmac='d' * 64, retrieval_query_hmac='e' * 64,
    )
    checks = iter((True, False))
    monkeypatch.setattr(
        v2_answer,
        'provider_logging_is_safe',
        lambda: next(checks),
    )
    permit = _Permit()

    with pytest.raises(RagAnswerModelBoundaryError) as exc_info:
        answer_model.invoke_once(prepared, permit)

    assert exc_info.value.code == 'provider_safety_unavailable'
    assert permit.calls == 1
    assert provider.calls == 1
    assert provider.configs == [{'callbacks': []}]
    assert cost.charged == []


def test_static_registry_overflow_is_runtime_unavailable_at_construction() -> None:
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError

    class StaticOverflow(_CostPolicy):
        def prepare_answer_generation(self, value):
            raise RagBudgetExceededError

    route = RoutedRagAnswerModel(
        model=_Model(None), provider='openai',
        model_name='gpt-5.4-mini-2026-03-17',
        model_config_snapshot_hmac='a' * 64,
    )

    with pytest.raises(RagAnswerModelBoundaryError) as exc_info:
        StructuredRagAnswerModel(routed_model=route, cost_policy=StaticOverflow())

    assert exc_info.value.code == 'runtime_version_unavailable'


def test_prepare_input_filters_only_credential_evidence_then_binds_fresh_authority() -> None:
    credential = _slot_and_observation(
        0,
        content='OPENAI_API_KEY=abcdefghijklmnopqrstuvwxyz123456',
    )
    safe = _slot_and_observation(1, content='safe evidence')
    answer_model, provider, _ = _answer_model(_valid_result())

    prepared_input = answer_model.prepare_input(
        question='질문',
        slots=(credential[0], safe[0]),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='e' * 64,
    )

    assert prepared_input.__class__.__name__ == 'PreparedAnswerInput'
    assert tuple(slot.slot_id for slot in prepared_input.evidence_slots) == ('E1',)
    assert prepared_input.evidence_slots[0].evidence.model_content == 'safe evidence'
    assert not hasattr(prepared_input, 'model_influence')

    fresh = replace(
        safe[1],
        ordinal=0,
        slot_id='E1',
        observation_hmac=f'{81:x}'.zfill(64),
    )
    invocation = answer_model.bind_influence(
        prepared_input,
        model_influence=(fresh,),
    )
    assert invocation.model_influence == (fresh,)
    assert invocation.model_influence[0] is fresh
    assert provider.calls == 0


def test_prepare_input_safe_empty_is_explicit_no_generation_and_cannot_invoke() -> None:
    credential = _slot_and_observation(
        content='OPENAI_API_KEY=abcdefghijklmnopqrstuvwxyz123456',
    )[0]
    answer_model, provider, _ = _answer_model(_valid_result())

    outcome = answer_model.prepare_input(
        question='질문',
        slots=(credential,),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='e' * 64,
    )

    assert outcome.__class__.__name__ == 'AnswerPreparationNoGeneration'
    assert outcome.outcome == 'safety_filter_empty'
    permit = _Permit()
    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.invoke_once(outcome, permit)
    assert permit.calls == 0
    assert provider.calls == 0


def test_compat_prepare_refuses_when_tail_drop_would_stale_supplied_influence() -> None:
    from backend.app.agent_runtime.rag_cost_policy import RagBudgetExceededError

    class RequestTailDrop(_CostPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def prepare_answer_generation(self, value):
            self.calls += 1
            if self.calls == 2:
                raise RagBudgetExceededError
            return super().prepare_answer_generation(value)

    pairs = (_slot_and_observation(0), _slot_and_observation(1))
    route = RoutedRagAnswerModel(
        model=_Model(None), provider='openai',
        model_name='gpt-5.4-mini-2026-03-17',
        model_config_snapshot_hmac='a' * 64,
    )
    answer_model = StructuredRagAnswerModel(
        routed_model=route,
        cost_policy=RequestTailDrop(),
    )

    with pytest.raises(RagAnswerModelBoundaryError, match='unavailable'):
        answer_model.prepare(
            question='질문',
            slots=tuple(pair[0] for pair in pairs),
            model_influence=tuple(pair[1] for pair in pairs),
            answer_question_hmac='d' * 64,
            retrieval_query_hmac='e' * 64,
        )
