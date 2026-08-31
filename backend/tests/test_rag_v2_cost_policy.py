from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal

import pytest
import tiktoken

from backend.app.agent_runtime import model_router, rag_cost_policy
from backend.app.agent_runtime.model_router import (
    build_rag_answer_model_config_snapshot_hmac,
)
from backend.app.agent_runtime.rag_cost_policy import (
    RAG_COMPONENT_CEILING_USD,
    RagBudgetExceededError,
    RagCostPolicy,
    RagPolicyUnavailableError,
    build_answer_output_schema_hmac,
)
from backend.app.core.config import Settings
from backend.app.rag.retrieval import (
    AnswerGenerationCostInput,
    QueryEmbeddingCostInput,
    StrictProviderUsage,
    build_query_embedding_model_config_snapshot_hmac,
)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        'agent_runtime_fingerprint_secret': 'task-nine-fingerprint-secret',
        'agent_runtime_fingerprint_key_version': 'task-nine-v1',
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _compact(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()


def _schema_bytes() -> bytes:
    return _compact({
        'name': 'rag_answer_blocks_v1',
        'schema': {'type': 'object'},
        'strict': True,
        'type': 'json_schema',
    })


def _policy(settings: Settings | None = None) -> RagCostPolicy:
    selected = settings or _settings()
    return RagCostPolicy(
        settings=selected,
        answer_output_schema_hmac=build_answer_output_schema_hmac(
            selected,
            _schema_bytes(),
        ),
        answer_prompt_renderer_hmac='b' * 64,
    )


def _answer_input(policy: RagCostPolicy, *, content: str = '질문  두 칸') -> AnswerGenerationCostInput:
    return AnswerGenerationCostInput(
        exact_messages_json=_compact([
            {'content': content, 'ordinal': 1, 'role': 'user'},
        ]),
        exact_response_schema_json=_schema_bytes(),
        model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
    )


@pytest.mark.parametrize('token_count', (7999, 8000))
def test_query_embedding_exact_tiktoken_bound_passes_without_overhead(
    token_count: int,
) -> None:
    policy = _policy()
    raw = ('x ' * (token_count - 1)).encode()
    assert len(tiktoken.get_encoding('cl100k_base').encode(
        raw.decode(), allowed_special=set(), disallowed_special=()
    )) == token_count

    prepared = policy.prepare_query_embedding(QueryEmbeddingCostInput(
        retrieval_query_utf8=raw,
        model_config_snapshot_hmac=policy.query_embedding_model_config_snapshot_hmac,
    ))

    assert prepared.component == 'query_embedding'
    assert prepared.estimated_input_tokens == token_count
    assert prepared.maximum_output_tokens == 0
    assert prepared.reserved_cost_usd == (
        Decimal(token_count) * Decimal('0.020000') / Decimal(1_000_000)
    ).quantize(Decimal('0.000001'), rounding='ROUND_CEILING')


def test_query_embedding_8001_and_component_ceiling_refuse_preflight() -> None:
    policy = _policy()
    raw = ('x ' * 8000).encode()

    with pytest.raises(RagBudgetExceededError) as exc_info:
        policy.prepare_query_embedding(QueryEmbeddingCostInput(
            retrieval_query_utf8=raw,
            model_config_snapshot_hmac=policy.query_embedding_model_config_snapshot_hmac,
        ))

    assert exc_info.value.code == 'budget_exceeded'


def test_query_embedding_strict_utf8_and_byte_identity_preserve_unicode_and_spaces() -> None:
    policy = _policy()
    decomposed = 'Cafe\u0301  repeated   spaces'.encode()
    composed = 'Café  repeated   spaces'.encode()
    first = policy.prepare_query_embedding(QueryEmbeddingCostInput(
        decomposed, policy.query_embedding_model_config_snapshot_hmac
    ))
    second = policy.prepare_query_embedding(QueryEmbeddingCostInput(
        composed, policy.query_embedding_model_config_snapshot_hmac
    ))

    assert first.estimator_input_hmac != second.estimator_input_hmac
    with pytest.raises(ValueError, match='UTF-8'):
        policy.prepare_query_embedding(QueryEmbeddingCostInput(
            b'\xff', policy.query_embedding_model_config_snapshot_hmac
        ))


def test_answer_generation_uses_exact_compact_estimator_o200k_and_full_reserve() -> None:
    policy = _policy()
    value = _answer_input(policy, content='Cafe\u0301  그대로')
    prepared = policy.prepare_answer_generation(value)
    estimator = _compact({
        'messages': json.loads(value.exact_messages_json),
        'model_config_identity': 'rag-answer-model-config:v1',
        'output_schema': json.loads(value.exact_response_schema_json),
        'output_schema_identity': 'rag-answer-blocks:v1',
    }).decode()
    encoded = len(tiktoken.get_encoding('o200k_base').encode(
        estimator, allowed_special=set(), disallowed_special=()
    ))

    assert prepared.component == 'answer_generation'
    assert prepared.estimated_input_tokens == encoded + 16 + 512
    assert prepared.maximum_output_tokens == 512
    assert prepared.reserved_cost_usd == policy.charge_actual(
        'answer_generation', StrictProviderUsage(encoded + 16 + 512, 512, encoded + 16 + 1024)
    )


def test_answer_estimator_preserves_unicode_and_repeated_whitespace_identity() -> None:
    policy = _policy()
    decomposed = policy.prepare_answer_generation(
        _answer_input(policy, content='Cafe\u0301  두 칸')
    )
    composed = policy.prepare_answer_generation(
        _answer_input(policy, content='Café 두 칸')
    )

    assert decomposed.estimator_input_hmac != composed.estimator_input_hmac


def test_answer_rejects_noncanonical_or_malformed_authoritative_json() -> None:
    policy = _policy()
    valid = _answer_input(policy)
    cases = (
        replace(valid, exact_messages_json=bytearray(valid.exact_messages_json)),
        replace(valid, exact_messages_json=b'[{"role":"user"}]'),
        replace(valid, exact_messages_json=b'[ {"content":"x","ordinal":1,"role":"user"} ]'),
        replace(valid, exact_response_schema_json=b'{"type":NaN}'),
    )
    for value in cases:
        with pytest.raises(ValueError):
            policy.prepare_answer_generation(value)


def test_six_place_ceiling_actual_is_unclamped_and_overrun_representable() -> None:
    policy = _policy()

    assert policy.charge_actual(
        'query_embedding', StrictProviderUsage(1, 0, 1)
    ) == Decimal('0.000001')
    assert policy.charge_actual(
        'answer_generation', StrictProviderUsage(1, 1, 2)
    ) == Decimal('0.000006')
    overrun = policy.charge_actual(
        'answer_generation', StrictProviderUsage(20_000, 512, 20_512)
    )
    assert overrun > RAG_COMPONENT_CEILING_USD
    assert overrun == Decimal('0.017304')


def test_model_config_and_policy_mutations_change_hmac_and_fail_readiness() -> None:
    settings = _settings()
    policy = _policy(settings)
    mutated_config = build_rag_answer_model_config_snapshot_hmac(
        settings,
        output_schema_hmac='c' * 64,
        prompt_renderer_hmac='b' * 64,
    )
    assert mutated_config != policy.answer_model_config_snapshot_hmac

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.prepare_answer_generation(replace(
            _answer_input(policy),
            model_config_snapshot_hmac=mutated_config,
        ))
    assert exc_info.value.code == 'model_unavailable'

    expected = policy.authorized_policy_snapshot_hmac('answer_generation')
    with pytest.raises(RagPolicyUnavailableError, match='unavailable'):
        policy.require_authorized_policy_snapshot(
            'answer_generation',
            ('0' if expected[0] != '0' else '1') + expected[1:],
        )


def test_query_model_config_drift_is_zero_call_retriever_refusal() -> None:
    policy = _policy()
    expected = build_query_embedding_model_config_snapshot_hmac(_settings())
    assert expected == policy.query_embedding_model_config_snapshot_hmac
    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.prepare_query_embedding(QueryEmbeddingCostInput(
            b'query',
            '0' * 64,
        ))
    assert exc_info.value.code == 'retriever_not_configured'


def test_cost_and_estimator_hmac_goldens_are_stable() -> None:
    policy = _policy()
    query = policy.prepare_query_embedding(QueryEmbeddingCostInput(
        'Cafe\u0301  repeated   spaces'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    ))
    answer = policy.prepare_answer_generation(
        _answer_input(policy, content='Cafe\u0301  그대로')
    )

    assert policy.query_embedding_model_config_snapshot_hmac == (
        '1372d4f69205743b0d6789208add510bbbf5bce1887fd2fe2fe172a986b79d54'
    )
    assert policy.answer_model_config_snapshot_hmac == (
        '7b3a70a63bda4ab5a061cce63cd730ab951c8e9d8cc1967ec3c234b8a01f0a0e'
    )
    assert policy.authorized_policy_snapshot_hmac('query_embedding') == (
        '2dbc321ad582544ae75fc142f64f57a3d057483da42a45c44814b33567467cdc'
    )
    assert policy.authorized_policy_snapshot_hmac('answer_generation') == (
        '4716809535676e6504b6365de9f222555bbae7aad759ce73cd2c914dcf82f1a2'
    )
    assert query.estimator_input_hmac == (
        '9b296952b02485dab7520f39dd02c037f3b950781e5274db3171a372e9b39128'
    )
    assert answer.estimator_input_hmac == (
        '92dd0b34b581c60a991b127013d417848f55a1aef61114f52f1acbb13a05bb54'
    )


@pytest.mark.parametrize(
    ('module', 'name', 'mutated'),
    (
        (
            rag_cost_policy,
            'ANSWER_INPUT_USD_PER_1M',
            Decimal('0.750001'),
        ),
        (
            rag_cost_policy,
            'ANSWER_TOKENIZER_ENCODING',
            'cl100k_base',
        ),
        (
            rag_cost_policy,
            'ANSWER_ESTIMATOR_VERSION',
            'openai-o200k-rag-answer:mutated',
        ),
        (
            model_router,
            'RAG_OPENAI_API_BASE_URL',
            'https://example.invalid/v1',
        ),
        (
            model_router,
            'RAG_ANSWER_SERVICE_TIER',
            'auto',
        ),
    ),
)
def test_each_price_tokenizer_estimator_endpoint_or_tier_mutation_rebinds_policy(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    name: str,
    mutated: object,
) -> None:
    original = _policy()
    authorized = original.authorized_policy_snapshot_hmac('answer_generation')
    monkeypatch.setattr(module, name, mutated)
    if name == 'ANSWER_TOKENIZER_ENCODING':
        with pytest.raises(ValueError, match='tokenizer'):
            _policy()
        return
    changed = _policy()
    changed_hmac = changed.authorized_policy_snapshot_hmac('answer_generation')

    assert changed_hmac != authorized
    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        changed.require_authorized_policy_snapshot(
            'answer_generation',
            authorized,
        )
    assert exc_info.value.code == 'model_unavailable'


def test_existing_policy_behavior_is_frozen_against_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    value = _answer_input(policy)
    before_budget = policy.prepare_answer_generation(value)
    before_actual = policy.charge_actual(
        'answer_generation',
        StrictProviderUsage(600, 10, 610),
    )
    before_hmac = policy.authorized_policy_snapshot_hmac('answer_generation')
    query_value = QueryEmbeddingCostInput(
        b'query',
        policy.query_embedding_model_config_snapshot_hmac,
    )
    before_query_budget = policy.prepare_query_embedding(query_value)
    before_query_actual = policy.charge_actual(
        'query_embedding',
        StrictProviderUsage(1, 0, 1),
    )
    before_query_hmac = policy.authorized_policy_snapshot_hmac(
        'query_embedding'
    )

    monkeypatch.setattr(
        rag_cost_policy,
        'ANSWER_INPUT_USD_PER_1M',
        Decimal('9.750000'),
    )
    monkeypatch.setattr(rag_cost_policy, 'MAX_ANSWER_INPUT_TOKENS', 1)
    monkeypatch.setattr(rag_cost_policy, 'ANSWER_REPLY_PRIMING_TOKENS', 999)
    monkeypatch.setattr(
        rag_cost_policy,
        'RAG_COMPONENT_CEILING_USD',
        Decimal('0.000001'),
    )
    monkeypatch.setattr(
        rag_cost_policy,
        'QUERY_EMBEDDING_INPUT_USD_PER_1M',
        Decimal('9.020000'),
    )
    monkeypatch.setattr(rag_cost_policy, 'MAX_QUERY_EMBEDDING_TOKENS', 1)
    monkeypatch.setattr(
        rag_cost_policy,
        'QUERY_EMBEDDING_TOKENIZER_ENCODING',
        'o200k_base',
    )
    monkeypatch.setattr(
        rag_cost_policy,
        'QUERY_EMBEDDING_ESTIMATOR_VERSION',
        'openai-cl100k:mutated-after-init',
    )
    monkeypatch.setattr(
        rag_cost_policy,
        'ANSWER_ESTIMATOR_VERSION',
        'openai-o200k-rag-answer:mutated-after-init',
    )

    assert policy.prepare_answer_generation(value) == before_budget
    assert policy.charge_actual(
        'answer_generation',
        StrictProviderUsage(600, 10, 610),
    ) == before_actual
    assert policy.authorized_policy_snapshot_hmac(
        'answer_generation'
    ) == before_hmac
    assert policy.prepare_query_embedding(query_value) == before_query_budget
    assert policy.charge_actual(
        'query_embedding',
        StrictProviderUsage(1, 0, 1),
    ) == before_query_actual
    assert policy.authorized_policy_snapshot_hmac(
        'query_embedding'
    ) == before_query_hmac
    with pytest.raises(AttributeError):
        policy._answer_policy_hmac = '0' * 64


def test_tokenizer_resolver_cannot_substitute_an_encoding_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o200k = tiktoken.get_encoding('o200k_base')
    original = tiktoken.get_encoding

    def substitute(name: str):
        if name == 'cl100k_base':
            return o200k
        return original(name)

    monkeypatch.setattr(rag_cost_policy.tiktoken, 'get_encoding', substitute)

    with pytest.raises(ValueError, match='tokenizer'):
        _policy()

    class SpoofedEncoding:
        name = 'cl100k_base'

        def encode(self, value, **kwargs):
            return o200k.encode(value, **kwargs)

    def spoof(name: str):
        if name == 'cl100k_base':
            return SpoofedEncoding()
        return original(name)

    monkeypatch.setattr(rag_cost_policy.tiktoken, 'get_encoding', spoof)
    with pytest.raises(ValueError, match='tokenizer'):
        _policy()


def _answer_estimator_char_count(value: AnswerGenerationCostInput) -> int:
    return len(_compact({
        'messages': json.loads(value.exact_messages_json),
        'model_config_identity': 'rag-answer-model-config:v1',
        'output_schema': json.loads(value.exact_response_schema_json),
        'output_schema_identity': 'rag-answer-blocks:v1',
    }).decode())


def _answer_input_with_estimator_chars(
    policy: RagCostPolicy,
    target: int,
) -> AnswerGenerationCostInput:
    empty = _answer_input(policy, content='')
    base = _answer_estimator_char_count(empty)
    assert base < target
    value = _answer_input(policy, content='x' * (target - base))
    assert _answer_estimator_char_count(value) == target
    return value


@pytest.mark.parametrize('character_count', (11_999, 12_000))
def test_answer_exact_serialized_character_bound_passes(
    character_count: int,
) -> None:
    policy = _policy()

    prepared = policy.prepare_answer_generation(
        _answer_input_with_estimator_chars(policy, character_count)
    )

    assert prepared.component == 'answer_generation'


def test_answer_12001_serialized_characters_refuses() -> None:
    policy = _policy()

    with pytest.raises(RagBudgetExceededError):
        policy.prepare_answer_generation(
            _answer_input_with_estimator_chars(policy, 12_001)
        )


def _answer_input_with_framed_tokens(
    policy: RagCostPolicy,
    framed_tokens: int,
) -> AnswerGenerationCostInput:
    tokenizer = tiktoken.get_encoding('o200k_base')

    def encoded_count(value: AnswerGenerationCostInput) -> int:
        estimator = _compact({
            'messages': json.loads(value.exact_messages_json),
            'model_config_identity': 'rag-answer-model-config:v1',
            'output_schema': json.loads(value.exact_response_schema_json),
            'output_schema_identity': 'rag-answer-blocks:v1',
        }).decode()
        return len(tokenizer.encode(
            estimator,
            allowed_special=set(),
            disallowed_special=(),
        ))

    base = encoded_count(_answer_input(policy, content=''))
    target_encoded = framed_tokens - 16 - 512
    content_length = target_encoded - base
    for _ in range(4):
        value = _answer_input(policy, content='가' * content_length)
        difference = target_encoded - encoded_count(value)
        if difference == 0:
            break
        content_length += difference
    assert encoded_count(value) == target_encoded
    return value


@pytest.mark.parametrize('framed_tokens', (9_999, 10_000))
def test_answer_exact_framed_token_bound_passes(
    framed_tokens: int,
) -> None:
    policy = _policy()

    prepared = policy.prepare_answer_generation(
        _answer_input_with_framed_tokens(policy, framed_tokens)
    )

    assert prepared.estimated_input_tokens == framed_tokens


def test_answer_10001_framed_tokens_refuses() -> None:
    policy = _policy()

    with pytest.raises(RagBudgetExceededError):
        policy.prepare_answer_generation(
            _answer_input_with_framed_tokens(policy, 10_001)
        )


def test_answer_real_component_ceiling_refuses_without_clamping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_cost_policy,
        'ANSWER_OUTPUT_USD_PER_1M',
        Decimal('100.000000'),
    )
    policy = _policy()

    with pytest.raises(RagBudgetExceededError):
        policy.prepare_answer_generation(_answer_input(policy))

    assert policy.charge_actual(
        'answer_generation',
        StrictProviderUsage(0, 512, 512),
    ) == Decimal('0.051200')


def test_policy_and_authority_have_no_attribute_mutation_bypass() -> None:
    policy = _policy()

    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(
            policy._authority,
            'query_input_usd_per_1m',
            Decimal('9.020000'),
        )
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(policy, '_query_policy_hmac', '0' * 64)

    prepared = policy.prepare_query_embedding(QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    ))
    assert prepared.reserved_cost_usd == Decimal('0.000001')


def test_tuple_base_constructor_cannot_forge_same_hmac_different_behavior() -> None:
    policy = _policy()
    forged_authority = policy._authority._replace(
        query_input_usd_per_1m=Decimal('9.020000'),
    )
    forged = tuple.__new__(
        RagCostPolicy,
        (forged_authority, *policy[1:]),
    )
    query = QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    )

    assert forged[8] == (
        policy.authorized_policy_snapshot_hmac('query_embedding')
    )
    with pytest.raises(RagPolicyUnavailableError) as readiness_exc:
        forged.authorized_policy_snapshot_hmac('query_embedding')
    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        forged.prepare_query_embedding(query)

    assert readiness_exc.value.code == 'retriever_not_configured'
    assert exc_info.value.code == 'retriever_not_configured'


@pytest.mark.parametrize('operation', (copy.copy, copy.deepcopy, pickle.dumps))
def test_policy_and_authority_reject_copy_deepcopy_and_pickle(
    operation: Callable[[object], object],
) -> None:
    policy = _policy()

    with pytest.raises(TypeError, match='non-transferable'):
        operation(policy)
    with pytest.raises(TypeError, match='non-transferable'):
        operation(policy._authority)


def test_query_tokenizer_core_drift_is_zero_call_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    authorized = policy.authorized_policy_snapshot_hmac('query_embedding')
    query = QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    )
    before = policy.prepare_query_embedding(query)
    query_tokenizer = tiktoken.get_encoding('cl100k_base')
    answer_tokenizer = tiktoken.get_encoding('o200k_base')
    monkeypatch.setattr(
        query_tokenizer,
        '_core_bpe',
        answer_tokenizer._core_bpe,
    )

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.prepare_query_embedding(query)

    assert before.estimated_input_tokens > 0
    assert exc_info.value.code == 'retriever_not_configured'
    assert policy.authorized_policy_snapshot_hmac('query_embedding') == authorized


def test_answer_tokenizer_core_drift_is_zero_call_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    authorized = policy.authorized_policy_snapshot_hmac('answer_generation')
    value = _answer_input(policy, content='한국어 근거 답변')
    answer_tokenizer = tiktoken.get_encoding('o200k_base')
    query_tokenizer = tiktoken.get_encoding('cl100k_base')
    monkeypatch.setattr(
        answer_tokenizer,
        '_core_bpe',
        query_tokenizer._core_bpe,
    )

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.prepare_answer_generation(value)

    assert exc_info.value.code == 'model_unavailable'
    assert policy.authorized_policy_snapshot_hmac('answer_generation') == authorized


def test_answer_exact_rounded_component_ceiling_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_cost_policy,
        'ANSWER_OUTPUT_USD_PER_1M',
        Decimal('22.558593'),
    )
    policy = _policy()

    prepared = policy.prepare_answer_generation(_answer_input(policy))

    assert prepared.reserved_cost_usd == Decimal('0.012000')


def test_answer_one_microdollar_above_component_ceiling_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_cost_policy,
        'ANSWER_OUTPUT_USD_PER_1M',
        Decimal('22.558594'),
    )
    policy = _policy()

    with pytest.raises(RagBudgetExceededError):
        policy.prepare_answer_generation(_answer_input(policy))
