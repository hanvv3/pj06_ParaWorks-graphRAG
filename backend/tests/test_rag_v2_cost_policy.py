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


def _insertion_order_schema_bytes() -> bytes:
    return json.dumps(
        {
            'name': 'rag_answer_blocks_v1',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': {'answer_blocks': {'type': 'array'}},
                'required': ['answer_blocks'],
                'additionalProperties': False,
            },
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
        separators=(',', ':'),
    ).encode()


def test_answer_cost_accepts_exact_hmac_bound_insertion_order_schema() -> None:
    settings = _settings()
    schema = _insertion_order_schema_bytes()
    policy = RagCostPolicy(
        settings=settings,
        answer_output_schema_hmac=build_answer_output_schema_hmac(settings, schema),
        answer_prompt_renderer_hmac='b' * 64,
    )

    prepared = policy.prepare_answer_generation(AnswerGenerationCostInput(
        exact_messages_json=_compact([
            {'content': 'question', 'ordinal': 1, 'role': 'user'},
        ]),
        exact_response_schema_json=schema,
        model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
    ))

    assert prepared.estimator_input_hmac == policy.prepare_answer_generation(
        AnswerGenerationCostInput(
            exact_messages_json=_compact([
                {'content': 'question', 'ordinal': 1, 'role': 'user'},
            ]),
            exact_response_schema_json=schema,
            model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
        )
    ).estimator_input_hmac


@pytest.mark.parametrize(
    'schema',
    (
        b'{"name":"x","name":"y"}',
        b'{"name":"x"} ',
        b'{"name":NaN}',
        b'{"name":',
        b'\xff',
    ),
)
def test_answer_cost_rejects_malformed_or_unstable_exact_schema(
    schema: bytes,
) -> None:
    settings = _settings()
    with pytest.raises(ValueError, match='schema'):
        policy = RagCostPolicy(
            settings=settings,
            answer_output_schema_hmac=build_answer_output_schema_hmac(
                settings,
                schema,
            ),
            answer_prompt_renderer_hmac='b' * 64,
        )
        policy.prepare_answer_generation(AnswerGenerationCostInput(
            exact_messages_json=_compact([
                {'content': 'question', 'ordinal': 1, 'role': 'user'},
            ]),
            exact_response_schema_json=schema,
            model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
        ))


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


def test_policy_is_opaque_and_exposes_no_secret_authority_or_tokenizer() -> None:
    settings = _settings()
    policy = _policy(settings)

    assert settings.agent_runtime_fingerprint_secret not in repr(policy)
    assert not hasattr(policy, '_authority')
    assert not hasattr(policy, '_secret')
    assert not hasattr(policy, '_integrity_hmac')
    with pytest.raises(TypeError):
        iter(policy)
    with pytest.raises(TypeError):
        tuple(policy)
    with pytest.raises(TypeError):
        policy[0]
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(policy, '_query_policy_hmac', '0' * 64)

    prepared = policy.prepare_query_embedding(QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    ))
    assert prepared.reserved_cost_usd == Decimal('0.000001')


def test_object_new_unregistered_policy_is_typed_zero_call_refusal() -> None:
    policy = _policy()
    forged = object.__new__(RagCostPolicy)
    query = QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    )

    with pytest.raises(RagPolicyUnavailableError) as readiness_exc:
        forged.authorized_policy_snapshot_hmac('query_embedding')
    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        forged.prepare_query_embedding(query)

    assert readiness_exc.value.code == 'retriever_not_configured'
    assert exc_info.value.code == 'retriever_not_configured'


def test_policy_subclass_is_rejected() -> None:
    with pytest.raises(TypeError, match='subclass'):
        class _ForgedPolicy(RagCostPolicy):
            pass


@pytest.mark.parametrize('operation', (copy.copy, copy.deepcopy, pickle.dumps))
def test_policy_rejects_copy_deepcopy_and_pickle(
    operation: Callable[[object], object],
) -> None:
    policy = _policy()

    with pytest.raises(TypeError, match='non-transferable'):
        operation(policy)


@pytest.mark.parametrize('encoding_name', ('cl100k_base', 'o200k_base'))
def test_fake_core_with_approved_token_bytes_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    encoding_name: str,
) -> None:
    tokenizer = tiktoken.get_encoding(encoding_name)
    approved_core = tokenizer._core_bpe

    class _FakeCore:
        def token_byte_values(self):
            return approved_core.token_byte_values()

        def encode(self, text, allowed_special):
            return [0]

    monkeypatch.setattr(tokenizer, '_core_bpe', _FakeCore())

    with pytest.raises(ValueError, match='tokenizer'):
        _policy()


@pytest.mark.parametrize(
    ('component', 'unavailable_code'),
    (
        ('query_embedding', 'retriever_not_configured'),
        ('answer_generation', 'model_unavailable'),
    ),
)
def test_encode_time_core_swap_discards_count_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    unavailable_code: str,
) -> None:
    policy = _policy()
    original_encode = tiktoken.Encoding.encode
    replacement_core = tiktoken.get_encoding(
        'o200k_base' if component == 'query_embedding' else 'cl100k_base'
    )._core_bpe
    changed: list[tuple[tiktoken.Encoding, object]] = []

    def encode_and_swap(self, text, **kwargs):
        result = original_encode(self, text, **kwargs)
        changed.append((self, self._core_bpe))
        self._core_bpe = replacement_core
        return result

    monkeypatch.setattr(tiktoken.Encoding, 'encode', encode_and_swap)
    try:
        with pytest.raises(RagPolicyUnavailableError) as exc_info:
            if component == 'query_embedding':
                policy.prepare_query_embedding(QueryEmbeddingCostInput(
                    '한국어 그래프 검색'.encode(),
                    policy.query_embedding_model_config_snapshot_hmac,
                ))
            else:
                policy.prepare_answer_generation(
                    _answer_input(policy, content='한국어 근거 답변')
                )
    finally:
        for tokenizer, original_core in changed:
            tokenizer._core_bpe = original_core

    assert exc_info.value.code == unavailable_code


def test_global_cached_tokenizer_core_drift_cannot_reach_private_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    query = QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    )
    answer = _answer_input(policy, content='한국어 근거 답변')
    before_query = policy.prepare_query_embedding(query)
    before_answer = policy.prepare_answer_generation(answer)
    query_tokenizer = tiktoken.get_encoding('cl100k_base')
    answer_tokenizer = tiktoken.get_encoding('o200k_base')
    query_core = query_tokenizer._core_bpe
    answer_core = answer_tokenizer._core_bpe
    monkeypatch.setattr(query_tokenizer, '_core_bpe', answer_core)
    monkeypatch.setattr(answer_tokenizer, '_core_bpe', query_core)

    assert policy.prepare_query_embedding(query) == before_query
    assert policy.prepare_answer_generation(answer) == before_answer


def test_module_exposes_no_canonical_policy_state_or_registry_accessor() -> None:
    for name in (
        '_POLICY_STATES',
        '_POLICY_STATES_LOCK',
        '_PolicyState',
        '_PrivateTokenizer',
        '_RagCostAuthority',
        '_policy_state',
    ):
        assert not hasattr(rag_cost_policy, name)


class _InvalidationHook:
    def __init__(self, target: str, mode: str) -> None:
        self.target = target
        self.mode = mode
        self.active = False

    def __call__(self, point, policy, invalidate) -> None:
        if self.active and point == self.target:
            self.active = False
            invalidate(self.mode)


class _ClassSwapHook(_InvalidationHook):
    def __init__(self, target: str) -> None:
        super().__init__(target, 'unused')
        self.original_type = None
        self.replacement_type = rag_cost_policy._create_rag_cost_policy_type()

    def __call__(self, point, policy, invalidate) -> None:
        if self.active and point == self.target:
            self.active = False
            self.original_type = type(policy)
            policy.__class__ = self.replacement_type

    def restore(self, policy) -> None:
        if self.original_type is not None:
            policy.__class__ = self.original_type


def _isolated_policy(hook: _InvalidationHook):
    policy_type = rag_cost_policy._create_rag_cost_policy_type(test_hook=hook)
    settings = _settings()
    schema_hmac = build_answer_output_schema_hmac(settings, _schema_bytes())
    policy = policy_type(
        settings=settings,
        answer_output_schema_hmac=schema_hmac,
        answer_prompt_renderer_hmac='b' * 64,
    )
    assert policy_type is not RagCostPolicy
    return policy


@pytest.mark.parametrize('mode', ('remove', 'replace'))
def test_property_registry_invalidation_is_typed_refusal(mode: str) -> None:
    hook = _InvalidationHook('query_property_before_return', mode)
    policy = _isolated_policy(hook)
    hook.active = True

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        _ = policy.query_embedding_model_config_snapshot_hmac

    assert exc_info.value.code == 'retriever_not_configured'


@pytest.mark.parametrize('mode', ('remove', 'replace'))
def test_query_encode_registry_invalidation_is_typed_refusal(mode: str) -> None:
    hook = _InvalidationHook('query_encode_after_call', mode)
    policy = _isolated_policy(hook)
    query = QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    )
    hook.active = True

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.prepare_query_embedding(query)

    assert exc_info.value.code == 'retriever_not_configured'


@pytest.mark.parametrize('mode', ('remove', 'replace'))
def test_answer_schema_registry_invalidation_is_typed_refusal(mode: str) -> None:
    hook = _InvalidationHook('answer_schema_after_hmac', mode)
    policy = _isolated_policy(hook)
    value = _answer_input(policy, content='한국어 근거 답변')
    hook.active = True

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.prepare_answer_generation(value)

    assert exc_info.value.code == 'model_unavailable'


@pytest.mark.parametrize('mode', ('remove', 'replace'))
def test_charge_registry_invalidation_is_typed_refusal(mode: str) -> None:
    hook = _InvalidationHook('charge_before_return', mode)
    policy = _isolated_policy(hook)
    hook.active = True

    with pytest.raises(RagPolicyUnavailableError) as exc_info:
        policy.charge_actual(
            'answer_generation',
            StrictProviderUsage(1, 1, 2),
        )

    assert exc_info.value.code == 'model_unavailable'


@pytest.mark.parametrize(
    ('point', 'operation', 'expected_code'),
    (
        (
            'query_property_before_return',
            'query_property',
            'retriever_not_configured',
        ),
        (
            'query_embedding_policy_before_return',
            'query_policy',
            'retriever_not_configured',
        ),
        (
            'query_encode_after_call',
            'query_prepare',
            'retriever_not_configured',
        ),
        ('answer_schema_after_hmac', 'answer_prepare', 'model_unavailable'),
        ('answer_encode_after_call', 'answer_prepare', 'model_unavailable'),
        ('charge_before_return', 'charge', 'model_unavailable'),
    ),
)
def test_class_swap_during_public_operation_is_typed_refusal(
    point: str,
    operation: str,
    expected_code: str,
) -> None:
    hook = _ClassSwapHook(point)
    policy = _isolated_policy(hook)
    original_type = type(policy)
    query = QueryEmbeddingCostInput(
        '한국어 그래프 검색'.encode(),
        policy.query_embedding_model_config_snapshot_hmac,
    )
    answer = _answer_input(policy, content='한국어 근거 답변')
    hook.active = True

    try:
        with pytest.raises(RagPolicyUnavailableError) as exc_info:
            if operation == 'query_property':
                _ = policy.query_embedding_model_config_snapshot_hmac
            elif operation == 'query_policy':
                policy.authorized_policy_snapshot_hmac('query_embedding')
            elif operation == 'query_prepare':
                policy.prepare_query_embedding(query)
            elif operation == 'answer_prepare':
                policy.prepare_answer_generation(answer)
            elif operation == 'charge':
                policy.charge_actual(
                    'answer_generation',
                    StrictProviderUsage(1, 1, 2),
                )
            else:
                raise AssertionError('unknown class-swap operation')
    finally:
        hook.restore(policy)

    assert exc_info.value.code == expected_code
    assert type(policy) is original_type
    assert len(policy.query_embedding_model_config_snapshot_hmac) == 64


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
