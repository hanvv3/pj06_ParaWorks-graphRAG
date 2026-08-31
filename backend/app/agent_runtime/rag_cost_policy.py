from __future__ import annotations

import hmac
import json
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Final

import tiktoken

from backend.app.admin.auto_review_keys import (
    fingerprint_key_material_verifier,
)
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.model_router import (
    RAG_ANSWER_MODEL_CONFIG_VERSION,
    build_rag_answer_model_config_snapshot_hmac,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.core.config import Settings
from backend.app.rag.retrieval import (
    QUERY_EMBEDDING_MODEL_CONFIG_VERSION,
    QUERY_EMBEDDING_PAYLOAD_VALIDATOR_VERSION,
    AnswerGenerationCostInput,
    PreparedPaidCallBudget,
    QueryEmbeddingCostInput,
    RagPaidComponent,
    StrictProviderUsage,
    build_query_embedding_model_config_snapshot_hmac,
    is_lower_hex_64,
    validate_strict_provider_usage,
)

RAG_COMPONENT_CEILING_USD: Final = Decimal('0.012000')
QUERY_EMBEDDING_INPUT_USD_PER_1M: Final = Decimal('0.020000')
ANSWER_INPUT_USD_PER_1M: Final = Decimal('0.750000')
ANSWER_OUTPUT_USD_PER_1M: Final = Decimal('4.500000')
QUERY_EMBEDDING_COST_POLICY_VERSION: Final = 'rag-query-embedding-cost:v1'
QUERY_EMBEDDING_ESTIMATOR_VERSION: Final = (
    'openai-cl100k-text-embedding-3-small:v1'
)
ANSWER_COST_POLICY_VERSION: Final = 'rag-answer-cost:v1'
ANSWER_ESTIMATOR_VERSION: Final = 'openai-o200k-rag-answer:v1'
QUERY_EMBEDDING_TOKENIZER_ENCODING: Final = 'cl100k_base'
ANSWER_TOKENIZER_ENCODING: Final = 'o200k_base'
MAX_QUERY_EMBEDDING_TOKENS: Final = 8_000
MAX_ANSWER_SERIALIZED_CHARS: Final = 12_000
MAX_ANSWER_INPUT_TOKENS: Final = 10_000
MAX_ANSWER_OUTPUT_TOKENS: Final = 512
ANSWER_REPLY_PRIMING_TOKENS: Final = 16
ANSWER_FRAMING_SAFETY_TOKENS: Final = 512
_MILLION: Final = Decimal(1_000_000)
_SIX_PLACES: Final = Decimal('0.000001')


class RagCostPolicyError(ValueError):
    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RagBudgetExceededError(RagCostPolicyError):
    def __init__(self) -> None:
        super().__init__('budget_exceeded', 'RAG provider budget is exceeded')


class RagPolicyUnavailableError(RagCostPolicyError):
    def __init__(self, code: str) -> None:
        super().__init__(code, 'RAG provider policy is unavailable')


def build_answer_output_schema_hmac(
    settings: Settings,
    exact_response_schema_json: bytes,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return _answer_output_schema_hmac(secret, exact_response_schema_json)


def _answer_output_schema_hmac(
    secret: bytes,
    exact_response_schema_json: bytes,
) -> str:
    if type(exact_response_schema_json) is not bytes:
        raise ValueError('answer output schema bytes are invalid')
    try:
        text = exact_response_schema_json.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        raise ValueError('answer output schema must be strict UTF-8') from None
    return keyed_fingerprint(
        exact_utf8_bytes(text),
        secret=secret,
        schema_version='rag-answer-output-schema-bytes:v1',
        policy_version='rag-answer-blocks:v1',
    )


class RagCostPolicy:
    __slots__ = (
        '_answer_config_hmac',
        '_answer_policy_hmac',
        '_answer_prompt_renderer_hmac',
        '_answer_schema_hmac',
        '_answer_tokenizer',
        '_key_material_verifier',
        '_key_version',
        '_query_config_hmac',
        '_query_policy_hmac',
        '_query_tokenizer',
        '_secret',
    )

    def __init__(
        self,
        *,
        settings: Settings,
        answer_output_schema_hmac: str,
        answer_prompt_renderer_hmac: str,
    ) -> None:
        if type(settings) is not Settings:
            raise ValueError('RAG cost policy settings are invalid')
        if not is_lower_hex_64(answer_output_schema_hmac) or not is_lower_hex_64(
            answer_prompt_renderer_hmac
        ):
            raise ValueError('RAG answer policy identities are invalid')
        _validate_frozen_prices()
        secret, key_version = fingerprint_secret_bytes(settings)
        self._secret = secret
        self._key_version = key_version
        self._key_material_verifier = fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        )
        self._answer_schema_hmac = answer_output_schema_hmac
        self._answer_prompt_renderer_hmac = answer_prompt_renderer_hmac
        self._query_config_hmac = (
            build_query_embedding_model_config_snapshot_hmac(settings)
        )
        self._answer_config_hmac = build_rag_answer_model_config_snapshot_hmac(
            settings,
            output_schema_hmac=answer_output_schema_hmac,
            prompt_renderer_hmac=answer_prompt_renderer_hmac,
        )
        self._query_policy_hmac = self._build_query_policy_hmac()
        self._answer_policy_hmac = self._build_answer_policy_hmac()
        self._query_tokenizer = tiktoken.get_encoding(
            QUERY_EMBEDDING_TOKENIZER_ENCODING
        )
        self._answer_tokenizer = tiktoken.get_encoding(
            ANSWER_TOKENIZER_ENCODING
        )

    @property
    def query_embedding_model_config_snapshot_hmac(self) -> str:
        return self._query_config_hmac

    @property
    def answer_model_config_snapshot_hmac(self) -> str:
        return self._answer_config_hmac

    def authorized_policy_snapshot_hmac(
        self,
        component: RagPaidComponent,
    ) -> str:
        selected = _exact_component(component)
        if selected == 'query_embedding':
            return self._query_policy_hmac
        return self._answer_policy_hmac

    def require_authorized_policy_snapshot(
        self,
        component: RagPaidComponent,
        observed_hmac: str,
    ) -> None:
        selected = _exact_component(component)
        if not is_lower_hex_64(observed_hmac) or not hmac.compare_digest(
            observed_hmac,
            self.authorized_policy_snapshot_hmac(selected),
        ):
            code = (
                'retriever_not_configured'
                if selected == 'query_embedding'
                else 'model_unavailable'
            )
            raise RagPolicyUnavailableError(code)

    def prepare_query_embedding(
        self,
        value: QueryEmbeddingCostInput,
    ) -> PreparedPaidCallBudget:
        if (
            type(value) is not QueryEmbeddingCostInput
            or type(value.retrieval_query_utf8) is not bytes
            or not is_lower_hex_64(value.model_config_snapshot_hmac)
        ):
            raise ValueError('query embedding cost input is invalid')
        if not hmac.compare_digest(
            value.model_config_snapshot_hmac,
            self._query_config_hmac,
        ):
            raise RagPolicyUnavailableError('retriever_not_configured')
        try:
            text = value.retrieval_query_utf8.decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            raise ValueError('query embedding input must be strict UTF-8') from None
        encoded_tokens = len(self._query_tokenizer.encode(
            text,
            allowed_special=set(),
            disallowed_special=(),
        ))
        if encoded_tokens > MAX_QUERY_EMBEDDING_TOKENS:
            raise RagBudgetExceededError
        reserved = _rounded_cost(
            input_tokens=encoded_tokens,
            output_tokens=0,
            input_price=QUERY_EMBEDDING_INPUT_USD_PER_1M,
            output_price=Decimal('0.000000'),
        )
        _require_component_ceiling(reserved)
        estimator_input_hmac = keyed_fingerprint(
            exact_utf8_bytes(text),
            secret=self._secret,
            schema_version='rag-query-embedding-estimator-input-bytes:v1',
            policy_version=QUERY_EMBEDDING_ESTIMATOR_VERSION,
        )
        return PreparedPaidCallBudget(
            component='query_embedding',
            estimated_input_tokens=encoded_tokens,
            maximum_output_tokens=0,
            reserved_cost_usd=reserved,
            cost_policy_snapshot_hmac=self._query_policy_hmac,
            estimator_input_hmac=estimator_input_hmac,
        )

    def prepare_answer_generation(
        self,
        value: AnswerGenerationCostInput,
    ) -> PreparedPaidCallBudget:
        if (
            type(value) is not AnswerGenerationCostInput
            or type(value.exact_messages_json) is not bytes
            or type(value.exact_response_schema_json) is not bytes
            or not is_lower_hex_64(value.model_config_snapshot_hmac)
        ):
            raise ValueError('answer generation cost input is invalid')
        if not hmac.compare_digest(
            value.model_config_snapshot_hmac,
            self._answer_config_hmac,
        ):
            raise RagPolicyUnavailableError('model_unavailable')

        messages = _decode_exact_json(value.exact_messages_json, 'messages')
        schema = _decode_exact_json(value.exact_response_schema_json, 'schema')
        _validate_exact_messages(messages)
        if type(schema) is not dict:
            raise ValueError('answer response schema must be an object')
        observed_schema_hmac = _answer_output_schema_hmac(
            self._secret,
            value.exact_response_schema_json,
        )
        if not hmac.compare_digest(
            observed_schema_hmac,
            self._answer_schema_hmac,
        ):
            raise RagPolicyUnavailableError('model_unavailable')

        estimator_bytes = _canonical_json_bytes({
            'messages': messages,
            'model_config_identity': RAG_ANSWER_MODEL_CONFIG_VERSION,
            'output_schema': schema,
            'output_schema_identity': 'rag-answer-blocks:v1',
        })
        estimator_text = estimator_bytes.decode('utf-8')
        if len(estimator_text) > MAX_ANSWER_SERIALIZED_CHARS:
            raise RagBudgetExceededError
        encoded_tokens = len(self._answer_tokenizer.encode(
            estimator_text,
            allowed_special=set(),
            disallowed_special=(),
        ))
        framed_tokens = (
            encoded_tokens
            + ANSWER_REPLY_PRIMING_TOKENS
            + ANSWER_FRAMING_SAFETY_TOKENS
        )
        if framed_tokens > MAX_ANSWER_INPUT_TOKENS:
            raise RagBudgetExceededError
        reserved = _rounded_cost(
            input_tokens=framed_tokens,
            output_tokens=MAX_ANSWER_OUTPUT_TOKENS,
            input_price=ANSWER_INPUT_USD_PER_1M,
            output_price=ANSWER_OUTPUT_USD_PER_1M,
        )
        _require_component_ceiling(reserved)
        estimator_hmac = keyed_fingerprint(
            exact_utf8_bytes(estimator_text),
            secret=self._secret,
            schema_version='rag-generation-estimator-input-bytes:v1',
            policy_version=ANSWER_ESTIMATOR_VERSION,
        )
        return PreparedPaidCallBudget(
            component='answer_generation',
            estimated_input_tokens=framed_tokens,
            maximum_output_tokens=MAX_ANSWER_OUTPUT_TOKENS,
            reserved_cost_usd=reserved,
            cost_policy_snapshot_hmac=self._answer_policy_hmac,
            estimator_input_hmac=estimator_hmac,
        )

    def charge_actual(
        self,
        component: RagPaidComponent,
        usage: StrictProviderUsage,
    ) -> Decimal:
        selected = _exact_component(component)
        strict_usage = validate_strict_provider_usage(usage)
        if selected == 'query_embedding':
            if strict_usage.output_tokens != 0:
                raise ValueError('query embedding output usage must be zero')
            return _rounded_cost(
                input_tokens=strict_usage.input_tokens,
                output_tokens=0,
                input_price=QUERY_EMBEDDING_INPUT_USD_PER_1M,
                output_price=Decimal('0.000000'),
            )
        return _rounded_cost(
            input_tokens=strict_usage.input_tokens,
            output_tokens=strict_usage.output_tokens,
            input_price=ANSWER_INPUT_USD_PER_1M,
            output_price=ANSWER_OUTPUT_USD_PER_1M,
        )

    def _build_query_policy_hmac(self) -> str:
        return keyed_fingerprint(
            {
                'authorized_model_config_snapshot_hmac': self._query_config_hmac,
                'authorized_model_config_version': (
                    QUERY_EMBEDDING_MODEL_CONFIG_VERSION
                ),
                'component': 'query_embedding',
                'cosine_indexability_policy_version': (
                    'pgvector-cosine-indexable:v1'
                ),
                'cost_policy_version': QUERY_EMBEDDING_COST_POLICY_VERSION,
                'credential_scan_policy_version': 'credential-scan:v1',
                'embedding_payload_validator_version': (
                    QUERY_EMBEDDING_PAYLOAD_VALIDATOR_VERSION
                ),
                'fingerprint_key_material_verifier': (
                    self._key_material_verifier
                ),
                'fingerprint_key_version': self._key_version,
                'input_usd_per_1m': _price_identity(
                    QUERY_EMBEDDING_INPUT_USD_PER_1M
                ),
                'max_input_tokens': MAX_QUERY_EMBEDDING_TOKENS,
                'rag_component_ceiling_usd': _price_identity(
                    RAG_COMPONENT_CEILING_USD
                ),
                'rounding': 'six-place-ROUND_CEILING',
                'token_count_rule': (
                    'exact-query-text:no-normalization:no-chat-overhead:v1'
                ),
                'token_estimator_version': QUERY_EMBEDDING_ESTIMATOR_VERSION,
                'tokenizer_encoding': QUERY_EMBEDDING_TOKENIZER_ENCODING,
            },
            secret=self._secret,
            schema_version='rag-query-embedding-provider-policy-snapshot:v1',
            policy_version=QUERY_EMBEDDING_COST_POLICY_VERSION,
        )

    def _build_answer_policy_hmac(self) -> str:
        return keyed_fingerprint(
            {
                'answer_block_joiner_version': 'rag-answer-block-joiner:v1',
                'authorized_model_config_snapshot_hmac': self._answer_config_hmac,
                'authorized_model_config_version': RAG_ANSWER_MODEL_CONFIG_VERSION,
                'component': 'answer_generation',
                'cost_policy_version': ANSWER_COST_POLICY_VERSION,
                'credential_scan_policy_version': 'credential-scan:v1',
                'fingerprint_key_material_verifier': (
                    self._key_material_verifier
                ),
                'fingerprint_key_version': self._key_version,
                'framing_safety_tokens': ANSWER_FRAMING_SAFETY_TOKENS,
                'input_usd_per_1m': _price_identity(
                    ANSWER_INPUT_USD_PER_1M
                ),
                'max_input_tokens': MAX_ANSWER_INPUT_TOKENS,
                'max_output_tokens': MAX_ANSWER_OUTPUT_TOKENS,
                'max_serialized_input_chars': MAX_ANSWER_SERIALIZED_CHARS,
                'output_contract_version': 'rag-answer-blocks:v1',
                'output_schema_hmac': self._answer_schema_hmac,
                'output_schema_name': 'rag_answer_blocks_v1',
                'output_usd_per_1m': _price_identity(
                    ANSWER_OUTPUT_USD_PER_1M
                ),
                'prompt_version': 'rag-answer:v2',
                'prompt_renderer_hmac': self._answer_prompt_renderer_hmac,
                'prompt_renderer_version': 'rag-answer-renderer:v1',
                'rag_component_ceiling_usd': _price_identity(
                    RAG_COMPONENT_CEILING_USD
                ),
                'reply_priming_tokens': ANSWER_REPLY_PRIMING_TOKENS,
                'rounding': 'six-place-ROUND_CEILING',
                'token_count_rule': (
                    'compact-json-messages-and-exact-schema:'
                    'no-unicode-normalization:v1'
                ),
                'token_estimator_version': ANSWER_ESTIMATOR_VERSION,
                'tokenizer_encoding': ANSWER_TOKENIZER_ENCODING,
            },
            secret=self._secret,
            schema_version='rag-answer-provider-policy-snapshot:v1',
            policy_version=ANSWER_COST_POLICY_VERSION,
        )


def _exact_component(component: object) -> RagPaidComponent:
    if type(component) is not str or component not in {
        'query_embedding',
        'answer_generation',
    }:
        raise ValueError('RAG paid component is invalid')
    return component


def _decode_exact_json(value: bytes, name: str) -> object:
    try:
        text = value.decode('utf-8', errors='strict')
        parsed = json.loads(
            text,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError):
        raise ValueError(f'answer {name} JSON is invalid') from None
    _validate_json_tree(parsed)
    if _canonical_json_bytes(parsed) != value:
        raise ValueError(f'answer {name} JSON is not canonical')
    return parsed


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError, RecursionError):
        raise ValueError('answer estimator JSON is invalid') from None


def _validate_exact_messages(value: object) -> None:
    if type(value) is not list or not value:
        raise ValueError('answer messages must be a nonempty list')
    for expected_ordinal, message in enumerate(value, start=1):
        if (
            type(message) is not dict
            or set(message) != {'content', 'ordinal', 'role'}
            or type(message['content']) is not str
            or type(message['ordinal']) is not int
            or message['ordinal'] != expected_ordinal
            or type(message['role']) is not str
            or message['role'] not in {'system', 'user'}
        ):
            raise ValueError('answer messages are invalid')


def _validate_json_tree(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for item in value:
            _validate_json_tree(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('answer estimator JSON keys are invalid')
            _validate_json_tree(item)
        return
    raise ValueError('answer estimator JSON values are invalid')


def _rounded_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_price: Decimal,
    output_price: Decimal,
) -> Decimal:
    with localcontext() as context:
        context.prec = 64
        value = (
            Decimal(input_tokens) * input_price
            + Decimal(output_tokens) * output_price
        ) / _MILLION
        return value.quantize(_SIX_PLACES, rounding=ROUND_CEILING)


def _require_component_ceiling(value: Decimal) -> None:
    if value > RAG_COMPONENT_CEILING_USD:
        raise RagBudgetExceededError


def _validate_frozen_prices() -> None:
    for value in (
        RAG_COMPONENT_CEILING_USD,
        QUERY_EMBEDDING_INPUT_USD_PER_1M,
        ANSWER_INPUT_USD_PER_1M,
        ANSWER_OUTPUT_USD_PER_1M,
    ):
        _price_identity(value)


def _price_identity(value: object) -> str:
    if (
        type(value) is not Decimal
        or not value.is_finite()
        or value <= 0
        or value.as_tuple().exponent != -6
        or value.is_signed()
    ):
        raise ValueError('RAG price registry is invalid')
    return format(value, 'f')
