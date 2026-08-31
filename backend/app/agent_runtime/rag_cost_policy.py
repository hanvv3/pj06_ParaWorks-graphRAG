from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
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
QUERY_EMBEDDING_COUNT_RULE: Final = (
    'exact-query-text:no-normalization:no-chat-overhead:v1'
)
ANSWER_COUNT_RULE: Final = (
    'compact-json-messages-and-exact-schema:no-unicode-normalization:v1'
)
ROUNDING_IDENTITY: Final = 'six-place-ROUND_CEILING'
MAX_QUERY_EMBEDDING_TOKENS: Final = 8_000
MAX_ANSWER_SERIALIZED_CHARS: Final = 12_000
MAX_ANSWER_INPUT_TOKENS: Final = 10_000
MAX_ANSWER_OUTPUT_TOKENS: Final = 512
ANSWER_REPLY_PRIMING_TOKENS: Final = 16
ANSWER_FRAMING_SAFETY_TOKENS: Final = 512
_MILLION: Final = Decimal(1_000_000)
_SIX_PLACES: Final = Decimal('0.000001')


@dataclass(frozen=True, slots=True)
class _RagCostAuthority:
    component_ceiling_usd: Decimal
    query_input_usd_per_1m: Decimal
    query_cost_policy_version: str
    query_estimator_version: str
    query_tokenizer_encoding: str
    query_count_rule: str
    max_query_tokens: int
    answer_input_usd_per_1m: Decimal
    answer_output_usd_per_1m: Decimal
    answer_cost_policy_version: str
    answer_estimator_version: str
    answer_tokenizer_encoding: str
    answer_count_rule: str
    max_answer_serialized_chars: int
    max_answer_input_tokens: int
    max_answer_output_tokens: int
    answer_reply_priming_tokens: int
    answer_framing_safety_tokens: int
    rounding_identity: str
    cost_divisor: Decimal
    rounding_quantum: Decimal
    rounding_mode: str
    answer_model_config_identity: str
    query_tokenizer: object
    answer_tokenizer: object


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
        '_authority',
        '_answer_config_hmac',
        '_answer_policy_hmac',
        '_answer_prompt_renderer_hmac',
        '_answer_schema_hmac',
        '_key_material_verifier',
        '_key_version',
        '_query_config_hmac',
        '_query_policy_hmac',
        '_secret',
        '_sealed',
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, '_sealed', False):
            raise AttributeError('RAG cost policy is immutable')
        object.__setattr__(self, name, value)

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
        authority = _build_cost_authority()
        secret, key_version = fingerprint_secret_bytes(settings)
        object.__setattr__(self, '_authority', authority)
        object.__setattr__(self, '_secret', secret)
        object.__setattr__(self, '_key_version', key_version)
        object.__setattr__(self, '_key_material_verifier', fingerprint_key_material_verifier(
            settings.agent_runtime_fingerprint_secret
        ))
        object.__setattr__(self, '_answer_schema_hmac', answer_output_schema_hmac)
        object.__setattr__(
            self,
            '_answer_prompt_renderer_hmac',
            answer_prompt_renderer_hmac,
        )
        object.__setattr__(
            self,
            '_query_config_hmac',
            build_query_embedding_model_config_snapshot_hmac(settings),
        )
        object.__setattr__(self, '_answer_config_hmac', build_rag_answer_model_config_snapshot_hmac(
            settings,
            output_schema_hmac=answer_output_schema_hmac,
            prompt_renderer_hmac=answer_prompt_renderer_hmac,
        ))
        object.__setattr__(
            self,
            '_query_policy_hmac',
            self._build_query_policy_hmac(),
        )
        object.__setattr__(
            self,
            '_answer_policy_hmac',
            self._build_answer_policy_hmac(),
        )
        object.__setattr__(self, '_sealed', True)

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
        authority = self._authority
        encoded_tokens = len(authority.query_tokenizer.encode(
            text,
            allowed_special=set(),
            disallowed_special=(),
        ))
        if encoded_tokens > authority.max_query_tokens:
            raise RagBudgetExceededError
        reserved = _rounded_cost(
            input_tokens=encoded_tokens,
            output_tokens=0,
            input_price=authority.query_input_usd_per_1m,
            output_price=Decimal('0.000000'),
            authority=authority,
        )
        _require_component_ceiling(
            reserved,
            ceiling=authority.component_ceiling_usd,
        )
        estimator_input_hmac = keyed_fingerprint(
            exact_utf8_bytes(text),
            secret=self._secret,
            schema_version='rag-query-embedding-estimator-input-bytes:v1',
            policy_version=authority.query_estimator_version,
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
            'model_config_identity': self._authority.answer_model_config_identity,
            'output_schema': schema,
            'output_schema_identity': 'rag-answer-blocks:v1',
        })
        estimator_text = estimator_bytes.decode('utf-8')
        authority = self._authority
        if len(estimator_text) > authority.max_answer_serialized_chars:
            raise RagBudgetExceededError
        encoded_tokens = len(authority.answer_tokenizer.encode(
            estimator_text,
            allowed_special=set(),
            disallowed_special=(),
        ))
        framed_tokens = (
            encoded_tokens
            + authority.answer_reply_priming_tokens
            + authority.answer_framing_safety_tokens
        )
        if framed_tokens > authority.max_answer_input_tokens:
            raise RagBudgetExceededError
        reserved = _rounded_cost(
            input_tokens=framed_tokens,
            output_tokens=authority.max_answer_output_tokens,
            input_price=authority.answer_input_usd_per_1m,
            output_price=authority.answer_output_usd_per_1m,
            authority=authority,
        )
        _require_component_ceiling(
            reserved,
            ceiling=authority.component_ceiling_usd,
        )
        estimator_hmac = keyed_fingerprint(
            exact_utf8_bytes(estimator_text),
            secret=self._secret,
            schema_version='rag-generation-estimator-input-bytes:v1',
            policy_version=authority.answer_estimator_version,
        )
        return PreparedPaidCallBudget(
            component='answer_generation',
            estimated_input_tokens=framed_tokens,
            maximum_output_tokens=authority.max_answer_output_tokens,
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
        authority = self._authority
        if selected == 'query_embedding':
            if strict_usage.output_tokens != 0:
                raise ValueError('query embedding output usage must be zero')
            return _rounded_cost(
                input_tokens=strict_usage.input_tokens,
                output_tokens=0,
                input_price=authority.query_input_usd_per_1m,
                output_price=Decimal('0.000000'),
                authority=authority,
            )
        return _rounded_cost(
            input_tokens=strict_usage.input_tokens,
            output_tokens=strict_usage.output_tokens,
            input_price=authority.answer_input_usd_per_1m,
            output_price=authority.answer_output_usd_per_1m,
            authority=authority,
        )

    def _build_query_policy_hmac(self) -> str:
        authority = self._authority
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
                'cost_policy_version': authority.query_cost_policy_version,
                'credential_scan_policy_version': 'credential-scan:v1',
                'embedding_payload_validator_version': (
                    QUERY_EMBEDDING_PAYLOAD_VALIDATOR_VERSION
                ),
                'fingerprint_key_material_verifier': (
                    self._key_material_verifier
                ),
                'fingerprint_key_version': self._key_version,
                'input_usd_per_1m': _price_identity(
                    authority.query_input_usd_per_1m
                ),
                'max_input_tokens': authority.max_query_tokens,
                'rag_component_ceiling_usd': _price_identity(
                    authority.component_ceiling_usd
                ),
                'rounding': authority.rounding_identity,
                'token_count_rule': authority.query_count_rule,
                'token_estimator_version': authority.query_estimator_version,
                'tokenizer_encoding': authority.query_tokenizer_encoding,
            },
            secret=self._secret,
            schema_version='rag-query-embedding-provider-policy-snapshot:v1',
            policy_version=authority.query_cost_policy_version,
        )

    def _build_answer_policy_hmac(self) -> str:
        authority = self._authority
        return keyed_fingerprint(
            {
                'answer_block_joiner_version': 'rag-answer-block-joiner:v1',
                'authorized_model_config_snapshot_hmac': self._answer_config_hmac,
                'authorized_model_config_version': (
                    authority.answer_model_config_identity
                ),
                'component': 'answer_generation',
                'cost_policy_version': authority.answer_cost_policy_version,
                'credential_scan_policy_version': 'credential-scan:v1',
                'fingerprint_key_material_verifier': (
                    self._key_material_verifier
                ),
                'fingerprint_key_version': self._key_version,
                'framing_safety_tokens': (
                    authority.answer_framing_safety_tokens
                ),
                'input_usd_per_1m': _price_identity(
                    authority.answer_input_usd_per_1m
                ),
                'max_input_tokens': authority.max_answer_input_tokens,
                'max_output_tokens': authority.max_answer_output_tokens,
                'max_serialized_input_chars': (
                    authority.max_answer_serialized_chars
                ),
                'output_contract_version': 'rag-answer-blocks:v1',
                'output_schema_hmac': self._answer_schema_hmac,
                'output_schema_name': 'rag_answer_blocks_v1',
                'output_usd_per_1m': _price_identity(
                    authority.answer_output_usd_per_1m
                ),
                'prompt_version': 'rag-answer:v2',
                'prompt_renderer_hmac': self._answer_prompt_renderer_hmac,
                'prompt_renderer_version': 'rag-answer-renderer:v1',
                'rag_component_ceiling_usd': _price_identity(
                    authority.component_ceiling_usd
                ),
                'reply_priming_tokens': (
                    authority.answer_reply_priming_tokens
                ),
                'rounding': authority.rounding_identity,
                'token_count_rule': authority.answer_count_rule,
                'token_estimator_version': authority.answer_estimator_version,
                'tokenizer_encoding': authority.answer_tokenizer_encoding,
            },
            secret=self._secret,
            schema_version='rag-answer-provider-policy-snapshot:v1',
            policy_version=authority.answer_cost_policy_version,
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
    authority: _RagCostAuthority,
) -> Decimal:
    with localcontext() as context:
        context.prec = 64
        value = (
            Decimal(input_tokens) * input_price
            + Decimal(output_tokens) * output_price
        ) / authority.cost_divisor
        return value.quantize(
            authority.rounding_quantum,
            rounding=authority.rounding_mode,
        )


def _require_component_ceiling(value: Decimal, *, ceiling: Decimal) -> None:
    if value > ceiling:
        raise RagBudgetExceededError


def _build_cost_authority() -> _RagCostAuthority:
    prices = (
        RAG_COMPONENT_CEILING_USD,
        QUERY_EMBEDDING_INPUT_USD_PER_1M,
        ANSWER_INPUT_USD_PER_1M,
        ANSWER_OUTPUT_USD_PER_1M,
    )
    for value in prices:
        _price_identity(value)
    exact_strings = (
        QUERY_EMBEDDING_COST_POLICY_VERSION,
        QUERY_EMBEDDING_ESTIMATOR_VERSION,
        QUERY_EMBEDDING_TOKENIZER_ENCODING,
        QUERY_EMBEDDING_COUNT_RULE,
        ANSWER_COST_POLICY_VERSION,
        ANSWER_ESTIMATOR_VERSION,
        ANSWER_TOKENIZER_ENCODING,
        ANSWER_COUNT_RULE,
        ROUNDING_IDENTITY,
        RAG_ANSWER_MODEL_CONFIG_VERSION,
    )
    if any(type(value) is not str or not value for value in exact_strings):
        raise ValueError('RAG cost policy identity registry is invalid')
    positive_integers = (
        MAX_QUERY_EMBEDDING_TOKENS,
        MAX_ANSWER_SERIALIZED_CHARS,
        MAX_ANSWER_INPUT_TOKENS,
        MAX_ANSWER_OUTPUT_TOKENS,
        ANSWER_REPLY_PRIMING_TOKENS,
        ANSWER_FRAMING_SAFETY_TOKENS,
    )
    if any(type(value) is not int or value <= 0 for value in positive_integers):
        raise ValueError('RAG cost policy bounds are invalid')
    if (
        type(_MILLION) is not Decimal
        or Decimal(1_000_000) != _MILLION
        or type(_SIX_PLACES) is not Decimal
        or Decimal('0.000001') != _SIX_PLACES
        or type(ROUND_CEILING) is not str
        or ROUND_CEILING != 'ROUND_CEILING'
    ):
        raise ValueError('RAG cost rounding registry is invalid')

    query_tokenizer = _resolve_exact_tokenizer(
        QUERY_EMBEDDING_TOKENIZER_ENCODING
    )
    answer_tokenizer = _resolve_exact_tokenizer(ANSWER_TOKENIZER_ENCODING)
    return _RagCostAuthority(
        component_ceiling_usd=RAG_COMPONENT_CEILING_USD,
        query_input_usd_per_1m=QUERY_EMBEDDING_INPUT_USD_PER_1M,
        query_cost_policy_version=QUERY_EMBEDDING_COST_POLICY_VERSION,
        query_estimator_version=QUERY_EMBEDDING_ESTIMATOR_VERSION,
        query_tokenizer_encoding=QUERY_EMBEDDING_TOKENIZER_ENCODING,
        query_count_rule=QUERY_EMBEDDING_COUNT_RULE,
        max_query_tokens=MAX_QUERY_EMBEDDING_TOKENS,
        answer_input_usd_per_1m=ANSWER_INPUT_USD_PER_1M,
        answer_output_usd_per_1m=ANSWER_OUTPUT_USD_PER_1M,
        answer_cost_policy_version=ANSWER_COST_POLICY_VERSION,
        answer_estimator_version=ANSWER_ESTIMATOR_VERSION,
        answer_tokenizer_encoding=ANSWER_TOKENIZER_ENCODING,
        answer_count_rule=ANSWER_COUNT_RULE,
        max_answer_serialized_chars=MAX_ANSWER_SERIALIZED_CHARS,
        max_answer_input_tokens=MAX_ANSWER_INPUT_TOKENS,
        max_answer_output_tokens=MAX_ANSWER_OUTPUT_TOKENS,
        answer_reply_priming_tokens=ANSWER_REPLY_PRIMING_TOKENS,
        answer_framing_safety_tokens=ANSWER_FRAMING_SAFETY_TOKENS,
        rounding_identity=ROUNDING_IDENTITY,
        cost_divisor=_MILLION,
        rounding_quantum=_SIX_PLACES,
        rounding_mode=ROUND_CEILING,
        answer_model_config_identity=RAG_ANSWER_MODEL_CONFIG_VERSION,
        query_tokenizer=query_tokenizer,
        answer_tokenizer=answer_tokenizer,
    )


def _resolve_exact_tokenizer(name: str) -> object:
    try:
        tokenizer = tiktoken.get_encoding(name)
        resolved_name = tokenizer.name
    except Exception:
        raise ValueError('RAG tokenizer is unavailable') from None
    if (
        type(tokenizer) is not tiktoken.Encoding
        or type(resolved_name) is not str
        or resolved_name != name
    ):
        raise ValueError('RAG tokenizer identity is invalid')
    return tokenizer


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
