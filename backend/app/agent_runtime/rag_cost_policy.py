from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext
from threading import RLock
from typing import Final, NamedTuple
from weakref import WeakKeyDictionary, WeakSet

import tiktoken
import tiktoken._tiktoken as _tiktoken

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
from backend.app.rag.evidence_projection import verify_answer_model_influence
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
_QUERY_TOKENIZER_REGISTRY_FINGERPRINT: Final = (
    'a65dff0b0e3cfda66925523c59433c0e929e02454ddd43f8cafe9de3d99f0fd3'
)
_ANSWER_TOKENIZER_REGISTRY_FINGERPRINT: Final = (
    'a3f5661718c52b8999530b4865249f2a51f7e69117b983191553a7805a8b8e08'
)
_MAX_TOKENIZER_TOKENS: Final = 250_000
_MAX_TOKENIZER_DEFINITION_BYTES: Final = 64 * 1_024 * 1_024
_MAX_TOKENIZER_PATTERN_BYTES: Final = 65_536
_MAX_TOKENIZER_SPECIAL_TOKENS: Final = 256
_MAX_TOKENIZER_SPECIAL_TEXT_BYTES: Final = 4_096


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


@dataclass(frozen=True, slots=True)
class RagAnswerCeilingReservation:
    """Conservative admission ceiling, never a rendered prompt/dispatch budget."""

    component: str
    estimated_input_tokens: int
    maximum_output_tokens: int
    reserved_cost_usd: Decimal
    cost_policy_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class RagUnusedComponentReservation:
    """Zero-cost route-inapplicable component; contains no prepared prompt."""

    component: RagPaidComponent
    cost_policy_snapshot_hmac: str
    estimated_input_tokens: int = 0
    maximum_output_tokens: int = 0
    reserved_cost_usd: Decimal = Decimal('0.000000')


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


def _create_rag_cost_policy_type(
    *,
    test_hook: Callable[[str, object, Callable[[str], None]], None] | None = None,
) -> type:
    class Authority(NamedTuple):
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
        query_tokenizer_fingerprint: str
        answer_tokenizer_fingerprint: str

    class PrivateTokenizer(NamedTuple):
        encoding: tiktoken.Encoding
        core: _tiktoken.CoreBPE
        definition_fingerprint: str

    class InfluenceFingerprintSettings(NamedTuple):
        agent_runtime_fingerprint_secret: str
        agent_runtime_fingerprint_key_version: str
        paraworks_env: str

    class State:
        __slots__ = (
            'answer_config_hmac',
            'answer_artifact_signer',
            'answer_estimator_signer',
            'answer_influence_verifier',
            'answer_policy_hmac',
            'answer_tokenizer',
            'authority',
            'query_config_hmac',
            'query_estimator_signer',
            'query_policy_hmac',
            'query_tokenizer',
            'schema_verifier',
        )

        def __init__(self, **values: object) -> None:
            for name, value in values.items():
                object.__setattr__(self, name, value)

        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError('RAG cost policy state is immutable')

        def __repr__(self) -> str:
            return '<RagCostPolicyState redacted>'

        def __copy__(self) -> None:
            raise TypeError('RAG cost policy state is non-transferable')

        def __deepcopy__(self, memo: object) -> None:
            raise TypeError('RAG cost policy state is non-transferable')

        def __reduce_ex__(self, protocol: int) -> None:
            raise TypeError('RAG cost policy state is non-transferable')

    states: WeakKeyDictionary[object, object] = WeakKeyDictionary()
    issued: WeakSet[object] = WeakSet()
    state_lock = RLock()

    def get_state(policy: object, unavailable_code: str) -> State:
        if type(policy) is not Policy:
            raise RagPolicyUnavailableError(unavailable_code)
        with state_lock:
            state = states.get(policy)
        if type(state) is not State:
            raise RagPolicyUnavailableError(unavailable_code)
        return state

    def require_same_state(
        policy: object,
        state: State,
        unavailable_code: str,
    ) -> None:
        with state_lock:
            unchanged = type(policy) is Policy and states.get(policy) is state
        if not unchanged:
            raise RagPolicyUnavailableError(unavailable_code)

    def invoke_hook(point: str, policy: object) -> None:
        if test_hook is None:
            return

        def invalidate(mode: str) -> None:
            with state_lock:
                if mode == 'remove':
                    states.pop(policy, None)
                elif mode == 'replace':
                    states[policy] = object()
                else:
                    raise ValueError('test invalidation mode is invalid')

        test_hook(point, policy, invalidate)

    class Policy:
        __slots__ = ('__weakref__',)

        def __init_subclass__(cls, **kwargs: object) -> None:
            raise TypeError('RAG cost policy subclassing is forbidden')

        def __init__(
            self,
            *,
            settings: Settings,
            answer_output_schema_hmac: str,
            answer_prompt_renderer_hmac: str,
        ) -> None:
            if type(self) is not Policy or type(settings) is not Settings:
                raise ValueError('RAG cost policy settings are invalid')
            with state_lock:
                if self in issued:
                    raise ValueError('RAG cost policy is already initialized')
            if not is_lower_hex_64(
                answer_output_schema_hmac
            ) or not is_lower_hex_64(answer_prompt_renderer_hmac):
                raise ValueError('RAG answer policy identities are invalid')
            authority = Authority(*_validated_cost_authority_values())
            query_tokenizer = PrivateTokenizer(*_clone_approved_tokenizer(
                authority.query_tokenizer_encoding,
                expected_fingerprint=authority.query_tokenizer_fingerprint,
            ))
            answer_tokenizer = PrivateTokenizer(*_clone_approved_tokenizer(
                authority.answer_tokenizer_encoding,
                expected_fingerprint=authority.answer_tokenizer_fingerprint,
            ))
            secret, key_version = fingerprint_secret_bytes(settings)
            key_material_verifier = fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            )
            query_config_hmac = (
                build_query_embedding_model_config_snapshot_hmac(settings)
            )
            answer_config_hmac = build_rag_answer_model_config_snapshot_hmac(
                settings,
                output_schema_hmac=answer_output_schema_hmac,
                prompt_renderer_hmac=answer_prompt_renderer_hmac,
            )
            query_policy_hmac = _build_query_policy_hmac(
                authority=authority,
                secret=secret,
                key_version=key_version,
                key_material_verifier=key_material_verifier,
                query_config_hmac=query_config_hmac,
            )
            answer_policy_hmac = _build_answer_policy_hmac(
                authority=authority,
                secret=secret,
                key_version=key_version,
                key_material_verifier=key_material_verifier,
                answer_config_hmac=answer_config_hmac,
                answer_output_schema_hmac=answer_output_schema_hmac,
                answer_prompt_renderer_hmac=answer_prompt_renderer_hmac,
            )

            def schema_verifier(
                exact_schema_json: bytes,
                after_hmac: Callable[[], None],
            ) -> bool:
                observed = _answer_output_schema_hmac(
                    secret,
                    exact_schema_json,
                )
                after_hmac()
                return hmac.compare_digest(
                    observed,
                    answer_output_schema_hmac,
                )

            def query_estimator_signer(payload: object) -> str:
                return keyed_fingerprint(
                    payload,
                    secret=secret,
                    schema_version=(
                        'rag-query-embedding-estimator-input-bytes:v1'
                    ),
                    policy_version=authority.query_estimator_version,
                )

            def answer_estimator_signer(payload: object) -> str:
                return keyed_fingerprint(
                    payload,
                    secret=secret,
                    schema_version='rag-generation-estimator-input-bytes:v1',
                    policy_version=authority.answer_estimator_version,
                )

            influence_settings = InfluenceFingerprintSettings(
                agent_runtime_fingerprint_secret=(
                    settings.agent_runtime_fingerprint_secret
                ),
                agent_runtime_fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                paraworks_env=settings.paraworks_env,
            )

            def answer_influence_verifier(
                slots: object,
                observations: object,
                prepared_set: object = None,
                db: object = None,
            ) -> None:
                verify_answer_model_influence(
                    slots,
                    observations,
                    settings=influence_settings,
                )
                if prepared_set is not None:
                    from backend.app.rag.evidence_projection import (
                        PreparedModelInfluenceSet,
                        ProjectionFence,
                        _valid_prepared_influence_set,
                    )
                    if type(prepared_set) is not PreparedModelInfluenceSet or prepared_set.observations != observations:
                        raise ValueError('prepared influence set is invalid')
                    fence = ProjectionFence(
                        prepared_corpus_generation=prepared_set.prepared_corpus_generation,
                        current_corpus_generation=prepared_set.prepared_corpus_generation,
                        prepared_index_generation=prepared_set.prepared_index_generation,
                        current_index_generation=prepared_set.prepared_index_generation,
                        prepared_readiness_hmac=prepared_set.prepared_readiness_hmac,
                        current_readiness_hmac=prepared_set.prepared_readiness_hmac)
                    if not _valid_prepared_influence_set(prepared_set, fence=fence, settings=influence_settings):
                        raise ValueError('prepared influence set is invalid')
                    if db is not None and prepared_set.graph_paths:
                        from backend.app.rag.neo4j_retriever import validate_graph_paths
                        if prepared_set.graph_scope is None or not validate_graph_paths(
                            db, prepared_set.graph_paths, settings=influence_settings,
                            scope=prepared_set.graph_scope,
                        ):
                            raise ValueError('graph influence changed before send')

            def answer_artifact_signer(kind: str, payload: object) -> str:
                domains = {
                    'rendered_input': (
                        'rag-rendered-model-input-bytes:v1',
                        'rag-answer:v2',
                    ),
                    'prepared_input': (
                        'rag-prepared-answer-input:v1',
                        'rag-answer:v2',
                    ),
                    'prepared_invocation': (
                        'rag-prepared-answer-invocation:v1',
                        'rag-answer:v2',
                    ),
                    'block_text': (
                        'rag-answer-block-text-bytes:v1',
                        'rag-answer:v2',
                    ),
                    'block_result': (
                        'rag-answer-block-result:v1',
                        'rag-answer-block-confidence:v1',
                    ),
                    'audit_set': (
                        'rag-answer-block-audit-set:v1',
                        'rag-answer-block-confidence:v1',
                    ),
                    'assembled': (
                        'rag-assembled-answer-bytes:v1',
                        'rag-answer-block-joiner:v1',
                    ),
                }
                if type(kind) is not str or kind not in domains:
                    raise ValueError('RAG answer artifact domain is invalid')
                schema_version, policy_version = domains[kind]
                return keyed_fingerprint(
                    payload,
                    secret=secret,
                    schema_version=schema_version,
                    policy_version=policy_version,
                )

            state = State(
                authority=authority,
                schema_verifier=schema_verifier,
                query_estimator_signer=query_estimator_signer,
                answer_estimator_signer=answer_estimator_signer,
                answer_influence_verifier=answer_influence_verifier,
                answer_artifact_signer=answer_artifact_signer,
                query_config_hmac=query_config_hmac,
                answer_config_hmac=answer_config_hmac,
                query_policy_hmac=query_policy_hmac,
                answer_policy_hmac=answer_policy_hmac,
                query_tokenizer=query_tokenizer,
                answer_tokenizer=answer_tokenizer,
            )
            with state_lock:
                if self in issued:
                    raise ValueError('RAG cost policy is already initialized')
                states[self] = state
                issued.add(self)

        def __copy__(self) -> None:
            raise TypeError('RAG cost policy is non-transferable')

        def __deepcopy__(self, memo: object) -> None:
            raise TypeError('RAG cost policy is non-transferable')

        def __reduce_ex__(self, protocol: int) -> None:
            raise TypeError('RAG cost policy is non-transferable')

        @property
        def query_embedding_model_config_snapshot_hmac(self) -> str:
            code = 'retriever_not_configured'
            state = get_state(self, code)
            value = state.query_config_hmac
            invoke_hook('query_property_before_return', self)
            require_same_state(self, state, code)
            return value

        @property
        def answer_model_config_snapshot_hmac(self) -> str:
            code = 'model_unavailable'
            state = get_state(self, code)
            value = state.answer_config_hmac
            invoke_hook('answer_property_before_return', self)
            require_same_state(self, state, code)
            return value

        def authorized_policy_snapshot_hmac(
            self,
            component: RagPaidComponent,
        ) -> str:
            selected = _exact_component(component)
            code = (
                'retriever_not_configured'
                if selected == 'query_embedding'
                else 'model_unavailable'
            )
            state = get_state(self, code)
            value = (
                state.query_policy_hmac
                if selected == 'query_embedding'
                else state.answer_policy_hmac
            )
            invoke_hook(f'{selected}_policy_before_return', self)
            require_same_state(self, state, code)
            return value

        def require_authorized_policy_snapshot(
            self,
            component: RagPaidComponent,
            observed_hmac: str,
        ) -> None:
            selected = _exact_component(component)
            code = (
                'retriever_not_configured'
                if selected == 'query_embedding'
                else 'model_unavailable'
            )
            state = get_state(self, code)
            expected = (
                state.query_policy_hmac
                if selected == 'query_embedding'
                else state.answer_policy_hmac
            )
            if not is_lower_hex_64(observed_hmac) or not hmac.compare_digest(
                observed_hmac,
                expected,
            ):
                raise RagPolicyUnavailableError(code)
            require_same_state(self, state, code)

        def prepare_query_embedding(
            self,
            value: QueryEmbeddingCostInput,
        ) -> PreparedPaidCallBudget:
            code = 'retriever_not_configured'
            state = get_state(self, code)
            if (
                type(value) is not QueryEmbeddingCostInput
                or type(value.retrieval_query_utf8) is not bytes
                or not is_lower_hex_64(value.model_config_snapshot_hmac)
            ):
                raise ValueError('query embedding cost input is invalid')
            if not hmac.compare_digest(
                value.model_config_snapshot_hmac,
                state.query_config_hmac,
            ):
                raise RagPolicyUnavailableError(code)
            try:
                text = value.retrieval_query_utf8.decode(
                    'utf-8',
                    errors='strict',
                )
            except UnicodeDecodeError:
                raise ValueError(
                    'query embedding input must be strict UTF-8'
                ) from None
            authority = state.authority
            encoded_tokens = _count_private_tokens(
                state.query_tokenizer.encoding,
                state.query_tokenizer.core,
                state.query_tokenizer.definition_fingerprint,
                text,
                unavailable_code=code,
                after_encode=lambda: invoke_hook(
                    'query_encode_after_call',
                    self,
                ),
            )
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
            result = PreparedPaidCallBudget(
                component='query_embedding',
                estimated_input_tokens=encoded_tokens,
                maximum_output_tokens=0,
                reserved_cost_usd=reserved,
                cost_policy_snapshot_hmac=state.query_policy_hmac,
                estimator_input_hmac=state.query_estimator_signer(
                    exact_utf8_bytes(text)
                ),
            )
            require_same_state(self, state, code)
            return result

        def prepare_answer_generation(
            self,
            value: AnswerGenerationCostInput,
        ) -> PreparedPaidCallBudget:
            code = 'model_unavailable'
            state = get_state(self, code)
            if (
                type(value) is not AnswerGenerationCostInput
                or type(value.exact_messages_json) is not bytes
                or type(value.exact_response_schema_json) is not bytes
                or not is_lower_hex_64(value.model_config_snapshot_hmac)
            ):
                raise ValueError('answer generation cost input is invalid')
            if not hmac.compare_digest(
                value.model_config_snapshot_hmac,
                state.answer_config_hmac,
            ):
                raise RagPolicyUnavailableError(code)
            messages = _decode_exact_json(
                value.exact_messages_json,
                'messages',
            )
            schema = _decode_exact_json(
                value.exact_response_schema_json,
                'schema',
                require_canonical=False,
            )
            _validate_exact_messages(messages)
            if type(schema) is not dict:
                raise ValueError('answer response schema must be an object')
            if not state.schema_verifier(
                value.exact_response_schema_json,
                lambda: invoke_hook('answer_schema_after_hmac', self),
            ):
                raise RagPolicyUnavailableError(code)
            require_same_state(self, state, code)
            authority = state.authority
            estimator_text = _canonical_json_bytes({
                'messages': messages,
                'model_config_identity': authority.answer_model_config_identity,
                'output_schema': schema,
                'output_schema_identity': 'rag-answer-blocks:v1',
            }).decode('utf-8')
            if len(estimator_text) > authority.max_answer_serialized_chars:
                raise RagBudgetExceededError
            encoded_tokens = _count_private_tokens(
                state.answer_tokenizer.encoding,
                state.answer_tokenizer.core,
                state.answer_tokenizer.definition_fingerprint,
                estimator_text,
                unavailable_code=code,
                after_encode=lambda: invoke_hook(
                    'answer_encode_after_call',
                    self,
                ),
            )
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
            result = PreparedPaidCallBudget(
                component='answer_generation',
                estimated_input_tokens=framed_tokens,
                maximum_output_tokens=authority.max_answer_output_tokens,
                reserved_cost_usd=reserved,
                cost_policy_snapshot_hmac=state.answer_policy_hmac,
                estimator_input_hmac=state.answer_estimator_signer(
                    exact_utf8_bytes(estimator_text)
                ),
            )
            require_same_state(self, state, code)
            return result

        def charge_actual(
            self,
            component: RagPaidComponent,
            usage: StrictProviderUsage,
        ) -> Decimal:
            selected = _exact_component(component)
            code = (
                'retriever_not_configured'
                if selected == 'query_embedding'
                else 'model_unavailable'
            )
            state = get_state(self, code)
            strict_usage = validate_strict_provider_usage(usage)
            authority = state.authority
            if selected == 'query_embedding':
                if strict_usage.output_tokens != 0:
                    raise ValueError('query embedding output usage must be zero')
                result = _rounded_cost(
                    input_tokens=strict_usage.input_tokens,
                    output_tokens=0,
                    input_price=authority.query_input_usd_per_1m,
                    output_price=Decimal('0.000000'),
                    authority=authority,
                )
            else:
                result = _rounded_cost(
                    input_tokens=strict_usage.input_tokens,
                    output_tokens=strict_usage.output_tokens,
                    input_price=authority.answer_input_usd_per_1m,
                    output_price=authority.answer_output_usd_per_1m,
                    authority=authority,
                )
            invoke_hook('charge_before_return', self)
            require_same_state(self, state, code)
            return result

        def usage_exceeds_authorized_cap(
            self,
            component: RagPaidComponent,
            usage: StrictProviderUsage,
        ) -> bool:
            selected = _exact_component(component)
            code = (
                'retriever_not_configured'
                if selected == 'query_embedding'
                else 'model_unavailable'
            )
            state = get_state(self, code)
            strict_usage = validate_strict_provider_usage(usage)
            authority = state.authority
            result = (
                (
                    strict_usage.input_tokens > authority.max_query_tokens
                    or strict_usage.output_tokens != 0
                )
                if selected == 'query_embedding'
                else (
                    strict_usage.input_tokens > authority.max_answer_input_tokens
                    or strict_usage.output_tokens > authority.max_answer_output_tokens
                )
            )
            require_same_state(self, state, code)
            return result

        def reserve_unused_component(self, component: RagPaidComponent) -> RagUnusedComponentReservation:
            return RagUnusedComponentReservation(
                component=_exact_component(component),
                cost_policy_snapshot_hmac=self.authorized_policy_snapshot_hmac(component),
            )

        def reserve_answer_generation(self) -> RagAnswerCeilingReservation:
            state = get_state(self, 'model_unavailable')
            authority = state.authority
            reserved = _rounded_cost(
                input_tokens=authority.max_answer_input_tokens,
                output_tokens=authority.max_answer_output_tokens,
                input_price=authority.answer_input_usd_per_1m,
                output_price=authority.answer_output_usd_per_1m,
                authority=authority,
            )
            _require_component_ceiling(reserved, ceiling=authority.component_ceiling_usd)
            result = RagAnswerCeilingReservation(
                component='answer_generation',
                estimated_input_tokens=authority.max_answer_input_tokens,
                maximum_output_tokens=authority.max_answer_output_tokens,
                reserved_cost_usd=reserved,
                cost_policy_snapshot_hmac=state.answer_policy_hmac,
            )
            require_same_state(self, state, 'model_unavailable')
            return result

        def validate_prepared_budget(
            self,
            prepared: PreparedPaidCallBudget,
        ) -> None:
            if type(prepared) is not PreparedPaidCallBudget:
                raise ValueError('RAG prepared budget is invalid')
            selected = _exact_component(prepared.component)
            code = (
                'retriever_not_configured'
                if selected == 'query_embedding'
                else 'model_unavailable'
            )
            state = get_state(self, code)
            expected_policy = (
                state.query_policy_hmac
                if selected == 'query_embedding'
                else state.answer_policy_hmac
            )
            authority = state.authority
            if (
                not is_lower_hex_64(prepared.cost_policy_snapshot_hmac)
                or not hmac.compare_digest(
                    prepared.cost_policy_snapshot_hmac,
                    expected_policy,
                )
                or not is_lower_hex_64(prepared.estimator_input_hmac)
                or type(prepared.estimated_input_tokens) is not int
                or prepared.estimated_input_tokens < 0
                or type(prepared.maximum_output_tokens) is not int
                or prepared.maximum_output_tokens < 0
                or selected == 'query_embedding'
                and prepared.maximum_output_tokens != 0
                or selected == 'query_embedding'
                and prepared.estimated_input_tokens > authority.max_query_tokens
                or selected == 'answer_generation'
                and prepared.estimated_input_tokens > authority.max_answer_input_tokens
                or selected == 'answer_generation'
                and prepared.maximum_output_tokens > authority.max_answer_output_tokens
            ):
                raise ValueError('RAG prepared budget is invalid')
            expected = _rounded_cost(
                input_tokens=prepared.estimated_input_tokens,
                output_tokens=prepared.maximum_output_tokens,
                input_price=(
                    authority.query_input_usd_per_1m
                    if selected == 'query_embedding'
                    else authority.answer_input_usd_per_1m
                ),
                output_price=(
                    Decimal('0.000000')
                    if selected == 'query_embedding'
                    else authority.answer_output_usd_per_1m
                ),
                authority=authority,
            )
            if (
                type(prepared.reserved_cost_usd) is not Decimal
                or prepared.reserved_cost_usd != expected
                or expected > authority.component_ceiling_usd
            ):
                raise ValueError('RAG prepared budget is invalid')
            require_same_state(self, state, code)

        def sign_answer_artifact(self, kind: str, payload: object) -> str:
            code = 'model_unavailable'
            state = get_state(self, code)
            result = state.answer_artifact_signer(kind, payload)
            require_same_state(self, state, code)
            return result

        def verify_answer_model_influence(
            self,
            slots: object,
            observations: object,
            *, prepared_set: object = None,
            db: object = None,
        ) -> None:
            code = 'model_unavailable'
            state = get_state(self, code)
            state.answer_influence_verifier(slots, observations, prepared_set, db)
            invoke_hook('answer_influence_after_verify', self)
            require_same_state(self, state, code)

    Policy.__name__ = 'RagCostPolicy'
    Policy.__qualname__ = 'RagCostPolicy'
    Policy.__module__ = __name__
    return Policy


RagCostPolicy = _create_rag_cost_policy_type()



def _build_query_policy_hmac(
    *,
    authority: object,
    secret: bytes,
    key_version: str,
    key_material_verifier: str,
    query_config_hmac: str,
) -> str:
    return keyed_fingerprint(
        {
            'authorized_model_config_snapshot_hmac': query_config_hmac,
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
            'fingerprint_key_material_verifier': key_material_verifier,
            'fingerprint_key_version': key_version,
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
        secret=secret,
        schema_version='rag-query-embedding-provider-policy-snapshot:v1',
        policy_version=authority.query_cost_policy_version,
    )


def _build_answer_policy_hmac(
    *,
    authority: object,
    secret: bytes,
    key_version: str,
    key_material_verifier: str,
    answer_config_hmac: str,
    answer_output_schema_hmac: str,
    answer_prompt_renderer_hmac: str,
) -> str:
    return keyed_fingerprint(
        {
            'answer_block_joiner_version': 'rag-answer-block-joiner:v1',
            'authorized_model_config_snapshot_hmac': answer_config_hmac,
            'authorized_model_config_version': (
                authority.answer_model_config_identity
            ),
            'component': 'answer_generation',
            'cost_policy_version': authority.answer_cost_policy_version,
            'credential_scan_policy_version': 'credential-scan:v1',
            'fingerprint_key_material_verifier': key_material_verifier,
            'fingerprint_key_version': key_version,
            'framing_safety_tokens': authority.answer_framing_safety_tokens,
            'input_usd_per_1m': _price_identity(
                authority.answer_input_usd_per_1m
            ),
            'max_input_tokens': authority.max_answer_input_tokens,
            'max_output_tokens': authority.max_answer_output_tokens,
            'max_serialized_input_chars': (
                authority.max_answer_serialized_chars
            ),
            'output_contract_version': 'rag-answer-blocks:v1',
            'output_schema_hmac': answer_output_schema_hmac,
            'output_schema_name': 'rag_answer_blocks_v1',
            'output_usd_per_1m': _price_identity(
                authority.answer_output_usd_per_1m
            ),
            'prompt_version': 'rag-answer:v2',
            'prompt_renderer_hmac': answer_prompt_renderer_hmac,
            'prompt_renderer_version': 'rag-answer-renderer:v1',
            'rag_component_ceiling_usd': _price_identity(
                authority.component_ceiling_usd
            ),
            'reply_priming_tokens': authority.answer_reply_priming_tokens,
            'rounding': authority.rounding_identity,
            'token_count_rule': authority.answer_count_rule,
            'token_estimator_version': authority.answer_estimator_version,
            'tokenizer_encoding': authority.answer_tokenizer_encoding,
        },
        secret=secret,
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


def _decode_exact_json(
    value: bytes,
    name: str,
    *,
    require_canonical: bool = True,
) -> object:
    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if type(key) is not str or key in result:
                raise ValueError
            result[key] = item
        return result

    try:
        text = value.decode('utf-8', errors='strict')
        parsed = json.loads(
            text,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            object_pairs_hook=strict_object,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError):
        raise ValueError(f'answer {name} JSON is invalid') from None
    _validate_json_tree(parsed)
    if require_canonical:
        if _canonical_json_bytes(parsed) != value:
            raise ValueError(f'answer {name} JSON is not canonical')
    elif _insertion_order_json_bytes(parsed) != value:
        raise ValueError(f'answer {name} JSON is not exact')
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


def _insertion_order_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
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
    authority: object,
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


def _validated_cost_authority_values() -> tuple[object, ...]:
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

    return (
        RAG_COMPONENT_CEILING_USD,
        QUERY_EMBEDDING_INPUT_USD_PER_1M,
        QUERY_EMBEDDING_COST_POLICY_VERSION,
        QUERY_EMBEDDING_ESTIMATOR_VERSION,
        QUERY_EMBEDDING_TOKENIZER_ENCODING,
        QUERY_EMBEDDING_COUNT_RULE,
        MAX_QUERY_EMBEDDING_TOKENS,
        ANSWER_INPUT_USD_PER_1M,
        ANSWER_OUTPUT_USD_PER_1M,
        ANSWER_COST_POLICY_VERSION,
        ANSWER_ESTIMATOR_VERSION,
        ANSWER_TOKENIZER_ENCODING,
        ANSWER_COUNT_RULE,
        MAX_ANSWER_SERIALIZED_CHARS,
        MAX_ANSWER_INPUT_TOKENS,
        MAX_ANSWER_OUTPUT_TOKENS,
        ANSWER_REPLY_PRIMING_TOKENS,
        ANSWER_FRAMING_SAFETY_TOKENS,
        ROUNDING_IDENTITY,
        _MILLION,
        _SIX_PLACES,
        ROUND_CEILING,
        RAG_ANSWER_MODEL_CONFIG_VERSION,
        _QUERY_TOKENIZER_REGISTRY_FINGERPRINT,
        _ANSWER_TOKENIZER_REGISTRY_FINGERPRINT,
    )


def _clone_approved_tokenizer(
    name: str,
    *,
    expected_fingerprint: str,
) -> tuple[tiktoken.Encoding, _tiktoken.CoreBPE, str]:
    try:
        source = tiktoken.get_encoding(name)
    except Exception:
        raise ValueError('RAG tokenizer is unavailable') from None
    if (
        type(source) is not tiktoken.Encoding
        or type(source._core_bpe) is not _tiktoken.CoreBPE
        or type(source.name) is not str
        or source.name != name
    ):
        raise ValueError('RAG tokenizer identity is invalid')
    snapshot = _tokenizer_definition_snapshot(source)
    observed_fingerprint = _tokenizer_snapshot_fingerprint(snapshot)
    if not hmac.compare_digest(observed_fingerprint, expected_fingerprint):
        raise ValueError('RAG tokenizer registry is not approved')
    _, pattern, mergeable_ranks, special_tokens, _, _ = snapshot
    try:
        clone = tiktoken.Encoding(
            name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
    except Exception:
        raise ValueError('RAG tokenizer clone is unavailable') from None
    if (
        type(clone) is not tiktoken.Encoding
        or clone is source
        or type(clone._core_bpe) is not _tiktoken.CoreBPE
    ):
        raise ValueError('RAG tokenizer clone identity is invalid')
    clone_fingerprint = _tokenizer_definition_fingerprint(clone)
    if not hmac.compare_digest(clone_fingerprint, expected_fingerprint):
        raise ValueError('RAG tokenizer clone definition is invalid')
    return clone, clone._core_bpe, clone_fingerprint


def _count_private_tokens(
    encoding: tiktoken.Encoding,
    core: _tiktoken.CoreBPE,
    definition_fingerprint: str,
    text: str,
    *,
    unavailable_code: str,
    after_encode: Callable[[], None] | None = None,
) -> int:
    tokenizer = _verified_private_encoding(
        encoding,
        core,
        definition_fingerprint,
        unavailable_code,
    )
    try:
        tokens = tokenizer.encode(
            text,
            allowed_special=set(),
            disallowed_special=(),
        )
    except Exception:
        raise RagPolicyUnavailableError(unavailable_code) from None
    if after_encode is not None:
        after_encode()
    _verified_private_encoding(
        encoding,
        core,
        definition_fingerprint,
        unavailable_code,
    )
    if type(tokens) is not list or any(type(token) is not int for token in tokens):
        raise RagPolicyUnavailableError(unavailable_code)
    return len(tokens)


def _verified_private_encoding(
    encoding: object,
    core: object,
    definition_fingerprint: object,
    unavailable_code: str,
) -> tiktoken.Encoding:
    try:
        if (
            type(encoding) is not tiktoken.Encoding
            or type(core) is not _tiktoken.CoreBPE
            or type(encoding._core_bpe) is not _tiktoken.CoreBPE
            or encoding._core_bpe is not core
            or type(definition_fingerprint) is not str
        ):
            raise ValueError
        observed = _tokenizer_definition_fingerprint(encoding)
    except (AttributeError, TypeError, ValueError):
        raise RagPolicyUnavailableError(unavailable_code) from None
    if not hmac.compare_digest(observed, definition_fingerprint):
        raise RagPolicyUnavailableError(unavailable_code)
    return encoding


def _tokenizer_definition_fingerprint(
    tokenizer: tiktoken.Encoding,
) -> str:
    return _tokenizer_snapshot_fingerprint(
        _tokenizer_definition_snapshot(tokenizer)
    )


def _tokenizer_definition_snapshot(
    tokenizer: tiktoken.Encoding,
) -> tuple[
    str,
    str,
    dict[bytes, int],
    dict[str, int],
    int,
    tuple[bytes, ...],
]:
    try:
        name = tokenizer.name
        pattern = tokenizer._pat_str
        mergeable_ranks = tokenizer._mergeable_ranks
        special_tokens = tokenizer._special_tokens
        core = tokenizer._core_bpe
        token_byte_values = core.token_byte_values()
        max_token_value = tokenizer.max_token_value
    except Exception:
        raise ValueError('RAG tokenizer definition is unavailable') from None
    if (
        type(tokenizer) is not tiktoken.Encoding
        or type(core) is not _tiktoken.CoreBPE
        or type(name) is not str
        or type(pattern) is not str
        or type(mergeable_ranks) is not dict
        or type(special_tokens) is not dict
        or type(token_byte_values) is not list
        or not token_byte_values
        or len(mergeable_ranks) > _MAX_TOKENIZER_TOKENS
        or len(token_byte_values) > _MAX_TOKENIZER_TOKENS
        or len(special_tokens) > _MAX_TOKENIZER_SPECIAL_TOKENS
        or type(max_token_value) is not int
        or max_token_value < 0
    ):
        raise ValueError('RAG tokenizer definition is invalid')

    name_bytes = name.encode('utf-8', errors='strict')
    pattern_bytes = pattern.encode('utf-8', errors='strict')
    if len(pattern_bytes) > _MAX_TOKENIZER_PATTERN_BYTES:
        raise ValueError('RAG tokenizer definition is invalid')
    total_bytes = len(name_bytes) + len(pattern_bytes)
    copied_ranks: dict[bytes, int] = {}
    seen_ranks: set[int] = set()
    try:
        rank_items = tuple(mergeable_ranks.items())
    except Exception:
        raise ValueError('RAG tokenizer definition is unstable') from None
    if len(rank_items) != len(mergeable_ranks):
        raise ValueError('RAG tokenizer definition is unstable')
    for token_bytes, rank in rank_items:
        if (
            type(token_bytes) is not bytes
            or type(rank) is not int
            or rank < 0
            or rank >= 2**64
            or rank in seen_ranks
        ):
            raise ValueError('RAG tokenizer definition is invalid')
        total_bytes += len(token_bytes)
        if total_bytes > _MAX_TOKENIZER_DEFINITION_BYTES:
            raise ValueError('RAG tokenizer definition is invalid')
        copied_ranks[token_bytes] = rank
        seen_ranks.add(rank)

    copied_token_values: list[bytes] = []
    for token_bytes in token_byte_values:
        if type(token_bytes) is not bytes:
            raise ValueError('RAG tokenizer definition is invalid')
        total_bytes += len(token_bytes)
        if total_bytes > _MAX_TOKENIZER_DEFINITION_BYTES:
            raise ValueError('RAG tokenizer definition is invalid')
        copied_token_values.append(token_bytes)

    copied_specials: dict[str, int] = {}
    for token, rank in special_tokens.items():
        if (
            type(token) is not str
            or type(rank) is not int
            or rank < 0
            or rank >= 2**64
        ):
            raise ValueError('RAG tokenizer definition is invalid')
        token_bytes = token.encode('utf-8', errors='strict')
        if len(token_bytes) > _MAX_TOKENIZER_SPECIAL_TEXT_BYTES:
            raise ValueError('RAG tokenizer definition is invalid')
        total_bytes += len(token_bytes)
        if total_bytes > _MAX_TOKENIZER_DEFINITION_BYTES:
            raise ValueError('RAG tokenizer definition is invalid')
        copied_specials[token] = rank
    return (
        name,
        pattern,
        copied_ranks,
        copied_specials,
        max_token_value,
        tuple(copied_token_values),
    )


def _tokenizer_snapshot_fingerprint(
    snapshot: tuple[
        str,
        str,
        dict[bytes, int],
        dict[str, int],
        int,
        tuple[bytes, ...],
    ],
) -> str:
    name, pattern, mergeable_ranks, special_tokens, max_token_value, core_tokens = (
        snapshot
    )
    digest = hashlib.sha256(b'rag-tokenizer-definition:v2\0')
    _update_sized_digest(digest, name.encode('utf-8', errors='strict'))
    _update_sized_digest(digest, pattern.encode('utf-8', errors='strict'))
    rank_items = sorted(
        mergeable_ranks.items(),
        key=lambda item: (item[1], item[0]),
    )
    digest.update(len(rank_items).to_bytes(8, 'big'))
    for token_bytes, rank in rank_items:
        _update_sized_digest(digest, token_bytes)
        digest.update(rank.to_bytes(8, 'big'))
    digest.update(len(core_tokens).to_bytes(8, 'big'))
    for token_bytes in core_tokens:
        _update_sized_digest(digest, token_bytes)
    special_items = sorted(special_tokens.items())
    digest.update(len(special_items).to_bytes(8, 'big'))
    for token, rank in special_items:
        token_bytes = token.encode('utf-8', errors='strict')
        _update_sized_digest(digest, token_bytes)
        digest.update(rank.to_bytes(8, 'big'))
    digest.update(max_token_value.to_bytes(8, 'big'))
    return digest.hexdigest()


def _update_sized_digest(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, 'big'))
    digest.update(value)


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
