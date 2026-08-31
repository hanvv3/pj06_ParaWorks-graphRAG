from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter_ns
from typing import Literal, Protocol

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_contracts import ProviderDispatchPermit
from backend.app.agent_runtime.rag_v2_identity import (
    exact_utf8_bytes,
    verify_serialized_security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.rag.embeddings import validate_query_embedding_vector
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import (
    EmbeddingUsageParserPort,
    PreparedPaidCallBudget,
    PreparedQueryEmbedding,
    QueryEmbeddingCallResult,
    QueryEmbeddingCostInput,
    QueryEmbeddingReceipt,
    RagCostPolicyPort,
    RetrievalRequest,
    StrictProviderUsage,
)

_MODEL = 'text-embedding-3-small'
_DIMENSIONS = 1536
_MODEL_CONFIG_VERSION = 'rag-query-embedding-config:v1'
_PROVIDER_POLICY_VERSION = 'openai-embeddings-api:v1'
_PAYLOAD_VALIDATOR_VERSION = 'rag-query-embedding-payload:v1'
_INDEX_POLICY_VERSION = 'rag-v2-serving-index:v1'

QueryEmbeddingErrorOutcome = Literal[
    'retriever_unavailable',
    'provider_response_identity_invalid',
    'provider_embedding_payload_invalid',
    'provider_usage_overrun',
    'provider_safety_unavailable',
]


class QueryEmbeddingTransportPort(Protocol):
    def dispatch(self, prepared: PreparedQueryEmbedding) -> object: ...


@dataclass(frozen=True, slots=True)
class _ParsedUsage:
    usage: StrictProviderUsage | None
    actual_cost_usd: Decimal | None


class QueryEmbeddingDispatchError(RuntimeError):
    """Sanitized typed failure; raw provider details never cross this boundary."""

    def __init__(
        self,
        outcome: QueryEmbeddingErrorOutcome,
        *,
        prepared: PreparedQueryEmbedding,
        receipt: QueryEmbeddingReceipt,
    ) -> None:
        super().__init__(outcome)
        self.outcome = outcome
        self.prepared = prepared
        self.receipt = receipt
        self.vector = None


class StrictQueryEmbeddingAdapter:
    def __init__(
        self,
        *,
        usage_parser: EmbeddingUsageParserPort,
        cost_policy: RagCostPolicyPort,
        transport: QueryEmbeddingTransportPort,
        settings: Settings,
    ) -> None:
        self._usage_parser = usage_parser
        self._cost_policy = cost_policy
        self._transport = transport
        self._settings = settings

    def prepare(
        self,
        request: RetrievalRequest,
        readiness: RagServingIndexReadiness,
    ) -> PreparedQueryEmbedding:
        verify_serialized_security_scope_fingerprint(
            request.security_scope,
            serialized_fingerprint=request.security_scope_fingerprint,
            settings=self._settings,
        )
        if (
            not readiness.ready
            or type(readiness.corpus_generation) is not int
            or readiness.corpus_generation < 0
            or type(readiness.vector_index_generation) is not int
            or readiness.vector_index_generation < 0
            or not _is_lower_hex_64(readiness.readiness_snapshot_hmac)
            or readiness.embedding_model != _MODEL
            or readiness.embedding_dimensions != _DIMENSIONS
            or readiness.index_policy_version != _INDEX_POLICY_VERSION
        ):
            raise ValueError('serving index is not ready for query embedding')
        query_utf8 = request.retrieval_query_text.encode('utf-8', errors='strict')
        model_snapshot_hmac = _model_config_snapshot_hmac(self._settings)
        budget = self._cost_policy.prepare_query_embedding(
            QueryEmbeddingCostInput(
                retrieval_query_utf8=query_utf8,
                model_config_snapshot_hmac=model_snapshot_hmac,
            )
        )
        _validate_budget(budget)
        secret, _ = fingerprint_secret_bytes(self._settings)
        query_hmac = keyed_fingerprint(
            exact_utf8_bytes(request.retrieval_query_text),
            secret=secret,
            schema_version='rag-query-embedding-query:v1',
            policy_version=_MODEL_CONFIG_VERSION,
        )
        provider_policy_hmac = keyed_fingerprint(
            {
                'provider': 'openai',
                'protocol_family': _PROVIDER_POLICY_VERSION,
                'payload_validator': _PAYLOAD_VALIDATOR_VERSION,
                'max_provider_attempts': 1,
            },
            secret=secret,
            schema_version='rag-query-embedding-provider-policy:v1',
            policy_version=_PROVIDER_POLICY_VERSION,
        )
        attempt_fence_hmac = keyed_fingerprint(
            {
                'query_hmac': query_hmac,
                'corpus_generation': readiness.corpus_generation,
                'vector_index_generation': readiness.vector_index_generation,
                'readiness_snapshot_hmac': readiness.readiness_snapshot_hmac,
                'model_config_snapshot_hmac': model_snapshot_hmac,
                'provider_policy_snapshot_hmac': provider_policy_hmac,
                'cost_policy_snapshot_hmac': budget.cost_policy_snapshot_hmac,
                'estimator_input_hmac': budget.estimator_input_hmac,
            },
            secret=secret,
            schema_version='rag-query-embedding-attempt-fence:v1',
            policy_version=_MODEL_CONFIG_VERSION,
        )
        return PreparedQueryEmbedding(
            retrieval_query_hmac=query_hmac,
            transient_query_utf8=query_utf8,
            corpus_generation=readiness.corpus_generation,
            vector_index_generation=readiness.vector_index_generation,
            readiness_snapshot_hmac=readiness.readiness_snapshot_hmac,
            model_config_snapshot_hmac=model_snapshot_hmac,
            provider_policy_snapshot_hmac=provider_policy_hmac,
            estimated_input_tokens=budget.estimated_input_tokens,
            reserved_cost_usd=budget.reserved_cost_usd,
            attempt_fence_hmac=attempt_fence_hmac,
            budget=budget,
        )

    def dispatch_once(
        self,
        prepared: PreparedQueryEmbedding,
        permit: ProviderDispatchPermit,
    ) -> QueryEmbeddingCallResult:
        permit.consume_at_dispatch()
        started_ns = perf_counter_ns()
        try:
            response = self._transport.dispatch(prepared)
        except Exception:
            raise self._failure(
                'retriever_unavailable',
                prepared=prepared,
                started_ns=started_ns,
                input_tokens=0,
                charge=prepared.reserved_cost_usd,
            ) from None

        usage = self._parse_usage_once(response)
        if _is_overrun(prepared, usage):
            assert usage.usage is not None
            assert usage.actual_cost_usd is not None
            raise self._failure(
                'provider_usage_overrun',
                prepared=prepared,
                started_ns=started_ns,
                input_tokens=usage.usage.input_tokens,
                charge=usage.actual_cost_usd,
            )
        item = _validated_response_identity(response)
        if item is None:
            raise self._failure_from_parsed(
                'provider_response_identity_invalid',
                prepared=prepared,
                parsed=usage,
                started_ns=started_ns,
            )
        try:
            vector = validate_query_embedding_vector(
                item.get('embedding'),
                expected_dimensions=_DIMENSIONS,
            )
        except (TypeError, ValueError):
            raise self._failure_from_parsed(
                'provider_embedding_payload_invalid',
                prepared=prepared,
                parsed=usage,
                started_ns=started_ns,
            ) from None
        if usage.usage is None or usage.actual_cost_usd is None:
            raise self._failure(
                'provider_safety_unavailable',
                prepared=prepared,
                started_ns=started_ns,
                input_tokens=0,
                charge=prepared.reserved_cost_usd,
            )
        latency_ms = _latency_ms(started_ns)
        receipt = QueryEmbeddingReceipt(
            attempted=True,
            input_tokens=usage.usage.input_tokens,
            actual_cost_usd=usage.actual_cost_usd,
            latency_ms=latency_ms,
            outcome='component_succeeded',
            model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
            provider_policy_snapshot_hmac=prepared.provider_policy_snapshot_hmac,
        )
        return QueryEmbeddingCallResult(
            prepared=prepared,
            vector=vector,
            attempted=True,
            validated_input_tokens=usage.usage.input_tokens,
            actual_cost_usd=usage.actual_cost_usd,
            receipt=receipt,
        )

    def _parse_usage_once(self, response: object) -> _ParsedUsage:
        raw_usage = response.get('usage') if isinstance(response, Mapping) else None
        try:
            usage = self._usage_parser.parse_usage(raw_usage)
            if not isinstance(usage, StrictProviderUsage):
                raise TypeError
            if max(usage.input_tokens, usage.output_tokens, usage.total_tokens) > (
                2**63 - 1
            ):
                raise ValueError
            actual = self._cost_policy.charge_actual('query_embedding', usage)
            if (
                type(actual) is not Decimal
                or not actual.is_finite()
                or actual < Decimal('0')
                or not _is_numeric_24_6_representable(actual)
            ):
                raise ValueError
            return _ParsedUsage(usage=usage, actual_cost_usd=actual)
        except Exception:
            return _ParsedUsage(usage=None, actual_cost_usd=None)

    def _failure_from_parsed(
        self,
        outcome: QueryEmbeddingErrorOutcome,
        *,
        prepared: PreparedQueryEmbedding,
        parsed: _ParsedUsage,
        started_ns: int,
    ) -> QueryEmbeddingDispatchError:
        if parsed.usage is not None and parsed.actual_cost_usd is not None:
            return self._failure(
                outcome,
                prepared=prepared,
                started_ns=started_ns,
                input_tokens=parsed.usage.input_tokens,
                charge=parsed.actual_cost_usd,
            )
        return self._failure(
            outcome,
            prepared=prepared,
            started_ns=started_ns,
            input_tokens=0,
            charge=prepared.reserved_cost_usd,
        )

    @staticmethod
    def _failure(
        outcome: QueryEmbeddingErrorOutcome,
        *,
        prepared: PreparedQueryEmbedding,
        started_ns: int,
        input_tokens: int,
        charge: Decimal,
    ) -> QueryEmbeddingDispatchError:
        return QueryEmbeddingDispatchError(
            outcome,
            prepared=prepared,
            receipt=QueryEmbeddingReceipt(
                attempted=True,
                input_tokens=input_tokens,
                actual_cost_usd=charge,
                latency_ms=_latency_ms(started_ns),
                outcome=outcome,
                model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
                provider_policy_snapshot_hmac=(
                    prepared.provider_policy_snapshot_hmac
                ),
            ),
        )


def share_query_embedding_result(
    result: QueryEmbeddingCallResult,
    *,
    legacy_query_text: str,
    v2_query_text: str,
) -> QueryEmbeddingCallResult:
    legacy_bytes = legacy_query_text.encode('utf-8', errors='strict')
    v2_bytes = v2_query_text.encode('utf-8', errors='strict')
    if (
        legacy_bytes != v2_bytes
        or result.prepared.transient_query_utf8 != legacy_bytes
    ):
        raise ValueError('shadow query bytes do not match')
    return result


def _validated_response_identity(response: object) -> Mapping[str, object] | None:
    if not isinstance(response, Mapping):
        return None
    if response.get('object') != 'list' or response.get('model') != _MODEL:
        return None
    data = response.get('data')
    if not isinstance(data, list) or len(data) != 1:
        return None
    item = data[0]
    if (
        not isinstance(item, Mapping)
        or item.get('object') != 'embedding'
        or type(item.get('index')) is not int
        or item.get('index') != 0
    ):
        return None
    return item


def _is_overrun(prepared: PreparedQueryEmbedding, parsed: _ParsedUsage) -> bool:
    return bool(
        parsed.usage is not None
        and parsed.actual_cost_usd is not None
        and (
            parsed.usage.input_tokens > prepared.estimated_input_tokens
            or parsed.usage.output_tokens > prepared.budget.maximum_output_tokens
            or parsed.actual_cost_usd > prepared.reserved_cost_usd
        )
    )


def _validate_budget(value: PreparedPaidCallBudget) -> None:
    if (
        value.component != 'query_embedding'
        or type(value.estimated_input_tokens) is not int
        or value.estimated_input_tokens < 0
        or type(value.maximum_output_tokens) is not int
        or value.maximum_output_tokens != 0
        or type(value.reserved_cost_usd) is not Decimal
        or not value.reserved_cost_usd.is_finite()
        or value.reserved_cost_usd < Decimal('0')
        or not _is_numeric_24_6_representable(value.reserved_cost_usd)
        or not _is_lower_hex_64(value.cost_policy_snapshot_hmac)
        or not _is_lower_hex_64(value.estimator_input_hmac)
    ):
        raise ValueError('query embedding budget is invalid')


def _model_config_snapshot_hmac(settings: Settings) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'api_base_url': 'https://api.openai.com/v1',
            'dimensions': _DIMENSIONS,
            'encoding_format': 'float',
            'endpoint_identity': 'openai-direct-standard-global:v1',
            'input_count_per_query_call': 1,
            'max_provider_attempts': 1,
            'model': _MODEL,
            'provider': 'openai',
            'provider_send_start_window_seconds': 5,
            'regional_processing': False,
            'sdk_retry': 0,
            'timeout_seconds': 30,
        },
        secret=secret,
        schema_version='rag-query-embedding-model-config-snapshot:v1',
        policy_version=_MODEL_CONFIG_VERSION,
    )


def _latency_ms(started_ns: int) -> int:
    return max(0, (perf_counter_ns() - started_ns) // 1_000_000)


def _is_lower_hex_64(value: object) -> bool:
    return bool(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value))


def _is_numeric_24_6_representable(value: Decimal) -> bool:
    normalized = value.normalize()
    if normalized.is_zero():
        return True
    return normalized.as_tuple().exponent >= -6 and normalized.adjusted() <= 17
