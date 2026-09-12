from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
    QueryEmbeddingDispatchError,
    QueryEmbeddingReadinessError,
    StrictQueryEmbeddingAdapter,
    share_query_embedding_result,
)
from backend.app.core.config import Settings
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import (
    PreparedPaidCallBudget,
    QueryEmbeddingCostInput,
    RetrievalRequest,
    StrictProviderUsage,
    decimal_storage_representation,
)


def _settings() -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret='task-7-embedding-secret-with-32-bytes',
        agent_runtime_fingerprint_key_version='task7-v1',
        openai_embedding_model='text-embedding-3-small',
        openai_embedding_dimensions=1536,
    )


def _scope() -> SecurityScope:
    return SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='user-1',
        workspace_scope_id='workspace-1',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public', 'internal'),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )


def _request(*, fingerprint: str | None = None, query: str = 'exact query') -> RetrievalRequest:
    settings = _settings()
    scope = _scope()
    return RetrievalRequest(
        retrieval_query_text=query,
        security_scope=scope,
        security_scope_fingerprint=(
            fingerprint
            if fingerprint is not None
            else security_scope_fingerprint(scope, settings=settings)
        ),
        query_embedding_result=None,
        candidate_scan_limit=50,
        visible_limit=5,
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )


def _readiness(*, ready: bool = True) -> RagServingIndexReadiness:
    return RagServingIndexReadiness(
        ready=ready,
        corpus_generation=7,
        vector_index_generation=11,
        expected_document_count=3,
        live_vector_count=3,
        tombstone_count=0,
        mismatch_count_capped_at_20=0 if ready else 1,
        embedding_model='text-embedding-3-small',
        embedding_dimensions=1536,
        index_policy_version='rag-v2-serving-index:v1',
        readiness_snapshot_hmac='a' * 64,
    )


def _raw_response(*, usage: object | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        'object': 'list',
        'model': 'text-embedding-3-small',
        'data': [
            {
                'object': 'embedding',
                'index': 0,
                'embedding': [1.0, *([0.0] * 1535)],
            }
        ],
    }
    payload['usage'] = (
        {'prompt_tokens': 3, 'total_tokens': 3}
        if usage is None
        else usage
    )
    return payload


class _UsageParser:
    def __init__(self, result: StrictProviderUsage | Exception) -> None:
        self.result = result
        self.seen: list[object] = []

    def parse_usage(self, usage: object) -> StrictProviderUsage:
        self.seen.append(usage)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _CostPolicy:
    def __init__(
        self,
        *,
        estimated_input_tokens: int = 3,
        reserved: Decimal = Decimal('0.000777'),
        actual: Decimal | Exception = Decimal('0.000123'),
    ) -> None:
        self.estimated_input_tokens = estimated_input_tokens
        self.reserved = reserved
        self.actual = actual
        self.prepared_inputs: list[QueryEmbeddingCostInput] = []
        self.charges: list[tuple[str, StrictProviderUsage]] = []

    def prepare_query_embedding(
        self,
        value: QueryEmbeddingCostInput,
    ) -> PreparedPaidCallBudget:
        self.prepared_inputs.append(value)
        return PreparedPaidCallBudget(
            component='query_embedding',
            estimated_input_tokens=self.estimated_input_tokens,
            maximum_output_tokens=0,
            reserved_cost_usd=self.reserved,
            cost_policy_snapshot_hmac='b' * 64,
            estimator_input_hmac='c' * 64,
        )

    def charge_actual(
        self,
        component: str,
        usage: StrictProviderUsage,
    ) -> Decimal:
        self.charges.append((component, usage))
        if isinstance(self.actual, Exception):
            raise self.actual
        return self.actual


class _Transport:
    def __init__(self, result: object | Exception) -> None:
        self.result = result
        self.calls = 0

    def dispatch(self, prepared) -> object:
        del prepared
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _Permit:
    def __init__(self) -> None:
        self.used = False
        self.uses = 0

    def consume_at_dispatch(self) -> None:
        if self.used:
            raise RuntimeError('permit already consumed')
        self.used = True
        self.uses += 1


def _adapter(
    *,
    parser: _UsageParser | None = None,
    policy: _CostPolicy | None = None,
    transport: _Transport | None = None,
) -> tuple[StrictQueryEmbeddingAdapter, _UsageParser, _CostPolicy, _Transport]:
    parser = parser or _UsageParser(
        StrictProviderUsage(input_tokens=3, output_tokens=0, total_tokens=3)
    )
    policy = policy or _CostPolicy()
    transport = transport or _Transport(_raw_response())
    return (
        StrictQueryEmbeddingAdapter(
            usage_parser=parser,
            cost_policy=policy,
            transport=transport,
            settings=_settings(),
        ),
        parser,
        policy,
        transport,
    )


def test_prepare_verifies_scope_before_cost_or_transport_and_binds_readiness() -> None:
    adapter, _, policy, transport = _adapter()
    with pytest.raises(ValueError, match='fingerprint'):
        adapter.prepare(_request(fingerprint='0' * 64), _readiness())
    assert policy.prepared_inputs == []
    assert transport.calls == 0

    prepared = adapter.prepare(_request(), _readiness())
    assert prepared.transient_query_utf8 == b'exact query'
    assert prepared.corpus_generation == 7
    assert prepared.vector_index_generation == 11
    assert prepared.readiness_snapshot_hmac == 'a' * 64
    assert prepared.estimated_input_tokens == 3


def test_prepare_rejects_not_ready_or_wrong_embedding_identity_without_dispatch() -> None:
    adapter, _, policy, transport = _adapter()
    with pytest.raises(ValueError, match='serving index is not ready'):
        adapter.prepare(_request(), _readiness(ready=False))
    wrong = _readiness()
    object.__setattr__(wrong, 'embedding_model', 'model-alias')
    with pytest.raises(ValueError, match='serving index is not ready'):
        adapter.prepare(_request(), wrong)
    assert policy.prepared_inputs == []
    assert transport.calls == 0


def test_prepare_rejects_malformed_or_bigint_overflow_readiness_before_cost() -> None:
    adapter, _, policy, transport = _adapter()
    overflow = _readiness()
    object.__setattr__(overflow, 'corpus_generation', 2**63)
    with pytest.raises(ValueError, match='serving index readiness'):
        adapter.prepare(_request(), overflow)

    forged_ready = _readiness()
    object.__setattr__(forged_ready, 'ready', 1)
    with pytest.raises(ValueError, match='serving index readiness'):
        adapter.prepare(_request(), forged_ready)

    malformed_hmac = _readiness()
    object.__setattr__(malformed_hmac, 'readiness_snapshot_hmac', 'not-a-hmac')
    with pytest.raises(ValueError, match='serving index readiness'):
        adapter.prepare(_request(), malformed_hmac)

    assert policy.prepared_inputs == []
    assert transport.calls == 0


@pytest.mark.parametrize(
    'mutate',
    (
        lambda value: replace(value, attempt_fence_hmac='0' * 64),
        lambda value: replace(
            value,
            estimated_input_tokens=4,
            budget=replace(value.budget, estimated_input_tokens=4),
        ),
        lambda value: replace(
            value,
            reserved_cost_usd=Decimal('-0.000000'),
            budget=replace(
                value.budget,
                reserved_cost_usd=Decimal('-0.000000'),
            ),
        ),
    ),
)
def test_dispatch_rejects_forged_prepared_before_permit_or_transport(mutate) -> None:
    adapter, _, _, transport = _adapter()
    prepared = mutate(adapter.prepare(_request(), _readiness()))
    permit = _Permit()

    with pytest.raises(ValueError, match='prepared query embedding'):
        adapter.dispatch_once(prepared, permit)

    assert permit.uses == 0
    assert transport.calls == 0


def test_dispatch_passes_raw_usage_once_and_preserves_parser_and_decimal_results() -> None:
    raw_usage = {'opaque': object()}
    usage = StrictProviderUsage(input_tokens=3, output_tokens=0, total_tokens=3)
    actual = Decimal('0.000321')
    parser = _UsageParser(usage)
    policy = _CostPolicy(actual=actual)
    transport = _Transport(_raw_response(usage=raw_usage))
    adapter, _, _, _ = _adapter(parser=parser, policy=policy, transport=transport)
    prepared = adapter.prepare(_request(), _readiness())
    permit = _Permit()

    result = adapter.dispatch_once(prepared, permit)

    assert parser.seen == [raw_usage]
    assert policy.charges == [('query_embedding', usage)]
    assert result.actual_cost_usd is actual
    assert result.validated_input_tokens == 3
    assert result.vector.coordinates[0] == 1.0
    assert len(result.vector.coordinates) == 1536
    assert result.vector.canonical_big_endian_float32_sha256 == (
        'e3fb2d174fad79a02167462d3d2cd9f49143ad375e626258e5a530eda3041a17'
    )
    with pytest.raises(FrozenInstanceError):
        result.vector.coordinates = ()  # type: ignore[misc]
    with pytest.raises(RuntimeError, match='already consumed'):
        adapter.dispatch_once(prepared, permit)
    assert transport.calls == 1


@pytest.mark.parametrize(
    ('mutate', 'outcome'),
    (
        (lambda value: value.update(object='object'), 'provider_response_identity_invalid'),
        (lambda value: value.update(model='text-embedding-3-small-latest'), 'provider_response_identity_invalid'),
        (lambda value: value.update(data=[]), 'provider_response_identity_invalid'),
        (lambda value: value['data'][0].update(object='vector'), 'provider_response_identity_invalid'),
        (lambda value: value['data'][0].update(index=True), 'provider_response_identity_invalid'),
        (lambda value: value['data'][0].update(index=1), 'provider_response_identity_invalid'),
        (lambda value: value['data'][0].update(embedding=[1.0]), 'provider_embedding_payload_invalid'),
        (lambda value: value['data'][0].update(embedding=[True, *([0.0] * 1535)]), 'provider_embedding_payload_invalid'),
        (lambda value: value['data'][0].update(embedding=[float('inf'), *([0.0] * 1535)]), 'provider_embedding_payload_invalid'),
        (lambda value: value['data'][0].update(embedding=[3.5e38, *([0.0] * 1535)]), 'provider_embedding_payload_invalid'),
        (lambda value: value['data'][0].update(embedding=[1e-50, *([0.0] * 1535)]), 'provider_embedding_payload_invalid'),
        (lambda value: value['data'][0].update(embedding=[0.0] * 1536), 'provider_embedding_payload_invalid'),
    ),
)
def test_strict_response_identity_and_vector_matrix(mutate, outcome: str) -> None:
    payload = _raw_response()
    mutate(payload)
    adapter, parser, _, transport = _adapter(transport=_Transport(payload))
    prepared = adapter.prepare(_request(), _readiness())

    with pytest.raises(QueryEmbeddingDispatchError) as caught:
        adapter.dispatch_once(prepared, _Permit())

    assert caught.value.outcome == outcome
    assert caught.value.vector is None
    assert len(parser.seen) == 1
    assert transport.calls == 1


class _EqualityForgingValue:
    def __eq__(self, other: object) -> bool:
        del other
        return True


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ('path', 'value'),
    (
        ('top_object', _EqualityForgingValue()),
        ('top_model', _StringSubclass('text-embedding-3-small')),
        ('item_object', _StringSubclass('embedding')),
    ),
)
def test_response_identity_rejects_equality_forgers_and_str_subclasses(
    path: str,
    value: object,
) -> None:
    payload = _raw_response()
    if path == 'top_object':
        payload['object'] = value
    elif path == 'top_model':
        payload['model'] = value
    else:
        payload['data'][0]['object'] = value
    adapter, _, _, transport = _adapter(transport=_Transport(payload))
    prepared = adapter.prepare(_request(), _readiness())

    with pytest.raises(QueryEmbeddingDispatchError) as caught:
        adapter.dispatch_once(prepared, _Permit())

    assert caught.value.outcome == 'provider_response_identity_invalid'
    assert transport.calls == 1


class _UsageSubclass(StrictProviderUsage):
    def __post_init__(self) -> None:
        pass


@pytest.mark.parametrize(
    'usage',
    (
        _UsageSubclass(input_tokens=3, output_tokens=0, total_tokens=3),
        StrictProviderUsage(input_tokens=3, output_tokens=0, total_tokens=3),
    ),
)
def test_usage_requires_exact_class_and_exact_bounded_arithmetic_without_charge(
    usage: StrictProviderUsage,
) -> None:
    if type(usage) is StrictProviderUsage:
        object.__setattr__(usage, 'input_tokens', True)
    parser = _UsageParser(usage)
    policy = _CostPolicy()
    adapter, _, _, _ = _adapter(parser=parser, policy=policy)
    prepared = adapter.prepare(_request(), _readiness())

    with pytest.raises(QueryEmbeddingDispatchError) as caught:
        adapter.dispatch_once(prepared, _Permit())

    assert caught.value.outcome == 'provider_safety_unavailable'
    assert policy.charges == []


def test_negative_zero_costs_are_never_admitted() -> None:
    adapter, _, policy, transport = _adapter(
        policy=_CostPolicy(reserved=Decimal('-0.000000'))
    )
    with pytest.raises(ValueError, match='budget'):
        adapter.prepare(_request(), _readiness())
    assert len(policy.prepared_inputs) == 1
    assert transport.calls == 0

    policy = _CostPolicy(actual=Decimal('-0.000000'))
    adapter, _, _, _ = _adapter(policy=policy)
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as caught:
        adapter.dispatch_once(prepared, _Permit())
    assert caught.value.outcome == 'provider_safety_unavailable'
    assert decimal_storage_representation(Decimal('0')) == '0'
    assert decimal_storage_representation(Decimal('0.000000')) == '0'


def test_parse_precedence_is_overrun_then_identity_then_vector_then_usage() -> None:
    malformed = _raw_response(usage={'malformed': True})
    malformed['object'] = 'wrong'
    malformed['data'][0]['embedding'] = [0.0] * 1536

    overrun_parser = _UsageParser(
        StrictProviderUsage(input_tokens=4, output_tokens=0, total_tokens=4)
    )
    adapter, _, _, _ = _adapter(
        parser=overrun_parser,
        policy=_CostPolicy(estimated_input_tokens=3, actual=Decimal('0.001')),
        transport=_Transport(malformed),
    )
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as overrun:
        adapter.dispatch_once(prepared, _Permit())
    assert overrun.value.outcome == 'provider_usage_overrun'

    adapter, _, _, _ = _adapter(
        parser=_UsageParser(ValueError('bad usage')),
        transport=_Transport(malformed),
    )
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as identity:
        adapter.dispatch_once(prepared, _Permit())
    assert identity.value.outcome == 'provider_response_identity_invalid'

    vector_only = _raw_response(usage={'malformed': True})
    vector_only['data'][0]['embedding'] = [0.0] * 1536
    adapter, _, _, _ = _adapter(
        parser=_UsageParser(ValueError('bad usage')),
        transport=_Transport(vector_only),
    )
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as vector:
        adapter.dispatch_once(prepared, _Permit())
    assert vector.value.outcome == 'provider_embedding_payload_invalid'

    usage_only = _raw_response(usage={'malformed': True})
    adapter, _, _, _ = _adapter(
        parser=_UsageParser(ValueError('bad usage')),
        transport=_Transport(usage_only),
    )
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as usage_error:
        adapter.dispatch_once(prepared, _Permit())
    assert usage_error.value.outcome == 'provider_safety_unavailable'

    adapter, _, _, _ = _adapter(
        policy=_CostPolicy(actual=Decimal('1e18')),
        transport=_Transport(_raw_response()),
    )
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as storage_error:
        adapter.dispatch_once(prepared, _Permit())
    assert storage_error.value.outcome == 'provider_safety_unavailable'


def test_response_less_transport_failure_does_not_invoke_usage_parser() -> None:
    parser = _UsageParser(ValueError('must not be called'))
    adapter, _, _, transport = _adapter(
        parser=parser,
        transport=_Transport(TimeoutError('raw secret must not escape')),
    )
    prepared = adapter.prepare(_request(), _readiness())
    with pytest.raises(QueryEmbeddingDispatchError) as caught:
        adapter.dispatch_once(prepared, _Permit())
    assert caught.value.outcome == 'retriever_unavailable'
    assert caught.value.receipt.actual_cost_usd == Decimal('0.000777')
    assert parser.seen == []
    assert transport.calls == 1
    assert 'raw secret' not in str(caught.value)


def test_shadow_reuses_exact_same_immutable_carrier_only_for_identical_query_bytes() -> None:
    adapter, _, _, _ = _adapter()
    prepared = adapter.prepare(_request(query='한글 Exact'), _readiness())
    result = adapter.dispatch_once(prepared, _Permit())
    assert share_query_embedding_result(
        result,
        legacy_query_text='한글 Exact',
        v2_query_text='한글 Exact',
    ) is result
    with pytest.raises(ValueError, match='query bytes'):
        share_query_embedding_result(
            result,
            legacy_query_text='한글 Exact',
            v2_query_text='한글 exact',
        )


def test_not_ready_shadow_can_prepare_legacy_only_carrier_without_weakening_enforce():
    """Catches serving-index readiness being bypassed outside the shadow bridge."""
    adapter, _, _, transport = _adapter()
    not_ready = _readiness(ready=False)

    with pytest.raises(QueryEmbeddingReadinessError):
        adapter.prepare(_request(query='legacy only'), not_ready)

    prepared = adapter.prepare_shadow_legacy(
        _request(query='legacy only'), not_ready
    )

    assert prepared.transient_query_utf8 == b'legacy only'
    assert prepared.corpus_generation == not_ready.corpus_generation
    assert prepared.vector_index_generation == not_ready.vector_index_generation
    assert transport.calls == 0
