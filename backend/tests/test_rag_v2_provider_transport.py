from __future__ import annotations

import inspect
import logging
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.model_router import RoutedRagAnswerModel
from backend.app.agent_runtime.provider_send_fence import (
    _assemble_rag_evidence_barrier,
)
from backend.app.agent_runtime.rag_cost_policy import RagCostPolicy
from backend.app.agent_runtime.rag_postgres_binding import (
    RagPostgresAdvisoryCleanupError,
)
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderDispatchAuthority,
    RagProviderTransportError,
    _assemble_rag_provider_dispatch_authority,
)
from backend.app.agents.rag_orchestrator_agent.v2_answer import StructuredRagAnswerModel
from backend.app.agents.rag_orchestrator_agent.v2_answer_schema import (
    ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
    ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT,
    build_answer_output_schema_hmac,
    build_answer_prompt_renderer_hmac,
)
from backend.app.core.config import Settings
from backend.app.models.agent_runs import AgentRun
from backend.app.models.auto_review import AutoReviewRuntimeKeyState
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.models.rag_serving import (
    RagLexicalServingProjection,
    RagServingCorpusGeneration,
)
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import (
    AnswerGenerationCostInput,
    PreparedQueryEmbedding,
    QueryEmbeddingCostInput,
    StrictProviderUsage,
    build_query_embedding_attempt_fence_hmac,
    build_query_embedding_model_config_snapshot_hmac,
    build_query_embedding_provider_policy_snapshot_hmac,
    build_query_embedding_retrieval_query_hmac_from_utf8,
    rank_evidence_slots,
)
from backend.tests.test_rag_v2_costs import (
    _TEST_COST_POLICY,
    _budget,
    _ledger,
    _snapshot,
)
from backend.tests.test_rag_v2_projection import (
    _candidate as _projection_candidate,
)
from backend.tests.test_rag_v2_projection import (
    _prepare_rows as _prepare_projection_rows,
)
from backend.tests.test_rag_v2_projection import (
    _projection as _canonical_projection,
)
from backend.tests.test_rag_v2_projection import (
    _settings as _projection_settings,
)


class _Client:
    def __init__(self, seen):
        self.seen = seen
        self.response = None

    def send(self, body: bytes, *, timeout_seconds: int, max_retries: int):
        self.seen.append((body, timeout_seconds, max_retries))
        return self.response


_CLIENTS: dict[int, object] = {}
_TEST_SETTINGS = Settings(
    _env_file=None,
    agent_runtime_fingerprint_secret='task-nine-fingerprint-secret',
    agent_runtime_fingerprint_key_version='task-nine-v1',
)


def _transport_ledger(
    tmp_path: Path,
    client: object,
    *,
    cost_policy: RagCostPolicy | None = None,
):
    ledger = _ledger(tmp_path, cost_policy=cost_policy)
    _CLIENTS[id(ledger)] = client
    return ledger


def _prepared_query(query_budget) -> PreparedQueryEmbedding:
    query_utf8 = '민감한 근거'.encode()
    query_hmac = build_query_embedding_retrieval_query_hmac_from_utf8(
        query_utf8,
        settings=_TEST_SETTINGS,
    )
    model_hmac = build_query_embedding_model_config_snapshot_hmac(_TEST_SETTINGS)
    provider_hmac = build_query_embedding_provider_policy_snapshot_hmac(
        _TEST_SETTINGS
    )
    return PreparedQueryEmbedding(
        retrieval_query_hmac=query_hmac,
        transient_query_utf8=query_utf8,
        corpus_generation=1,
        vector_index_generation=1,
        readiness_snapshot_hmac='4' * 64,
        model_config_snapshot_hmac=model_hmac,
        provider_policy_snapshot_hmac=provider_hmac,
        estimated_input_tokens=query_budget.estimated_input_tokens,
        reserved_cost_usd=query_budget.reserved_cost_usd,
        attempt_fence_hmac=build_query_embedding_attempt_fence_hmac(
            retrieval_query_hmac=query_hmac,
            corpus_generation=1,
            vector_index_generation=1,
            readiness_snapshot_hmac='4' * 64,
            model_config_snapshot_hmac=model_hmac,
            provider_policy_snapshot_hmac=provider_hmac,
            budget=query_budget,
            settings=_TEST_SETTINGS,
        ),
        budget=query_budget,
    )


def _admit_transport(ledger, run_id: int, *, surface='ask', scope_hmac='3' * 64):
    from backend.app.agents.rag_orchestrator_agent.v2_input import (
        prepare_direct_request_text,
    )

    text = prepare_direct_request_text(
        '민감한 근거', key=_TEST_SETTINGS.agent_runtime_fingerprint_secret.encode(),
    )
    query_budget = _TEST_COST_POLICY.prepare_query_embedding(
        QueryEmbeddingCostInput(
            retrieval_query_utf8='민감한 근거'.encode(),
            model_config_snapshot_hmac=(
                _TEST_COST_POLICY.query_embedding_model_config_snapshot_hmac
            ),
        )
    )
    ledger.create_admission(
        agent_run_id=run_id,
        surface=surface,
        mode='enforce',
        cutover_stage=surface,
        configured_backend='pgvector',
        query_context_version='direct-query:v1',
        current_text_hmac=text.current_text_hmac,
        retrieval_query_hmac=text.retrieval_query_hmac,
        security_scope_fingerprint=scope_hmac,
        admission_cache_identity_hmac=None,
        source_window=f'rag-v2:admission:enforce:{surface}:pgvector',
        components=(
            (_snapshot('query_embedding', _TEST_COST_POLICY), query_budget),
            (
                _snapshot('answer_generation', _TEST_COST_POLICY),
                _TEST_COST_POLICY.reserve_unused_component('answer_generation')
                if surface == 'search' else _budget('answer_generation', '0.002000'),
            ),
        ),
    )
    if ledger._session.get(RagServingCorpusGeneration, 1) is None:
        ledger._session.add_all([
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version='task-nine-v1',
                fingerprint_key_material_verifier='b' * 64,
                generation=1,
                ready=True,
            ),
            RagServingCorpusGeneration(
                id=1,
                corpus_generation=1,
                vector_index_generation=1,
                embedding_model='text-embedding-3-small',
                embedding_dimensions=1536,
                index_policy_version='rag-v2-serving-index:v1',
                pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
                fingerprint_key_version='task-nine-v1',
                fingerprint_key_material_verifier='b' * 64,
            ),
        ])
        ledger._session.commit()
    return query_budget


def _authority(
    ledger,
    *,
    current='2' * 64,
    settings: Settings = _TEST_SETTINGS,
    answer_model: StructuredRagAnswerModel | None = None,
    readiness: object | None = None,
):
    barrier = _assemble_rag_evidence_barrier(
        load_current_identity=(
            current
            if callable(current)
            else lambda: current[0] if type(current) is list else current
        )
    )
    return _assemble_rag_provider_dispatch_authority(
        store=ledger,
        provider_safety=ledger.provider_safety_authority,
        provider_connection_factory=ledger.provider_connection_factory,
        evidence_barrier=barrier,
        identity_secret=b'task-12-test-identity-secret',
        timeout_seconds=30,
        provider_client=_CLIENTS.get(id(ledger)),
        settings=settings,
        answer_model=answer_model,
        runtime_health=ledger.runtime_health_authority,
        load_current_readiness=lambda: (
            readiness()
            if callable(readiness)
            else readiness
        ) or RagServingIndexReadiness(
            ready=True,
            corpus_generation=1,
            vector_index_generation=1,
            expected_document_count=0,
            live_vector_count=0,
            tombstone_count=0,
            mismatch_count_capped_at_20=0,
            embedding_model='text-embedding-3-small',
            embedding_dimensions=1536,
            index_policy_version='rag-v2-serving-index:v1',
            readiness_snapshot_hmac='4' * 64,
        ),
    )


def _dispatch_embedding_response(tmp_path: Path, run_id: int, response: object):
    seen = []
    client = _Client(seen)
    client.response = response
    ledger = _transport_ledger(tmp_path, client)
    query_budget = _admit_transport(ledger, run_id)
    grant = ledger.claim_component(
        run_id=run_id,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    dispatch = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )
    observation = authority.dispatch(grant=grant, prepared=dispatch)
    return ledger, authority, grant, observation, query_budget


def test_transport_public_api_has_no_constructible_envelope_or_call_callbacks():
    import backend.app.agent_runtime.rag_provider_transport as module

    assert not hasattr(module, 'PreparedProviderDispatchEnvelope')
    parameters = inspect.signature(RagProviderDispatchAuthority.dispatch).parameters
    assert not {'send', 'evidence_fence', 'safety_before', 'safety_after'} & set(
        parameters
    )
    init_parameters = inspect.signature(
        RagProviderDispatchAuthority.__init__
    ).parameters
    assert 'provider_client' not in init_parameters


def test_server_builds_canonical_request_and_consumes_store_owned_grant_once(
    tmp_path: Path, caplog
):
    seen = []
    client = _Client(seen)
    ledger = _transport_ledger(tmp_path, client)
    query_budget = _admit_transport(ledger, 30)
    client.response = {
        'object': 'list',
        'model': 'text-embedding-3-small',
        'data': [
            {
                'object': 'embedding',
                'index': 0,
                'embedding': [1.0, *([0.0] * 1535)],
            }
        ],
        'usage': {
            'prompt_tokens': query_budget.estimated_input_tokens,
            'total_tokens': query_budget.estimated_input_tokens,
        },
    }
    grant = ledger.claim_component(
        run_id=30,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )

    with caplog.at_level(logging.DEBUG):
        observation = authority.dispatch(grant=grant, prepared=prepared)
        result = authority.finalize(grant=grant, observation=observation)

    assert result.terminal_outcome == 'component_succeeded'
    assert result.charge_basis == 'actual'
    assert seen == [(
        ('{"dimensions":1536,"encoding_format":"float","input":'
         '"민감한 근거","model":"text-embedding-3-small"}').encode(),
        30,
        0,
    )]
    assert '민감한 근거' not in caplog.text
    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=prepared)


def test_forged_prepared_or_grant_and_evidence_drift_are_zero_call(tmp_path: Path):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 31)
    grant = ledger.claim_component(
        run_id=31,
        component='query_embedding',
        prepared=query_budget,
    )
    current = ['2' * 64]
    authority = _authority(ledger, current=current)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )
    current[0] = '3' * 64

    with pytest.raises(RagProviderTransportError, match='evidence'):
        authority.dispatch(grant=grant, prepared=prepared)
    with pytest.raises((TypeError, RagProviderTransportError)):
        authority.dispatch(grant=grant, prepared=object())
    assert seen == []


def test_runtime_poison_after_prepare_refuses_send_and_closes_claim_at_zero(
    tmp_path: Path,
) -> None:
    seen: list[object] = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 311)
    grant = ledger.claim_component(
        run_id=311,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )
    ledger.runtime_health_authority._poison()

    with pytest.raises(RagProviderTransportError, match='runtime health'):
        authority.dispatch(grant=grant, prepared=prepared)

    ledger._session.expire_all()
    parent = ledger._session.get(AgentRun, 311)
    children = tuple(
        ledger._session.query(AgentRunCostComponent)
        .filter(AgentRunCostComponent.agent_run_id == 311)
        .order_by(AgentRunCostComponent.component_ordinal)
    )
    assert seen == []
    assert parent is not None
    assert parent.status == 'failed'
    assert parent.metadata_['outcome'] == 'provider_safety_unavailable'
    assert parent.total_charged_cost_usd == 0
    assert len(children) == 2
    assert all(child.attempted is False for child in children)
    assert all(child.dispatch_count == 0 for child in children)
    assert all(child.charged_cost_usd == 0 for child in children)


def test_provider_advisory_cleanup_before_send_is_exact_terminal_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[object] = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 312)
    grant = ledger.claim_component(
        run_id=312,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )

    @contextmanager
    def refuse_before_owner(*_args, **_kwargs):
        raise RagPostgresAdvisoryCleanupError(
            'RAG PostgreSQL advisory cleanup failed'
        )
        yield

    monkeypatch.setattr(
        type(ledger),
        'projection_owner_barrier',
        refuse_before_owner,
    )

    with pytest.raises(RagProviderTransportError, match='transport failed'):
        authority.dispatch(grant=grant, prepared=prepared)

    ledger._session.expire_all()
    parent = ledger._session.get(AgentRun, 312)
    children = tuple(
        ledger._session.query(AgentRunCostComponent)
        .filter(AgentRunCostComponent.agent_run_id == 312)
        .order_by(AgentRunCostComponent.component_ordinal)
    )
    assert seen == []
    assert parent is not None
    assert parent.status == 'failed'
    assert parent.metadata_['outcome'] == 'provider_safety_unavailable'
    assert len(children) == 2
    assert all(child.dispatch_state == 'terminal' for child in children)
    assert all(child.attempted is False for child in children)
    assert all(child.dispatch_count == 0 for child in children)
    assert all(child.charged_cost_usd == 0 for child in children)


def test_provider_advisory_cleanup_after_send_preserves_attempted_unknown_cost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[object] = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 313)
    grant = ledger.claim_component(
        run_id=313,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )

    @contextmanager
    def fail_cleanup_after_send(*_args, **_kwargs):
        yield
        raise RagPostgresAdvisoryCleanupError(
            'RAG PostgreSQL advisory cleanup failed'
        )

    monkeypatch.setattr(
        type(ledger.provider_safety_authority),
        'dispatch_barrier',
        fail_cleanup_after_send,
    )

    with pytest.raises(RagProviderTransportError, match='transport failed'):
        authority.dispatch(grant=grant, prepared=prepared)

    ledger._session.expire_all()
    parent = ledger._session.get(AgentRun, 313)
    children = tuple(
        ledger._session.query(AgentRunCostComponent)
        .filter(AgentRunCostComponent.agent_run_id == 313)
        .order_by(AgentRunCostComponent.component_ordinal)
    )
    query_child = children[0]
    assert len(seen) == 1
    assert grant.consumed is True
    assert parent is not None
    assert parent.status == 'running'
    assert query_child.component == 'query_embedding'
    assert query_child.dispatch_state == 'dispatching'
    assert query_child.attempted is True
    assert query_child.dispatch_count == 1
    assert query_child.charge_basis == 'reserved'
    assert query_child.charged_cost_usd == query_child.reserved_cost_usd
    assert query_child.terminal_outcome is None
    assert children[1].dispatch_state == 'not_attempted'


def test_request_identity_binds_rendered_input_and_dispatch_fence(tmp_path: Path):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 32)
    grant = ledger.claim_component(
        run_id=32,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    frozen = _prepared_query(query_budget)
    left = authority.prepare(grant=grant, prepared=frozen)
    with pytest.raises(RagProviderTransportError, match='invocation is invalid'):
        authority.prepare(
            grant=grant,
            prepared=replace(
                frozen,
                transient_query_utf8='변경된 근거'.encode(),
            ),
        )
    assert left.request_identity_hmac
    parent = ledger._session.get(AgentRun, 32)
    assert parent is not None
    assert parent.status == 'failed'
    assert parent.run_record_phase == 'final'


def test_same_budget_altered_query_carrier_is_rejected_before_canonicalization(
    tmp_path: Path,
):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 320)
    grant = ledger.claim_component(
        run_id=320,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    authentic = _prepared_query(query_budget)
    altered_budget = _TEST_COST_POLICY.prepare_query_embedding(
        QueryEmbeddingCostInput(
            retrieval_query_utf8='민감한 증거'.encode(),
            model_config_snapshot_hmac=authentic.model_config_snapshot_hmac,
        )
    )
    assert altered_budget.estimated_input_tokens == query_budget.estimated_input_tokens
    assert altered_budget.reserved_cost_usd == query_budget.reserved_cost_usd
    altered = replace(
        authentic,
        transient_query_utf8='민감한 증거'.encode(),
        estimated_input_tokens=altered_budget.estimated_input_tokens,
        reserved_cost_usd=altered_budget.reserved_cost_usd,
        budget=altered_budget,
    )

    with pytest.raises(RagProviderTransportError):
        authority.prepare(grant=grant, prepared=altered)

    assert seen == []


def test_prepare_evidence_snapshot_failure_cancels_committed_claim(tmp_path: Path):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 33)
    grant = ledger.claim_component(
        run_id=33,
        component='query_embedding',
        prepared=query_budget,
    )

    def unavailable():
        raise RuntimeError('sensitive evidence backend detail')

    authority = _authority(ledger, current=unavailable)
    with pytest.raises(RagProviderTransportError, match='evidence identity') as exc:
        authority.prepare(
            grant=grant,
            prepared=_prepared_query(query_budget),
        )
    parent = ledger._session.get(AgentRun, 33)
    assert parent is not None
    assert parent.status == 'failed'
    assert parent.run_record_phase == 'final'
    assert 'sensitive' not in str(exc.value)
    assert seen == []


def test_dispatch_requires_store_configured_client_and_sanitizes_client_failure(
    tmp_path: Path,
):
    ledger_without_client = _ledger(tmp_path)
    with pytest.raises(TypeError, match='authority'):
        _authority(ledger_without_client)

    class FailingClient:
        def send(self, *_args, **_kwargs):
            raise RuntimeError('raw sensitive provider response')

    ledger = _transport_ledger(tmp_path, FailingClient())
    query_budget = _admit_transport(ledger, 34)
    grant = ledger.claim_component(
        run_id=34,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )
    observation = authority.dispatch(grant=grant, prepared=prepared)
    result = authority.finalize(grant=grant, observation=observation)
    assert result.terminal_outcome == 'retriever_unavailable'


def test_first_forged_prepared_dispatch_cancels_committed_claim(tmp_path: Path):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 35)
    grant = ledger.claim_component(
        run_id=35,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)

    with pytest.raises(RagProviderTransportError, match='prepared'):
        authority.dispatch(grant=grant, prepared=object())

    parent = ledger._session.get(AgentRun, 35)
    assert parent is not None
    assert parent.status == 'failed'
    assert parent.run_record_phase == 'final'
    assert seen == []


def test_transport_owns_embedding_classification_precedence_and_one_use(
    tmp_path: Path,
):
    overrun_tokens = 20
    ledger, authority, grant, observation, _ = _dispatch_embedding_response(
        tmp_path,
        36,
        {
            'object': 'wrong',
            'model': 'wrong',
            'data': [],
            'usage': {
                'prompt_tokens': overrun_tokens,
                'total_tokens': overrun_tokens,
            },
        },
    )
    overrun = authority.finalize(grant=grant, observation=observation)
    assert overrun.terminal_outcome == 'provider_usage_overrun'
    assert overrun.charged_cost_usd == _TEST_COST_POLICY.charge_actual(
        'query_embedding',
        StrictProviderUsage(overrun_tokens, 0, overrun_tokens),
    )
    with pytest.raises(TypeError, match='unavailable'):
        authority.finalize(grant=grant, observation=observation)
    assert ledger.total_charged_cost(36) == overrun.charged_cost_usd


def test_all_zero_embedding_is_not_cosine_indexable(tmp_path: Path):
    budget = _TEST_COST_POLICY.prepare_query_embedding(
        QueryEmbeddingCostInput(
            retrieval_query_utf8='민감한 근거'.encode(),
            model_config_snapshot_hmac=(
                _TEST_COST_POLICY.query_embedding_model_config_snapshot_hmac
            ),
        )
    )
    response = {
        'object': 'list',
        'model': 'text-embedding-3-small',
        'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.0] * 1536}],
        'usage': {
            'prompt_tokens': budget.estimated_input_tokens,
            'total_tokens': budget.estimated_input_tokens,
        },
    }
    _, authority, grant, observation, _ = _dispatch_embedding_response(
        tmp_path, 360, response
    )

    result = authority.finalize(grant=grant, observation=observation)

    assert result.terminal_outcome == 'provider_embedding_payload_invalid'


def test_embedding_item_identity_precedes_vector_and_usage_contract(
    tmp_path: Path,
):
    response = {
        'object': 'list',
        'model': 'text-embedding-3-small',
        'data': [
            {
                'object': 'wrong',
                'index': 0,
                'embedding': [float('nan')] * 1536,
            }
        ],
        'usage': {'prompt_tokens': 'coerced', 'total_tokens': 'coerced'},
    }
    _, authority, grant, observation, _ = _dispatch_embedding_response(
        tmp_path, 37, response
    )
    result = authority.finalize(grant=grant, observation=observation)
    assert result.terminal_outcome == 'provider_response_identity_invalid'


def test_direct_provider_client_uses_exact_openai_and_langchain_bindings(
    monkeypatch,
):
    import backend.app.agent_runtime.model_router as model_router
    import backend.app.agent_runtime.rag_provider_transport as transport

    openai_init = []
    embedding_calls = []
    answer_calls = []

    class FakeEmbeddingResponse:
        def model_dump(self, *, mode):
            assert mode == 'json'
            return {'object': 'list'}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            openai_init.append(kwargs)
            self.embeddings = SimpleNamespace(
                create=lambda **kwargs: (
                    embedding_calls.append(kwargs) or FakeEmbeddingResponse()
                )
            )

    class FakeAnswerModel:
        def invoke(self, value, *, config):
            answer_calls.append((value, config))
            return {'raw': object(), 'parsed': object(), 'parsing_error': None}

    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(
        model_router,
        'build_rag_answer_model_route',
        lambda *, settings: SimpleNamespace(model=FakeAnswerModel()),
    )
    settings = Settings(_env_file=None, openai_api_key='test-key-not-live')

    client = transport._DirectOpenAIProviderClient(settings)
    embedding = client.send(
        ('{"dimensions":1536,"encoding_format":"float","input":"질문",'
         '"model":"text-embedding-3-small"}').encode(),
        timeout_seconds=30,
        max_retries=0,
    )
    answer = client.send(
        canonical_json_bytes({
            'input': [{'content': '질문', 'role': 'user'}],
            'max_output_tokens': 512,
            'model': 'gpt-5.4-mini-2026-03-17',
            'reasoning': {'effort': 'none'},
            'service_tier': 'default',
            'store': False,
            'stream': False,
            'text': {'format': ANSWER_OUTPUT_SCHEMA_PROVIDER_FORMAT},
            'tools': [],
        }),
        timeout_seconds=30,
        max_retries=0,
    )

    assert embedding == {'object': 'list'}
    assert answer['parsing_error'] is None
    assert embedding_calls == [{
        'dimensions': 1536,
        'encoding_format': 'float',
        'input': '질문',
        'model': 'text-embedding-3-small',
    }]
    assert answer_calls == [
        ([{'content': '질문', 'role': 'user'}], {'callbacks': []})
    ]
    assert openai_init[0]['base_url'] == 'https://api.openai.com/v1'
    assert openai_init[0]['max_retries'] == 0
    assert openai_init[0]['http_client']._trust_env is False


def test_dispatch_freshly_rechecks_durable_cost_row_before_provider_send(
    tmp_path: Path,
):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    query_budget = _admit_transport(ledger, 38)
    grant = ledger.claim_component(
        run_id=38,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(
        grant=grant,
        prepared=_prepared_query(query_budget),
    )
    with Session(ledger._session.get_bind()) as competing:
        competing.execute(
            update(AgentRunCostComponent)
            .where(
                AgentRunCostComponent.agent_run_id == 38,
                AgentRunCostComponent.component == 'query_embedding',
            )
            .values(dispatch_fence_hmac='f' * 64)
        )
        competing.commit()

    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=prepared)
    assert seen == []


def _answer_transport_case(tmp_path: Path, run_id: int):
    seen = []
    client = _Client(seen)
    policy_settings = _projection_settings()
    policy = RagCostPolicy(
        settings=policy_settings,
        answer_output_schema_hmac=build_answer_output_schema_hmac(
            policy_settings
        ),
        answer_prompt_renderer_hmac=build_answer_prompt_renderer_hmac(
            policy_settings
        ),
    )
    ledger = _transport_ledger(
        tmp_path, client, cost_policy=policy
    )
    projection = _canonical_projection(1)
    slot = rank_evidence_slots((_projection_candidate(projection, 0.9),))[0]
    _, prepared_influence = _prepare_projection_rows(
        (projection,), rendered_input_hmac='d' * 64
    )
    observation = prepared_influence.observations[0]
    answer_model = StructuredRagAnswerModel(
        routed_model=RoutedRagAnswerModel(
            model=object(),
            provider='openai',
            model_name='gpt-5.4-mini-2026-03-17',
            model_config_snapshot_hmac=policy.answer_model_config_snapshot_hmac,
        ),
        cost_policy=policy,
    )
    prepared_answer = answer_model.prepare(
        question='질문',
        slots=(slot,),
        model_influence=(observation,),
        answer_question_hmac='d' * 64,
        retrieval_query_hmac='2' * 64,
    )
    budget = prepared_answer.budget
    ledger.create_admission(
        agent_run_id=run_id,
        surface='ask',
        mode='enforce',
        cutover_stage='ask',
        configured_backend='keyword',
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:enforce:ask:keyword',
        components=(
            (_snapshot('query_embedding', policy), policy.prepare_query_embedding(QueryEmbeddingCostInput(
                retrieval_query_utf8=b'unused',
                model_config_snapshot_hmac=policy.query_embedding_model_config_snapshot_hmac,
            ))),
            (_snapshot('answer_generation', policy), budget),
        ),
    )
    key_version = policy_settings.agent_runtime_fingerprint_key_version
    key_verifier = 'b' * 64
    ledger._session.add_all([
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=key_version,
            fingerprint_key_material_verifier=key_verifier,
            generation=1,
            ready=True,
        ),
        RagServingCorpusGeneration(
            id=1,
            corpus_generation=1,
            vector_index_generation=1,
            embedding_model='text-embedding-3-small',
            embedding_dimensions=1536,
            index_policy_version='rag-v2-serving-index:v1',
            pgvector_cosine_policy_version='pgvector-cosine-indexable:v1',
            fingerprint_key_version=key_version,
            fingerprint_key_material_verifier=key_verifier,
        ),
        RagLexicalServingProjection(
            corpus_generation_id=1,
            corpus_generation=1,
            serving_document_id=slot.evidence.serving_document_id,
            serving_kind=slot.evidence.serving_kind,
            support_mode=slot.support_mode,
            effective_permission=slot.evidence.effective_permission,
            serving_identity_hmac=slot.evidence.serving_identity_hmac,
            serving_version_fingerprint=slot.evidence.serving_version_fingerprint,
            model_content_hmac=slot.evidence.model_content_hmac,
            canonical_citation_projection_hmac=(
                slot.evidence.canonical_citation_projection_hmac
            ),
            title_lower='title',
            searchable_lower='content',
            lexical_contract_version='rag-keyword-lexical-compat:v1',
            fingerprint_key_version=key_version,
            fingerprint_key_material_verifier=key_verifier,
        ),
    ])
    ledger._session.commit()
    grant = ledger.claim_component(
        run_id=run_id,
        component='answer_generation',
        prepared=budget,
    )
    authority = _authority(
        ledger,
        settings=policy_settings,
        answer_model=answer_model,
    )
    dispatch = authority.prepare(grant=grant, prepared=prepared_answer)
    return ledger, authority, grant, dispatch, client, prepared_answer


def test_answer_dispatch_freshly_rechecks_c5_key_corpus_and_projection_rows(
    tmp_path: Path,
):
    ledger, authority, grant, dispatch, _client, _prepared = (
        _answer_transport_case(tmp_path, 39)
    )
    with Session(ledger._session.get_bind()) as competing:
        competing.execute(
            update(RagLexicalServingProjection)
            .values(model_content_hmac='c' * 64)
        )
        competing.commit()

    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=dispatch)
    assert _client.seen == []


def test_same_budget_altered_answer_messages_are_rejected_before_send(tmp_path: Path):
    ledger, authority, grant, _dispatch, client, prepared = (
        _answer_transport_case(tmp_path, 390)
    )
    altered_messages = (
        prepared.messages[0],
        (prepared.messages[1][0], prepared.messages[1][1].replace('질문', '문질')),
    )
    messages_json = canonical_json_bytes([
        {'content': content, 'ordinal': ordinal, 'role': role}
        for ordinal, (role, content) in enumerate(altered_messages, start=1)
    ])
    altered_budget = ledger.cost_policy_authority.prepare_answer_generation(
        AnswerGenerationCostInput(
            exact_messages_json=messages_json,
            exact_response_schema_json=ANSWER_OUTPUT_SCHEMA_PROVIDER_BYTES,
            model_config_snapshot_hmac=prepared.model_config_snapshot_hmac,
        )
    )
    assert altered_budget.estimated_input_tokens == prepared.budget.estimated_input_tokens
    assert altered_budget.reserved_cost_usd == prepared.budget.reserved_cost_usd
    altered = replace(
        prepared,
        messages=altered_messages,
        generation_estimator_input_hmac=altered_budget.estimator_input_hmac,
        encoded_input_tokens=altered_budget.estimated_input_tokens - 528,
        framed_input_tokens=altered_budget.estimated_input_tokens,
        reserved_cost_usd=altered_budget.reserved_cost_usd,
        budget=altered_budget,
    )

    with pytest.raises(RagProviderTransportError):
        authority.prepare(grant=grant, prepared=altered)

    assert client.seen == []


def test_query_dispatch_rechecks_corpus_and_vector_generation_before_send(
    tmp_path: Path,
):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    budget = _admit_transport(ledger, 391)
    grant = ledger.claim_component(
        run_id=391, component='query_embedding', prepared=budget
    )
    authority = _authority(ledger)
    dispatch = authority.prepare(grant=grant, prepared=_prepared_query(budget))
    with Session(ledger._session.get_bind()) as competing:
        competing.execute(
            update(RagServingCorpusGeneration)
            .where(RagServingCorpusGeneration.id == 1)
            .values(vector_index_generation=2)
        )
        competing.commit()

    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=dispatch)

    assert seen == []


def test_query_dispatch_rechecks_exact_readiness_snapshot_before_send(
    tmp_path: Path,
):
    seen = []
    ledger = _transport_ledger(tmp_path, _Client(seen))
    budget = _admit_transport(ledger, 392)
    current = ['4' * 64]

    def readiness():
        return RagServingIndexReadiness(
            ready=True,
            corpus_generation=1,
            vector_index_generation=1,
            expected_document_count=0,
            live_vector_count=0,
            tombstone_count=0,
            mismatch_count_capped_at_20=0,
            embedding_model='text-embedding-3-small',
            embedding_dimensions=1536,
            index_policy_version='rag-v2-serving-index:v1',
            readiness_snapshot_hmac=current[0],
        )

    grant = ledger.claim_component(
        run_id=392, component='query_embedding', prepared=budget
    )
    authority = _authority(ledger, readiness=readiness)
    dispatch = authority.prepare(grant=grant, prepared=_prepared_query(budget))
    current[0] = '6' * 64

    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=dispatch)

    assert seen == []


@pytest.mark.parametrize(
    ('run_id', 'metadata', 'parsed', 'expected'),
    (
        (
            40,
            {'model': 'wrong', 'object': 'response', 'service_tier': 'default'},
            [],
            'provider_response_identity_invalid',
        ),
        (
            41,
            {
                'model': 'gpt-5.4-mini-2026-03-17',
                'object': 'response',
                'service_tier': 'default',
            },
            {
                'answer_blocks': [{
                    'text': '답변',
                    'evidence_slot_ids': ['E9'],
                    'support_mode': 'source_observation',
                }],
                'insufficient_evidence_reason': None,
            },
            'citation_validation_failed',
        ),
        (
            42,
            {
                'model': 'gpt-5.4-mini-2026-03-17',
                'object': 'response',
                'service_tier': 'default',
            },
            [],
            'structured_output_invalid',
        ),
    ),
)
def test_transport_owns_answer_identity_schema_and_citation_precedence(
    tmp_path: Path,
    run_id: int,
    metadata: dict[str, object],
    parsed: object,
    expected: str,
):
    ledger, authority, grant, dispatch, client, _prepared = (
        _answer_transport_case(tmp_path, run_id)
    )
    client.response = {
        'raw': AIMessage(
            content='',
            usage_metadata={
                'input_tokens': 10,
                'output_tokens': 5,
                'total_tokens': 15,
            },
            response_metadata=metadata,
        ),
        'parsed': parsed,
        'parsing_error': None,
    }

    observation = authority.dispatch(grant=grant, prepared=dispatch)
    final = authority.finalize(grant=grant, observation=observation)

    assert final.terminal_outcome == expected
    assert final.charge_basis == 'actual'
    assert len(client.seen) == 1
    assert ledger.total_charged_cost(run_id) == final.charged_cost_usd


def test_answer_usage_invalid_precedes_schema_and_citation_validation(
    tmp_path: Path,
):
    ledger, authority, grant, dispatch, client, _prepared = (
        _answer_transport_case(tmp_path, 420)
    )
    raw = AIMessage(
        content='',
        usage_metadata={
            'input_tokens': 10,
            'output_tokens': 5,
            'total_tokens': 15,
        },
        response_metadata={
            'model': 'gpt-5.4-mini-2026-03-17',
            'object': 'response',
            'service_tier': 'default',
        },
    )
    raw.usage_metadata = {
        'input_tokens': '10',
        'output_tokens': 5,
        'total_tokens': 15,
    }
    client.response = {
        'raw': raw,
        'parsed': [],
        'parsing_error': None,
    }

    observation = authority.dispatch(grant=grant, prepared=dispatch)
    final = authority.finalize(grant=grant, observation=observation)

    assert final.terminal_outcome == 'provider_safety_unavailable'
    assert final.charge_basis == 'reserved'


def test_answer_overrun_precedes_identity_usage_and_schema_failures(tmp_path: Path):
    ledger, authority, grant, dispatch, client, prepared = (
        _answer_transport_case(tmp_path, 421)
    )
    overrun_input = prepared.budget.estimated_input_tokens + 1
    client.response = {
        'raw': AIMessage(
            content='',
            usage_metadata={
                'input_tokens': overrun_input,
                'output_tokens': 1,
                'total_tokens': overrun_input + 1,
            },
            response_metadata={
                'model': 'wrong',
                'object': 'wrong',
                'service_tier': 'wrong',
            },
        ),
        'parsed': [],
        'parsing_error': RuntimeError('not exposed'),
    }

    observation = authority.dispatch(grant=grant, prepared=dispatch)
    final = authority.finalize(grant=grant, observation=observation)

    assert final.terminal_outcome == 'provider_usage_overrun'
    assert final.charge_basis == 'actual'
    assert ledger.total_charged_cost(421) == final.charged_cost_usd
