from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest

from backend.app.agent_runtime.provider_send_fence import (
    RagEvidenceFreshnessAuthority,
    RagEvidenceSendBarrier,
)
from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderDispatchAuthority,
    RagProviderRequest,
    RagProviderTransportError,
)
from backend.app.models.agent_runs import AgentRun
from backend.app.rag.retrieval import QueryEmbeddingCostInput
from backend.tests.test_rag_v2_costs import (
    _TEST_COST_POLICY,
    _budget,
    _ledger,
    _snapshot,
)


class _Client:
    def __init__(self, seen):
        self.seen = seen

    def send(self, body: bytes, *, timeout_seconds: int, max_retries: int):
        self.seen.append((body, timeout_seconds, max_retries))
        return {'ok': True}


def _request() -> RagProviderRequest:
    return RagProviderRequest(
        component='query_embedding',
        provider='openai',
        model='text-embedding-3-small',
        endpoint='/v1/embeddings',
        service_tier='default',
        model_config_snapshot_hmac=(
            _TEST_COST_POLICY.query_embedding_model_config_snapshot_hmac
        ),
        rendered_input_utf8='민감한 근거'.encode(),
    )


def _admit_transport(ledger, run_id: int):
    query_budget = _TEST_COST_POLICY.prepare_query_embedding(
        QueryEmbeddingCostInput(
            retrieval_query_utf8=_request().rendered_input_utf8,
            model_config_snapshot_hmac=(
                _TEST_COST_POLICY.query_embedding_model_config_snapshot_hmac
            ),
        )
    )
    ledger.create_admission(
        agent_run_id=run_id,
        surface='ask',
        mode='enforce',
        cutover_stage='ask',
        configured_backend='pgvector',
        query_context_version='direct-query:v1',
        current_text_hmac='1' * 64,
        retrieval_query_hmac='2' * 64,
        security_scope_fingerprint='3' * 64,
        admission_cache_identity_hmac=None,
        source_window='rag-v2:admission:enforce:ask:pgvector',
        components=(
            (_snapshot('query_embedding', _TEST_COST_POLICY), query_budget),
            (
                _snapshot('answer_generation', _TEST_COST_POLICY),
                _budget('answer_generation', '0.002000'),
            ),
        ),
    )
    return query_budget


def _authority(ledger, *, current='2' * 64):
    freshness = RagEvidenceFreshnessAuthority(
        current
        if callable(current)
        else lambda: current[0] if type(current) is list else current
    )
    barrier = RagEvidenceSendBarrier(freshness=freshness)
    return RagProviderDispatchAuthority(
        store=ledger,
        provider_safety=ledger.provider_safety_authority,
        provider_connection_factory=ledger.provider_connection_factory,
        evidence_barrier=barrier,
        identity_secret=b'task-12-test-identity-secret',
        timeout_seconds=30,
    )


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
    ledger = _ledger(tmp_path, provider_client=_Client(seen))
    query_budget = _admit_transport(ledger, 30)
    grant = ledger.claim_component(
        run_id=30,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(grant=grant, request=_request())

    with caplog.at_level(logging.DEBUG):
        result = authority.dispatch(grant=grant, prepared=prepared)

    assert result == {'ok': True}
    assert seen == [(
        '{"input":"민감한 근거","model":"text-embedding-3-small"}'.encode(),
        30,
        0,
    )]
    assert '민감한 근거' not in caplog.text
    with pytest.raises(RagProviderTransportError):
        authority.dispatch(grant=grant, prepared=prepared)


def test_forged_prepared_or_grant_and_evidence_drift_are_zero_call(tmp_path: Path):
    seen = []
    ledger = _ledger(tmp_path, provider_client=_Client(seen))
    query_budget = _admit_transport(ledger, 31)
    grant = ledger.claim_component(
        run_id=31,
        component='query_embedding',
        prepared=query_budget,
    )
    current = ['2' * 64]
    authority = _authority(ledger, current=current)
    prepared = authority.prepare(grant=grant, request=_request())
    current[0] = '3' * 64

    with pytest.raises(RagProviderTransportError, match='evidence'):
        authority.dispatch(grant=grant, prepared=prepared)
    with pytest.raises((TypeError, RagProviderTransportError)):
        authority.dispatch(grant=grant, prepared=object())
    assert seen == []


def test_request_identity_binds_rendered_input_and_dispatch_fence(tmp_path: Path):
    seen = []
    ledger = _ledger(tmp_path, provider_client=_Client(seen))
    query_budget = _admit_transport(ledger, 32)
    grant = ledger.claim_component(
        run_id=32,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    left = authority.prepare(grant=grant, request=_request())
    with pytest.raises(RagProviderTransportError, match='cost identity drifted'):
        authority.prepare(
            grant=grant,
            request=RagProviderRequest(
                component='query_embedding',
                provider='openai',
                model='text-embedding-3-small',
                endpoint='/v1/embeddings',
                service_tier='default',
                model_config_snapshot_hmac=(
                    _TEST_COST_POLICY.query_embedding_model_config_snapshot_hmac
                ),
                rendered_input_utf8='변경된 근거'.encode(),
            ),
        )
    assert left.request_identity_hmac
    parent = ledger._session.get(AgentRun, 32)
    assert parent is not None
    assert parent.status == 'failed'
    assert parent.run_record_phase == 'final'


def test_prepare_evidence_snapshot_failure_cancels_committed_claim(tmp_path: Path):
    seen = []
    ledger = _ledger(tmp_path, provider_client=_Client(seen))
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
        authority.prepare(grant=grant, request=_request())
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

    ledger = _ledger(tmp_path, provider_client=FailingClient())
    query_budget = _admit_transport(ledger, 34)
    grant = ledger.claim_component(
        run_id=34,
        component='query_embedding',
        prepared=query_budget,
    )
    authority = _authority(ledger)
    prepared = authority.prepare(grant=grant, request=_request())
    with pytest.raises(RagProviderTransportError, match='transport failed') as exc:
        authority.dispatch(grant=grant, prepared=prepared)
    assert exc.value.__cause__ is None
    assert 'sensitive' not in str(exc.value)


def test_first_forged_prepared_dispatch_cancels_committed_claim(tmp_path: Path):
    seen = []
    ledger = _ledger(tmp_path, provider_client=_Client(seen))
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
