from __future__ import annotations

import logging

import pytest

from backend.app.agent_runtime.rag_provider_transport import (
    PreparedProviderDispatchEnvelope,
    RagProviderTransportError,
    dispatch_fenced_provider_call,
)
from backend.tests.test_rag_v2_costs import _admit, _budget, _ledger


class _Send:
    server_owned_fenced = True

    def __init__(self, seen):
        self.seen = seen

    def __call__(self, body, **kwargs):
        self.seen.append((body, kwargs))
        return {'ok': True}


def _envelope(grant, payload=b'{"input":"sensitive evidence"}'):
    return PreparedProviderDispatchEnvelope(
        component='query_embedding', request_bytes=payload,
        endpoint='/v1/embeddings', provider='openai',
        model='text-embedding-3-small', service_tier='default',
        model_config_snapshot_hmac='a' * 64,
        provider_safety_snapshot_hmac=grant.provider_safety_snapshot_hmac,
    )


def test_transport_accepts_only_store_owned_one_use_grant_and_forwards_no_runtime_hooks(caplog):
    ledger = _ledger()
    _admit(ledger, 30)
    grant = ledger.claim_component(
        run_id=30,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    seen = []
    envelope = _envelope(grant)
    barriers = []

    with caplog.at_level(logging.DEBUG):
        result = dispatch_fenced_provider_call(
            store=ledger,
            grant=grant,
            prepared=envelope,
            send=_Send(seen),
            evidence_fence=lambda operation: barriers.append('fence') or operation(),
            safety_before=lambda: grant.provider_safety_snapshot_hmac,
            safety_after=lambda: grant.provider_safety_snapshot_hmac,
        )
    assert result == {'ok': True}
    assert seen == [(envelope.request_bytes, {
        'max_retries': 0,
        'callbacks': [],
        'cache': False,
        'tracing': False,
    })]
    assert barriers == ['fence']
    assert 'sensitive evidence' not in caplog.text
    with pytest.raises(RagProviderTransportError):
        dispatch_fenced_provider_call(
            store=ledger, grant=grant, prepared=envelope, send=_Send([]),
            evidence_fence=lambda operation: operation(),
            safety_before=lambda: grant.provider_safety_snapshot_hmac,
            safety_after=lambda: grant.provider_safety_snapshot_hmac,
        )


def test_forged_grant_never_reaches_provider():
    ledger = _ledger()
    called = False

    class Forged:
        agent_run_id = 1
        component = 'query_embedding'
        reserved_cost_usd = 0
        dispatch_fence_hmac = 'a' * 64
        provider_safety_snapshot_hmac = 'b' * 64

        def consume_at_dispatch(self):
            pass

    def send(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(RagProviderTransportError):
        dispatch_fenced_provider_call(
            store=ledger, grant=Forged(), prepared=_envelope(Forged()), send=send,
            evidence_fence=lambda operation: operation(),
            safety_before=lambda: 'b' * 64, safety_after=lambda: 'b' * 64,
        )
    assert called is False


def test_transport_rejects_safety_drift_before_or_after_and_non_server_send():
    ledger = _ledger()
    _admit(ledger, 31)
    grant = ledger.claim_component(
        run_id=31, component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    with pytest.raises(RagProviderTransportError):
        dispatch_fenced_provider_call(
            store=ledger, grant=grant, prepared=_envelope(grant),
            send=lambda *_a, **_k: None,
            evidence_fence=lambda operation: operation(),
            safety_before=lambda: grant.provider_safety_snapshot_hmac,
            safety_after=lambda: grant.provider_safety_snapshot_hmac,
        )
