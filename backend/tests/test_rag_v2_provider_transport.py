from __future__ import annotations

import logging

import pytest

from backend.app.agent_runtime.rag_provider_transport import (
    RagProviderTransportError,
    dispatch_fenced_provider_call,
)
from backend.tests.test_rag_v2_costs import _admit, _budget, _ledger


def test_transport_accepts_only_store_owned_one_use_grant_and_forwards_no_runtime_hooks(caplog):
    ledger = _ledger()
    _admit(ledger, 30)
    grant = ledger.claim_component(
        run_id=30,
        component='query_embedding',
        prepared=_budget('query_embedding', '0.000010'),
    )
    seen = []
    payload = {'input': 'sensitive evidence'}

    with caplog.at_level(logging.DEBUG):
        result = dispatch_fenced_provider_call(
            store=ledger,
            grant=grant,
            payload=payload,
            send=lambda body, **kwargs: seen.append((body, kwargs)) or {'ok': True},
        )
    assert result == {'ok': True}
    assert seen == [(payload, {
        'max_retries': 0,
        'callbacks': [],
        'cache': False,
        'tracing': False,
    })]
    assert 'sensitive evidence' not in caplog.text
    with pytest.raises(RagProviderTransportError):
        dispatch_fenced_provider_call(
            store=ledger, grant=grant, payload=payload, send=lambda *_a, **_k: None
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
            store=ledger, grant=Forged(), payload={}, send=send
        )
    assert called is False
