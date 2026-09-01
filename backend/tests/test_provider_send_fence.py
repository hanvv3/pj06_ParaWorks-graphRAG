import copy
import json
import pickle
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from backend.app.agent_runtime import provider_send_fence as fence_module
from backend.app.agent_runtime.contracts import (
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
)
from backend.app.agent_runtime.provider_send_fence import (
    FencedOpenAITransport,
    FencedProviderSendPermit,
    ProviderAttemptGrant,
    ProviderSendFenceError,
    _run_shared_advisory_send_fence,
)
from backend.app.agent_runtime.review_v21_extraction import (
    ExtractionCallLedger,
    ExtractionCallStateError,
    ExtractionProviderSafetySnapshot,
    ProviderUsage,
    build_prepared_extraction_plan_set,
    invoke_prepared_extraction,
)
from backend.app.core.config import Settings


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _attempt(clock: _Clock, *, commit=None, db_clock=None):
    settings = Settings(
        agent_runtime_fingerprint_secret='send-fence-test-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='send-fence-v1',
    )
    packet = EvidencePacket(
        'company_memory',
        'window',
        [
            EvidenceMessage(
                source_id='gmail:message',
                source_url='https://secret.test/message',
                text='verified evidence',
                author='person@example.test',
                timestamp='2026-08-28T00:00:00Z',
                permission_level='internal',
                source_snippet_override='verified evidence',
                metadata={
                    'stable_message_identity': 'message-1',
                    'workflow_evidence_ref_id': 1,
                },
            )
        ],
        PermissionContext('owner', 'employee'),
    )
    safety = ExtractionProviderSafetySnapshot(
        purpose='extraction',
        provider='openai',
        model='gpt-5.4-mini-2026-03-17',
        reasoning_effort='none',
        state_version=1,
        cost_policy_version='auto-review-extraction-cost:v1',
        token_estimator_version='openai-o200k-extraction:v1',
        tokenizer_encoding='o200k_base',
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        input_usd_per_1m=Decimal('0.750000'),
        output_usd_per_1m=Decimal('4.500000'),
        breaker_open=False,
    )
    plan = build_prepared_extraction_plan_set(
        packet=packet,
        selected_agent_names=('history_agent',),
        settings=settings,
        fingerprint_key_material_verifier='a' * 64,
        safety_snapshots=(safety,),
    ).plans[0]
    ledger = ExtractionCallLedger(permit_monotonic=clock, db_clock=db_clock)
    context = ledger.claim_or_replay('workflow', plan)
    grant = ledger.mark_attempt_started(context, commit=commit)
    return ledger, context, plan, grant


def _grant(clock: _Clock, *, commit=None) -> ProviderAttemptGrant:
    return _attempt(clock, commit=commit)[3]


def test_extraction_send_permit_is_one_use_nonserializable_and_expires_before_recovery():
    clock = _Clock()
    grant = _grant(clock)
    for serializer in (
        pickle.dumps,
        copy.copy,
        copy.deepcopy,
        lambda value: json.dumps(value),
    ):
        with pytest.raises((TypeError, ValueError)):
            serializer(grant.permit)
    grant.permit.consume_at_dispatch()
    with pytest.raises(ProviderSendFenceError, match='consumed'):
        grant.permit.consume_at_dispatch()

    late = _grant(clock)
    clock.value += 5
    with pytest.raises(ProviderSendFenceError, match='expired'):
        late.permit.consume_at_dispatch()


def test_store_owned_fence_consumes_only_at_dispatch_and_never_exposes_transport(
    caplog,
):
    clock = _Clock()
    ledger, _context, plan, grant = _attempt(clock)
    calls = []

    assert not hasattr(grant, 'transport')
    assert not grant.permit.consumed
    result = invoke_prepared_extraction(
        plan.invocation,
        lambda _invocation, *, timeout: calls.append(timeout) or 'ok',
        store=ledger,
        grant=grant,
    )
    assert result == 'ok'
    assert calls == [60]
    assert grant.permit.consumed
    assert plan.invocation.canonical_text not in caplog.text


def test_second_store_dispatch_reuses_no_consumed_permit():
    ledger, _context, plan, grant = _attempt(_Clock())
    invoke_prepared_extraction(
        plan.invocation,
        lambda _invocation, *, timeout: timeout,
        store=ledger,
        grant=grant,
    )
    with pytest.raises((ProviderSendFenceError, ExtractionCallStateError)):
        invoke_prepared_extraction(
            plan.invocation,
            lambda _invocation, *, timeout: timeout,
            store=ledger,
            grant=grant,
        )


def test_restart_has_no_reconstructable_send_permit_and_marker_is_not_send_authority():
    with pytest.raises(TypeError):
        FencedProviderSendPermit('attempt-hmac', 15.0)  # type: ignore[call-arg]
    with pytest.raises((AttributeError, TypeError)):
        ProviderAttemptGrant.from_marker(  # type: ignore[attr-defined]
            attempt_id='attempt-hmac',
            provider_timeout_seconds=60,
        )
    assert not hasattr(FencedOpenAITransport, 'dispatch')


def test_attempt_grant_is_not_returned_before_marker_commit():
    events = []

    def commit():
        events.append('commit')

    grant = _grant(_Clock(), commit=commit)
    assert events == ['commit']
    assert grant.permit.attempt_id == grant.attempt_id


def test_permit_and_grant_public_construction_copy_pickle_json_and_repr_are_closed():
    with pytest.raises(TypeError):
        ProviderAttemptGrant(  # type: ignore[call-arg]
            attempt_id='forged',
            permit=object(),
            provider_timeout_seconds=60,
            authoritative_lease_expires_at=datetime.now(UTC),
        )
    with pytest.raises((AttributeError, TypeError)):
        FencedProviderSendPermit.issue(  # type: ignore[attr-defined]
            attempt_id='forged', send_start_window_seconds=5
        )
    grant = _grant(_Clock())
    for value in (grant, grant.permit):
        assert grant.attempt_id not in repr(value)
        for serializer in (copy.copy, copy.deepcopy, pickle.dumps, json.dumps):
            with pytest.raises((TypeError, ValueError)):
                serializer(value)


def test_no_importable_callable_can_issue_a_provider_attempt_grant():
    issuer = getattr(fence_module, '_issue_provider_attempt_grant', None)
    assert issuer is None


def test_store_dispatch_authenticates_task1_server_owned_hook(monkeypatch):
    from backend.app.agent_runtime import auto_review_cost_policy

    monkeypatch.setattr(
        auto_review_cost_policy,
        'is_server_owned_fenced_send_hook',
        lambda _hook: False,
    )
    ledger, _context, plan, grant = _attempt(_Clock())
    with pytest.raises(ProviderSendFenceError, match='server-owned'):
        invoke_prepared_extraction(
            plan.invocation,
            lambda *_args, **_kwargs: pytest.fail('provider called'),
            store=ledger,
            grant=grant,
        )


@pytest.mark.parametrize(
    'terminal_cause',
    ('complete', 'fail', 'recovery', 'drift', 'lease_expiry', 'corruption'),
)
def test_every_terminal_or_authority_loss_removes_indirect_and_permit_dispatch(
    terminal_cause: str,
):
    now = [datetime(2026, 8, 28, tzinfo=UTC)]
    clock = _Clock()
    ledger, context, plan, grant = _attempt(clock, db_clock=lambda: now[0])
    if terminal_cause == 'complete':
        ledger.complete(
            context,
            result={
                'result_kind': 'no_candidate',
                'candidate': None,
                'no_candidate_reason': 'no_relevant_evidence',
            },
            usage=ProviderUsage(1, 1),
        )
    elif terminal_cause == 'fail':
        ledger.fail(context, reason_code='provider_failure')
    elif terminal_cause == 'recovery':
        now[0] += timedelta(seconds=121)
        ledger.recover_expired(context)
    elif terminal_cause == 'drift':
        ledger.set_drift(context)
    elif terminal_cause == 'lease_expiry':
        now[0] += timedelta(seconds=121)
    else:
        ledger.force_corrupt_completed(context)

    called = False

    def provider(_invocation, *, timeout):
        nonlocal called
        called = True
        return timeout

    with pytest.raises((ExtractionCallStateError, ProviderSendFenceError)):
        invoke_prepared_extraction(
            plan.invocation,
            provider,
            store=ledger,
            grant=grant,
        )
    with pytest.raises(ProviderSendFenceError):
        grant.permit.consume_at_dispatch()
    assert called is False


def test_rag_shared_advisory_fence_uses_dedicated_connection_and_exact_unlock():
    events = []

    class Result:
        def scalar_one(self):
            return True

    class Connection:
        def exec_driver_sql(self, sql, params):
            events.append((sql, params))
            return Result()

        def close(self):
            events.append('close')

    connection = Connection()
    value = _run_shared_advisory_send_fence(
        connection_factory=lambda: connection,
        key=(10, -20),
        recheck=lambda: events.append('recheck'),
        send=lambda: events.append('send') or 'ok',
    )
    assert value == 'ok'
    assert events == [
        ('SELECT pg_advisory_lock_shared(%s, %s)', (10, -20)),
        'recheck',
        'send',
        ('SELECT pg_advisory_unlock_shared(%s, %s)', (10, -20)),
        'close',
    ]


def test_rag_shared_advisory_fence_invalidates_and_closes_on_uncertain_unlock():
    events = []

    class Result:
        def scalar_one(self):
            return False

    class Connection:
        def exec_driver_sql(self, sql, params):
            events.append((sql, params))
            return Result()

        def invalidate(self):
            events.append('invalidate')

        def close(self):
            events.append('close')

    with pytest.raises(ProviderSendFenceError, match='unlock'):
        _run_shared_advisory_send_fence(
            connection_factory=Connection,
            key=(1, 2),
            recheck=lambda: None,
            send=lambda: None,
        )
    assert events[-2:] == ['invalidate', 'close']
