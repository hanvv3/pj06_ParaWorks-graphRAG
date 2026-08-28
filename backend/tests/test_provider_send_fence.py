import copy
import json
import pickle
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from backend.app.agent_runtime import provider_send_fence as fence_module
from backend.app.agent_runtime.contracts import (
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
)
from backend.app.agent_runtime.provider_send_fence import (
    FencedProviderSendPermit,
    ProviderAttemptGrant,
    ProviderSendFenceError,
)
from backend.app.agent_runtime.review_v21_extraction import (
    ExtractionCallLedger,
    ExtractionProviderSafetySnapshot,
    build_prepared_extraction_plan_set,
)
from backend.app.core.config import Settings


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _grant(clock: _Clock, *, commit=None) -> ProviderAttemptGrant:
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
    ledger = ExtractionCallLedger(permit_monotonic=clock)
    context = ledger.claim_or_replay('workflow', plan)
    return ledger.mark_attempt_started(context, commit=commit)


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


def test_fenced_transport_consumes_only_at_dispatch_and_never_reads_or_logs_body(
    caplog,
):
    clock = _Clock()
    grant = _grant(clock)
    calls = []

    class _OpaqueBody:
        def __str__(self):  # pragma: no cover - a read is a contract failure
            raise AssertionError('transport inspected body')

    transport = grant.transport
    assert not grant.permit.consumed
    result = transport.dispatch(
        lambda *, timeout: calls.append(timeout) or 'ok',
        request_body=_OpaqueBody(),
    )
    assert result == 'ok'
    assert calls == [60]
    assert grant.permit.consumed
    assert 'OpaqueBody' not in caplog.text


def test_redirect_retry_or_second_dispatch_reuses_no_consumed_permit():
    grant = _grant(_Clock())
    transport = grant.transport
    transport.dispatch(lambda *, timeout: timeout)
    for kwargs in ({}, {'is_redirect': True}, {'is_retry': True}):
        with pytest.raises(ProviderSendFenceError):
            transport.dispatch(lambda *, timeout: timeout, **kwargs)


def test_restart_has_no_reconstructable_send_permit_and_marker_is_not_send_authority():
    with pytest.raises(TypeError):
        FencedProviderSendPermit('attempt-hmac', 15.0)  # type: ignore[call-arg]
    with pytest.raises((AttributeError, TypeError)):
        ProviderAttemptGrant.from_marker(  # type: ignore[attr-defined]
            attempt_id='attempt-hmac',
            provider_timeout_seconds=60,
        )


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


def test_fenced_transport_uses_the_task1_server_owned_hook_identity():
    from backend.app.agent_runtime.auto_review_cost_policy import (
        is_server_owned_fenced_send_hook,
    )

    transport = _grant(_Clock()).transport
    assert is_server_owned_fenced_send_hook(transport.http_hook)


def test_no_importable_callable_can_issue_a_provider_attempt_grant():
    issuer = getattr(fence_module, '_issue_provider_attempt_grant', None)
    assert issuer is None
