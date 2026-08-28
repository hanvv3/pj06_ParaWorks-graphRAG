import copy
import json
import pickle
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.agent_runtime.provider_send_fence import (
    FencedOpenAITransport,
    FencedProviderSendPermit,
    ProviderAttemptGrant,
    ProviderSendFenceError,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _grant(clock: _Clock, *, window: float = 5.0) -> ProviderAttemptGrant:
    permit = FencedProviderSendPermit.issue(
        attempt_id='attempt-hmac',
        send_start_window_seconds=window,
        monotonic=clock,
    )
    return ProviderAttemptGrant(
        attempt_id='attempt-hmac',
        permit=permit,
        provider_timeout_seconds=60,
        authoritative_lease_expires_at=datetime.now(UTC) + timedelta(seconds=120),
    )


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

    late = _grant(clock, window=1)
    clock.value += 1
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

    transport = FencedOpenAITransport(grant)
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
    transport = FencedOpenAITransport(grant)
    transport.dispatch(lambda *, timeout: timeout)
    for kwargs in ({}, {'is_redirect': True}, {'is_retry': True}):
        with pytest.raises(ProviderSendFenceError):
            transport.dispatch(lambda *, timeout: timeout, **kwargs)


def test_restart_has_no_reconstructable_send_permit_and_marker_is_not_send_authority():
    with pytest.raises(TypeError):
        FencedProviderSendPermit('attempt-hmac', 15.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ProviderAttemptGrant.from_marker(  # type: ignore[attr-defined]
            attempt_id='attempt-hmac',
            provider_timeout_seconds=60,
        )


def test_attempt_grant_is_not_returned_before_marker_commit():
    events = []

    def commit():
        events.append('commit')

    grant = ProviderAttemptGrant.after_committed_marker(
        attempt_id='attempt-hmac',
        provider_timeout_seconds=60,
        send_start_window_seconds=5,
        authoritative_lease_expires_at=datetime.now(UTC) + timedelta(seconds=120),
        commit=commit,
        monotonic=_Clock(),
    )
    assert events == ['commit']
    assert grant.permit.attempt_id == 'attempt-hmac'
