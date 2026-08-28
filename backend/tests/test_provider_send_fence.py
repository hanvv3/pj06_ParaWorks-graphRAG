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
    _issue_provider_attempt_grant,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _grant(clock: _Clock, *, window: float = 5.0) -> ProviderAttemptGrant:
    return _issue_provider_attempt_grant(
        attempt_id='attempt-hmac',
        provider_timeout_seconds=60,
        send_start_window_seconds=window,
        authoritative_lease_expires_at=datetime.now(UTC) + timedelta(seconds=120),
        commit=lambda: None,
        monotonic=clock,
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

    grant = _issue_provider_attempt_grant(
        attempt_id='attempt-hmac',
        provider_timeout_seconds=60,
        send_start_window_seconds=5,
        authoritative_lease_expires_at=datetime.now(UTC) + timedelta(seconds=120),
        commit=commit,
        monotonic=_Clock(),
    )
    assert events == ['commit']
    assert grant.permit.attempt_id == 'attempt-hmac'


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
        assert 'attempt-hmac' not in repr(value)
        for serializer in (copy.copy, copy.deepcopy, pickle.dumps, json.dumps):
            with pytest.raises((TypeError, ValueError)):
                serializer(value)


def test_fenced_transport_uses_the_task1_server_owned_hook_identity():
    from backend.app.agent_runtime.auto_review_cost_policy import (
        is_server_owned_fenced_send_hook,
    )

    transport = FencedOpenAITransport(_grant(_Clock()))
    assert is_server_owned_fenced_send_hook(transport.http_hook)
