from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from threading import Lock
from time import monotonic as _monotonic
from typing import Any, Generic, TypeVar


class ProviderSendFenceError(RuntimeError):
    """Bounded send-fence refusal; never contains a request or response body."""


_ISSUER = object()


class FencedProviderSendPermit:
    """Opaque, process-local, one-use authority consumed at HTTP dispatch."""

    __slots__ = (
        '_attempt_id',
        '_deadline',
        '_monotonic',
        '_consumed',
        '_invalidated',
        '_lock',
    )

    def __init__(
        self,
        issuer: object,
        attempt_id: str,
        deadline: float,
        monotonic: Callable[[], float],
    ) -> None:
        if issuer is not _ISSUER:
            raise TypeError(
                'provider send permits can only be issued by the call store'
            )
        self._attempt_id = attempt_id
        self._deadline = deadline
        self._monotonic = monotonic
        self._consumed = False
        self._invalidated = False
        self._lock = Lock()

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def send_start_deadline_monotonic(self) -> float:
        return self._deadline

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    def consume_at_dispatch(self) -> None:
        with self._lock:
            if self._consumed:
                raise ProviderSendFenceError(
                    'provider send permit was already consumed'
                )
            if self._invalidated or self._monotonic() >= self._deadline:
                raise ProviderSendFenceError('provider send permit expired')
            self._consumed = True

    def _invalidate_from_store(self) -> None:
        with self._lock:
            self._invalidated = True

    def __copy__(self):
        raise TypeError('provider send permits cannot be copied')

    def __deepcopy__(self, memo):
        del memo
        raise TypeError('provider send permits cannot be copied')

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError('provider send permits cannot be serialized')

    def __repr__(self) -> str:
        return '<FencedProviderSendPermit opaque>'


class ProviderAttemptGrant:
    """Opaque grant issued only after a call-store marker commit."""

    __slots__ = (
        '_attempt_id',
        '_permit',
        '_provider_timeout_seconds',
        '_authoritative_lease_expires_at',
    )

    def __init__(
        self,
        issuer: object,
        *,
        attempt_id: str,
        permit: FencedProviderSendPermit,
        provider_timeout_seconds: int,
        authoritative_lease_expires_at: datetime,
    ) -> None:
        if issuer is not _ISSUER:
            raise TypeError(
                'provider attempt grants can only be issued by the call store'
            )
        self._attempt_id = attempt_id
        self._permit = permit
        self._provider_timeout_seconds = provider_timeout_seconds
        self._authoritative_lease_expires_at = authoritative_lease_expires_at

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def permit(self) -> FencedProviderSendPermit:
        return self._permit

    @property
    def provider_timeout_seconds(self) -> int:
        return self._provider_timeout_seconds

    @property
    def authoritative_lease_expires_at(self) -> datetime:
        return self._authoritative_lease_expires_at

    @classmethod
    def from_marker(cls, **values: Any) -> ProviderAttemptGrant:
        del values
        raise TypeError('a durable attempt marker is not provider send authority')

    def __copy__(self):
        raise TypeError('provider attempt grants cannot be copied')

    def __deepcopy__(self, memo):
        del memo
        raise TypeError('provider attempt grants cannot be copied')

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError('provider attempt grants cannot be serialized')

    def __repr__(self) -> str:
        return '<ProviderAttemptGrant opaque>'


def _issue_provider_attempt_grant(
    *,
    attempt_id: str,
    provider_timeout_seconds: int,
    send_start_window_seconds: int | float,
    authoritative_lease_expires_at: datetime,
    commit: Callable[[], None],
    monotonic: Callable[[], float] = _monotonic,
) -> ProviderAttemptGrant:
    """Call-store-only factory; commit must return before authority exists."""
    if (
        not attempt_id
        or provider_timeout_seconds <= 0
        or send_start_window_seconds <= 0
    ):
        raise ValueError('attempt, timeout, and send-start window are required')
    commit()
    permit = FencedProviderSendPermit(
        _ISSUER,
        attempt_id,
        monotonic() + send_start_window_seconds,
        monotonic,
    )
    return ProviderAttemptGrant(
        _ISSUER,
        attempt_id=attempt_id,
        permit=permit,
        provider_timeout_seconds=provider_timeout_seconds,
        authoritative_lease_expires_at=authoritative_lease_expires_at,
    )


_T = TypeVar('_T')


class FencedOpenAITransport(Generic[_T]):
    """Body-blind one-dispatch transport boundary."""

    __slots__ = ('_grant', '_dispatched', '_http_hook')

    def __init__(self, grant: ProviderAttemptGrant) -> None:
        from backend.app.agent_runtime.auto_review_cost_policy import (
            _SERVER_OWNED_FENCED_SEND_HOOK,
        )

        self._grant = grant
        self._dispatched = False
        self._http_hook = _SERVER_OWNED_FENCED_SEND_HOOK

    @property
    def http_hook(self) -> object:
        return self._http_hook

    def dispatch(
        self,
        send: Callable[..., _T],
        *,
        request_body: Any = None,
        is_redirect: bool = False,
        is_retry: bool = False,
    ) -> _T:
        # The body is deliberately accepted but never inspected, rendered, or logged.
        del request_body
        if self._dispatched or is_redirect or is_retry:
            raise ProviderSendFenceError(
                'redirect, retry, or second dispatch is forbidden'
            )
        self._grant.permit.consume_at_dispatch()
        self._dispatched = True
        return send(timeout=self._grant.provider_timeout_seconds)
