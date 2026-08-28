from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar


class ProviderSendFenceError(RuntimeError):
    """Bounded send-fence refusal; never contains a request or response body."""


class FencedProviderSendPermit(Protocol):
    """Read-only shape of a store-owned, process-local send permit."""

    @property
    def attempt_id(self) -> str: ...

    @property
    def send_start_deadline_monotonic(self) -> float: ...

    @property
    def consumed(self) -> bool: ...

    def consume_at_dispatch(self) -> None: ...


_T = TypeVar('_T')


class FencedOpenAITransport(Generic[_T]):
    """Non-constructible base for the body-blind store-owned transport."""

    __slots__ = ()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError('fenced transports can only be issued by the call store')

    @property
    def http_hook(self) -> object:
        raise NotImplementedError

    def _consume_from_store(self) -> None:
        raise NotImplementedError

    def _dispatch_consumed(
        self,
        send: Callable[..., _T],
        *,
        request_body: Any = None,
    ) -> _T:
        raise NotImplementedError


class ProviderAttemptGrant(Protocol, Generic[_T]):
    """Read-only shape of authority returned by a committed call store."""

    @property
    def attempt_id(self) -> str: ...

    @property
    def permit(self) -> FencedProviderSendPermit: ...

    @property
    def provider_timeout_seconds(self) -> int: ...

    @property
    def authoritative_lease_expires_at(self) -> datetime: ...
