from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
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


def run_shared_advisory_send_fence(
    *,
    connection_factory: Callable[[], object],
    key: tuple[int, int],
    recheck: Callable[[], None],
    send: Callable[[], _T],
) -> _T:
    """Run the final evidence recheck/send under a dedicated shared PG lock."""
    connection = connection_factory()
    locked = False
    try:
        connection.exec_driver_sql(  # type: ignore[attr-defined]
            'SELECT pg_advisory_lock_shared(%s, %s)', key
        )
        locked = True
        recheck()
        return send()
    finally:
        unlock_error: BaseException | None = None
        valid_unlock = not locked
        if locked:
            try:
                result = connection.exec_driver_sql(  # type: ignore[attr-defined]
                    'SELECT pg_advisory_unlock_shared(%s, %s)', key
                )
                valid_unlock = result.scalar_one() is True
            except BaseException as exc:
                unlock_error = exc
                with suppress(Exception):
                    connection.invalidate()  # type: ignore[attr-defined]
            finally:
                if not valid_unlock:
                    with suppress(Exception):
                        connection.invalidate()  # type: ignore[attr-defined]
        connection.close()  # type: ignore[attr-defined]
        if unlock_error is not None:
            raise unlock_error
        if locked and not valid_unlock:
            raise ProviderSendFenceError('shared advisory unlock was not confirmed')
