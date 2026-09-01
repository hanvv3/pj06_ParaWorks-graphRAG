from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from threading import RLock
from typing import Any, Generic, Protocol, TypeVar

from backend.app.agent_runtime.rag_advisory_locks import (
    RagLockOrderCapability,
    RagLockOrderCoordinator,
    RegisteredAdvisoryLock,
    acquire_advisory_lock,
    release_advisory_lock,
)
from backend.app.agent_runtime.rag_safety_identity import identities_match


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


class RagEvidenceFreshnessAuthority:
    """Concrete server-owned reader for the current C.5 evidence identity."""

    __slots__ = ('_load_current_identity',)

    def __init__(self, load_current_identity: Callable[[], str]) -> None:
        if not callable(load_current_identity):
            raise TypeError('evidence freshness authority is unavailable')
        self._load_current_identity = load_current_identity

    def require_current(self, expected_identity_hmac: str) -> None:
        if not identities_match(
            self._load_current_identity(), expected_identity_hmac
        ):
            raise ProviderSendFenceError('evidence identity changed before send')

    def snapshot_identity(self) -> str:
        value = self._load_current_identity()
        if not identities_match(value, value):
            raise ProviderSendFenceError('evidence identity is unavailable')
        return value


class RagEvidenceSendBarrier:
    """Concrete fresh-evidence barrier with SQLite and dedicated PG paths."""

    __slots__ = ('_connection_factory', '_freshness', '_mutex', '_registered_lock')

    def __init__(
        self,
        *,
        freshness: RagEvidenceFreshnessAuthority,
        connection_factory: Callable[[], object] | None = None,
        registered_lock: RegisteredAdvisoryLock | None = None,
    ) -> None:
        if type(freshness) is not RagEvidenceFreshnessAuthority:
            raise TypeError('evidence freshness authority is unavailable')
        if (connection_factory is None) != (registered_lock is None):
            raise ValueError('PostgreSQL evidence barrier is incomplete')
        self._freshness = freshness
        self._connection_factory = connection_factory
        self._registered_lock = registered_lock
        self._mutex = RLock()

    def snapshot_identity(self) -> str:
        return self._freshness.snapshot_identity()

    def run(
        self,
        *,
        expected_identity_hmac: str,
        operation: Callable[[], _T],
        order: RagLockOrderCoordinator,
        evidence_capability: RagLockOrderCapability,
        c5_capability: RagLockOrderCapability,
    ) -> _T:
        if not callable(operation):
            raise TypeError('provider operation is unavailable')
        order.require(evidence_capability, stage='evidence_shared_barrier')
        order.require(c5_capability, stage='c5_key_corpus')
        if self._connection_factory is None:
            with self._mutex:
                self._freshness.require_current(expected_identity_hmac)
                return operation()
        connection = self._connection_factory()
        locked = False
        try:
            acquire_advisory_lock(
                connection, self._registered_lock, shared=True  # type: ignore[arg-type]
            )
            locked = True
            self._freshness.require_current(expected_identity_hmac)
            return operation()
        finally:
            try:
                if locked:
                    release_advisory_lock(
                        connection,
                        self._registered_lock,  # type: ignore[arg-type]
                        shared=True,
                    )
            finally:
                connection.close()  # type: ignore[attr-defined]


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
