from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from langsmith import tracing_context

from backend.app.agent_runtime.rag_cost_ledger import RagCostLedgerError

_T = TypeVar('_T')


class RagProviderTransportError(RuntimeError):
    """Body-blind provider transport refusal."""


class _GrantStore(Protocol):
    def consume_committed_grant(self, grant: object) -> None: ...


def dispatch_fenced_provider_call(
    *,
    store: _GrantStore,
    grant: object,
    payload: Any,
    send: Callable[..., _T],
) -> _T:
    """Dispatch exactly once with every ambient provider hook disabled.

    Neither payload nor response is passed to the cost authority.
    """
    if not callable(send):
        raise RagProviderTransportError('provider send callable is unavailable')
    try:
        store.consume_committed_grant(grant)
    except (AttributeError, TypeError, RagCostLedgerError):
        raise RagProviderTransportError('committed dispatch grant is unavailable') from None
    try:
        with tracing_context(enabled=False):
            return send(
                payload,
                max_retries=0,
                callbacks=[],
                cache=False,
                tracing=False,
            )
    except BaseException:
        raise
