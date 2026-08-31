from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from langsmith import tracing_context

from backend.app.agent_runtime.rag_cost_ledger import RagCostLedgerError

_T = TypeVar('_T')


class RagProviderTransportError(RuntimeError):
    """Body-blind provider transport refusal."""


class _GrantStore(Protocol):
    def consume_committed_grant(self, grant: object) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedProviderDispatchEnvelope:
    component: str
    request_bytes: bytes
    endpoint: str
    provider: str
    model: str
    service_tier: str
    model_config_snapshot_hmac: str
    provider_safety_snapshot_hmac: str

    def __post_init__(self) -> None:
        targets = {
            'query_embedding': (
                '/v1/embeddings', 'text-embedding-3-small'
            ),
            'answer_generation': (
                '/v1/chat/completions', 'gpt-5.4-mini-2026-03-17'
            ),
        }
        if (
            type(self.request_bytes) is not bytes or not self.request_bytes
            or self.component not in targets
            or (self.endpoint, self.model) != targets[self.component]
            or self.provider != 'openai' or self.service_tier != 'default'
            or any(
                type(value) is not str or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)
                for value in (
                    self.model_config_snapshot_hmac,
                    self.provider_safety_snapshot_hmac,
                )
            )
        ):
            raise ValueError('prepared provider dispatch envelope is invalid')


def dispatch_fenced_provider_call(
    *,
    store: _GrantStore,
    grant: object,
    prepared: PreparedProviderDispatchEnvelope,
    send: Callable[..., _T],
    evidence_fence: Callable[[Callable[[], _T]], _T],
    safety_before: Callable[[], str],
    safety_after: Callable[[], str],
) -> _T:
    """Dispatch exactly once with every ambient provider hook disabled.

    Neither payload nor response is passed to the cost authority.
    """
    if (
        type(prepared) is not PreparedProviderDispatchEnvelope
        or not callable(send)
        or getattr(send, 'server_owned_fenced', False) is not True
        or not callable(evidence_fence)
    ):
        raise RagProviderTransportError('provider send callable is unavailable')

    def dispatch() -> _T:
        try:
            grant_safety = grant.provider_safety_snapshot_hmac
            grant_component = grant.component
        except Exception:
            raise RagProviderTransportError(
                'committed dispatch grant is unavailable'
            ) from None
        before = safety_before()
        if (
            grant_component != prepared.component
            or before != prepared.provider_safety_snapshot_hmac
            or grant_safety != prepared.provider_safety_snapshot_hmac
        ):
            raise RagProviderTransportError('provider safety binding drifted')
        try:
            store.consume_committed_grant(grant)
        except (AttributeError, TypeError, RagCostLedgerError):
            raise RagProviderTransportError(
                'committed dispatch grant is unavailable'
            ) from None
        with tracing_context(enabled=False):
            result = send(
                prepared.request_bytes,
                max_retries=0, callbacks=[], cache=False, tracing=False,
            )
        if safety_after() != prepared.provider_safety_snapshot_hmac:
            raise RagProviderTransportError('provider safety binding drifted')
        return result

    return evidence_fence(dispatch)
