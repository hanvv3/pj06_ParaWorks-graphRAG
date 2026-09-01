from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from langsmith import tracing_context
from sqlalchemy.engine import Connection

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.provider_send_fence import (
    ProviderSendFenceError,
    RagEvidenceSendBarrier,
)
from backend.app.agent_runtime.rag_advisory_locks import begin_rag_lock_order
from backend.app.agent_runtime.rag_cost_ledger import RagCostLedger, RagCostLedgerError
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyService,
)
from backend.app.agent_runtime.rag_runtime_contracts import (
    AuthorizedProviderPolicySnapshot,
    CommittedRagDispatchGrant,
    RagProviderSafetyBinding,
)
from backend.app.agent_runtime.rag_safety_identity import (
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.rag.retrieval import (
    AnswerGenerationCostInput,
    QueryEmbeddingCostInput,
)

_T = TypeVar('_T')
_PREPARED_SEAL = object()


class RagProviderTransportError(RuntimeError):
    """Body-blind provider transport refusal."""


@dataclass(frozen=True, slots=True)
class RagProviderRequest:
    """Typed server input; callers never provide serialized provider bytes."""

    component: str
    provider: str
    model: str
    endpoint: str
    service_tier: str
    model_config_snapshot_hmac: str
    rendered_input_utf8: bytes = field(repr=False)
    response_schema_json: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        targets = {
            'query_embedding': ('/v1/embeddings', 'text-embedding-3-small'),
            'answer_generation': (
                '/v1/chat/completions', 'gpt-5.4-mini-2026-03-17'
            ),
        }
        try:
            text = self.rendered_input_utf8.decode('utf-8')
        except (AttributeError, UnicodeDecodeError):
            raise ValueError('rendered provider input is invalid') from None
        if (
            type(self.rendered_input_utf8) is not bytes
            or not text
            or self.component not in targets
            or (self.endpoint, self.model) != targets[self.component]
            or self.provider != 'openai'
            or self.service_tier != 'default'
            or (self.component == 'query_embedding')
            != (self.response_schema_json is None)
        ):
            raise ValueError('provider request target is invalid')
        for value in (
            self.model_config_snapshot_hmac,
        ):
            require_lower_hmac(value)


@dataclass(frozen=True, slots=True)
class _PreparedProviderDispatch:
    request_identity_hmac: str
    evidence_identity_hmac: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PreparedState:
    grant: object
    request: RagProviderRequest
    request_bytes: bytes = field(repr=False)
    snapshot: AuthorizedProviderPolicySnapshot
    binding: RagProviderSafetyBinding
    evidence_identity_hmac: str
    query_identity_hmac: str


class RagProviderDispatchAuthority:
    """Store-owned request builder and one-shot provider dispatch authority."""

    __slots__ = (
        '_barrier', '_client', '_connection_factory', '_prepared', '_safety',
        '_secret', '_store', '_timeout_seconds',
    )

    def __init__(
        self,
        *,
        store: RagCostLedger,
        provider_safety: RagProviderSafetyService,
        provider_connection_factory: Callable[[], Connection],
        evidence_barrier: RagEvidenceSendBarrier,
        identity_secret: bytes,
        timeout_seconds: int,
    ) -> None:
        if (
            type(store) is not RagCostLedger
            or type(provider_safety) is not RagProviderSafetyService
            or provider_safety is not store.provider_safety_authority
            or provider_connection_factory is not store.provider_connection_factory
            or type(evidence_barrier) is not RagEvidenceSendBarrier
            or type(identity_secret) is not bytes
            or not identity_secret
            or type(timeout_seconds) is not int
            or timeout_seconds <= 0
        ):
            raise TypeError('provider dispatch authority is unavailable')
        self._store = store
        self._safety = provider_safety
        self._connection_factory = provider_connection_factory
        self._barrier = evidence_barrier
        try:
            self._client = store._transport_provider_client()
        except RagCostLedgerError as exc:
            raise TypeError('provider dispatch authority is unavailable') from exc
        self._secret = identity_secret
        self._timeout_seconds = timeout_seconds
        self._prepared: dict[int, _PreparedState] = {}

    def prepare(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        request: RagProviderRequest,
    ) -> _PreparedProviderDispatch:
        if type(request) is not RagProviderRequest:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise TypeError('typed provider request is required')
        try:
            snapshot, binding, budget, query_identity_hmac = (
                self._store._transport_context(grant)
            )
        except RagCostLedgerError as exc:
            raise RagProviderTransportError('committed dispatch grant is unavailable') from exc
        if (
            request.component != grant.component
            or request.provider != snapshot.provider
            or request.model != snapshot.model
            or not hmac.compare_digest(
                request.model_config_snapshot_hmac,
                snapshot.authorized_model_config_snapshot_hmac,
            )
            or not hmac.compare_digest(
                grant.provider_safety_snapshot_hmac,
                binding.provider_safety_snapshot_hmac,
            )
        ):
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('provider request authority is misaligned')
        try:
            if request.component == 'query_embedding':
                expected_budget = self._store.cost_policy_authority.prepare_query_embedding(
                    QueryEmbeddingCostInput(
                        retrieval_query_utf8=request.rendered_input_utf8,
                        model_config_snapshot_hmac=(
                            request.model_config_snapshot_hmac
                        ),
                    )
                )
                body = {
                    'input': request.rendered_input_utf8.decode('utf-8'),
                    'model': request.model,
                }
            else:
                expected_budget = self._store.cost_policy_authority.prepare_answer_generation(
                    AnswerGenerationCostInput(
                        exact_messages_json=request.rendered_input_utf8,
                        exact_response_schema_json=request.response_schema_json,
                        model_config_snapshot_hmac=(
                            request.model_config_snapshot_hmac
                        ),
                    )
                )
                messages = json.loads(request.rendered_input_utf8)
                body = {
                    'messages': messages,
                    'model': request.model,
                    'response_format': json.loads(request.response_schema_json),
                    'service_tier': request.service_tier,
                }
        except (TypeError, ValueError):
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError(
                'provider request cost identity is invalid'
            ) from None
        if expected_budget != budget:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('provider request cost identity drifted')
        request_bytes = canonical_json_bytes(body)
        try:
            evidence_identity_hmac = self._barrier.snapshot_identity()
        except Exception:
            self._cancel_unconsumed_claim(
                grant,
                outcome='evidence_unavailable',
            )
            raise RagProviderTransportError(
                'evidence identity is unavailable'
            ) from None
        identity = self._request_identity(
            request=request,
            request_bytes=request_bytes,
            grant=grant,
            binding=binding,
            query_identity_hmac=query_identity_hmac,
            evidence_identity_hmac=evidence_identity_hmac,
        )
        prepared = _PreparedProviderDispatch(
            request_identity_hmac=identity,
            evidence_identity_hmac=evidence_identity_hmac,
            _seal=_PREPARED_SEAL,
        )
        self._prepared[id(prepared)] = _PreparedState(
            grant=grant,
            request=request,
            request_bytes=request_bytes,
            snapshot=snapshot,
            binding=binding,
            evidence_identity_hmac=evidence_identity_hmac,
            query_identity_hmac=query_identity_hmac,
        )
        return prepared

    def _cancel_unconsumed_claim(
        self,
        grant: object,
        *,
        outcome: str,
    ) -> None:
        if getattr(grant, 'consumed', None) is not False:
            return
        try:
            self._store.cancel_claim_before_dispatch(
                grant=grant,  # type: ignore[arg-type]
                outcome=outcome,
            )
        except RagCostLedgerError:
            raise RagProviderTransportError(
                'pre-send refusal could not be finalized'
            ) from None

    def dispatch(
        self,
        *,
        grant: CommittedRagDispatchGrant,
        prepared: object,
    ) -> _T:
        state = self._prepared.get(id(prepared))
        if (
            type(prepared) is not _PreparedProviderDispatch
            or prepared._seal is not _PREPARED_SEAL
            or state is None
            or state.grant is not grant
            or not hmac.compare_digest(
                prepared.request_identity_hmac,
                self._request_identity(
                    request=state.request,
                    request_bytes=state.request_bytes,
                    grant=grant,
                    binding=state.binding,
                    query_identity_hmac=state.query_identity_hmac,
                    evidence_identity_hmac=state.evidence_identity_hmac,
                ),
            )
        ):
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('prepared provider dispatch is unavailable')
        self._prepared.pop(id(prepared), None)
        order = begin_rag_lock_order('ordinary')
        sidecar_capability = order.acquire('provider_stable_sidecar')
        safety_capability = order.acquire('provider_safety_rows')
        try:
            def send() -> _T:
                cost_capability = order.acquire('agent_run_cost')
                self._store.consume_transport_grant(
                    grant,
                    order=order,
                    order_capability=cost_capability,
                )
                assistant_capability = order.acquire('optional_assistant')
                order.finish()
                with tracing_context(enabled=False):
                    result = self._client.send(
                        state.request_bytes,
                        timeout_seconds=self._timeout_seconds,
                        max_retries=0,
                        order=order,
                        order_capability=assistant_capability,
                    )
                return result

            with (
                self._connection_factory() as connection,
                self._safety.dispatch_barrier(
                    connection,
                    state.request.component,
                    state.snapshot,
                    expected_binding=state.binding,
                    order=order,
                    sidecar_capability=sidecar_capability,
                    safety_capability=safety_capability,
                ),
            ):
                projection_capability = order.acquire('projection_owner')
                with self._store.projection_owner_barrier(
                    grant.agent_run_id,
                    order=order,
                    order_capability=projection_capability,
                ):
                    evidence_capability = order.acquire('evidence_shared_barrier')
                    c5_capability = order.acquire('c5_key_corpus')
                    return self._barrier.run(
                        expected_identity_hmac=prepared.evidence_identity_hmac,
                        operation=send,
                        order=order,
                        evidence_capability=evidence_capability,
                        c5_capability=c5_capability,
                    )
        except (ProviderSendFenceError, RagCostLedgerError, RagProviderSafetyError) as exc:
            if getattr(grant, 'consumed', None) is False:
                refusal_outcome = (
                    'evidence_unavailable'
                    if isinstance(exc, ProviderSendFenceError)
                    else 'provider_safety_unavailable'
                )
                try:
                    self._store.cancel_claim_before_dispatch(
                        grant=grant,
                        outcome=refusal_outcome,
                    )
                except RagCostLedgerError as cancel_exc:
                    raise RagProviderTransportError(
                        'pre-send refusal could not be finalized'
                    ) from cancel_exc
            message = (
                'evidence provider-send fence refused dispatch'
                if isinstance(exc, ProviderSendFenceError)
                else 'provider dispatch authority refused dispatch'
            )
            raise RagProviderTransportError(message) from None
        except Exception:
            self._cancel_unconsumed_claim(
                grant,
                outcome='provider_safety_unavailable',
            )
            raise RagProviderTransportError('provider transport failed') from None

    def _request_identity(
        self,
        *,
        request: RagProviderRequest,
        request_bytes: bytes,
        grant: CommittedRagDispatchGrant,
        binding: RagProviderSafetyBinding,
        query_identity_hmac: str,
        evidence_identity_hmac: str,
    ) -> str:
        return rag_identity_hmac(
            {
                'component': request.component,
                'dispatch_fence_hmac': grant.dispatch_fence_hmac,
                'endpoint': request.endpoint,
                'evidence_identity_hmac': evidence_identity_hmac,
                'model': request.model,
                'model_config_snapshot_hmac': request.model_config_snapshot_hmac,
                'provider': request.provider,
                'provider_safety_snapshot_hmac': (
                    binding.provider_safety_snapshot_hmac
                ),
                'query_identity_hmac': query_identity_hmac,
                'rendered_input_sha256': hashlib.sha256(
                    request.rendered_input_utf8
                ).hexdigest(),
                'request_bytes_sha256': hashlib.sha256(request_bytes).hexdigest(),
                'service_tier': request.service_tier,
            },
            secret=self._secret,
            schema_version='rag-provider-dispatch-request:v1',
            policy_version='rag-provider-transport:v1',
        )
