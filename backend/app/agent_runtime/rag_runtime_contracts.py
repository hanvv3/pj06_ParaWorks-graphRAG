from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol, TypeAlias
from uuid import UUID

from backend.app.agent_runtime.provider_usage import RagResultOutcome
from backend.app.agent_runtime.rag_v2_contracts import (
    ProviderDispatchPermit,
    RagCutoverStage,
    RagRetrievalBackend,
    RagSurface,
)
from backend.app.rag.retrieval import RagPaidComponent, StrictProviderUsage

RagComponentClassification = Literal[
    'validated_success', 'pre_send_refusal', 'response_less_failure',
    'response_identity_invalid', 'embedding_payload_invalid',
    'usage_contract_invalid', 'usage_storage_invalid', 'known_overrun',
    'structured_output_invalid', 'citation_validation_failed',
    'evidence_validation_failed',
]
RagComponentTerminalOutcome = Literal[
    'component_succeeded', 'retriever_unavailable', 'model_provider_failed',
    'provider_response_identity_invalid', 'provider_embedding_payload_invalid',
    'provider_usage_overrun', 'provider_safety_unavailable',
    'structured_output_invalid', 'citation_validation_failed',
    'evidence_unavailable', 'abandoned_unknown',
]
RagProviderSafetyAction = Literal['unchanged', 'block_overrun', 'block_remediation']
RagRunTerminalOutcome: TypeAlias = RagResultOutcome | Literal[
    'serving_index_not_ready', 'shadow_match', 'shadow_mismatch',
    'persistence_failed', 'abandoned_unknown', 'live_corpus_snapshot_changed',
]

ADMISSION_SOURCE_WINDOWS = (
    'rag-v2:admission:enforce:ask:keyword',
    'rag-v2:admission:enforce:ask:pgvector',
    'rag-v2:admission:enforce:search:keyword',
    'rag-v2:admission:enforce:search:pgvector',
    'rag-v2:admission:enforce:assistant:keyword',
    'rag-v2:admission:enforce:assistant:pgvector',
    'rag-v2:admission:shadow:ask:pgvector',
    'rag-v2:admission:shadow:search:pgvector',
    'rag-v2:admission:shadow:assistant:pgvector',
)
RagAdmissionSourceWindow: TypeAlias = Literal[
    'rag-v2:admission:enforce:ask:keyword',
    'rag-v2:admission:enforce:ask:pgvector',
    'rag-v2:admission:enforce:search:keyword',
    'rag-v2:admission:enforce:search:pgvector',
    'rag-v2:admission:enforce:assistant:keyword',
    'rag-v2:admission:enforce:assistant:pgvector',
    'rag-v2:admission:shadow:ask:pgvector',
    'rag-v2:admission:shadow:search:pgvector',
    'rag-v2:admission:shadow:assistant:pgvector',
]
TERMINAL_SOURCE_WINDOWS = (
    'rag-v2:ask:keyword', 'rag-v2:ask:pgvector',
    'rag-v2:search:keyword', 'rag-v2:search:pgvector',
    'rag-v2:assistant:keyword', 'rag-v2:assistant:pgvector',
    'rag-v2:shadow:ask:keyword', 'rag-v2:shadow:ask:pgvector',
    'rag-v2:shadow:search:keyword', 'rag-v2:shadow:search:pgvector',
    'rag-v2:shadow:assistant:keyword', 'rag-v2:shadow:assistant:pgvector',
    'rag-v2:final-error:ask:keyword', 'rag-v2:final-error:ask:pgvector',
    'rag-v2:final-error:search:keyword', 'rag-v2:final-error:search:pgvector',
    'rag-v2:final-error:assistant:keyword', 'rag-v2:final-error:assistant:pgvector',
)


def admission_source_window(*, mode: str, surface: str, backend: str) -> str:
    candidate = f'rag-v2:admission:{mode}:{surface}:{backend}'
    if candidate not in ADMISSION_SOURCE_WINDOWS:
        raise ValueError('RAG admission source window is invalid')
    return candidate


def terminal_source_window(*, stage: str, surface: str, backend: str) -> str:
    prefixes = {
        'product': 'rag-v2',
        'shadow': 'rag-v2:shadow',
        'final_error': 'rag-v2:final-error',
    }
    prefix = prefixes.get(stage)
    candidate = '' if prefix is None else f'{prefix}:{surface}:{backend}'
    if candidate not in TERMINAL_SOURCE_WINDOWS:
        raise ValueError('RAG terminal source window is invalid')
    return candidate
RagTerminalSourceWindow: TypeAlias = Literal[
    'rag-v2:ask:keyword', 'rag-v2:ask:pgvector',
    'rag-v2:search:keyword', 'rag-v2:search:pgvector',
    'rag-v2:assistant:keyword', 'rag-v2:assistant:pgvector',
    'rag-v2:shadow:ask:keyword', 'rag-v2:shadow:ask:pgvector',
    'rag-v2:shadow:search:keyword', 'rag-v2:shadow:search:pgvector',
    'rag-v2:shadow:assistant:keyword', 'rag-v2:shadow:assistant:pgvector',
    'rag-v2:final-error:ask:keyword', 'rag-v2:final-error:ask:pgvector',
    'rag-v2:final-error:search:keyword', 'rag-v2:final-error:search:pgvector',
    'rag-v2:final-error:assistant:keyword', 'rag-v2:final-error:assistant:pgvector',
]


@dataclass(frozen=True, slots=True)
class RagRunAdmission:
    agent_run_id: int
    surface: RagSurface
    mode: Literal['shadow', 'enforce']
    cutover_stage: RagCutoverStage
    configured_backend: RagRetrievalBackend
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1']
    current_text_hmac: str
    retrieval_query_hmac: str
    security_scope_fingerprint: str
    query_embedding_provider_policy_snapshot_hmac: str | None
    answer_provider_policy_snapshot_hmac: str | None
    admission_cache_identity_hmac: str
    source_window: RagAdmissionSourceWindow
    status: Literal['running']
    run_record_phase: Literal['admission']
    component_order: tuple[Literal['query_embedding'], Literal['answer_generation']]
    total_reserved_cost_usd: Decimal
    total_charged_cost_usd: Decimal
    runtime_cost_snapshot_hmac: str


_CLASSIFIED_OBSERVATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class _ClassifiedProviderObservation:
    component: RagPaidComponent
    classification: RagComponentClassification
    terminal_outcome: RagComponentTerminalOutcome
    provider_dispatch_started: bool
    provider_response_received: bool
    strict_usage: StrictProviderUsage | None
    actual_cost_usd: Decimal | None
    safety_action: RagProviderSafetyAction
    _seal: object
    _consumed: bool = False

    def _consume(self) -> None:
        if self._seal is not _CLASSIFIED_OBSERVATION_SEAL or self._consumed:
            raise TypeError('classified provider observation is unavailable')
        object.__setattr__(self, '_consumed', True)

    def __copy__(self):
        raise TypeError('classified provider observations cannot be copied')

    def __deepcopy__(self, _memo: object):
        raise TypeError('classified provider observations cannot be copied')

    def __reduce_ex__(self, _protocol: int):
        raise TypeError('classified provider observations cannot be serialized')


def _issue_classified_provider_observation(
    *,
    component: RagPaidComponent,
    classification: RagComponentClassification,
    terminal_outcome: RagComponentTerminalOutcome,
    provider_dispatch_started: bool,
    provider_response_received: bool,
    strict_usage: StrictProviderUsage | None,
    actual_cost_usd: Decimal | None,
    safety_action: RagProviderSafetyAction,
) -> _ClassifiedProviderObservation:
    return _ClassifiedProviderObservation(
        component=component,
        classification=classification,
        terminal_outcome=terminal_outcome,
        provider_dispatch_started=provider_dispatch_started,
        provider_response_received=provider_response_received,
        strict_usage=strict_usage,
        actual_cost_usd=actual_cost_usd,
        safety_action=safety_action,
        _seal=_CLASSIFIED_OBSERVATION_SEAL,
    )


@dataclass(frozen=True, slots=True)
class RagComponentFinal:
    agent_run_id: int
    component: RagPaidComponent
    terminal_outcome: RagComponentTerminalOutcome | None
    dispatch_state: Literal['terminal', 'abandoned_unknown']
    attempted: bool
    dispatch_count: Literal[0, 1]
    reserved_input_tokens: int
    reserved_output_tokens: int
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    reserved_cost_usd: Decimal
    charged_cost_usd: Decimal
    charge_basis: Literal['zero', 'actual', 'reserved']
    overrun: bool
    provider: str
    model: str
    authorized_model_config_version: str
    authorized_model_config_snapshot_hmac: str
    authorized_cost_policy_version: str
    authorized_token_estimator_version: str
    authorized_policy_snapshot_hmac: str
    dispatch_fence_hmac: str | None
    process_instance_hmac: str | None
    parent_status: Literal['running', 'complete', 'failed']
    parent_run_record_phase: Literal[
        'admission', 'cost_finalized_pending_projection', 'final', 'admission_only'
    ]
    parent_total_charged_cost_usd: Decimal
    projection_owner_fence_hmac: str | None
    runtime_cost_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class RagRunTerminal:
    agent_run_id: int
    status: Literal['complete', 'failed']
    run_record_phase: Literal['final', 'admission_only']
    outcome: RagRunTerminalOutcome
    admission_cache_identity_hmac: str
    source_window: RagTerminalSourceWindow | RagAdmissionSourceWindow
    cache_key: str
    total_reserved_cost_usd: Decimal
    total_charged_cost_usd: Decimal
    component_finals: tuple[RagComponentFinal, RagComponentFinal]
    projection_owner_fence_hmac: str | None
    runtime_cost_snapshot_hmac: str
    terminal_identity_hmac: str | None
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class AuthorizedProviderPolicySnapshot:
    component: RagPaidComponent
    provider: str
    model: str
    reasoning_or_config_identity: str
    authorized_model_config_version: str
    authorized_model_config_snapshot_hmac: str
    authorized_cost_policy_version: str
    authorized_token_estimator_version: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    authorized_policy_snapshot_hmac: str


@dataclass(frozen=True, slots=True)
class RagProviderSafetyBinding:
    policy_snapshot: AuthorizedProviderPolicySnapshot
    authority_uuid: UUID
    designated_environment_id: str
    envelope_digest: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    global_safety_generation: int
    family_state: Literal['ready']
    family_state_version: int
    family_safety_generation: int
    provider_safety_snapshot_hmac: str


class CommittedRagDispatchGrant(ProviderDispatchPermit, Protocol):
    @property
    def agent_run_id(self) -> int: ...
    @property
    def component(self) -> RagPaidComponent: ...
    @property
    def reserved_cost_usd(self) -> Decimal: ...
    @property
    def dispatch_fence_hmac(self) -> str: ...
    @property
    def provider_safety_snapshot_hmac(self) -> str: ...
