from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from hashlib import sha256
from threading import Lock, RLock
from time import monotonic
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.agent_runtime.auto_review_cost_policy import (
    _SERVER_OWNED_FENCED_SEND_HOOK,
    AUTO_REVIEW_COST_POLICY_VERSION,
    AUTO_REVIEW_FRAMING_SAFETY_TOKENS,
    AUTO_REVIEW_MAX_BATCH_COST_USD,
    AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH,
    AUTO_REVIEW_MAX_INPUT_TOKENS,
    AUTO_REVIEW_MAX_OUTPUT_TOKENS,
    AUTO_REVIEW_MAX_PROVIDER_ATTEMPTS,
    AUTO_REVIEW_MAX_VALIDATION_COST_USD,
    AUTO_REVIEW_REASONING_EFFORT,
    AUTO_REVIEW_REPLY_PRIMING_TOKENS,
    AUTO_REVIEW_TOKENIZER_ENCODING,
    AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
    AUTO_REVIEW_VALIDATOR_MODEL,
    AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
    AUTO_REVIEW_VALIDATOR_PROVIDER,
)
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.agent_runtime.keyed_mutation_guard import KeyedMutationGuard
from backend.app.agent_runtime.provider_send_fence import ProviderSendFenceError
from backend.app.core.config import Settings
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AutoReviewExtractionCall,
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    AutoReviewValidation,
    AutoReviewValidationCall,
    Source,
)
from backend.app.schemas.auto_review import (
    AUTO_REVIEW_MAX_INPUT_CHARS,
    AUTO_REVIEW_TOKEN_ESTIMATOR_VERSION,
    AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
    AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
    AutoReviewPolicyDecision,
    AutoReviewPolicyReasonCode,
    CandidateValidationResult,
)

VALIDATION_KEY_SCHEMA_VERSION = 'auto-review-validation-key:v1'
VALIDATION_BATCH_SCHEMA_VERSION = 'auto-review-validation-batch:v1'
VALIDATION_OWNER_PERMISSION_SCHEMA_VERSION = 'auto-review-owner-permission:v1'
VALIDATION_SOURCE_STATE_SCHEMA_VERSION = 'auto-review-source-state:v1'


@dataclass(frozen=True, slots=True)
class ValidationSourceState:
    canonical_row_id: int
    canonical_source_type: str
    content_signature: str
    content_fingerprint: str
    permission_level: str


def derive_validation_owner_permission_hmac(
    *,
    owner_subject_id: str,
    permission_levels: Sequence[str],
    settings: Settings,
) -> str:
    return build_keyed_fingerprint(
        {
            'owner_subject_id': owner_subject_id,
            'permission_levels': sorted(set(permission_levels)),
        },
        settings=settings,
        schema_version=VALIDATION_OWNER_PERMISSION_SCHEMA_VERSION,
        policy_version=VALIDATION_OWNER_PERMISSION_SCHEMA_VERSION,
    )


def derive_validation_source_state_hmac(
    *,
    workflow_thread_id: str,
    sources: Sequence[ValidationSourceState],
    settings: Settings,
) -> str:
    ordered = sorted(
        sources,
        key=lambda item: (
            item.canonical_row_id,
            item.canonical_source_type,
            item.content_signature,
        ),
    )
    return build_keyed_fingerprint(
        {
            'workflow_thread_id': workflow_thread_id,
            'sources': [
                {
                    'canonical_row_id': item.canonical_row_id,
                    'canonical_source_type': item.canonical_source_type,
                    'content_signature': item.content_signature,
                    'content_fingerprint': item.content_fingerprint,
                    'permission_level': item.permission_level,
                }
                for item in ordered
            ],
        },
        settings=settings,
        schema_version=VALIDATION_SOURCE_STATE_SCHEMA_VERSION,
        policy_version=VALIDATION_SOURCE_STATE_SCHEMA_VERSION,
    )


@dataclass(frozen=True, slots=True)
class AutoReviewValidationIdentity:
    workflow_execution_identity_hmac: str
    security_scope_hmac: str
    candidate_key: str
    evidence_version_hash: str
    normalized_claim_fingerprint: str
    candidate_generation_fingerprint: str
    validator_provider: Literal['openai']
    validator_model: Literal['gpt-5.6-terra']
    reasoning_effort: Literal['medium']
    validator_prompt_version: Literal['auto-review-validation:v1']
    validator_output_contract_version: Literal['candidate-validation-batch:v1']
    policy_version: Literal['auto-review-policy:v1']
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    token_estimator_version: Literal['openai-o200k-chat:v1']
    tokenizer_encoding: Literal['o200k_base']
    max_input_tokens: Literal[6000]
    max_output_tokens: Literal[3072]
    max_candidates_per_batch: Literal[4]
    max_batches_per_workflow: Literal[2]
    max_candidates_per_workflow: Literal[5]
    max_provider_attempts: Literal[1]
    provider_timeout_seconds: int
    provider_send_start_window_seconds: int
    provider_attempt_lease_seconds: int
    provider_commit_grace_seconds: int
    input_cost_per_1m_tokens: Decimal
    output_cost_per_1m_tokens: Decimal
    cost_policy_version: Literal['auto-review-cost:v1']
    provider_safety_state_version: int
    rollout_control_epoch: int
    authorized_percentage_at_launch: Literal[0, 10, 100]
    rollout_authorization_generation: int
    confirmed_extraction_cost_ceiling_usd: Decimal
    confirmed_validation_cost_ceiling_usd: Decimal
    confirmed_total_cost_ceiling_usd: Decimal

    def fingerprint_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = (
                format(value, 'f') if isinstance(value, Decimal) else value
            )
        return payload


def derive_validation_key(
    identity: AutoReviewValidationIdentity,
    *,
    settings: Settings,
) -> str:
    return build_keyed_fingerprint(
        identity.fingerprint_payload(),
        settings=settings,
        schema_version=VALIDATION_KEY_SCHEMA_VERSION,
        policy_version=identity.policy_version,
    )


@dataclass(frozen=True, slots=True)
class ValidationCandidateClaim:
    review_item_id: int
    validation_key: str
    identity: AutoReviewValidationIdentity


@dataclass(frozen=True, slots=True)
class ValidationBatchClaimRequest:
    workflow_thread_id: str
    batch_fingerprint: str
    candidates: tuple[ValidationCandidateClaim, ...]
    max_provider_attempts: Literal[1]
    prepared_content_hmac: str
    serialized_character_count: int
    framed_input_tokens: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class ValidationClaimResult:
    disposition: Literal['claimed', 'replayed', 'busy']
    validation_call_id: int
    lease_token: str | None
    lease_expires_at: datetime | None
    terminal_status: Literal['completed', 'failed'] | None
    failure_reason_code: AutoReviewPolicyReasonCode | None
    completed: tuple[CompletedValidationProjection, ...]


@dataclass(frozen=True, slots=True)
class ValidationAttemptAdmission:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str
    prepared_content_hmac: str
    serialized_character_count: int
    framed_input_tokens: int
    max_output_tokens: int
    recomputed_reserved_cost_usd: Decimal
    expected_owner_permission_hmac: str
    expected_source_state_hmac: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    token_estimator_version: str
    cost_policy_version: str
    expected_effective_mode: Literal['shadow', 'enforce']
    provider_timeout_seconds: int
    provider_send_start_window_seconds: int
    provider_attempt_lease_seconds: int
    provider_commit_grace_seconds: int


@dataclass(frozen=True, slots=True)
class ValidationLockedContext:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str


@dataclass(frozen=True, slots=True)
class ValidationCandidateCompletion:
    review_item_id: int
    validation_key: str
    result: CandidateValidationResult
    policy_decision: AutoReviewPolicyDecision
    policy_reason_codes: tuple[AutoReviewPolicyReasonCode, ...]


@dataclass(frozen=True, slots=True)
class ValidationBatchCompletion:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str
    candidates: tuple[ValidationCandidateCompletion, ...]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ValidationBatchFailure:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str
    reason_code: AutoReviewPolicyReasonCode
    usage_known: bool
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class CompletedValidationProjection:
    validation_call_id: int
    validation_id: int
    review_item_id: int
    result: CandidateValidationResult
    policy_decision: AutoReviewPolicyDecision
    policy_reason_codes: tuple[AutoReviewPolicyReasonCode, ...]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class ValidationCallSnapshot:
    status: Literal['claimed', 'completed', 'failed']
    provider_attempt_count: int
    charged_input_tokens: int | None
    charged_output_tokens: int | None
    charged_cost_usd: Decimal
    budget_overrun: bool
    budget_overrun_cost_usd: Decimal
    terminal_at: datetime | None


class _CommittedPermit:
    __slots__ = (
        '_attempt_id',
        '_clock',
        '_consumed',
        '_deadline',
        '_invalidated',
        '_lock',
    )

    def __init__(
        self,
        *,
        attempt_id: str,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self._attempt_id = attempt_id
        self._deadline = deadline
        self._clock = clock
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
                raise ProviderSendFenceError('provider send permit was consumed')
            if self._invalidated or self._clock() >= self._deadline:
                raise ProviderSendFenceError('provider send permit expired')
            self._consumed = True

    def invalidate_from_store(self) -> None:
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


@dataclass(frozen=True, slots=True)
class _CommittedGrant:
    attempt_id: str
    permit: _CommittedPermit
    provider_timeout_seconds: int
    authoritative_lease_expires_at: datetime

    def __repr__(self) -> str:
        return '<ProviderAttemptGrant opaque>'


@dataclass(slots=True)
class _CallState:
    request: ValidationBatchClaimRequest
    validation_call_id: int
    lease_token: str
    lease_expires_at: datetime
    provider_attempt_count: int = 0
    attempt_started_at: datetime | None = None
    provider_dispatched: bool = False
    status: Literal['claimed', 'completed', 'failed'] = 'claimed'
    completed: tuple[CompletedValidationProjection, ...] = ()
    failure_reason_code: AutoReviewPolicyReasonCode | None = None
    charged_input_tokens: int | None = None
    charged_output_tokens: int | None = None
    charged_cost_usd: Decimal = Decimal('0')
    budget_overrun: bool = False
    budget_overrun_cost_usd: Decimal = Decimal('0')
    terminal_at: datetime | None = None
    cancelled: bool = False


class ValidationStoreError(RuntimeError):
    """Bounded validation-ledger refusal."""


class ValidationCallLedger:
    """Thread-safe deterministic claim ledger used by SQLite/unit tests."""

    def __init__(
        self,
        *,
        settings: Settings,
        db_clock: Callable[[], datetime],
        lease_token_factory: Callable[[], str] | None = None,
        permit_monotonic: Callable[[], float] = monotonic,
    ) -> None:
        self._settings = settings
        self._db_clock = db_clock
        self._lease_token_factory = lease_token_factory or (
            lambda: secrets.token_hex(32)
        )
        self._permit_monotonic = permit_monotonic
        self._calls: dict[tuple[str, str], _CallState] = {}
        self._active_grants: dict[str, tuple[_CommittedGrant, _CallState]] = {}
        self._next_call_id = 1
        self._next_validation_id = 1
        self._lock = RLock()
        self._validation_breaker_open = False
        self._provider_safety_event_count = 0

    @property
    def validation_breaker_open(self) -> bool:
        with self._lock:
            return self._validation_breaker_open

    @property
    def provider_safety_event_count(self) -> int:
        with self._lock:
            return self._provider_safety_event_count

    def claim_or_replay(
        self,
        request: ValidationBatchClaimRequest,
    ) -> ValidationClaimResult:
        self._validate_request(request)
        key = (request.workflow_thread_id, request.batch_fingerprint)
        with self._lock:
            now = self._db_clock()
            state = self._calls.get(key)
            if state is None:
                state = self._new_claim(request, now=now)
                self._calls[key] = state
                return self._claimed_result(state)
            if state.request != request:
                raise ValidationStoreError('validation claim identity changed')
            if state.status == 'completed':
                if not state.completed:
                    raise ValidationStoreError(
                        'completed validation projection is unavailable'
                    )
                return ValidationClaimResult(
                    disposition='replayed',
                    validation_call_id=state.validation_call_id,
                    lease_token=None,
                    lease_expires_at=None,
                    terminal_status='completed',
                    failure_reason_code=None,
                    completed=state.completed,
                )
            if state.status == 'failed':
                if state.failure_reason_code is None:
                    raise ValidationStoreError(
                        'failed validation reason is unavailable'
                    )
                return ValidationClaimResult(
                    disposition='replayed',
                    validation_call_id=state.validation_call_id,
                    lease_token=None,
                    lease_expires_at=None,
                    terminal_status='failed',
                    failure_reason_code=state.failure_reason_code,
                    completed=(),
                )
            if now < state.lease_expires_at or state.provider_attempt_count != 0:
                return ValidationClaimResult(
                    disposition='busy',
                    validation_call_id=state.validation_call_id,
                    lease_token=None,
                    lease_expires_at=state.lease_expires_at,
                    terminal_status=None,
                    failure_reason_code=None,
                    completed=(),
                )
            state.lease_token = self._next_lease_token()
            state.lease_expires_at = now + timedelta(
                seconds=self._lease_seconds(request)
            )
            return self._claimed_result(state)

    def mark_attempt_started(
        self,
        admission: ValidationAttemptAdmission,
        *,
        commit: Callable[[], None],
    ) -> _CommittedGrant:
        key = (admission.workflow_thread_id, admission.batch_fingerprint)
        with self._lock:
            state = self._calls.get(key)
            now = self._db_clock()
            if (
                state is None
                or state.validation_call_id != admission.validation_call_id
                or state.lease_token != admission.lease_token
                or state.provider_attempt_count != 0
                or now >= state.lease_expires_at
            ):
                raise ValidationStoreError('validation attempt is not claimable')
            self._validate_admission(state, admission)
            prior_expiry = state.lease_expires_at
            state.provider_attempt_count = 1
            state.attempt_started_at = now
            state.lease_expires_at = now + timedelta(
                seconds=admission.provider_attempt_lease_seconds
            )
            attempt_id = sha256(
                f'{state.lease_token}:attempt:1'.encode()
            ).hexdigest()
            try:
                commit()
            except Exception:
                state.provider_attempt_count = 0
                state.attempt_started_at = None
                state.lease_expires_at = prior_expiry
                raise
            permit = _CommittedPermit(
                attempt_id=attempt_id,
                deadline=(
                    self._permit_monotonic()
                    + admission.provider_send_start_window_seconds
                ),
                clock=self._permit_monotonic,
            )
            grant = _CommittedGrant(
                attempt_id=attempt_id,
                permit=permit,
                provider_timeout_seconds=admission.provider_timeout_seconds,
                authoritative_lease_expires_at=state.lease_expires_at,
            )
            self._active_grants[attempt_id] = (grant, state)
            return grant

    def dispatch_prepared_validation(
        self,
        *,
        invocation: Any,
        provider: Callable[..., Any],
        grant: object,
    ) -> Any:
        attempt_id = getattr(grant, 'attempt_id', None)
        with self._lock:
            active = self._active_grants.get(attempt_id)
            if active is None or active[0] is not grant:
                raise ValidationStoreError(
                    'committed provider attempt grant is required'
                )
            committed_grant, state = active
            request = state.request
            if (
                self._db_clock() >= state.lease_expires_at
                or getattr(invocation, 'prepared_content_hmac', None)
                != request.prepared_content_hmac
                or getattr(invocation, 'character_count', None)
                != request.serialized_character_count
                or getattr(invocation, 'framed_input_tokens', None)
                != request.framed_input_tokens
                or getattr(invocation, 'max_output_tokens', None)
                != request.candidates[0].identity.max_output_tokens
            ):
                self._active_grants.pop(attempt_id, None)
                raise ValidationStoreError('prepared validation dispatch changed')
            try:
                committed_grant.permit.consume_at_dispatch()
            finally:
                self._active_grants.pop(attempt_id, None)
            state.provider_dispatched = True
        return provider(
            invocation,
            timeout=committed_grant.provider_timeout_seconds,
            http_hook=_SERVER_OWNED_FENCED_SEND_HOOK,
        )

    def complete(
        self,
        context: ValidationLockedContext,
        completion: ValidationBatchCompletion,
    ) -> tuple[CompletedValidationProjection, ...]:
        with self._lock:
            state = self._locked_state(context)
            if (
                state.status != 'claimed'
                or state.provider_attempt_count != 1
                or not state.provider_dispatched
                or self._db_clock() >= state.lease_expires_at
                or (
                    completion.validation_call_id,
                    completion.workflow_thread_id,
                    completion.batch_fingerprint,
                    completion.lease_token,
                )
                != (
                    context.validation_call_id,
                    context.workflow_thread_id,
                    context.batch_fingerprint,
                    context.lease_token,
                )
            ):
                raise ValidationStoreError('validation call is not completable')
            ordered_claims = tuple(
                sorted(
                    state.request.candidates,
                    key=lambda candidate: candidate.validation_key,
                )
            )
            ordered_completions = tuple(
                sorted(
                    completion.candidates,
                    key=lambda candidate: candidate.validation_key,
                )
            )
            if tuple(
                (item.review_item_id, item.validation_key)
                for item in ordered_completions
            ) != tuple(
                (item.review_item_id, item.validation_key)
                for item in ordered_claims
            ):
                raise ValidationStoreError('validation completion set changed')
            for index, item in enumerate(ordered_completions, start=1):
                if item.result.candidate_slot_id != f'C{index:02d}':
                    raise ValidationStoreError(
                        'validation completion slot changed'
                    )
            input_allocations = _allocate_integer_tokens(
                completion.input_tokens, len(ordered_completions)
            )
            output_allocations = _allocate_integer_tokens(
                completion.output_tokens, len(ordered_completions)
            )
            identity = ordered_claims[0].identity
            call_cost = _token_cost(
                completion.input_tokens,
                completion.output_tokens,
                input_price=identity.input_cost_per_1m_tokens,
                output_price=identity.output_cost_per_1m_tokens,
            )
            validation_obligation = call_cost + sum(
                (
                    other.charged_cost_usd
                    if other.status in {'completed', 'failed'}
                    else other.request.reserved_cost_usd
                )
                for other in self._calls.values()
                if other is not state
                and other.request.workflow_thread_id
                == state.request.workflow_thread_id
            )
            overrun = (
                completion.input_tokens > identity.max_input_tokens
                or completion.output_tokens > identity.max_output_tokens
                or call_cost > state.request.reserved_cost_usd
                or validation_obligation
                > min(
                    AUTO_REVIEW_MAX_VALIDATION_COST_USD,
                    identity.confirmed_validation_cost_ceiling_usd,
                )
                or identity.confirmed_extraction_cost_ceiling_usd
                + validation_obligation
                > identity.confirmed_total_cost_ceiling_usd
            )
            if overrun:
                state.status = 'failed'
                state.failure_reason_code = 'budget_exceeded'
                state.charged_input_tokens = completion.input_tokens
                state.charged_output_tokens = completion.output_tokens
                state.charged_cost_usd = call_cost
                state.budget_overrun = True
                state.budget_overrun_cost_usd = max(
                    call_cost - state.request.reserved_cost_usd,
                    Decimal('0'),
                )
                state.terminal_at = self._db_clock()
                if not self._validation_breaker_open:
                    self._validation_breaker_open = True
                    self._provider_safety_event_count += 1
                return ()
            if state.cancelled:
                state.status = 'failed'
                state.failure_reason_code = 'post_validation_drift'
                state.charged_input_tokens = completion.input_tokens
                state.charged_output_tokens = completion.output_tokens
                state.charged_cost_usd = call_cost
                state.terminal_at = self._db_clock()
                return ()
            costs = _allocate_micro_usd(
                input_allocations,
                output_allocations,
                input_price=identity.input_cost_per_1m_tokens,
                output_price=identity.output_cost_per_1m_tokens,
                total_cost=call_cost,
                tie_breakers=tuple(item.validation_key for item in ordered_claims),
            )
            projections: list[CompletedValidationProjection] = []
            for index, item in enumerate(ordered_completions):
                projections.append(
                    CompletedValidationProjection(
                        validation_call_id=state.validation_call_id,
                        validation_id=self._next_validation_id,
                        review_item_id=item.review_item_id,
                        result=item.result,
                        policy_decision=item.policy_decision,
                        policy_reason_codes=item.policy_reason_codes,
                        input_tokens=input_allocations[index],
                        output_tokens=output_allocations[index],
                        estimated_cost_usd=costs[index],
                        cache_hit=False,
                    )
                )
                self._next_validation_id += 1
            state.completed = tuple(projections)
            state.status = 'completed'
            state.charged_input_tokens = completion.input_tokens
            state.charged_output_tokens = completion.output_tokens
            state.charged_cost_usd = call_cost
            state.terminal_at = self._db_clock()
            return state.completed

    def snapshot(self, context: ValidationLockedContext) -> ValidationCallSnapshot:
        with self._lock:
            state = self._locked_state(context)
            return ValidationCallSnapshot(
                status=state.status,
                provider_attempt_count=state.provider_attempt_count,
                charged_input_tokens=state.charged_input_tokens,
                charged_output_tokens=state.charged_output_tokens,
                charged_cost_usd=state.charged_cost_usd,
                budget_overrun=state.budget_overrun,
                budget_overrun_cost_usd=state.budget_overrun_cost_usd,
                terminal_at=state.terminal_at,
            )

    def cancel(self, context: ValidationLockedContext) -> None:
        with self._lock:
            state = self._locked_state(context)
            if state.status != 'claimed':
                return
            if state.provider_attempt_count == 0:
                state.status = 'failed'
                state.failure_reason_code = 'post_validation_drift'
                state.charged_input_tokens = 0
                state.charged_output_tokens = 0
                state.charged_cost_usd = Decimal('0')
                state.terminal_at = self._db_clock()
            else:
                state.cancelled = True

    def recover_expired(self, context: ValidationLockedContext) -> None:
        with self._lock:
            state = self._locked_state(context)
            if (
                state.status != 'claimed'
                or state.provider_attempt_count != 1
                or self._db_clock() < state.lease_expires_at
            ):
                raise ValidationStoreError(
                    'expired validation attempt is not recoverable'
                )
            for attempt_id, active in tuple(self._active_grants.items()):
                if active[1] is state:
                    active[0].permit.invalidate_from_store()
                    self._active_grants.pop(attempt_id, None)
            state.status = 'failed'
            state.failure_reason_code = 'validator_unavailable'
            state.charged_input_tokens = state.request.reserved_input_tokens
            state.charged_output_tokens = state.request.reserved_output_tokens
            state.charged_cost_usd = state.request.reserved_cost_usd
            state.terminal_at = self._db_clock()

    def fail(
        self,
        context: ValidationLockedContext,
        failure: ValidationBatchFailure,
    ) -> None:
        with self._lock:
            state = self._locked_state(context)
            if (
                state.status != 'claimed'
                or state.provider_attempt_count != 1
                or (
                    failure.validation_call_id,
                    failure.workflow_thread_id,
                    failure.batch_fingerprint,
                    failure.lease_token,
                )
                != (
                    context.validation_call_id,
                    context.workflow_thread_id,
                    context.batch_fingerprint,
                    context.lease_token,
                )
            ):
                raise ValidationStoreError('validation call is not fail-able')
            if failure.usage_known:
                if failure.input_tokens is None or failure.output_tokens is None:
                    raise ValidationStoreError('validation failure usage is ambiguous')
                identity = state.request.candidates[0].identity
                charged_input = failure.input_tokens
                charged_output = failure.output_tokens
                charged_cost = _token_cost(
                    charged_input,
                    charged_output,
                    input_price=identity.input_cost_per_1m_tokens,
                    output_price=identity.output_cost_per_1m_tokens,
                )
            else:
                charged_input = state.request.reserved_input_tokens
                charged_output = state.request.reserved_output_tokens
                charged_cost = state.request.reserved_cost_usd
            state.status = 'failed'
            state.failure_reason_code = failure.reason_code
            state.charged_input_tokens = charged_input
            state.charged_output_tokens = charged_output
            state.charged_cost_usd = charged_cost
            state.terminal_at = self._db_clock()

    def _locked_state(self, context: ValidationLockedContext) -> _CallState:
        state = self._calls.get(
            (context.workflow_thread_id, context.batch_fingerprint)
        )
        if (
            state is None
            or state.validation_call_id != context.validation_call_id
            or state.lease_token != context.lease_token
        ):
            raise ValidationStoreError('locked validation context is required')
        return state

    def _new_claim(
        self,
        request: ValidationBatchClaimRequest,
        *,
        now: datetime,
    ) -> _CallState:
        state = _CallState(
            request=request,
            validation_call_id=self._next_call_id,
            lease_token=self._next_lease_token(),
            lease_expires_at=now + timedelta(seconds=self._lease_seconds(request)),
        )
        self._next_call_id += 1
        return state

    def _next_lease_token(self) -> str:
        token = self._lease_token_factory()
        if len(token) != 64:
            raise ValidationStoreError('validation lease token is unavailable')
        return token

    @staticmethod
    def _lease_seconds(request: ValidationBatchClaimRequest) -> int:
        values = {
            candidate.identity.provider_attempt_lease_seconds
            for candidate in request.candidates
        }
        if len(values) != 1:
            raise ValidationStoreError('validation timing identity changed')
        return next(iter(values))

    @staticmethod
    def _claimed_result(state: _CallState) -> ValidationClaimResult:
        return ValidationClaimResult(
            disposition='claimed',
            validation_call_id=state.validation_call_id,
            lease_token=state.lease_token,
            lease_expires_at=state.lease_expires_at,
            terminal_status=None,
            failure_reason_code=None,
            completed=(),
        )

    def _validate_request(self, request: ValidationBatchClaimRequest) -> None:
        if not 1 <= len(request.candidates) <= 4:
            raise ValidationStoreError('validation candidate count is invalid')
        identities = tuple(candidate.identity for candidate in request.candidates)
        first = identities[0]
        validation_keys = tuple(
            candidate.validation_key for candidate in request.candidates
        )
        if (
            len(set(validation_keys)) != len(validation_keys)
            or any(
                derive_validation_key(candidate.identity, settings=self._settings)
                != candidate.validation_key
                for candidate in request.candidates
            )
        ):
            raise ValidationStoreError('validation key identity changed')
        variable_fields = {
            'candidate_key',
            'evidence_version_hash',
            'normalized_claim_fingerprint',
            'candidate_generation_fingerprint',
        }
        shared_fields = tuple(
            field.name
            for field in fields(AutoReviewValidationIdentity)
            if field.name not in variable_fields
        )
        if any(
            tuple(getattr(identity, name) for name in shared_fields)
            != tuple(getattr(first, name) for name in shared_fields)
            for identity in identities[1:]
        ):
            raise ValidationStoreError('validation batch identity changed')
        exact_registry = (
            first.validator_provider,
            first.validator_model,
            first.reasoning_effort,
            first.validator_prompt_version,
            first.validator_output_contract_version,
            first.token_estimator_version,
            first.tokenizer_encoding,
            first.max_input_tokens,
            first.max_output_tokens,
            first.max_candidates_per_batch,
            first.max_provider_attempts,
            first.input_cost_per_1m_tokens,
            first.output_cost_per_1m_tokens,
            first.cost_policy_version,
            first.fingerprint_key_version,
        )
        expected_registry = (
            AUTO_REVIEW_VALIDATOR_PROVIDER,
            AUTO_REVIEW_VALIDATOR_MODEL,
            AUTO_REVIEW_REASONING_EFFORT,
            AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
            AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
            AUTO_REVIEW_TOKEN_ESTIMATOR_VERSION,
            AUTO_REVIEW_TOKENIZER_ENCODING,
            AUTO_REVIEW_MAX_INPUT_TOKENS,
            AUTO_REVIEW_MAX_OUTPUT_TOKENS,
            AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH,
            AUTO_REVIEW_MAX_PROVIDER_ATTEMPTS,
            AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
            AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
            AUTO_REVIEW_COST_POLICY_VERSION,
            self._settings.agent_runtime_fingerprint_key_version,
        )
        if exact_registry != expected_registry:
            raise ValidationStoreError('validation registry identity changed')
        expected_reserve = _token_cost(
            request.reserved_input_tokens,
            request.reserved_output_tokens,
            input_price=first.input_cost_per_1m_tokens,
            output_price=first.output_cost_per_1m_tokens,
        )
        if (
            request.max_provider_attempts != 1
            or len(request.prepared_content_hmac) != 64
            or not 0 < request.serialized_character_count <= AUTO_REVIEW_MAX_INPUT_CHARS
            or not 0 < request.framed_input_tokens <= first.max_input_tokens
            or request.reserved_input_tokens != request.framed_input_tokens
            or request.reserved_output_tokens != first.max_output_tokens
            or request.reserved_cost_usd != expected_reserve
            or request.reserved_cost_usd > AUTO_REVIEW_MAX_BATCH_COST_USD
            or request.reserved_cost_usd
            > first.confirmed_validation_cost_ceiling_usd
            or first.confirmed_extraction_cost_ceiling_usd
            + request.reserved_cost_usd
            > first.confirmed_total_cost_ceiling_usd
        ):
            raise ValidationStoreError('validation reservation is invalid')
        expected = derive_validation_batch_fingerprint(
            workflow_thread_id=request.workflow_thread_id,
            candidates=request.candidates,
            prepared_content_hmac=request.prepared_content_hmac,
            serialized_character_count=request.serialized_character_count,
            framed_input_tokens=request.framed_input_tokens,
            reserved_input_tokens=request.reserved_input_tokens,
            reserved_output_tokens=request.reserved_output_tokens,
            reserved_cost_usd=request.reserved_cost_usd,
            settings=self._settings,
        )
        if expected != request.batch_fingerprint:
            raise ValidationStoreError('validation batch identity changed')

    @staticmethod
    def _validate_admission(
        state: _CallState,
        admission: ValidationAttemptAdmission,
    ) -> None:
        request = state.request
        identity = request.candidates[0].identity
        expected = (
            request.prepared_content_hmac,
            request.serialized_character_count,
            request.framed_input_tokens,
            identity.max_output_tokens,
            request.reserved_cost_usd,
            identity.fingerprint_key_version,
            identity.fingerprint_key_material_verifier,
            identity.token_estimator_version,
            identity.cost_policy_version,
            identity.provider_timeout_seconds,
            identity.provider_send_start_window_seconds,
            identity.provider_attempt_lease_seconds,
            identity.provider_commit_grace_seconds,
        )
        actual = (
            admission.prepared_content_hmac,
            admission.serialized_character_count,
            admission.framed_input_tokens,
            admission.max_output_tokens,
            admission.recomputed_reserved_cost_usd,
            admission.fingerprint_key_version,
            admission.fingerprint_key_material_verifier,
            admission.token_estimator_version,
            admission.cost_policy_version,
            admission.provider_timeout_seconds,
            admission.provider_send_start_window_seconds,
            admission.provider_attempt_lease_seconds,
            admission.provider_commit_grace_seconds,
        )
        if actual != expected:
            raise ValidationStoreError('validation admission identity changed')


class AutoReviewValidationStore:
    """PostgreSQL-authoritative validation call and child-result ledger.

    The store deliberately keeps the prepared invocation and provider grant out
    of SQL. Only bounded identities, token/cost accounting, allowlisted result
    fields, and terminal status are persisted.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        settings: Settings,
        db_clock: Callable[[Session], datetime] | None = None,
        lease_token_factory: Callable[[], str] | None = None,
        permit_monotonic: Callable[[], float] = monotonic,
        current_permission_resolver: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._db_clock_override = db_clock
        self._lease_token_factory = lease_token_factory or (
            lambda: secrets.token_hex(32)
        )
        self._permit_monotonic = permit_monotonic
        self._current_permission_resolver = current_permission_resolver
        self._active_grants: dict[str, tuple[_CommittedGrant, ValidationLockedContext]] = {}
        self._dispatched_calls: set[int] = set()
        self._attempt_guards: dict[int, tuple[str, str]] = {}
        self._grant_lock = RLock()

    @contextmanager
    def _locked_transaction(
        self,
    ) -> Iterator[tuple[Session, AutoReviewRuntimeKeyState | None]]:
        with (
            self._session_factory() as db,
            db.begin(),
            KeyedMutationGuard.generation_barrier(db),
        ):
            runtime = KeyedMutationGuard.lock_runtime_key_state(
                db, for_update=False
            )
            yield db, runtime

    @staticmethod
    def _runtime_matches(
        runtime: AutoReviewRuntimeKeyState | None,
        *,
        key_version: str,
        material_verifier: str,
    ) -> bool:
        return bool(
            runtime is not None
            and runtime.ready
            and runtime.fingerprint_key_version == key_version
            and runtime.fingerprint_key_material_verifier == material_verifier
        )

    def claim_or_replay(
        self,
        request: ValidationBatchClaimRequest,
    ) -> ValidationClaimResult:
        self._validate_request(request)
        try:
            with self._locked_transaction() as (db, runtime):
                now = self._db_now(db)
                first = request.candidates[0].identity
                if not self._runtime_matches(
                    runtime,
                    key_version=first.fingerprint_key_version,
                    material_verifier=first.fingerprint_key_material_verifier,
                ):
                    raise ValidationStoreError(
                        'validation runtime key identity changed'
                    )
                safety = self._lock_validation_safety(db, allow_open=True)
                rollout_matches = self._lock_rollout_authority(
                    db,
                    workflow_thread_id=request.workflow_thread_id,
                )
                source_states = self._lock_source_states(
                    db,
                    workflow_thread_id=request.workflow_thread_id,
                    allow_drift=True,
                )
                workflow_request = self._locked_workflow_request(
                    db, request=request
                )
                call = db.scalar(
                    select(AutoReviewValidationCall)
                    .where(
                        AutoReviewValidationCall.workflow_thread_id
                        == request.workflow_thread_id,
                        AutoReviewValidationCall.batch_fingerprint
                        == request.batch_fingerprint,
                    )
                    .with_for_update()
                )
                if call is not None:
                    self._verify_call_request(db, call=call, request=request)
                    if call.status == 'completed':
                        completed = self._completed_projections(db, call=call)
                        if not completed:
                            raise ValidationStoreError(
                                'completed validation projection is unavailable'
                            )
                        return ValidationClaimResult(
                            disposition='replayed',
                            validation_call_id=call.id,
                            lease_token=None,
                            lease_expires_at=None,
                            terminal_status='completed',
                            failure_reason_code=None,
                            completed=completed,
                        )
                    if call.status == 'failed':
                        reason = self._failed_reason(db, call=call)
                        return ValidationClaimResult(
                            disposition='replayed',
                            validation_call_id=call.id,
                            lease_token=None,
                            lease_expires_at=None,
                            terminal_status='failed',
                            failure_reason_code=reason,
                            completed=(),
                        )
                    if (
                        not source_states
                        or not rollout_matches
                        or safety.breaker_open
                        or safety.state_version
                        != call.provider_safety_state_version
                    ):
                        raise ValidationStoreError(
                            'validation provider safety identity changed'
                        )
                    if (
                        call.lease_expires_at is None
                        or now < call.lease_expires_at
                        or call.provider_attempt_count != 0
                    ):
                        return ValidationClaimResult(
                            disposition='busy',
                            validation_call_id=call.id,
                            lease_token=None,
                            lease_expires_at=call.lease_expires_at,
                            terminal_status=None,
                            failure_reason_code=None,
                            completed=(),
                        )
                    call.lease_token = self._next_lease_token()
                    call.lease_expires_at = now + timedelta(
                        seconds=request.candidates[0].identity.provider_attempt_lease_seconds
                    )
                    db.flush()
                    return self._claimed_result(call)

                if (
                    not source_states
                    or not rollout_matches
                    or safety.breaker_open
                    or safety.state_version != first.provider_safety_state_version
                ):
                    raise ValidationStoreError(
                        'validation provider safety identity changed'
                    )
                self._require_workflow_budget(
                    db,
                    workflow_request=workflow_request,
                    request=request,
                )
                call = AutoReviewValidationCall(
                    workflow_thread_id=request.workflow_thread_id,
                    batch_fingerprint=request.batch_fingerprint,
                    status='claimed',
                    candidate_count=len(request.candidates),
                    max_provider_attempts=1,
                    provider_attempt_count=0,
                    reserved_input_tokens=request.reserved_input_tokens,
                    reserved_output_tokens=request.reserved_output_tokens,
                    reserved_cost_usd=request.reserved_cost_usd,
                    charged_cost_usd=Decimal('0'),
                    prepared_content_hmac=request.prepared_content_hmac,
                    serialized_character_count=request.serialized_character_count,
                    framed_input_token_count=request.framed_input_tokens,
                    max_output_tokens=first.max_output_tokens,
                    token_estimator_version=first.token_estimator_version,
                    tokenizer_encoding=first.tokenizer_encoding,
                    reply_priming_tokens=AUTO_REVIEW_REPLY_PRIMING_TOKENS,
                    framing_safety_tokens=AUTO_REVIEW_FRAMING_SAFETY_TOKENS,
                    input_usd_per_1m=first.input_cost_per_1m_tokens,
                    output_usd_per_1m=first.output_cost_per_1m_tokens,
                    cost_policy_version=first.cost_policy_version,
                    provider_safety_state_version=first.provider_safety_state_version,
                    fingerprint_key_version=first.fingerprint_key_version,
                    fingerprint_key_material_verifier=(
                        first.fingerprint_key_material_verifier
                    ),
                    workflow_extraction_cost_ceiling_usd=(
                        first.confirmed_extraction_cost_ceiling_usd
                    ),
                    workflow_validation_cost_ceiling_usd=(
                        first.confirmed_validation_cost_ceiling_usd
                    ),
                    workflow_total_cost_ceiling_usd=(
                        first.confirmed_total_cost_ceiling_usd
                    ),
                    provider_timeout_seconds=first.provider_timeout_seconds,
                    provider_send_start_window_seconds=(
                        first.provider_send_start_window_seconds
                    ),
                    provider_attempt_lease_seconds=(
                        first.provider_attempt_lease_seconds
                    ),
                    provider_commit_grace_seconds=first.provider_commit_grace_seconds,
                    lease_token=self._next_lease_token(),
                    lease_expires_at=now
                    + timedelta(seconds=first.provider_attempt_lease_seconds),
                    budget_overrun=False,
                    budget_overrun_cost_usd=Decimal('0'),
                    claimed_at=now,
                )
                db.add(call)
                db.flush()
                for candidate in sorted(
                    request.candidates, key=lambda item: item.validation_key
                ):
                    identity = candidate.identity
                    db.add(
                        AutoReviewValidation(
                            review_item_id=candidate.review_item_id,
                            validation_call_id=call.id,
                            workflow_thread_id=request.workflow_thread_id,
                            validation_key=candidate.validation_key,
                            evidence_version_hash=identity.evidence_version_hash,
                            candidate_generation_fingerprint=(
                                identity.candidate_generation_fingerprint
                            ),
                            status='claimed',
                            validator_provider=identity.validator_provider,
                            validator_model=identity.validator_model,
                            reasoning_effort=identity.reasoning_effort,
                            validator_prompt_version=(
                                identity.validator_prompt_version
                            ),
                            validator_output_contract_version=(
                                identity.validator_output_contract_version
                            ),
                            policy_version=identity.policy_version,
                            fingerprint_key_version=identity.fingerprint_key_version,
                            fingerprint_key_material_verifier=(
                                identity.fingerprint_key_material_verifier
                            ),
                            cost_policy_version=identity.cost_policy_version,
                            confirmed_validation_cost_ceiling_usd=(
                                identity.confirmed_validation_cost_ceiling_usd
                            ),
                            claim_results=[],
                            uncertainty_codes=[],
                            conflict_codes=[],
                            policy_reason_codes=[],
                            input_tokens=0,
                            output_tokens=0,
                            estimated_cost_usd=Decimal('0'),
                            cache_hit=False,
                            created_at=now,
                        )
                    )
                db.flush()
                return self._claimed_result(call)
        except IntegrityError:
            return self._replay_after_claim_race(request)

    def mark_attempt_started(
        self,
        admission: ValidationAttemptAdmission,
    ) -> _CommittedGrant:
        context = ValidationLockedContext(
            validation_call_id=admission.validation_call_id,
            workflow_thread_id=admission.workflow_thread_id,
            batch_fingerprint=admission.batch_fingerprint,
            lease_token=admission.lease_token,
        )
        refusal: str | None = None
        with self._locked_transaction() as (db, runtime):
            now = self._db_now(db)
            safety = self._lock_validation_safety(
                db,
                allow_open=True,
                require_current_registry=False,
            )
            runtime_matches = self._runtime_matches(
                runtime,
                key_version=admission.fingerprint_key_version,
                material_verifier=admission.fingerprint_key_material_verifier,
            )
            safety_matches = self._provider_safety_matches(
                safety,
                allow_open=False,
            )
            rollout_matches = self._lock_rollout_authority(
                db,
                workflow_thread_id=context.workflow_thread_id,
                expected_mode=admission.expected_effective_mode,
            )
            source_states = self._lock_source_states(
                db,
                workflow_thread_id=context.workflow_thread_id,
                allow_drift=True,
            )
            self._locked_workflow_request_for_context(db, context=context)
            call = self._locked_call(db, context)
            if (
                call.status != 'claimed'
                or call.provider_attempt_count != 0
                or call.lease_expires_at is None
                or now >= call.lease_expires_at
            ):
                raise ValidationStoreError('validation attempt is not claimable')
            self._validate_sql_admission(call, admission)
            try:
                current_guards = self._current_guard_hmacs(
                    db,
                    context=context,
                    source_states=source_states,
                )
            except ValidationStoreError:
                current_guards = ('', '')
            expected_guards = (
                admission.expected_owner_permission_hmac,
                admission.expected_source_state_hmac,
            )
            if (
                not runtime_matches
                or not safety_matches
                or not rollout_matches
                or safety.state_version != call.provider_safety_state_version
                or current_guards != expected_guards
            ):
                children = self._locked_children(db, call=call)
                self._terminal_failure(
                    call=call,
                    children=children,
                    reason='post_validation_drift',
                    input_tokens=0,
                    output_tokens=0,
                    charged_cost=Decimal('0'),
                    now=now,
                )
                if not runtime_matches:
                    refusal = 'validation runtime key identity changed'
                elif (
                    not safety_matches
                    or safety.state_version != call.provider_safety_state_version
                ):
                    refusal = 'validation provider safety identity changed'
                elif not rollout_matches:
                    refusal = 'validation rollout authority changed'
                else:
                    refusal = 'validation execution guard changed'
                attempt_id = ''
                timeout = 0
                send_window = 0
                lease_expires_at = now
            else:
                call.provider_attempt_count = 1
                call.attempt_started_at = now
                call.lease_expires_at = now + timedelta(
                    seconds=call.provider_attempt_lease_seconds
                )
                attempt_id = sha256(
                    f'{call.lease_token}:attempt:1'.encode()
                ).hexdigest()
                timeout = call.provider_timeout_seconds
                send_window = call.provider_send_start_window_seconds
                lease_expires_at = call.lease_expires_at
        if refusal is not None:
            raise ValidationStoreError(refusal)
        permit = _CommittedPermit(
            attempt_id=attempt_id,
            deadline=self._permit_monotonic() + send_window,
            clock=self._permit_monotonic,
        )
        grant = _CommittedGrant(
            attempt_id=attempt_id,
            permit=permit,
            provider_timeout_seconds=timeout,
            authoritative_lease_expires_at=lease_expires_at,
        )
        with self._grant_lock:
            self._active_grants[attempt_id] = (grant, context)
            self._attempt_guards[context.validation_call_id] = current_guards
        return grant

    def dispatch_prepared_validation(
        self,
        *,
        invocation: Any,
        provider: Callable[..., Any],
        grant: object,
    ) -> Any:
        attempt_id = getattr(grant, 'attempt_id', None)
        with self._grant_lock:
            active = self._active_grants.get(attempt_id)
            if active is None or active[0] is not grant:
                raise ValidationStoreError(
                    'committed provider attempt grant is required'
                )
            committed_grant, context = active
        with self._locked_transaction() as (db, runtime):
            safety = self._lock_validation_safety(
                db,
                allow_open=True,
                require_current_registry=False,
            )
            rollout_matches = self._lock_rollout_authority(
                db,
                workflow_thread_id=context.workflow_thread_id,
            )
            source_states = self._lock_source_states(
                db,
                workflow_thread_id=context.workflow_thread_id,
                allow_drift=True,
            )
            self._locked_workflow_request_for_context(db, context=context)
            call = self._locked_call(db, context, for_update=False)
            now = self._db_now(db)
            current_guards = self._current_guard_hmacs(
                db,
                context=context,
                source_states=source_states,
            )
            with self._grant_lock:
                expected_guards = self._attempt_guards.get(
                    context.validation_call_id
                )
            valid = (
                call.status == 'claimed'
                and call.provider_attempt_count == 1
                and call.lease_expires_at is not None
                and now < call.lease_expires_at
                and getattr(invocation, 'prepared_content_hmac', None)
                == call.prepared_content_hmac
                and getattr(invocation, 'character_count', None)
                == call.serialized_character_count
                and getattr(invocation, 'framed_input_tokens', None)
                == call.framed_input_token_count
                and getattr(invocation, 'max_output_tokens', None)
                == call.max_output_tokens
                and expected_guards is not None
                and current_guards == expected_guards
                and self._runtime_matches(
                    runtime,
                    key_version=call.fingerprint_key_version,
                    material_verifier=(
                        call.fingerprint_key_material_verifier
                    ),
                )
                and self._provider_safety_matches(safety, allow_open=False)
                and rollout_matches
                and safety.state_version == call.provider_safety_state_version
            )
        if not valid:
            with self._grant_lock:
                self._active_grants.pop(attempt_id, None)
            committed_grant.permit.invalidate_from_store()
            raise ValidationStoreError('prepared validation dispatch changed')
        try:
            committed_grant.permit.consume_at_dispatch()
        finally:
            with self._grant_lock:
                self._active_grants.pop(attempt_id, None)
        with self._grant_lock:
            self._dispatched_calls.add(context.validation_call_id)
        return provider(
            invocation,
            timeout=committed_grant.provider_timeout_seconds,
            http_hook=_SERVER_OWNED_FENCED_SEND_HOOK,
        )

    def complete(
        self,
        context: ValidationLockedContext,
        completion: ValidationBatchCompletion,
    ) -> tuple[CompletedValidationProjection, ...]:
        self._require_matching_completion(context, completion)
        with self._grant_lock:
            if context.validation_call_id not in self._dispatched_calls:
                raise ValidationStoreError('validation provider was not dispatched')
        with self._locked_transaction() as (db, runtime):
            now = self._db_now(db)
            safety = self._lock_validation_safety(
                db,
                allow_open=True,
                require_current_registry=False,
            )
            rollout_matches = self._lock_rollout_authority(
                db,
                workflow_thread_id=context.workflow_thread_id,
            )
            source_states = self._lock_source_states(
                db,
                workflow_thread_id=context.workflow_thread_id,
                allow_drift=True,
            )
            workflow_request = self._locked_workflow_request_for_context(
                db, context=context, allow_cancelled=True
            )
            cancelled = self._workflow_is_cancelled(db, context=context)
            call = self._locked_call(db, context)
            with self._grant_lock:
                expected_guards = self._attempt_guards.get(
                    context.validation_call_id
                )
            try:
                current_guards = self._current_guard_hmacs(
                    db,
                    context=context,
                    source_states=source_states,
                )
            except ValidationStoreError:
                current_guards = ('', '')
            guard_matches = (
                expected_guards is not None
                and current_guards == expected_guards
                and self._runtime_matches(
                    runtime,
                    key_version=call.fingerprint_key_version,
                    material_verifier=(
                        call.fingerprint_key_material_verifier
                    ),
                )
                and self._provider_safety_matches(safety, allow_open=False)
                and rollout_matches
                and safety.state_version == call.provider_safety_state_version
            )
            children = self._locked_children(db, call=call)
            if (
                call.status != 'claimed'
                or call.provider_attempt_count != 1
                or call.lease_expires_at is None
                or now >= call.lease_expires_at
            ):
                raise ValidationStoreError('validation call is not completable')
            ordered = tuple(
                sorted(completion.candidates, key=lambda item: item.validation_key)
            )
            if tuple((row.review_item_id, row.validation_key) for row in children) != tuple(
                (item.review_item_id, item.validation_key) for item in ordered
            ):
                raise ValidationStoreError('validation completion identity changed')
            charge = _token_cost(
                completion.input_tokens,
                completion.output_tokens,
                input_price=Decimal(call.input_usd_per_1m),
                output_price=Decimal(call.output_usd_per_1m),
            )
            input_allocations = _allocate_integer_tokens(
                completion.input_tokens, len(children)
            )
            output_allocations = _allocate_integer_tokens(
                completion.output_tokens, len(children)
            )
            costs = _allocate_micro_usd(
                input_allocations,
                output_allocations,
                input_price=Decimal(call.input_usd_per_1m),
                output_price=Decimal(call.output_usd_per_1m),
                total_cost=charge,
                tie_breakers=tuple(row.validation_key for row in children),
            )
            extraction_obligation, validation_obligation = (
                self._workflow_obligation_after_charge(
                    db,
                    context=context,
                    charged_cost=charge,
                )
            )
            overrun = (
                completion.input_tokens > call.framed_input_token_count
                or completion.output_tokens > call.max_output_tokens
                or charge > Decimal(call.reserved_cost_usd)
                or validation_obligation
                > min(
                    AUTO_REVIEW_MAX_VALIDATION_COST_USD,
                    Decimal(workflow_request.confirmed_validation_cost_ceiling_usd),
                )
                or extraction_obligation + validation_obligation
                > min(
                    Decimal(workflow_request.confirmed_total_cost_ceiling_usd),
                    Decimal(workflow_request.auto_review_budget_limit_usd),
                )
            )
            if overrun:
                self._terminal_failure(
                    call=call,
                    children=children,
                    reason='budget_exceeded',
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    charged_cost=charge,
                    now=now,
                )
                call.budget_overrun = True
                call.budget_overrun_cost_usd = max(
                    charge - Decimal(call.reserved_cost_usd), Decimal('0')
                )
                self._open_budget_breaker(
                    db,
                    safety=safety,
                    call=call,
                    charged_cost=charge,
                    now=now,
                )
                with self._grant_lock:
                    self._dispatched_calls.discard(context.validation_call_id)
                    self._attempt_guards.pop(context.validation_call_id, None)
                return ()
            if cancelled or not guard_matches:
                self._terminal_failure(
                    call=call,
                    children=children,
                    reason='post_validation_drift',
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    charged_cost=charge,
                    now=now,
                )
                with self._grant_lock:
                    self._dispatched_calls.discard(context.validation_call_id)
                    self._attempt_guards.pop(context.validation_call_id, None)
                return ()
            projections: list[CompletedValidationProjection] = []
            for index, (row, item) in enumerate(zip(children, ordered, strict=True)):
                payload = item.result.model_dump(mode='json')
                row.status = 'completed'
                row.claim_results = payload['claim_results']
                row.minimum_entailment_score = min(
                    Decimal(result['entailment_score'])
                    for result in payload['claim_results']
                )
                row.uncertainty_codes = payload['uncertainty_codes']
                row.conflict_codes = payload['conflict_codes']
                row.policy_decision = item.policy_decision
                row.policy_reason_codes = list(item.policy_reason_codes)
                row.input_tokens = input_allocations[index]
                row.output_tokens = output_allocations[index]
                row.estimated_cost_usd = costs[index]
                row.cache_hit = False
                row.completed_at = now
                projections.append(
                    self._projection(row, slot_index=index, cache_hit=False)
                )
            call.status = 'completed'
            call.charged_input_tokens = completion.input_tokens
            call.charged_output_tokens = completion.output_tokens
            call.charged_cost_usd = charge
            call.terminal_at = now
        with self._grant_lock:
            self._dispatched_calls.discard(context.validation_call_id)
            self._attempt_guards.pop(context.validation_call_id, None)
        return tuple(projections)

    def fail(
        self,
        context: ValidationLockedContext,
        failure: ValidationBatchFailure,
    ) -> None:
        self._require_matching_failure(context, failure)
        with self._locked_transaction() as (db, _runtime):
            safety = self._lock_validation_safety(db, allow_open=True)
            self._lock_rollout_authority(
                db,
                workflow_thread_id=context.workflow_thread_id,
            )
            self._lock_source_states(
                db,
                workflow_thread_id=context.workflow_thread_id,
                allow_drift=True,
            )
            self._locked_workflow_request_for_context(
                db, context=context, allow_cancelled=True
            )
            call = self._locked_call(db, context)
            children = self._locked_children(db, call=call)
            if call.status != 'claimed' or call.provider_attempt_count != 1:
                raise ValidationStoreError('validation call is not fail-able')
            if failure.usage_known:
                if failure.input_tokens is None or failure.output_tokens is None:
                    raise ValidationStoreError('validation failure usage is ambiguous')
                input_tokens = failure.input_tokens
                output_tokens = failure.output_tokens
                charge = _token_cost(
                    input_tokens,
                    output_tokens,
                    input_price=Decimal(call.input_usd_per_1m),
                    output_price=Decimal(call.output_usd_per_1m),
                )
            else:
                input_tokens = call.reserved_input_tokens
                output_tokens = call.reserved_output_tokens
                charge = Decimal(call.reserved_cost_usd)
            self._terminal_failure(
                call=call,
                children=children,
                reason=failure.reason_code,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                charged_cost=charge,
                now=self._db_now(db),
            )
            if (
                input_tokens > call.framed_input_token_count
                or output_tokens > call.max_output_tokens
                or charge > Decimal(call.reserved_cost_usd)
            ):
                call.budget_overrun = True
                call.budget_overrun_cost_usd = max(
                    charge - Decimal(call.reserved_cost_usd), Decimal('0')
                )
                self._open_budget_breaker(
                    db,
                    safety=safety,
                    call=call,
                    charged_cost=charge,
                    now=call.terminal_at,
                )
        with self._grant_lock:
            self._dispatched_calls.discard(context.validation_call_id)
            self._attempt_guards.pop(context.validation_call_id, None)

    def cancel(self, context: ValidationLockedContext) -> None:
        with self._locked_transaction() as (db, _runtime):
            self._lock_validation_safety(db, allow_open=True)
            self._lock_rollout_authority(
                db,
                workflow_thread_id=context.workflow_thread_id,
            )
            self._lock_source_states(
                db,
                workflow_thread_id=context.workflow_thread_id,
                allow_drift=True,
            )
            self._locked_workflow_request_for_context(
                db, context=context, allow_cancelled=True
            )
            call = self._locked_call(db, context)
            children = self._locked_children(db, call=call)
            if call.status != 'claimed' or call.provider_attempt_count != 0:
                return
            self._terminal_failure(
                call=call,
                children=children,
                reason='post_validation_drift',
                input_tokens=0,
                output_tokens=0,
                charged_cost=Decimal('0'),
                now=self._db_now(db),
            )

    def recover_expired(self, context: ValidationLockedContext) -> None:
        with self._locked_transaction() as (db, _runtime):
            self._lock_validation_safety(db, allow_open=True)
            self._lock_rollout_authority(
                db,
                workflow_thread_id=context.workflow_thread_id,
            )
            self._lock_source_states(
                db,
                workflow_thread_id=context.workflow_thread_id,
                allow_drift=True,
            )
            self._locked_workflow_request_for_context(
                db, context=context, allow_cancelled=True
            )
            call = self._locked_call(db, context)
            children = self._locked_children(db, call=call)
            if (
                call.status != 'claimed'
                or call.provider_attempt_count != 1
                or call.lease_expires_at is None
                or self._db_now(db) < call.lease_expires_at
            ):
                raise ValidationStoreError(
                    'expired validation attempt is not recoverable'
                )
            self._terminal_failure(
                call=call,
                children=children,
                reason='validator_unavailable',
                input_tokens=call.reserved_input_tokens,
                output_tokens=call.reserved_output_tokens,
                charged_cost=Decimal(call.reserved_cost_usd),
                now=self._db_now(db),
            )
        with self._grant_lock:
            for attempt_id, (grant, active_context) in tuple(
                self._active_grants.items()
            ):
                if active_context == context:
                    grant.permit.invalidate_from_store()
                    self._active_grants.pop(attempt_id, None)
            self._dispatched_calls.discard(context.validation_call_id)
            self._attempt_guards.pop(context.validation_call_id, None)

    def _validate_request(self, request: ValidationBatchClaimRequest) -> None:
        validator = ValidationCallLedger(
            settings=self._settings,
            db_clock=lambda: datetime.min,
        )
        validator._validate_request(request)

    def _replay_after_claim_race(
        self, request: ValidationBatchClaimRequest
    ) -> ValidationClaimResult:
        with self._session_factory() as db:
            call = db.scalar(
                select(AutoReviewValidationCall).where(
                    AutoReviewValidationCall.workflow_thread_id
                    == request.workflow_thread_id,
                    AutoReviewValidationCall.batch_fingerprint
                    == request.batch_fingerprint,
                )
            )
            if call is None:
                raise ValidationStoreError('validation claim race lost without owner')
            self._verify_call_request(db, call=call, request=request)
            if call.status == 'completed':
                return ValidationClaimResult(
                    disposition='replayed',
                    validation_call_id=call.id,
                    lease_token=None,
                    lease_expires_at=None,
                    terminal_status='completed',
                    failure_reason_code=None,
                    completed=self._completed_projections(db, call=call),
                )
            if call.status == 'failed':
                return ValidationClaimResult(
                    disposition='replayed',
                    validation_call_id=call.id,
                    lease_token=None,
                    lease_expires_at=None,
                    terminal_status='failed',
                    failure_reason_code=self._failed_reason(db, call=call),
                    completed=(),
                )
            return ValidationClaimResult(
                disposition='busy',
                validation_call_id=call.id,
                lease_token=None,
                lease_expires_at=call.lease_expires_at,
                terminal_status=None,
                failure_reason_code=None,
                completed=(),
            )

    def _verify_call_request(
        self,
        db: Session,
        *,
        call: AutoReviewValidationCall,
        request: ValidationBatchClaimRequest,
    ) -> None:
        first = request.candidates[0].identity
        expected_call = (
            len(request.candidates),
            request.max_provider_attempts,
            request.prepared_content_hmac,
            request.serialized_character_count,
            request.framed_input_tokens,
            request.reserved_input_tokens,
            request.reserved_output_tokens,
            request.reserved_cost_usd,
            first.max_output_tokens,
            first.token_estimator_version,
            first.tokenizer_encoding,
            first.input_cost_per_1m_tokens,
            first.output_cost_per_1m_tokens,
            first.cost_policy_version,
            first.provider_safety_state_version,
            first.fingerprint_key_version,
            first.fingerprint_key_material_verifier,
            first.confirmed_extraction_cost_ceiling_usd,
            first.confirmed_validation_cost_ceiling_usd,
            first.confirmed_total_cost_ceiling_usd,
            first.provider_timeout_seconds,
            first.provider_send_start_window_seconds,
            first.provider_attempt_lease_seconds,
            first.provider_commit_grace_seconds,
        )
        actual_call = (
            call.candidate_count,
            call.max_provider_attempts,
            call.prepared_content_hmac,
            call.serialized_character_count,
            call.framed_input_token_count,
            call.reserved_input_tokens,
            call.reserved_output_tokens,
            Decimal(call.reserved_cost_usd),
            call.max_output_tokens,
            call.token_estimator_version,
            call.tokenizer_encoding,
            Decimal(call.input_usd_per_1m),
            Decimal(call.output_usd_per_1m),
            call.cost_policy_version,
            call.provider_safety_state_version,
            call.fingerprint_key_version,
            call.fingerprint_key_material_verifier,
            Decimal(call.workflow_extraction_cost_ceiling_usd),
            Decimal(call.workflow_validation_cost_ceiling_usd),
            Decimal(call.workflow_total_cost_ceiling_usd),
            call.provider_timeout_seconds,
            call.provider_send_start_window_seconds,
            call.provider_attempt_lease_seconds,
            call.provider_commit_grace_seconds,
        )
        if actual_call != expected_call:
            raise ValidationStoreError('validation claim identity changed')
        children = self._locked_children(db, call=call, for_update=False)
        expected_children = tuple(
            sorted(request.candidates, key=lambda item: item.validation_key)
        )
        actual_children = tuple(
            (
                row.review_item_id,
                row.validation_key,
                row.evidence_version_hash,
                row.candidate_generation_fingerprint,
                row.validator_provider,
                row.validator_model,
                row.reasoning_effort,
                row.validator_prompt_version,
                row.validator_output_contract_version,
                row.policy_version,
                row.fingerprint_key_version,
                row.fingerprint_key_material_verifier,
                row.cost_policy_version,
                Decimal(row.confirmed_validation_cost_ceiling_usd),
            )
            for row in children
        )
        expected_child_values = tuple(
            (
                item.review_item_id,
                item.validation_key,
                item.identity.evidence_version_hash,
                item.identity.candidate_generation_fingerprint,
                item.identity.validator_provider,
                item.identity.validator_model,
                item.identity.reasoning_effort,
                item.identity.validator_prompt_version,
                item.identity.validator_output_contract_version,
                item.identity.policy_version,
                item.identity.fingerprint_key_version,
                item.identity.fingerprint_key_material_verifier,
                item.identity.cost_policy_version,
                item.identity.confirmed_validation_cost_ceiling_usd,
            )
            for item in expected_children
        )
        if actual_children != expected_child_values:
            raise ValidationStoreError('validation child identity changed')

    def _validate_sql_admission(
        self,
        call: AutoReviewValidationCall,
        admission: ValidationAttemptAdmission,
    ) -> None:
        expected = (
            call.prepared_content_hmac,
            call.serialized_character_count,
            call.framed_input_token_count,
            call.max_output_tokens,
            Decimal(call.reserved_cost_usd),
            call.fingerprint_key_version,
            call.fingerprint_key_material_verifier,
            call.token_estimator_version,
            call.cost_policy_version,
            call.provider_timeout_seconds,
            call.provider_send_start_window_seconds,
            call.provider_attempt_lease_seconds,
            call.provider_commit_grace_seconds,
        )
        actual = (
            admission.prepared_content_hmac,
            admission.serialized_character_count,
            admission.framed_input_tokens,
            admission.max_output_tokens,
            admission.recomputed_reserved_cost_usd,
            admission.fingerprint_key_version,
            admission.fingerprint_key_material_verifier,
            admission.token_estimator_version,
            admission.cost_policy_version,
            admission.provider_timeout_seconds,
            admission.provider_send_start_window_seconds,
            admission.provider_attempt_lease_seconds,
            admission.provider_commit_grace_seconds,
        )
        if actual != expected:
            raise ValidationStoreError('validation admission identity changed')

    def _locked_workflow_request(
        self,
        db: Session,
        *,
        request: ValidationBatchClaimRequest,
    ) -> AgentWorkflowRequest:
        thread = db.scalar(
            select(AgentWorkflowThread)
            .where(
                AgentWorkflowThread.thread_id == request.workflow_thread_id
            )
            .with_for_update()
        )
        workflow_request = db.scalar(
            select(AgentWorkflowRequest)
            .where(
                AgentWorkflowRequest.workflow_thread_id
                == request.workflow_thread_id
            )
            .with_for_update()
        )
        if thread is None or thread.cancelled_at is not None:
            raise ValidationStoreError('validation workflow is unavailable')
        if workflow_request is None:
            raise ValidationStoreError('validation workflow request is unavailable')
        self._verify_workflow_request(
            workflow_request=workflow_request,
            identity=request.candidates[0].identity,
        )
        return workflow_request

    @staticmethod
    def _lock_validation_safety(
        db: Session,
        *,
        allow_open: bool,
        require_current_registry: bool = True,
    ) -> AutoReviewProviderSafetyState:
        state = db.scalar(
            select(AutoReviewProviderSafetyState)
            .where(
                AutoReviewProviderSafetyState.purpose == 'validation',
                AutoReviewProviderSafetyState.provider
                == AUTO_REVIEW_VALIDATOR_PROVIDER,
                AutoReviewProviderSafetyState.model == AUTO_REVIEW_VALIDATOR_MODEL,
                AutoReviewProviderSafetyState.reasoning_effort
                == AUTO_REVIEW_REASONING_EFFORT,
            )
            .with_for_update()
        )
        if state is None:
            raise ValidationStoreError('validation provider safety is unavailable')
        if not AutoReviewValidationStore._provider_safety_matches(
            state,
            allow_open=allow_open,
            require_current_registry=require_current_registry,
        ):
            raise ValidationStoreError('validation provider safety is unavailable')
        return state

    @staticmethod
    def _provider_safety_matches(
        state: AutoReviewProviderSafetyState,
        *,
        allow_open: bool,
        require_current_registry: bool = True,
    ) -> bool:
        registry_matches = (
            state.authorized_cost_policy_version
            == AUTO_REVIEW_COST_POLICY_VERSION
            and state.token_estimator_version
            == AUTO_REVIEW_TOKEN_ESTIMATOR_VERSION
            and state.tokenizer_encoding == AUTO_REVIEW_TOKENIZER_ENCODING
            and state.reply_priming_tokens == AUTO_REVIEW_REPLY_PRIMING_TOKENS
            and state.framing_safety_tokens == AUTO_REVIEW_FRAMING_SAFETY_TOKENS
            and Decimal(state.input_usd_per_1m)
            == AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M
            and Decimal(state.output_usd_per_1m)
            == AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M
        )
        return (
            (allow_open or not state.breaker_open)
            and (not require_current_registry or registry_matches)
        )

    @staticmethod
    def _lock_rollout_authority(
        db: Session,
        *,
        workflow_thread_id: str,
        expected_mode: Literal['shadow', 'enforce'] | None = None,
    ) -> bool:
        thread = db.get(AgentWorkflowThread, workflow_thread_id)
        workflow_request = db.get(AgentWorkflowRequest, workflow_thread_id)
        KeyedMutationGuard.acquire_projection_lock(db)
        if thread is None or workflow_request is None:
            return False
        rollout = db.scalar(
            select(AutoReviewRolloutState)
            .where(
                AutoReviewRolloutState.security_scope_id
                == thread.security_scope_id,
                AutoReviewRolloutState.policy_version
                == workflow_request.auto_review_policy_version,
            )
            .with_for_update(read=True)
        )
        effective_mode = expected_mode or workflow_request.auto_review_mode
        if rollout is None:
            return effective_mode != 'enforce'
        return bool(
            not rollout.breaker_open
            and rollout.control_epoch == workflow_request.rollout_control_epoch
            and rollout.authorization_generation
            == workflow_request.rollout_authorization_generation
            and rollout.max_authorized_percentage
            >= (workflow_request.authorized_percentage_at_launch or 0)
        )

    @staticmethod
    def _lock_source_states(
        db: Session,
        *,
        workflow_thread_id: str,
        allow_drift: bool = False,
    ) -> tuple[ValidationSourceState, ...]:
        refs = tuple(
            db.scalars(
                select(AgentWorkflowEvidenceRef)
                .where(
                    AgentWorkflowEvidenceRef.workflow_thread_id
                    == workflow_thread_id
                )
                .order_by(AgentWorkflowEvidenceRef.canonical_row_id)
            )
        )
        source_ids = sorted({ref.canonical_row_id for ref in refs})
        sources = tuple(
            db.scalars(
                select(Source)
                .where(Source.id.in_(source_ids))
                .order_by(Source.id)
                .with_for_update(read=True)
            )
        )
        by_id = {source.id: source for source in sources}
        states: list[ValidationSourceState] = []
        for ref in refs:
            source = by_id.get(ref.canonical_row_id)
            if (
                source is None
                or source.source_type != ref.canonical_source_type
                or source.permission_level != ref.permission_level_snapshot
                or source.server_content_signature_schema
                != 'server-source-content:v1'
                or source.server_content_signature != ref.content_signature
            ):
                if allow_drift:
                    return ()
                raise ValidationStoreError('validation source state changed')
            states.append(
                ValidationSourceState(
                    canonical_row_id=ref.canonical_row_id,
                    canonical_source_type=ref.canonical_source_type,
                    content_signature=ref.content_signature,
                    content_fingerprint=ref.content_fingerprint,
                    permission_level=ref.permission_level_snapshot,
                )
            )
        if not states or len(sources) != len(source_ids):
            if allow_drift:
                return ()
            raise ValidationStoreError('validation source state is unavailable')
        return tuple(states)

    def _current_guard_hmacs(
        self,
        db: Session,
        *,
        context: ValidationLockedContext,
        source_states: tuple[ValidationSourceState, ...],
    ) -> tuple[str, str]:
        thread = db.get(AgentWorkflowThread, context.workflow_thread_id)
        resolver = self._current_permission_resolver
        if thread is None or resolver is None or not source_states:
            raise ValidationStoreError('validation permission guard is unavailable')
        levels = tuple(sorted(set(resolver(thread.owner_subject_id))))
        if not levels or any(
            source.permission_level not in levels for source in source_states
        ):
            raise ValidationStoreError('validation owner permission changed')
        return (
            derive_validation_owner_permission_hmac(
                owner_subject_id=thread.owner_subject_id,
                permission_levels=levels,
                settings=self._settings,
            ),
            derive_validation_source_state_hmac(
                workflow_thread_id=context.workflow_thread_id,
                sources=source_states,
                settings=self._settings,
            ),
        )

    def _locked_workflow_request_for_context(
        self,
        db: Session,
        *,
        context: ValidationLockedContext,
        allow_cancelled: bool = False,
    ) -> AgentWorkflowRequest:
        thread = db.scalar(
            select(AgentWorkflowThread)
            .where(AgentWorkflowThread.thread_id == context.workflow_thread_id)
            .with_for_update()
        )
        workflow_request = db.scalar(
            select(AgentWorkflowRequest)
            .where(
                AgentWorkflowRequest.workflow_thread_id
                == context.workflow_thread_id
            )
            .with_for_update()
        )
        if thread is None or (
            not allow_cancelled and thread.cancelled_at is not None
        ):
            raise ValidationStoreError('validation workflow is unavailable')
        if workflow_request is None:
            raise ValidationStoreError('validation workflow request is unavailable')
        return workflow_request

    @staticmethod
    def _workflow_is_cancelled(
        db: Session,
        *,
        context: ValidationLockedContext,
    ) -> bool:
        thread = db.get(AgentWorkflowThread, context.workflow_thread_id)
        if thread is None:
            raise ValidationStoreError('validation workflow is unavailable')
        return thread.cancelled_at is not None

    @staticmethod
    def _verify_workflow_request(
        *,
        workflow_request: AgentWorkflowRequest,
        identity: AutoReviewValidationIdentity,
    ) -> None:
        expected = (
            identity.validator_provider,
            identity.validator_model,
            identity.reasoning_effort,
            identity.validator_prompt_version,
            identity.validator_output_contract_version,
            identity.policy_version,
            identity.cost_policy_version,
            identity.token_estimator_version,
            identity.tokenizer_encoding,
            AUTO_REVIEW_REPLY_PRIMING_TOKENS,
            AUTO_REVIEW_FRAMING_SAFETY_TOKENS,
            identity.max_input_tokens,
            identity.max_output_tokens,
            identity.max_candidates_per_batch,
            identity.max_batches_per_workflow,
            identity.max_candidates_per_workflow,
            identity.max_provider_attempts,
            identity.provider_timeout_seconds,
            identity.provider_send_start_window_seconds,
            identity.provider_attempt_lease_seconds,
            identity.provider_commit_grace_seconds,
            identity.input_cost_per_1m_tokens,
            identity.output_cost_per_1m_tokens,
            identity.authorized_percentage_at_launch,
            identity.rollout_authorization_generation,
            identity.provider_safety_state_version,
            identity.rollout_control_epoch,
            identity.confirmed_extraction_cost_ceiling_usd,
            identity.confirmed_validation_cost_ceiling_usd,
            identity.confirmed_total_cost_ceiling_usd,
            identity.fingerprint_key_version,
            identity.fingerprint_key_material_verifier,
        )
        actual = (
            workflow_request.auto_review_validator_provider,
            workflow_request.auto_review_validator_model,
            workflow_request.auto_review_reasoning_effort,
            workflow_request.auto_review_validator_prompt_version,
            workflow_request.auto_review_validator_output_contract_version,
            workflow_request.auto_review_policy_version,
            workflow_request.auto_review_cost_policy_version,
            workflow_request.auto_review_token_estimator_version,
            workflow_request.auto_review_tokenizer_encoding,
            workflow_request.auto_review_reply_priming_tokens,
            workflow_request.auto_review_framing_safety_tokens,
            workflow_request.auto_review_max_input_tokens,
            workflow_request.auto_review_max_output_tokens,
            workflow_request.auto_review_max_candidates_per_batch,
            workflow_request.auto_review_max_batches_per_workflow,
            workflow_request.auto_review_max_candidates_per_workflow,
            workflow_request.auto_review_max_provider_attempts,
            workflow_request.auto_review_provider_timeout_seconds,
            workflow_request.auto_review_provider_send_start_window_seconds,
            workflow_request.auto_review_provider_attempt_lease_seconds,
            workflow_request.auto_review_provider_commit_grace_seconds,
            Decimal(workflow_request.auto_review_validator_input_usd_per_1m),
            Decimal(workflow_request.auto_review_validator_output_usd_per_1m),
            workflow_request.authorized_percentage_at_launch,
            workflow_request.rollout_authorization_generation,
            workflow_request.validation_provider_safety_state_version,
            workflow_request.rollout_control_epoch,
            Decimal(workflow_request.confirmed_extraction_cost_ceiling_usd),
            Decimal(workflow_request.confirmed_validation_cost_ceiling_usd),
            Decimal(workflow_request.confirmed_total_cost_ceiling_usd),
            workflow_request.fingerprint_key_version,
            workflow_request.fingerprint_key_material_verifier,
        )
        if actual != expected:
            raise ValidationStoreError('validation workflow identity changed')

    @staticmethod
    def _require_workflow_budget(
        db: Session,
        *,
        workflow_request: AgentWorkflowRequest,
        request: ValidationBatchClaimRequest,
    ) -> None:
        extraction_calls = tuple(
            db.scalars(
                select(AutoReviewExtractionCall).where(
                    AutoReviewExtractionCall.workflow_thread_id
                    == request.workflow_thread_id
                )
            )
        )
        validation_calls = tuple(
            db.scalars(
                select(AutoReviewValidationCall)
                .where(
                    AutoReviewValidationCall.workflow_thread_id
                    == request.workflow_thread_id
                )
                .order_by(AutoReviewValidationCall.id)
                .with_for_update()
            )
        )
        candidate_count = sum(call.candidate_count for call in validation_calls)
        if (
            len(validation_calls) >= request.candidates[0].identity.max_batches_per_workflow
            or candidate_count + len(request.candidates)
            > request.candidates[0].identity.max_candidates_per_workflow
        ):
            raise ValidationStoreError('validation workflow batch cap exceeded')
        extraction_obligation = sum(
            (
                Decimal(call.charged_cost_usd)
                if call.status in {'completed', 'failed'}
                else Decimal(call.reserved_cost_usd)
            )
            for call in extraction_calls
        )
        validation_obligation = request.reserved_cost_usd + sum(
            (
                Decimal(call.charged_cost_usd)
                if call.status in {'completed', 'failed'}
                else Decimal(call.reserved_cost_usd)
            )
            for call in validation_calls
        )
        validation_ceiling = min(
            AUTO_REVIEW_MAX_VALIDATION_COST_USD,
            Decimal(workflow_request.confirmed_validation_cost_ceiling_usd),
        )
        total_ceiling = min(
            Decimal(workflow_request.confirmed_total_cost_ceiling_usd),
            Decimal(workflow_request.auto_review_budget_limit_usd),
        )
        if (
            validation_obligation > validation_ceiling
            or extraction_obligation + validation_obligation > total_ceiling
        ):
            raise ValidationStoreError('signed validation workflow budget exceeded')

    @staticmethod
    def _workflow_obligation_after_charge(
        db: Session,
        *,
        context: ValidationLockedContext,
        charged_cost: Decimal,
    ) -> tuple[Decimal, Decimal]:
        extraction_calls = tuple(
            db.scalars(
                select(AutoReviewExtractionCall).where(
                    AutoReviewExtractionCall.workflow_thread_id
                    == context.workflow_thread_id
                )
            )
        )
        validation_calls = tuple(
            db.scalars(
                select(AutoReviewValidationCall).where(
                    AutoReviewValidationCall.workflow_thread_id
                    == context.workflow_thread_id,
                    AutoReviewValidationCall.id != context.validation_call_id,
                )
            )
        )
        extraction_obligation = sum(
            (
                Decimal(call.charged_cost_usd)
                if call.status in {'completed', 'failed'}
                else Decimal(call.reserved_cost_usd)
            )
            for call in extraction_calls
        )
        validation_obligation = charged_cost + sum(
            (
                Decimal(call.charged_cost_usd)
                if call.status in {'completed', 'failed'}
                else Decimal(call.reserved_cost_usd)
            )
            for call in validation_calls
        )
        return extraction_obligation, validation_obligation

    def _open_budget_breaker(
        self,
        db: Session,
        *,
        safety: AutoReviewProviderSafetyState,
        call: AutoReviewValidationCall,
        charged_cost: Decimal,
        now: datetime,
    ) -> None:
        if safety.breaker_open:
            return
        prior_version = safety.state_version
        next_version = prior_version + 1
        next_sequence = safety.last_event_sequence + 1
        event = AutoReviewProviderSafetyEvent(
            provider_safety_state_id=safety.id,
            purpose=safety.purpose,
            provider=safety.provider,
            model=safety.model,
            reasoning_effort=safety.reasoning_effort,
            event_sequence=next_sequence,
            event_kind='budget_overrun',
            prior_state_version=prior_version,
            new_state_version=next_version,
            cost_policy_version=safety.authorized_cost_policy_version,
            token_estimator_version=safety.token_estimator_version,
            tokenizer_encoding=safety.tokenizer_encoding,
            reply_priming_tokens=safety.reply_priming_tokens,
            framing_safety_tokens=safety.framing_safety_tokens,
            input_usd_per_1m=safety.input_usd_per_1m,
            output_usd_per_1m=safety.output_usd_per_1m,
            prior_breaker_open=False,
            new_breaker_open=True,
            reason_code='budget_exceeded',
            actor_subject_hmac=None,
            call_hmac=build_keyed_fingerprint(
                {
                    'validation_call_id': call.id,
                    'workflow_thread_id': call.workflow_thread_id,
                    'batch_fingerprint': call.batch_fingerprint,
                },
                settings=self._settings,
                schema_version='auto-review-validation-call-overrun:v1',
                policy_version='auto-review-validation-call-overrun:v1',
            ),
            fingerprint_key_version=call.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                call.fingerprint_key_material_verifier
            ),
            created_at=now,
        )
        db.add(event)
        db.flush()
        safety.state_version = next_version
        safety.breaker_open = True
        safety.breaker_reason_code = 'budget_exceeded'
        safety.overrun_count += 1
        safety.last_overrun_cost_usd = charged_cost
        safety.last_overrun_at = now
        safety.last_event_sequence = next_sequence
        safety.last_event_id = event.id
        safety.updated_at = now

    def _locked_call(
        self,
        db: Session,
        context: ValidationLockedContext,
        *,
        for_update: bool = True,
    ) -> AutoReviewValidationCall:
        statement = select(AutoReviewValidationCall).where(
            AutoReviewValidationCall.id == context.validation_call_id,
            AutoReviewValidationCall.workflow_thread_id
            == context.workflow_thread_id,
            AutoReviewValidationCall.batch_fingerprint == context.batch_fingerprint,
            AutoReviewValidationCall.lease_token == context.lease_token,
        )
        if for_update:
            statement = statement.with_for_update()
        call = db.scalar(statement)
        if call is None:
            raise ValidationStoreError('locked validation context is required')
        return call

    @staticmethod
    def _locked_children(
        db: Session,
        *,
        call: AutoReviewValidationCall,
        for_update: bool = True,
    ) -> tuple[AutoReviewValidation, ...]:
        statement = (
            select(AutoReviewValidation)
            .where(AutoReviewValidation.validation_call_id == call.id)
            .order_by(AutoReviewValidation.validation_key)
        )
        if for_update:
            statement = statement.with_for_update()
        children = tuple(db.scalars(statement))
        if len(children) != call.candidate_count:
            raise ValidationStoreError('validation child set is corrupt')
        return children

    def _completed_projections(
        self, db: Session, *, call: AutoReviewValidationCall
    ) -> tuple[CompletedValidationProjection, ...]:
        children = self._locked_children(db, call=call, for_update=False)
        if any(row.status != 'completed' for row in children):
            raise ValidationStoreError('completed validation child set is corrupt')
        return tuple(
            self._projection(row, slot_index=index, cache_hit=True)
            for index, row in enumerate(children)
        )

    @staticmethod
    def _projection(
        row: AutoReviewValidation,
        *,
        slot_index: int,
        cache_hit: bool,
    ) -> CompletedValidationProjection:
        result = CandidateValidationResult.model_validate(
            {
                'candidate_slot_id': f'C{slot_index + 1:02d}',
                'claim_results': row.claim_results,
                'uncertainty_codes': row.uncertainty_codes,
                'conflict_codes': row.conflict_codes,
            }
        )
        if row.policy_decision is None or not row.policy_reason_codes:
            raise ValidationStoreError('completed validation decision is unavailable')
        return CompletedValidationProjection(
            validation_call_id=row.validation_call_id,
            validation_id=row.id,
            review_item_id=row.review_item_id,
            result=result,
            policy_decision=row.policy_decision,
            policy_reason_codes=tuple(row.policy_reason_codes),
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            estimated_cost_usd=Decimal(row.estimated_cost_usd),
            cache_hit=cache_hit,
        )

    def _failed_reason(
        self, db: Session, *, call: AutoReviewValidationCall
    ) -> AutoReviewPolicyReasonCode:
        children = self._locked_children(db, call=call, for_update=False)
        reasons = {
            tuple(row.policy_reason_codes)
            for row in children
            if row.status == 'failed'
        }
        if len(reasons) != 1:
            raise ValidationStoreError('failed validation reason is unavailable')
        reason_tuple = next(iter(reasons))
        if len(reason_tuple) != 1:
            raise ValidationStoreError('failed validation reason is unavailable')
        return reason_tuple[0]

    @staticmethod
    def _terminal_failure(
        *,
        call: AutoReviewValidationCall,
        children: tuple[AutoReviewValidation, ...],
        reason: AutoReviewPolicyReasonCode,
        input_tokens: int,
        output_tokens: int,
        charged_cost: Decimal,
        now: datetime,
    ) -> None:
        input_allocations = _allocate_integer_tokens(input_tokens, len(children))
        output_allocations = _allocate_integer_tokens(output_tokens, len(children))
        costs = _allocate_micro_usd(
            input_allocations,
            output_allocations,
            input_price=Decimal(call.input_usd_per_1m),
            output_price=Decimal(call.output_usd_per_1m),
            total_cost=charged_cost,
            tie_breakers=tuple(row.validation_key for row in children),
        )
        for index, row in enumerate(children):
            row.status = 'failed'
            row.policy_reason_codes = [reason]
            row.input_tokens = input_allocations[index]
            row.output_tokens = output_allocations[index]
            row.estimated_cost_usd = costs[index]
            row.completed_at = now
        call.status = 'failed'
        call.charged_input_tokens = input_tokens
        call.charged_output_tokens = output_tokens
        call.charged_cost_usd = charged_cost
        call.terminal_at = now

    @staticmethod
    def _require_matching_completion(
        context: ValidationLockedContext,
        completion: ValidationBatchCompletion,
    ) -> None:
        if (
            completion.validation_call_id,
            completion.workflow_thread_id,
            completion.batch_fingerprint,
            completion.lease_token,
        ) != (
            context.validation_call_id,
            context.workflow_thread_id,
            context.batch_fingerprint,
            context.lease_token,
        ):
            raise ValidationStoreError('validation completion context changed')

    @staticmethod
    def _require_matching_failure(
        context: ValidationLockedContext,
        failure: ValidationBatchFailure,
    ) -> None:
        if (
            failure.validation_call_id,
            failure.workflow_thread_id,
            failure.batch_fingerprint,
            failure.lease_token,
        ) != (
            context.validation_call_id,
            context.workflow_thread_id,
            context.batch_fingerprint,
            context.lease_token,
        ):
            raise ValidationStoreError('validation failure context changed')

    def _next_lease_token(self) -> str:
        token = self._lease_token_factory()
        if len(token) != 64:
            raise ValidationStoreError('validation lease token is unavailable')
        return token

    @staticmethod
    def _claimed_result(call: AutoReviewValidationCall) -> ValidationClaimResult:
        return ValidationClaimResult(
            disposition='claimed',
            validation_call_id=call.id,
            lease_token=call.lease_token,
            lease_expires_at=call.lease_expires_at,
            terminal_status=None,
            failure_reason_code=None,
            completed=(),
        )

    def _db_now(self, db: Session) -> datetime:
        if self._db_clock_override is not None:
            return self._db_clock_override(db)
        value = db.scalar(select(func.clock_timestamp()))
        if not isinstance(value, datetime):
            raise ValidationStoreError('database clock is unavailable')
        return value


def derive_validation_batch_fingerprint(
    *,
    workflow_thread_id: str,
    candidates: tuple[ValidationCandidateClaim, ...],
    prepared_content_hmac: str,
    serialized_character_count: int,
    framed_input_tokens: int,
    reserved_input_tokens: int,
    reserved_output_tokens: int,
    reserved_cost_usd: Decimal,
    settings: Settings,
) -> str:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.validation_key))
    return build_keyed_fingerprint(
        {
            'workflow_thread_id': workflow_thread_id,
            'candidates': [
                {
                    'review_item_id': candidate.review_item_id,
                    'validation_key': candidate.validation_key,
                    'identity': candidate.identity.fingerprint_payload(),
                }
                for candidate in ordered
            ],
            'prepared_content_hmac': prepared_content_hmac,
            'serialized_character_count': serialized_character_count,
            'framed_input_tokens': framed_input_tokens,
            'reserved_input_tokens': reserved_input_tokens,
            'reserved_output_tokens': reserved_output_tokens,
            'reserved_cost_usd': format(reserved_cost_usd, 'f'),
        },
        settings=settings,
        schema_version=VALIDATION_BATCH_SCHEMA_VERSION,
        policy_version='auto-review-cost:v1',
    )


def _token_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    input_price: Decimal,
    output_price: Decimal,
) -> Decimal:
    if input_tokens < 0 or output_tokens < 0:
        raise ValidationStoreError('validation usage is invalid')
    return (
        (
            Decimal(input_tokens) * input_price
            + Decimal(output_tokens) * output_price
        )
        / Decimal(1_000_000)
    ).quantize(Decimal('0.000001'), rounding=ROUND_CEILING)


def _allocate_integer_tokens(total: int, count: int) -> tuple[int, ...]:
    if total < 0 or count <= 0:
        raise ValidationStoreError('validation allocation is invalid')
    base, remainder = divmod(total, count)
    return tuple(base + (1 if index < remainder else 0) for index in range(count))


def _allocate_micro_usd(
    input_allocations: tuple[int, ...],
    output_allocations: tuple[int, ...],
    *,
    input_price: Decimal,
    output_price: Decimal,
    total_cost: Decimal,
    tie_breakers: tuple[str, ...],
) -> tuple[Decimal, ...]:
    raw = tuple(
        Decimal(input_tokens) * input_price
        + Decimal(output_tokens) * output_price
        for input_tokens, output_tokens in zip(
            input_allocations, output_allocations, strict=True
        )
    )
    floors = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in raw]
    total_micro = int(total_cost / Decimal('0.000001'))
    remaining = total_micro - sum(floors)
    if remaining < 0:
        raise ValidationStoreError('validation cost allocation is invalid')
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - Decimal(floors[index])), tie_breakers[index]),
    )
    for index in order[:remaining]:
        floors[index] += 1
    return tuple(Decimal(value) * Decimal('0.000001') for value in floors)
