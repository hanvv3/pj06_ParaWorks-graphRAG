from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from itertools import combinations
from threading import Lock, RLock
from time import monotonic
from types import MappingProxyType
from typing import Any, TypeVar
from uuid import uuid4

import tiktoken
from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.agent_runtime.auto_review_cost_policy import (
    AUTO_REVIEW_MAX_WORKFLOW_COST_USD,
    EXTRACTION_ROUTE_POLICIES,
    ExtractionRoutePolicy,
)
from backend.app.agent_runtime.auto_review_input_safety import (
    provider_logging_is_safe,
    scan_auto_review_plaintext,
)
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.agent_runtime.contracts import EvidencePacket
from backend.app.agent_runtime.keyed_mutation_guard import KeyedMutationGuard
from backend.app.agent_runtime.provider_send_fence import (
    FencedOpenAITransport,
    ProviderAttemptGrant,
    ProviderSendFenceError,
)
from backend.app.agent_runtime.review_v2_drafting import (
    CandidateEvidenceRefBinding,
    derive_candidate_evidence_version_hash,
)
from backend.app.core.config import Settings
from backend.app.models.agent_runs import AgentRun
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.auto_review import (
    AutoReviewExtractionCall,
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewValidationCall,
    ReviewItemEvidenceRef,
)
from backend.app.models.review import ReviewItem
from backend.app.models.source import Source
from backend.app.schemas.auto_review import (
    DecisionRecordExtractionResult,
    HistoryExtractionResult,
    MailDocumentExtractionResult,
    TimelineExtractionResult,
    TodoExtractionResult,
)

EXTRACTION_REGISTRY_VERSION = 'auto-review-extraction-registry:v1'
EXTRACTION_PLAN_SCHEMA = 'auto-review-extraction-plan:v1'
EXTRACTION_PLAN_SET_SCHEMA = 'auto-review-extraction-plan-set:v1'
EXTRACTION_SAFETY_SET_SCHEMA = 'auto-review-extraction-safety-set:v1'
EXTRACTION_RESULT_SET_SCHEMA = 'auto-review-extraction-result-set:v1'


class ExtractionCallStateError(RuntimeError):
    """Bounded extraction refusal; raw evidence/provider errors are excluded."""


@dataclass(frozen=True)
class ExtractionProviderSafetySnapshot:
    purpose: str
    provider: str
    model: str
    reasoning_effort: str
    state_version: int
    cost_policy_version: str
    token_estimator_version: str
    tokenizer_encoding: str
    reply_priming_tokens: int
    framing_safety_tokens: int
    input_usd_per_1m: Decimal
    output_usd_per_1m: Decimal
    breaker_open: bool


@dataclass(frozen=True)
class PreparedEvidenceSlotIdentity:
    slot_id: str
    source_id: str
    stable_message_identity: str
    text_fingerprint: str
    permission_level: str


@dataclass(frozen=True)
class PreparedExtractionInvocation:
    agent_name: str
    canonical_text: str
    canonical_bytes: bytes
    prepared_content_hmac: str
    character_count: int
    encoded_input_tokens: int
    framed_input_tokens: int
    max_output_tokens: int
    evidence_slot_ids: tuple[str, ...]
    evidence_slot_identities: tuple[PreparedEvidenceSlotIdentity, ...]
    evidence_slot_set_hmac: str
    output_schema: type[BaseModel]
    provider_options: Mapping[str, Any]

    def __repr__(self) -> str:
        return (
            f'<PreparedExtractionInvocation agent={self.agent_name!r} '
            f'characters={self.character_count} framed_tokens={self.framed_input_tokens}>'
        )


@dataclass(frozen=True)
class PreparedExtractionPlan:
    agent_name: str
    provider: str
    model: str
    reasoning_effort: str
    route_version: str
    prompt_version: str
    output_contract_version: str
    extraction_registry_version: str
    cost_policy_version: str
    token_estimator_version: str
    tokenizer_encoding: str
    reply_priming_tokens: int
    framing_safety_tokens: int
    max_input_chars: int
    max_input_tokens: int
    max_output_tokens: int
    max_candidates: int
    max_provider_attempts: int
    input_usd_per_1m: Decimal
    output_usd_per_1m: Decimal
    provider_safety_state_version: int
    provider_timeout_seconds: int
    provider_send_start_window_seconds: int
    provider_attempt_lease_seconds: int
    provider_commit_grace_seconds: int
    invocation: PreparedExtractionInvocation
    reserved_cost_usd: Decimal
    plan_hmac: str

    @property
    def timing_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.provider_timeout_seconds,
            self.provider_send_start_window_seconds,
            self.provider_attempt_lease_seconds,
            self.provider_commit_grace_seconds,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'provider': self.provider,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort,
            'route_version': self.route_version,
            'prompt_version': self.prompt_version,
            'output_contract_version': self.output_contract_version,
            'extraction_registry_version': self.extraction_registry_version,
            'cost_policy_version': self.cost_policy_version,
            'token_estimator_version': self.token_estimator_version,
            'tokenizer_encoding': self.tokenizer_encoding,
            'reply_priming_tokens': self.reply_priming_tokens,
            'framing_safety_tokens': self.framing_safety_tokens,
            'max_input_chars': self.max_input_chars,
            'max_input_tokens': self.max_input_tokens,
            'max_output_tokens': self.max_output_tokens,
            'max_candidates': self.max_candidates,
            'max_provider_attempts': self.max_provider_attempts,
            'input_usd_per_1m': format(self.input_usd_per_1m, 'f'),
            'output_usd_per_1m': format(self.output_usd_per_1m, 'f'),
            'provider_safety_state_version': self.provider_safety_state_version,
            'timing': list(self.timing_tuple),
            'prepared_content_hmac': self.invocation.prepared_content_hmac,
            'prepared_character_count': self.invocation.character_count,
            'framed_input_tokens': self.invocation.framed_input_tokens,
            'reserved_cost_usd': format(self.reserved_cost_usd, 'f'),
            'evidence_slot_set_hmac': self.invocation.evidence_slot_set_hmac,
            'provider_options': {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in sorted(self.invocation.provider_options.items())
            },
        }

    def identity_hmac(self, settings: Settings) -> str:
        return build_keyed_fingerprint(
            self.identity_payload(),
            settings=settings,
            schema_version=EXTRACTION_PLAN_SCHEMA,
            policy_version=EXTRACTION_PLAN_SCHEMA,
        )


@dataclass(frozen=True)
class PreparedExtractionPlanSet:
    plans: tuple[PreparedExtractionPlan, ...]
    selected_agent_names: tuple[str, ...]
    plan_set_hmac: str
    provider_safety_snapshot_set_hmac: str
    confirmed_cost_ceiling_usd: Decimal


_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    'mail_document_agent': MailDocumentExtractionResult,
    'timeline_agent': TimelineExtractionResult,
    'history_agent': HistoryExtractionResult,
    'decision_record_agent': DecisionRecordExtractionResult,
    'todo_agent': TodoExtractionResult,
}


def _canonical_json(value: Any) -> str:
    return unicodedata.normalize(
        'NFC',
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
    )


def _maximum_cost(policy: ExtractionRoutePolicy) -> Decimal:
    return (
        (
            Decimal(policy.max_input_tokens) * policy.input_cost_per_1m_tokens
            + Decimal(policy.max_output_tokens) * policy.output_cost_per_1m_tokens
        )
        / Decimal(1_000_000)
    ).quantize(Decimal('0.000001'), rounding=ROUND_CEILING)


def _render_invocation(
    packet: EvidencePacket,
    policy: ExtractionRoutePolicy,
    *,
    settings: Settings,
) -> PreparedExtractionInvocation:
    plaintext = '\n'.join(message.text for message in packet.messages)
    if not scan_auto_review_plaintext(plaintext).allowed:
        raise ExtractionCallStateError('sensitive input detected')
    evidence = []
    slots: list[str] = []
    slot_identities: list[PreparedEvidenceSlotIdentity] = []
    for index, message in enumerate(packet.messages, start=1):
        if index > 12:
            break
        slot = f'S{index:02d}'
        slots.append(slot)
        stable_identity = str(
            message.metadata.get('stable_message_identity')
            or f'{message.source_id}:{message.timestamp}:{index}'
        )
        slot_identities.append(
            PreparedEvidenceSlotIdentity(
                slot_id=slot,
                source_id=message.source_id,
                stable_message_identity=stable_identity,
                text_fingerprint=build_keyed_fingerprint(
                    message.text,
                    settings=settings,
                    schema_version='candidate-message-content:v1',
                    policy_version='candidate-message-content:v1',
                ),
                permission_level=message.permission_level,
            )
        )
        evidence.append(
            {
                'slot_id': slot,
                'text': message.text,
                'timestamp': message.timestamp,
                'message_alias': f'M{index:02d}',
                'participant_alias': f'P{index:02d}',
            }
        )
    dto = {
        'task': 'extract_zero_or_one_review_candidate',
        'agent_name': policy.agent_name,
        'prompt_version': policy.prompt_version,
        'output_contract_version': policy.output_contract_version,
        'requirements': {
            'result_kind': ['candidate', 'no_candidate'],
            'maximum_candidates': 1,
            'evidence_slots_must_be_exact': True,
        },
        'evidence': evidence,
    }
    canonical_text = _canonical_json(dto)
    character_count = len(canonical_text)
    if character_count > policy.max_input_chars:
        raise ExtractionCallStateError('extraction input character cap exceeded')
    try:
        encoded = len(
            tiktoken.get_encoding(policy.tokenizer_encoding).encode(canonical_text)
        )
    except Exception:
        raise ExtractionCallStateError(
            'extraction input tokenizer unavailable'
        ) from None
    framed = encoded + policy.reply_priming_tokens + policy.framing_safety_tokens
    if framed > policy.max_input_tokens:
        raise ExtractionCallStateError('extraction input token cap exceeded')
    canonical_bytes = canonical_text.encode('utf-8')
    content_hmac = build_keyed_fingerprint(
        canonical_text,
        settings=settings,
        schema_version='auto-review-extraction-content:v1',
        policy_version=policy.token_estimator_version,
    )
    return PreparedExtractionInvocation(
        agent_name=policy.agent_name,
        canonical_text=canonical_text,
        canonical_bytes=canonical_bytes,
        prepared_content_hmac=content_hmac,
        character_count=character_count,
        encoded_input_tokens=encoded,
        framed_input_tokens=framed,
        max_output_tokens=policy.max_output_tokens,
        evidence_slot_ids=tuple(slots),
        evidence_slot_identities=tuple(slot_identities),
        evidence_slot_set_hmac=build_keyed_fingerprint(
            [asdict(slot) for slot in slot_identities],
            settings=settings,
            schema_version='auto-review-extraction-evidence-slots:v1',
            policy_version='auto-review-extraction-evidence-slots:v1',
        ),
        output_schema=_OUTPUT_SCHEMAS[policy.agent_name],
        provider_options=MappingProxyType(
            {
                'transport': 'responses',
                'cache': False,
                'max_retries': 0,
                'callbacks': (),
                'tracing': False,
                'verbose': False,
            }
        ),
    )


def build_prepared_extraction_plan_set(
    *,
    packet: EvidencePacket,
    selected_agent_names: Sequence[str],
    settings: Settings,
    fingerprint_key_material_verifier: str,
    safety_snapshots: Sequence[ExtractionProviderSafetySnapshot],
) -> PreparedExtractionPlanSet:
    names = tuple(sorted(set(selected_agent_names)))
    if not names or len(names) != len(selected_agent_names):
        raise ExtractionCallStateError(
            'selected extraction agents must be non-empty and unique'
        )
    policies = {policy.agent_name: policy for policy in EXTRACTION_ROUTE_POLICIES}
    if any(name not in policies for name in names):
        raise ExtractionCallStateError('extraction registry unavailable')
    safety_by_identity = {
        (item.purpose, item.provider, item.model, item.reasoning_effort): item
        for item in safety_snapshots
    }
    if len(safety_by_identity) != len(safety_snapshots):
        raise ExtractionCallStateError('duplicate extraction safety snapshot')
    selected_safety_identities = {
        (
            'extraction',
            policies[name].provider,
            policies[name].model,
            policies[name].reasoning_effort,
        )
        for name in names
    }
    if set(safety_by_identity) != selected_safety_identities:
        raise ExtractionCallStateError('unselected extraction safety snapshot')
    plans: list[PreparedExtractionPlan] = []
    for name in names:
        policy = policies[name]
        safety = safety_by_identity.get(
            ('extraction', policy.provider, policy.model, policy.reasoning_effort)
        )
        if safety is None or safety.breaker_open:
            raise ExtractionCallStateError('extraction provider unavailable')
        expected = (
            policy.extraction_cost_policy_version,
            policy.token_estimator_version,
            policy.tokenizer_encoding,
            policy.reply_priming_tokens,
            policy.framing_safety_tokens,
            policy.input_cost_per_1m_tokens,
            policy.output_cost_per_1m_tokens,
        )
        actual = (
            safety.cost_policy_version,
            safety.token_estimator_version,
            safety.tokenizer_encoding,
            safety.reply_priming_tokens,
            safety.framing_safety_tokens,
            safety.input_usd_per_1m,
            safety.output_usd_per_1m,
        )
        if actual != expected:
            raise ExtractionCallStateError('extraction route price or policy drift')
        invocation = _render_invocation(packet, policy, settings=settings)
        seed = PreparedExtractionPlan(
            agent_name=name,
            provider=policy.provider,
            model=policy.model,
            reasoning_effort=policy.reasoning_effort,
            route_version=policy.route_version,
            prompt_version=policy.prompt_version,
            output_contract_version=policy.output_contract_version,
            extraction_registry_version=EXTRACTION_REGISTRY_VERSION,
            cost_policy_version=policy.extraction_cost_policy_version,
            token_estimator_version=policy.token_estimator_version,
            tokenizer_encoding=policy.tokenizer_encoding,
            reply_priming_tokens=policy.reply_priming_tokens,
            framing_safety_tokens=policy.framing_safety_tokens,
            max_input_chars=policy.max_input_chars,
            max_input_tokens=policy.max_input_tokens,
            max_output_tokens=policy.max_output_tokens,
            max_candidates=policy.max_candidates,
            max_provider_attempts=policy.max_provider_attempts,
            input_usd_per_1m=policy.input_cost_per_1m_tokens,
            output_usd_per_1m=policy.output_cost_per_1m_tokens,
            provider_safety_state_version=safety.state_version,
            provider_timeout_seconds=settings.auto_review_provider_timeout_seconds,
            provider_send_start_window_seconds=(
                settings.auto_review_provider_send_start_window_seconds
            ),
            provider_attempt_lease_seconds=(
                settings.auto_review_provider_attempt_lease_seconds
            ),
            provider_commit_grace_seconds=(
                settings.auto_review_provider_commit_grace_seconds
            ),
            invocation=invocation,
            reserved_cost_usd=_maximum_cost(policy),
            plan_hmac='',
        )
        plans.append(replace(seed, plan_hmac=seed.identity_hmac(settings)))
    ordered_plans = tuple(sorted(plans, key=lambda item: item.agent_name))
    plan_set_hmac = build_keyed_fingerprint(
        [plan.identity_payload() for plan in ordered_plans],
        settings=settings,
        schema_version=EXTRACTION_PLAN_SET_SCHEMA,
        policy_version=EXTRACTION_PLAN_SET_SCHEMA,
    )
    safety_values = [
        {
            **asdict(snapshot),
            'input_usd_per_1m': format(snapshot.input_usd_per_1m, 'f'),
            'output_usd_per_1m': format(snapshot.output_usd_per_1m, 'f'),
        }
        for snapshot in sorted(
            safety_snapshots,
            key=lambda item: (
                item.purpose,
                item.provider,
                item.model,
                item.reasoning_effort,
            ),
        )
    ]
    safety_hmac = build_keyed_fingerprint(
        {
            'fingerprint_key_material_verifier': fingerprint_key_material_verifier,
            'snapshots': safety_values,
        },
        settings=settings,
        schema_version=EXTRACTION_SAFETY_SET_SCHEMA,
        policy_version=EXTRACTION_SAFETY_SET_SCHEMA,
    )
    return PreparedExtractionPlanSet(
        plans=ordered_plans,
        selected_agent_names=names,
        plan_set_hmac=plan_set_hmac,
        provider_safety_snapshot_set_hmac=safety_hmac,
        confirmed_cost_ceiling_usd=sum(
            (plan.reserved_cost_usd for plan in ordered_plans), Decimal('0')
        ),
    )


def parse_extraction_result(
    schema: type[BaseModel],
    value: Any,
    *,
    native_truncated: bool = False,
    canonical_token_count_override: int | None = None,
) -> BaseModel:
    if native_truncated:
        raise ExtractionCallStateError('extraction output was truncated')
    try:
        if isinstance(value, str):
            value = json.loads(value)
        result = schema.model_validate(value)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
        raise ExtractionCallStateError('extraction output was malformed') from None
    counter = getattr(result, 'canonical_token_count', None)
    count = canonical_token_count_override
    if count is None:
        count = counter() if callable(counter) else 0
    if count > 2048:
        raise ExtractionCallStateError('extraction output token cap exceeded')
    return result


_R = TypeVar('_R')


def invoke_prepared_extraction(
    invocation: PreparedExtractionInvocation,
    provider: Callable[..., _R],
    *,
    store: Any,
    grant: ProviderAttemptGrant | None,
) -> _R:
    if not provider_logging_is_safe():
        raise ExtractionCallStateError('provider logging controls are unsafe')
    if grant is None:
        raise ExtractionCallStateError('committed provider attempt grant is required')
    dispatcher = getattr(store, 'dispatch_prepared_extraction', None)
    if not callable(dispatcher):
        raise ExtractionCallStateError('committed provider attempt store is required')
    return dispatcher(
        invocation=invocation,
        provider=provider,
        grant=grant,
    )


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ExtractionLockedContext:
    workflow_thread_id: str
    agent_name: str
    plan_hmac: str
    lease_token: str
    owner_subject_id: str | None = None
    allowed_permission_levels: tuple[str, ...] = ()


@dataclass
class _CallState:
    context: ExtractionLockedContext
    plan: PreparedExtractionPlan
    status: str
    claimed_at: datetime
    lease_expires_at: datetime
    reserved_cost_usd: Decimal
    provider_attempt_count: int = 0
    attempt_started_at: datetime | None = None
    charged_input_tokens: int | None = None
    charged_output_tokens: int | None = None
    charged_cost_usd: Decimal = Decimal('0')
    budget_overrun: bool = False
    result_kind: str | None = None
    result_candidate_count: int | None = None
    result_candidate_set_hmac: str | None = None
    terminal_at: datetime | None = None
    cancelled: bool = False
    drifted: bool = False


@dataclass(frozen=True)
class ExtractionCallSnapshot:
    status: str
    reserved_cost_usd: Decimal
    charged_cost_usd: Decimal
    provider_attempt_count: int
    attempt_started_at: datetime | None
    lease_expires_at: datetime
    budget_overrun: bool
    result_kind: str | None
    result_candidate_count: int | None
    result_candidate_set_hmac: str | None
    timing_tuple: tuple[int, int, int, int]

    @property
    def cacheable(self) -> bool:
        return (
            self.status == 'completed'
            and self.result_kind in {'candidate', 'no_candidate'}
            and self.result_candidate_set_hmac is not None
        )


def _result_set_hmac(items: Sequence[tuple[str, str]]) -> str:
    canonical = _canonical_json(
        {
            'schema': EXTRACTION_RESULT_SET_SCHEMA,
            'items': [list(item) for item in sorted(items)],
        }
    ).encode('utf-8')
    return hmac.new(
        b'process-local-test-ledger-key', canonical, hashlib.sha256
    ).hexdigest()


class ExtractionCallLedger:
    """Deterministic store used by SQLite/fake tests; SQL stores use the same state rules."""

    def __init__(
        self,
        *,
        db_clock: Callable[[], datetime] | None = None,
        workflow_budget_limit: Decimal = AUTO_REVIEW_MAX_WORKFLOW_COST_USD,
        permit_monotonic: Callable[[], float] = monotonic,
    ) -> None:
        self._db_clock = db_clock or (lambda: datetime.now(UTC))
        self._budget_limit = workflow_budget_limit
        self._permit_monotonic = permit_monotonic
        self._calls: dict[tuple[str, str], _CallState] = {}
        self._permits: dict[
            tuple[str, str], tuple[ProviderAttemptGrant, FencedOpenAITransport[Any]]
        ] = {}
        self._lock = RLock()
        self.disabled = False
        self.validation_breaker_open = False
        self.breaker_open = False
        self.last_lock_order: tuple[str, ...] = ()

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def claim_or_replay(
        self,
        workflow_thread_id: str,
        plan: PreparedExtractionPlan,
        *,
        validation_obligation: Decimal = Decimal('0'),
        expected_now: datetime | None = None,
    ) -> ExtractionLockedContext:
        del expected_now
        with self._lock:
            self.last_lock_order = (
                'runtime_key_state',
                'provider_safety:extraction',
                'provider_safety:validation',
                'sources',
                'workflow',
                'agent_run',
                'extraction_call',
            )
            if self.disabled or self.validation_breaker_open or self.breaker_open:
                raise ExtractionCallStateError(
                    'extraction is disabled or a breaker is open'
                )
            if plan.reserved_cost_usd + validation_obligation > self._budget_limit:
                raise ExtractionCallStateError('signed workflow budget exceeded')
            key = (workflow_thread_id, plan.agent_name)
            current = self._calls.get(key)
            if current is not None:
                if current.plan.plan_hmac != plan.plan_hmac:
                    raise ExtractionCallStateError(
                        'workflow agent already owns another plan'
                    )
                return current.context
            now = self._db_clock()
            token = hashlib.sha256(
                f'{workflow_thread_id}:{plan.agent_name}:{now.isoformat()}'.encode()
            ).hexdigest()
            context = ExtractionLockedContext(
                workflow_thread_id, plan.agent_name, plan.plan_hmac, token
            )
            self._calls[key] = _CallState(
                context=context,
                plan=plan,
                status='claimed',
                claimed_at=now,
                lease_expires_at=now
                + timedelta(seconds=plan.provider_attempt_lease_seconds),
                reserved_cost_usd=plan.reserved_cost_usd,
            )
            return context

    def mark_attempt_started(
        self,
        context: ExtractionLockedContext,
        *,
        commit: Callable[[], None] | None = None,
    ) -> ProviderAttemptGrant:
        with self._lock:
            state = self._state(context)
            if state.status != 'claimed' or state.provider_attempt_count != 0:
                raise ExtractionCallStateError('provider attempt is not claimable')
            if (
                self.disabled
                or self.validation_breaker_open
                or self.breaker_open
                or state.drifted
            ):
                self._terminal_fail(state, charge=False)
                raise ExtractionCallStateError('pre-attempt state changed')
            if state.cancelled:
                self._terminal_fail(state, charge=False)
                raise ExtractionCallStateError('extraction was cancelled')
            now = self._db_clock()
            state.provider_attempt_count = 1
            state.attempt_started_at = now
            state.lease_expires_at = now + timedelta(
                seconds=state.plan.provider_attempt_lease_seconds
            )
            attempt_id = hashlib.sha256(
                f'{context.lease_token}:attempt:1'.encode()
            ).hexdigest()
            (commit or (lambda: None))()
            deadline = (
                self._permit_monotonic()
                + state.plan.provider_send_start_window_seconds
            )
            timeout = state.plan.provider_timeout_seconds
            lease_expiry = state.lease_expires_at
            clock = self._permit_monotonic

            class _CommittedPermit:
                __slots__ = ('_consumed', '_invalidated', '_lock')

                def __init__(self) -> None:
                    self._consumed = False
                    self._invalidated = False
                    self._lock = Lock()

                @property
                def attempt_id(self) -> str:
                    return attempt_id

                @property
                def send_start_deadline_monotonic(self) -> float:
                    return deadline

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
                        if self._invalidated or clock() >= deadline:
                            raise ProviderSendFenceError(
                                'provider send permit expired'
                            )
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

            permit = _CommittedPermit()

            class _CommittedTransport(FencedOpenAITransport[Any]):
                __slots__ = ('_dispatched', '_http_hook')

                def __init__(self) -> None:
                    from backend.app.agent_runtime.auto_review_cost_policy import (
                        _SERVER_OWNED_FENCED_SEND_HOOK,
                    )

                    self._dispatched = False
                    self._http_hook = _SERVER_OWNED_FENCED_SEND_HOOK

                @property
                def http_hook(self) -> object:
                    return self._http_hook

                def _consume_from_store(self) -> None:
                    from backend.app.agent_runtime.auto_review_cost_policy import (
                        is_server_owned_fenced_send_hook,
                    )

                    if not is_server_owned_fenced_send_hook(self._http_hook):
                        raise ProviderSendFenceError(
                            'server-owned provider send hook is required'
                        )
                    if self._dispatched:
                        raise ProviderSendFenceError(
                            'redirect, retry, or second dispatch is forbidden'
                        )
                    permit.consume_at_dispatch()
                    self._dispatched = True

                def _dispatch_consumed(
                    self,
                    send: Callable[..., Any],
                    *,
                    request_body: Any = None,
                ) -> Any:
                    del request_body
                    if not self._dispatched:
                        raise ProviderSendFenceError(
                            'store-owned dispatch authentication is required'
                        )
                    return send(timeout=timeout)

            transport = _CommittedTransport()

            class _CommittedGrant:
                __slots__ = ()

                @property
                def attempt_id(self) -> str:
                    return attempt_id

                @property
                def permit(self) -> _CommittedPermit:
                    return permit

                @property
                def provider_timeout_seconds(self) -> int:
                    return timeout

                @property
                def authoritative_lease_expires_at(self) -> datetime:
                    return lease_expiry

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

            grant: ProviderAttemptGrant[Any] = _CommittedGrant()
            self._permits[(context.workflow_thread_id, context.agent_name)] = (
                grant,
                transport,
            )
            return grant

    def dispatch_prepared_extraction(
        self,
        *,
        invocation: PreparedExtractionInvocation,
        provider: Callable[..., _R],
        grant: ProviderAttemptGrant,
    ) -> _R:
        with self._lock:
            matches = [
                (key, state, active[1])
                for key, state in self._calls.items()
                if (active := self._permits.get(key)) is not None
                and active[0] is grant
            ]
            if len(matches) != 1:
                raise ExtractionCallStateError(
                    'committed provider attempt grant is required'
                )
            key, state, transport = matches[0]
            if (
                state.status != 'claimed'
                or state.provider_attempt_count != 1
                or state.plan.invocation is not invocation
                or state.drifted
                or self._db_clock() >= state.lease_expires_at
            ):
                self._invalidate_permit(state.context)
                raise ExtractionCallStateError('prepared extraction dispatch changed')
            transport._consume_from_store()
            self._permits.pop(key, None)
        return transport._dispatch_consumed(
            lambda *, timeout: provider(invocation, timeout=timeout),
            request_body=invocation.canonical_bytes,
        )

    def complete(
        self,
        context: ExtractionLockedContext,
        *,
        result: Any,
        usage: ProviderUsage,
        canonical_output_tokens: int | None = None,
        native_truncated: bool = False,
        candidate_pair: tuple[str, str] | None = None,
    ) -> ExtractionCallSnapshot:
        with self._lock:
            state = self._state(context)
            if state.status != 'claimed' or state.provider_attempt_count != 1:
                raise ExtractionCallStateError('extraction call is not completable')
            charge = self._usage_cost(state.plan, usage)
            state.charged_input_tokens = max(usage.input_tokens, 0)
            state.charged_output_tokens = max(usage.output_tokens, 0)
            state.charged_cost_usd = charge
            if (
                usage.input_tokens > state.plan.max_input_tokens
                or usage.output_tokens > state.plan.max_output_tokens
                or charge > state.reserved_cost_usd
            ):
                state.budget_overrun = True
                self.breaker_open = True
                self._terminal_fail(state, charge=True, preserve_known_charge=True)
                raise ExtractionCallStateError('provider usage budget overrun')
            if state.cancelled or state.drifted:
                self._terminal_fail(state, charge=True, preserve_known_charge=True)
                raise ExtractionCallStateError('post-call state changed')
            try:
                parsed = parse_extraction_result(
                    state.plan.invocation.output_schema,
                    result,
                    native_truncated=native_truncated,
                    canonical_token_count_override=canonical_output_tokens,
                )
            except ExtractionCallStateError:
                self._terminal_fail(state, charge=True, preserve_known_charge=True)
                raise
            result_kind = parsed.result_kind
            if result_kind == 'candidate':
                if candidate_pair is None:
                    self._terminal_fail(state, charge=True, preserve_known_charge=True)
                    raise ExtractionCallStateError(
                        'candidate evidence binding is missing'
                    )
                candidate = parsed.candidate
                slots = {
                    binding.evidence_slot_id
                    for binding in candidate.field_evidence_bindings
                }
                if not slots.issubset(state.plan.invocation.evidence_slot_ids):
                    self._terminal_fail(state, charge=True, preserve_known_charge=True)
                    raise ExtractionCallStateError(
                        'candidate evidence binding is invalid'
                    )
                pairs = [candidate_pair]
                count = 1
            else:
                pairs = []
                count = 0
            state.status = 'completed'
            state.result_kind = result_kind
            state.result_candidate_count = count
            state.result_candidate_set_hmac = _result_set_hmac(pairs)
            state.terminal_at = self._db_clock()
            self._invalidate_permit(context)
            return self.snapshot(context)

    def fail(self, context: ExtractionLockedContext, *, reason_code: str) -> None:
        del reason_code
        with self._lock:
            state = self._state(context)
            self._terminal_fail(state, charge=state.provider_attempt_count == 1)

    def cancel(self, context: ExtractionLockedContext) -> None:
        with self._lock:
            state = self._state(context)
            state.cancelled = True
            if state.provider_attempt_count == 0 and state.status == 'claimed':
                self._invalidate_permit(context)
                self._terminal_fail(state, charge=False)

    def set_drift(self, context: ExtractionLockedContext) -> None:
        with self._lock:
            self._state(context).drifted = True
            self._invalidate_permit(context)

    def recover_expired(self, context: ExtractionLockedContext) -> None:
        with self._lock:
            state = self._state(context)
            if self._db_clock() < state.lease_expires_at:
                raise ExtractionCallStateError('extraction lease is still live')
            self._terminal_fail(state, charge=state.provider_attempt_count == 1)

    def verify_replay(
        self,
        context: ExtractionLockedContext,
        *,
        candidate_pairs: Sequence[tuple[str, str]],
    ) -> None:
        with self._lock:
            state = self._state(context)
            if not self.snapshot(context).cacheable:
                raise ExtractionCallStateError('completed extraction result is corrupt')
            if state.result_candidate_set_hmac != _result_set_hmac(candidate_pairs):
                raise ExtractionCallStateError(
                    'completed extraction child set is corrupt'
                )
            if len(candidate_pairs) != state.result_candidate_count:
                raise ExtractionCallStateError(
                    'completed extraction child count is corrupt'
                )

    def force_corrupt_completed(self, context: ExtractionLockedContext) -> None:
        with self._lock:
            state = self._state(context)
            state.status = 'completed'
            state.provider_attempt_count = 1
            state.attempt_started_at = self._db_clock()
            state.terminal_at = self._db_clock()
            self._invalidate_permit(context)

    def snapshot(self, context: ExtractionLockedContext) -> ExtractionCallSnapshot:
        with self._lock:
            state = self._state(context)
            return ExtractionCallSnapshot(
                status=state.status,
                reserved_cost_usd=state.reserved_cost_usd,
                charged_cost_usd=state.charged_cost_usd,
                provider_attempt_count=state.provider_attempt_count,
                attempt_started_at=state.attempt_started_at,
                lease_expires_at=state.lease_expires_at,
                budget_overrun=state.budget_overrun,
                result_kind=state.result_kind,
                result_candidate_count=state.result_candidate_count,
                result_candidate_set_hmac=state.result_candidate_set_hmac,
                timing_tuple=state.plan.timing_tuple,
            )

    def _state(self, context: ExtractionLockedContext) -> _CallState:
        state = self._calls.get((context.workflow_thread_id, context.agent_name))
        if state is None or state.context != context:
            raise ExtractionCallStateError('extraction call context is invalid')
        return state

    @staticmethod
    def _usage_cost(plan: PreparedExtractionPlan, usage: ProviderUsage) -> Decimal:
        return (
            (
                Decimal(max(usage.input_tokens, 0)) * plan.input_usd_per_1m
                + Decimal(max(usage.output_tokens, 0)) * plan.output_usd_per_1m
            )
            / Decimal(1_000_000)
        ).quantize(Decimal('0.000001'), rounding=ROUND_CEILING)

    def _terminal_fail(
        self,
        state: _CallState,
        *,
        charge: bool,
        preserve_known_charge: bool = False,
    ) -> None:
        if state.status != 'claimed':
            return
        state.status = 'failed'
        state.terminal_at = self._db_clock()
        if charge:
            if not preserve_known_charge:
                state.charged_input_tokens = state.plan.max_input_tokens
                state.charged_output_tokens = state.plan.max_output_tokens
                state.charged_cost_usd = state.reserved_cost_usd
        else:
            state.charged_input_tokens = 0
            state.charged_output_tokens = 0
            state.charged_cost_usd = Decimal('0')
        state.result_kind = None
        state.result_candidate_count = None
        state.result_candidate_set_hmac = None
        self._invalidate_permit(state.context)

    def _invalidate_permit(self, context: ExtractionLockedContext) -> None:
        active = self._permits.pop(
            (context.workflow_thread_id, context.agent_name), None
        )
        if active is not None:
            active[0].permit._invalidate_from_store()


class ExtractionCallStore:
    """PostgreSQL-authoritative E1/E2/E3 extraction ledger.

    Plaintext lives only in the caller's immutable invocation. This store writes
    keyed identities, bounded codes, token counts, and Decimal charges.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        settings: Settings,
        prepared_plan_set: PreparedExtractionPlanSet,
        db_clock: Callable[[Session], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._plan_set = prepared_plan_set
        self._plans = {plan.agent_name: plan for plan in prepared_plan_set.plans}
        self._db_clock_override = db_clock
        self._active_grants: dict[
            str,
            tuple[
                ProviderAttemptGrant,
                ExtractionLockedContext,
                FencedOpenAITransport[Any],
            ],
        ] = {}
        self._grant_lock = RLock()

    def claim_or_replay(
        self,
        workflow_thread_id: str,
        plan: PreparedExtractionPlan,
        *,
        actor_subject_id: str,
        allowed_permission_levels: tuple[str, ...],
        validation_obligation: Decimal = Decimal('0'),
        expected_now: datetime | None = None,
    ) -> ExtractionLockedContext:
        del expected_now
        self._require_signed_plan(plan)
        try:
            with self._session_factory() as db, db.begin():
                thread, request, refs, sources, safety = self._lock_prefix(
                    db, workflow_thread_id=workflow_thread_id
                )
                now = self._db_now(db)
                existing = db.scalar(
                    select(AutoReviewExtractionCall).where(
                        AutoReviewExtractionCall.workflow_thread_id
                        == workflow_thread_id,
                        AutoReviewExtractionCall.agent_name == plan.agent_name,
                    )
                )
                self._ensure_claim_ready(
                    db=db,
                    thread=thread,
                    request=request,
                    refs=refs,
                    sources=sources,
                    safety=safety,
                    plan=plan,
                    actor_subject_id=actor_subject_id,
                    allowed_permission_levels=allowed_permission_levels,
                    validation_obligation=validation_obligation,
                    prospective_reserve=(
                        plan.reserved_cost_usd if existing is None else Decimal('0')
                    ),
                )
                if existing is not None:
                    self._verify_existing_plan(existing, plan)
                    if existing.status == 'failed':
                        raise ExtractionCallStateError(
                            'failed extraction call is not cacheable'
                        )
                    self._verify_completed_replay(
                        db,
                        thread=thread,
                        request=request,
                        call=existing,
                    )
                    return self._context(
                        existing,
                        owner_subject_id=actor_subject_id,
                        allowed_permission_levels=allowed_permission_levels,
                    )
                permission = _strictest_permission(
                    tuple(ref.permission_level_snapshot for ref in refs)
                )
                run = AgentRun(
                    agent_name=plan.agent_name,
                    prompt_version=plan.prompt_version,
                    status='claimed',
                    source_window='company-memory-review-v2.1',
                    cache_key=plan.plan_hmac,
                    model_name=plan.model,
                    generation_provider=plan.provider,
                    generation_reasoning_effort=plan.reasoning_effort,
                    generation_route_version=plan.route_version,
                    generation_output_contract_version=plan.output_contract_version,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    permission_level=permission,
                    metadata_={
                        'extraction_registry_version': plan.extraction_registry_version,
                        'cost_policy_version': plan.cost_policy_version,
                    },
                    workflow_thread_id=workflow_thread_id,
                    effect_key=plan.plan_hmac,
                    started_at=now,
                )
                db.add(run)
                db.flush()
                lease_token = uuid4().hex
                call = AutoReviewExtractionCall(
                    agent_run_id=run.id,
                    workflow_thread_id=workflow_thread_id,
                    agent_name=plan.agent_name,
                    extraction_plan_hmac=plan.plan_hmac,
                    provider=plan.provider,
                    model=plan.model,
                    reasoning_effort=plan.reasoning_effort,
                    route_version=plan.route_version,
                    prompt_version=plan.prompt_version,
                    output_contract_version=plan.output_contract_version,
                    extraction_registry_version=plan.extraction_registry_version,
                    cost_policy_version=plan.cost_policy_version,
                    provider_safety_state_version=plan.provider_safety_state_version,
                    token_estimator_version=plan.token_estimator_version,
                    tokenizer_encoding=plan.tokenizer_encoding,
                    reply_priming_tokens=plan.reply_priming_tokens,
                    framing_safety_tokens=plan.framing_safety_tokens,
                    prepared_content_hmac=plan.invocation.prepared_content_hmac,
                    prepared_character_count=plan.invocation.character_count,
                    framed_input_token_cap=plan.invocation.framed_input_tokens,
                    total_output_token_cap=plan.max_output_tokens,
                    max_candidates_per_agent=1,
                    input_usd_per_1m=plan.input_usd_per_1m,
                    output_usd_per_1m=plan.output_usd_per_1m,
                    fingerprint_key_version=request.fingerprint_key_version,
                    fingerprint_key_material_verifier=(
                        request.fingerprint_key_material_verifier
                    ),
                    provider_timeout_seconds=plan.provider_timeout_seconds,
                    provider_send_start_window_seconds=(
                        plan.provider_send_start_window_seconds
                    ),
                    provider_attempt_lease_seconds=(
                        plan.provider_attempt_lease_seconds
                    ),
                    provider_commit_grace_seconds=plan.provider_commit_grace_seconds,
                    workflow_extraction_cost_ceiling_usd=(
                        request.confirmed_extraction_cost_ceiling_usd
                    ),
                    workflow_total_cost_ceiling_usd=(
                        request.confirmed_total_cost_ceiling_usd
                    ),
                    status='claimed',
                    lease_token=lease_token,
                    lease_expires_at=now
                    + timedelta(seconds=plan.provider_attempt_lease_seconds),
                    max_provider_attempts=1,
                    provider_attempt_count=0,
                    reserved_input_tokens=plan.max_input_tokens,
                    reserved_output_tokens=plan.max_output_tokens,
                    reserved_cost_usd=plan.reserved_cost_usd,
                    charged_cost_usd=Decimal('0'),
                    budget_overrun=False,
                    budget_overrun_cost_usd=Decimal('0'),
                    claimed_at=now,
                )
                db.add(call)
                db.flush()
                return self._context(
                    call,
                    owner_subject_id=actor_subject_id,
                    allowed_permission_levels=allowed_permission_levels,
                )
        except ExtractionCallStateError as exc:
            if str(exc) == 'evidence_binding_mismatch':
                self._persist_replay_binding_failure(
                    workflow_thread_id=workflow_thread_id,
                    plan=plan,
                )
            raise
        except IntegrityError:
            with self._session_factory() as db:
                existing = db.scalar(
                    select(AutoReviewExtractionCall).where(
                        AutoReviewExtractionCall.workflow_thread_id
                        == workflow_thread_id,
                        AutoReviewExtractionCall.agent_name == plan.agent_name,
                    )
                )
                if existing is None:
                    raise
                self._verify_existing_plan(existing, plan)
                return self._context(
                    existing,
                    owner_subject_id=actor_subject_id,
                    allowed_permission_levels=allowed_permission_levels,
                )

    def mark_attempt_started(
        self,
        context: ExtractionLockedContext,
        *,
        commit: Callable[[], None] | None = None,
    ) -> ProviderAttemptGrant:
        del commit
        refusal: str | None = None
        with self._session_factory() as db:
            with db.begin():
                thread, request, refs, sources, safety = self._lock_prefix(
                    db, workflow_thread_id=context.workflow_thread_id
                )
                call = self._locked_call(db, context)
                run = db.scalar(
                    select(AgentRun)
                    .where(AgentRun.id == call.agent_run_id)
                    .with_for_update()
                )
                if run is None:
                    raise ExtractionCallStateError('extraction agent run is missing')
                plan = self._plans[context.agent_name]
                try:
                    self._ensure_claim_ready(
                        db=db,
                        thread=thread,
                        request=request,
                        refs=refs,
                        sources=sources,
                        safety=safety,
                        plan=plan,
                        actor_subject_id=context.owner_subject_id,
                        allowed_permission_levels=context.allowed_permission_levels,
                        validation_obligation=Decimal('0'),
                        prospective_reserve=Decimal('0'),
                    )
                except ExtractionCallStateError as exc:
                    refusal = str(exc)
                    now = self._db_now(db)
                    call.charged_input_tokens = 0
                    call.charged_output_tokens = 0
                    call.charged_cost_usd = Decimal('0')
                    self._sql_fail(call, run, now=now)
                if refusal is not None:
                    timeout = 0
                    window = 0
                    lease_expiry = self._db_now(db)
                    attempt_id = ''
                    continue_marker = False
                else:
                    continue_marker = True
                self._verify_call_identity(call, plan)
                if continue_marker and (
                    call.status != 'claimed' or call.provider_attempt_count != 0
                ):
                    raise ExtractionCallStateError('provider attempt is not claimable')
                if continue_marker:
                    now = self._db_now(db)
                    call.provider_attempt_count = 1
                    call.attempt_started_at = now
                    call.lease_expires_at = now + timedelta(
                        seconds=call.provider_attempt_lease_seconds
                    )
                    attempt_id = build_keyed_fingerprint(
                        {
                            'call_id': call.id,
                            'attempt': 1,
                            'started_at': now.isoformat(),
                        },
                        settings=self._settings,
                        schema_version='auto-review-extraction-attempt:v1',
                        policy_version='auto-review-extraction-attempt:v1',
                    )
                    timeout = call.provider_timeout_seconds
                    window = call.provider_send_start_window_seconds
                    lease_expiry = call.lease_expires_at
            if refusal is not None:
                raise ExtractionCallStateError(refusal)
            deadline = monotonic() + window

            class _CommittedPermit:
                __slots__ = ('_consumed', '_invalidated', '_lock')

                def __init__(self) -> None:
                    self._consumed = False
                    self._invalidated = False
                    self._lock = Lock()

                @property
                def attempt_id(self) -> str:
                    return attempt_id

                @property
                def send_start_deadline_monotonic(self) -> float:
                    return deadline

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
                        if self._invalidated or monotonic() >= deadline:
                            raise ProviderSendFenceError(
                                'provider send permit expired'
                            )
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

            permit = _CommittedPermit()

            class _CommittedTransport(FencedOpenAITransport[Any]):
                __slots__ = ('_dispatched', '_http_hook')

                def __init__(self) -> None:
                    from backend.app.agent_runtime.auto_review_cost_policy import (
                        _SERVER_OWNED_FENCED_SEND_HOOK,
                    )

                    self._dispatched = False
                    self._http_hook = _SERVER_OWNED_FENCED_SEND_HOOK

                @property
                def http_hook(self) -> object:
                    return self._http_hook

                def _consume_from_store(self) -> None:
                    from backend.app.agent_runtime.auto_review_cost_policy import (
                        is_server_owned_fenced_send_hook,
                    )

                    if not is_server_owned_fenced_send_hook(self._http_hook):
                        raise ProviderSendFenceError(
                            'server-owned provider send hook is required'
                        )
                    if self._dispatched:
                        raise ProviderSendFenceError(
                            'redirect, retry, or second dispatch is forbidden'
                        )
                    permit.consume_at_dispatch()
                    self._dispatched = True

                def _dispatch_consumed(
                    self,
                    send: Callable[..., Any],
                    *,
                    request_body: Any = None,
                ) -> Any:
                    del request_body
                    if not self._dispatched:
                        raise ProviderSendFenceError(
                            'store-owned dispatch authentication is required'
                        )
                    return send(timeout=timeout)

            transport = _CommittedTransport()

            class _CommittedGrant:
                __slots__ = ()

                @property
                def attempt_id(self) -> str:
                    return attempt_id

                @property
                def permit(self) -> _CommittedPermit:
                    return permit

                @property
                def provider_timeout_seconds(self) -> int:
                    return timeout

                @property
                def authoritative_lease_expires_at(self) -> datetime:
                    return lease_expiry

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

            grant: ProviderAttemptGrant[Any] = _CommittedGrant()
            with self._grant_lock:
                self._active_grants[grant.attempt_id] = (grant, context, transport)
            return grant

    def dispatch_prepared_extraction(
        self,
        *,
        invocation: PreparedExtractionInvocation,
        provider: Callable[..., _R],
        grant: ProviderAttemptGrant,
    ) -> _R:
        active: tuple[
            ProviderAttemptGrant,
            ExtractionLockedContext,
            FencedOpenAITransport[Any],
        ]
        with self._grant_lock:
            found = self._active_grants.get(grant.attempt_id)
            if found is None or found[0] is not grant:
                raise ExtractionCallStateError(
                    'committed provider attempt grant is required'
                )
            active = found
        context = active[1]
        transport = active[2]
        plan = self._plans[context.agent_name]
        if plan.invocation is not invocation:
            self._invalidate_active_grant(context)
            raise ExtractionCallStateError('prepared extraction dispatch changed')
        try:
            with self._session_factory() as db, db.begin():
                thread, request, refs, sources, safety = self._lock_prefix(
                    db, workflow_thread_id=context.workflow_thread_id
                )
                call = self._locked_call(db, context)
                now = self._db_now(db)
                if (
                    call.status != 'claimed'
                    or call.provider_attempt_count != 1
                    or call.lease_expires_at is None
                    or now >= call.lease_expires_at
                    or not self._runtime_key_matches(db, request)
                    or context.owner_subject_id is None
                    or thread.owner_subject_id != context.owner_subject_id
                    or any(
                        ref.permission_level_snapshot
                        not in context.allowed_permission_levels
                        for ref in refs
                    )
                    or not self._current_identity_matches(
                        thread=thread,
                        request=request,
                        refs=refs,
                        sources=sources,
                        safety=safety,
                        plan=plan,
                        allow_cancelled=True,
                    )
                ):
                    live = False
                else:
                    live = True
        except Exception:
            self._invalidate_active_grant(context)
            raise
        if not live:
            self._invalidate_active_grant(context)
            raise ExtractionCallStateError(
                'committed provider attempt grant is not live'
            )
        with self._grant_lock:
            current = self._active_grants.get(grant.attempt_id)
            if current is not active or current[0] is not grant:
                raise ExtractionCallStateError(
                    'committed provider attempt grant is not live'
                )
            try:
                transport._consume_from_store()
            except Exception:
                self._active_grants.pop(grant.attempt_id, None)
                grant.permit._invalidate_from_store()
                raise
            self._active_grants.pop(grant.attempt_id, None)
        return transport._dispatch_consumed(
            lambda *, timeout: provider(invocation, timeout=timeout),
            request_body=invocation.canonical_bytes,
        )

    def complete(
        self,
        context: ExtractionLockedContext,
        *,
        result: Any,
        usage: ProviderUsage | None,
        candidate_writer: Callable[
            [Session, AgentRun, PreparedExtractionPlan, BaseModel], tuple[str, str]
        ]
        | None = None,
        native_truncated: bool = False,
    ) -> None:
        self._invalidate_active_grant(context)
        plan = self._plans[context.agent_name]
        parsed: BaseModel | None = None
        parse_error = False
        try:
            parsed = parse_extraction_result(
                plan.invocation.output_schema,
                result,
                native_truncated=native_truncated,
            )
        except ExtractionCallStateError:
            parse_error = True
        with self._session_factory() as db, db.begin():
            thread, request, refs, sources, safety = self._lock_prefix(
                db, workflow_thread_id=context.workflow_thread_id
            )
            call = self._locked_call(db, context)
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == call.agent_run_id)
                .with_for_update()
            )
            if (
                run is None
                or call.status != 'claimed'
                or call.provider_attempt_count != 1
            ):
                raise ExtractionCallStateError('extraction call is not completable')
            now = self._db_now(db)
            drifted = not self._current_identity_matches(
                thread=thread,
                request=request,
                refs=refs,
                sources=sources,
                safety=safety,
                plan=plan,
            )
            drifted = drifted or bool(
                not self._runtime_key_matches(db, request)
                or
                context.owner_subject_id is None
                or thread.owner_subject_id != context.owner_subject_id
                or any(
                    ref.permission_level_snapshot
                    not in context.allowed_permission_levels
                    for ref in refs
                )
                or call.lease_expires_at is None
                or now >= call.lease_expires_at
            )
            charge = (
                plan.reserved_cost_usd
                if usage is None
                else ExtractionCallLedger._usage_cost(plan, usage)
            )
            input_tokens = (
                plan.max_input_tokens if usage is None else usage.input_tokens
            )
            output_tokens = (
                plan.max_output_tokens if usage is None else usage.output_tokens
            )
            overrun = (
                input_tokens > plan.max_input_tokens
                or output_tokens > plan.max_output_tokens
                or charge > plan.reserved_cost_usd
            )
            other_extraction = sum(
                (
                    Decimal(row.charged_cost_usd)
                    if row.status in {'completed', 'failed'}
                    else Decimal(row.reserved_cost_usd)
                )
                for row in db.scalars(
                    select(AutoReviewExtractionCall).where(
                        AutoReviewExtractionCall.workflow_thread_id
                        == context.workflow_thread_id,
                        AutoReviewExtractionCall.id != call.id,
                    )
                ).all()
            )
            validation_spend = sum(
                (
                    Decimal(row.charged_cost_usd)
                    if row.status in {'completed', 'failed'}
                    else Decimal(row.reserved_cost_usd)
                )
                for row in db.scalars(
                    select(AutoReviewValidationCall).where(
                        AutoReviewValidationCall.workflow_thread_id
                        == context.workflow_thread_id
                    )
                ).all()
            )
            overrun = overrun or (
                other_extraction + charge
                > Decimal(request.confirmed_extraction_cost_ceiling_usd)
                or other_extraction + charge + validation_spend
                > Decimal(request.confirmed_total_cost_ceiling_usd)
            )
            call.charged_input_tokens = max(input_tokens, 0)
            call.charged_output_tokens = max(output_tokens, 0)
            call.charged_cost_usd = charge
            call.budget_overrun = overrun
            call.budget_overrun_cost_usd = max(
                charge - plan.reserved_cost_usd, Decimal('0')
            )
            run.input_tokens = call.charged_input_tokens
            run.output_tokens = call.charged_output_tokens
            run.total_tokens = call.charged_input_tokens + call.charged_output_tokens
            run.estimated_cost_usd = float(charge)
            if overrun:
                self._sql_fail(call, run, now=now)
                self._open_overrun_breaker(
                    db, safety=safety['extraction'], call=call, now=now
                )
            if parse_error or drifted or thread.cancelled_at is not None or overrun:
                self._sql_fail(call, run, now=now)
                return
            assert parsed is not None
            if parsed.result_kind == 'candidate':  # type: ignore[attr-defined]
                if candidate_writer is None:
                    self._sql_fail(call, run, now=now)
                    raise ExtractionCallStateError(
                        'candidate evidence binding is missing'
                    )
                # ``begin_nested`` flushes pending state before the savepoint.
                # Keep that intermediate database-visible state terminal and
                # non-cacheable until exact relational candidate proof passes.
                self._sql_fail(call, run, now=now)
                try:
                    with db.begin_nested():
                        pair = candidate_writer(db, run, plan, parsed)
                        db.flush()
                        verified_pair = self._candidate_pair_from_rows(
                            db,
                            thread=thread,
                            request=request,
                            call=call,
                            parsed=parsed,
                        )
                        if pair != verified_pair:
                            raise ExtractionCallStateError(
                                'evidence_binding_mismatch'
                            )
                except (ExtractionCallStateError, IntegrityError):
                    run.metadata_ = {
                        **(run.metadata_ or {}),
                        'failure_reason_code': 'evidence_binding_mismatch',
                    }
                    self._sql_fail(call, run, now=now)
                    return
                items = [pair]
                result_kind = 'candidate'
            else:
                items = []
                result_kind = 'no_candidate'
            call.result_kind = result_kind
            call.result_candidate_count = len(items)
            call.result_candidate_set_hmac = self._signed_result_set(items)
            call.status = 'completed'
            call.terminal_at = now
            run.status = 'complete'
            run.completed_at = now

    def fail(
        self,
        context: ExtractionLockedContext,
        *,
        reason_code: str,
        usage: ProviderUsage | None = None,
    ) -> None:
        self._invalidate_active_grant(context)
        with self._session_factory() as db, db.begin():
            _thread, request, _refs, _sources, safety = self._lock_prefix(
                db, workflow_thread_id=context.workflow_thread_id
            )
            call = self._locked_call(db, context)
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == call.agent_run_id)
                .with_for_update()
            )
            if run is None or call.status != 'claimed':
                raise ExtractionCallStateError('extraction call is not failable')
            now = self._db_now(db)
            overrun = False
            if call.provider_attempt_count == 0:
                call.charged_input_tokens = 0
                call.charged_output_tokens = 0
                call.charged_cost_usd = Decimal('0')
            elif usage is None:
                call.charged_input_tokens = call.reserved_input_tokens
                call.charged_output_tokens = call.reserved_output_tokens
                call.charged_cost_usd = call.reserved_cost_usd
            else:
                plan = self._plans[context.agent_name]
                call.charged_input_tokens = max(usage.input_tokens, 0)
                call.charged_output_tokens = max(usage.output_tokens, 0)
                call.charged_cost_usd = ExtractionCallLedger._usage_cost(plan, usage)
                overrun = (
                    call.charged_input_tokens > plan.max_input_tokens
                    or call.charged_output_tokens > plan.max_output_tokens
                    or Decimal(call.charged_cost_usd) > plan.reserved_cost_usd
                )
            if call.provider_attempt_count == 1:
                with db.no_autoflush:
                    other_extraction = sum(
                        (
                            Decimal(row.charged_cost_usd)
                            if row.status in {'completed', 'failed'}
                            else Decimal(row.reserved_cost_usd)
                        )
                        for row in db.scalars(
                            select(AutoReviewExtractionCall).where(
                                AutoReviewExtractionCall.workflow_thread_id
                                == context.workflow_thread_id,
                                AutoReviewExtractionCall.id != call.id,
                            )
                        ).all()
                    )
                    validation_spend = sum(
                        (
                            Decimal(row.charged_cost_usd)
                            if row.status in {'completed', 'failed'}
                            else Decimal(row.reserved_cost_usd)
                        )
                        for row in db.scalars(
                            select(AutoReviewValidationCall).where(
                                AutoReviewValidationCall.workflow_thread_id
                                == context.workflow_thread_id
                            )
                        ).all()
                    )
                charge = Decimal(call.charged_cost_usd)
                overrun = overrun or (
                    other_extraction + charge
                    > Decimal(request.confirmed_extraction_cost_ceiling_usd)
                    or other_extraction + charge + validation_spend
                    > Decimal(request.confirmed_total_cost_ceiling_usd)
                )
                call.budget_overrun = overrun
                call.budget_overrun_cost_usd = max(
                    charge - Decimal(call.reserved_cost_usd), Decimal('0')
                )
            run.input_tokens = call.charged_input_tokens
            run.output_tokens = call.charged_output_tokens
            run.total_tokens = call.charged_input_tokens + call.charged_output_tokens
            run.estimated_cost_usd = float(call.charged_cost_usd)
            bounded_reason = (
                reason_code
                if reason_code
                in {
                    'provider_failure',
                    'provider_timeout',
                    'provider_response_invalid',
                    'evidence_binding_mismatch',
                    'cancelled',
                    'lease_expired',
                }
                else 'provider_failure'
            )
            run.metadata_ = {
                **(run.metadata_ or {}),
                'failure_reason_code': bounded_reason,
            }
            self._sql_fail(call, run, now=now)
            if overrun:
                self._open_overrun_breaker(
                    db, safety=safety['extraction'], call=call, now=now
                )

    def cancel(
        self,
        context: ExtractionLockedContext,
        *,
        actor_subject_id: str,
    ) -> None:
        revoke_unstarted_authority = False
        with self._session_factory() as db, db.begin():
            thread, _request, _refs, _sources, _safety = self._lock_prefix(
                db, workflow_thread_id=context.workflow_thread_id
            )
            if (
                context.owner_subject_id is None
                or actor_subject_id != context.owner_subject_id
                or thread.owner_subject_id != actor_subject_id
            ):
                raise ExtractionCallStateError('workflow owner mismatch')
            call = self._locked_call(db, context)
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == call.agent_run_id)
                .with_for_update()
            )
            if run is None or call.status != 'claimed':
                raise ExtractionCallStateError('extraction call is not cancellable')
            now = self._db_now(db)
            if thread.cancelled_at is None:
                thread.cancelled_at = now
                thread.cancelled_by_subject_id = actor_subject_id
            if call.provider_attempt_count == 0:
                revoke_unstarted_authority = True
                call.charged_input_tokens = 0
                call.charged_output_tokens = 0
                call.charged_cost_usd = Decimal('0')
                run.metadata_ = {
                    **(run.metadata_ or {}),
                    'failure_reason_code': 'cancelled',
                }
                self._sql_fail(call, run, now=now)
        if revoke_unstarted_authority:
            self._invalidate_active_grant(context)

    def recover_expired(
        self,
        context: ExtractionLockedContext,
    ) -> ExtractionLockedContext | None:
        """Reclaim an unsent claim or conservatively finalize a sent attempt."""
        refusal: str | None = None
        recovered: ExtractionLockedContext | None = None
        with self._session_factory() as db:
            with db.begin():
                thread, request, refs, sources, safety = self._lock_prefix(
                    db, workflow_thread_id=context.workflow_thread_id
                )
                call = self._locked_call(db, context)
                run = db.scalar(
                    select(AgentRun)
                    .where(AgentRun.id == call.agent_run_id)
                    .with_for_update()
                )
                if run is None or call.status != 'claimed':
                    raise ExtractionCallStateError('extraction call is not recoverable')
                now = self._db_now(db)
                if call.lease_expires_at is None or now < call.lease_expires_at:
                    raise ExtractionCallStateError('extraction lease has not expired')
                if call.provider_attempt_count == 0:
                    plan = self._plans[context.agent_name]
                    try:
                        self._ensure_claim_ready(
                            db=db,
                            thread=thread,
                            request=request,
                            refs=refs,
                            sources=sources,
                            safety=safety,
                            plan=plan,
                            actor_subject_id=context.owner_subject_id,
                            allowed_permission_levels=(
                                context.allowed_permission_levels
                            ),
                            validation_obligation=Decimal('0'),
                            prospective_reserve=Decimal('0'),
                        )
                    except ExtractionCallStateError as exc:
                        refusal = str(exc)
                        call.charged_input_tokens = 0
                        call.charged_output_tokens = 0
                        call.charged_cost_usd = Decimal('0')
                        run.metadata_ = {
                            **(run.metadata_ or {}),
                            'failure_reason_code': 'lease_expired',
                        }
                        self._sql_fail(call, run, now=now)
                    if refusal is None:
                        call.lease_token = uuid4().hex
                        call.claimed_at = now
                        call.lease_expires_at = now + timedelta(
                            seconds=call.provider_attempt_lease_seconds
                        )
                        recovered = self._context(
                            call,
                            owner_subject_id=context.owner_subject_id,
                            allowed_permission_levels=(
                                context.allowed_permission_levels
                            ),
                        )
                else:
                    call.charged_input_tokens = call.reserved_input_tokens
                    call.charged_output_tokens = call.reserved_output_tokens
                    call.charged_cost_usd = call.reserved_cost_usd
                    run.input_tokens = call.reserved_input_tokens
                    run.output_tokens = call.reserved_output_tokens
                    run.total_tokens = (
                        call.reserved_input_tokens + call.reserved_output_tokens
                    )
                    run.estimated_cost_usd = float(call.reserved_cost_usd)
                    run.metadata_ = {
                        **(run.metadata_ or {}),
                        'failure_reason_code': 'lease_expired',
                    }
                    self._sql_fail(call, run, now=now)
            if refusal is not None:
                self._invalidate_active_grant(context)
                raise ExtractionCallStateError(refusal)
            self._invalidate_active_grant(context)
            return recovered

    def _lock_prefix(
        self,
        db: Session,
        *,
        workflow_thread_id: str,
    ) -> tuple[
        AgentWorkflowThread,
        AgentWorkflowRequest,
        tuple[AgentWorkflowEvidenceRef, ...],
        dict[int, Source],
        dict[str, AutoReviewProviderSafetyState],
    ]:
        with KeyedMutationGuard.generation_barrier(db):
            runtime = KeyedMutationGuard.lock_runtime_key_state(db, for_update=False)
            if runtime is None or not runtime.ready:
                raise ExtractionCallStateError('runtime key state is unavailable')
            safety_rows = tuple(
                db.scalars(
                    select(AutoReviewProviderSafetyState)
                    .where(
                        or_(
                            and_(
                                AutoReviewProviderSafetyState.purpose == 'extraction',
                                AutoReviewProviderSafetyState.provider == 'openai',
                                AutoReviewProviderSafetyState.model
                                == 'gpt-5.4-mini-2026-03-17',
                                AutoReviewProviderSafetyState.reasoning_effort
                                == 'none',
                            ),
                            and_(
                                AutoReviewProviderSafetyState.purpose == 'validation',
                                AutoReviewProviderSafetyState.provider == 'openai',
                                AutoReviewProviderSafetyState.model == 'gpt-5.6-terra',
                                AutoReviewProviderSafetyState.reasoning_effort
                                == 'medium',
                            ),
                        )
                    )
                    .order_by(
                        AutoReviewProviderSafetyState.purpose,
                        AutoReviewProviderSafetyState.provider,
                        AutoReviewProviderSafetyState.model,
                        AutoReviewProviderSafetyState.reasoning_effort,
                    )
                    .with_for_update(read=True)
                ).all()
            )
            refs = tuple(
                db.scalars(
                    select(AgentWorkflowEvidenceRef)
                    .where(
                        AgentWorkflowEvidenceRef.workflow_thread_id
                        == workflow_thread_id
                    )
                    .order_by(AgentWorkflowEvidenceRef.canonical_row_id)
                ).all()
            )
            source_ids = sorted({ref.canonical_row_id for ref in refs})
            source_rows: tuple[Source, ...] = ()
            if source_ids:
                source_rows = tuple(
                    db.scalars(
                        select(Source)
                        .where(Source.id.in_(source_ids))
                        .order_by(Source.id)
                        .with_for_update(read=True)
                    ).all()
                )
            thread = db.scalar(
                select(AgentWorkflowThread)
                .where(AgentWorkflowThread.thread_id == workflow_thread_id)
                .with_for_update()
            )
            request = db.get(AgentWorkflowRequest, workflow_thread_id)
            if thread is None or request is None or not refs:
                raise ExtractionCallStateError('workflow evidence is incomplete')
            safety = {row.purpose: row for row in safety_rows}
            return thread, request, refs, {row.id: row for row in source_rows}, safety

    def _ensure_claim_ready(
        self,
        *,
        db: Session,
        thread: AgentWorkflowThread,
        request: AgentWorkflowRequest,
        refs: tuple[AgentWorkflowEvidenceRef, ...],
        sources: dict[int, Source],
        safety: dict[str, AutoReviewProviderSafetyState],
        plan: PreparedExtractionPlan,
        actor_subject_id: str | None,
        allowed_permission_levels: tuple[str, ...],
        validation_obligation: Decimal,
        prospective_reserve: Decimal,
    ) -> None:
        if (
            self._settings.auto_review_mode == 'disabled'
            or thread.cancelled_at is not None
        ):
            raise ExtractionCallStateError('extraction is disabled or cancelled')
        if actor_subject_id is None or thread.owner_subject_id != actor_subject_id:
            raise ExtractionCallStateError('workflow owner mismatch')
        if not allowed_permission_levels or any(
            ref.permission_level_snapshot not in allowed_permission_levels for ref in refs
        ):
            raise ExtractionCallStateError('workflow evidence permission denied')
        if not self._runtime_key_matches(db, request):
            raise ExtractionCallStateError('runtime fingerprint key changed')
        if not self._current_identity_matches(
            thread=thread,
            request=request,
            refs=refs,
            sources=sources,
            safety=safety,
            plan=plan,
        ):
            raise ExtractionCallStateError('workflow extraction identity changed')
        total_ceiling = Decimal(request.confirmed_total_cost_ceiling_usd)
        extraction_calls = tuple(
            db.scalars(
                select(AutoReviewExtractionCall).where(
                    AutoReviewExtractionCall.workflow_thread_id == thread.thread_id
                )
            ).all()
        )
        validation_calls = tuple(
            db.scalars(
                select(AutoReviewValidationCall).where(
                    AutoReviewValidationCall.workflow_thread_id == thread.thread_id
                )
            ).all()
        )
        extraction_obligation = sum(
            (
                Decimal(call.charged_cost_usd)
                if call.status in {'completed', 'failed'}
                else Decimal(call.reserved_cost_usd)
            )
            for call in extraction_calls
        )
        validation_spend = sum(
            (
                Decimal(call.charged_cost_usd)
                if call.status in {'completed', 'failed'}
                else Decimal(call.reserved_cost_usd)
            )
            for call in validation_calls
        )
        authoritative_obligation = (
            extraction_obligation
            + validation_spend
            + prospective_reserve
            + validation_obligation
        )
        if authoritative_obligation > total_ceiling:
            raise ExtractionCallStateError('signed workflow budget exceeded')

    def _current_identity_matches(
        self,
        *,
        thread: AgentWorkflowThread,
        request: AgentWorkflowRequest,
        refs: tuple[AgentWorkflowEvidenceRef, ...],
        sources: dict[int, Source],
        safety: dict[str, AutoReviewProviderSafetyState],
        plan: PreparedExtractionPlan,
        allow_cancelled: bool = False,
    ) -> bool:
        extraction = safety.get('extraction')
        validation = safety.get('validation')
        sources_match = len(sources) == len({ref.canonical_row_id for ref in refs})
        for ref in refs:
            source = sources.get(ref.canonical_row_id)
            sources_match = sources_match and bool(
                source is not None
                and source.source_type == ref.canonical_source_type
                and source.permission_level == ref.permission_level_snapshot
                and source.server_content_signature_schema == 'server-source-content:v1'
                and source.server_content_signature == ref.content_signature
            )
        return bool(
            thread.graph_version == 'company-memory-review-v2.1-auto-review'
            and (allow_cancelled or thread.cancelled_at is None)
            and refs
            and sources_match
            and request.extraction_plan_set_hmac == self._plan_set.plan_set_hmac
            and request.extraction_provider_safety_snapshot_set_hmac
            == self._plan_set.provider_safety_snapshot_set_hmac
            and request.auto_review_extraction_provider == plan.provider
            and request.auto_review_extraction_model == plan.model
            and request.auto_review_extraction_reasoning_effort == plan.reasoning_effort
            and request.auto_review_extraction_route_version == plan.route_version
            and request.auto_review_extraction_cost_policy_version
            == plan.cost_policy_version
            and request.auto_review_extraction_token_estimator_version
            == plan.token_estimator_version
            and request.auto_review_tokenizer_encoding == plan.tokenizer_encoding
            and request.auto_review_reply_priming_tokens == plan.reply_priming_tokens
            and request.auto_review_framing_safety_tokens == plan.framing_safety_tokens
            and request.auto_review_max_provider_attempts == 1
            and request.auto_review_provider_timeout_seconds
            == plan.provider_timeout_seconds
            and request.auto_review_provider_send_start_window_seconds
            == plan.provider_send_start_window_seconds
            and request.auto_review_provider_attempt_lease_seconds
            == plan.provider_attempt_lease_seconds
            and request.auto_review_provider_commit_grace_seconds
            == plan.provider_commit_grace_seconds
            and request.selected_extraction_agent_count == len(self._plan_set.plans)
            and request.extraction_max_input_chars_per_agent == plan.max_input_chars
            and request.extraction_max_input_tokens_per_agent == plan.max_input_tokens
            and request.extraction_max_output_tokens_per_agent == plan.max_output_tokens
            and request.extraction_max_candidates_per_agent == 1
            and Decimal(request.auto_review_extraction_input_usd_per_1m)
            == plan.input_usd_per_1m
            and Decimal(request.auto_review_extraction_output_usd_per_1m)
            == plan.output_usd_per_1m
            and extraction is not None
            and validation is not None
            and not extraction.breaker_open
            and not validation.breaker_open
            and extraction.state_version == plan.provider_safety_state_version
            and extraction.authorized_cost_policy_version == plan.cost_policy_version
            and extraction.token_estimator_version == plan.token_estimator_version
            and extraction.tokenizer_encoding == plan.tokenizer_encoding
            and extraction.reply_priming_tokens == plan.reply_priming_tokens
            and extraction.framing_safety_tokens == plan.framing_safety_tokens
            and Decimal(extraction.input_usd_per_1m) == plan.input_usd_per_1m
            and Decimal(extraction.output_usd_per_1m) == plan.output_usd_per_1m
            and validation.state_version
            == request.validation_provider_safety_state_version
            and validation.authorized_cost_policy_version
            == request.auto_review_cost_policy_version
            and validation.token_estimator_version
            == request.auto_review_token_estimator_version
            and validation.tokenizer_encoding == request.auto_review_tokenizer_encoding
            and validation.reply_priming_tokens
            == request.auto_review_reply_priming_tokens
            and validation.framing_safety_tokens
            == request.auto_review_framing_safety_tokens
            and Decimal(validation.input_usd_per_1m)
            == Decimal(request.auto_review_validator_input_usd_per_1m)
            and Decimal(validation.output_usd_per_1m)
            == Decimal(request.auto_review_validator_output_usd_per_1m)
        )

    @staticmethod
    def _runtime_key_matches(db: Session, request: AgentWorkflowRequest) -> bool:
        runtime = KeyedMutationGuard.lock_runtime_key_state(db, for_update=False)
        return bool(
            runtime is not None
            and runtime.ready
            and runtime.fingerprint_key_version == request.fingerprint_key_version
            and runtime.fingerprint_key_material_verifier
            == request.fingerprint_key_material_verifier
        )

    def _candidate_pair_from_rows(
        self,
        db: Session,
        *,
        thread: AgentWorkflowThread,
        request: AgentWorkflowRequest,
        call: AutoReviewExtractionCall,
        parsed: BaseModel | None = None,
    ) -> tuple[str, str]:
        items = tuple(
            db.scalars(
                select(ReviewItem).where(
                    ReviewItem.workflow_thread_id == thread.thread_id,
                    ReviewItem.agent_run_id == call.agent_run_id,
                )
            ).all()
        )
        if len(items) != 1:
            raise ExtractionCallStateError('evidence_binding_mismatch')
        item = items[0]
        if (
            item.candidate_key is None
            or item.candidate_contract_version != 'c5-v1'
            or item.agent_run_id != call.agent_run_id
        ):
            raise ExtractionCallStateError('evidence_binding_mismatch')
        children = tuple(
            db.scalars(
                select(ReviewItemEvidenceRef)
                .where(
                    ReviewItemEvidenceRef.review_item_id == item.id,
                    ReviewItemEvidenceRef.workflow_thread_id == thread.thread_id,
                )
                .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
            ).all()
        )
        if not children or tuple(
            child.candidate_slot_ordinal for child in children
        ) != tuple(range(1, len(children) + 1)):
            raise ExtractionCallStateError('evidence_binding_mismatch')
        workflow_refs = {
            row.id: row
            for row in db.scalars(
                select(AgentWorkflowEvidenceRef).where(
                    AgentWorkflowEvidenceRef.workflow_thread_id == thread.thread_id,
                    AgentWorkflowEvidenceRef.id.in_(
                        [child.workflow_evidence_ref_id for child in children]
                    ),
                )
            ).all()
        }
        if len(workflow_refs) != len(children):
            raise ExtractionCallStateError('evidence_binding_mismatch')
        bindings = tuple(
            CandidateEvidenceRefBinding(
                workflow_evidence_ref_id=child.workflow_evidence_ref_id,
                ordinal=child.candidate_slot_ordinal,
                canonical_source_kind=workflow_refs[
                    child.workflow_evidence_ref_id
                ].canonical_source_type,
                canonical_source_id=workflow_refs[
                    child.workflow_evidence_ref_id
                ].canonical_row_id,
                canonical_version_or_signature=(
                    workflow_refs[child.workflow_evidence_ref_id].external_revision
                    or workflow_refs[child.workflow_evidence_ref_id].content_signature
                ),
                content_fingerprint=workflow_refs[
                    child.workflow_evidence_ref_id
                ].content_fingerprint,
                message_set_hmac=child.message_content_fingerprint,
                permission_level=workflow_refs[
                    child.workflow_evidence_ref_id
                ].permission_level_snapshot,
                fingerprint_key_version=child.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    child.fingerprint_key_material_verifier
                ),
            )
            for child in children
        )
        if any(
            binding.fingerprint_key_version != request.fingerprint_key_version
            or binding.fingerprint_key_material_verifier
            != request.fingerprint_key_material_verifier
            for binding in bindings
        ):
            raise ExtractionCallStateError('evidence_binding_mismatch')
        plan = self._plans.get(call.agent_name)
        if plan is None:
            raise ExtractionCallStateError('evidence_binding_mismatch')
        source_rows = {
            row.id: row
            for row in db.scalars(
                select(Source).where(
                    Source.id.in_(
                        [ref.canonical_row_id for ref in workflow_refs.values()]
                    )
                )
            ).all()
        }
        slots_by_ref: dict[int, tuple[PreparedEvidenceSlotIdentity, ...]] = {}
        for ref_id, ref in workflow_refs.items():
            source = source_rows.get(ref.canonical_row_id)
            if source is None or source.source_id == '':
                raise ExtractionCallStateError('evidence_binding_mismatch')
            slots_by_ref[ref_id] = tuple(
                slot
                for slot in plan.invocation.evidence_slot_identities
                if slot.source_id == source.source_id
            )
            if not slots_by_ref[ref_id]:
                raise ExtractionCallStateError('evidence_binding_mismatch')
        selected_slot_ids: set[str] | None = None
        if parsed is not None:
            candidate = getattr(parsed, 'candidate', None)
            field_bindings = getattr(candidate, 'field_evidence_bindings', ())
            selected_slot_ids = {
                str(binding.evidence_slot_id) for binding in field_bindings
            }
            all_slot_ids = {
                slot.slot_id for slot in plan.invocation.evidence_slot_identities
            }
            if not selected_slot_ids or not selected_slot_ids <= all_slot_ids:
                raise ExtractionCallStateError('evidence_binding_mismatch')
        selected_permissions: list[str] = []
        for binding in bindings:
            possible_slots = slots_by_ref[binding.workflow_evidence_ref_id]
            if selected_slot_ids is not None:
                selected = tuple(
                    slot
                    for slot in possible_slots
                    if slot.slot_id in selected_slot_ids
                )
                candidate_sets = (selected,) if selected else ()
            else:
                candidate_sets = tuple(
                    subset
                    for count in range(1, len(possible_slots) + 1)
                    for subset in combinations(possible_slots, count)
                )
            matching_sets = tuple(
                slot_set
                for slot_set in candidate_sets
                if self._message_set_hmac(
                    ref=workflow_refs[binding.workflow_evidence_ref_id],
                    slots=slot_set,
                    request=request,
                )
                == binding.message_set_hmac
            )
            if len(matching_sets) != 1:
                raise ExtractionCallStateError('evidence_binding_mismatch')
            selected_permissions.extend(
                slot.permission_level for slot in matching_sets[0]
            )
        if (
            not selected_permissions
            or item.permission_level
            != _strictest_permission(tuple(selected_permissions))
        ):
            raise ExtractionCallStateError('evidence_binding_mismatch')
        security_scope_hmac = build_keyed_fingerprint(
            thread.security_scope_id,
            settings=self._settings,
            schema_version='candidate-security-scope:v1',
            policy_version='candidate-security-scope:v1',
        )
        evidence_hmac = derive_candidate_evidence_version_hash(
            workflow_execution_hmac=thread.input_hash,
            security_scope_hmac=security_scope_hmac,
            candidate_key=item.candidate_key,
            refs=bindings,
            settings=self._settings,
        )
        return item.candidate_key, evidence_hmac

    def _message_set_hmac(
        self,
        *,
        ref: AgentWorkflowEvidenceRef,
        slots: Sequence[PreparedEvidenceSlotIdentity],
        request: AgentWorkflowRequest,
    ) -> str:
        ordered = tuple(sorted(slots, key=lambda item: item.stable_message_identity))
        if not ordered:
            raise ExtractionCallStateError('evidence_binding_mismatch')
        permission = _strictest_permission(
            tuple(slot.permission_level for slot in ordered)
        )
        if permission != ref.permission_level_snapshot:
            raise ExtractionCallStateError('evidence_binding_mismatch')
        return build_keyed_fingerprint(
            {
                'canonical_source_kind': ref.canonical_source_type,
                'canonical_source_id': ref.canonical_row_id,
                'canonical_version_or_signature': (
                    ref.external_revision or ref.content_signature
                ),
                'content_fingerprint': ref.content_fingerprint,
                'permission_level': permission,
                'fingerprint_key_version': request.fingerprint_key_version,
                'fingerprint_key_material_verifier': (
                    request.fingerprint_key_material_verifier
                ),
                'messages': [
                    {
                        'stable_message_identity': slot.stable_message_identity,
                        'text_fingerprint': slot.text_fingerprint,
                    }
                    for slot in ordered
                ],
            },
            settings=self._settings,
            schema_version='candidate-message-set:v1',
            policy_version='candidate-message-set:v1',
        )

    def _verify_completed_replay(
        self,
        db: Session,
        *,
        thread: AgentWorkflowThread,
        request: AgentWorkflowRequest,
        call: AutoReviewExtractionCall,
    ) -> None:
        if call.status != 'completed':
            return
        run = db.get(AgentRun, call.agent_run_id)
        if run is None or run.status != 'complete':
            raise ExtractionCallStateError('completed extraction result is corrupt')
        if call.result_kind == 'no_candidate':
            item_count = db.scalar(
                select(func.count())
                .select_from(ReviewItem)
                .where(
                    ReviewItem.workflow_thread_id == thread.thread_id,
                    ReviewItem.agent_run_id == call.agent_run_id,
                )
            )
            expected = self._signed_result_set(())
            valid = (
                item_count == 0
                and call.result_candidate_count == 0
                and call.result_candidate_set_hmac == expected
            )
        elif call.result_kind == 'candidate':
            pair = self._candidate_pair_from_rows(
                db,
                thread=thread,
                request=request,
                call=call,
            )
            valid = (
                call.result_candidate_count == 1
                and call.result_candidate_set_hmac == self._signed_result_set((pair,))
            )
        else:
            valid = False
        if not valid:
            raise ExtractionCallStateError('completed extraction result is corrupt')

    def _persist_replay_binding_failure(
        self,
        *,
        workflow_thread_id: str,
        plan: PreparedExtractionPlan,
    ) -> None:
        with self._session_factory() as db, db.begin():
            self._lock_prefix(db, workflow_thread_id=workflow_thread_id)
            call = db.scalar(
                select(AutoReviewExtractionCall)
                .where(
                    AutoReviewExtractionCall.workflow_thread_id
                    == workflow_thread_id,
                    AutoReviewExtractionCall.agent_name == plan.agent_name,
                )
                .with_for_update()
            )
            if (
                call is None
                or call.status != 'completed'
                or call.extraction_plan_hmac != plan.plan_hmac
            ):
                return
            run = db.scalar(
                select(AgentRun)
                .where(AgentRun.id == call.agent_run_id)
                .with_for_update()
            )
            if run is None:
                return
            items = tuple(
                db.scalars(
                    select(ReviewItem)
                    .where(
                        ReviewItem.workflow_thread_id == workflow_thread_id,
                        ReviewItem.agent_run_id == call.agent_run_id,
                    )
                    .with_for_update()
                ).all()
            )
            if items:
                children = tuple(
                    db.scalars(
                        select(ReviewItemEvidenceRef)
                        .where(
                            ReviewItemEvidenceRef.review_item_id.in_(
                                [item.id for item in items]
                            )
                        )
                        .with_for_update()
                    ).all()
                )
                for child in children:
                    db.delete(child)
                for item in items:
                    db.delete(item)
            run.metadata_ = {
                **(run.metadata_ or {}),
                'failure_reason_code': 'evidence_binding_mismatch',
            }
            self._sql_fail(call, run, now=self._db_now(db))

    def _locked_call(
        self,
        db: Session,
        context: ExtractionLockedContext,
    ) -> AutoReviewExtractionCall:
        call = db.scalar(
            select(AutoReviewExtractionCall)
            .where(
                AutoReviewExtractionCall.workflow_thread_id
                == context.workflow_thread_id,
                AutoReviewExtractionCall.agent_name == context.agent_name,
            )
            .with_for_update()
        )
        if (
            call is None
            or call.extraction_plan_hmac != context.plan_hmac
            or call.lease_token != context.lease_token
        ):
            raise ExtractionCallStateError('extraction call context is invalid')
        return call

    def _db_now(self, db: Session) -> datetime:
        if self._db_clock_override is not None:
            return self._db_clock_override(db)
        if db.get_bind().dialect.name == 'postgresql':
            value = db.scalar(select(func.clock_timestamp()))
            if value is None:
                raise ExtractionCallStateError('database clock is unavailable')
            return value
        return datetime.now(UTC)

    def _invalidate_active_grant(
        self,
        context: ExtractionLockedContext,
    ) -> None:
        with self._grant_lock:
            attempt_ids = [
                attempt_id
                for attempt_id, active in self._active_grants.items()
                if active[1].workflow_thread_id == context.workflow_thread_id
                and active[1].agent_name == context.agent_name
                and active[1].lease_token == context.lease_token
            ]
            for attempt_id in attempt_ids:
                active = self._active_grants.pop(attempt_id)
                active[0].permit._invalidate_from_store()

    def _require_signed_plan(self, plan: PreparedExtractionPlan) -> None:
        current = self._plans.get(plan.agent_name)
        if current is None or current != plan:
            raise ExtractionCallStateError('plan is not in the signed extraction set')

    @staticmethod
    def _verify_existing_plan(
        call: AutoReviewExtractionCall,
        plan: PreparedExtractionPlan,
    ) -> None:
        if call.extraction_plan_hmac != plan.plan_hmac:
            raise ExtractionCallStateError('workflow agent already owns another plan')

    @staticmethod
    def _context(
        call: AutoReviewExtractionCall,
        *,
        owner_subject_id: str | None = None,
        allowed_permission_levels: tuple[str, ...] = (),
    ) -> ExtractionLockedContext:
        if call.lease_token is None:
            raise ExtractionCallStateError('extraction call lease is missing')
        return ExtractionLockedContext(
            workflow_thread_id=call.workflow_thread_id,
            agent_name=call.agent_name,
            plan_hmac=call.extraction_plan_hmac,
            lease_token=call.lease_token,
            owner_subject_id=owner_subject_id,
            allowed_permission_levels=allowed_permission_levels,
        )

    @staticmethod
    def _verify_call_identity(
        call: AutoReviewExtractionCall,
        plan: PreparedExtractionPlan,
    ) -> None:
        if (
            call.prepared_content_hmac != plan.invocation.prepared_content_hmac
            or call.prepared_character_count != plan.invocation.character_count
            or call.framed_input_token_cap != plan.invocation.framed_input_tokens
            or call.total_output_token_cap != plan.max_output_tokens
            or call.provider_timeout_seconds != plan.provider_timeout_seconds
            or call.provider_send_start_window_seconds
            != plan.provider_send_start_window_seconds
            or call.provider_attempt_lease_seconds
            != plan.provider_attempt_lease_seconds
            or call.provider_commit_grace_seconds != plan.provider_commit_grace_seconds
        ):
            raise ExtractionCallStateError('prepared extraction invocation changed')

    @staticmethod
    def _sql_fail(
        call: AutoReviewExtractionCall,
        run: AgentRun,
        *,
        now: datetime,
    ) -> None:
        call.status = 'failed'
        call.terminal_at = now
        call.result_kind = None
        call.result_candidate_count = None
        call.result_candidate_set_hmac = None
        run.status = 'failed'
        run.completed_at = now

    def _signed_result_set(self, items: Sequence[tuple[str, str]]) -> str:
        return build_keyed_fingerprint(
            {'items': [list(item) for item in sorted(items)]},
            settings=self._settings,
            schema_version=EXTRACTION_RESULT_SET_SCHEMA,
            policy_version=EXTRACTION_RESULT_SET_SCHEMA,
        )

    def _open_overrun_breaker(
        self,
        db: Session,
        *,
        safety: AutoReviewProviderSafetyState,
        call: AutoReviewExtractionCall,
        now: datetime,
    ) -> None:
        prior_version = safety.state_version
        event = AutoReviewProviderSafetyEvent(
            provider_safety_state_id=safety.id,
            purpose=safety.purpose,
            provider=safety.provider,
            model=safety.model,
            reasoning_effort=safety.reasoning_effort,
            event_sequence=safety.last_event_sequence + 1,
            event_kind='budget_overrun',
            prior_state_version=prior_version,
            new_state_version=prior_version + 1,
            cost_policy_version=safety.authorized_cost_policy_version,
            token_estimator_version=safety.token_estimator_version,
            tokenizer_encoding=safety.tokenizer_encoding,
            reply_priming_tokens=safety.reply_priming_tokens,
            framing_safety_tokens=safety.framing_safety_tokens,
            input_usd_per_1m=safety.input_usd_per_1m,
            output_usd_per_1m=safety.output_usd_per_1m,
            prior_breaker_open=safety.breaker_open,
            new_breaker_open=True,
            reason_code='budget_overrun',
            regression_gate_reference=safety.regression_gate_reference,
            actor_subject_hmac=None,
            call_hmac=build_keyed_fingerprint(
                {'extraction_call_id': call.id},
                settings=self._settings,
                schema_version='auto-review-extraction-call:v1',
                policy_version='auto-review-extraction-call:v1',
            ),
            fingerprint_key_version=call.fingerprint_key_version,
            fingerprint_key_material_verifier=(call.fingerprint_key_material_verifier),
            created_at=now,
        )
        db.add(event)
        db.flush()
        safety.state_version = prior_version + 1
        safety.breaker_open = True
        safety.breaker_reason_code = 'budget_overrun'
        safety.overrun_count += 1
        safety.last_overrun_cost_usd = call.charged_cost_usd
        safety.last_overrun_at = now
        safety.updated_at = now
        safety.last_event_sequence = event.event_sequence
        safety.last_event_id = event.id


def _strictest_permission(levels: tuple[str, ...]) -> str:
    rank = {'public': 0, 'internal': 1, 'restricted': 2}
    return max(levels, key=lambda value: rank.get(value, 2))
