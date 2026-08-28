from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import (
    fingerprint_key_material_verifier as build_key_material_verifier,
)
from backend.app.agent_runtime.canonical_sources import (
    ResolvedSourceVersion,
    ReviewWorkflowPreflightError,
    build_keyed_fingerprint,
    resolve_source_versions,
)
from backend.app.agent_runtime.contracts import (
    AgentCostBudgetDecision,
    AgentRunResult,
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
    ReviewCandidate,
)
from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.review_v2_agents import (
    ReviewAgentAdapter,
    ReviewAgentCatalog,
)
from backend.app.agent_runtime.review_v2_preflight import (
    PreparedReviewRequest,
    build_prepared_review_identity,
    find_matching_review_thread,
    review_thread_matches_prepared,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_versions import SourceVersionRef
from backend.app.models.agent_runs import AgentRun
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.auto_review import ReviewItemEvidenceRef
from backend.app.models.review import ReviewItem
from backend.app.models.source import (
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    ReviewItemResolutionStatus,
    ReviewWorkflowDryRunResponse,
    normalize_agent_names,
)

EFFECT_KEY_SCHEMA = 'review-agent-effect:v1'
EFFECT_KEY_POLICY = 'review-agent-effect-key:v1'
PERMISSION_FINGERPRINT_SCHEMA = 'review-permission-levels:v1'
PERMISSION_FINGERPRINT_POLICY = 'review-permission-fingerprint:v1'
PACKET_FINGERPRINT_SCHEMA = 'review-evidence-packet:v1'
PACKET_FINGERPRINT_POLICY = 'review-evidence-selection:v1'
CANDIDATE_KEY_SCHEMA = 'review-agent-candidate:v1'
CANDIDATE_KEY_POLICY = 'review-agent-candidate-key:v1'
CANDIDATE_CONTRACT_VERSION = 'c5-v1'
CANDIDATE_MESSAGE_SET_SCHEMA = 'candidate-message-set:v1'
CANDIDATE_MESSAGE_SET_POLICY = 'candidate-message-set:v1'
CANDIDATE_EVIDENCE_STATE_SCHEMA = 'candidate-evidence-state:v1'
CANDIDATE_EVIDENCE_STATE_POLICY = 'candidate-evidence-state:v1'
CANDIDATE_GENERATION_SCHEMA = 'candidate-generation:v1'
CANDIDATE_GENERATION_POLICY = 'candidate-generation:v1'
DEFAULT_LEASE_TTL_SECONDS = 120
MAX_PROVIDER_ROUTES = 3
MAX_ATTEMPTS_PER_PROVIDER = 2
LEASE_GRACE_SECONDS = 30
REVIEW_STATUSES: tuple[ReviewItemResolutionStatus, ...] = (
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
)
_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}
_DRAFTABLE_STATUSES = frozenset(
    {
        'created',
        'drafting',
        'checkpoint_pending',
        'checkpoint_failed',
    }
)


class ReviewDraftError(ReviewWorkflowPreflightError):
    pass


class CandidateEvidenceBindingError(ValueError):
    pass


@dataclass(frozen=True)
class GenerationIdentity:
    agent_name: str
    provider: str | None
    model: str
    reasoning_effort: str | None
    prompt_version: str
    route_version: str | None
    output_contract_version: str | None


@dataclass(frozen=True)
class CandidateEvidenceRefBinding:
    workflow_evidence_ref_id: int
    ordinal: int
    canonical_source_kind: str
    canonical_source_id: int | str
    canonical_version_or_signature: str
    content_fingerprint: str
    message_set_hmac: str
    permission_level: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str


@dataclass(frozen=True)
class _CandidateParentRow:
    candidate_contract_version: str


@dataclass(frozen=True)
class CandidateEvidenceBindingSet:
    refs: tuple[CandidateEvidenceRefBinding, ...]
    evidence_version_hash: str
    candidate_contract_version: str = CANDIDATE_CONTRACT_VERSION

    def assert_same_workflow_agent_run(
        self,
        *,
        workflow_thread_id: str,
        agent_run_workflow_thread_id: str,
    ) -> None:
        if workflow_thread_id != agent_run_workflow_thread_id:
            raise CandidateEvidenceBindingError(
                'agent run must belong to the same workflow'
            )

    def verify_replay(
        self,
        *,
        agent_run_id: int,
        stored_agent_run_id: int | None,
        stored_refs: tuple[int, ...],
        stored_bindings: tuple[CandidateEvidenceRefBinding, ...] | None = None,
    ) -> None:
        expected = tuple(row.workflow_evidence_ref_id for row in self.refs)
        if (
            stored_agent_run_id != agent_run_id
            or stored_refs != expected
            or (stored_bindings is not None and stored_bindings != self.refs)
        ):
            raise CandidateEvidenceBindingError(
                'candidate evidence binding is immutable'
            )

    def is_auto_review_eligible(self, graph_version: str) -> bool:
        return graph_version == 'company-memory-review-v2.1-auto-review'

    def atomic_rows(
        self,
    ) -> tuple[_CandidateParentRow, tuple[CandidateEvidenceRefBinding, ...]]:
        return _CandidateParentRow(self.candidate_contract_version), self.refs


def build_candidate_generation_fingerprint(
    identity: GenerationIdentity,
    *,
    settings: Settings,
    fingerprint_key_material_verifier: str,
) -> str:
    _, key_version = fingerprint_secret_bytes(settings)
    return build_keyed_fingerprint(
        {
            'agent_name': identity.agent_name,
            'generation_provider': identity.provider,
            'model_name': identity.model,
            'generation_reasoning_effort': identity.reasoning_effort,
            'prompt_version': identity.prompt_version,
            'generation_route_version': identity.route_version,
            'generation_output_contract_version': identity.output_contract_version,
            'fingerprint_key_version': key_version,
            'fingerprint_key_material_verifier': fingerprint_key_material_verifier,
        },
        settings=settings,
        schema_version=CANDIDATE_GENERATION_SCHEMA,
        policy_version=CANDIDATE_GENERATION_POLICY,
    )


def generation_identity_is_auto_eligible(
    identity: GenerationIdentity,
    *,
    validator_provider: str,
    validator_model: str,
) -> bool:
    return bool(
        identity.provider in {'openai', 'azure_openai', 'gemini'}
        and identity.reasoning_effort == 'none'
        and identity.route_version
        and identity.output_contract_version
        and (identity.provider, identity.model) != (validator_provider, validator_model)
    )


def build_candidate_evidence_bindings(
    *,
    candidate: ReviewCandidate,
    packet: EvidencePacket,
    workflow_execution_hmac: str,
    security_scope_hmac: str,
    candidate_key: str,
    settings: Settings,
    fingerprint_key_material_verifier: str,
    workflow_thread_id: str | None = None,
    on_persist: Callable[[CandidateEvidenceBindingSet], None] | None = None,
) -> CandidateEvidenceBindingSet:
    if len(candidate.source_links) != len(candidate.source_snippets):
        raise CandidateEvidenceBindingError('candidate evidence mapping is ambiguous')
    selected: list[EvidenceMessage] = []
    for link, snippet in zip(
        candidate.source_links, candidate.source_snippets, strict=True
    ):
        matches = [
            message
            for message in packet.messages
            if message.source_url == link and message.source_snippet == snippet
        ]
        if len(matches) != 1:
            raise CandidateEvidenceBindingError(
                'candidate evidence mapping is ambiguous'
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    if not selected:
        raise CandidateEvidenceBindingError('candidate evidence mapping is empty')
    secret, key_version = fingerprint_secret_bytes(settings)
    groups: dict[int, list[EvidenceMessage]] = {}
    for message in selected:
        metadata = message.metadata
        ref_id = metadata.get('workflow_evidence_ref_id')
        if not isinstance(ref_id, int):
            raise CandidateEvidenceBindingError(
                'candidate evidence lacks an exact workflow ref'
            )
        message_workflow = metadata.get('workflow_thread_id')
        if (
            workflow_thread_id
            and message_workflow
            and message_workflow != workflow_thread_id
        ):
            raise CandidateEvidenceBindingError(
                'candidate evidence ref must belong to the same workflow'
            )
        groups.setdefault(ref_id, []).append(message)
    rows: list[CandidateEvidenceRefBinding] = []
    for ref_id, messages in groups.items():
        ordered = sorted(
            messages,
            key=lambda item: str(item.metadata.get('stable_message_identity') or ''),
        )
        first = ordered[0]
        metadata = first.metadata
        required = (
            'canonical_source_kind',
            'canonical_source_id',
            'canonical_version_or_signature',
            'content_fingerprint',
        )
        if any(metadata.get(key) in {None, ''} for key in required):
            raise CandidateEvidenceBindingError('candidate evidence ref is incomplete')
        common = tuple(metadata.get(key) for key in required)
        if any(
            tuple(message.metadata.get(key) for key in required) != common
            for message in ordered
        ):
            raise CandidateEvidenceBindingError('candidate evidence ref is ambiguous')
        permission_level = _strictest_permission(
            *(message.permission_level for message in ordered)
        )
        message_set_hmac = build_keyed_fingerprint(
            {
                'canonical_source_kind': common[0],
                'canonical_source_id': common[1],
                'canonical_version_or_signature': common[2],
                'content_fingerprint': common[3],
                'permission_level': permission_level,
                'fingerprint_key_version': key_version,
                'fingerprint_key_material_verifier': (
                    fingerprint_key_material_verifier
                ),
                'messages': [
                    {
                        'stable_message_identity': message.metadata.get(
                            'stable_message_identity'
                        ),
                        'text_fingerprint': build_keyed_fingerprint(
                            message.text,
                            settings=settings,
                            schema_version='candidate-message-content:v1',
                            policy_version='candidate-message-content:v1',
                        ),
                    }
                    for message in ordered
                ],
            },
            settings=settings,
            schema_version=CANDIDATE_MESSAGE_SET_SCHEMA,
            policy_version=CANDIDATE_MESSAGE_SET_POLICY,
        )
        rows.append(
            CandidateEvidenceRefBinding(
                workflow_evidence_ref_id=ref_id,
                ordinal=0,
                canonical_source_kind=str(common[0]),
                canonical_source_id=cast(int | str, common[1]),
                canonical_version_or_signature=str(common[2]),
                content_fingerprint=str(common[3]),
                message_set_hmac=message_set_hmac,
                permission_level=permission_level,
                fingerprint_key_version=key_version,
                fingerprint_key_material_verifier=fingerprint_key_material_verifier,
            )
        )
    ordered_rows = tuple(
        replace(row, ordinal=index)
        for index, row in enumerate(
            sorted(
                rows,
                key=lambda row: (
                    row.workflow_evidence_ref_id,
                    row.canonical_source_kind,
                    str(row.canonical_source_id),
                ),
            ),
            start=1,
        )
    )
    evidence_version_hash = derive_candidate_evidence_version_hash(
        workflow_execution_hmac=workflow_execution_hmac,
        security_scope_hmac=security_scope_hmac,
        candidate_key=candidate_key,
        refs=ordered_rows,
        settings=settings,
    )
    result = CandidateEvidenceBindingSet(ordered_rows, evidence_version_hash)
    if on_persist is not None:
        on_persist(result)
    del secret
    return result


def derive_candidate_evidence_version_hash(
    *,
    workflow_execution_hmac: str,
    security_scope_hmac: str,
    candidate_key: str,
    refs: Sequence[CandidateEvidenceRefBinding],
    settings: Settings,
) -> str:
    """Re-derive the immutable candidate evidence identity from relational rows."""
    ordered_rows = tuple(
        sorted(
            refs,
            key=lambda row: (
                row.workflow_evidence_ref_id,
                row.canonical_source_kind,
                str(row.canonical_source_id),
            ),
        )
    )
    return build_keyed_fingerprint(
        {
            'workflow_execution_hmac': workflow_execution_hmac,
            'security_scope_hmac': security_scope_hmac,
            'candidate_key': candidate_key,
            'refs': [
                {
                    'workflow_evidence_ref_id': row.workflow_evidence_ref_id,
                    'canonical_source_kind': row.canonical_source_kind,
                    'canonical_source_id': row.canonical_source_id,
                    'canonical_version_or_signature': row.canonical_version_or_signature,
                    'content_fingerprint': row.content_fingerprint,
                    'aggregate_message_set_hmac': row.message_set_hmac,
                    'permission_level': row.permission_level,
                    'fingerprint_key_version': row.fingerprint_key_version,
                    'fingerprint_key_material_verifier': row.fingerprint_key_material_verifier,
                }
                for row in ordered_rows
            ],
        },
        settings=settings,
        schema_version=CANDIDATE_EVIDENCE_STATE_SCHEMA,
        policy_version=CANDIDATE_EVIDENCE_STATE_POLICY,
    )


@dataclass(frozen=True)
class ReviewDraftResult:
    review_item_ids: tuple[int, ...]
    review_status_counts: dict[ReviewItemResolutionStatus, int]


@dataclass(frozen=True)
class _DraftInputs:
    thread_id: str
    security_scope_id: str
    status: str
    state_version: int
    evidence_version_hash: str
    selection_policy_version: str
    agent_names: tuple[str, ...]
    refs: tuple[ResolvedSourceVersion, ...]
    prepared: PreparedReviewRequest


@dataclass(frozen=True)
class _AdapterPlan:
    adapter: ReviewAgentAdapter
    effect_key: str
    decision: AgentCostBudgetDecision
    cached_run_id: int | None


@dataclass(frozen=True)
class _Lease:
    token: str
    expires_at: datetime
    state_version: int
    packet_hash: str
    evidence_version_hash: str
    permission_fingerprint: str
    selection_policy_version: str
    agent_names: tuple[str, ...]
    prior_status: str
    plans: tuple[_AdapterPlan, ...]
    cached_review_item_ids: tuple[int, ...]


@dataclass
class _LeaseRenewalClaims:
    prior: _Lease
    prospective: _Lease | None = None

    @property
    def cleanup_order(self) -> tuple[_Lease, ...]:
        if self.prospective is None:
            return (self.prior,)
        return (self.prospective, self.prior)


class ReviewDraftService:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        catalog: ReviewAgentCatalog,
        settings: Settings,
        lease_ttl_seconds: int | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if lease_ttl_seconds is None:
            lease_ttl_seconds = max(
                DEFAULT_LEASE_TTL_SECONDS,
                ceil(
                    settings.agent_llm_timeout_seconds
                    * MAX_PROVIDER_ROUTES
                    * MAX_ATTEMPTS_PER_PROVIDER
                )
                + LEASE_GRACE_SECONDS,
            )
        if lease_ttl_seconds <= 0:
            raise ValueError('lease ttl must be positive')
        self._session_factory = session_factory
        self._catalog = catalog
        self._settings = settings
        self._lease_ttl = timedelta(seconds=lease_ttl_seconds)
        self._now = now or (lambda: datetime.now(UTC))

    def preview_prepared(
        self,
        *,
        prepared: PreparedReviewRequest,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewWorkflowDryRunResponse:
        permission_levels = _normalize_permission_levels(allowed_permission_levels)
        with self._session_factory() as db:
            packet = _build_exact_packet(
                db,
                refs=prepared.source_refs,
                workflow_thread_id=None,
                actor_subject_id=actor_subject_id,
                allowed_permission_levels=permission_levels,
                settings=self._settings,
            )
            packet_hash = _packet_fingerprint(packet, settings=self._settings)
            permission_fingerprint = build_permission_fingerprint(
                settings=self._settings,
                allowed_permission_levels=permission_levels,
            )
            matching_thread = find_matching_review_thread(
                db,
                prepared=prepared,
                actor=_actor(actor_subject_id, permission_levels),
                settings=self._settings,
            )
            plans = _build_adapter_plans(
                db,
                catalog=self._catalog,
                thread_id=matching_thread.thread_id if matching_thread else None,
                agent_names=prepared.agent_names,
                evidence_hash=packet_hash,
                permission_fingerprint=permission_fingerprint,
                selection_policy_version=prepared.selection_policy_version,
                packet=packet,
                settings=self._settings,
            )
            response = _aggregate_preview(
                prepared=prepared,
                plans=plans,
                source_count=len(prepared.source_refs),
                budget_limit_usd=self._settings.agent_llm_max_estimated_cost_usd,
            )
            db.rollback()
            return response

    def draft(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        allowed_permission_levels: Sequence[str],
    ) -> ReviewDraftResult:
        permission_levels = _normalize_permission_levels(allowed_permission_levels)
        preview, cached_ids = self._preview_thread(
            workflow_thread_id=workflow_thread_id,
            actor_subject_id=actor_subject_id,
            permission_levels=permission_levels,
        )
        if preview.budget_status == 'over_budget':
            raise ReviewDraftError('budget_exceeded', 'review budget exceeded')
        if preview.budget_status == 'no_input':
            return _draft_result((), ())
        if preview.budget_status == 'cached':
            return self._result_for_ids(cached_ids)

        lease = self._claim_lease(
            workflow_thread_id=workflow_thread_id,
            actor_subject_id=actor_subject_id,
            permission_levels=permission_levels,
        )
        if not lease.plans:
            return self._result_for_ids(lease.cached_review_item_ids)

        try:
            with self._session_factory() as db:
                inputs = _load_draft_inputs(
                    db,
                    workflow_thread_id,
                    lock=False,
                    settings=self._settings,
                )
                _ensure_thread_scope(inputs, self._settings)
                packet = _build_exact_packet(
                    db,
                    refs=inputs.refs,
                    workflow_thread_id=inputs.thread_id,
                    actor_subject_id=actor_subject_id,
                    allowed_permission_levels=permission_levels,
                    settings=self._settings,
                )
                current_packet_hash = _packet_fingerprint(
                    packet,
                    settings=self._settings,
                )
                db.rollback()
            if current_packet_hash != lease.packet_hash:
                raise ReviewDraftError('evidence_changed', 'source evidence changed')
        except ReviewDraftError:
            self._release_lease_after_failure(workflow_thread_id, lease)
            raise
        except Exception:
            self._release_lease_after_failure(workflow_thread_id, lease)
            raise

        results: list[tuple[_AdapterPlan, AgentRunResult]] = []
        for index, plan in enumerate(lease.plans):
            try:
                result = _validated_result(
                    plan.adapter.run(packet),
                    plan=plan,
                    packet=packet,
                )
            except ReviewDraftError:
                self._release_lease_after_failure(workflow_thread_id, lease)
                raise
            except Exception:
                self._release_lease_after_failure(workflow_thread_id, lease)
                raise ReviewDraftError(
                    'model_unavailable',
                    'review model is unavailable',
                ) from None
            results.append((plan, result))
            if index + 1 < len(lease.plans):
                renewal_claims = _LeaseRenewalClaims(prior=lease)
                try:
                    lease = self._renew_lease(
                        workflow_thread_id,
                        lease,
                        claims=renewal_claims,
                    )
                except Exception:
                    self._release_lease_claims_after_failure(
                        workflow_thread_id,
                        renewal_claims.cleanup_order,
                    )
                    raise

        try:
            return self._persist_results(
                workflow_thread_id=workflow_thread_id,
                actor_subject_id=actor_subject_id,
                permission_levels=permission_levels,
                lease=lease,
                results=tuple(results),
            )
        except ReviewDraftError:
            self._release_lease_after_failure(workflow_thread_id, lease)
            raise
        except Exception:
            self._release_lease_after_failure(workflow_thread_id, lease)
            raise

    def _preview_thread(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        permission_levels: tuple[str, ...],
    ) -> tuple[ReviewWorkflowDryRunResponse, tuple[int, ...]]:
        with self._session_factory() as db:
            inputs = _load_draft_inputs(
                db,
                workflow_thread_id,
                lock=False,
                settings=self._settings,
            )
            _ensure_thread_scope(inputs, self._settings)
            _ensure_thread_state(inputs.status)
            packet = _build_exact_packet(
                db,
                refs=inputs.refs,
                workflow_thread_id=inputs.thread_id,
                actor_subject_id=actor_subject_id,
                allowed_permission_levels=permission_levels,
                settings=self._settings,
            )
            permission_fingerprint = build_permission_fingerprint(
                settings=self._settings,
                allowed_permission_levels=permission_levels,
            )
            plans = _build_adapter_plans(
                db,
                catalog=self._catalog,
                thread_id=inputs.thread_id,
                agent_names=inputs.agent_names,
                evidence_hash=_packet_fingerprint(packet, settings=self._settings),
                permission_fingerprint=permission_fingerprint,
                selection_policy_version=inputs.selection_policy_version,
                packet=packet,
                settings=self._settings,
            )
            response = _aggregate_preview(
                prepared=inputs.prepared,
                plans=plans,
                source_count=len(inputs.refs),
                budget_limit_usd=self._settings.agent_llm_max_estimated_cost_usd,
            )
            cached_ids = _cached_review_item_ids(
                db, plans, inputs.thread_id, settings=self._settings
            )
            db.rollback()
            return response, cached_ids

    def _claim_lease(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        permission_levels: tuple[str, ...],
    ) -> _Lease:
        with self._session_factory() as db:
            thread = _locked_thread(db, workflow_thread_id)
            inputs = _draft_inputs_for_thread(db, thread, settings=self._settings)
            _ensure_thread_scope(inputs, self._settings)
            _ensure_thread_state(inputs.status)
            now = self._now()
            if thread.cancelled_at is not None or thread.status == 'cancelled':
                raise ReviewDraftError(
                    'invalid_state_transition', 'workflow is cancelled'
                )
            if (
                thread.lease_token
                and thread.lease_expires_at is not None
                and not _lease_expired(thread.lease_expires_at, now)
            ):
                raise ReviewDraftError(
                    'concurrent_resume', 'workflow is already drafting'
                )
            prior_status = _lease_restore_status(thread.status)

            packet = _build_exact_packet(
                db,
                refs=inputs.refs,
                workflow_thread_id=inputs.thread_id,
                actor_subject_id=actor_subject_id,
                allowed_permission_levels=permission_levels,
                settings=self._settings,
            )
            packet_hash = _packet_fingerprint(packet, settings=self._settings)
            permission_fingerprint = build_permission_fingerprint(
                settings=self._settings,
                allowed_permission_levels=permission_levels,
            )
            plans = _build_adapter_plans(
                db,
                catalog=self._catalog,
                thread_id=inputs.thread_id,
                agent_names=inputs.agent_names,
                evidence_hash=packet_hash,
                permission_fingerprint=permission_fingerprint,
                selection_policy_version=inputs.selection_policy_version,
                packet=packet,
                settings=self._settings,
            )
            preview = _aggregate_preview(
                prepared=inputs.prepared,
                plans=plans,
                source_count=len(inputs.refs),
                budget_limit_usd=self._settings.agent_llm_max_estimated_cost_usd,
            )
            if preview.budget_status == 'over_budget':
                db.rollback()
                raise ReviewDraftError('budget_exceeded', 'review budget exceeded')
            cached_ids = _cached_review_item_ids(
                db, plans, inputs.thread_id, settings=self._settings
            )
            pending_plans = tuple(
                plan
                for plan in plans
                if plan.cached_run_id is None
                and plan.decision.budget_status != 'no_input'
            )
            if not pending_plans:
                db.rollback()
                return _Lease(
                    token='',
                    expires_at=now,
                    state_version=thread.state_version,
                    packet_hash=packet_hash,
                    evidence_version_hash=inputs.evidence_version_hash,
                    permission_fingerprint=permission_fingerprint,
                    selection_policy_version=inputs.selection_policy_version,
                    agent_names=inputs.agent_names,
                    prior_status=prior_status,
                    plans=(),
                    cached_review_item_ids=cached_ids,
                )

            token = uuid4().hex
            expires_at = now + self._lease_ttl
            thread.lease_token = token
            thread.lease_expires_at = expires_at
            thread.status = 'drafting'
            thread.state_version += 1
            state_version = thread.state_version
            db.commit()
            return _Lease(
                token=token,
                expires_at=expires_at,
                state_version=state_version,
                packet_hash=packet_hash,
                evidence_version_hash=inputs.evidence_version_hash,
                permission_fingerprint=permission_fingerprint,
                selection_policy_version=inputs.selection_policy_version,
                agent_names=inputs.agent_names,
                prior_status=prior_status,
                plans=pending_plans,
                cached_review_item_ids=cached_ids,
            )

    def _persist_results(
        self,
        *,
        workflow_thread_id: str,
        actor_subject_id: str,
        permission_levels: tuple[str, ...],
        lease: _Lease,
        results: tuple[tuple[_AdapterPlan, AgentRunResult], ...],
    ) -> ReviewDraftResult:
        with self._session_factory() as db:
            thread = _locked_thread(db, workflow_thread_id)
            now = self._now()
            if (
                thread.lease_token != lease.token
                or thread.state_version != lease.state_version
                or thread.status != 'drafting'
                or thread.lease_expires_at is None
                or not _same_datetime(thread.lease_expires_at, lease.expires_at)
                or _lease_expired(thread.lease_expires_at, now)
            ):
                db.rollback()
                raise ReviewDraftError(
                    'concurrent_resume', 'draft lease is no longer valid'
                )
            if thread.cancelled_at is not None:
                db.rollback()
                raise ReviewDraftError(
                    'invalid_state_transition', 'workflow is cancelled'
                )

            inputs = _draft_inputs_for_thread(db, thread, settings=self._settings)
            _ensure_thread_scope(inputs, self._settings)
            if inputs.evidence_version_hash != lease.evidence_version_hash:
                db.rollback()
                raise ReviewDraftError('evidence_changed', 'source evidence changed')
            if (
                inputs.selection_policy_version != lease.selection_policy_version
                or inputs.agent_names != lease.agent_names
            ):
                db.rollback()
                raise ReviewDraftError(
                    'invalid_state_transition',
                    'workflow request changed during drafting',
                )
            packet = _build_exact_packet(
                db,
                refs=inputs.refs,
                workflow_thread_id=inputs.thread_id,
                actor_subject_id=actor_subject_id,
                allowed_permission_levels=permission_levels,
                settings=self._settings,
            )
            packet_hash = _packet_fingerprint(packet, settings=self._settings)
            if packet_hash != lease.packet_hash:
                db.rollback()
                raise ReviewDraftError('evidence_changed', 'source evidence changed')
            permission_fingerprint = build_permission_fingerprint(
                settings=self._settings,
                allowed_permission_levels=permission_levels,
            )
            if permission_fingerprint != lease.permission_fingerprint:
                db.rollback()
                raise ReviewDraftError(
                    'permission_denied', 'permission context changed'
                )

            item_ids = list(lease.cached_review_item_ids)
            for plan, result in results:
                existing = _find_agent_run(db, workflow_thread_id, plan.effect_key)
                if existing is not None:
                    item_ids.extend(
                        _review_item_ids_for_run(
                            db,
                            existing.id,
                            workflow_thread_id,
                            settings=self._settings,
                        )
                    )
                    continue
                agent_run, created = _insert_or_get_agent_run(
                    db,
                    thread=thread,
                    plan=plan,
                    result=result,
                    packet=packet,
                    packet_hash=packet_hash,
                    permission_fingerprint=permission_fingerprint,
                    selection_policy_version=inputs.selection_policy_version,
                    now=now,
                )
                if not created:
                    item_ids.extend(
                        _review_item_ids_for_run(
                            db,
                            agent_run.id,
                            workflow_thread_id,
                            settings=self._settings,
                        )
                    )
                    continue
                try:
                    with db.begin_nested():
                        for candidate in result.candidates:
                            item, _ = _insert_or_get_review_item(
                                db,
                                thread=thread,
                                agent_run=agent_run,
                                plan=plan,
                                result=result,
                                candidate=candidate,
                                packet=packet,
                                settings=self._settings,
                                now=now,
                            )
                            item_ids.append(item.id)
                except ReviewDraftError as exc:
                    if exc.code != 'evidence_binding_mismatch':
                        raise
                    agent_run.status = 'failed'
                    agent_run.completed_at = now
                    agent_run.metadata_ = {
                        **(agent_run.metadata_ or {}),
                        'failure_reason_code': 'evidence_binding_mismatch',
                    }
                    thread.status = lease.prior_status
                    thread.lease_token = None
                    thread.lease_expires_at = None
                    thread.state_version += 1
                    db.commit()
                    raise

            thread.status = 'checkpoint_pending'
            thread.lease_token = None
            thread.lease_expires_at = None
            thread.state_version += 1
            result_value = _draft_result_from_db(
                db,
                tuple(sorted(set(item_ids))),
            )
            db.commit()
            return result_value

    def _renew_lease(
        self,
        workflow_thread_id: str,
        lease: _Lease,
        *,
        claims: _LeaseRenewalClaims,
    ) -> _Lease:
        with self._session_factory() as db:
            thread = _locked_thread(db, workflow_thread_id)
            now = self._now()
            if not _thread_has_exact_lease(thread, lease) or _lease_expired(
                lease.expires_at,
                now,
            ):
                db.rollback()
                raise ReviewDraftError(
                    'concurrent_resume',
                    'draft lease is no longer valid',
                )
            if thread.cancelled_at is not None:
                db.rollback()
                raise ReviewDraftError(
                    'invalid_state_transition',
                    'workflow is cancelled',
                )
            expires_at = now + self._lease_ttl
            prospective = replace(
                lease,
                expires_at=expires_at,
                state_version=thread.state_version + 1,
            )
            claims.prospective = prospective
            thread.lease_expires_at = expires_at
            thread.state_version += 1
            db.commit()
            return prospective

    def _release_lease_after_failure(
        self,
        workflow_thread_id: str,
        lease: _Lease,
    ) -> None:
        self._release_lease_claims_after_failure(workflow_thread_id, (lease,))

    def _release_lease_claims_after_failure(
        self,
        workflow_thread_id: str,
        leases: Sequence[_Lease],
    ) -> None:
        for lease in leases:
            try:
                self._release_lease(workflow_thread_id, lease)
            except Exception:
                # Cleanup must never replace the original provider,
                # renewal, validation, serialization, or persistence error.
                continue

    def _release_lease(self, workflow_thread_id: str, lease: _Lease) -> None:
        with self._session_factory() as db:
            thread = _locked_thread(db, workflow_thread_id)
            if not _thread_has_exact_lease(thread, lease):
                db.rollback()
                return
            thread.lease_token = None
            thread.lease_expires_at = None
            thread.status = lease.prior_status
            thread.state_version += 1
            db.commit()

    def _result_for_ids(self, review_item_ids: tuple[int, ...]) -> ReviewDraftResult:
        with self._session_factory() as db:
            result = _draft_result_from_db(db, review_item_ids)
            db.rollback()
            return result


def build_permission_fingerprint(
    *,
    settings: Settings,
    allowed_permission_levels: Sequence[str],
) -> str:
    return build_keyed_fingerprint(
        list(_normalize_permission_levels(allowed_permission_levels)),
        settings=settings,
        schema_version=PERMISSION_FINGERPRINT_SCHEMA,
        policy_version=PERMISSION_FINGERPRINT_POLICY,
    )


def build_review_effect_key(
    *,
    settings: Settings,
    workflow_thread_id: str,
    agent_name: str,
    prompt_version: str,
    model_route_version: str,
    evidence_hash: str,
    permission_fingerprint: str,
    selection_policy_version: str,
) -> str:
    return build_keyed_fingerprint(
        {
            'workflow_thread_id': workflow_thread_id,
            'agent_name': agent_name,
            'prompt_version': prompt_version,
            'model_route_version': model_route_version,
            'evidence_hash': evidence_hash,
            'permission_fingerprint': permission_fingerprint,
            'selection_policy_version': selection_policy_version,
        },
        settings=settings,
        schema_version=EFFECT_KEY_SCHEMA,
        policy_version=EFFECT_KEY_POLICY,
    )


def _build_adapter_plans(
    db: Session,
    *,
    catalog: ReviewAgentCatalog,
    thread_id: str | None,
    agent_names: tuple[str, ...],
    evidence_hash: str,
    permission_fingerprint: str,
    selection_policy_version: str,
    packet: EvidencePacket,
    settings: Settings,
) -> tuple[_AdapterPlan, ...]:
    plans: list[_AdapterPlan] = []
    for name in agent_names:
        adapter = catalog.get(name)
        prompt_version = adapter.manifest.prompt_versions[0]
        effect_key = build_review_effect_key(
            settings=settings,
            workflow_thread_id=thread_id or 'unpersisted-preview',
            agent_name=name,
            prompt_version=prompt_version,
            model_route_version=adapter.model_route_version,
            evidence_hash=evidence_hash,
            permission_fingerprint=permission_fingerprint,
            selection_policy_version=selection_policy_version,
        )
        run = _find_agent_run(db, thread_id, effect_key) if thread_id else None
        decision = adapter.preflight(packet)
        if run is not None:
            decision = replace(
                decision,
                action='use_cache',
                reason='cache_hit',
                budget_status='cached',
                estimated_cost_usd=0.0,
                cache_hit=True,
            )
        plans.append(
            _AdapterPlan(
                adapter=adapter,
                effect_key=effect_key,
                decision=decision,
                cached_run_id=run.id if run is not None else None,
            )
        )
    return tuple(plans)


def _aggregate_preview(
    *,
    prepared: PreparedReviewRequest,
    plans: tuple[_AdapterPlan, ...],
    source_count: int,
    budget_limit_usd: float | None,
) -> ReviewWorkflowDryRunResponse:
    statuses = [plan.decision.budget_status for plan in plans]
    estimated_cost = sum(plan.decision.estimated_cost_usd for plan in plans)
    over_budget = 'over_budget' in statuses or (
        budget_limit_usd is not None and estimated_cost > budget_limit_usd
    )
    if over_budget:
        budget_status = 'over_budget'
    elif statuses and all(status == 'no_input' for status in statuses):
        budget_status = 'no_input'
    elif all(status in {'cached', 'no_input'} for status in statuses) and any(
        status == 'cached' for status in statuses
    ):
        budget_status = 'cached'
    else:
        budget_status = 'within_budget'
    cache_hit = (
        bool(statuses)
        and any(status == 'cached' for status in statuses)
        and all(status in {'cached', 'no_input'} for status in statuses)
    )
    return ReviewWorkflowDryRunResponse(
        workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
        graph_version=COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
        source_count=source_count,
        agent_names=list(prepared.agent_names),
        selection_policy_version=prepared.selection_policy_version,
        estimated_input_tokens=sum(
            plan.decision.token_usage.input_tokens for plan in plans
        ),
        estimated_output_tokens=sum(
            plan.decision.token_usage.output_tokens for plan in plans
        ),
        estimated_cost_usd=estimated_cost,
        budget_limit_usd=budget_limit_usd,
        budget_status=cast(Any, budget_status),
        cache_hit=cache_hit,
        requires_explicit_run=True,
    )


def _load_draft_inputs(
    db: Session,
    thread_id: str,
    *,
    lock: bool,
    settings: Settings,
) -> _DraftInputs:
    query = select(AgentWorkflowThread).where(
        AgentWorkflowThread.thread_id == thread_id
    )
    if lock:
        query = query.with_for_update()
    thread = db.scalar(query)
    if thread is None:
        raise ReviewDraftError('not_found', 'workflow was not found')
    return _draft_inputs_for_thread(db, thread, settings=settings)


def _draft_inputs_for_thread(
    db: Session,
    thread: AgentWorkflowThread,
    *,
    settings: Settings,
) -> _DraftInputs:
    if thread.security_scope_id != settings.agent_runtime_security_scope_id:
        raise ReviewDraftError('not_found', 'workflow was not found')
    request = db.scalar(
        select(AgentWorkflowRequest).where(
            AgentWorkflowRequest.workflow_thread_id == thread.thread_id
        )
    )
    rows = db.scalars(
        select(AgentWorkflowEvidenceRef)
        .where(AgentWorkflowEvidenceRef.workflow_thread_id == thread.thread_id)
        .order_by(AgentWorkflowEvidenceRef.ordinal)
    ).all()
    if request is None or not rows:
        raise ReviewDraftError('invalid_input', 'workflow evidence is incomplete')
    refs = tuple(
        ResolvedSourceVersion(
            source_type=row.canonical_source_type,
            canonical_table=row.canonical_table,
            canonical_row_id=row.canonical_row_id,
            document_version_id=row.document_version_id,
            external_revision=row.external_revision,
            content_signature=row.content_signature,
            permission_level=row.permission_level_snapshot,
            content_fingerprint=row.content_fingerprint,
        )
        for row in rows
    )
    raw_agent_names = tuple(request.agent_names)
    try:
        agent_names = normalize_agent_names(raw_agent_names)
    except (TypeError, ValueError):
        raise ReviewDraftError(
            'invalid_state_transition',
            'workflow identity changed',
        ) from None
    if agent_names != raw_agent_names:
        raise ReviewDraftError(
            'invalid_state_transition',
            'workflow identity changed',
        )
    prepared = build_prepared_review_identity(
        source_refs=refs,
        agent_names=agent_names,
        settings=settings,
    )
    if thread.evidence_version_hash != prepared.evidence_version_hash:
        raise ReviewDraftError('evidence_changed', 'source evidence changed')
    if not review_thread_matches_prepared(
        db,
        thread=thread,
        prepared=prepared,
        settings=settings,
    ):
        raise ReviewDraftError(
            'invalid_state_transition',
            'workflow identity changed',
        )
    return _DraftInputs(
        thread_id=thread.thread_id,
        security_scope_id=thread.security_scope_id,
        status=thread.status,
        state_version=thread.state_version,
        evidence_version_hash=prepared.evidence_version_hash,
        selection_policy_version=prepared.selection_policy_version,
        agent_names=prepared.agent_names,
        refs=refs,
        prepared=prepared,
    )


def _locked_thread(db: Session, thread_id: str) -> AgentWorkflowThread:
    thread = db.scalar(
        select(AgentWorkflowThread)
        .where(AgentWorkflowThread.thread_id == thread_id)
        .with_for_update()
    )
    if thread is None:
        raise ReviewDraftError('not_found', 'workflow was not found')
    return thread


def _ensure_thread_state(status: str) -> None:
    if status not in _DRAFTABLE_STATUSES:
        raise ReviewDraftError(
            'invalid_state_transition',
            'workflow cannot draft in its current state',
        )


def _ensure_thread_scope(inputs: _DraftInputs, settings: Settings) -> None:
    if inputs.security_scope_id != settings.agent_runtime_security_scope_id:
        raise ReviewDraftError('not_found', 'workflow was not found')


def _build_exact_packet(
    db: Session,
    *,
    refs: tuple[ResolvedSourceVersion, ...],
    workflow_thread_id: str | None,
    actor_subject_id: str,
    allowed_permission_levels: tuple[str, ...],
    settings: Settings,
) -> EvidencePacket:
    current_refs, sources = _revalidate_refs(
        db,
        refs=refs,
        actor_subject_id=actor_subject_id,
        allowed_permission_levels=allowed_permission_levels,
        settings=settings,
    )
    ranked: list[tuple[int, float, int, str, int, EvidenceMessage]] = []
    workflow_refs_by_source: dict[int, AgentWorkflowEvidenceRef] = {}
    if workflow_thread_id is not None:
        workflow_rows = db.scalars(
            select(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.workflow_thread_id == workflow_thread_id
            )
        ).all()
        workflow_refs_by_source = {row.canonical_row_id: row for row in workflow_rows}
    for ref in current_refs:
        source = sources[ref.canonical_row_id]
        parser_metadata = _authoritative_parser_metadata(
            db,
            source_id=source.id,
            document_version_id=ref.document_version_id,
        )
        chunk_query = select(DocumentChunk).where(DocumentChunk.source_id == source.id)
        if ref.document_version_id is not None:
            chunk_query = chunk_query.where(
                DocumentChunk.version_id == ref.document_version_id
            )
        chunks = db.scalars(chunk_query.order_by(DocumentChunk.chunk_index)).all()
        has_non_empty_chunk = False
        for chunk in chunks:
            if chunk.permission_level not in allowed_permission_levels:
                raise ReviewDraftError(
                    'permission_denied', 'evidence permission changed'
                )
            if not chunk.text.strip():
                continue
            has_non_empty_chunk = True
            message = _message_from_chunk(
                source, chunk, parser_metadata=parser_metadata
            )
            message = _bind_message_to_workflow_ref(
                message,
                workflow_thread_id=workflow_thread_id,
                workflow_ref=workflow_refs_by_source.get(source.id),
                stable_message_identity=f'chunk:{chunk.id}',
            )
            ranked.append(
                (
                    _importance_score(message.text, source.source_type),
                    _source_timestamp(source),
                    -chunk.chunk_index,
                    source.source_id,
                    chunk.id,
                    message,
                )
            )
        if has_non_empty_chunk:
            continue
        fallback = _fallback_message(
            db,
            source=source,
            ref=ref,
            parser_metadata=parser_metadata,
        )
        if fallback is not None:
            fallback = _bind_message_to_workflow_ref(
                fallback,
                workflow_thread_id=workflow_thread_id,
                workflow_ref=workflow_refs_by_source.get(source.id),
                stable_message_identity=f'source:{source.id}:fallback',
            )
            ranked.append(
                (
                    _importance_score(fallback.text, source.source_type),
                    _source_timestamp(source),
                    0,
                    source.source_id,
                    0,
                    fallback,
                )
            )

    selected: list[EvidenceMessage] = []
    seen: set[tuple[str, str]] = set()
    remaining_chars = max(settings.agent_llm_max_input_chars, 0)
    max_messages = max(settings.agent_llm_max_evidence_messages, 0)
    for rank, (_, _, _, _, _, message) in enumerate(
        sorted(ranked, reverse=True),
        start=1,
    ):
        normalized_text = ' '.join(message.text.split()).casefold()
        dedupe_key = (message.source_id, normalized_text)
        if not normalized_text or dedupe_key in seen:
            continue
        if len(selected) >= max_messages or remaining_chars <= 0:
            break
        seen.add(dedupe_key)
        text = message.text[:remaining_chars]
        remaining_chars -= len(text)
        selected.append(
            replace(
                message,
                text=text,
                metadata={
                    **message.metadata,
                    'evidence_rank': rank,
                    'importance_score': _importance_score(
                        message.text,
                        str(message.metadata.get('source_type') or ''),
                    ),
                    'selection_policy_version': COMPANY_MEMORY_SELECTION_POLICY_VERSION,
                },
            )
        )
    return EvidencePacket(
        source_type='company_memory',
        source_window=(
            f'{COMPANY_MEMORY_SELECTION_POLICY_VERSION}:ranked:{max_messages}'
        ),
        messages=selected,
        permission_context=PermissionContext(
            user_id=actor_subject_id,
            role='workflow_actor',
            allowed_permission_levels=allowed_permission_levels,
        ),
    )


def _revalidate_refs(
    db: Session,
    *,
    refs: tuple[ResolvedSourceVersion, ...],
    actor_subject_id: str,
    allowed_permission_levels: tuple[str, ...],
    settings: Settings,
) -> tuple[tuple[ResolvedSourceVersion, ...], dict[int, Source]]:
    source_ids = [ref.canonical_row_id for ref in refs]
    source_rows = db.scalars(select(Source).where(Source.id.in_(source_ids))).all()
    sources = {source.id: source for source in source_rows}
    if len(sources) != len(source_ids):
        raise ReviewDraftError('not_found', 'source reference was not found')
    for ref in refs:
        source = sources[ref.canonical_row_id]
        if ref.canonical_table != 'sources' or source.source_type != ref.source_type:
            raise ReviewDraftError('evidence_changed', 'source evidence changed')
        if source.permission_level not in allowed_permission_levels:
            raise ReviewDraftError('permission_denied', 'evidence permission changed')
        if source.permission_level != ref.permission_level:
            raise ReviewDraftError('permission_denied', 'evidence permission changed')
    source_refs = tuple(
        SourceVersionRef(
            source_type=cast(Any, ref.source_type),
            source_id=sources[ref.canonical_row_id].source_id,
            version_or_signature=ref.content_signature,
        )
        for ref in refs
    )
    try:
        current = resolve_source_versions(
            db,
            refs=source_refs,
            actor=_actor(actor_subject_id, allowed_permission_levels),
            settings=settings,
        )
    except ReviewWorkflowPreflightError as exc:
        raise ReviewDraftError(exc.code, str(exc)) from None
    if current != refs:
        raise ReviewDraftError('evidence_changed', 'source evidence changed')
    return current, sources


def _authoritative_parser_metadata(
    db: Session,
    *,
    source_id: int,
    document_version_id: int | None,
) -> dict[str, object]:
    if document_version_id is None:
        return {}
    parser_run = db.scalar(
        select(DocumentParserRun)
        .where(
            DocumentParserRun.source_id == source_id,
            DocumentParserRun.document_version_id == document_version_id,
        )
        .order_by(DocumentParserRun.finished_at.desc(), DocumentParserRun.id.desc())
    )
    if parser_run is None:
        return {}
    return {
        'parser_run_id': parser_run.id,
        'parser_name': parser_run.parser_name,
        'parser_status': parser_run.parser_status,
        'parser_status_reason': parser_run.parser_status_reason,
        'mime_type': parser_run.mime_type,
        'document_version_label': parser_run.document_version_label,
        'revision_id': parser_run.revision_id,
        'content_signature': parser_run.content_signature,
    }


def _message_from_chunk(
    source: Source,
    chunk: DocumentChunk,
    *,
    parser_metadata: dict[str, object],
) -> EvidenceMessage:
    metadata = source.raw_metadata or {}
    quality_keys = (
        'parser_name',
        'parser_status',
        'parser_status_reason',
        'mime_type',
        'section_path',
        'page_number',
        'calendar_id',
        'calendar_summary',
        'event_context_key',
        'event_status',
        'organizer_email',
        'attendee_domains',
        'location',
        'event_start',
        'event_end',
        'start',
        'end',
    )
    return EvidenceMessage(
        source_id=source.source_id,
        source_url=source.source_url,
        text=chunk.text,
        author=source.author,
        timestamp=str(metadata.get('ts') or source.created_at.isoformat()),
        permission_level=_strictest_permission(
            source.permission_level,
            chunk.permission_level,
        ),
        metadata={
            'source_type': source.source_type,
            'chunk_id': chunk.id,
            **{
                key: chunk.metadata_.get(key)
                for key in quality_keys
                if chunk.metadata_.get(key) is not None
            },
            **parser_metadata,
        },
        source_snippet_override=chunk.source_snippet,
    )


def _bind_message_to_workflow_ref(
    message: EvidenceMessage,
    *,
    workflow_thread_id: str | None,
    workflow_ref: AgentWorkflowEvidenceRef | None,
    stable_message_identity: str,
) -> EvidenceMessage:
    if workflow_thread_id is None:
        return message
    if workflow_ref is None:
        raise ReviewDraftError(
            'invalid_state_transition', 'workflow evidence is incomplete'
        )
    return replace(
        message,
        metadata={
            **message.metadata,
            'workflow_thread_id': workflow_thread_id,
            'workflow_evidence_ref_id': workflow_ref.id,
            'canonical_source_kind': workflow_ref.canonical_source_type,
            'canonical_source_id': workflow_ref.canonical_row_id,
            'canonical_version_or_signature': (
                workflow_ref.external_revision or workflow_ref.content_signature
            ),
            'content_fingerprint': workflow_ref.content_fingerprint,
            'stable_message_identity': stable_message_identity,
        },
    )


def _fallback_message(
    db: Session,
    *,
    source: Source,
    ref: ResolvedSourceVersion,
    parser_metadata: dict[str, object],
) -> EvidenceMessage | None:
    text = ''
    if ref.document_version_id is not None:
        version = db.get(DocumentVersion, ref.document_version_id)
        if version is not None:
            text = version.body
    if not text:
        snippet = (source.raw_metadata or {}).get('source_snippet')
        text = snippet if isinstance(snippet, str) else ''
    if not text.strip():
        return None
    return EvidenceMessage(
        source_id=source.source_id,
        source_url=source.source_url,
        text=text,
        author=source.author,
        timestamp=str(
            (source.raw_metadata or {}).get('ts') or source.created_at.isoformat()
        ),
        permission_level=source.permission_level,
        metadata={
            'source_type': source.source_type,
            'fallback_body': True,
            **parser_metadata,
        },
        source_snippet_override=text[:240],
    )


def _packet_fingerprint(packet: EvidencePacket, *, settings: Settings) -> str:
    return build_keyed_fingerprint(
        {
            'source_type': packet.source_type,
            'source_window': packet.source_window,
            'messages': [
                {
                    'source_id': message.source_id,
                    'source_url': message.source_url,
                    'text': message.text,
                    'author': message.author,
                    'timestamp': message.timestamp,
                    'permission_level': message.permission_level,
                    'source_snippet': message.source_snippet,
                    'metadata': message.metadata,
                }
                for message in packet.messages
            ],
        },
        settings=settings,
        schema_version=PACKET_FINGERPRINT_SCHEMA,
        policy_version=PACKET_FINGERPRINT_POLICY,
    )


def _validated_result(
    result: AgentRunResult,
    *,
    plan: _AdapterPlan,
    packet: EvidencePacket,
) -> AgentRunResult:
    expected_prompt = plan.adapter.manifest.prompt_versions[0]
    if (
        result.agent_name != plan.adapter.manifest.name
        or result.prompt_version != expected_prompt
    ):
        raise ReviewDraftError('invalid_input', 'agent result contract mismatch')
    packet_links = set(packet.source_links)
    packet_snippets = set(packet.source_snippets)
    candidates: list[ReviewCandidate] = []
    for candidate in result.candidates:
        if candidate.item_type not in plan.adapter.allowed_item_types:
            raise ReviewDraftError(
                'invalid_input',
                'agent result item type is not allowed',
            )
        try:
            candidate.validate_evidence()
        except ValueError:
            raise ReviewDraftError(
                'invalid_input',
                'review candidate requires source evidence',
            ) from None
        if not set(candidate.source_links).issubset(packet_links) or not set(
            candidate.source_snippets
        ).issubset(packet_snippets):
            raise ReviewDraftError('invalid_input', 'candidate evidence is not bound')
        if not 0.0 <= candidate.confidence_score <= 1.0:
            raise ReviewDraftError('invalid_input', 'candidate confidence is invalid')
        candidates.append(
            replace(
                candidate,
                permission_level=_strictest_permission(
                    packet.strictest_permission,
                    candidate.permission_level,
                ),
            )
        )
    return replace(result, candidates=candidates)


def _insert_or_get_agent_run(
    db: Session,
    *,
    thread: AgentWorkflowThread,
    plan: _AdapterPlan,
    result: AgentRunResult,
    packet: EvidencePacket,
    packet_hash: str,
    permission_fingerprint: str,
    selection_policy_version: str,
    now: datetime,
) -> tuple[AgentRun, bool]:
    values = {
        'agent_name': result.agent_name,
        'prompt_version': result.prompt_version,
        'status': 'complete',
        'source_window': packet.source_window,
        'cache_key': plan.effect_key,
        'model_name': result.cost.model_name,
        'generation_provider': result.model_provider,
        'generation_reasoning_effort': result.model_reasoning_effort,
        'generation_route_version': result.route_version,
        'generation_output_contract_version': result.output_contract_version,
        'input_tokens': result.cost.token_usage.input_tokens,
        'output_tokens': result.cost.token_usage.output_tokens,
        'total_tokens': result.cost.token_usage.total_tokens,
        'estimated_cost_usd': result.cost.estimated_cost_usd,
        'permission_level': packet.strictest_permission,
        'metadata_': {
            'cache_hit': result.cost.cache_hit,
            'message_count': len(packet.messages),
            'source_count': len({message.source_id for message in packet.messages}),
            'evidence_hash': packet_hash,
            'permission_fingerprint': permission_fingerprint,
            'model_route_version': plan.adapter.model_route_version,
            'selection_policy_version': selection_policy_version,
        },
        'workflow_thread_id': thread.thread_id,
        'effect_key': plan.effect_key,
        'started_at': now,
        'completed_at': now,
    }
    inserted_id = _insert_do_nothing(db, AgentRun, values)
    run = (
        db.get(AgentRun, inserted_id)
        if inserted_id is not None
        else _find_agent_run(db, thread.thread_id, plan.effect_key)
    )
    if run is None:
        raise ReviewDraftError('concurrent_resume', 'agent effect insert was lost')
    return run, inserted_id is not None


def _insert_or_get_review_item(
    db: Session,
    *,
    thread: AgentWorkflowThread,
    agent_run: AgentRun,
    plan: _AdapterPlan,
    result: AgentRunResult,
    candidate: ReviewCandidate,
    packet: EvidencePacket,
    settings: Settings,
    now: datetime,
) -> tuple[ReviewItem, bool]:
    candidate_key = build_keyed_fingerprint(
        {
            'effect_key': plan.effect_key,
            'candidate': _candidate_identity(candidate),
        },
        settings=settings,
        schema_version=CANDIDATE_KEY_SCHEMA,
        policy_version=CANDIDATE_KEY_POLICY,
    )
    source_ids = _unique_strings(message.source_id for message in packet.messages)
    permission_level = _strictest_permission(
        packet.strictest_permission,
        candidate.permission_level,
    )
    predecessor_id = _find_predecessor_id(
        db,
        security_scope_id=thread.security_scope_id,
        agent_name=result.agent_name,
        item_type=candidate.item_type,
        source_ids=source_ids,
    )
    material_verifier = build_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    security_scope_hmac = build_keyed_fingerprint(
        thread.security_scope_id,
        settings=settings,
        schema_version='candidate-security-scope:v1',
        policy_version='candidate-security-scope:v1',
    )
    try:
        binding_set = build_candidate_evidence_bindings(
            candidate=candidate,
            packet=packet,
            workflow_execution_hmac=thread.input_hash,
            security_scope_hmac=security_scope_hmac,
            candidate_key=candidate_key,
            settings=settings,
            fingerprint_key_material_verifier=material_verifier,
            workflow_thread_id=thread.thread_id,
        )
    except CandidateEvidenceBindingError:
        raise ReviewDraftError(
            'evidence_binding_mismatch',
            'candidate evidence does not map exactly to workflow evidence',
        ) from None
    payload = {
        **candidate.payload_fields,
        'title': candidate.title,
        'summary': candidate.summary,
        'agent_name': result.agent_name,
        'agent_run_id': agent_run.id,
        'prompt_version': result.prompt_version,
        'cache_key': plan.effect_key,
        'estimated_cost_usd': result.cost.estimated_cost_usd,
        'token_usage': {
            'input_tokens': result.cost.token_usage.input_tokens,
            'output_tokens': result.cost.token_usage.output_tokens,
            'total_tokens': result.cost.token_usage.total_tokens,
        },
        'uncertainty_reason': candidate.uncertainty_reason,
        'source_ids': source_ids,
        'source_types': _unique_strings(
            str(message.metadata.get('source_type') or packet.source_type)
            for message in packet.messages
        ),
        'source_authors': _unique_strings(
            message.author for message in packet.messages if message.author
        ),
    }
    values = {
        'item_type': candidate.item_type,
        'payload': payload,
        'source_links': list(candidate.source_links),
        'source_snippets': list(candidate.source_snippets),
        'confidence_score': candidate.confidence_score,
        'permission_level': permission_level,
        'status': 'pending_review',
        'workflow_thread_id': thread.thread_id,
        'candidate_key': candidate_key,
        'predecessor_review_item_id': predecessor_id,
        'agent_run_id': agent_run.id,
        'candidate_contract_version': CANDIDATE_CONTRACT_VERSION,
        'created_at': now,
    }
    inserted_id = _insert_do_nothing(db, ReviewItem, values)
    item = (
        db.get(ReviewItem, inserted_id)
        if inserted_id is not None
        else db.scalar(
            select(ReviewItem).where(
                ReviewItem.workflow_thread_id == thread.thread_id,
                ReviewItem.candidate_key == candidate_key,
            )
        )
    )
    if item is None:
        raise ReviewDraftError('concurrent_resume', 'review candidate insert was lost')
    created = inserted_id is not None
    if created:
        for binding in binding_set.refs:
            db.add(
                ReviewItemEvidenceRef(
                    review_item_id=item.id,
                    workflow_thread_id=thread.thread_id,
                    workflow_evidence_ref_id=binding.workflow_evidence_ref_id,
                    candidate_slot_ordinal=binding.ordinal,
                    message_content_fingerprint=binding.message_set_hmac,
                    fingerprint_key_version=binding.fingerprint_key_version,
                    fingerprint_key_material_verifier=(
                        binding.fingerprint_key_material_verifier
                    ),
                    created_at=now,
                )
            )
        db.flush()
    else:
        stored_refs = db.scalars(
            select(ReviewItemEvidenceRef)
            .where(ReviewItemEvidenceRef.review_item_id == item.id)
            .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
        ).all()
        workflow_refs = {
            row.id: row
            for row in db.scalars(
                select(AgentWorkflowEvidenceRef).where(
                    AgentWorkflowEvidenceRef.workflow_thread_id == thread.thread_id,
                    AgentWorkflowEvidenceRef.id.in_(
                        [row.workflow_evidence_ref_id for row in stored_refs]
                    ),
                )
            ).all()
        }
        stored_bindings = tuple(
            CandidateEvidenceRefBinding(
                workflow_evidence_ref_id=row.workflow_evidence_ref_id,
                ordinal=row.candidate_slot_ordinal,
                canonical_source_kind=workflow_refs[
                    row.workflow_evidence_ref_id
                ].canonical_source_type,
                canonical_source_id=workflow_refs[
                    row.workflow_evidence_ref_id
                ].canonical_row_id,
                canonical_version_or_signature=(
                    workflow_refs[row.workflow_evidence_ref_id].external_revision
                    or workflow_refs[row.workflow_evidence_ref_id].content_signature
                ),
                content_fingerprint=workflow_refs[
                    row.workflow_evidence_ref_id
                ].content_fingerprint,
                message_set_hmac=row.message_content_fingerprint,
                permission_level=workflow_refs[
                    row.workflow_evidence_ref_id
                ].permission_level_snapshot,
                fingerprint_key_version=row.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    row.fingerprint_key_material_verifier
                ),
            )
            for row in stored_refs
            if row.workflow_evidence_ref_id in workflow_refs
        )
        try:
            binding_set.verify_replay(
                agent_run_id=agent_run.id,
                stored_agent_run_id=item.agent_run_id,
                stored_refs=tuple(row.workflow_evidence_ref_id for row in stored_refs),
                stored_bindings=stored_bindings,
            )
        except CandidateEvidenceBindingError:
            raise ReviewDraftError(
                'invalid_state_transition',
                'candidate evidence binding changed',
            ) from None
    return item, created


def _insert_do_nothing(db: Session, model: type, values: dict[str, Any]) -> int | None:
    dialect = db.get_bind().dialect.name
    if dialect == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == 'sqlite':
        from sqlalchemy.dialects.sqlite import insert
    else:
        instance = model(**values)
        db.add(instance)
        db.flush()
        return cast(int, instance.id)
    statement = (
        insert(model).values(**values).on_conflict_do_nothing().returning(model.id)
    )
    return db.execute(statement).scalar_one_or_none()


def _candidate_identity(candidate: ReviewCandidate) -> dict[str, object]:
    return {
        'item_type': candidate.item_type,
        'title': candidate.title,
        'summary': candidate.summary,
        'source_links': list(candidate.source_links),
        'source_snippets': list(candidate.source_snippets),
        'confidence_score': candidate.confidence_score,
        'permission_level': candidate.permission_level,
        'uncertainty_reason': candidate.uncertainty_reason,
        'payload_fields': candidate.payload_fields,
    }


def _find_predecessor_id(
    db: Session,
    *,
    security_scope_id: str,
    agent_name: str,
    item_type: str,
    source_ids: list[str],
) -> int | None:
    expected_sources = _normalized_source_id_set(source_ids)
    if expected_sources is None:
        return None
    candidates = db.scalars(
        select(ReviewItem)
        .join(
            AgentWorkflowThread,
            AgentWorkflowThread.thread_id == ReviewItem.workflow_thread_id,
        )
        .where(
            AgentWorkflowThread.security_scope_id == security_scope_id,
            ReviewItem.status == 'needs_more_evidence',
            ReviewItem.item_type == item_type,
        )
        .order_by(ReviewItem.created_at.desc(), ReviewItem.id.desc())
    ).all()
    for item in candidates:
        if item.payload.get('agent_name') != agent_name:
            continue
        if (
            _normalized_source_id_set(item.payload.get('source_ids'))
            == expected_sources
        ):
            return item.id
    return None


def _normalized_source_id_set(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            return None
        source_id = item.strip()
        prefix, separator, suffix = source_id.partition(':')
        if not prefix or not separator or not suffix:
            return None
        normalized.add(source_id)
    return tuple(sorted(normalized))


def _find_agent_run(
    db: Session,
    thread_id: str | None,
    effect_key: str,
) -> AgentRun | None:
    if thread_id is None:
        return None
    return db.scalar(
        select(AgentRun).where(
            AgentRun.workflow_thread_id == thread_id,
            AgentRun.effect_key == effect_key,
            AgentRun.status == 'complete',
        )
    )


def _review_item_ids_for_run(
    db: Session,
    agent_run_id: int,
    workflow_thread_id: str,
    *,
    settings: Settings,
) -> tuple[int, ...]:
    request = db.get(AgentWorkflowRequest, workflow_thread_id)
    if request is None:
        raise ReviewDraftError(
            'invalid_state_transition',
            'cached candidate provenance is incomplete',
        )
    _, expected_key_version = fingerprint_secret_bytes(settings)
    expected_material_verifier = build_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    items = db.scalars(
        select(ReviewItem).where(
            ReviewItem.workflow_thread_id == workflow_thread_id,
            ReviewItem.agent_run_id == agent_run_id,
        )
    ).all()
    for item in items:
        if item.candidate_contract_version != CANDIDATE_CONTRACT_VERSION:
            raise ReviewDraftError(
                'invalid_state_transition',
                'cached candidate provenance is incomplete',
            )
        refs = tuple(
            db.scalars(
                select(ReviewItemEvidenceRef)
                .where(
                    ReviewItemEvidenceRef.review_item_id == item.id,
                    ReviewItemEvidenceRef.workflow_thread_id == workflow_thread_id,
                )
                .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
            ).all()
        )
        workflow_ref_ids = tuple(ref.workflow_evidence_ref_id for ref in refs)
        workflow_ref_count = db.scalar(
            select(func.count())
            .select_from(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id == workflow_thread_id,
                AgentWorkflowEvidenceRef.id.in_(workflow_ref_ids),
            )
        )
        if (
            item.candidate_key is None
            or not refs
            or tuple(ref.candidate_slot_ordinal for ref in refs)
            != tuple(range(1, len(refs) + 1))
            or len(set(workflow_ref_ids)) != len(refs)
            or workflow_ref_count != len(refs)
            or any(
                    ref.fingerprint_key_version != expected_key_version
                    or ref.fingerprint_key_material_verifier
                    != expected_material_verifier
                or len(ref.message_content_fingerprint) != 64
                for ref in refs
            )
        ):
            raise ReviewDraftError(
                'invalid_state_transition',
                'cached candidate provenance is incomplete',
            )
    return tuple(sorted(item.id for item in items))


def _cached_review_item_ids(
    db: Session,
    plans: tuple[_AdapterPlan, ...],
    workflow_thread_id: str,
    *,
    settings: Settings,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                item_id
                for plan in plans
                if plan.cached_run_id is not None
                for item_id in _review_item_ids_for_run(
                    db,
                    plan.cached_run_id,
                    workflow_thread_id,
                    settings=settings,
                )
            }
        )
    )


def _draft_result_from_db(
    db: Session,
    review_item_ids: tuple[int, ...],
) -> ReviewDraftResult:
    if not review_item_ids:
        return _draft_result((), ())
    statuses = tuple(
        db.scalars(
            select(ReviewItem.status).where(ReviewItem.id.in_(review_item_ids))
        ).all()
    )
    if len(statuses) != len(review_item_ids):
        raise ReviewDraftError(
            'concurrent_resume', 'review candidate replay is incomplete'
        )
    return _draft_result(review_item_ids, statuses)


def _draft_result(
    review_item_ids: tuple[int, ...],
    statuses: Sequence[str],
) -> ReviewDraftResult:
    counts: dict[ReviewItemResolutionStatus, int] = dict.fromkeys(REVIEW_STATUSES, 0)
    for status in statuses:
        if status not in counts:
            raise ReviewDraftError(
                'invalid_state_transition', 'review status is invalid'
            )
        counts[cast(ReviewItemResolutionStatus, status)] += 1
    return ReviewDraftResult(
        review_item_ids=review_item_ids,
        review_status_counts=counts,
    )


def _actor(subject_id: str, permission_levels: tuple[str, ...]) -> DemoUser:
    return DemoUser(
        id=subject_id,
        email=f'{subject_id}@workflow.invalid',
        role='workflow_actor',
        permission_levels=set(permission_levels),
        name=subject_id,
        title='Workflow Actor',
        department='Workflow',
    )


def _normalize_permission_levels(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({value.strip() for value in values if value.strip()}))
    if not normalized or any(value not in _PERMISSION_RANK for value in normalized):
        raise ReviewDraftError('permission_denied', 'permission context is invalid')
    return normalized


def _strictest_permission(*levels: str) -> str:
    try:
        return max(levels, key=lambda level: _PERMISSION_RANK[level])
    except KeyError:
        raise ReviewDraftError(
            'permission_denied', 'permission level is invalid'
        ) from None


def _importance_score(text: str, source_type: str) -> int:
    lowered = text.lower()
    score = 0
    if any(
        value in lowered
        for value in ('decision', 'decided', 'approved', '결정', '승인')
    ):
        score += 50
    if any(
        value in lowered
        for value in ('todo', 'due', 'deadline', '검토', '준비', '배포')
    ):
        score += 35
    if source_type == 'gmail':
        score += 10
    elif source_type in {'drive', 'gmail_attachment'}:
        score += 8
    elif source_type == 'calendar':
        score += 6
    if 40 <= len(text) <= 1600:
        score += 5
    return score


def _source_timestamp(source: Source) -> float:
    value = (source.raw_metadata or {}).get('ts')
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    created_at = source.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.timestamp()


def _lease_expired(expires_at: datetime, now: datetime) -> bool:
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return expires_at <= now


def _thread_has_exact_lease(
    thread: AgentWorkflowThread,
    lease: _Lease,
) -> bool:
    return bool(lease.token) and (
        thread.lease_token == lease.token
        and thread.state_version == lease.state_version
        and thread.status == 'drafting'
        and thread.lease_expires_at is not None
        and _same_datetime(thread.lease_expires_at, lease.expires_at)
    )


def _lease_restore_status(status: str) -> str:
    if status == 'drafting':
        # Taking over an expired drafting lease is the one approved cleanup
        # transition that cannot restore its prior transient state.
        return 'created'
    if status in _DRAFTABLE_STATUSES:
        return status
    raise ReviewDraftError(
        'invalid_state_transition',
        'workflow cannot draft in its current state',
    )


def _same_datetime(left: datetime, right: datetime) -> bool:
    if left.tzinfo is None:
        left = left.replace(tzinfo=UTC)
    if right.tzinfo is None:
        right = right.replace(tzinfo=UTC)
    return left == right


def _unique_strings(values) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
