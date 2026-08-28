from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Protocol

from fastapi import HTTPException
from sqlalchemy import and_, or_, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    KeyGenerationLockedContext,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.rbac import PERMISSION_ORDER
from backend.app.knowledge.claim_fingerprints import (
    normalized_claim_fingerprint,
    promoted_effect_fingerprint,
    trusted_project_scope_fingerprint,
    trusted_title_collision_bucket,
)
from backend.app.knowledge.promotion import (
    IncompleteReviewPromotionError,
    build_promotion_preview,
    find_review_item_promotion,
    promote_review_item,
    validate_review_item_for_approval,
)
from backend.app.knowledge.trusted_fingerprint_projection import (
    TRUSTED_FINGERPRINT_PROJECTION_COMPONENT,
    build_hidden_collision_exists_statement,
    build_missing_active_projection_exists_statement,
    demote_trusted_fingerprint_projection,
    project_approved_effects,
    projection_identity_ready,
    source_projection_snapshots,
)
from backend.app.knowledge.trusted_provenance import (
    TrustedPromotionBundle,
    effects_for_created_promotion,
    find_explicit_promotion,
    record_or_verify_explicit_provenance,
    replay_effect_for_bundle,
    validate_reuse_bundle,
)
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AutoReviewExtractionCall,
    AutoReviewProviderSafetyState,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    AutoReviewValidation,
    AutoReviewValidationCall,
    Document,
    DocumentVersion,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TrustedKnowledgeFingerprint,
    TrustedKnowledgeFingerprintProjectionState,
)
from backend.app.review.actors import (
    SYSTEM_AUTO_REVIEW_ACTOR_ID,
    ApprovalDirective,
    CreateNewPromotion,
    ReuseExistingPromotion,
    ReviewResolutionActor,
    _assert_approval_directive,
    _assert_review_resolution_actor,
)
from backend.app.schemas.auto_review import AUTO_REVIEW_POLICY_VERSION
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
)

ReviewAction = Literal['approve', 'reject', 'needs_more_evidence']


class CurrentPermissionResolver(Protocol):
    def __call__(
        self,
        subject_id: str,
    ) -> tuple[str, Sequence[str]] | None: ...


@dataclass(frozen=True)
class PromotionResult:
    target_type: str | None
    created_record_ids: tuple[int, ...]
    created_timeline_event_ids: tuple[int, ...]
    effect: Literal['created', 'reaffirmed'] = 'created'


@dataclass(frozen=True)
class ReviewTransitionResult:
    item_id: int
    status: str
    replayed: bool
    promotion: PromotionResult | None


@dataclass(frozen=True)
class ReviewBatchTransitionResult:
    results: tuple[ReviewTransitionResult, ...]
    failed_items: tuple[dict[str, object], ...]
    skipped_items: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _ProviderCallSnapshot:
    purpose: str
    call_id: int
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


@dataclass(frozen=True, slots=True)
class _AutoApprovalLocator:
    review_item_id: int
    workflow_thread_id: str
    security_scope_id: str
    policy_version: str
    validation_id: int
    validation_call_id: int
    validation_snapshot: _ProviderCallSnapshot
    extraction_snapshots: tuple[_ProviderCallSnapshot, ...]
    authorized_percentage_at_launch: int
    rollout_authorization_generation: int
    rollout_control_epoch: int


class InvalidReviewTransition(ValueError):  # noqa: N818 - name is a frozen public contract
    code = 'invalid_state_transition'

    def __init__(self, *, action: ReviewAction, status: str) -> None:
        super().__init__(f'Cannot {action} review item from {status}')


class AutoReviewAdmissionNotReady(ValueError):  # noqa: N818 - policy outcome
    pass


class CanonicalEvidenceDrift(ValueError):  # noqa: N818 - policy outcome
    pass


class ReviewTransitionService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        current_permission_resolver: CurrentPermissionResolver | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._current_permission_resolver = current_permission_resolver

    def transition(
        self,
        *,
        db: Session,
        item_id: int,
        action: ReviewAction,
        actor: ReviewResolutionActor,
        note: str | None = None,
        approval_directive: ApprovalDirective | None = None,
    ) -> ReviewTransitionResult:
        _assert_review_resolution_actor(actor)
        preview = db.get(ReviewItem, item_id)
        if preview is None:
            raise ValueError('Review item not found')
        directive = self._authorize_action(
            item=preview,
            action=action,
            actor=actor,
            approval_directive=approval_directive,
        )
        if action == 'approve':
            auto_locator = (
                self._discover_auto_approval_locator(db, preview=preview, actor=actor)
                if actor.actor_type == 'auto_policy'
                else None
            )
            with KeyedMutationGuard.generation_barrier(db):
                context = lock_runtime_state(db, mode='share')
                safety = (
                    self._lock_and_require_provider_safety(db, locator=auto_locator)
                    if auto_locator is not None
                    else {}
                )
                acquire_projection(db, context)
                if actor.actor_type == 'auto_policy':
                    self._require_auto_projection_admission(db, context=context)
                    self._require_exact_reaffirmation_collision_clear(
                        db,
                        item=preview,
                        actor=actor,
                        directive=directive,
                        locator=auto_locator,
                    )
                    self._lock_and_require_rollout(db, locator=auto_locator)
                validation_ids: list[int] = []
                item = self._lock_approval_item(
                    db,
                    preview=preview,
                    before_item_lock=(
                        lambda: validation_ids.append(
                            self._lock_and_require_auto_calls(
                                db,
                                locator=auto_locator,
                                safety=safety,
                            )
                        )
                        if auto_locator is not None
                        else None
                    ),
                )
                if auto_locator is not None:
                    self._require_auto_validation_multiplicity_after_item_lock(
                        db,
                        locator=auto_locator,
                    )
                    self._require_current_owner_permissions(
                        db,
                        item=item,
                        actor=actor,
                    )
                return self._transition_item(
                    db=db,
                    item=item,
                    action=action,
                    actor=actor,
                    note=note,
                    approval_directive=directive,
                    locked_auto_validation_id=(
                        validation_ids[0] if validation_ids else None
                    ),
                )
        if preview.candidate_contract_version == 'c5-v1':
            item, _ = CanonicalReviewEvidenceStalenessResolver(
                settings=self._settings
            ).lock_item_and_resolve_drift(
                db,
                item_id=item_id,
            )
        else:
            items = self._load_items(db, [item_id], for_update=True)
            if not items:
                raise ValueError('Review item not found')
            item = items[0]
        return self._transition_item(
            db=db,
            item=item,
            action=action,
            actor=actor,
            note=note,
            approval_directive=directive,
            locked_auto_validation_id=None,
        )

    def _lock_approval_item(
        self,
        db: Session,
        *,
        preview: ReviewItem,
        before_item_lock: Callable[[], None] | None,
    ) -> ReviewItem:
        if preview.candidate_contract_version == 'c5-v1':
            item, drifted = (
                CanonicalReviewEvidenceStalenessResolver(
                    settings=self._settings
                ).lock_item_and_resolve_drift(
                    db,
                    item_id=preview.id,
                    before_item_lock=before_item_lock,
                )
            )
            if drifted:
                raise CanonicalEvidenceDrift(
                    'Review item canonical evidence has changed'
                )
            return item
        if before_item_lock is not None:
            before_item_lock()
        items = ReviewTransitionService._load_items(
            db,
            [preview.id],
            for_update=True,
        )
        if not items:
            raise ValueError('Review item not found')
        return items[0]

    def _require_auto_projection_admission(
        self,
        db: Session,
        *,
        context: KeyGenerationLockedContext | None,
    ) -> None:
        if (
            self._settings.auto_review_mode != 'enforce'
            or context is None
            or db.get_bind().dialect.name != 'postgresql'
        ):
            raise AutoReviewAdmissionNotReady('Automatic review admission is not ready')
        configured_verifier = fingerprint_key_material_verifier(
            self._settings.agent_runtime_fingerprint_secret
        )
        if (
            context.key_version
            != self._settings.agent_runtime_fingerprint_key_version
            or context.material_verifier != configured_verifier
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review key identity is not ready'
            )
        runtime = db.scalar(
            select(AutoReviewRuntimeKeyState).where(
                AutoReviewRuntimeKeyState.component
                == 'auto_review_trust_promotion'
            )
        )
        projection = db.scalar(
            select(TrustedKnowledgeFingerprintProjectionState).where(
                TrustedKnowledgeFingerprintProjectionState.component
                == TRUSTED_FINGERPRINT_PROJECTION_COMPONENT
            )
        )
        missing_active_row = bool(
            db.scalar(
                build_missing_active_projection_exists_statement(
                    expected_rows=source_projection_snapshots(
                        db,
                        settings=self._settings,
                        fingerprint_key_material_verifier=context.material_verifier,
                    ),
                )
            )
        )
        if not projection_identity_ready(
            runtime,
            projection,
            missing_active_row=missing_active_row,
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review projection is not ready'
            )

    def _discover_auto_approval_locator(
        self,
        db: Session,
        *,
        preview: ReviewItem,
        actor: ReviewResolutionActor,
    ) -> _AutoApprovalLocator:
        if preview.workflow_thread_id is None or actor.policy_version is None:
            raise AutoReviewAdmissionNotReady('Automatic review workflow is missing')
        workflow = db.get(AgentWorkflowThread, preview.workflow_thread_id)
        request = db.get(AgentWorkflowRequest, preview.workflow_thread_id)
        validations = tuple(
            db.scalars(
                select(AutoReviewValidation).where(
                    AutoReviewValidation.review_item_id == preview.id,
                    AutoReviewValidation.workflow_thread_id
                    == preview.workflow_thread_id,
                    AutoReviewValidation.status == 'completed',
                    AutoReviewValidation.policy_version == actor.policy_version,
                )
            ).all()
        )
        if workflow is None or request is None or len(validations) != 1:
            raise AutoReviewAdmissionNotReady(
                'Automatic review control snapshot is incomplete'
            )
        validation = validations[0]
        validation_call = db.get(
            AutoReviewValidationCall,
            validation.validation_call_id,
        )
        extraction_calls = tuple(
            db.scalars(
                select(AutoReviewExtractionCall)
                .where(
                    AutoReviewExtractionCall.workflow_thread_id
                    == preview.workflow_thread_id
                )
                .order_by(AutoReviewExtractionCall.id)
            ).all()
        )
        required_rollout = (
            request.authorized_percentage_at_launch,
            request.rollout_authorization_generation,
            request.rollout_control_epoch,
        )
        if (
            validation_call is None
            or not extraction_calls
            or any(value is None for value in required_rollout)
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review provider snapshot is incomplete'
            )
        return _AutoApprovalLocator(
            review_item_id=preview.id,
            workflow_thread_id=workflow.thread_id,
            security_scope_id=workflow.security_scope_id,
            policy_version=actor.policy_version,
            validation_id=validation.id,
            validation_call_id=validation_call.id,
            validation_snapshot=_validation_provider_snapshot(
                validation,
                validation_call,
            ),
            extraction_snapshots=tuple(
                _extraction_provider_snapshot(call) for call in extraction_calls
            ),
            authorized_percentage_at_launch=int(required_rollout[0]),
            rollout_authorization_generation=int(required_rollout[1]),
            rollout_control_epoch=int(required_rollout[2]),
        )

    def _require_exact_reaffirmation_collision_clear(
        self,
        db: Session,
        *,
        item: ReviewItem,
        actor: ReviewResolutionActor,
        directive: ApprovalDirective | None,
        locator: _AutoApprovalLocator | None,
    ) -> None:
        if not isinstance(directive, ReuseExistingPromotion) or locator is None:
            raise AutoReviewAdmissionNotReady(
                'Automatic review exact reaffirmation is missing'
            )
        if item.item_type not in {'timeline_event', 'history_event'}:
            raise AutoReviewAdmissionNotReady(
                'Automatic review target type is not reusable'
            )
        preview = build_promotion_preview(item)
        normalized = preview.get('normalized_payload')
        if not isinstance(normalized, dict):
            raise AutoReviewAdmissionNotReady(
                'Automatic review candidate payload is invalid'
            )
        project_key = (item.payload or {}).get('project_key')
        if not isinstance(project_key, str):
            project_key = None
        entries: list[tuple[str, int, str, str]] = [
            (
                item.item_type,
                directive.expected_id,
                directive.expected_claim_fingerprint,
                str(normalized['title']),
            )
        ]
        candidate_primary = normalized_claim_fingerprint(
            item=item,
            security_scope_id=locator.security_scope_id,
            settings=self._settings,
        )
        if candidate_primary != directive.expected_claim_fingerprint:
            raise AutoReviewAdmissionNotReady(
                'Automatic review candidate claim is not exact'
            )
        if item.item_type == 'history_event':
            if (
                directive.expected_companion_id is None
                or directive.expected_companion_claim_fingerprint is None
            ):
                raise AutoReviewAdmissionNotReady(
                    'Automatic review companion reaffirmation is missing'
                )
            companion_claim = promoted_effect_fingerprint(
                knowledge_type='timeline_event',
                normalized_persisted_fields={
                    'title': str(normalized['title']),
                    'result_summary': str(normalized['reason']),
                },
                project_key=project_key,
                security_scope_id=locator.security_scope_id,
                settings=self._settings,
            )
            if companion_claim != directive.expected_companion_claim_fingerprint:
                raise AutoReviewAdmissionNotReady(
                    'Automatic review companion claim is not exact'
                )
            entries.append(
                (
                    'timeline_event',
                    directive.expected_companion_id,
                    companion_claim,
                    str(normalized['title']),
                )
            )
        project_hmac = trusted_project_scope_fingerprint(
            project_key=project_key,
            settings=self._settings,
        )
        for knowledge_type, knowledge_id, claim, title in entries:
            title_hmac = trusted_title_collision_bucket(
                item_type=knowledge_type,  # type: ignore[arg-type]
                normalized_title=title,
                settings=self._settings,
            )
            if db.scalar(
                build_hidden_collision_exists_statement(
                    knowledge_type=knowledge_type,
                    security_scope_id=locator.security_scope_id,
                    project_scope_hmac=project_hmac,
                    normalized_title_bucket_hmac=title_hmac,
                    visible_permission_levels=actor.allowed_permission_levels,
                )
            ):
                raise AutoReviewAdmissionNotReady(
                    'Automatic review hidden collision is present'
                )
            visible_statement = (
                select(TrustedKnowledgeFingerprint)
                .where(
                    TrustedKnowledgeFingerprint.knowledge_type == knowledge_type,
                    TrustedKnowledgeFingerprint.project_scope_hmac == project_hmac,
                    TrustedKnowledgeFingerprint.normalized_title_bucket_hmac
                    == title_hmac,
                    TrustedKnowledgeFingerprint.review_status == 'approved',
                    TrustedKnowledgeFingerprint.permission_level.in_(
                        actor.allowed_permission_levels
                    ),
                    or_(
                        and_(
                            TrustedKnowledgeFingerprint.scope_resolution == 'exact',
                            TrustedKnowledgeFingerprint.security_scope_id
                            == locator.security_scope_id,
                        ),
                        TrustedKnowledgeFingerprint.scope_resolution
                        == 'legacy_unknown',
                    ),
                )
                .order_by(TrustedKnowledgeFingerprint.id)
            )
            if db.get_bind().dialect.name == 'postgresql':
                visible_statement = visible_statement.with_for_update(read=True)
            visible = tuple(db.scalars(visible_statement).all())
            if len(visible) != 1:
                raise AutoReviewAdmissionNotReady(
                    'Automatic review visible collision is ambiguous'
                )
            row = visible[0]
            if (
                row.scope_resolution != 'exact'
                or row.security_scope_id != locator.security_scope_id
                or row.knowledge_id != knowledge_id
                or row.normalized_claim_fingerprint != claim
            ):
                raise AutoReviewAdmissionNotReady(
                    'Automatic review visible collision is not exact'
                )

    @staticmethod
    def _lock_and_require_provider_safety(
        db: Session,
        *,
        locator: _AutoApprovalLocator,
    ) -> dict[tuple[str, str, str, str], AutoReviewProviderSafetyState]:
        snapshots = (*locator.extraction_snapshots, locator.validation_snapshot)
        identities = sorted(
            {
                (
                    snapshot.purpose,
                    snapshot.provider,
                    snapshot.model,
                    snapshot.reasoning_effort,
                )
                for snapshot in snapshots
            }
        )
        statement = (
            select(AutoReviewProviderSafetyState)
            .where(
                tuple_(
                    AutoReviewProviderSafetyState.purpose,
                    AutoReviewProviderSafetyState.provider,
                    AutoReviewProviderSafetyState.model,
                    AutoReviewProviderSafetyState.reasoning_effort,
                ).in_(identities)
            )
            .order_by(
                AutoReviewProviderSafetyState.purpose,
                AutoReviewProviderSafetyState.provider,
                AutoReviewProviderSafetyState.model,
                AutoReviewProviderSafetyState.reasoning_effort,
            )
        )
        if db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update(read=True)
        rows = tuple(db.scalars(statement).all())
        safety = {
            (row.purpose, row.provider, row.model, row.reasoning_effort): row
            for row in rows
        }
        if len(safety) != len(identities):
            raise AutoReviewAdmissionNotReady(
                'Automatic review provider safety is incomplete'
            )
        for snapshot in snapshots:
            row = safety[
                (
                    snapshot.purpose,
                    snapshot.provider,
                    snapshot.model,
                    snapshot.reasoning_effort,
                )
            ]
            if row.breaker_open or not _provider_snapshot_matches_safety(
                snapshot,
                row,
            ):
                raise AutoReviewAdmissionNotReady(
                    'Automatic review provider safety is stale'
                )
        return safety

    @staticmethod
    def _lock_and_require_rollout(
        db: Session,
        *,
        locator: _AutoApprovalLocator | None,
    ) -> None:
        if locator is None:
            raise AutoReviewAdmissionNotReady('Automatic review rollout is missing')
        statement = select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id == locator.security_scope_id,
            AutoReviewRolloutState.policy_version == locator.policy_version,
        )
        if db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update(read=True)
        rollout = db.scalar(statement)
        if (
            rollout is None
            or rollout.breaker_open
            or locator.authorized_percentage_at_launch not in {10, 100}
            or rollout.max_authorized_percentage
            != locator.authorized_percentage_at_launch
            or rollout.authorization_generation
            != locator.rollout_authorization_generation
            or rollout.control_epoch != locator.rollout_control_epoch
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review rollout authorization is stale'
            )

    def _lock_and_require_auto_calls(
        self,
        db: Session,
        *,
        locator: _AutoApprovalLocator | None,
        safety: dict[tuple[str, str, str, str], AutoReviewProviderSafetyState],
    ) -> int:
        if locator is None:
            raise AutoReviewAdmissionNotReady('Automatic review call snapshot is missing')
        extraction_statement = (
            select(AutoReviewExtractionCall)
            .where(
                AutoReviewExtractionCall.workflow_thread_id
                == locator.workflow_thread_id
            )
            .order_by(AutoReviewExtractionCall.id)
        )
        validation_locators = tuple(
            db.scalars(
                select(AutoReviewValidation)
                .where(
                    AutoReviewValidation.review_item_id == locator.review_item_id,
                    AutoReviewValidation.workflow_thread_id
                    == locator.workflow_thread_id,
                    AutoReviewValidation.status == 'completed',
                    AutoReviewValidation.policy_version == locator.policy_version,
                )
                .order_by(AutoReviewValidation.id)
            ).all()
        )
        validation_call_ids = tuple(
            sorted(row.validation_call_id for row in validation_locators)
        )
        validation_call_statement = (
            select(AutoReviewValidationCall)
            .where(
                AutoReviewValidationCall.id.in_(validation_call_ids),
                AutoReviewValidationCall.workflow_thread_id
                == locator.workflow_thread_id,
            )
            .order_by(AutoReviewValidationCall.id)
        )
        validation_statement = (
            select(AutoReviewValidation)
            .where(
                AutoReviewValidation.review_item_id == locator.review_item_id,
                AutoReviewValidation.workflow_thread_id
                == locator.workflow_thread_id,
                AutoReviewValidation.status == 'completed',
                AutoReviewValidation.policy_version == locator.policy_version,
            )
            .order_by(AutoReviewValidation.id)
        )
        if db.get_bind().dialect.name == 'postgresql':
            extraction_statement = extraction_statement.with_for_update(read=True)
            validation_call_statement = validation_call_statement.with_for_update(
                read=True
            )
            validation_statement = validation_statement.with_for_update(read=True)
        extraction_calls = tuple(db.scalars(extraction_statement).all())
        validation_calls = tuple(db.scalars(validation_call_statement).all())
        validations = tuple(db.scalars(validation_statement).all())
        validation_call = validation_calls[0] if len(validation_calls) == 1 else None
        validation = validations[0] if len(validations) == 1 else None
        current_extraction = tuple(
            _extraction_provider_snapshot(call) for call in extraction_calls
        )
        if (
            validation_call is None
            or validation is None
            or validation.id != locator.validation_id
            or validation_call.id != locator.validation_call_id
            or current_extraction != locator.extraction_snapshots
            or _validation_provider_snapshot(validation, validation_call)
            != locator.validation_snapshot
            or any(call.status != 'completed' for call in extraction_calls)
            or validation_call.status != 'completed'
            or validation.status != 'completed'
            or validation.policy_version != locator.policy_version
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review call snapshot is stale'
            )
        workflow = db.get(AgentWorkflowThread, locator.workflow_thread_id)
        verifier = fingerprint_key_material_verifier(
            self._settings.agent_runtime_fingerprint_secret
        )
        if (
            workflow is None
            or workflow.evidence_version_hash != validation.evidence_version_hash
            or validation.fingerprint_key_version
            != self._settings.agent_runtime_fingerprint_key_version
            or validation.fingerprint_key_material_verifier != verifier
            or validation_call.fingerprint_key_version
            != self._settings.agent_runtime_fingerprint_key_version
            or validation_call.fingerprint_key_material_verifier != verifier
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review validation identity is stale'
            )
        for snapshot in (*current_extraction, locator.validation_snapshot):
            identity = (
                snapshot.purpose,
                snapshot.provider,
                snapshot.model,
                snapshot.reasoning_effort,
            )
            if identity not in safety or not _provider_snapshot_matches_safety(
                snapshot,
                safety[identity],
            ):
                raise AutoReviewAdmissionNotReady(
                    'Automatic review provider call identity is stale'
                )
        return validation.id

    @staticmethod
    def _require_auto_validation_multiplicity_after_item_lock(
        db: Session,
        *,
        locator: _AutoApprovalLocator,
    ) -> None:
        validation_ids = tuple(
            db.scalars(
                select(AutoReviewValidation.id)
                .where(
                    AutoReviewValidation.review_item_id
                    == locator.review_item_id,
                    AutoReviewValidation.workflow_thread_id
                    == locator.workflow_thread_id,
                    AutoReviewValidation.status == 'completed',
                    AutoReviewValidation.policy_version
                    == locator.policy_version,
                )
                .order_by(AutoReviewValidation.id)
            ).all()
        )
        if validation_ids != (locator.validation_id,):
            raise AutoReviewAdmissionNotReady(
                'Automatic review validation multiplicity changed'
            )

    def _require_current_owner_permissions(
        self,
        db: Session,
        *,
        item: ReviewItem,
        actor: ReviewResolutionActor,
    ) -> None:
        if (
            self._current_permission_resolver is None
            or item.workflow_thread_id is None
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review current owner permissions are unavailable'
            )
        workflow = db.scalar(
            select(AgentWorkflowThread)
            .where(AgentWorkflowThread.thread_id == item.workflow_thread_id)
            .execution_options(populate_existing=True)
        )
        if workflow is None:
            raise AutoReviewAdmissionNotReady(
                'Automatic review current owner is unavailable'
            )
        resolved = self._current_permission_resolver(workflow.owner_subject_id)
        if resolved is None or not isinstance(resolved, tuple) or len(resolved) != 2:
            raise AutoReviewAdmissionNotReady(
                'Automatic review current owner is unavailable'
            )
        resolved_subject_id, raw_levels = resolved
        if (
            resolved_subject_id != workflow.owner_subject_id
            or isinstance(raw_levels, (str, bytes))
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review current owner identity is stale'
            )
        current_levels = tuple(raw_levels)
        if any(level not in PERMISSION_ORDER for level in current_levels):
            raise AutoReviewAdmissionNotReady(
                'Automatic review current owner permissions are invalid'
            )
        effective_levels = frozenset(current_levels).intersection(
            actor.allowed_permission_levels
        )
        if item.permission_level not in effective_levels:
            raise AutoReviewAdmissionNotReady(
                'Automatic review item permission is no longer authorized'
            )
        source_ids = tuple(
            sorted(
                set(
                    db.scalars(
                        select(AgentWorkflowEvidenceRef.canonical_row_id)
                        .join(
                            ReviewItemEvidenceRef,
                            ReviewItemEvidenceRef.workflow_evidence_ref_id
                            == AgentWorkflowEvidenceRef.id,
                        )
                        .where(
                            ReviewItemEvidenceRef.review_item_id == item.id,
                            AgentWorkflowEvidenceRef.workflow_thread_id
                            == item.workflow_thread_id,
                        )
                    ).all()
                )
            )
        )
        sources = tuple(
            db.scalars(
                select(Source)
                .where(Source.id.in_(source_ids))
                .order_by(Source.id)
                .execution_options(populate_existing=True)
            ).all()
        )
        if (
            not source_ids
            or tuple(source.id for source in sources) != source_ids
            or any(source.permission_level not in effective_levels for source in sources)
        ):
            raise AutoReviewAdmissionNotReady(
                'Automatic review source permission is no longer authorized'
            )

    def transition_many(
        self,
        *,
        db: Session,
        item_ids: Sequence[int],
        action: ReviewAction,
        actor: ReviewResolutionActor,
        note: str | None = None,
    ) -> ReviewBatchTransitionResult:
        _assert_review_resolution_actor(actor)
        if actor.actor_type != 'human':
            _raise_review_permission_denied()
        requested_ids = sorted(set(item_ids))
        items = self._load_items(db, requested_ids, for_update=False)
        items_by_id = {item.id: item for item in items}
        results: list[ReviewTransitionResult] = []
        failed_items: list[dict[str, object]] = []
        skipped_items = [
            {'id': item_id, 'detail': 'Review item not found'}
            for item_id in requested_ids
            if item_id not in items_by_id
        ]

        for item_id in requested_ids:
            item = items_by_id.get(item_id)
            if item is None:
                continue
            try:
                with db.begin_nested():
                    result = self.transition(
                        db=db,
                        item_id=item.id,
                        action=action,
                        actor=actor,
                        note=note,
                    )
                results.append(result)
            except InvalidReviewTransition as exc:
                failed_items.append(
                    {
                        'id': item.id,
                        'code': exc.code,
                        'detail': str(exc),
                    }
                )
            except HTTPException as exc:
                failed_items.append({'id': item.id, 'detail': exc.detail})
            except ValueError as exc:
                failed_items.append({'id': item.id, 'detail': str(exc)})
            except IntegrityError:
                failed_items.append(
                    {
                        'id': item_id,
                        'detail': 'Review promotion provenance is incomplete',
                    }
                )

        return ReviewBatchTransitionResult(
            results=tuple(results),
            failed_items=tuple(failed_items),
            skipped_items=tuple(skipped_items),
        )

    @staticmethod
    def _load_items(
        db: Session,
        item_ids: Sequence[int],
        *,
        for_update: bool = True,
    ) -> list[ReviewItem]:
        if not item_ids:
            return []
        statement = (
            select(ReviewItem)
            .where(ReviewItem.id.in_(item_ids))
            .order_by(ReviewItem.id)
            .execution_options(populate_existing=True)
        )
        if for_update and db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update()
        return list(db.scalars(statement).all())

    def _transition_item(
        self,
        *,
        db: Session,
        item: ReviewItem,
        action: ReviewAction,
        actor: ReviewResolutionActor,
        note: str | None,
        approval_directive: ApprovalDirective | None,
        locked_auto_validation_id: int | None,
    ) -> ReviewTransitionResult:
        directive = self._authorize_action(
            item=item,
            action=action,
            actor=actor,
            approval_directive=approval_directive,
        )

        if item.status == 'approved' and action == 'approve':
            explicit = find_explicit_promotion(
                db,
                item=item,
                settings=self._settings,
            )
            return ReviewTransitionResult(
                item_id=item.id,
                status=item.status,
                replayed=True,
                promotion=(
                    _to_promotion_result(
                        _bundle_as_raw(
                            explicit,
                            effect=replay_effect_for_bundle(
                                db,
                                item=item,
                                bundle=explicit,
                            ),
                        )
                    )
                    if explicit is not None
                    else _to_promotion_result(find_review_item_promotion(db, item))
                ),
            )
        if item.status != 'pending_review':
            raise InvalidReviewTransition(action=action, status=item.status)

        reviewed_at = datetime.now(UTC)
        if action == 'approve':
            if directive is None:
                raise RuntimeError('Approval directive is required')
            _validate_source_evidence(item)
            validate_review_item_for_approval(item)
            validation_id = (
                None
                if actor.actor_type == 'human'
                else locked_auto_validation_id
            )
            if actor.actor_type == 'auto_policy' and validation_id is None:
                raise AutoReviewAdmissionNotReady(
                    'Auto approval validation is not locked'
                )
            reuse_bundle = (
                validate_reuse_bundle(
                    db,
                    item=item,
                    directive=directive,
                    settings=self._settings,
                )
                if isinstance(directive, ReuseExistingPromotion)
                else None
            )
            created_result: PromotionResult | None = None
            created_bundle: TrustedPromotionBundle | None = None
            if isinstance(directive, CreateNewPromotion):
                created_result = self._promote_exactly_once(db, item)
                if item.candidate_contract_version == 'c5-v1':
                    created_bundle = effects_for_created_promotion(
                        db,
                        item=item,
                        record_ids=created_result.created_record_ids,
                        timeline_ids=created_result.created_timeline_event_ids,
                        settings=self._settings,
                    )
            item.status = 'approved'
            item.reviewer_id = actor.subject_id
            item.reviewed_at = reviewed_at
            if actor.actor_type == 'human':
                item.resolution_source = 'human'
                item.resolution_policy_version = None
                item.auto_validation_id = None
            else:
                item.resolution_source = 'auto_policy'
                item.resolution_policy_version = actor.policy_version
                item.auto_validation_id = validation_id
            db.flush([item])
            if reuse_bundle is not None:
                record_or_verify_explicit_provenance(
                    db,
                    item=item,
                    bundle=reuse_bundle,
                    settings=self._settings,
                )
                promotion = _to_promotion_result(
                    _bundle_as_raw(reuse_bundle, effect='reaffirmed')
                )
            else:
                if created_result is None:
                    raise RuntimeError('Created promotion result is required')
                if created_bundle is not None:
                    record_or_verify_explicit_provenance(
                        db,
                        item=item,
                        bundle=created_bundle,
                        settings=self._settings,
                    )
                promotion = created_result
            if item.candidate_contract_version == 'c5-v1':
                document_drifted = (
                    CanonicalReviewEvidenceStalenessResolver(
                        settings=self._settings
                    )
                    .lock_current_documents_and_resolve_drift(db, item=item)
                )
                if document_drifted:
                    raise CanonicalEvidenceDrift(
                        'Review item canonical document has changed'
                    )
                projection_bundle = reuse_bundle or created_bundle
                if projection_bundle is None:
                    raise RuntimeError('C.5 projection bundle is required')
                project_approved_effects(
                    db,
                    effects=projection_bundle.effects,
                    settings=self._settings,
                    new_target_effects=created_bundle is not None,
                )
            elif promotion.created_timeline_event_ids:
                demote_trusted_fingerprint_projection(db)
        elif action == 'reject':
            item.status = 'rejected'
            item.reviewer_id = actor.subject_id
            item.reviewed_at = reviewed_at
            item.resolution_source = 'human'
            item.resolution_policy_version = None
            item.auto_validation_id = None
            db.flush([item])
            promotion = None
        elif action == 'needs_more_evidence':
            item.status = 'needs_more_evidence'
            item.reviewer_id = actor.subject_id
            item.reviewed_at = reviewed_at
            item.resolution_source = 'human'
            item.resolution_policy_version = None
            item.auto_validation_id = None
            payload = dict(item.payload or {})
            payload['needs_more_evidence'] = {
                'requested_at': reviewed_at.isoformat(),
                'requested_by': actor.subject_id,
                'note': (note or '').strip(),
                'source_count': len(item.source_snippets or []),
                'previous_status': 'pending_review',
            }
            item.payload = payload
            db.flush([item])
            promotion = None
        else:
            raise ValueError(f'Unsupported review action: {action}')

        return ReviewTransitionResult(
            item_id=item.id,
            status=item.status,
            replayed=False,
            promotion=promotion,
        )

    @staticmethod
    def _authorize_action(
        *,
        item: ReviewItem,
        action: ReviewAction,
        actor: ReviewResolutionActor,
        approval_directive: ApprovalDirective | None,
    ) -> ApprovalDirective | None:
        _assert_review_resolution_actor(actor)
        if approval_directive is not None:
            _assert_approval_directive(approval_directive)
        if item.permission_level not in actor.allowed_permission_levels:
            _raise_review_permission_denied()
        if actor.actor_type == 'human':
            if 'human_review' not in actor.capabilities:
                _raise_review_permission_denied()
            if isinstance(approval_directive, ReuseExistingPromotion):
                _raise_review_permission_denied()
            if action == 'approve':
                return approval_directive or CreateNewPromotion()
            if approval_directive is not None:
                raise ValueError('Approval directive is valid only for approval')
            return None
        if (
            actor.subject_id != SYSTEM_AUTO_REVIEW_ACTOR_ID
            or 'auto_review' not in actor.capabilities
            or actor.policy_version != AUTO_REVIEW_POLICY_VERSION
            or action != 'approve'
            or approval_directive is None
        ):
            _raise_review_permission_denied()
        return approval_directive

    @staticmethod
    def _promote_exactly_once(db: Session, item: ReviewItem) -> PromotionResult:
        try:
            with db.begin_nested():
                raw_result = promote_review_item(db, item)
        except IntegrityError as integrity_error:
            db.refresh(item)
            try:
                raw_result = find_review_item_promotion(db, item)
            except IncompleteReviewPromotionError:
                raise integrity_error from None
            if raw_result is None:
                raise
        result = _to_promotion_result(raw_result)
        if result is None:
            raise RuntimeError('Approved review item has no promotion result')
        return result


class ReviewEvidenceStalenessResolver(Protocol):
    def lock_item_and_resolve_drift(
        self,
        db: Session,
        *,
        item_id: int,
        before_item_lock: Callable[[], None] | None = None,
    ) -> tuple[ReviewItem, bool]: ...


class CanonicalReviewEvidenceStalenessResolver:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def lock_item_and_resolve_drift(
        self,
        db: Session,
        *,
        item_id: int,
        before_item_lock: Callable[[], None] | None = None,
    ) -> tuple[ReviewItem, bool]:
        preview = db.scalar(
            select(ReviewItem)
            .where(ReviewItem.id == item_id)
            .execution_options(populate_existing=True)
        )
        if preview is None:
            raise ValueError('Review item not found')
        if preview.workflow_thread_id is None:
            item = self._lock_review_item(db, item_id=item_id)
            return item, False
        workflow_thread_id = preview.workflow_thread_id

        locator_children = tuple(
            db.scalars(
                select(ReviewItemEvidenceRef).where(
                    ReviewItemEvidenceRef.review_item_id == item_id,
                    ReviewItemEvidenceRef.workflow_thread_id
                    == workflow_thread_id,
                )
            ).all()
        )
        locator_ref_ids = tuple(
            sorted(child.workflow_evidence_ref_id for child in locator_children)
        )
        locator_refs = tuple(
            db.scalars(
                select(AgentWorkflowEvidenceRef).where(
                    AgentWorkflowEvidenceRef.workflow_thread_id
                    == workflow_thread_id,
                ).order_by(AgentWorkflowEvidenceRef.id)
            ).all()
        )
        source_ids = tuple(
            sorted({ref.canonical_row_id for ref in locator_refs})
        )

        source_statement = (
            select(Source)
            .where(Source.id.in_(source_ids))
            .order_by(Source.id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
            source_statement = source_statement.with_for_update(read=True)
        sources = {
            source.id: source for source in db.scalars(source_statement).all()
        }

        workflow_statement = (
            select(AgentWorkflowThread)
            .where(AgentWorkflowThread.thread_id == workflow_thread_id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
            workflow_statement = workflow_statement.with_for_update()
        workflow = db.scalar(workflow_statement)
        if before_item_lock is not None:
            before_item_lock()
        item = self._lock_review_item(db, item_id=item_id)
        if workflow is None or item.workflow_thread_id != workflow_thread_id:
            return item, True

        child_statement = (
            select(ReviewItemEvidenceRef)
            .where(
                ReviewItemEvidenceRef.review_item_id == item.id,
                ReviewItemEvidenceRef.workflow_thread_id == workflow_thread_id,
            )
            .order_by(ReviewItemEvidenceRef.id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
            child_statement = child_statement.with_for_update()
        children = tuple(db.scalars(child_statement).all())
        locked_ref_ids = tuple(
            sorted(child.workflow_evidence_ref_id for child in children)
        )
        ref_statement = (
            select(AgentWorkflowEvidenceRef)
            .where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == workflow_thread_id,
            )
            .order_by(AgentWorkflowEvidenceRef.id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
            ref_statement = ref_statement.with_for_update()
        workflow_refs = tuple(db.scalars(ref_statement).all())
        refs_by_id = {ref.id: ref for ref in workflow_refs}
        refs = tuple(
            refs_by_id[ref_id]
            for ref_id in locked_ref_ids
            if ref_id in refs_by_id
        )
        locked_source_ids = tuple(
            sorted({ref.canonical_row_id for ref in workflow_refs})
        )
        if locked_source_ids != source_ids:
            raise CanonicalEvidenceDrift(
                'Canonical evidence changed during lock acquisition'
            )

        if not children:
            return item, item.candidate_contract_version == 'c5-v1'
        if (
            locked_ref_ids != locator_ref_ids
            or len(refs) != len(children)
            or tuple(ref.id for ref in workflow_refs)
            != tuple(ref.id for ref in locator_refs)
        ):
            return item, True
        for ref in workflow_refs:
            source = sources.get(ref.canonical_row_id)
            if source is None:
                return item, True
            if (
                ref.canonical_table != 'sources'
                or ref.canonical_source_type != source.source_type
                or ref.permission_level_snapshot != source.permission_level
                or source.server_content_signature_schema
                != 'server-source-content:v1'
                or source.server_content_signature != ref.content_signature
            ):
                return item, True
        derived_evidence_hash = build_keyed_fingerprint(
            [
                {
                    'source_type': ref.canonical_source_type,
                    'canonical_table': ref.canonical_table,
                    'canonical_row_id': ref.canonical_row_id,
                    'document_version_id': ref.document_version_id,
                    'external_revision': ref.external_revision,
                    'content_signature': ref.content_signature,
                    'permission_level': ref.permission_level_snapshot,
                    'content_fingerprint': ref.content_fingerprint,
                }
                for ref in sorted(workflow_refs, key=lambda row: row.ordinal)
            ],
            settings=self._settings,
            schema_version='review-evidence-versions:v1',
            policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        )
        if workflow.evidence_version_hash != derived_evidence_hash:
            return item, True
        source_permissions = [sources[ref.canonical_row_id].permission_level for ref in refs]
        if (
            not source_permissions
            or any(permission not in PERMISSION_ORDER for permission in source_permissions)
            or item.permission_level
            != max(source_permissions, key=lambda value: PERMISSION_ORDER[value])
        ):
            return item, True
        return item, False

    def lock_current_documents_and_resolve_drift(
        self,
        db: Session,
        *,
        item: ReviewItem,
    ) -> bool:
        if item.workflow_thread_id is None:
            return False
        refs = tuple(
            db.scalars(
                select(AgentWorkflowEvidenceRef)
                .join(
                    ReviewItemEvidenceRef,
                    ReviewItemEvidenceRef.workflow_evidence_ref_id
                    == AgentWorkflowEvidenceRef.id,
                )
                .where(
                    ReviewItemEvidenceRef.review_item_id == item.id,
                    AgentWorkflowEvidenceRef.workflow_thread_id
                    == item.workflow_thread_id,
                )
                .order_by(AgentWorkflowEvidenceRef.id)
                .execution_options(populate_existing=True)
            ).all()
        )
        source_ids = tuple(sorted({ref.canonical_row_id for ref in refs}))
        document_statement = (
            select(Document)
            .where(Document.source_id.in_(source_ids))
            .order_by(Document.id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
            document_statement = document_statement.with_for_update()
        documents = tuple(db.scalars(document_statement).all())
        current_version_ids = tuple(
            sorted(
                {
                    document.current_document_version_id
                    for document in documents
                    if document.current_document_version_id is not None
                }
                | {
                    ref.document_version_id
                    for ref in refs
                    if ref.document_version_id is not None
                }
            )
        )
        version_statement = (
            select(DocumentVersion)
            .where(DocumentVersion.id.in_(current_version_ids))
            .order_by(DocumentVersion.id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
            version_statement = version_statement.with_for_update()
        tuple(db.scalars(version_statement).all())
        documents_by_source: dict[int, list[Document]] = {}
        for document in documents:
            documents_by_source.setdefault(document.source_id, []).append(document)
        for ref in refs:
            source_documents = documents_by_source.get(ref.canonical_row_id, [])
            if ref.document_version_id is None:
                if not source_documents:
                    continue
                if (
                    len(source_documents) != 1
                    or source_documents[0].current_document_version_id is not None
                ):
                    return True
                continue
            if len(source_documents) != 1:
                return True
            if source_documents[0].current_document_version_id != ref.document_version_id:
                return True
        return False

    @staticmethod
    def _lock_review_item(db: Session, *, item_id: int) -> ReviewItem:
        items = ReviewTransitionService._load_items(db, [item_id])
        if not items:
            raise ValueError('Review item not found')
        return items[0]


class InternalReviewTransitionService:
    def __init__(
        self,
        *,
        evidence_staleness_resolver: ReviewEvidenceStalenessResolver,
    ) -> None:
        self._evidence_staleness_resolver = evidence_staleness_resolver

    def mark_evidence_stale(
        self,
        *,
        db: Session,
        item_id: int,
    ) -> ReviewTransitionResult:
        item, evidence_has_drifted = (
            self._evidence_staleness_resolver.lock_item_and_resolve_drift(
                db,
                item_id=item_id,
            )
        )
        if isinstance(
            self._evidence_staleness_resolver,
            CanonicalReviewEvidenceStalenessResolver,
        ):
            evidence_has_drifted = (
                self._evidence_staleness_resolver
                .lock_current_documents_and_resolve_drift(db, item=item)
                or evidence_has_drifted
            )
        if item.status != 'pending_review':
            raise InvalidReviewTransition(
                action='needs_more_evidence',
                status=item.status,
            )
        if not evidence_has_drifted:
            raise ValueError('Review item canonical evidence is current')
        reviewed_at = datetime.now(UTC)
        item.status = 'needs_more_evidence'
        item.reviewer_id = SYSTEM_AUTO_REVIEW_ACTOR_ID
        item.reviewed_at = reviewed_at
        item.resolution_source = 'auto_policy'
        item.resolution_policy_version = None
        item.auto_validation_id = None
        payload = dict(item.payload or {})
        payload['needs_more_evidence'] = {
            'requested_at': reviewed_at.isoformat(),
            'requested_by': SYSTEM_AUTO_REVIEW_ACTOR_ID,
            'reason_code': 'evidence_version_changed',
            'previous_status': 'pending_review',
        }
        item.payload = payload
        db.flush([item])
        return ReviewTransitionResult(
            item_id=item.id,
            status=item.status,
            replayed=False,
            promotion=None,
        )


def _raise_review_permission_denied() -> None:
    raise HTTPException(
        status_code=403,
        detail='Review approval permission required.',
    )


def _validate_source_evidence(item: ReviewItem) -> None:
    has_link = any(isinstance(value, str) and value.strip() for value in item.source_links or [])
    has_snippet = any(isinstance(value, str) and value.strip() for value in item.source_snippets or [])
    if not has_link or not has_snippet:
        raise ValueError('Review item requires source evidence')


def _to_promotion_result(raw_result: dict | None) -> PromotionResult | None:
    if raw_result is None:
        return None
    return PromotionResult(
        target_type=raw_result['target_type'],
        created_record_ids=tuple(raw_result['created_record_ids']),
        created_timeline_event_ids=tuple(raw_result['created_timeline_event_ids']),
        effect=raw_result.get('effect', 'created'),
    )


def _bundle_as_raw(
    bundle: TrustedPromotionBundle,
    *,
    effect: Literal['created', 'reaffirmed'],
) -> dict[str, object]:
    return {
        'target_type': bundle.target_type,
        'created_record_ids': list(bundle.record_ids),
        'created_timeline_event_ids': list(bundle.timeline_ids),
        'effect': effect,
    }


def _validation_provider_snapshot(
    validation: AutoReviewValidation,
    call: AutoReviewValidationCall,
) -> _ProviderCallSnapshot:
    return _ProviderCallSnapshot(
        purpose='validation',
        call_id=call.id,
        provider=validation.validator_provider,
        model=validation.validator_model,
        reasoning_effort=validation.reasoning_effort,
        state_version=call.provider_safety_state_version,
        cost_policy_version=call.cost_policy_version,
        token_estimator_version=call.token_estimator_version,
        tokenizer_encoding=call.tokenizer_encoding,
        reply_priming_tokens=call.reply_priming_tokens,
        framing_safety_tokens=call.framing_safety_tokens,
        input_usd_per_1m=call.input_usd_per_1m,
        output_usd_per_1m=call.output_usd_per_1m,
    )


def _extraction_provider_snapshot(
    call: AutoReviewExtractionCall,
) -> _ProviderCallSnapshot:
    return _ProviderCallSnapshot(
        purpose='extraction',
        call_id=call.id,
        provider=call.provider,
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        state_version=call.provider_safety_state_version,
        cost_policy_version=call.cost_policy_version,
        token_estimator_version=call.token_estimator_version,
        tokenizer_encoding=call.tokenizer_encoding,
        reply_priming_tokens=call.reply_priming_tokens,
        framing_safety_tokens=call.framing_safety_tokens,
        input_usd_per_1m=call.input_usd_per_1m,
        output_usd_per_1m=call.output_usd_per_1m,
    )


def _provider_snapshot_matches_safety(
    snapshot: _ProviderCallSnapshot,
    safety: AutoReviewProviderSafetyState,
) -> bool:
    return (
        snapshot.state_version == safety.state_version
        and snapshot.cost_policy_version == safety.authorized_cost_policy_version
        and snapshot.token_estimator_version == safety.token_estimator_version
        and snapshot.tokenizer_encoding == safety.tokenizer_encoding
        and snapshot.reply_priming_tokens == safety.reply_priming_tokens
        and snapshot.framing_safety_tokens == safety.framing_safety_tokens
        and snapshot.input_usd_per_1m == safety.input_usd_per_1m
        and snapshot.output_usd_per_1m == safety.output_usd_per_1m
    )
