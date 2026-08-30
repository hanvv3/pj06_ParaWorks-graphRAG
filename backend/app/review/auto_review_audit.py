from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.core.config import Settings
from backend.app.models import (
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRolloutControlEvent,
    AutoReviewRolloutState,
    ReviewItem,
)
from backend.app.review.actors import (
    ReviewResolutionActor,
    _assert_review_resolution_actor,
)
from backend.app.review.auto_review_rollout import (
    RolloutGateError,
    RolloutPercentage,
    RolloutSelectionInput,
    stable_rollout_selection,
)


@dataclass(frozen=True, slots=True)
class PromotionReservationInput:
    review_item_id: int
    security_scope_id: str
    policy_version: str
    selection: RolloutSelectionInput
    requested_percentage: RolloutPercentage
    stored_percentage: RolloutPercentage
    authorized_percentage: RolloutPercentage


@dataclass(frozen=True, slots=True)
class PromotionReservation:
    decision: AutoReviewPromotionDecision
    audit: AutoReviewPostAudit | None
    cache_hit: bool


class AutoReviewPromotionReservationService:
    """Reserves one immutable promotion/audit identity under rollout lock."""

    def __init__(self, db: Session, *, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def reserve_auto_promotion(
        self,
        request: PromotionReservationInput,
        *,
        locked_rollout: AutoReviewRolloutState | None = None,
        locked_item: ReviewItem | None = None,
    ) -> PromotionReservation:
        existing = self._db.scalar(
            select(AutoReviewPromotionDecision).where(
                AutoReviewPromotionDecision.review_item_id
                == request.review_item_id
            )
        )
        if existing is not None:
            self._verify_replay(existing, request=request)
            return PromotionReservation(
                decision=existing,
                audit=self._audit_for_decision(existing.id),
                cache_hit=True,
            )

        rollout_statement = select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id
            == request.security_scope_id,
            AutoReviewRolloutState.policy_version == request.policy_version,
        )
        item_statement = select(ReviewItem).where(
            ReviewItem.id == request.review_item_id
        )
        if self._db.get_bind().dialect.name == 'postgresql':
            rollout_statement = rollout_statement.with_for_update()
            item_statement = item_statement.with_for_update()
        rollout = locked_rollout or self._db.scalar(rollout_statement)
        item = locked_item or self._db.scalar(item_statement)
        if locked_rollout is not None and (
            locked_rollout.security_scope_id != request.security_scope_id
            or locked_rollout.policy_version != request.policy_version
        ):
            raise RolloutGateError('held rollout identity changed')
        if locked_item is not None and locked_item.id != request.review_item_id:
            raise RolloutGateError('held review item identity changed')
        if rollout is None or item is None or item.status != 'pending_review':
            raise RolloutGateError('promotion reservation is unavailable')
        self._verify_control(rollout=rollout, request=request)

        enforce = stable_rollout_selection(
            request.selection,
            percentage=request.stored_percentage,
            domain='auto-review-enforce-selection:v1',
            settings=self._settings,
        )
        if not enforce.selected:
            raise RolloutGateError('candidate is outside enforce selection')

        ordinal = rollout.enforce_promotion_ordinal + 1
        selection_result, audit_selected, audit_fingerprint = (
            self._select_audit(
                rollout=rollout,
                request=request,
                promotion_ordinal=ordinal,
                enforce_fingerprint=enforce.fingerprint,
            )
        )
        decision = AutoReviewPromotionDecision(
            review_item_id=request.review_item_id,
            security_scope_id=request.security_scope_id,
            policy_version=request.policy_version,
            rollout_authorization_generation=(
                request.selection.rollout_authorization_generation
            ),
            promotion_ordinal=ordinal,
            requested_percentage=request.requested_percentage,
            stored_percentage=request.stored_percentage,
            authorized_percentage=request.authorized_percentage,
            enforce_selection_fingerprint=enforce.fingerprint,
            audit_selection_fingerprint=audit_fingerprint,
            selection_result=selection_result,
            fingerprint_key_version=request.selection.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                request.selection.fingerprint_key_material_verifier
            ),
        )
        self._db.add(decision)
        self._db.flush()
        audit: AutoReviewPostAudit | None = None
        if audit_selected:
            audit = AutoReviewPostAudit(
                review_item_id=request.review_item_id,
                promotion_decision_id=decision.id,
                sample_cohort=selection_result,
                status='pending',
                outcome=None,
            )
            self._db.add(audit)
            rollout.post_audit_selected_count += 1
            if selection_result == 'first_50':
                rollout.pending_mandatory_audit_count += 1
        rollout.enforce_promotion_ordinal = ordinal
        rollout.state_version += 1
        self._db.flush()
        return PromotionReservation(
            decision=decision,
            audit=audit,
            cache_hit=False,
        )

    def _verify_control(
        self,
        *,
        rollout: AutoReviewRolloutState,
        request: PromotionReservationInput,
    ) -> None:
        selection = request.selection
        expected_verifier = fingerprint_key_material_verifier(
            self._settings.agent_runtime_fingerprint_secret
        )
        if (
            rollout.breaker_open
            or rollout.control_epoch != selection.rollout_control_epoch
            or rollout.authorization_generation
            != selection.rollout_authorization_generation
            or rollout.max_authorized_percentage
            < request.authorized_percentage
            or selection.policy_version != request.policy_version
            or selection.requested_percentage != request.requested_percentage
            or selection.authorized_percentage != request.authorized_percentage
            or selection.fingerprint_key_version
            != self._settings.agent_runtime_fingerprint_key_version
            or selection.fingerprint_key_material_verifier != expected_verifier
            or request.stored_percentage
            > min(request.requested_percentage, request.authorized_percentage)
            or request.stored_percentage not in {10, 100}
        ):
            raise RolloutGateError('promotion rollout control changed')

    def _select_audit(
        self,
        *,
        rollout: AutoReviewRolloutState,
        request: PromotionReservationInput,
        promotion_ordinal: int,
        enforce_fingerprint: str,
    ) -> tuple[
        Literal['first_50', 'sample_10', 'sample_2', 'not_selected'],
        bool,
        str,
    ]:
        payload = {
            **asdict(request.selection),
            'promotion_ordinal': promotion_ordinal,
            'enforce_selection_fingerprint': enforce_fingerprint,
            'confirmed_mandatory_audit_count': (
                rollout.confirmed_mandatory_audit_count
            ),
            'pending_mandatory_audit_count': (
                rollout.pending_mandatory_audit_count
            ),
        }
        fingerprint = build_keyed_fingerprint(
            payload,
            settings=self._settings,
            schema_version='auto-review-audit-selection:v1',
            policy_version=request.policy_version,
        )
        if rollout.confirmed_mandatory_audit_count < 50:
            if (
                rollout.confirmed_mandatory_audit_count
                + rollout.pending_mandatory_audit_count
                >= 50
            ):
                raise RolloutGateError('mandatory audit slots are full')
            return 'first_50', True, fingerprint
        percentage = 2 if request.stored_percentage == 100 else 10
        selected = int(fingerprint, 16) % 100 < percentage
        if not selected:
            return 'not_selected', False, fingerprint
        return ('sample_2' if percentage == 2 else 'sample_10'), True, fingerprint

    def _audit_for_decision(
        self,
        decision_id: int,
    ) -> AutoReviewPostAudit | None:
        return self._db.scalar(
            select(AutoReviewPostAudit).where(
                AutoReviewPostAudit.promotion_decision_id == decision_id
            )
        )

    @staticmethod
    def _verify_replay(
        decision: AutoReviewPromotionDecision,
        *,
        request: PromotionReservationInput,
    ) -> None:
        expected = (
            request.security_scope_id,
            request.policy_version,
            request.selection.rollout_authorization_generation,
            request.requested_percentage,
            request.stored_percentage,
            request.authorized_percentage,
            request.selection.fingerprint_key_version,
            request.selection.fingerprint_key_material_verifier,
        )
        actual = (
            decision.security_scope_id,
            decision.policy_version,
            decision.rollout_authorization_generation,
            decision.requested_percentage,
            decision.stored_percentage,
            decision.authorized_percentage,
            decision.fingerprint_key_version,
            decision.fingerprint_key_material_verifier,
        )
        if actual != expected:
            raise RolloutGateError('promotion reservation replay changed')


AuditOutcome = Literal[
    'confirmed',
    'incorrect',
    'permission_violation',
    'source_version_violation',
    'policy_violation',
]


@dataclass(frozen=True, slots=True)
class PostAuditTransitionResult:
    audit_id: int
    changed: bool
    replayed: bool
    remediation_required: bool


class AutoReviewPostAuditTransitionService:
    """The single human writer for selected post-audit outcomes."""

    def __init__(self, db: Session, *, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def ensure_manual_audit(
        self,
        *,
        review_item_id: int,
        actor: ReviewResolutionActor,
    ) -> AutoReviewPostAudit:
        """Create the bounded manual audit for an unsampled auto approval."""
        _assert_review_resolution_actor(actor)
        if actor.actor_type != 'human' or 'human_review' not in actor.capabilities:
            raise RolloutGateError('human audit authority is required')
        item = self._db.get(ReviewItem, review_item_id)
        if (
            item is None
            or item.permission_level not in actor.allowed_permission_levels
            or item.status not in {'approved', 'revoked'}
            or item.resolution_source != 'auto_policy'
        ):
            raise RolloutGateError('post audit is unavailable')
        decision = self._db.scalar(
            select(AutoReviewPromotionDecision).where(
                AutoReviewPromotionDecision.review_item_id == review_item_id
            )
        )
        if decision is None:
            raise RolloutGateError('post audit promotion is unavailable')
        existing = self._db.scalar(
            select(AutoReviewPostAudit).where(
                AutoReviewPostAudit.review_item_id == review_item_id
            )
        )
        if existing is not None:
            return existing
        if decision.selection_result != 'not_selected':
            raise RolloutGateError('selected post audit is unavailable')
        audit = AutoReviewPostAudit(
            review_item_id=review_item_id,
            promotion_decision_id=decision.id,
            sample_cohort='manual',
            status='pending',
            outcome=None,
        )
        self._db.add(audit)
        self._db.flush([audit])
        return audit

    def complete(
        self,
        *,
        audit_id: int,
        actor: ReviewResolutionActor,
        outcome: AuditOutcome,
        reason: str,
    ) -> PostAuditTransitionResult:
        _assert_review_resolution_actor(actor)
        if actor.actor_type != 'human' or 'human_review' not in actor.capabilities:
            raise RolloutGateError('human audit authority is required')
        normalized_reason = ' '.join(reason.split())
        if not 1 <= len(normalized_reason) <= 500:
            raise RolloutGateError('audit reason must contain 1 to 500 characters')
        actor_hmac = build_keyed_fingerprint(
            {'subject_id': actor.subject_id},
            settings=self._settings,
            schema_version='auto-review-audit-actor:v1',
            policy_version='auto-review-audit-actor:v1',
        )
        locator = self._db.get(AutoReviewPostAudit, audit_id)
        if locator is None:
            raise RolloutGateError('post audit is unavailable')
        decision = self._db.get(
            AutoReviewPromotionDecision, locator.promotion_decision_id
        )
        if decision is None:
            raise RolloutGateError('post audit promotion is unavailable')
        rollout_statement = select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id
            == decision.security_scope_id,
            AutoReviewRolloutState.policy_version == decision.policy_version,
        )
        audit_statement = select(AutoReviewPostAudit).where(
            AutoReviewPostAudit.id == audit_id
        )
        if self._db.get_bind().dialect.name == 'postgresql':
            rollout_statement = rollout_statement.with_for_update()
            audit_statement = audit_statement.with_for_update()
        rollout = self._db.scalar(rollout_statement)
        audit = self._db.scalar(audit_statement)
        if rollout is None or audit is None:
            raise RolloutGateError('post audit state is unavailable')
        if audit.status != 'pending':
            if (
                audit.outcome == outcome
                and audit.audit_reason == normalized_reason
                and audit.auditor_subject_hmac == actor_hmac
            ):
                return PostAuditTransitionResult(
                    audit_id=audit.id,
                    changed=False,
                    replayed=True,
                    remediation_required=audit.status
                    == 'remediation_required',
                )
            raise RolloutGateError('post audit outcome is immutable')

        now = datetime.now(UTC)
        if decision.selection_result == 'first_50':
            if rollout.pending_mandatory_audit_count <= 0:
                raise RolloutGateError('mandatory audit counter is inconsistent')
            rollout.pending_mandatory_audit_count -= 1
        critical = outcome != 'confirmed'
        rollout.post_audit_completed_count += 1
        if critical:
            rollout.post_audit_critical_count += 1
        elif decision.selection_result == 'first_50':
            rollout.confirmed_mandatory_audit_count += 1

        audit.outcome = outcome
        audit.auditor_subject_hmac = actor_hmac
        audit.auditor_fingerprint_key_version = (
            self._settings.agent_runtime_fingerprint_key_version
        )
        audit.auditor_fingerprint_key_material_verifier = (
            fingerprint_key_material_verifier(
                self._settings.agent_runtime_fingerprint_secret
            )
        )
        audit.audit_reason = normalized_reason
        audit.audited_at = now
        if critical:
            audit.status = 'remediation_required'
            audit.system_resolution_code = 'revoke_pending'
            audit.remediation_code = outcome
            self._open_breaker(
                rollout=rollout,
                actor_hmac=actor_hmac,
                reason_code=outcome,
                now=now,
            )
        else:
            rollout.state_version += 1
            rollout.updated_at = now
            audit.status = 'completed'
            audit.system_resolution_code = None
            audit.remediation_code = None
        self._db.flush()
        return PostAuditTransitionResult(
            audit_id=audit.id,
            changed=True,
            replayed=False,
            remediation_required=critical,
        )

    def _open_breaker(
        self,
        *,
        rollout: AutoReviewRolloutState,
        actor_hmac: str,
        reason_code: str,
        now: datetime,
    ) -> None:
        prior_state_version = rollout.state_version
        prior_control_epoch = rollout.control_epoch
        prior_percentage = rollout.max_authorized_percentage
        prior_generation = rollout.authorization_generation
        prior_breaker_open = rollout.breaker_open
        if prior_breaker_open:
            rollout.state_version += 1
            rollout.updated_at = now
            return
        rollout.state_version += 1
        rollout.control_epoch += 1
        rollout.breaker_open = True
        rollout.breaker_reason_code = reason_code
        rollout.breaker_opened_at = now
        rollout.max_authorized_percentage = 0
        rollout.authorization_generation += 1
        rollout.updated_at = now
        event = AutoReviewRolloutControlEvent(
            rollout_state_id=rollout.id,
            security_scope_id=rollout.security_scope_id,
            policy_version=rollout.policy_version,
            event_sequence=rollout.last_event_sequence + 1,
            event_kind='breaker_opened',
            prior_state_version=prior_state_version,
            new_state_version=rollout.state_version,
            prior_control_epoch=prior_control_epoch,
            new_control_epoch=rollout.control_epoch,
            prior_max_authorized_percentage=prior_percentage,
            new_max_authorized_percentage=0,
            prior_breaker_open=prior_breaker_open,
            new_breaker_open=True,
            prior_authorization_generation=prior_generation,
            new_authorization_generation=rollout.authorization_generation,
            reason_code=reason_code,
            regression_gate_reference=rollout.regression_gate_reference,
            actor_subject_hmac=actor_hmac,
            fingerprint_key_version=(
                self._settings.agent_runtime_fingerprint_key_version
            ),
            fingerprint_key_material_verifier=(
                fingerprint_key_material_verifier(
                    self._settings.agent_runtime_fingerprint_secret
                )
            ),
            created_at=now,
        )
        self._db.add(event)
        self._db.flush()
        rollout.last_event_sequence = event.event_sequence
        rollout.last_event_id = event.id
