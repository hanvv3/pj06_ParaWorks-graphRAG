from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.canonical_sources import build_keyed_fingerprint
from backend.app.core.config import Settings
from backend.app.models import (
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRevocationAssessment,
    AutoReviewRolloutState,
    ReviewItem,
)
from backend.app.review.actors import (
    ReviewResolutionActor,
    _assert_review_resolution_actor,
    auto_review_actor,
)
from backend.app.review.auto_review_audit import (
    AutoReviewPostAuditTransitionService,
)
from backend.app.review.auto_review_revoke import (
    AutoReviewRevokeService,
    QualityRevokeContext,
    mint_quality_revoke_context,
)
from backend.app.review.auto_review_rollout import RolloutGateError

QualityReasonCode = Literal[
    'incorrect_content',
    'permission_violation',
    'wrong_source_version',
    'policy_violation',
]

_QUALITY_REASONS = {
    'incorrect_content',
    'permission_violation',
    'wrong_source_version',
    'policy_violation',
}
_OUTCOME_BY_REASON: dict[str, str] = {
    'incorrect_content': 'incorrect',
    'permission_violation': 'permission_violation',
    'wrong_source_version': 'source_version_violation',
    'policy_violation': 'policy_violation',
}


class QualityRevokeRefused(ValueError):  # noqa: N818 - bounded domain refusal
    pass


class QualityRevokeCallback(Protocol):
    def __call__(
        self,
        *,
        context: QualityRevokeContext,
        actor: ReviewResolutionActor,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class QualityRevokeResult:
    review_item_id: int
    replayed: bool
    revoked: bool
    remediation_required: bool


class AutoReviewQualityRevokeService:
    """Commits immutable quality authority before exact knowledge revoke."""

    def __init__(
        self,
        db: Session,
        *,
        settings: Settings,
        revoke_callback: QualityRevokeCallback | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._revoke_callback = revoke_callback

    def revoke_quality(
        self,
        *,
        review_item_id: int,
        actor: ReviewResolutionActor,
        reason_code: QualityReasonCode,
        reason: str,
    ) -> QualityRevokeResult:
        _assert_review_resolution_actor(actor)
        if reason_code == 'business_withdrawal':
            raise QualityRevokeRefused('business_withdrawal uses the direct path')
        if reason_code not in _QUALITY_REASONS:
            raise QualityRevokeRefused('invalid_quality_reason')
        if actor.actor_type != 'human' or 'human_review' not in actor.capabilities:
            raise QualityRevokeRefused('forbidden')
        normalized_reason = ' '.join(reason.split())
        if not 1 <= len(normalized_reason) <= 500:
            raise QualityRevokeRefused('reason must contain 1 to 500 characters')

        context, already_completed = self._commit_quality_authority(
            review_item_id=review_item_id,
            actor=actor,
            reason_code=reason_code,
            reason=normalized_reason,
        )
        if already_completed:
            return QualityRevokeResult(
                review_item_id=review_item_id,
                replayed=True,
                revoked=True,
                remediation_required=False,
            )
        try:
            callback = self._revoke_callback
            if callback is None:
                callback = AutoReviewRevokeService(
                    self._db, settings=self._settings
                ).revoke_quality
            callback(context=context, actor=actor)
        except Exception:  # the committed breaker must survive bounded failures
            self._finalize(context=context, succeeded=False)
            return QualityRevokeResult(
                review_item_id=review_item_id,
                replayed=False,
                revoked=False,
                remediation_required=True,
            )
        self._finalize(context=context, succeeded=True)
        return QualityRevokeResult(
            review_item_id=review_item_id,
            replayed=False,
            revoked=True,
            remediation_required=False,
        )

    def recover_pending_remediation(self, *, limit: int = 100) -> int:
        if not 1 <= limit <= 100:
            raise QualityRevokeRefused('recovery limit must be between 1 and 100')
        pending: list[tuple[str, int, int, int, str]] = []
        audits = tuple(
            self._db.scalars(
                select(AutoReviewPostAudit)
                .where(
                    AutoReviewPostAudit.status == 'remediation_required',
                    AutoReviewPostAudit.system_resolution_code.in_(
                        {'revoke_pending', 'revoke_failed'}
                    ),
                )
                .order_by(AutoReviewPostAudit.id)
                .limit(limit)
            ).all()
        )
        for audit in audits:
            assessment = self._db.scalar(
                select(AutoReviewRevocationAssessment).where(
                    AutoReviewRevocationAssessment.review_item_id
                    == audit.review_item_id
                )
            )
            if assessment is not None:
                pending.append(
                    (
                        'audit',
                        audit.id,
                        audit.review_item_id,
                        assessment.id,
                        assessment.reason_code,
                    )
                )
        remaining = limit - len(pending)
        if remaining > 0:
            corrections = tuple(
                self._db.scalars(
                    select(AutoReviewAuditCorrection)
                    .where(
                        AutoReviewAuditCorrection.status
                        == 'remediation_required',
                        AutoReviewAuditCorrection.system_resolution_code.in_(
                            {'revoke_pending', 'revoke_failed'}
                        ),
                    )
                    .order_by(AutoReviewAuditCorrection.id)
                    .limit(remaining)
                ).all()
            )
            for correction in corrections:
                assessment = self._db.get(
                    AutoReviewRevocationAssessment, correction.assessment_id
                )
                if assessment is not None:
                    pending.append(
                        (
                            'correction',
                            correction.id,
                            correction.review_item_id,
                            assessment.id,
                            assessment.reason_code,
                        )
                    )
        self._db.rollback()  # unlocked locator scan ends before normal revoke order
        recovered = 0
        actor = auto_review_actor(
            policy_version='auto-review-quality-recovery:v1'
        )
        for kind, remediation_id, item_id, assessment_id, reason_code in pending:
            context = mint_quality_revoke_context(
                self._db,
                review_item_id=item_id,
                assessment_id=assessment_id,
                reason_code=reason_code,  # type: ignore[arg-type]
                remediation_kind=kind,  # type: ignore[arg-type]
                remediation_id=remediation_id,
            )
            callback = self._revoke_callback
            if callback is None:
                callback = AutoReviewRevokeService(
                    self._db, settings=self._settings
                ).revoke_quality
            try:
                callback(context=context, actor=actor)
            except Exception:
                self._finalize(context=context, succeeded=False)
                continue
            self._finalize(context=context, succeeded=True)
            recovered += 1
        return recovered

    def _commit_quality_authority(
        self,
        *,
        review_item_id: int,
        actor: ReviewResolutionActor,
        reason_code: QualityReasonCode,
        reason: str,
    ) -> tuple[QualityRevokeContext, bool]:
        try:
            item_statement = select(ReviewItem).where(ReviewItem.id == review_item_id)
            if self._db.get_bind().dialect.name == 'postgresql':
                item_statement = item_statement.with_for_update()
            item = self._db.scalar(item_statement)
            if item is None or item.permission_level not in actor.allowed_permission_levels:
                raise QualityRevokeRefused('not_found')
            if item.resolution_source != 'auto_policy' or item.status not in {
                'approved',
                'revoked',
            }:
                raise QualityRevokeRefused('unsupported_transition')
            decision = self._db.scalar(
                select(AutoReviewPromotionDecision).where(
                    AutoReviewPromotionDecision.review_item_id == item.id
                )
            )
            if decision is None:
                raise QualityRevokeRefused('promotion_decision_required')
            rollout_statement = select(AutoReviewRolloutState).where(
                AutoReviewRolloutState.security_scope_id == decision.security_scope_id,
                AutoReviewRolloutState.policy_version == decision.policy_version,
            )
            if self._db.get_bind().dialect.name == 'postgresql':
                rollout_statement = rollout_statement.with_for_update()
            rollout = self._db.scalar(rollout_statement)
            if rollout is None:
                raise QualityRevokeRefused('rollout_state_required')

            actor_hmac = self._actor_hmac(actor)
            assessment = self._db.scalar(
                select(AutoReviewRevocationAssessment).where(
                    AutoReviewRevocationAssessment.review_item_id == item.id
                )
            )
            if assessment is not None and (
                assessment.reason_code != reason_code
                or assessment.actor_subject_hmac != actor_hmac
            ):
                raise QualityRevokeRefused('revoke_reason_conflict')
            if assessment is None:
                assessment = AutoReviewRevocationAssessment(
                    review_item_id=item.id,
                    reason_code=reason_code,
                    actor_subject_hmac=actor_hmac,
                    actor_fingerprint_key_version=(
                        self._settings.agent_runtime_fingerprint_key_version
                    ),
                    actor_fingerprint_key_material_verifier=(
                        fingerprint_key_material_verifier(
                            self._settings.agent_runtime_fingerprint_secret
                        )
                    ),
                )
                self._db.add(assessment)
                self._db.flush([assessment])

            audit = self._db.scalar(
                select(AutoReviewPostAudit).where(
                    AutoReviewPostAudit.review_item_id == item.id
                )
            )
            if audit is None:
                audit = AutoReviewPostAudit(
                    review_item_id=item.id,
                    promotion_decision_id=decision.id,
                    sample_cohort='manual',
                    status='pending',
                    outcome=None,
                )
                self._db.add(audit)
                self._db.flush([audit])

            if audit.status == 'pending':
                AutoReviewPostAuditTransitionService(
                    self._db, settings=self._settings
                ).complete(
                    audit_id=audit.id,
                    actor=actor,
                    outcome=_OUTCOME_BY_REASON[reason_code],  # type: ignore[arg-type]
                    reason=reason,
                )
                context = mint_quality_revoke_context(
                    self._db,
                    review_item_id=item.id,
                    assessment_id=assessment.id,
                    reason_code=reason_code,
                    remediation_kind='audit',
                    remediation_id=audit.id,
                )
                self._db.commit()
                return context, False

            if audit.status == 'completed' and audit.outcome == 'confirmed':
                correction = self._db.scalar(
                    select(AutoReviewAuditCorrection).where(
                        AutoReviewAuditCorrection.review_item_id == item.id
                    )
                )
                if correction is None:
                    correction = AutoReviewAuditCorrection(
                        post_audit_id=audit.id,
                        review_item_id=item.id,
                        assessment_id=assessment.id,
                        effective_outcome=_OUTCOME_BY_REASON[reason_code],
                        status='remediation_required',
                        system_resolution_code='revoke_pending',
                        actor_subject_hmac=actor_hmac,
                        actor_fingerprint_key_version=(
                            self._settings.agent_runtime_fingerprint_key_version
                        ),
                        actor_fingerprint_key_material_verifier=(
                            fingerprint_key_material_verifier(
                                self._settings.agent_runtime_fingerprint_secret
                            )
                        ),
                    )
                    self._db.add(correction)
                    rollout.corrected_critical_count += 1
                    AutoReviewPostAuditTransitionService(
                        self._db, settings=self._settings
                    )._open_breaker(  # same locked rollout and event writer
                        rollout=rollout,
                        actor_hmac=actor_hmac,
                        reason_code=_OUTCOME_BY_REASON[reason_code],
                        now=datetime.now(UTC),
                    )
                    self._db.flush([correction])
                context = mint_quality_revoke_context(
                    self._db,
                    review_item_id=item.id,
                    assessment_id=assessment.id,
                    reason_code=reason_code,
                    remediation_kind='correction',
                    remediation_id=correction.id,
                )
                already_completed = correction.status == 'completed'
                self._db.commit()
                return context, already_completed

            if audit.outcome != _OUTCOME_BY_REASON[reason_code]:
                raise QualityRevokeRefused('revoke_reason_conflict')
            context = mint_quality_revoke_context(
                self._db,
                review_item_id=item.id,
                assessment_id=assessment.id,
                reason_code=reason_code,
                remediation_kind='audit',
                remediation_id=audit.id,
            )
            already_completed = audit.status == 'completed'
            self._db.commit()
            return context, already_completed
        except Exception:
            self._db.rollback()
            raise

    def _finalize(self, *, context: QualityRevokeContext, succeeded: bool) -> None:
        if context.remediation_kind == 'audit':
            row = self._db.get(AutoReviewPostAudit, context.remediation_id)
            if row is None:
                raise RolloutGateError('quality remediation audit disappeared')
            if succeeded:
                row.status = 'completed'
                row.system_resolution_code = None
                row.remediation_code = None
            else:
                row.status = 'remediation_required'
                row.system_resolution_code = 'revoke_failed'
        else:
            correction = self._db.get(
                AutoReviewAuditCorrection, context.remediation_id
            )
            if correction is None:
                raise RolloutGateError('quality remediation correction disappeared')
            if succeeded:
                correction.status = 'completed'
                correction.system_resolution_code = 'revoked'
                correction.completed_at = datetime.now(UTC)
            else:
                correction.status = 'remediation_required'
                correction.system_resolution_code = 'revoke_failed'
        self._db.commit()

    def _actor_hmac(self, actor: ReviewResolutionActor) -> str:
        return build_keyed_fingerprint(
            {'subject_id': actor.subject_id},
            settings=self._settings,
            schema_version='auto-review-audit-actor:v1',
            policy_version='auto-review-audit-actor:v1',
        )
