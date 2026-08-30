from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import (
    AgentWorkflowThread,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRolloutState,
    AutoReviewValidation,
    ReviewItem,
)
from backend.app.review.auto_review_revoke import (
    _SOURCE_INVALIDATION_CONTEXTS_INFO_KEY,
    SourceInvalidationRevokeContext,
)


@dataclass(frozen=True, slots=True)
class SourceInvalidationAuditResult:
    review_item_id: int
    changed: bool
    replayed: bool


class AutoReviewAuditTransitionStore:
    """The single source-reconciliation writer for selected audit transitions."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def complete_source_invalidated(
        self, context: SourceInvalidationRevokeContext
    ) -> SourceInvalidationAuditResult:
        self._validate_context(context)
        review_item_id = context.review_item_id
        statement = select(AutoReviewPostAudit).where(
            AutoReviewPostAudit.review_item_id == review_item_id
        )
        audit = self._db.scalar(statement)
        if audit is None:
            return SourceInvalidationAuditResult(
                review_item_id=review_item_id,
                changed=False,
                replayed=False,
            )
        if audit.status != 'pending':
            replayed = bool(
                audit.status == 'completed'
                and audit.outcome is None
                and audit.system_resolution_code
                == 'source_invalidated_before_audit'
            )
            return SourceInvalidationAuditResult(
                review_item_id=review_item_id,
                changed=False,
                replayed=replayed,
            )
        decision = self._db.get(
            AutoReviewPromotionDecision, audit.promotion_decision_id
        )
        if decision is None or decision.review_item_id != review_item_id:
            raise ValueError('source invalidation audit promotion is missing')
        rollout_statement = select(AutoReviewRolloutState).where(
            AutoReviewRolloutState.security_scope_id
            == decision.security_scope_id,
            AutoReviewRolloutState.policy_version == decision.policy_version,
        )
        rollout = self._db.scalar(rollout_statement)
        if rollout is None:
            raise ValueError('source invalidation audit rollout is missing')
        if decision.selection_result == 'first_50':
            if rollout.pending_mandatory_audit_count <= 0:
                raise ValueError(
                    'source invalidation mandatory audit counter is inconsistent'
                )
            rollout.pending_mandatory_audit_count -= 1
        rollout.invalidated_before_audit_count += 1
        rollout.updated_at = datetime.now(UTC)
        audit.status = 'completed'
        audit.outcome = None
        audit.system_resolution_code = 'source_invalidated_before_audit'
        audit.remediation_code = None
        audit.auditor_subject_hmac = None
        audit.auditor_fingerprint_key_version = None
        audit.auditor_fingerprint_key_material_verifier = None
        audit.audit_reason = None
        audit.audited_at = datetime.now(UTC)
        return SourceInvalidationAuditResult(
            review_item_id=review_item_id,
            changed=True,
            replayed=False,
        )

    def _validate_context(
        self, context: SourceInvalidationRevokeContext
    ) -> None:
        if not isinstance(context, SourceInvalidationRevokeContext):
            raise TypeError('A source-invalidation context is required')
        if context.session_identity != id(self._db):
            raise TypeError('Source-invalidation context belongs to another session')
        issued = self._db.info.get(
            _SOURCE_INVALIDATION_CONTEXTS_INFO_KEY, {}
        ).get(id(context))
        if issued is not context:
            raise TypeError(
                'Source-invalidation context was not issued by reconciliation'
            )


@dataclass(frozen=True, slots=True)
class ShadowComparisonResult:
    review_item_id: int
    changed: bool
    supported: bool
    excluded: bool


class AutoReviewShadowComparisonStore:
    """Records one later human outcome against an unchanged shadow prediction."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def record(
        self,
        *,
        item: ReviewItem,
        locked_rollout: AutoReviewRolloutState,
    ) -> ShadowComparisonResult:
        if item.workflow_thread_id is None:
            return ShadowComparisonResult(item.id, False, False, False)
        validations = tuple(
            self._db.scalars(
                select(AutoReviewValidation).where(
                    AutoReviewValidation.review_item_id == item.id,
                    AutoReviewValidation.workflow_thread_id
                    == item.workflow_thread_id,
                    AutoReviewValidation.status == 'completed',
                    AutoReviewValidation.shadow_comparison_status == 'pending',
                )
            ).all()
        )
        if len(validations) != 1:
            return ShadowComparisonResult(item.id, False, False, False)
        validation = validations[0]
        thread = self._db.get(AgentWorkflowThread, item.workflow_thread_id)
        if (
            thread is None
            or locked_rollout.security_scope_id != thread.security_scope_id
            or locked_rollout.policy_version != validation.policy_version
        ):
            raise ValueError('shadow comparison rollout identity changed')
        now = datetime.now(UTC)
        if thread.evidence_version_hash != validation.evidence_version_hash:
            validation.shadow_comparison_status = 'excluded'
            validation.shadow_exclusion_code = 'evidence_version_changed'
            validation.shadow_human_resolution = None
            validation.shadow_compared_at = now
            self._db.flush([validation])
            return ShadowComparisonResult(item.id, True, False, True)
        if item.status not in {'approved', 'rejected', 'needs_more_evidence'}:
            return ShadowComparisonResult(item.id, False, False, False)
        supported = item.status == 'approved'
        validation.shadow_comparison_status = 'completed'
        validation.shadow_human_resolution = item.status
        validation.shadow_exclusion_code = None
        validation.shadow_compared_at = now
        locked_rollout.shadow_completed_count += 1
        if supported:
            locked_rollout.shadow_supported_count += 1
        locked_rollout.state_version += 1
        locked_rollout.updated_at = now
        self._db.flush([validation, locked_rollout])
        return ShadowComparisonResult(item.id, True, supported, False)
