from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.agent_runtime.auto_review_orchestrator import (
    AutoReviewValidationOrchestrator,
    PreparedValidationBatch,
    ValidationOrchestrationResult,
)
from backend.app.core.config import Settings
from backend.app.knowledge.trusted_provenance import TrustedProvenanceMismatch
from backend.app.review.actors import ReuseExistingPromotion, auto_review_actor
from backend.app.review.transitions import (
    AutoReviewAdmissionNotReady,
    CanonicalEvidenceDrift,
    CurrentPermissionResolver,
    ReviewTransitionResult,
    ReviewTransitionService,
)
from backend.app.schemas.auto_review import AUTO_REVIEW_POLICY_VERSION


class AutoReviewHumanOnly(ValueError):  # noqa: N818 - policy outcome contract
    """Exact safe reuse was not proven; leave the item for human review."""


class AutoReviewResolutionService:
    def __init__(
        self,
        *,
        settings: Settings,
        current_permission_resolver: CurrentPermissionResolver,
        validation_orchestrator: AutoReviewValidationOrchestrator | None = None,
    ) -> None:
        self._transitions = ReviewTransitionService(
            settings=settings,
            current_permission_resolver=current_permission_resolver,
        )
        self._validation_orchestrator = validation_orchestrator

    def validate_prepared(
        self,
        batch: PreparedValidationBatch,
    ) -> ValidationOrchestrationResult:
        """Persist a bounded observation; Task 10 remains approval authority."""
        if self._validation_orchestrator is None:
            raise AutoReviewHumanOnly('automatic validation is unavailable')
        return self._validation_orchestrator.execute(batch)

    def resolve(
        self,
        *,
        db: Session,
        item_id: int,
        directive: ReuseExistingPromotion,
    ) -> ReviewTransitionResult:
        try:
            return self._transitions.transition(
                db=db,
                item_id=item_id,
                action='approve',
                actor=auto_review_actor(policy_version=AUTO_REVIEW_POLICY_VERSION),
                approval_directive=directive,
            )
        except (
            AutoReviewAdmissionNotReady,
            CanonicalEvidenceDrift,
            TrustedProvenanceMismatch,
        ) as exc:
            db.expire_all()
            raise AutoReviewHumanOnly('exact trusted bundle reuse was not proven') from exc
