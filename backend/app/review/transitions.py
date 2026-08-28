from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.knowledge.promotion import (
    IncompleteReviewPromotionError,
    find_review_item_promotion,
    promote_review_item,
    validate_review_item_for_approval,
)
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AutoReviewValidation,
    Document,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
)
from backend.app.review.actors import (
    SYSTEM_AUTO_REVIEW_ACTOR_ID,
    ApprovalDirective,
    CreateNewPromotion,
    ReuseExistingPromotion,
    ReviewResolutionActor,
)
from backend.app.schemas.auto_review import AUTO_REVIEW_POLICY_VERSION

ReviewAction = Literal['approve', 'reject', 'needs_more_evidence']


@dataclass(frozen=True)
class PromotionResult:
    target_type: str | None
    created_record_ids: tuple[int, ...]
    created_timeline_event_ids: tuple[int, ...]


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


class InvalidReviewTransition(ValueError):  # noqa: N818 - name is a frozen public contract
    code = 'invalid_state_transition'

    def __init__(self, *, action: ReviewAction, status: str) -> None:
        super().__init__(f'Cannot {action} review item from {status}')


class ReviewTransitionService:
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
        preview = db.get(ReviewItem, item_id)
        if preview is None:
            raise ValueError('Review item not found')
        directive = self._authorize_action(
            item=preview,
            action=action,
            actor=actor,
            approval_directive=approval_directive,
        )
        items = self._load_items(db, [item_id])
        if not items:
            raise ValueError('Review item not found')
        return self._transition_item(
            db=db,
            item=items[0],
            action=action,
            actor=actor,
            note=note,
            approval_directive=directive,
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
        if actor.actor_type != 'human' or 'human_review' not in actor.capabilities:
            _raise_review_permission_denied()
        requested_ids = sorted(set(item_ids))
        items = self._load_items(db, requested_ids)
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
                    result = self._transition_item(
                        db=db,
                        item=item,
                        action=action,
                        actor=actor,
                        note=note,
                        approval_directive=self._authorize_action(
                            item=item,
                            action=action,
                            actor=actor,
                            approval_directive=None,
                        ),
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
    def _load_items(db: Session, item_ids: Sequence[int]) -> list[ReviewItem]:
        if not item_ids:
            return []
        statement = (
            select(ReviewItem)
            .where(ReviewItem.id.in_(item_ids))
            .order_by(ReviewItem.id)
            .execution_options(populate_existing=True)
        )
        if db.get_bind().dialect.name == 'postgresql':
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
    ) -> ReviewTransitionResult:
        directive = self._authorize_action(
            item=item,
            action=action,
            actor=actor,
            approval_directive=approval_directive,
        )

        if item.status == 'approved' and action == 'approve':
            return ReviewTransitionResult(
                item_id=item.id,
                status=item.status,
                replayed=True,
                promotion=_to_promotion_result(find_review_item_promotion(db, item)),
            )
        if item.status != 'pending_review':
            raise InvalidReviewTransition(action=action, status=item.status)

        reviewed_at = datetime.now(UTC)
        if action == 'approve':
            if directive is None:
                raise RuntimeError('Approval directive is required')
            _validate_source_evidence(item)
            validate_review_item_for_approval(item)
            validation_id = self._validation_id_for_actor(db, item=item, actor=actor)
            if isinstance(directive, ReuseExistingPromotion):
                raise ValueError(
                    'Existing promotion reuse requires the trusted provenance service'
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
            promotion = self._promote_with_directive(db, item, directive)
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
    def _validation_id_for_actor(
        db: Session,
        *,
        item: ReviewItem,
        actor: ReviewResolutionActor,
    ) -> int | None:
        if actor.actor_type == 'human':
            return None
        statement = select(AutoReviewValidation).where(
            AutoReviewValidation.review_item_id == item.id,
            AutoReviewValidation.workflow_thread_id == item.workflow_thread_id,
            AutoReviewValidation.status == 'completed',
            AutoReviewValidation.policy_version == actor.policy_version,
        )
        if db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update()
        validations = tuple(db.scalars(statement).all())
        if len(validations) != 1:
            raise ValueError('Auto approval requires one matching completed validation')
        return validations[0].id

    def _promote_with_directive(
        self,
        db: Session,
        item: ReviewItem,
        directive: ApprovalDirective,
    ) -> PromotionResult:
        if isinstance(directive, ReuseExistingPromotion):
            raise ValueError(
                'Existing promotion reuse requires the trusted provenance service'
            )
        return self._promote_exactly_once(db, item)

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
    def canonical_evidence_has_drifted(
        self,
        db: Session,
        *,
        item: ReviewItem,
    ) -> bool: ...


class CanonicalReviewEvidenceStalenessResolver:
    def canonical_evidence_has_drifted(
        self,
        db: Session,
        *,
        item: ReviewItem,
    ) -> bool:
        if item.workflow_thread_id is None:
            return False
        children = tuple(
            db.scalars(
                select(ReviewItemEvidenceRef).where(
                    ReviewItemEvidenceRef.review_item_id == item.id,
                    ReviewItemEvidenceRef.workflow_thread_id
                    == item.workflow_thread_id,
                )
            ).all()
        )
        if not children:
            return False
        refs = tuple(
            db.scalars(
                select(AgentWorkflowEvidenceRef).where(
                    AgentWorkflowEvidenceRef.workflow_thread_id
                    == item.workflow_thread_id,
                    AgentWorkflowEvidenceRef.id.in_(
                        [child.workflow_evidence_ref_id for child in children]
                    ),
                )
            ).all()
        )
        if len(refs) != len(children):
            return True
        source_statement = select(Source).where(
            Source.id.in_([ref.canonical_row_id for ref in refs])
        )
        if db.get_bind().dialect.name == 'postgresql':
            source_statement = source_statement.with_for_update()
        sources = {
            source.id: source for source in db.scalars(source_statement).all()
        }
        documents = {
            document.source_id: document
            for document in db.scalars(
                select(Document).where(
                    Document.source_id.in_(
                        [ref.canonical_row_id for ref in refs]
                    )
                )
            ).all()
        }
        for ref in refs:
            source = sources.get(ref.canonical_row_id)
            if source is None:
                return True
            if (
                ref.canonical_table != 'sources'
                or ref.canonical_source_type != source.source_type
                or source.server_content_signature_schema
                != 'server-source-content:v1'
                or source.server_content_signature != ref.content_signature
            ):
                return True
            document = documents.get(source.id)
            current_version_id = (
                document.current_document_version_id if document is not None else None
            )
            if current_version_id != ref.document_version_id:
                return True
        return False


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
        items = ReviewTransitionService._load_items(db, [item_id])
        if not items:
            raise ValueError('Review item not found')
        item = items[0]
        if item.status != 'pending_review':
            raise InvalidReviewTransition(
                action='needs_more_evidence',
                status=item.status,
            )
        if not self._evidence_staleness_resolver.canonical_evidence_has_drifted(
            db,
            item=item,
        ):
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
    )
