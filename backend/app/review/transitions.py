from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser
from backend.app.core.rbac import ensure_can_review_permission
from backend.app.knowledge.promotion import (
    IncompleteReviewPromotionError,
    find_review_item_promotion,
    promote_review_item,
    validate_review_item_for_approval,
)
from backend.app.models import ReviewItem

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
        actor: DemoUser,
        note: str | None = None,
    ) -> ReviewTransitionResult:
        items = self._load_items(db, [item_id])
        if not items:
            raise ValueError('Review item not found')
        return self._transition_item(
            db=db,
            item=items[0],
            action=action,
            actor=actor,
            note=note,
        )

    def transition_many(
        self,
        *,
        db: Session,
        item_ids: Sequence[int],
        action: ReviewAction,
        actor: DemoUser,
        note: str | None = None,
    ) -> ReviewBatchTransitionResult:
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
        actor: DemoUser,
        note: str | None,
    ) -> ReviewTransitionResult:
        ensure_can_review_permission(actor, item.permission_level)
        if item.permission_level not in actor.permission_levels:
            raise HTTPException(status_code=403, detail='Review approval permission required.')

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
            _validate_source_evidence(item)
            validate_review_item_for_approval(item)
            item.status = 'approved'
            item.reviewer_id = actor.id
            item.reviewed_at = reviewed_at
            db.flush([item])
            promotion = self._promote_exactly_once(db, item)
        elif action == 'reject':
            item.status = 'rejected'
            item.reviewer_id = actor.id
            item.reviewed_at = reviewed_at
            db.flush([item])
            promotion = None
        elif action == 'needs_more_evidence':
            item.status = 'needs_more_evidence'
            item.reviewer_id = actor.id
            item.reviewed_at = reviewed_at
            payload = dict(item.payload or {})
            payload['needs_more_evidence'] = {
                'requested_at': reviewed_at.isoformat(),
                'requested_by': actor.id,
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
