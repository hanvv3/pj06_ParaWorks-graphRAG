from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser
from backend.app.knowledge.trusted_serving_eligibility import (
    canonical_evidence_version_is_current,
)
from backend.app.models import (
    AgentWorkflowEvidenceRef,
    AutoReviewPostAudit,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)

_KNOWN_PERMISSIONS = {'public', 'internal', 'restricted'}


class ReviewEvidenceNotFound(LookupError):  # noqa: N818 - concealment sentinel
    def __init__(self) -> None:
        super().__init__('Review item not found')


@dataclass(frozen=True, slots=True)
class ReviewEvidenceProjection:
    review_item_id: int
    visible: bool
    evidence_available: bool
    evidence_status: str
    effective_permission: str
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    action_required: bool


class ReviewEvidenceVisibilityService:
    """Actor-aware Review Queue projection, separate from trusted serving."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def project(
        self, review_item_id: int, actor: DemoUser
    ) -> ReviewEvidenceProjection:
        try:
            return self._project(review_item_id, actor)
        except ReviewEvidenceNotFound:
            raise
        except (SQLAlchemyError, TypeError, ValueError):
            raise ReviewEvidenceNotFound() from None

    def _project(
        self, review_item_id: int, actor: DemoUser
    ) -> ReviewEvidenceProjection:
        item = self._db.get(ReviewItem, review_item_id)
        if item is None or item.permission_level not in actor.permission_levels:
            raise ReviewEvidenceNotFound()
        sources, complete, permission_snapshots = (
            self._resolve_current_sources(item)
        )
        evidence_shaped = bool(
            (item.payload or {}).get('source_ids')
            or item.source_links
            or item.source_snippets
        )
        if (
            not sources
            and not evidence_shaped
            and item.candidate_contract_version != 'c5-v1'
        ):
            return ReviewEvidenceProjection(
                review_item_id=item.id,
                visible=True,
                evidence_available=True,
                evidence_status='legacy_available',
                effective_permission=item.permission_level,
                source_ids=tuple(
                    value
                    for value in (item.payload or {}).get('source_ids', ())
                    if isinstance(value, str)
                ),
                source_links=tuple(item.source_links or ()),
                source_snippets=tuple(item.source_snippets or ()),
                action_required=item.status == 'pending_review',
            )
        for source in sources:
            if (
                source.permission_level in _KNOWN_PERMISSIONS
                and source.permission_level not in actor.permission_levels
            ):
                raise ReviewEvidenceNotFound()
        supported = bool(sources) and all(
            source.permission_level in _KNOWN_PERMISSIONS for source in sources
        )
        evidence_available = complete and supported
        audit = self._db.scalar(
            select(AutoReviewPostAudit).where(
                AutoReviewPostAudit.review_item_id == item.id
            )
        )
        action_required = bool(
            audit is not None
            and audit.status in {'pending', 'remediation_required'}
        )
        effective_permission = _strictest_permission(
            [
                item.permission_level,
                *(
                    source.permission_level
                    for source in sources
                    if source.permission_level in _KNOWN_PERMISSIONS
                ),
                *(
                    permission
                    for permission in permission_snapshots
                    if permission in _KNOWN_PERMISSIONS
                ),
            ]
        )
        if not evidence_available:
            return ReviewEvidenceProjection(
                review_item_id=item.id,
                visible=True,
                evidence_available=False,
                evidence_status='evidence_unavailable',
                effective_permission=effective_permission,
                source_ids=(),
                source_links=(),
                source_snippets=(),
                action_required=action_required,
            )
        return ReviewEvidenceProjection(
            review_item_id=item.id,
            visible=True,
            evidence_available=True,
            evidence_status='available',
            effective_permission=effective_permission,
            source_ids=tuple(source.source_id for source in sources),
            source_links=tuple(item.source_links or ()),
            source_snippets=tuple(item.source_snippets or ()),
            action_required=action_required,
        )

    def _resolve_current_sources(
        self, item: ReviewItem
    ) -> tuple[tuple[Source, ...], bool, tuple[str, ...]]:
        links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.review_item_id == item.id
                )
            ).all()
        )
        children = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink)
                .where(
                    TrustedKnowledgeEvidenceLink.approval_link_id.in_(
                        [link.id for link in links]
                    )
                )
                .order_by(TrustedKnowledgeEvidenceLink.id)
            ).all()
        ) if links else ()
        if not children:
            refs = tuple(
                self._db.execute(
                    select(AgentWorkflowEvidenceRef)
                    .join(
                        ReviewItemEvidenceRef,
                        ReviewItemEvidenceRef.workflow_evidence_ref_id
                        == AgentWorkflowEvidenceRef.id,
                    )
                    .where(
                        ReviewItemEvidenceRef.review_item_id == item.id,
                        ReviewItemEvidenceRef.workflow_thread_id
                        == AgentWorkflowEvidenceRef.workflow_thread_id,
                    )
                    .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
                ).scalars().all()
            )
            if not refs:
                external_ids = tuple(
                    value
                    for value in (item.payload or {}).get('source_ids', ())
                    if isinstance(value, str)
                )
                if not external_ids:
                    return (), False, ()
                sources = tuple(
                    self._db.scalars(
                        select(Source)
                        .where(Source.source_id.in_(external_ids))
                        .order_by(Source.id)
                    ).all()
                )
                return sources, len(sources) == len(set(external_ids)), ()
            sources_by_id: dict[int, Source] = {}
            complete = True
            for ref in refs:
                source = self._db.get(Source, ref.canonical_row_id)
                if source is None:
                    complete = False
                    continue
                sources_by_id[source.id] = source
                if (
                    ref.canonical_table != 'sources'
                    or source.source_type != ref.canonical_source_type
                    or source.server_content_signature_schema
                    != 'server-source-content:v1'
                    or source.server_content_signature
                    != ref.content_signature
                ):
                    complete = False
            return (
                tuple(
                    sources_by_id[key] for key in sorted(sources_by_id)
                ),
                complete,
                tuple(ref.permission_level_snapshot for ref in refs),
            )
        sources_by_id: dict[int, Source] = {}
        complete = True
        for child in children:
            try:
                source_id = int(child.canonical_source_id)
            except ValueError:
                complete = False
                continue
            source = self._db.get(Source, source_id)
            if source is None:
                complete = False
                continue
            sources_by_id[source.id] = source
            if (
                source.source_type != child.canonical_source_kind
                or not canonical_evidence_version_is_current(
                    self._db,
                    source=source,
                    version_or_signature=(
                        child.canonical_version_or_signature
                    ),
                )
            ):
                complete = False
        return (
            tuple(sources_by_id[key] for key in sorted(sources_by_id)),
            complete,
            tuple(link.permission_level for link in links),
        )


def _strictest_permission(levels: list[str]) -> str:
    rank = {'public': 0, 'internal': 1, 'restricted': 2}
    known = [level for level in levels if level in rank]
    return max(known, key=rank.__getitem__) if known else 'restricted'
