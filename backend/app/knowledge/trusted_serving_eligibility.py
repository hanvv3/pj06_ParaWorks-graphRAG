from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.ingestion.source_authority import (
    exact_authority_contains_chunk,
    resolve_exact_source_authority,
)
from backend.app.knowledge.trusted_provenance import has_legacy_human_base
from backend.app.models import (
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewValidation,
    DecisionRecord,
    DocumentChunk,
    HistoryEvent,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    VectorServingTombstone,
)

_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}
_CRITICAL_AUDIT_OUTCOMES = {
    'incorrect',
    'permission_violation',
    'source_version_violation',
    'policy_violation',
}


@dataclass(frozen=True, slots=True)
class TrustedServingEligibility:
    eligible: bool
    effective_permission: str | None


class TrustedServingEligibilityService:
    """The fail-closed authority for trusted knowledge serving."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def for_document(self, document_id: str) -> TrustedServingEligibility:
        if ':' not in document_id:
            return _ineligible()
        knowledge_type, raw_id = document_id.split(':', maxsplit=1)
        if knowledge_type == 'chunk':
            try:
                return self._for_chunk(int(raw_id))
            except (SQLAlchemyError, TypeError, ValueError):
                return _ineligible()
        try:
            knowledge_id = int(raw_id)
        except ValueError:
            return _ineligible()
        return self.for_knowledge(knowledge_type, knowledge_id)

    def _for_chunk(self, chunk_id: int) -> TrustedServingEligibility:
        chunk = self._db.get(DocumentChunk, chunk_id)
        if chunk is None:
            return _ineligible()
        source = self._db.get(Source, chunk.source_id)
        if (
            source is None
            or source.source_type == 'slack'
            or source.permission_level not in _PERMISSION_RANK
            or chunk.permission_level not in _PERMISSION_RANK
        ):
            return _ineligible()
        approved_payloads = tuple(
            self._db.scalars(
                select(ReviewItem.payload).where(
                    ReviewItem.status == 'approved',
                    or_(
                        ReviewItem.resolution_source.is_(None),
                        ReviewItem.resolution_source == 'human',
                    ),
                )
            ).all()
        )
        if not any(
            source.source_id
            in {
                value
                for value in (payload or {}).get('source_ids', ())
                if isinstance(value, str)
            }
            for payload in approved_payloads
        ):
            return _ineligible()
        effective = _strictest_permission(
            [source.permission_level, chunk.permission_level]
        )
        if chunk.parser_run_id is None:
            return _ineligible()
        authority = resolve_exact_source_authority(self._db, source=source)
        if (
            authority is None
            or not exact_authority_contains_chunk(authority, chunk)
        ):
            return _ineligible()
        return TrustedServingEligibility(True, effective)

    def for_knowledge(
        self, knowledge_type: str, knowledge_id: int
    ) -> TrustedServingEligibility:
        try:
            return self._for_knowledge(knowledge_type, knowledge_id)
        except (SQLAlchemyError, TypeError, ValueError):
            return _ineligible()

    def approval_link_is_live(self, approval_link_id: int) -> bool:
        """Return whether one exact explicit approval effect is still authoritative."""
        try:
            link = self._db.get(
                TrustedKnowledgeApprovalLink, approval_link_id
            )
            if (
                link is None
                or not link.active
                or link.permission_level not in _PERMISSION_RANK
            ):
                return False
            target = self._db.get(
                knowledge_model_for_type(link.knowledge_type),
                link.knowledge_id,
            )
            item = self._db.get(ReviewItem, link.review_item_id)
            document_id = canonical_knowledge_document_id(
                link.knowledge_type,
                link.knowledge_id,
            )
            if (
                target is None
                or target.review_status != 'approved'
                or target.permission_level not in _PERMISSION_RANK
                or item is None
                or item.status != 'approved'
                or item.resolution_source != link.resolution_source
                or item.permission_level not in _PERMISSION_RANK
                or self._db.scalar(
                    select(VectorServingTombstone.id).where(
                        VectorServingTombstone.document_id == document_id
                    )
                )
                is not None
            ):
                return False
            evidence = self._current_evidence(link)
            if not evidence.valid:
                return False
            if link.resolution_source == 'human':
                return item.candidate_contract_version == 'c5-v1'
            if link.resolution_source != 'auto_policy' or self._is_quarantined(
                item.id
            ):
                return False
            validation = (
                self._db.get(AutoReviewValidation, item.auto_validation_id)
                if item.auto_validation_id is not None
                else None
            )
            return bool(
                validation is not None
                and validation.review_item_id == item.id
                and validation.status == 'completed'
                and validation.policy_decision
                in {'auto_approve', 'reuse_trusted'}
                and all(
                    permission in {'public', 'internal'}
                    for permission in evidence.permission_levels
                )
            )
        except (SQLAlchemyError, TypeError, ValueError):
            return False

    def _for_knowledge(
        self, knowledge_type: str, knowledge_id: int
    ) -> TrustedServingEligibility:
        canonical_type = canonical_knowledge_type(knowledge_type)
        model = knowledge_model_for_type(canonical_type)
        target = self._db.get(model, knowledge_id)
        document_id = canonical_knowledge_document_id(
            canonical_type,
            knowledge_id,
        )
        if (
            target is None
            or target.review_status != 'approved'
            or target.permission_level not in _PERMISSION_RANK
            or self._db.scalar(
                select(VectorServingTombstone.id).where(
                    VectorServingTombstone.document_id == document_id
                )
            )
            is not None
        ):
            return _ineligible()

        links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink)
                .where(
                    TrustedKnowledgeApprovalLink.knowledge_type.in_(
                        knowledge_type_storage_aliases(canonical_type)
                    ),
                    TrustedKnowledgeApprovalLink.knowledge_id == knowledge_id,
                    TrustedKnowledgeApprovalLink.active.is_(True),
                )
                .order_by(TrustedKnowledgeApprovalLink.id)
            ).all()
        )
        if not links:
            return self._legacy_human_eligibility(target)

        permission_levels = [target.permission_level]
        valid_human = has_legacy_human_base(
            self._db,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
        )
        nonquarantined_auto_results: list[bool] = []
        for link in links:
            item = self._db.get(ReviewItem, link.review_item_id)
            if (
                item is None
                or item.status != 'approved'
                or item.resolution_source != link.resolution_source
                or link.permission_level not in _PERMISSION_RANK
                or item.permission_level not in _PERMISSION_RANK
            ):
                if link.resolution_source == 'auto_policy':
                    nonquarantined_auto_results.append(False)
                continue
            permission_levels.extend([link.permission_level, item.permission_level])
            evidence_result = self._current_evidence(link)
            permission_levels.extend(evidence_result.permission_levels)
            if link.resolution_source == 'human':
                if item.candidate_contract_version == 'c5-v1' and evidence_result.valid:
                    valid_human = True
                continue
            if link.resolution_source != 'auto_policy':
                nonquarantined_auto_results.append(False)
                continue
            if self._is_quarantined(item.id):
                continue
            validation = (
                self._db.get(AutoReviewValidation, item.auto_validation_id)
                if item.auto_validation_id is not None
                else None
            )
            validation_complete = bool(
                validation is not None
                and validation.review_item_id == item.id
                and validation.status == 'completed'
                and validation.policy_decision in {'auto_approve', 'reuse_trusted'}
            )
            supported_permissions = all(
                permission in {'public', 'internal'}
                for permission in evidence_result.permission_levels
            )
            nonquarantined_auto_results.append(
                validation_complete
                and evidence_result.valid
                and supported_permissions
            )

        if valid_human:
            return TrustedServingEligibility(
                eligible=True,
                effective_permission=_strictest_permission(permission_levels),
            )
        if nonquarantined_auto_results and all(nonquarantined_auto_results):
            return TrustedServingEligibility(
                eligible=True,
                effective_permission=_strictest_permission(permission_levels),
            )
        return _ineligible()

    def _legacy_human_eligibility(self, target: object) -> TrustedServingEligibility:
        source_review_item_id = getattr(target, 'source_review_item_id', None)
        if source_review_item_id is None:
            # V2.0 trusted rows predate explicit provenance. They remain
            # readable only at their already-stored permission and can never
            # participate in C.5 auto reuse or permission broadening.
            return TrustedServingEligibility(
                eligible=True,
                effective_permission=target.permission_level,
            )
        item = self._db.get(ReviewItem, source_review_item_id)
        if (
            item is None
            or item.status != 'approved'
            or item.resolution_source not in {None, 'human'}
            or item.candidate_contract_version == 'c5-v1'
        ):
            return _ineligible()
        return TrustedServingEligibility(
            eligible=True,
            effective_permission=target.permission_level,
        )

    def _current_evidence(
        self, link: TrustedKnowledgeApprovalLink
    ) -> _EvidenceEligibility:
        children = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink)
                .where(
                    TrustedKnowledgeEvidenceLink.approval_link_id == link.id
                )
                .order_by(TrustedKnowledgeEvidenceLink.id)
            ).all()
        )
        if not children:
            return _EvidenceEligibility(False, ())
        permissions: list[str] = []
        for child in children:
            try:
                source_pk = int(child.canonical_source_id)
            except ValueError:
                return _EvidenceEligibility(False, tuple(permissions))
            source = self._db.get(Source, source_pk)
            if source is None or source.permission_level not in _PERMISSION_RANK:
                return _EvidenceEligibility(False, tuple(permissions))
            permissions.append(source.permission_level)
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
                return _EvidenceEligibility(False, tuple(permissions))
        return _EvidenceEligibility(True, tuple(permissions))

    def _is_quarantined(self, review_item_id: int) -> bool:
        audit = self._db.scalar(
            select(AutoReviewPostAudit).where(
                AutoReviewPostAudit.review_item_id == review_item_id
            )
        )
        if audit is not None and (
            audit.status == 'remediation_required'
            or audit.outcome in _CRITICAL_AUDIT_OUTCOMES
        ):
            return True
        correction = self._db.scalar(
            select(AutoReviewAuditCorrection).where(
                AutoReviewAuditCorrection.review_item_id == review_item_id
            )
        )
        return bool(
            correction is not None
            and correction.effective_outcome in _CRITICAL_AUDIT_OUTCOMES
        )


@dataclass(frozen=True, slots=True)
class _EvidenceEligibility:
    valid: bool
    permission_levels: tuple[str, ...]


def knowledge_model_for_type(knowledge_type: str) -> type:
    canonical_type = canonical_knowledge_type(knowledge_type)
    try:
        return {
            'decision_record': DecisionRecord,
            'history_event': HistoryEvent,
            'timeline_event': TimelineEvent,
            'todo': Todo,
        }[canonical_type]
    except KeyError:
        raise ValueError('trusted knowledge type is unsupported') from None


def canonical_knowledge_type(knowledge_type: str) -> str:
    return 'decision_record' if knowledge_type == 'decision' else knowledge_type


def knowledge_type_storage_aliases(knowledge_type: str) -> tuple[str, ...]:
    canonical_type = canonical_knowledge_type(knowledge_type)
    if canonical_type == 'decision_record':
        return ('decision', 'decision_record')
    return (canonical_type,)


def canonical_knowledge_document_id(
    knowledge_type: str,
    knowledge_id: int,
) -> str:
    return f'{canonical_knowledge_type(knowledge_type)}:{knowledge_id}'


def canonical_evidence_version_is_current(
    db: Session,
    *,
    source: Source,
    version_or_signature: str,
) -> bool:
    authority = resolve_exact_source_authority(db, source=source)
    if authority is None:
        return False
    if version_or_signature == source.server_content_signature:
        return True
    revision_id = authority.parser_run.revision_id
    return bool(revision_id and version_or_signature == revision_id)


def _strictest_permission(permission_levels: list[str]) -> str:
    return max(permission_levels, key=_PERMISSION_RANK.__getitem__)


def _ineligible() -> TrustedServingEligibility:
    return TrustedServingEligibility(eligible=False, effective_permission=None)
