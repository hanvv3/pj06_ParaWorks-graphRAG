from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.knowledge.trusted_fingerprint_projection import (
    ProjectionSummary,
    ProjectionSummaryDelta,
    projection_row_digest,
    snapshot_from_projection,
)
from backend.app.knowledge.trusted_serving_eligibility import (
    canonical_evidence_version_is_current,
    knowledge_model_for_type,
)
from backend.app.models import (
    Document,
    DocumentParserRun,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    TrustedKnowledgeFingerprint,
    TrustedKnowledgeFingerprintProjectionState,
    VectorIndexState,
)
from backend.app.rag.indexing import (
    VectorIndexWriter,
    build_rag_index_documents,
    compute_vector_document_hash,
)
from backend.app.review.auto_review_revoke import (
    _SOURCE_INVALIDATION_CONTEXTS_INFO_KEY,
    AutoReviewRevokeService,
    SourceInvalidationRevokeContext,
)

_KNOWN_SERVING_PERMISSIONS = {'public', 'internal'}
_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}


@dataclass(frozen=True, slots=True)
class SourceReconciliationResult:
    stale_count: int = 0
    reconciled_count: int = 0
    revoked_count: int = 0
    narrowed_count: int = 0
    repaired_count: int = 0
    ambiguous_count: int = 0
    remaining_count: int = 0
    failure_count: int = 0
    readiness: bool = True


class AutoReviewSourceReconciliationService:
    """Bounded, idempotent cleanup for canonical-source drift.

    The live serving predicate remains authoritative; this service only makes
    durable knowledge/vector state converge with that predicate.
    """

    def __init__(
        self,
        db: Session,
        *,
        settings: Settings,
        vector_writer: VectorIndexWriter | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._vector_writer = vector_writer

    def status(self, *, limit: int = 100) -> SourceReconciliationResult:
        bounded = _validate_limit(limit)
        try:
            stale = self._stale_source_ids(limit=bounded)
            return SourceReconciliationResult(
                stale_count=len(stale),
                remaining_count=len(stale),
                readiness=not stale,
            )
        except SQLAlchemyError:
            self._db.rollback()
            return SourceReconciliationResult(
                failure_count=1,
                remaining_count=1,
                readiness=False,
            )

    def recover_stale_sources(
        self, *, limit: int = 100
    ) -> SourceReconciliationResult:
        bounded = _validate_limit(limit)
        stale = self._stale_source_ids(limit=bounded)
        if not stale:
            return SourceReconciliationResult()
        result = self.reconcile_source_ids(stale)
        remaining = len(self._stale_source_ids(limit=bounded))
        return SourceReconciliationResult(
            stale_count=len(stale),
            reconciled_count=result.reconciled_count,
            revoked_count=result.revoked_count,
            narrowed_count=result.narrowed_count,
            repaired_count=result.repaired_count,
            ambiguous_count=result.ambiguous_count,
            remaining_count=remaining,
            failure_count=result.failure_count,
            readiness=remaining == 0 and result.failure_count == 0,
        )

    def reconcile_source_ids(
        self, source_ids: Sequence[int]
    ) -> SourceReconciliationResult:
        reconciled = revoked = narrowed = failures = 0
        for source_id in sorted(set(source_ids))[:100]:
            try:
                source_result = self._reconcile_source(source_id)
                reconciled += int(source_result[0])
                revoked += source_result[1]
                narrowed += source_result[2]
            except (SQLAlchemyError, ValueError):
                self._db.rollback()
                failures += 1
        return SourceReconciliationResult(
            reconciled_count=reconciled,
            revoked_count=revoked,
            narrowed_count=narrowed,
            failure_count=failures,
            remaining_count=failures,
            readiness=failures == 0,
        )

    def repair_current_document_versions(
        self, *, limit: int = 100
    ) -> SourceReconciliationResult:
        bounded = _validate_limit(limit)
        documents = tuple(
            self._db.scalars(
                select(Document)
                .where(Document.current_document_version_id.is_(None))
                .order_by(Document.id)
                .limit(bounded)
            ).all()
        )
        repaired = ambiguous = unresolved = 0
        for document in documents:
            source = self._db.get(Source, document.source_id)
            if (
                source is None
                or source.server_content_signature_schema
                != 'server-source-content:v1'
                or source.server_content_signature is None
            ):
                unresolved += 1
                continue
            runs = tuple(
                self._db.scalars(
                    select(DocumentParserRun).where(
                        DocumentParserRun.document_id == document.id,
                        DocumentParserRun.source_id == source.id,
                        DocumentParserRun.server_content_signature_schema
                        == 'server-source-content:v1',
                        DocumentParserRun.server_content_signature
                        == source.server_content_signature,
                        DocumentParserRun.parser_policy_version.is_not(None),
                        DocumentParserRun.parser_version.is_not(None),
                        DocumentParserRun.chunk_policy_version.is_not(None),
                    )
                ).all()
            )
            version_ids = {run.document_version_id for run in runs}
            if len(version_ids) == 1 and len(runs) == 1:
                document.current_document_version_id = version_ids.pop()
                repaired += 1
            elif len(version_ids) > 1 or len(runs) > 1:
                ambiguous += 1
            else:
                unresolved += 1
        if repaired:
            self._db.commit()
        remaining = ambiguous + unresolved
        return SourceReconciliationResult(
            repaired_count=repaired,
            ambiguous_count=ambiguous,
            remaining_count=remaining,
            readiness=remaining == 0,
        )

    def _stale_source_ids(self, *, limit: int) -> list[int]:
        links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.active.is_(True),
                    TrustedKnowledgeApprovalLink.resolution_source
                    == 'auto_policy',
                )
            ).all()
        )
        stale: set[int] = set()
        for link in links:
            for evidence in self._evidence(link.id):
                source_id = _source_id(evidence.canonical_source_id)
                if source_id is None:
                    continue
                source = self._db.get(Source, source_id)
                if (
                    not _evidence_is_current(self._db, source, evidence)
                    or self._permission_reconciliation_needed(link, source)
                ):
                    stale.add(source_id)
        return sorted(stale)[:limit]

    def _permission_reconciliation_needed(
        self,
        link: TrustedKnowledgeApprovalLink,
        source: Source | None,
    ) -> bool:
        if source is None:
            return False
        target = self._db.get(
            knowledge_model_for_type(link.knowledge_type),
            link.knowledge_id,
        )
        item = self._db.get(ReviewItem, link.review_item_id)
        if target is None or item is None:
            return False
        desired = _strictest_many(
            target.permission_level,
            link.permission_level,
            item.permission_level,
            source.permission_level,
        )
        current_levels = [
            target.permission_level,
            link.permission_level,
            item.permission_level,
        ]
        fingerprint = self._db.scalar(
            select(TrustedKnowledgeFingerprint).where(
                TrustedKnowledgeFingerprint.knowledge_type
                == link.knowledge_type,
                TrustedKnowledgeFingerprint.knowledge_id
                == link.knowledge_id,
            )
        )
        if fingerprint is not None:
            current_levels.append(fingerprint.permission_level)
        return any(_strictest(level, desired) != level for level in current_levels)

    def _reconcile_source(self, source_id: int) -> tuple[bool, int, int]:
        source = self._db.get(Source, source_id)
        evidence_rows = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.canonical_source_id
                    == str(source_id)
                )
            ).all()
        )
        if not evidence_rows:
            return False, 0, 0
        revoked = narrowed = 0
        item_ids: set[int] = set()
        narrowed_documents: set[str] = set()
        for evidence in evidence_rows:
            link = self._db.get(
                TrustedKnowledgeApprovalLink, evidence.approval_link_id
            )
            if (
                link is None
                or not link.active
                or link.resolution_source != 'auto_policy'
            ):
                continue
            item_ids.add(link.review_item_id)
            if _evidence_is_current(self._db, source, evidence):
                target = self._db.get(
                    knowledge_model_for_type(link.knowledge_type),
                    link.knowledge_id,
                )
                if target is None:
                    continue
                current = target.permission_level
                item = self._db.get(ReviewItem, link.review_item_id)
                strictest = _strictest_many(
                    current,
                    link.permission_level,
                    item.permission_level if item is not None else 'restricted',
                    source.permission_level,
                )
                target_changed = strictest != current
                link_changed = (
                    _strictest(link.permission_level, strictest)
                    != link.permission_level
                )
                item_changed = bool(
                    item is not None
                    and _strictest(item.permission_level, strictest)
                    != item.permission_level
                )
                if target_changed:
                    target.permission_level = strictest
                link.permission_level = _strictest(
                    link.permission_level, strictest
                )
                if item is not None:
                    item.permission_level = _strictest(
                        item.permission_level, strictest
                    )
                document_id = f'{link.knowledge_type}:{link.knowledge_id}'
                changed = self._narrow_fingerprint(
                    document_id=document_id,
                    permission_level=strictest,
                )
                if target_changed or link_changed or item_changed or changed:
                    if self._vector_writer is not None:
                        self._vector_writer.narrow_permissions(
                            [document_id], strictest
                        )
                    self._refresh_index_state_hash(document_id)
                    if document_id not in narrowed_documents:
                        narrowed += 1
                        narrowed_documents.add(document_id)
                continue
            context = self._mint_source_invalidation_context(
                review_item_id=link.review_item_id,
                canonical_source_id=str(source_id),
            )
            result = AutoReviewRevokeService(
                self._db,
                settings=self._settings,
                vector_writer=self._vector_writer,
            ).revoke_source_invalidated(
                context=context,
            )
            revoked += int(not result.replayed)
        if narrowed:
            self._db.commit()
        return bool(item_ids), revoked, narrowed

    def _narrow_fingerprint(
        self, *, document_id: str, permission_level: str
    ) -> bool:
        knowledge_type, raw_id = document_id.rsplit(':', maxsplit=1)
        row = self._db.scalar(
            select(TrustedKnowledgeFingerprint).where(
                TrustedKnowledgeFingerprint.knowledge_type == knowledge_type,
                TrustedKnowledgeFingerprint.knowledge_id == int(raw_id),
            )
        )
        if row is None:
            return False
        narrowed = _strictest(row.permission_level, permission_level)
        if narrowed == row.permission_level:
            return False
        state = self._db.scalar(
            select(TrustedKnowledgeFingerprintProjectionState).where(
                TrustedKnowledgeFingerprintProjectionState.component
                == 'trusted_knowledge_fingerprints'
            )
        )
        old_snapshot = snapshot_from_projection(row)
        row.permission_level = narrowed
        row.updated_at = datetime.now(UTC)
        if state is None:
            return True
        try:
            old_digest = projection_row_digest(
                old_snapshot, settings=self._settings
            )
            new_digest = projection_row_digest(
                snapshot_from_projection(row), settings=self._settings
            )
            if state.ready and not state.rebuild_required:
                delta = ProjectionSummaryDelta(old_digest, new_digest)
                source = delta.apply(
                    ProjectionSummary(
                        state.source_active_count,
                        state.source_checksum or '0' * 64,
                    )
                )
                projected = delta.apply(
                    ProjectionSummary(
                        state.projected_active_count,
                        state.projected_checksum or '0' * 64,
                    )
                )
                state.source_checksum = source.checksum_hex
                state.projected_checksum = projected.checksum_hex
            else:
                state.ready = False
                state.rebuild_required = True
        except (TypeError, ValueError):
            state.ready = False
            state.rebuild_required = True
        state.updated_at = datetime.now(UTC)
        return True

    def _refresh_index_state_hash(self, document_id: str) -> None:
        document = next(
            (
                candidate
                for candidate in build_rag_index_documents(self._db)
                if candidate.document_id == document_id
            ),
            None,
        )
        if document is None:
            return
        content_hash = compute_vector_document_hash(document)
        states = tuple(
            self._db.scalars(
                select(VectorIndexState).where(
                    VectorIndexState.document_id == document_id
                )
            ).all()
        )
        for state in states:
            state.content_hash = content_hash

    def _mint_source_invalidation_context(
        self, *, review_item_id: int, canonical_source_id: str
    ) -> SourceInvalidationRevokeContext:
        context = object.__new__(SourceInvalidationRevokeContext)
        object.__setattr__(context, 'session_identity', id(self._db))
        object.__setattr__(context, 'review_item_id', review_item_id)
        object.__setattr__(
            context, 'canonical_source_id', canonical_source_id
        )
        self._db.info.setdefault(
            _SOURCE_INVALIDATION_CONTEXTS_INFO_KEY, {}
        )[id(context)] = context
        return context

    def _evidence(
        self, approval_link_id: int
    ) -> tuple[TrustedKnowledgeEvidenceLink, ...]:
        return tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id
                    == approval_link_id
                )
            ).all()
        )


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError('limit must be between 1 and 100')
    return limit


def _source_id(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _evidence_is_current(
    db: Session,
    source: Source | None,
    evidence: TrustedKnowledgeEvidenceLink,
) -> bool:
    return bool(
        source is not None
        and source.source_type == evidence.canonical_source_kind
        and canonical_evidence_version_is_current(
            db,
            source=source,
            version_or_signature=evidence.canonical_version_or_signature,
        )
        and source.permission_level in _KNOWN_SERVING_PERMISSIONS
    )


def _strictest(left: str, right: str) -> str:
    if left not in _PERMISSION_RANK or right not in _PERMISSION_RANK:
        return 'restricted'
    return max((left, right), key=_PERMISSION_RANK.__getitem__)


def _strictest_many(*levels: str) -> str:
    if any(level not in _PERMISSION_RANK for level in levels):
        return 'restricted'
    return max(levels, key=_PERMISSION_RANK.__getitem__)
