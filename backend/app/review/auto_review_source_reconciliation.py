from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import String, and_, case, cast, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SourceContentUnverifiableError,
    server_parser_policy_for_source,
)
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
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
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
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    build_serving_lock_plan,
)
from backend.app.review.auto_review_revoke import (
    _SOURCE_INVALIDATION_CONTEXTS_INFO_KEY,
    AutoReviewRevokeService,
    SourceInvalidationRevokeContext,
)

_KNOWN_SERVING_PERMISSIONS = {'public', 'internal'}
_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}


def build_source_reconciliation_service(
    db: Session,
    *,
    settings: Settings,
) -> AutoReviewSourceReconciliationService:
    bind = db.get_bind()
    vector_writer = None
    if bind.dialect.name == 'postgresql':
        vector_writer = PgVectorStore(
            session=db,
            config=PgVectorConfig(
                embedding_dimensions=settings.openai_embedding_dimensions
            ),
            settings=settings,
        )
    return AutoReviewSourceReconciliationService(
        db,
        settings=settings,
        vector_writer=vector_writer,
    )


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


@dataclass(frozen=True, slots=True)
class CommittedSourceStateChange:
    """Bounded post-commit handoff consumed from Task 6B ingestion."""

    source_id: int
    content_changed: bool
    permission_changed: bool
    parser_policy_changed: bool
    primary_code: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_id, bool)
            or not isinstance(self.source_id, int)
            or self.source_id <= 0
        ):
            raise ValueError('source_id must be a positive integer')
        flags = (
            self.content_changed,
            self.permission_changed,
            self.parser_policy_changed,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise ValueError('source change flags must be booleans')
        if (
            not isinstance(self.primary_code, str)
            or not self.primary_code
            or len(self.primary_code) > 64
        ):
            raise ValueError('source change primary_code is inconsistent')


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
            pointer_count = self._null_pointer_count()
            ambiguous = self._ambiguous_pointer_count(limit=bounded)
            return SourceReconciliationResult(
                stale_count=len(stale),
                ambiguous_count=ambiguous,
                remaining_count=len(stale) + pointer_count,
                readiness=not stale and pointer_count == 0,
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
        try:
            stale = self._stale_source_ids(limit=bounded)
        except SQLAlchemyError:
            self._db.rollback()
            return SourceReconciliationResult(
                failure_count=1,
                remaining_count=1,
                readiness=False,
            )
        if not stale:
            return SourceReconciliationResult()
        result = self.reconcile_source_ids(stale)
        try:
            remaining = len(self._stale_source_ids(limit=bounded))
        except SQLAlchemyError:
            self._db.rollback()
            return SourceReconciliationResult(
                stale_count=len(stale),
                reconciled_count=result.reconciled_count,
                revoked_count=result.revoked_count,
                narrowed_count=result.narrowed_count,
                repaired_count=result.repaired_count,
                ambiguous_count=result.ambiguous_count,
                remaining_count=max(1, result.remaining_count),
                failure_count=result.failure_count + 1,
                readiness=False,
            )
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
            except (SQLAlchemyError, RuntimeError, TypeError, ValueError):
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

    def reconcile(
        self,
        changed_states: Sequence[CommittedSourceStateChange],
    ) -> SourceReconciliationResult:
        if not isinstance(changed_states, Sequence):
            raise TypeError('changed_states must be a bounded sequence')
        normalized: list[CommittedSourceStateChange] = []
        for changed in changed_states[:100]:
            if not isinstance(changed, CommittedSourceStateChange):
                raise TypeError('changed_states contains an invalid DTO')
            changed.__post_init__()
            if (
                changed.content_changed
                or changed.permission_changed
                or changed.parser_policy_changed
            ):
                normalized.append(changed)
        return self.reconcile_source_ids(
            [changed.source_id for changed in normalized]
        )

    def repair_current_document_versions(
        self, *, limit: int = 100
    ) -> SourceReconciliationResult:
        bounded = _validate_limit(limit)
        try:
            document_rows = tuple(
                self._db.execute(
                    select(Document.id, Document.source_id)
                    .where(Document.current_document_version_id.is_(None))
                    .order_by(Document.id)
                    .limit(bounded)
                ).all()
            )
        except SQLAlchemyError:
            self._db.rollback()
            return SourceReconciliationResult(
                failure_count=1,
                remaining_count=1,
                readiness=False,
            )
        self._db.rollback()
        repaired = ambiguous = unresolved = failures = 0
        for document_id, source_id in document_rows:
            try:
                outcome = self._repair_current_document_version(
                    document_id=document_id,
                    source_id=source_id,
                )
            except (SQLAlchemyError, RuntimeError, TypeError, ValueError):
                self._db.rollback()
                failures += 1
                continue
            repaired += int(outcome == 'repaired')
            ambiguous += int(outcome == 'ambiguous')
            unresolved += int(outcome == 'unresolved')
        try:
            remaining = self._null_pointer_count()
        except SQLAlchemyError:
            self._db.rollback()
            remaining = max(1, ambiguous + unresolved + failures)
            failures += 1
        return SourceReconciliationResult(
            repaired_count=repaired,
            ambiguous_count=ambiguous,
            remaining_count=remaining,
            failure_count=failures,
            readiness=remaining == 0 and failures == 0,
        )

    def _null_pointer_count(self) -> int:
        return int(
            self._db.scalar(
                select(func.count(Document.id)).where(
                    Document.current_document_version_id.is_(None)
                )
            )
            or 0
        )

    def _ambiguous_pointer_count(self, *, limit: int) -> int:
        rows = tuple(
            self._db.execute(
                select(Document.id, Document.source_id)
                .where(Document.current_document_version_id.is_(None))
                .order_by(Document.id)
                .limit(limit)
            ).all()
        )
        ambiguous = 0
        for document_id, source_id in rows:
            outcome, _ = self._pointer_candidate(
                document_id=document_id,
                source_id=source_id,
                for_update=False,
            )
            ambiguous += int(outcome == 'ambiguous')
        self._db.rollback()
        return ambiguous

    def _repair_current_document_version(
        self, *, document_id: int, source_id: int
    ) -> str:
        with KeyedMutationGuard.generation_barrier(self._db):
            key_context = lock_runtime_state(self._db)
            if key_context is None:
                self._db.rollback()
                return 'unresolved'
            acquire_projection(self._db, key_context)
            source_statement = select(Source).where(Source.id == source_id)
            document_statement = select(Document).where(
                Document.id == document_id
            )
            if self._db.get_bind().dialect.name == 'postgresql':
                source_statement = source_statement.with_for_update()
                document_statement = document_statement.with_for_update()
            source = self._db.scalar(source_statement)
            document = self._db.scalar(document_statement)
            if (
                source is None
                or document is None
                or document.source_id != source.id
                or document.current_document_version_id is not None
                or source.server_content_signature_schema
                != 'server-source-content:v1'
                or source.server_content_signature is None
            ):
                self._db.rollback()
                return 'unresolved'
            outcome, selected_version_id = self._pointer_candidate(
                document_id=document.id,
                source_id=source.id,
                for_update=(
                    self._db.get_bind().dialect.name == 'postgresql'
                ),
            )
            if outcome != 'repairable' or selected_version_id is None:
                self._db.rollback()
                return outcome
            document.current_document_version_id = selected_version_id
            self._db.commit()
            return 'repaired'

    def _pointer_candidate(
        self,
        *,
        document_id: int,
        source_id: int,
        for_update: bool,
    ) -> tuple[str, int | None]:
        source = self._db.get(Source, source_id)
        document = self._db.get(Document, document_id)
        if (
            source is None
            or document is None
            or document.source_id != source.id
            or document.current_document_version_id is not None
            or source.server_content_signature_schema
            != 'server-source-content:v1'
            or source.server_content_signature is None
        ):
            return 'unresolved', None
        try:
            policy = server_parser_policy_for_source(
                source_type=source.source_type,
                mime_type=(source.raw_metadata or {}).get('mime_type'),
            )
        except SourceContentUnverifiableError:
            return 'ambiguous', None
        versions_statement = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .order_by(DocumentVersion.id)
        )
        runs_statement = (
            select(DocumentParserRun)
            .where(
                DocumentParserRun.document_id == document.id,
                DocumentParserRun.source_id == source.id,
            )
            .order_by(DocumentParserRun.id)
        )
        if for_update:
            versions_statement = versions_statement.with_for_update()
            runs_statement = runs_statement.with_for_update()
        versions = tuple(self._db.scalars(versions_statement).all())
        runs = tuple(self._db.scalars(runs_statement).all())
        exact_runs = tuple(
            run
            for run in runs
            if run.server_content_signature_schema
            == 'server-source-content:v1'
            and run.server_content_signature
            == source.server_content_signature
            and run.parser_status == 'parsed'
            and run.parser_name == policy.parser_name
            and run.mime_type == policy.mime_type
            and run.parser_policy_version == policy.parser_policy_version
            and run.parser_version == policy.parser_version
            and run.chunk_policy_version == policy.chunk_policy_version
        )
        if len(exact_runs) != 1:
            return ('ambiguous' if runs else 'unresolved'), None
        run = exact_runs[0]
        version_ids = {version.id for version in versions}
        if run.document_version_id not in version_ids:
            return 'ambiguous', None
        chunks_statement = (
            select(DocumentChunk)
            .where(DocumentChunk.version_id == run.document_version_id)
            .order_by(DocumentChunk.chunk_index, DocumentChunk.id)
        )
        if for_update:
            chunks_statement = chunks_statement.with_for_update()
        chunks = tuple(self._db.scalars(chunks_statement).all())
        if (
            run.chunk_count <= 0
            or len(chunks) != run.chunk_count
            or [chunk.chunk_index for chunk in chunks]
            != list(range(run.chunk_count))
            or any(
                chunk.source_id != source.id
                or chunk.parser_run_id != run.id
                for chunk in chunks
            )
        ):
            return 'ambiguous', None
        return 'repairable', run.document_version_id

    def _stale_source_ids(self, *, limit: int) -> list[int]:
        evidence = TrustedKnowledgeEvidenceLink
        link = TrustedKnowledgeApprovalLink
        source_join = cast(Source.id, String(128)) == evidence.canonical_source_id
        parser_match_count = (
            select(func.count(DocumentParserRun.id))
            .select_from(DocumentParserRun)
            .join(Document, Document.id == DocumentParserRun.document_id)
            .where(
                Document.source_id == Source.id,
                Document.current_document_version_id
                == DocumentParserRun.document_version_id,
                DocumentParserRun.source_id == Source.id,
                DocumentParserRun.server_content_signature_schema
                == 'server-source-content:v1',
                DocumentParserRun.server_content_signature
                == Source.server_content_signature,
                DocumentParserRun.parser_policy_version.is_not(None),
                DocumentParserRun.parser_version.is_not(None),
                DocumentParserRun.chunk_policy_version.is_not(None),
                DocumentParserRun.revision_id
                == evidence.canonical_version_or_signature,
            )
            .correlate(Source, evidence)
            .scalar_subquery()
        )
        evidence_is_current = and_(
            Source.id.is_not(None),
            Source.source_type == evidence.canonical_source_kind,
            Source.server_content_signature_schema
            == 'server-source-content:v1',
            Source.server_content_signature.is_not(None),
            Source.permission_level.in_(_KNOWN_SERVING_PERMISSIONS),
            or_(
                evidence.canonical_version_or_signature
                == Source.server_content_signature,
                parser_match_count == 1,
            ),
        )
        stale_source_ids = set(
            self._db.scalars(
                select(evidence.canonical_source_id)
                .select_from(evidence)
                .join(link, link.id == evidence.approval_link_id)
                .outerjoin(Source, source_join)
                .where(
                    link.active.is_(True),
                    link.resolution_source == 'auto_policy',
                    ~evidence_is_current,
                )
                .distinct()
                .order_by(evidence.canonical_source_id)
                .limit(limit)
            ).all()
        )
        target_branches = (
            (DecisionRecord, ('decision_record', 'decision')),
            (HistoryEvent, ('history_event',)),
            (TimelineEvent, ('timeline_event',)),
            (Todo, ('todo',)),
        )
        for target, knowledge_types in target_branches:
            fingerprint = TrustedKnowledgeFingerprint
            permission_rank = _permission_rank_expression
            core_permissions = (
                target.permission_level,
                ReviewItem.permission_level,
                link.permission_level,
            )
            authoritative_permissions = (*core_permissions, Source.permission_level)
            permission_is_stale = or_(
                *(level.not_in(_PERMISSION_RANK) for level in core_permissions),
                and_(
                    fingerprint.id.is_not(None),
                    fingerprint.permission_level.not_in(_PERMISSION_RANK),
                ),
                *(
                    permission_rank(current) < permission_rank(authority)
                    for current in core_permissions
                    for authority in authoritative_permissions
                ),
                *(
                    and_(
                        fingerprint.id.is_not(None),
                        permission_rank(fingerprint.permission_level)
                        < permission_rank(authority),
                    )
                    for authority in authoritative_permissions
                ),
            )
            stale_source_ids.update(
                self._db.scalars(
                    select(evidence.canonical_source_id)
                    .select_from(evidence)
                    .join(link, link.id == evidence.approval_link_id)
                    .join(Source, source_join)
                    .join(ReviewItem, ReviewItem.id == link.review_item_id)
                    .join(target, target.id == link.knowledge_id)
                    .outerjoin(
                        fingerprint,
                        and_(
                            fingerprint.knowledge_type == link.knowledge_type,
                            fingerprint.knowledge_id == link.knowledge_id,
                        ),
                    )
                    .where(
                        link.active.is_(True),
                        link.resolution_source == 'auto_policy',
                        link.knowledge_type.in_(knowledge_types),
                        evidence_is_current,
                        permission_is_stale,
                    )
                    .distinct()
                    .order_by(evidence.canonical_source_id)
                    .limit(limit)
                ).all()
            )
        return sorted(
            source_id
            for value in stale_source_ids
            if (source_id := _source_id(value)) is not None
        )[:limit]

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
        evidence_ids = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink.id)
                .where(
                    TrustedKnowledgeEvidenceLink.canonical_source_id
                    == str(source_id)
                )
                .order_by(TrustedKnowledgeEvidenceLink.id)
                .limit(100)
            ).all()
        )
        if not evidence_ids:
            return False, 0, 0
        reconciled = revoked = narrowed = 0
        handled_items: set[int] = set()
        for evidence_id in evidence_ids:
            evidence = self._db.get(TrustedKnowledgeEvidenceLink, evidence_id)
            link = (
                self._db.get(
                    TrustedKnowledgeApprovalLink, evidence.approval_link_id
                )
                if evidence is not None
                else None
            )
            if (
                link is None
                or not link.active
                or link.resolution_source != 'auto_policy'
            ):
                continue
            reconciled = 1
            if link.review_item_id in handled_items:
                continue
            outcome = self._narrow_current_effect(
                source_id=source_id,
                evidence_id=evidence_id,
                approval_link_id=link.id,
            )
            if outcome == 'narrowed':
                narrowed += 1
                continue
            if outcome in {'current', 'gone'}:
                continue
            handled_items.add(link.review_item_id)
            context = self._mint_source_invalidation_context(
                review_item_id=link.review_item_id,
                canonical_source_id=str(source_id),
            )
            result = AutoReviewRevokeService(
                self._db,
                settings=self._settings,
                vector_writer=self._vector_writer,
            ).revoke_source_invalidated(context=context)
            revoked += int(not result.replayed)
        return bool(reconciled), revoked, narrowed

    def _narrow_current_effect(
        self,
        *,
        source_id: int,
        evidence_id: int,
        approval_link_id: int,
    ) -> str:
        link = self._db.get(TrustedKnowledgeApprovalLink, approval_link_id)
        if link is None:
            return 'gone'
        document_id = f'{link.knowledge_type}:{link.knowledge_id}'
        plan = build_serving_lock_plan(self._db, [document_id])
        self._db.rollback()
        with KeyedMutationGuard.generation_barrier(self._db):
            key_context = lock_runtime_state(self._db)
            if key_context is None:
                raise RuntimeError('key runtime unavailable')
            locked_context = ServingMutationLockCoordinator(
                db=self._db, settings=self._settings
            ).acquire(key_context=key_context, plan=plan)
            source = self._db.get(Source, source_id)
            evidence = self._db.get(
                TrustedKnowledgeEvidenceLink, evidence_id
            )
            link = self._db.get(
                TrustedKnowledgeApprovalLink, approval_link_id
            )
            if (
                evidence is None
                or link is None
                or not link.active
                or evidence.approval_link_id != link.id
            ):
                self._db.rollback()
                return 'gone'
            if not _evidence_is_current(self._db, source, evidence):
                self._db.rollback()
                return 'stale'
            target = self._db.get(
                knowledge_model_for_type(link.knowledge_type),
                link.knowledge_id,
            )
            item = self._db.get(ReviewItem, link.review_item_id)
            if target is None or item is None:
                self._db.rollback()
                return 'gone'
            strictest = _strictest_many(
                target.permission_level,
                link.permission_level,
                item.permission_level,
                source.permission_level,
            )
            target_changed = strictest != target.permission_level
            link_changed = strictest != link.permission_level
            item_changed = strictest != item.permission_level
            target.permission_level = strictest
            link.permission_level = strictest
            item.permission_level = strictest
            fingerprint_changed = self._narrow_fingerprint(
                document_id=document_id,
                permission_level=strictest,
            )
            changed = any(
                (target_changed, link_changed, item_changed, fingerprint_changed)
            )
            if changed and self._vector_writer is not None:
                if self._vector_writer.__class__.__name__ == 'PgVectorStore':
                    self._vector_writer.narrow_permissions(
                        [document_id],
                        strictest,
                        locked_context=locked_context,  # type: ignore[call-arg]
                    )
                else:
                    self._vector_writer.narrow_permissions(
                        [document_id], strictest
                    )
            if changed:
                self._refresh_index_state_hash(document_id)
            self._db.commit()
            return 'narrowed' if changed else 'current'

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


def _permission_rank_expression(column: object) -> object:
    return case(
        (column == 'public', _PERMISSION_RANK['public']),
        (column == 'internal', _PERMISSION_RANK['internal']),
        (column == 'restricted', _PERMISSION_RANK['restricted']),
        else_=len(_PERMISSION_RANK),
    )
