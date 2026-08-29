from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, NoReturn

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.knowledge.trusted_provenance import has_legacy_human_base
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
    canonical_evidence_version_is_current,
    knowledge_model_for_type,
)
from backend.app.models import (
    AutoReviewAuditCorrection,
    AutoReviewPostAudit,
    AutoReviewRevocationAssessment,
    AutoReviewValidation,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    TrustedKnowledgeFingerprint,
    TrustedKnowledgeFingerprintProjectionState,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.indexing import VectorIndexWriter
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    build_serving_lock_plan,
)
from backend.app.review.actors import (
    ReviewResolutionActor,
    _assert_review_resolution_actor,
    auto_review_actor,
)

AutoReviewRevokeReasonCode = Literal[
    'business_withdrawal',
    'incorrect_content',
    'permission_violation',
    'wrong_source_version',
    'policy_violation',
]

_REASON_CODES = {
    'business_withdrawal',
    'incorrect_content',
    'permission_violation',
    'wrong_source_version',
    'policy_violation',
}


class AutoReviewRevokeRefused(ValueError):  # noqa: N818 - bounded domain refusal
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, init=False)
class SourceInvalidationRevokeContext:
    session_identity: int
    review_item_id: int
    canonical_source_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            'Source-invalidation contexts are minted only by reconciliation'
        )

    def __copy__(self) -> NoReturn:
        raise TypeError('Source-invalidation contexts cannot be copied')

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError('Source-invalidation contexts cannot be copied')

    def __reduce__(self) -> NoReturn:
        raise TypeError('Source-invalidation contexts cannot be serialized')


_SOURCE_INVALIDATION_CONTEXTS_INFO_KEY = (
    'paraworks_c5_source_invalidation_contexts'
)


@dataclass(frozen=True, slots=True)
class AutoReviewRevokeResult:
    review_item_id: int
    status: Literal['revoked']
    replayed: bool
    knowledge_remains_trusted: bool
    revoked_document_count: int


class AutoReviewRevokeService:
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

    def revoke(
        self,
        *,
        review_item_id: int,
        actor: ReviewResolutionActor,
        reason_code: AutoReviewRevokeReasonCode,
    ) -> AutoReviewRevokeResult:
        try:
            return self._revoke(
                review_item_id=review_item_id,
                actor=actor,
                reason_code=reason_code,
            )
        except Exception:
            self._db.rollback()
            raise

    def revoke_source_invalidated(
        self,
        *,
        context: SourceInvalidationRevokeContext,
    ) -> AutoReviewRevokeResult:
        """Server-only monotonic revoke after proving canonical-source drift."""
        self._validate_source_invalidation_context(context)
        try:
            return self._revoke(
                review_item_id=context.review_item_id,
                actor=auto_review_actor(policy_version='source-reconciliation:v1'),
                reason_code='wrong_source_version',
                source_invalidation_context=context,
            )
        except Exception:
            self._db.rollback()
            raise
        finally:
            contexts = self._db.info.get(
                _SOURCE_INVALIDATION_CONTEXTS_INFO_KEY, {}
            )
            contexts.pop(id(context), None)

    def _revoke(
        self,
        *,
        review_item_id: int,
        actor: ReviewResolutionActor,
        reason_code: AutoReviewRevokeReasonCode,
        source_invalidation_context: SourceInvalidationRevokeContext | None = None,
    ) -> AutoReviewRevokeResult:
        _assert_review_resolution_actor(actor)
        system_invalidation = source_invalidation_context is not None
        if system_invalidation:
            if actor.actor_type != 'auto_policy':
                raise AutoReviewRevokeRefused('forbidden')
        else:
            if (
                actor.actor_type != 'human'
                or 'human_review' not in actor.capabilities
            ):
                raise AutoReviewRevokeRefused('forbidden')
            if reason_code not in _REASON_CODES:
                raise AutoReviewRevokeRefused('invalid_reason_code')
            if reason_code != 'business_withdrawal':
                raise AutoReviewRevokeRefused('quality_audit_required')

        planned_links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.review_item_id == review_item_id
                )
            ).all()
        )
        planned_documents = sorted({
            f'{link.knowledge_type}:{link.knowledge_id}'
            for link in planned_links
        })
        if not planned_documents:
            raise AutoReviewRevokeRefused('incomplete_auto_approval')
        plan = build_serving_lock_plan(
            self._db,
            planned_documents,
            extra_review_item_ids=(review_item_id,),
        )
        self._db.rollback()
        with KeyedMutationGuard.generation_barrier(self._db):
            key_context = lock_runtime_state(self._db)
            if key_context is None:
                raise AutoReviewRevokeRefused('key_runtime_unavailable')
            locked_context = ServingMutationLockCoordinator(
                db=self._db, settings=self._settings
            ).acquire(key_context=key_context, plan=plan)
            item = self._db.get(ReviewItem, review_item_id)
            if item is None:
                raise AutoReviewRevokeRefused('not_found')
            if (
                not system_invalidation
                and (
                    item.permission_level not in actor.allowed_permission_levels
                    or (
                        item.status != 'revoked'
                        and not self._actor_can_access_current_evidence(item, actor)
                    )
                )
            ):
                raise AutoReviewRevokeRefused('not_found')

            assessment = self._db.scalar(
                select(AutoReviewRevocationAssessment).where(
                    AutoReviewRevocationAssessment.review_item_id == item.id
                )
            )
            if item.status == 'revoked':
                if system_invalidation:
                    return self._replay(item)
                if assessment is None or assessment.reason_code != reason_code:
                    raise AutoReviewRevokeRefused('revoke_reason_conflict')
                return self._replay(item)
            self._validate_complete_auto_item(item)
            if system_invalidation:
                self._validate_source_invalidation(
                    item.id, source_invalidation_context.canonical_source_id
                )
                from backend.app.review.auto_review_audit_transitions import (
                    AutoReviewAuditTransitionStore,
                )

                AutoReviewAuditTransitionStore(
                    self._db
                ).complete_source_invalidated(source_invalidation_context)
            else:
                self._validate_audit_gate(item.id)
            if (
                not system_invalidation
                and assessment is not None
                and assessment.reason_code != reason_code
            ):
                raise AutoReviewRevokeRefused('revoke_reason_conflict')

            links = self._lock_active_links(item.id)
            self._validate_complete_links(links)
            if assessment is None and not system_invalidation:
                assessment = self._create_assessment(
                    item=item,
                    actor=actor,
                    reason_code=reason_code,
                    key_version=key_context.key_version,
                    material_verifier=key_context.material_verifier,
                )
                self._db.add(assessment)
                self._db.flush([assessment])

            now = datetime.now(UTC)
            documents_to_revoke: list[str] = []
            any_target_remains = False
            for link in links:
                link.active = False
                link.revoked_at = now
                if self._target_has_other_provenance(link):
                    any_target_remains = True
                    continue
                target = self._db.get(
                    knowledge_model_for_type(link.knowledge_type),
                    link.knowledge_id,
                )
                if target is None:
                    raise AutoReviewRevokeRefused('provenance_target_missing')
                target.review_status = 'revoked'
                documents_to_revoke.append(
                    f'{link.knowledge_type}:{link.knowledge_id}'
                )

            documents_to_revoke = sorted(set(documents_to_revoke))
            tombstone_count = 0
            for document_id in documents_to_revoke:
                tombstone = self._db.scalar(
                    select(VectorServingTombstone).where(
                        VectorServingTombstone.document_id == document_id
                    )
                )
                if tombstone is None:
                    self._db.add(
                        VectorServingTombstone(
                            document_id=document_id,
                            source_review_item_id=item.id,
                            reason_code=reason_code,
                            revoked_at=now,
                        )
                    )
                    tombstone_count += 1
                elif tombstone.source_review_item_id == item.id:
                    tombstone_count += 1

            if documents_to_revoke:
                self._db.execute(
                    delete(VectorIndexState).where(
                        VectorIndexState.document_id.in_(documents_to_revoke)
                    )
                )
                for document_id in documents_to_revoke:
                    knowledge_type, raw_id = document_id.rsplit(':', maxsplit=1)
                    self._db.execute(
                        delete(TrustedKnowledgeFingerprint).where(
                            TrustedKnowledgeFingerprint.knowledge_type
                            == knowledge_type,
                            TrustedKnowledgeFingerprint.knowledge_id
                            == int(raw_id),
                        )
                    )
                if self._vector_writer is not None:
                    if self._vector_writer.__class__.__name__ == 'PgVectorStore':
                        self._vector_writer.delete_many(
                            documents_to_revoke,
                            locked_context=locked_context,  # type: ignore[call-arg]
                        )
                    else:
                        self._vector_writer.delete_many(documents_to_revoke)

            projection = self._db.scalar(
                select(TrustedKnowledgeFingerprintProjectionState).where(
                    TrustedKnowledgeFingerprintProjectionState.component
                    == 'trusted_knowledge_fingerprints'
                )
            )
            if projection is not None and documents_to_revoke:
                projection.ready = False
                projection.rebuild_required = True
                projection.completed_at = None
                projection.updated_at = now

            secret, _ = fingerprint_secret_bytes(self._settings)
            item.status = 'revoked'
            item.revoked_at = now
            item.revoked_by_subject_hmac = keyed_fingerprint(
                {'subject_id': actor.subject_id},
                secret=secret,
                schema_version=(
                    'auto-review-source-reconciler:v1'
                    if system_invalidation
                    else 'auto-review-revoke-actor:v1'
                ),
                policy_version=(
                    'auto-review-source-reconciler:v1'
                    if system_invalidation
                    else 'auto-review-revoke-actor:v1'
                ),
            )
            item.revoked_by_fingerprint_key_version = key_context.key_version
            item.revoked_by_fingerprint_key_material_verifier = (
                key_context.material_verifier
            )
            item.revoke_knowledge_remained_trusted = any_target_remains
            item.revoke_document_count = tombstone_count
            self._db.commit()
            return AutoReviewRevokeResult(
                review_item_id=item.id,
                status='revoked',
                replayed=False,
                knowledge_remains_trusted=any_target_remains,
                revoked_document_count=tombstone_count,
            )

    def _lock_item(self, review_item_id: int) -> ReviewItem | None:
        statement = select(ReviewItem).where(ReviewItem.id == review_item_id)
        if self._db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update()
        return self._db.scalar(statement)

    def _validate_complete_auto_item(self, item: ReviewItem) -> None:
        if item.status != 'approved' or item.resolution_source != 'auto_policy':
            raise AutoReviewRevokeRefused('unsupported_transition')
        validation = (
            self._db.get(AutoReviewValidation, item.auto_validation_id)
            if item.auto_validation_id is not None
            else None
        )
        if (
            validation is None
            or validation.review_item_id != item.id
            or validation.status != 'completed'
            or validation.policy_decision not in {'auto_approve', 'reuse_trusted'}
            or not item.resolution_policy_version
        ):
            raise AutoReviewRevokeRefused('incomplete_auto_approval')

    def _validate_audit_gate(self, review_item_id: int) -> None:
        correction = self._db.scalar(
            select(AutoReviewAuditCorrection).where(
                AutoReviewAuditCorrection.review_item_id == review_item_id
            )
        )
        if correction is not None:
            raise AutoReviewRevokeRefused('audit_required')
        audit = self._db.scalar(
            select(AutoReviewPostAudit).where(
                AutoReviewPostAudit.review_item_id == review_item_id
            )
        )
        if audit is None:
            return
        if audit.status == 'completed' and audit.outcome == 'confirmed':
            return
        raise AutoReviewRevokeRefused('audit_required')

    def _validate_source_invalidation(
        self, review_item_id: int, canonical_source_id: str
    ) -> None:
        links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.review_item_id == review_item_id,
                    TrustedKnowledgeApprovalLink.active.is_(True),
                )
            ).all()
        )
        evidence = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id.in_(
                        [link.id for link in links]
                    ),
                    TrustedKnowledgeEvidenceLink.canonical_source_id
                    == canonical_source_id,
                )
            ).all()
        ) if links else ()
        if not evidence:
            raise AutoReviewRevokeRefused('source_invalidation_not_proven')
        try:
            source = self._db.get(Source, int(canonical_source_id))
        except ValueError:
            source = None
        if source is None:
            return
        if all(
            source.source_type == child.canonical_source_kind
            and canonical_evidence_version_is_current(
                self._db,
                source=source,
                version_or_signature=child.canonical_version_or_signature,
            )
            and source.permission_level in {'public', 'internal'}
            for child in evidence
        ):
            raise AutoReviewRevokeRefused('source_invalidation_not_proven')

    def _validate_source_invalidation_context(
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

    def _lock_active_links(
        self, review_item_id: int
    ) -> tuple[TrustedKnowledgeApprovalLink, ...]:
        statement = (
            select(TrustedKnowledgeApprovalLink)
            .where(
                TrustedKnowledgeApprovalLink.review_item_id == review_item_id,
                TrustedKnowledgeApprovalLink.active.is_(True),
            )
            .order_by(
                TrustedKnowledgeApprovalLink.knowledge_type,
                TrustedKnowledgeApprovalLink.knowledge_id,
                TrustedKnowledgeApprovalLink.promotion_effect_kind,
            )
        )
        if self._db.get_bind().dialect.name == 'postgresql':
            statement = statement.with_for_update()
        links = tuple(self._db.scalars(statement).all())
        if not links:
            raise AutoReviewRevokeRefused('incomplete_auto_approval')
        return links

    def _validate_complete_links(
        self, links: tuple[TrustedKnowledgeApprovalLink, ...]
    ) -> None:
        key_identities = {
            (
                link.fingerprint_key_version,
                link.fingerprint_key_material_verifier,
            )
            for link in links
        }
        for link in links:
            children = tuple(
                self._db.scalars(
                    select(TrustedKnowledgeEvidenceLink).where(
                        TrustedKnowledgeEvidenceLink.approval_link_id == link.id
                    )
                ).all()
            )
            if not children:
                raise AutoReviewRevokeRefused('incomplete_auto_approval')
            key_identities.update(
                (
                    child.fingerprint_key_version,
                    child.fingerprint_key_material_verifier,
                )
                for child in children
            )
        if len(key_identities) != 1:
            raise AutoReviewRevokeRefused('mixed_fingerprint_key_identity')

    def _target_has_other_provenance(
        self, selected: TrustedKnowledgeApprovalLink
    ) -> bool:
        remaining = tuple(self._db.scalars(
            select(TrustedKnowledgeApprovalLink.id).where(
                TrustedKnowledgeApprovalLink.knowledge_type
                == selected.knowledge_type,
                TrustedKnowledgeApprovalLink.knowledge_id == selected.knowledge_id,
                TrustedKnowledgeApprovalLink.id != selected.id,
                TrustedKnowledgeApprovalLink.active.is_(True),
            )
        ).all())
        eligibility = TrustedServingEligibilityService(self._db)
        return any(
            eligibility.approval_link_is_live(link_id)
            for link_id in remaining
        ) or has_legacy_human_base(
            self._db,
            knowledge_type=selected.knowledge_type,
            knowledge_id=selected.knowledge_id,
        )

    def _actor_can_access_current_evidence(
        self,
        item: ReviewItem,
        actor: ReviewResolutionActor,
    ) -> bool:
        if item.permission_level not in actor.allowed_permission_levels:
            return False
        links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink).where(
                    TrustedKnowledgeApprovalLink.review_item_id == item.id,
                    TrustedKnowledgeApprovalLink.active.is_(True),
                )
            ).all()
        )
        if not links:
            return False
        children = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id.in_(
                        [link.id for link in links]
                    )
                )
            ).all()
        )
        if not children:
            return False
        for child in children:
            try:
                source = self._db.get(Source, int(child.canonical_source_id))
            except ValueError:
                return False
            if (
                source is None
                or source.permission_level not in actor.allowed_permission_levels
                or source.source_type != child.canonical_source_kind
                or not canonical_evidence_version_is_current(
                    self._db,
                    source=source,
                    version_or_signature=child.canonical_version_or_signature,
                )
            ):
                return False
        return True

    def _create_assessment(
        self,
        *,
        item: ReviewItem,
        actor: ReviewResolutionActor,
        reason_code: str,
        key_version: str,
        material_verifier: str,
    ) -> AutoReviewRevocationAssessment:
        secret, _ = fingerprint_secret_bytes(self._settings)
        return AutoReviewRevocationAssessment(
            review_item_id=item.id,
            reason_code=reason_code,
            actor_subject_hmac=keyed_fingerprint(
                {'subject_id': actor.subject_id},
                secret=secret,
                schema_version='auto-review-revoke-actor:v1',
                policy_version='auto-review-revoke-actor:v1',
            ),
            actor_fingerprint_key_version=key_version,
            actor_fingerprint_key_material_verifier=material_verifier,
        )

    @staticmethod
    def _replay(item: ReviewItem) -> AutoReviewRevokeResult:
        if (
            item.revoke_knowledge_remained_trusted is None
            or item.revoke_document_count is None
        ):
            raise AutoReviewRevokeRefused('incomplete_revoke_snapshot')
        return AutoReviewRevokeResult(
            review_item_id=item.id,
            status='revoked',
            replayed=True,
            knowledge_remains_trusted=item.revoke_knowledge_remained_trusted,
            revoked_document_count=item.revoke_document_count,
        )
