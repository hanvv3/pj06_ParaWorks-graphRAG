from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import (
    RagPublicCitationUrlValidator,
    SecurityScope,
)
from backend.app.core.config import Settings
from backend.app.ingestion.source_authority import resolve_exact_source_authority
from backend.app.knowledge.serving_text import canonical_knowledge_text
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
    canonical_evidence_version_is_current,
    canonical_knowledge_document_id,
    canonical_knowledge_type,
    knowledge_model_for_type,
    knowledge_type_storage_aliases,
)
from backend.app.models import (
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    PermissionLevel,
    SelectedCitationChild,
    ServingEvidence,
    ServingEvidenceIdentity,
    TrustedEvidenceLinkIdentity,
    TrustedServingEnvelope,
    TrustedServingVersionEnvelope,
    build_approval_evidence_link_hmac,
    build_approval_provenance_hmac,
    build_canonical_citation_projection_hmac,
    build_evidence_link_set_hmac,
    build_legacy_evidence_pairs_hmac,
    build_model_content_hmac,
    build_selected_citation_child_hmac,
    build_serving_identity_hmac,
    build_serving_version_fingerprint,
    build_source_signature_hmac,
    build_source_version_identity_hmac,
    canonical_source_snippet,
    known_permission,
    require_exact_nonblank,
    require_lower_hex_64,
)
from backend.app.rag.source_observations import (
    CanonicalSourceObservationEligibilityService,
    CanonicalSourceObservationResolver,
)

_TRUSTED_TYPES = {
    'decision_record',
    'history_event',
    'timeline_event',
    'todo',
}
_SUPPORTED_SOURCE_TYPES = {'gmail', 'gmail_attachment', 'drive', 'calendar'}
_RESOLUTION_PRIORITY = {'human': 0, 'auto_policy': 1}


@dataclass(frozen=True, slots=True)
class _ResolvedCitationChild:
    identity: TrustedEvidenceLinkIdentity
    selected_child: SelectedCitationChild
    source: Source
    source_url: str
    source_snippet: str
    source_permission: PermissionLevel
    citation_projection_hmac: str
    evidence_link_hmac: str


@dataclass(frozen=True, slots=True)
class _ResolvedExplicitLink:
    link: TrustedKnowledgeApprovalLink
    item: ReviewItem
    identities: tuple[TrustedEvidenceLinkIdentity, ...]
    citations: tuple[_ResolvedCitationChild, ...]


@dataclass(frozen=True, slots=True)
class TrustedResourceAccessClassification:
    """Resource-only stage-one result with no citation or source projection."""

    global_eligibility: Literal['eligible', 'ineligible']
    resource_scope: Literal['in_scope', 'out_of_scope', 'invalid_scope']


class TrustedServingEnvelopeResolver:
    """Resolve canonical trusted evidence and its exact provenance branch."""

    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings
        self._eligibility = TrustedServingEligibilityService(db)

    def resolve_for_index(
        self,
        knowledge_type: str,
        knowledge_id: int,
    ) -> TrustedServingEnvelope | None:
        return self._resolve(
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            scope=None,
        )

    def resolve_for_scope(
        self,
        knowledge_type: str,
        knowledge_id: int,
        *,
        scope: SecurityScope,
    ) -> TrustedServingEnvelope | None:
        return self._resolve(
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
            scope=scope,
        )

    def _resolve(
        self,
        *,
        knowledge_type: str,
        knowledge_id: int,
        scope: SecurityScope | None,
    ) -> TrustedServingEnvelope | None:
        try:
            return self._resolve_exact(
                knowledge_type=knowledge_type,
                knowledge_id=knowledge_id,
                scope=scope,
            )
        except (SQLAlchemyError, TypeError, UnicodeError, ValueError):
            return None

    def _resolve_exact(
        self,
        *,
        knowledge_type: str,
        knowledge_id: int,
        scope: SecurityScope | None,
    ) -> TrustedServingEnvelope | None:
        if type(knowledge_id) is not int or knowledge_id <= 0:
            return None
        canonical_type = canonical_knowledge_type(knowledge_type)
        if canonical_type not in _TRUSTED_TYPES:
            return None
        target = self._db.get(
            knowledge_model_for_type(canonical_type),
            knowledge_id,
        )
        eligibility = self._eligibility.for_serving_envelope(
            canonical_type,
            knowledge_id,
        )
        effective_permission = known_permission(
            eligibility.effective_permission
        )
        if target is None or not eligibility.eligible or effective_permission is None:
            return None
        if scope is not None and not _target_is_visible_before_link_selection(
            scope=scope,
            target=target,
            effective_permission=effective_permission,
        ):
            return None
        model_content = require_exact_nonblank(
            canonical_knowledge_text(canonical_type, target)
        )
        title = require_exact_nonblank(target.title)
        serving_document_id = canonical_knowledge_document_id(
            canonical_type,
            knowledge_id,
        )
        model_content_hmac = build_model_content_hmac(
            serving_kind='trusted_knowledge',
            model_content=model_content,
            settings=self._settings,
        )
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
        resolved_links: list[_ResolvedExplicitLink] = []
        authorized_link_seen = False
        for link in links:
            if scope is not None:
                if not _link_scope_can_be_considered(scope=scope, link=link):
                    continue
                if not self._link_children_are_authorized(
                    scope=scope,
                    link=link,
                ):
                    continue
            authorized_link_seen = True
            resolved = self._resolve_explicit_link(
                link=link,
                scope=scope,
            )
            if resolved is None:
                return None
            resolved_links.append(resolved)
        if resolved_links:
            selected = min(
                resolved_links,
                key=lambda value: (
                    _RESOLUTION_PRIORITY[value.link.resolution_source],
                    value.link.id,
                ),
            )
            return self._build_explicit_envelope(
                canonical_type=canonical_type,
                knowledge_id=knowledge_id,
                serving_document_id=serving_document_id,
                model_content=model_content,
                title=title,
                model_content_hmac=model_content_hmac,
                effective_permission=effective_permission,
                selected=selected,
            )
        if authorized_link_seen:
            return None
        return self._build_legacy_envelope(
            canonical_type=canonical_type,
            knowledge_id=knowledge_id,
            serving_document_id=serving_document_id,
            model_content=model_content,
            title=title,
            model_content_hmac=model_content_hmac,
            effective_permission=effective_permission,
            target=target,
            scope=scope,
        )

    def _link_children_are_authorized(
        self,
        *,
        scope: SecurityScope,
        link: TrustedKnowledgeApprovalLink,
    ) -> bool:
        children = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id == link.id
                )
            ).all()
        )
        if not children:
            return False
        for child in children:
            try:
                source_id = int(child.canonical_source_id)
            except ValueError:
                return False
            source = self._db.get(Source, source_id)
            permission = (
                known_permission(source.permission_level)
                if source is not None
                else None
            )
            if (
                source is None
                or child.canonical_source_id != str(source_id)
                or permission not in scope.allowed_permission_levels
                or scope.source_constraints
                and f'source_pk:{source_id}' not in scope.source_constraints
            ):
                return False
        return True

    def _resolve_explicit_link(
        self,
        *,
        link: TrustedKnowledgeApprovalLink,
        scope: SecurityScope | None,
    ) -> _ResolvedExplicitLink | None:
        if (
            link.resolution_source not in _RESOLUTION_PRIORITY
            or not self._eligibility.approval_link_is_live(link.id)
        ):
            return None
        require_exact_nonblank(link.security_scope_id)
        require_exact_nonblank(link.promotion_effect_kind)
        require_lower_hex_64(link.claim_fingerprint)
        if known_permission(link.permission_level) is None:
            return None
        require_exact_nonblank(link.fingerprint_key_version)
        require_lower_hex_64(link.fingerprint_key_material_verifier)
        item = self._db.get(ReviewItem, link.review_item_id)
        if item is None:
            return None
        pairs = _validated_review_pairs(item)
        children = tuple(
            self._db.scalars(
                select(TrustedKnowledgeEvidenceLink).where(
                    TrustedKnowledgeEvidenceLink.approval_link_id == link.id
                )
            ).all()
        )
        if not children:
            return None
        identities = tuple(
            sorted(
                (_identity_from_row(child) for child in children),
                key=_identity_order,
            )
        )
        if (
            len(set(identities)) != len(identities)
            or len({value.trusted_knowledge_evidence_link_id for value in identities})
            != len(identities)
        ):
            return None
        citations: list[_ResolvedCitationChild] = []
        used_pair_ordinals: set[int] = set()
        for identity in identities:
            citation = self._resolve_citation_child(
                identity=identity,
                pairs=pairs,
                scope=scope,
            )
            if (
                citation is None
                or citation.selected_child.review_item_source_pair_ordinal
                in used_pair_ordinals
            ):
                return None
            used_pair_ordinals.add(
                citation.selected_child.review_item_source_pair_ordinal
            )
            citations.append(citation)
        return _ResolvedExplicitLink(
            link=link,
            item=item,
            identities=identities,
            citations=tuple(citations),
        )

    def _resolve_citation_child(
        self,
        *,
        identity: TrustedEvidenceLinkIdentity,
        pairs: tuple[tuple[str, str], ...],
        scope: SecurityScope | None,
    ) -> _ResolvedCitationChild | None:
        try:
            source_row_id = int(identity.canonical_source_id)
        except ValueError:
            return None
        if source_row_id <= 0 or identity.canonical_source_id != str(source_row_id):
            return None
        source = self._db.get(Source, source_row_id)
        source_permission = (
            known_permission(source.permission_level) if source is not None else None
        )
        if (
            source is None
            or source.source_type not in _SUPPORTED_SOURCE_TYPES
            or source.source_type != identity.canonical_source_kind
            or source_permission is None
            or not canonical_evidence_version_is_current(
                self._db,
                source=source,
                version_or_signature=identity.canonical_version_or_signature,
            )
        ):
            return None
        if scope is not None and (
            source_permission not in scope.allowed_permission_levels
            or scope.source_constraints
            and f'source_pk:{source.id}' not in scope.source_constraints
        ):
            return None
        authority = resolve_exact_source_authority(self._db, source=source)
        if authority is None:
            return None
        source_url = RagPublicCitationUrlValidator().validate(source.source_url)
        matching_ordinals: list[int] = []
        for ordinal, (pair_url, pair_snippet) in enumerate(pairs):
            if pair_url != source_url:
                continue
            if any(
                pair_snippet == chunk.source_snippet
                and pair_snippet == canonical_source_snippet(chunk.text)
                for chunk in authority.chunks
            ):
                matching_ordinals.append(ordinal)
        if not matching_ordinals:
            return None
        pair_ordinal = min(matching_ordinals)
        source_snippet = pairs[pair_ordinal][1]
        selected_child = SelectedCitationChild(
            trusted_knowledge_evidence_link_id=(
                identity.trusted_knowledge_evidence_link_id
            ),
            source_row_id=source.id,
            canonical_source_type=source.source_type,
            canonical_version_or_signature=(
                identity.canonical_version_or_signature
            ),
            review_item_source_pair_ordinal=pair_ordinal,
        )
        citation_hmac = build_canonical_citation_projection_hmac(
            public_source_id=source.source_id,
            public_source_type=source.source_type,
            source_url=source_url,
            source_snippet=source_snippet,
            effective_permission=source_permission,
            settings=self._settings,
        )
        signature_hmac = build_source_signature_hmac(
            canonical_source_id=identity.canonical_source_id,
            canonical_source_kind=identity.canonical_source_kind,
            canonical_version_or_signature=(
                identity.canonical_version_or_signature
            ),
            settings=self._settings,
        )
        version_hmac = build_source_version_identity_hmac(
            identity=identity,
            source_row_id=source.id,
            settings=self._settings,
        )
        return _ResolvedCitationChild(
            identity=identity,
            selected_child=selected_child,
            source=source,
            source_url=source_url,
            source_snippet=source_snippet,
            source_permission=source_permission,
            citation_projection_hmac=citation_hmac,
            evidence_link_hmac=build_approval_evidence_link_hmac(
                approval_evidence_link_id=(
                    identity.trusted_knowledge_evidence_link_id
                ),
                canonical_citation_projection_hmac=citation_hmac,
                source_id=source.id,
                source_permission=source_permission,
                source_signature_hmac=signature_hmac,
                source_version_identity_hmac=version_hmac,
                settings=self._settings,
            ),
        )

    def _build_explicit_envelope(
        self,
        *,
        canonical_type: str,
        knowledge_id: int,
        serving_document_id: str,
        model_content: str,
        title: str,
        model_content_hmac: str,
        effective_permission: PermissionLevel,
        selected: _ResolvedExplicitLink,
    ) -> TrustedServingEnvelope:
        selected_citation = selected.citations[0]
        citation_hmac = build_canonical_citation_projection_hmac(
            public_source_id=serving_document_id,
            public_source_type=canonical_type,
            source_url=selected_citation.source_url,
            source_snippet=selected_citation.source_snippet,
            effective_permission=effective_permission,
            settings=self._settings,
        )
        link = selected.link
        provenance = ExplicitApprovalProvenance(
            branch='explicit_approval',
            approval_link_id=link.id,
            review_item_id=selected.item.id,
            security_scope_id=link.security_scope_id,
            promotion_effect_kind=link.promotion_effect_kind,
            resolution_source=link.resolution_source,
            claim_fingerprint=link.claim_fingerprint,
            approval_permission_level=link.permission_level,
            approval_fingerprint_key_version=link.fingerprint_key_version,
            approval_fingerprint_key_material_verifier=(
                link.fingerprint_key_material_verifier
            ),
            evidence_links=selected.identities,
            selected_citation_child=selected_citation.selected_child,
        )
        evidence_link_set_hmac = build_evidence_link_set_hmac(
            approval_link_id=link.id,
            ordered_link_hmacs=tuple(
                (
                    citation.identity.trusted_knowledge_evidence_link_id,
                    citation.evidence_link_hmac,
                )
                for citation in selected.citations
            ),
            settings=self._settings,
        )
        selected_child_hmac = build_selected_citation_child_hmac(
            selected_citation.selected_child,
            settings=self._settings,
        )
        build_approval_provenance_hmac(
            branch='explicit_approval',
            approval_fingerprint_key_material_verifier=(
                link.fingerprint_key_material_verifier
            ),
            approval_fingerprint_key_version=link.fingerprint_key_version,
            approval_link_id=link.id,
            approval_permission=link.permission_level,
            claim_fingerprint=link.claim_fingerprint,
            evidence_link_set_hmac=evidence_link_set_hmac,
            legacy_evidence_pairs_hmac=None,
            promotion_effect_kind=link.promotion_effect_kind,
            resolution_source=link.resolution_source,
            review_item_id=selected.item.id,
            security_scope_id=link.security_scope_id,
            selected_citation_child_hmac=selected_child_hmac,
            settings=self._settings,
        )
        return _build_trusted_envelope(
            canonical_type=canonical_type,
            knowledge_id=knowledge_id,
            serving_document_id=serving_document_id,
            model_content=model_content,
            title=title,
            model_content_hmac=model_content_hmac,
            citation_hmac=citation_hmac,
            effective_permission=effective_permission,
            provenance=provenance,
            settings=self._settings,
        )

    def _build_legacy_envelope(
        self,
        *,
        canonical_type: str,
        knowledge_id: int,
        serving_document_id: str,
        model_content: str,
        title: str,
        model_content_hmac: str,
        effective_permission: PermissionLevel,
        target: object,
        scope: SecurityScope | None,
    ) -> TrustedServingEnvelope | None:
        if scope is not None and scope.source_constraints:
            return None
        review_item_id = getattr(target, 'source_review_item_id', None)
        if type(review_item_id) is not int or review_item_id <= 0:
            return None
        item = self._db.get(ReviewItem, review_item_id)
        if not _is_valid_legacy_binding(target=target, item=item):
            return None
        target_links = getattr(target, 'source_links', None)
        target_snippets = getattr(target, 'source_snippets', None)
        item_pairs = _validated_review_pairs(item)
        if (
            target_links != [pair[0] for pair in item_pairs]
            or target_snippets != [pair[1] for pair in item_pairs]
        ):
            return None
        legacy_hmac = build_legacy_evidence_pairs_hmac(
            knowledge_type=canonical_type,
            knowledge_id=knowledge_id,
            knowledge_review_status=target.review_status,
            knowledge_permission=target.permission_level,
            review_item=item,
            settings=self._settings,
        )
        citation_hmac = build_canonical_citation_projection_hmac(
            public_source_id=serving_document_id,
            public_source_type=canonical_type,
            source_url=item_pairs[0][0],
            source_snippet=item_pairs[0][1],
            effective_permission=effective_permission,
            settings=self._settings,
        )
        provenance = LegacyHumanProvenance(
            branch='legacy_human_base',
            legacy_binding='review_item',
            legacy_evidence_pairs_hmac=legacy_hmac,
            legacy_review_item_permission_level=item.permission_level,
            legacy_source_review_item_id=item.id,
        )
        build_approval_provenance_hmac(
            branch='legacy_human_base',
            approval_fingerprint_key_material_verifier=None,
            approval_fingerprint_key_version=None,
            approval_link_id=None,
            approval_permission=effective_permission,
            claim_fingerprint=None,
            evidence_link_set_hmac=None,
            legacy_evidence_pairs_hmac=legacy_hmac,
            promotion_effect_kind=None,
            resolution_source=None,
            review_item_id=item.id,
            security_scope_id=None,
            selected_citation_child_hmac=None,
            settings=self._settings,
        )
        return _build_trusted_envelope(
            canonical_type=canonical_type,
            knowledge_id=knowledge_id,
            serving_document_id=serving_document_id,
            model_content=model_content,
            title=title,
            model_content_hmac=model_content_hmac,
            citation_hmac=citation_hmac,
            effective_permission=effective_permission,
            provenance=provenance,
            settings=self._settings,
        )


class TrustedEvidenceAuthorizer:
    """Classify trusted access without projecting any citation/source bytes."""

    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._resolver = TrustedServingEnvelopeResolver(
            db=db,
            settings=settings,
        )

    def classify_access(
        self,
        scope: SecurityScope,
        evidence: TrustedServingEnvelope,
    ) -> EvidenceAccessClassification:
        permission = known_permission(evidence.identity.effective_permission)
        fresh = None
        if _trusted_envelope_is_consistent(evidence):
            fresh = self._resolver.resolve_for_index(
                evidence.trusted_version.knowledge_type,
                evidence.trusted_version.knowledge_id,
            )
        if fresh is None or not _same_trusted_authority(evidence, fresh):
            return EvidenceAccessClassification(
                global_eligibility='ineligible',
                resource_scope='invalid_scope',
                permission_visibility=(
                    'unknown_permission'
                    if permission is None
                    else 'denied_known'
                ),
            )
        if permission is None:
            visibility = 'unknown_permission'
        elif permission in scope.allowed_permission_levels:
            visibility = 'visible'
        else:
            visibility = 'denied_known'
        scoped = self._resolver.resolve_for_scope(
            fresh.trusted_version.knowledge_type,
            fresh.trusted_version.knowledge_id,
            scope=scope,
        )
        return EvidenceAccessClassification(
            global_eligibility='eligible',
            resource_scope='in_scope' if scoped is not None else 'out_of_scope',
            permission_visibility=visibility,
        )

    def classify_resource_access(
        self,
        scope: SecurityScope,
        evidence: TrustedServingEnvelope,
    ) -> TrustedResourceAccessClassification:
        """Classify exact resource membership without widening actor permissions."""
        fresh = None
        if _trusted_envelope_is_consistent(evidence):
            fresh = self._resolver.resolve_for_index(
                evidence.trusted_version.knowledge_type,
                evidence.trusted_version.knowledge_id,
            )
        if fresh is None or not _same_trusted_authority(evidence, fresh):
            return TrustedResourceAccessClassification(
                global_eligibility='ineligible',
                resource_scope='invalid_scope',
            )
        knowledge_type = fresh.trusted_version.knowledge_type
        knowledge_id = fresh.trusted_version.knowledge_id
        model = knowledge_model_for_type(knowledge_type)
        project_key = self._db.scalar(
            select(model.project_key).where(model.id == knowledge_id)
        )
        link_snapshot = _trusted_resource_link_snapshot(
            self._db,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
        )
        resource_scope = _classify_trusted_resource_snapshot(
            scope=scope,
            project_key=project_key,
            provenance=fresh.trusted_version.provenance,
            link_snapshot=link_snapshot,
        )
        final_fresh = self._resolver.resolve_for_index(
            knowledge_type,
            knowledge_id,
        )
        final_project_key = self._db.scalar(
            select(model.project_key)
            .where(model.id == knowledge_id)
            .execution_options(populate_existing=True)
        )
        final_link_snapshot = _trusted_resource_link_snapshot(
            self._db,
            knowledge_type=knowledge_type,
            knowledge_id=knowledge_id,
        )
        if (
            final_fresh is None
            or not _same_trusted_authority(fresh, final_fresh)
            or final_project_key != project_key
            or final_link_snapshot != link_snapshot
        ):
            return TrustedResourceAccessClassification(
                global_eligibility='ineligible',
                resource_scope='invalid_scope',
            )
        return TrustedResourceAccessClassification(
            global_eligibility='eligible',
            resource_scope=resource_scope,
        )


class ServingEvidenceResolver:
    """Fresh visible-only resolver shared by every future D serving consumer."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    def resolve_candidate(
        self,
        *,
        db: Session,
        identity: ServingEvidenceIdentity,
        scope: SecurityScope,
    ) -> ServingEvidence | None:
        if identity.serving_kind == 'raw_chunk':
            chunk_id = _canonical_positive_document_id(
                identity.serving_document_id,
                expected_prefix='chunk',
            )
            if chunk_id is None:
                return None
            observation = CanonicalSourceObservationResolver(
                db=db,
                settings=self._settings,
            ).resolve_for_index(chunk_id)
            if observation is None or observation.identity != identity:
                return None
            classification = CanonicalSourceObservationEligibilityService().classify_access(
                scope,
                observation,
            )
            if not _is_visible(classification):
                return None
            return observation.evidence
        if identity.serving_kind != 'trusted_knowledge':
            return None
        parsed = _canonical_trusted_document_id(identity.serving_document_id)
        if parsed is None:
            return None
        knowledge_type, knowledge_id = parsed
        envelope_resolver = TrustedServingEnvelopeResolver(
            db=db,
            settings=self._settings,
        )
        actorless = envelope_resolver.resolve_for_index(
            knowledge_type,
            knowledge_id,
        )
        if actorless is None or actorless.identity != identity:
            return None
        classification = TrustedEvidenceAuthorizer(
            db=db,
            settings=self._settings,
        ).classify_access(
            scope,
            actorless,
        )
        if not _is_visible(classification):
            return None
        scoped = envelope_resolver.resolve_for_scope(
            knowledge_type,
            knowledge_id,
            scope=scope,
        )
        return scoped.evidence if scoped is not None else None

    def resolve_trusted_candidate(
        self,
        *,
        db: Session,
        knowledge_type: str,
        knowledge_id: int,
        scope: SecurityScope,
    ) -> ServingEvidence | None:
        envelope_resolver = TrustedServingEnvelopeResolver(
            db=db,
            settings=self._settings,
        )
        actorless = envelope_resolver.resolve_for_index(
            knowledge_type,
            knowledge_id,
        )
        if actorless is None:
            return None
        classification = TrustedEvidenceAuthorizer(
            db=db,
            settings=self._settings,
        ).classify_access(
            scope,
            actorless,
        )
        if not _is_visible(classification):
            return None
        scoped = envelope_resolver.resolve_for_scope(
            knowledge_type,
            knowledge_id,
            scope=scope,
        )
        return scoped.evidence if scoped is not None else None


def _build_trusted_envelope(
    *,
    canonical_type: str,
    knowledge_id: int,
    serving_document_id: str,
    model_content: str,
    title: str,
    model_content_hmac: str,
    citation_hmac: str,
    effective_permission: PermissionLevel,
    provenance: ExplicitApprovalProvenance | LegacyHumanProvenance,
    settings: Settings,
) -> TrustedServingEnvelope:
    trusted_version = TrustedServingVersionEnvelope(
        serving_document_id=serving_document_id,
        knowledge_type=canonical_type,
        knowledge_id=knowledge_id,
        model_content_hmac=model_content_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        effective_permission=effective_permission,
        provenance=provenance,
    )
    version_fingerprint = build_serving_version_fingerprint(
        trusted_version,
        settings=settings,
    )
    identity = ServingEvidenceIdentity(
        serving_document_id=serving_document_id,
        serving_kind='trusted_knowledge',
        public_source_id=serving_document_id,
        public_source_type=canonical_type,
        effective_permission=effective_permission,
        model_content_hmac=model_content_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        serving_version_fingerprint=version_fingerprint,
        version_envelope=trusted_version,
    )
    evidence = ServingEvidence(
        serving_document_id=serving_document_id,
        serving_kind='trusted_knowledge',
        public_source_id=serving_document_id,
        public_source_type=canonical_type,
        support_mode='trusted_fact',
        model_content=model_content,
        title=title,
        effective_permission=effective_permission,
        serving_identity_hmac=build_serving_identity_hmac(
            serving_document_id=serving_document_id,
            serving_kind='trusted_knowledge',
            settings=settings,
        ),
        serving_version_fingerprint=version_fingerprint,
        model_content_hmac=model_content_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        version_envelope=trusted_version,
        provenance=provenance,
    )
    return TrustedServingEnvelope(
        identity=identity,
        evidence=evidence,
        trusted_version=trusted_version,
    )


def _identity_from_row(
    row: TrustedKnowledgeEvidenceLink,
) -> TrustedEvidenceLinkIdentity:
    identity = TrustedEvidenceLinkIdentity(
        trusted_knowledge_evidence_link_id=row.id,
        approval_link_id=row.approval_link_id,
        canonical_source_kind=require_exact_nonblank(row.canonical_source_kind),
        canonical_source_id=require_exact_nonblank(row.canonical_source_id),
        canonical_version_or_signature=require_exact_nonblank(
            row.canonical_version_or_signature
        ),
        evidence_hash=require_lower_hex_64(row.evidence_hash),
        fingerprint_key_version=require_exact_nonblank(
            row.fingerprint_key_version
        ),
        fingerprint_key_material_verifier=require_lower_hex_64(
            row.fingerprint_key_material_verifier
        ),
    )
    return identity


def _identity_order(identity: TrustedEvidenceLinkIdentity) -> tuple[object, ...]:
    return (
        identity.canonical_source_kind,
        identity.canonical_source_id,
        identity.canonical_version_or_signature,
        identity.evidence_hash,
        identity.trusted_knowledge_evidence_link_id,
    )


def _validated_review_pairs(item: ReviewItem) -> tuple[tuple[str, str], ...]:
    links = item.source_links
    snippets = item.source_snippets
    if (
        not isinstance(links, list)
        or not isinstance(snippets, list)
        or not links
        or len(links) != len(snippets)
    ):
        raise ValueError('review evidence pairs are invalid')
    pairs: list[tuple[str, str]] = []
    for source_url, source_snippet in zip(links, snippets, strict=True):
        pairs.append(
            (
                RagPublicCitationUrlValidator().validate(source_url),
                require_exact_nonblank(source_snippet),
            )
        )
    return tuple(pairs)


def _is_valid_legacy_binding(*, target: object, item: ReviewItem | None) -> bool:
    return bool(
        item is not None
        and target.review_status == 'approved'
        and known_permission(target.permission_level) is not None
        and target.source_review_item_id == item.id
        and item.status == 'approved'
        and item.resolution_source in {None, 'human'}
        and item.candidate_contract_version != 'c5-v1'
        and known_permission(item.permission_level) is not None
    )


def _target_is_visible_before_link_selection(
    *,
    scope: SecurityScope,
    target: object,
    effective_permission: PermissionLevel,
) -> bool:
    if effective_permission not in scope.allowed_permission_levels:
        return False
    if scope.project_constraints:
        project_key = getattr(target, 'project_key', None)
        if (
            type(project_key) is not str
            or f'project_key:{project_key}' not in scope.project_constraints
        ):
            return False
    return True


def _link_scope_can_be_considered(
    *,
    scope: SecurityScope,
    link: TrustedKnowledgeApprovalLink,
) -> bool:
    return bool(
        link.security_scope_id == scope.workspace_scope_id
        and link.resolution_source in _RESOLUTION_PRIORITY
    )


def _trusted_resource_scope(
    *,
    scope: SecurityScope,
    target: object,
    provenance: ExplicitApprovalProvenance | LegacyHumanProvenance,
) -> str:
    if scope.project_constraints:
        project_key = getattr(target, 'project_key', None)
        if (
            type(project_key) is not str
            or f'project_key:{project_key}' not in scope.project_constraints
        ):
            return 'out_of_scope'
    if isinstance(provenance, LegacyHumanProvenance):
        return 'out_of_scope' if scope.source_constraints else 'in_scope'
    if provenance.security_scope_id != scope.workspace_scope_id:
        return 'out_of_scope'
    if scope.source_constraints and any(
        f'source_pk:{identity.canonical_source_id}' not in scope.source_constraints
        for identity in provenance.evidence_links
    ):
        return 'out_of_scope'
    return 'in_scope'


def _trusted_resource_link_snapshot(
    db: Session,
    *,
    knowledge_type: str,
    knowledge_id: int,
) -> tuple[tuple[int, str, str, tuple[str, ...]], ...]:
    links = tuple(
        db.scalars(
            select(TrustedKnowledgeApprovalLink)
            .where(
                TrustedKnowledgeApprovalLink.knowledge_type.in_(
                    knowledge_type_storage_aliases(knowledge_type)
                ),
                TrustedKnowledgeApprovalLink.knowledge_id == knowledge_id,
                TrustedKnowledgeApprovalLink.active.is_(True),
            )
            .order_by(TrustedKnowledgeApprovalLink.id)
        ).all()
    )
    snapshot: list[tuple[int, str, str, tuple[str, ...]]] = []
    for link in links:
        child_ids = tuple(
            db.scalars(
                select(TrustedKnowledgeEvidenceLink.canonical_source_id)
                .where(TrustedKnowledgeEvidenceLink.approval_link_id == link.id)
                .order_by(TrustedKnowledgeEvidenceLink.id)
            ).all()
        )
        snapshot.append(
            (
                link.id,
                link.security_scope_id,
                link.resolution_source,
                child_ids,
            )
        )
    return tuple(snapshot)


def _classify_trusted_resource_snapshot(
    *,
    scope: SecurityScope,
    project_key: object,
    provenance: ExplicitApprovalProvenance | LegacyHumanProvenance,
    link_snapshot: tuple[tuple[int, str, str, tuple[str, ...]], ...],
) -> Literal['in_scope', 'out_of_scope']:
    if scope.project_constraints and (
        type(project_key) is not str
        or f'project_key:{project_key}' not in scope.project_constraints
    ):
        return 'out_of_scope'
    if isinstance(provenance, LegacyHumanProvenance):
        return 'out_of_scope' if scope.source_constraints else 'in_scope'
    for _, security_scope_id, resolution_source, child_ids in link_snapshot:
        if (
            security_scope_id != scope.workspace_scope_id
            or resolution_source not in _RESOLUTION_PRIORITY
            or not child_ids
        ):
            continue
        if scope.source_constraints and any(
            f'source_pk:{source_id}' not in scope.source_constraints
            for source_id in child_ids
        ):
            continue
        return 'in_scope'
    return 'out_of_scope'


def _trusted_envelope_is_consistent(envelope: TrustedServingEnvelope) -> bool:
    identity = envelope.identity
    evidence = envelope.evidence
    trusted_version = envelope.trusted_version
    return bool(
        identity.serving_kind == 'trusted_knowledge'
        and evidence.serving_kind == 'trusted_knowledge'
        and evidence.support_mode == 'trusted_fact'
        and identity.version_envelope is trusted_version
        and evidence.version_envelope is trusted_version
        and evidence.provenance is trusted_version.provenance
        and identity.serving_document_id == evidence.serving_document_id
        == trusted_version.serving_document_id
        and identity.public_source_id == evidence.public_source_id
        and identity.public_source_type == evidence.public_source_type
        == trusted_version.knowledge_type
        and identity.effective_permission == evidence.effective_permission
        == trusted_version.effective_permission
        and identity.model_content_hmac == evidence.model_content_hmac
        == trusted_version.model_content_hmac
        and identity.canonical_citation_projection_hmac
        == evidence.canonical_citation_projection_hmac
        == trusted_version.canonical_citation_projection_hmac
        and identity.serving_version_fingerprint
        == evidence.serving_version_fingerprint
    )


def _same_trusted_authority(
    supplied: TrustedServingEnvelope,
    fresh: TrustedServingEnvelope,
) -> bool:
    return bool(
        supplied.identity == fresh.identity
        and supplied.trusted_version == fresh.trusted_version
        and supplied.evidence.provenance == fresh.evidence.provenance
    )


def _canonical_positive_document_id(
    value: str,
    *,
    expected_prefix: str,
) -> int | None:
    prefix, separator, raw_id = value.partition(':')
    if (
        separator != ':'
        or prefix != expected_prefix
        or not raw_id.isascii()
        or not raw_id.isdecimal()
    ):
        return None
    parsed = int(raw_id)
    return parsed if parsed > 0 and str(parsed) == raw_id else None


def _canonical_trusted_document_id(value: str) -> tuple[str, int] | None:
    prefix, separator, _ = value.partition(':')
    if separator != ':' or prefix not in _TRUSTED_TYPES:
        return None
    parsed = _canonical_positive_document_id(value, expected_prefix=prefix)
    return (prefix, parsed) if parsed is not None else None


def _is_visible(classification: EvidenceAccessClassification) -> bool:
    return bool(
        classification.global_eligibility == 'eligible'
        and classification.resource_scope == 'in_scope'
        and classification.permission_visibility == 'visible'
    )
