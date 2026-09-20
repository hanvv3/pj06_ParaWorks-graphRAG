from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import (
    RagPublicCitationUrlValidator,
    SecurityScope,
)
from backend.app.core.config import Settings
from backend.app.ingestion.source_authority import (
    exact_authority_contains_chunk,
    resolve_exact_source_authority,
)
from backend.app.ingestion.source_versions import source_version_ref
from backend.app.models import DocumentChunk, Source, VectorServingTombstone
from backend.app.rag.serving_contracts import (
    CanonicalServingProjection,
    EvidenceAccessClassification,
    IndexableSourceObservation,
    RawChunkProvenance,
    RawServingVersionEnvelope,
    ServingEvidence,
    ServingEvidenceIdentity,
    build_canonical_citation_projection_hmac,
    build_model_content_hmac,
    build_serving_identity_hmac,
    build_serving_version_fingerprint,
    canonical_source_snippet,
    known_permission,
    require_exact_nonblank,
    strictest_permission,
)

_SUPPORTED_SOURCE_TYPES = {'gmail', 'gmail_attachment', 'drive', 'calendar'}


class CanonicalSourceObservationResolver:
    """Resolve one actor-independent current raw serving observation."""

    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def resolve_for_index(self, chunk_id: int) -> IndexableSourceObservation | None:
        try:
            return self.resolve_for_index_strict(chunk_id)
        except (SQLAlchemyError, TypeError, UnicodeError, ValueError):
            return None

    def resolve_for_index_strict(
        self, chunk_id: int
    ) -> IndexableSourceObservation | None:
        return self._resolve_for_index(chunk_id)

    def resolve_projection_for_scope(
        self,
        chunk_id: int,
        *,
        scope: SecurityScope,
    ) -> CanonicalServingProjection | None:
        try:
            return self.resolve_projection_for_scope_strict(chunk_id, scope=scope)
        except (SQLAlchemyError, TypeError, UnicodeError, ValueError):
            return None

    def resolve_projection_for_scope_strict(
        self,
        chunk_id: int,
        *,
        scope: SecurityScope,
    ) -> CanonicalServingProjection | None:
        """Return public raw bytes only after a fresh scoped canonical read."""
        observation = self.resolve_for_index_strict(chunk_id)
        return self._project_observation(observation, scope=scope)

    def resolve_approved_slack_child_for_scope_strict(
        self, chunk_id: int, *, scope: SecurityScope
    ) -> CanonicalServingProjection | None:
        """Approved graph child/exact candidate reconstruction, never discovery.

        A link row or caller-supplied identity is not authority. Reconstruct the
        entire current approved envelope, including every source binding, before
        resolving this exact source/snippet. Bound fan-out and fail closed.
        """
        from backend.app.models import (
            ReviewItem,
            TrustedKnowledgeApprovalLink,
            TrustedKnowledgeEvidenceLink,
        )
        from backend.app.rag.serving_contracts import ExplicitApprovalProvenance
        from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver

        chunk = self._db.get(DocumentChunk, chunk_id)
        source = self._db.get(Source, chunk.source_id) if chunk is not None else None
        if (
            source is None
            or source.source_type != 'slack'
            or source_version_ref(source) is None
            or self._is_tombstoned(f'chunk:{chunk_id}')
        ):
            return None
        links = tuple(
            self._db.scalars(
                select(TrustedKnowledgeApprovalLink)
                .join(
                    TrustedKnowledgeEvidenceLink,
                    TrustedKnowledgeEvidenceLink.approval_link_id
                    == TrustedKnowledgeApprovalLink.id,
                )
                .where(
                    TrustedKnowledgeEvidenceLink.canonical_source_id == str(source.id),
                    TrustedKnowledgeEvidenceLink.canonical_source_kind == 'slack',
                    TrustedKnowledgeApprovalLink.active.is_(True),
                )
                .order_by(TrustedKnowledgeApprovalLink.id)
                .limit(51)
            )
        )
        if len(links) > 50:
            return None
        resolver = TrustedServingEnvelopeResolver(db=self._db, settings=self._settings)
        for link in links:
            envelope = resolver.resolve_for_scope_strict(
                link.knowledge_type, link.knowledge_id, scope=scope
            )
            if envelope is None:
                continue
            provenance = envelope.evidence.provenance
            if (
                not isinstance(provenance, ExplicitApprovalProvenance)
                or provenance.approval_link_id != link.id
                or not any(
                    child.canonical_source_kind == 'slack'
                    and child.canonical_source_id == str(source.id)
                    for child in provenance.evidence_links
                )
            ):
                continue
            item = self._db.get(ReviewItem, provenance.review_item_id)
            if (source.source_url, chunk.source_snippet) not in set(
                zip(item.source_links, item.source_snippets, strict=True)
            ):
                continue
            observation = self._resolve_current_chunk(chunk, source)
            return self._project_observation(observation, scope=scope)
        return None

    def _project_observation(self, observation, *, scope):
        if observation is None:
            return None
        access = CanonicalSourceObservationEligibilityService().classify_access(
            scope, observation
        )
        if (
            access.global_eligibility != 'eligible'
            or access.resource_scope != 'in_scope'
            or access.permission_visibility != 'visible'
        ):
            return None
        source = self._db.get(Source, observation.raw_version.source_row_id)
        chunk = self._db.get(DocumentChunk, observation.raw_version.document_chunk_id)
        authority = (
            resolve_exact_source_authority(self._db, source=source)
            if source is not None
            else None
        )
        if (
            source is None
            or chunk is None
            or authority is None
            or authority.parser_run.id != observation.raw_version.parser_run_id
        ):
            return None
        source_url = RagPublicCitationUrlValidator().validate(source.source_url)
        source_snippet = require_exact_nonblank(chunk.source_snippet)
        return CanonicalServingProjection(
            identity=observation.identity,
            evidence=observation.evidence,
            public_result_id=chunk.id,
            source_url=source_url,
            source_snippet=source_snippet,
            parser_status=authority.parser_run.parser_status,
            parser_status_reason=authority.parser_run.parser_status_reason,
            revision_id=authority.parser_run.revision_id or None,
            approval_provenance_hmac=None,
            evidence_link_set_hmac=None,
        )

    def _resolve_for_index(self, chunk_id: int) -> IndexableSourceObservation | None:
        if type(chunk_id) is not int or chunk_id <= 0:
            return None
        chunk = self._db.get(DocumentChunk, chunk_id)
        if chunk is None or chunk.parser_run_id is None:
            return None
        source = self._db.get(Source, chunk.source_id)
        if (
            source is None
            or source.source_type not in _SUPPORTED_SOURCE_TYPES
            or source_version_ref(source) is None
            or self._is_tombstoned(f'chunk:{chunk.id}')
        ):
            return None
        return self._resolve_current_chunk(chunk, source)

    def _resolve_current_chunk(self, chunk, source):
        authority = resolve_exact_source_authority(self._db, source=source)
        if (
            authority is None
            or not exact_authority_contains_chunk(authority, chunk)
            or authority.document.source_id != source.id
            or authority.version.document_id != authority.document.id
            or authority.document.current_document_version_id != authority.version.id
            or chunk.version_id != authority.version.id
            or chunk.source_id != source.id
            or chunk.parser_run_id != authority.parser_run.id
        ):
            return None
        permission = strictest_permission(
            (source.permission_level, chunk.permission_level)
        )
        if permission is None:
            return None
        require_exact_nonblank(source.source_id)
        source_url = RagPublicCitationUrlValidator().validate(source.source_url)
        title = require_exact_nonblank(source.title)
        model_content = require_exact_nonblank(chunk.text)
        source_snippet = require_exact_nonblank(chunk.source_snippet)
        if source_snippet != canonical_source_snippet(model_content):
            return None
        parser_run = authority.parser_run
        server_signature_schema = require_exact_nonblank(
            parser_run.server_content_signature_schema
        )
        server_signature = require_exact_nonblank(parser_run.server_content_signature)
        if (
            server_signature_schema != source.server_content_signature_schema
            or server_signature != source.server_content_signature
        ):
            return None
        parser_policy_version = require_exact_nonblank(parser_run.parser_policy_version)
        parser_version = require_exact_nonblank(parser_run.parser_version)
        chunk_policy_version = require_exact_nonblank(parser_run.chunk_policy_version)
        external_revision = parser_run.revision_id or None
        if external_revision is not None:
            require_exact_nonblank(external_revision)

        serving_document_id = f'chunk:{chunk.id}'
        model_content_hmac = build_model_content_hmac(
            serving_kind='raw_chunk',
            model_content=model_content,
            settings=self._settings,
        )
        citation_hmac = build_canonical_citation_projection_hmac(
            public_source_id=source.source_id,
            public_source_type=source.source_type,
            source_url=source_url,
            source_snippet=source_snippet,
            effective_permission=permission,
            settings=self._settings,
        )
        raw_version = RawServingVersionEnvelope(
            serving_document_id=serving_document_id,
            source_row_id=source.id,
            public_source_id=source.source_id,
            document_id=authority.document.id,
            document_version_id=authority.version.id,
            current_document_version_id=authority.document.current_document_version_id,
            document_chunk_id=chunk.id,
            parser_run_id=parser_run.id,
            external_revision=external_revision,
            server_content_signature_schema=server_signature_schema,
            server_content_signature=server_signature,
            parser_policy_version=parser_policy_version,
            parser_version=parser_version,
            chunk_policy_version=chunk_policy_version,
            model_content_hmac=model_content_hmac,
            canonical_citation_projection_hmac=citation_hmac,
            effective_permission=permission,
        )
        version_fingerprint = build_serving_version_fingerprint(
            raw_version,
            settings=self._settings,
        )
        identity = ServingEvidenceIdentity(
            serving_document_id=serving_document_id,
            serving_kind='raw_chunk',
            public_source_id=source.source_id,
            public_source_type=source.source_type,
            effective_permission=permission,
            model_content_hmac=model_content_hmac,
            canonical_citation_projection_hmac=citation_hmac,
            serving_version_fingerprint=version_fingerprint,
            version_envelope=raw_version,
        )
        provenance = RawChunkProvenance(
            branch='raw_chunk',
            raw_version=raw_version,
        )
        evidence = ServingEvidence(
            serving_document_id=serving_document_id,
            serving_kind='raw_chunk',
            public_source_id=source.source_id,
            public_source_type=source.source_type,
            support_mode='source_observation',
            model_content=model_content,
            title=title,
            effective_permission=permission,
            serving_identity_hmac=build_serving_identity_hmac(
                serving_document_id=serving_document_id,
                serving_kind='raw_chunk',
                settings=self._settings,
            ),
            serving_version_fingerprint=version_fingerprint,
            model_content_hmac=model_content_hmac,
            canonical_citation_projection_hmac=citation_hmac,
            version_envelope=raw_version,
            provenance=provenance,
        )
        return IndexableSourceObservation(
            identity=identity,
            evidence=evidence,
            raw_version=raw_version,
        )

    def _is_tombstoned(self, document_id: str) -> bool:
        return (
            self._db.scalar(
                select(VectorServingTombstone.id).where(
                    VectorServingTombstone.document_id == document_id
                )
            )
            is not None
        )


class CanonicalSourceObservationEligibilityService:
    """Classify a canonical raw observation without returning citation bytes."""

    def classify_access(
        self,
        scope: SecurityScope,
        observation: IndexableSourceObservation,
    ) -> EvidenceAccessClassification:
        permission = known_permission(observation.identity.effective_permission)
        globally_eligible = _observation_is_consistent(observation)
        if not globally_eligible:
            return EvidenceAccessClassification(
                global_eligibility='ineligible',
                resource_scope='invalid_scope',
                permission_visibility=(
                    'unknown_permission' if permission is None else 'denied_known'
                ),
            )
        if scope.project_constraints:
            resource_scope = 'invalid_scope'
        elif (
            scope.resource_scope_mode == 'all_current_scope'
            or f'source_pk:{observation.raw_version.source_row_id}'
            in scope.source_constraints
        ):
            resource_scope = 'in_scope'
        else:
            resource_scope = 'out_of_scope'
        if permission is None:
            visibility = 'unknown_permission'
        elif permission in scope.allowed_permission_levels:
            visibility = 'visible'
        else:
            visibility = 'denied_known'
        return EvidenceAccessClassification(
            global_eligibility='eligible',
            resource_scope=resource_scope,
            permission_visibility=visibility,
        )


def _observation_is_consistent(observation: IndexableSourceObservation) -> bool:
    identity = observation.identity
    evidence = observation.evidence
    raw_version = observation.raw_version
    return bool(
        identity.serving_kind == 'raw_chunk'
        and evidence.serving_kind == 'raw_chunk'
        and evidence.support_mode == 'source_observation'
        and identity.version_envelope is raw_version
        and evidence.version_envelope is raw_version
        and type(evidence.provenance) is RawChunkProvenance
        and evidence.provenance.raw_version is raw_version
        and identity.serving_document_id
        == evidence.serving_document_id
        == raw_version.serving_document_id
        and identity.public_source_id
        == evidence.public_source_id
        == raw_version.public_source_id
        and identity.effective_permission
        == evidence.effective_permission
        == raw_version.effective_permission
        and identity.model_content_hmac
        == evidence.model_content_hmac
        == raw_version.model_content_hmac
        and identity.canonical_citation_projection_hmac
        == evidence.canonical_citation_projection_hmac
        == raw_version.canonical_citation_projection_hmac
        and identity.serving_version_fingerprint == evidence.serving_version_fingerprint
    )
