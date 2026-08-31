from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import (
    RagPublicCitationUrlValidator,
    StrictUnicodeScalarValidator,
    exact_utf8_bytes,
)
from backend.app.core.config import Settings

SupportMode = Literal['trusted_fact', 'source_observation']
ServingKind = Literal['raw_chunk', 'trusted_knowledge']
PermissionLevel = Literal['public', 'internal', 'restricted']
KnowledgeType = Literal[
    'decision_record',
    'history_event',
    'timeline_event',
    'todo',
]

SERVING_EVIDENCE_POLICY_VERSION = 'rag-serving-evidence:v1'
_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}
_LOWER_HEX_64 = re.compile(r'[0-9a-f]{64}')


@dataclass(frozen=True, slots=True)
class RawServingVersionEnvelope:
    serving_document_id: str
    source_row_id: int
    public_source_id: str
    document_id: int
    document_version_id: int
    current_document_version_id: int
    document_chunk_id: int
    parser_run_id: int
    external_revision: str | None
    server_content_signature_schema: str
    server_content_signature: str
    parser_policy_version: str
    parser_version: str
    chunk_policy_version: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    effective_permission: PermissionLevel


@dataclass(frozen=True, slots=True)
class RawChunkProvenance:
    branch: Literal['raw_chunk']
    raw_version: RawServingVersionEnvelope


@dataclass(frozen=True, slots=True)
class TrustedEvidenceLinkIdentity:
    trusted_knowledge_evidence_link_id: int
    approval_link_id: int
    canonical_source_kind: str
    canonical_source_id: str
    canonical_version_or_signature: str
    evidence_hash: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str


@dataclass(frozen=True, slots=True)
class SelectedCitationChild:
    trusted_knowledge_evidence_link_id: int
    source_row_id: int
    canonical_source_type: str
    canonical_version_or_signature: str
    review_item_source_pair_ordinal: int


@dataclass(frozen=True, slots=True)
class ExplicitApprovalProvenance:
    branch: Literal['explicit_approval']
    approval_link_id: int
    review_item_id: int
    security_scope_id: str
    promotion_effect_kind: str
    resolution_source: Literal['human', 'auto_policy']
    claim_fingerprint: str
    approval_permission_level: PermissionLevel
    approval_fingerprint_key_version: str
    approval_fingerprint_key_material_verifier: str
    evidence_links: tuple[TrustedEvidenceLinkIdentity, ...]
    selected_citation_child: SelectedCitationChild


@dataclass(frozen=True, slots=True)
class LegacyHumanProvenance:
    branch: Literal['legacy_human_base']
    legacy_binding: Literal['review_item']
    legacy_evidence_pairs_hmac: str
    legacy_review_item_permission_level: PermissionLevel
    legacy_source_review_item_id: int


@dataclass(frozen=True, slots=True)
class TrustedServingVersionEnvelope:
    serving_document_id: str
    knowledge_type: KnowledgeType
    knowledge_id: int
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    effective_permission: PermissionLevel
    provenance: ExplicitApprovalProvenance | LegacyHumanProvenance


ServingVersionEnvelope = RawServingVersionEnvelope | TrustedServingVersionEnvelope


@dataclass(frozen=True, slots=True)
class ServingEvidenceIdentity:
    serving_document_id: str
    serving_kind: ServingKind
    public_source_id: str
    public_source_type: str | None
    effective_permission: PermissionLevel
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    serving_version_fingerprint: str
    version_envelope: ServingVersionEnvelope


@dataclass(frozen=True, slots=True)
class ServingEvidence:
    serving_document_id: str
    serving_kind: ServingKind
    public_source_id: str
    public_source_type: str | None
    support_mode: SupportMode
    model_content: str
    title: str
    effective_permission: PermissionLevel
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    version_envelope: ServingVersionEnvelope
    provenance: RawChunkProvenance | ExplicitApprovalProvenance | LegacyHumanProvenance


@dataclass(frozen=True, slots=True)
class IndexableSourceObservation:
    identity: ServingEvidenceIdentity
    evidence: ServingEvidence
    raw_version: RawServingVersionEnvelope


@dataclass(frozen=True, slots=True)
class TrustedServingEnvelope:
    identity: ServingEvidenceIdentity
    evidence: ServingEvidence
    trusted_version: TrustedServingVersionEnvelope
    approval_provenance_hmac: str
    evidence_link_set_hmac: str | None
    evidence_link_hmacs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalServingProjection:
    """Fresh resolver-owned public bytes plus internal serving authority.

    This DTO may only be assembled from canonical rows.  Vector metadata and
    model output are deliberately unable to construct public citation bytes.
    """

    identity: ServingEvidenceIdentity
    evidence: ServingEvidence
    public_result_id: int
    source_url: str
    source_snippet: str
    parser_status: str | None
    parser_status_reason: str | None
    revision_id: str | None
    approval_provenance_hmac: str | None
    evidence_link_set_hmac: str | None
    evidence_link_hmacs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceAccessClassification:
    global_eligibility: Literal['eligible', 'ineligible']
    resource_scope: Literal['in_scope', 'out_of_scope', 'invalid_scope']
    permission_visibility: Literal[
        'visible',
        'denied_known',
        'unknown_permission',
    ]


def canonical_source_snippet(value: str) -> str:
    StrictUnicodeScalarValidator.validate(value)
    return ' '.join(value.split())[:240]


def known_permission(value: object) -> PermissionLevel | None:
    if type(value) is str and value in _PERMISSION_RANK:
        return cast(PermissionLevel, value)
    return None


def strictest_permission(values: tuple[object, ...]) -> PermissionLevel | None:
    permissions = tuple(known_permission(value) for value in values)
    if not permissions or any(value is None for value in permissions):
        return None
    return max(
        cast(tuple[PermissionLevel, ...], permissions),
        key=_PERMISSION_RANK.__getitem__,
    )


def require_exact_nonblank(value: object) -> str:
    if type(value) is not str:
        raise ValueError('canonical evidence text must be a string')
    StrictUnicodeScalarValidator.validate(value)
    if not value.strip():
        raise ValueError('canonical evidence text must be nonblank')
    return value


def require_positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError('canonical evidence identifier must be positive')
    return value


def require_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError('canonical evidence ordinal must be nonnegative')
    return value


def require_lower_hex_64(value: object) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise ValueError('canonical evidence HMAC must be lower hexadecimal')
    return value


def build_model_content_hmac(
    *,
    serving_kind: ServingKind,
    model_content: str,
    settings: Settings,
) -> str:
    _require_serving_kind(serving_kind)
    return _fingerprint(
        {
            'serving_kind': serving_kind,
            'model_content_bytes': exact_utf8_bytes(model_content),
        },
        schema_version='rag-model-content:v1',
        settings=settings,
    )


def build_canonical_citation_projection_hmac(
    *,
    public_source_id: str,
    public_source_type: str | None,
    source_url: str,
    source_snippet: str,
    effective_permission: PermissionLevel,
    settings: Settings,
) -> str:
    require_exact_nonblank(public_source_id)
    if public_source_type is not None:
        require_exact_nonblank(public_source_type)
    RagPublicCitationUrlValidator().validate(source_url)
    require_exact_nonblank(source_snippet)
    _require_permission(effective_permission)
    return _fingerprint(
        {
            'public_source_id_bytes': exact_utf8_bytes(public_source_id),
            'public_source_type_bytes': (
                exact_utf8_bytes(public_source_type)
                if public_source_type is not None
                else None
            ),
            'source_url_bytes': exact_utf8_bytes(source_url),
            'source_snippet_bytes': exact_utf8_bytes(source_snippet),
            'effective_permission': effective_permission,
        },
        schema_version='rag-canonical-citation:v1',
        settings=settings,
    )


def build_serving_identity_hmac(
    *,
    serving_document_id: str,
    serving_kind: ServingKind,
    settings: Settings,
) -> str:
    require_exact_nonblank(serving_document_id)
    _require_serving_kind(serving_kind)
    return _fingerprint(
        {
            'serving_document_id_bytes': exact_utf8_bytes(serving_document_id),
            'serving_kind': serving_kind,
        },
        schema_version='rag-serving-identity:v1',
        settings=settings,
    )


def build_serving_version_fingerprint(
    envelope: ServingVersionEnvelope,
    *,
    settings: Settings,
) -> str:
    return _fingerprint(
        serving_version_payload(envelope),
        schema_version='rag-serving-version:v1',
        settings=settings,
    )


def serving_version_payload(envelope: ServingVersionEnvelope) -> dict[str, object]:
    if isinstance(envelope, RawServingVersionEnvelope):
        _validate_raw_envelope(envelope)
        return {
            'kind': 'raw_chunk',
            'serving_document_id': exact_utf8_bytes(envelope.serving_document_id),
            'source_row_id': envelope.source_row_id,
            'public_source_id': exact_utf8_bytes(envelope.public_source_id),
            'document_id': envelope.document_id,
            'document_version_id': envelope.document_version_id,
            'current_document_version_id': envelope.current_document_version_id,
            'document_chunk_id': envelope.document_chunk_id,
            'parser_run_id': envelope.parser_run_id,
            'external_revision': _optional_exact_bytes(envelope.external_revision),
            'server_content_signature_schema': exact_utf8_bytes(
                envelope.server_content_signature_schema
            ),
            'server_content_signature': exact_utf8_bytes(
                envelope.server_content_signature
            ),
            'parser_policy_version': exact_utf8_bytes(envelope.parser_policy_version),
            'parser_version': exact_utf8_bytes(envelope.parser_version),
            'chunk_policy_version': exact_utf8_bytes(envelope.chunk_policy_version),
            'model_content_hash': envelope.model_content_hmac,
            'canonical_citation_projection_hash': (
                envelope.canonical_citation_projection_hmac
            ),
            'effective_permission': envelope.effective_permission,
        }
    if not isinstance(envelope, TrustedServingVersionEnvelope):
        raise TypeError('serving version envelope type is unsupported')
    _validate_trusted_envelope(envelope)
    return {
        'kind': 'trusted_knowledge',
        'serving_document_id': exact_utf8_bytes(envelope.serving_document_id),
        'knowledge_type': envelope.knowledge_type,
        'knowledge_id': envelope.knowledge_id,
        'model_content_hash': envelope.model_content_hmac,
        'canonical_citation_projection_hash': (
            envelope.canonical_citation_projection_hmac
        ),
        'effective_permission': envelope.effective_permission,
        'provenance': _provenance_payload(envelope.provenance),
    }


def build_source_signature_hmac(
    *,
    canonical_source_id: str,
    canonical_source_kind: str,
    canonical_version_or_signature: str,
    settings: Settings,
) -> str:
    return _fingerprint(
        {
            'canonical_source_id_bytes': exact_utf8_bytes(
                require_exact_nonblank(canonical_source_id)
            ),
            'canonical_source_kind_bytes': exact_utf8_bytes(
                require_exact_nonblank(canonical_source_kind)
            ),
            'canonical_version_or_signature_bytes': exact_utf8_bytes(
                require_exact_nonblank(canonical_version_or_signature)
            ),
        },
        schema_version='rag-approval-source-signature:v1',
        settings=settings,
    )


def build_source_version_identity_hmac(
    *,
    identity: TrustedEvidenceLinkIdentity,
    source_row_id: int,
    settings: Settings,
) -> str:
    _validate_evidence_link_identity(identity)
    require_positive_int(source_row_id)
    return _fingerprint(
        {
            'canonical_source_id_bytes': exact_utf8_bytes(identity.canonical_source_id),
            'canonical_source_kind_bytes': exact_utf8_bytes(
                identity.canonical_source_kind
            ),
            'canonical_version_or_signature_bytes': exact_utf8_bytes(
                identity.canonical_version_or_signature
            ),
            'evidence_hash_bytes': exact_utf8_bytes(identity.evidence_hash),
            'fingerprint_key_material_verifier': (
                identity.fingerprint_key_material_verifier
            ),
            'fingerprint_key_version_bytes': exact_utf8_bytes(
                identity.fingerprint_key_version
            ),
            'source_row_id': source_row_id,
        },
        schema_version='rag-approval-source-version-identity:v1',
        settings=settings,
    )


def build_selected_citation_child_hmac(
    child: SelectedCitationChild,
    *,
    settings: Settings,
) -> str:
    _validate_selected_child(child)
    return _fingerprint(
        {
            'canonical_source_type_bytes': exact_utf8_bytes(
                child.canonical_source_type
            ),
            'canonical_version_or_signature_bytes': exact_utf8_bytes(
                child.canonical_version_or_signature
            ),
            'review_item_source_pair_ordinal': (child.review_item_source_pair_ordinal),
            'source_row_id': child.source_row_id,
            'trusted_knowledge_evidence_link_id': (
                child.trusted_knowledge_evidence_link_id
            ),
        },
        schema_version='rag-approval-selected-citation-child:v1',
        settings=settings,
    )


def build_approval_evidence_link_hmac(
    *,
    approval_evidence_link_id: int,
    canonical_citation_projection_hmac: str,
    source_id: int,
    source_permission: PermissionLevel,
    source_signature_hmac: str,
    source_version_identity_hmac: str,
    settings: Settings,
) -> str:
    require_positive_int(approval_evidence_link_id)
    require_positive_int(source_id)
    _require_permission(source_permission)
    for value in (
        canonical_citation_projection_hmac,
        source_signature_hmac,
        source_version_identity_hmac,
    ):
        require_lower_hex_64(value)
    return _fingerprint(
        {
            'approval_evidence_link_id': approval_evidence_link_id,
            'canonical_citation_projection_hmac': (canonical_citation_projection_hmac),
            'source_id': source_id,
            'source_permission': source_permission,
            'source_signature_hmac': source_signature_hmac,
            'source_version_identity_hmac': source_version_identity_hmac,
        },
        schema_version='rag-approval-evidence-link:v1',
        settings=settings,
    )


def build_evidence_link_set_hmac(
    *,
    approval_link_id: int,
    ordered_link_hmacs: tuple[tuple[int, str], ...],
    settings: Settings,
) -> str:
    require_positive_int(approval_link_id)
    seen: set[int] = set()
    links: list[dict[str, object]] = []
    for ordinal, (link_id, link_hmac) in enumerate(ordered_link_hmacs):
        require_positive_int(link_id)
        require_lower_hex_64(link_hmac)
        if link_id in seen:
            raise ValueError('evidence link set contains a duplicate identifier')
        seen.add(link_id)
        links.append(
            {
                'approval_evidence_link_id': link_id,
                'evidence_link_hmac': link_hmac,
                'ordinal': ordinal,
            }
        )
    if not links:
        raise ValueError('evidence link set must be nonempty')
    return _fingerprint(
        {'approval_link_id': approval_link_id, 'links': links},
        schema_version='rag-approval-evidence-link-set:v1',
        settings=settings,
    )


def build_legacy_evidence_pairs_hmac(
    *,
    knowledge_type: str,
    knowledge_id: int,
    knowledge_review_status: str,
    knowledge_permission: PermissionLevel,
    review_item: object,
    settings: Settings,
) -> str:
    require_exact_nonblank(knowledge_type)
    require_positive_int(knowledge_id)
    require_exact_nonblank(knowledge_review_status)
    _require_permission(knowledge_permission)
    item_id = require_positive_int(getattr(review_item, 'id', None))
    item_permission = _require_permission(
        getattr(review_item, 'permission_level', None)
    )
    item_status = require_exact_nonblank(getattr(review_item, 'status', None))
    links = getattr(review_item, 'source_links', None)
    snippets = getattr(review_item, 'source_snippets', None)
    if (
        not isinstance(links, list)
        or not isinstance(snippets, list)
        or not links
        or len(links) != len(snippets)
    ):
        raise ValueError('legacy evidence pairs are invalid')
    pairs: list[dict[str, object]] = []
    for ordinal, (source_url, source_snippet) in enumerate(
        zip(links, snippets, strict=True)
    ):
        RagPublicCitationUrlValidator().validate(source_url)
        require_exact_nonblank(source_snippet)
        pairs.append(
            {
                'ordinal': ordinal,
                'source_snippet_bytes': exact_utf8_bytes(source_snippet),
                'source_url_bytes': exact_utf8_bytes(source_url),
            }
        )
    contract_version = getattr(review_item, 'candidate_contract_version', None)
    resolution_source = getattr(review_item, 'resolution_source', None)
    if contract_version is not None:
        require_exact_nonblank(contract_version)
    if resolution_source is not None:
        require_exact_nonblank(resolution_source)
    return _fingerprint(
        {
            'knowledge_permission': knowledge_permission,
            'knowledge_review_status_bytes': exact_utf8_bytes(knowledge_review_status),
            'knowledge_target_id': knowledge_id,
            'knowledge_type_bytes': exact_utf8_bytes(knowledge_type),
            'pairs': pairs,
            'review_item_contract_version_bytes': _optional_exact_bytes(
                contract_version
            ),
            'review_item_id': item_id,
            'review_item_permission': item_permission,
            'review_item_resolution_source_bytes': _optional_exact_bytes(
                resolution_source
            ),
            'review_item_status_bytes': exact_utf8_bytes(item_status),
        },
        schema_version='rag-legacy-human-evidence:v1',
        settings=settings,
    )


def build_approval_provenance_hmac(
    *,
    branch: Literal['explicit_approval', 'legacy_human_base'],
    approval_fingerprint_key_material_verifier: str | None,
    approval_fingerprint_key_version: str | None,
    approval_link_id: int | None,
    approval_permission: PermissionLevel,
    claim_fingerprint: str | None,
    evidence_link_set_hmac: str | None,
    legacy_evidence_pairs_hmac: str | None,
    promotion_effect_kind: str | None,
    resolution_source: str | None,
    review_item_id: int,
    security_scope_id: str | None,
    selected_citation_child_hmac: str | None,
    settings: Settings,
) -> str:
    if branch not in {'explicit_approval', 'legacy_human_base'}:
        raise ValueError('approval provenance branch is invalid')
    _require_permission(approval_permission)
    require_positive_int(review_item_id)
    for value in (
        approval_fingerprint_key_material_verifier,
        claim_fingerprint,
        evidence_link_set_hmac,
        legacy_evidence_pairs_hmac,
        selected_citation_child_hmac,
    ):
        if value is not None:
            require_lower_hex_64(value)
    if approval_link_id is not None:
        require_positive_int(approval_link_id)
    for value in (
        approval_fingerprint_key_version,
        promotion_effect_kind,
        resolution_source,
        security_scope_id,
    ):
        if value is not None:
            require_exact_nonblank(value)
    if branch == 'explicit_approval':
        required = (
            approval_fingerprint_key_material_verifier,
            approval_fingerprint_key_version,
            approval_link_id,
            claim_fingerprint,
            evidence_link_set_hmac,
            promotion_effect_kind,
            resolution_source,
            security_scope_id,
            selected_citation_child_hmac,
        )
        if (
            any(value is None for value in required)
            or legacy_evidence_pairs_hmac is not None
        ):
            raise ValueError('explicit approval provenance is incomplete')
    elif (
        any(
            value is not None
            for value in (
                approval_fingerprint_key_material_verifier,
                approval_fingerprint_key_version,
                approval_link_id,
                claim_fingerprint,
                evidence_link_set_hmac,
                promotion_effect_kind,
                resolution_source,
                security_scope_id,
            )
        )
        or legacy_evidence_pairs_hmac is None
        or selected_citation_child_hmac is not None
    ):
        raise ValueError('legacy approval provenance is incomplete')
    return _fingerprint(
        {
            'approval_fingerprint_key_material_verifier': (
                approval_fingerprint_key_material_verifier
            ),
            'approval_fingerprint_key_version_bytes': _optional_exact_bytes(
                approval_fingerprint_key_version
            ),
            'approval_link_id': approval_link_id,
            'approval_permission': approval_permission,
            'branch': branch,
            'claim_fingerprint': claim_fingerprint,
            'evidence_link_set_hmac': evidence_link_set_hmac,
            'legacy_evidence_pairs_hmac': legacy_evidence_pairs_hmac,
            'promotion_effect_kind_bytes': _optional_exact_bytes(promotion_effect_kind),
            'resolution_source_bytes': _optional_exact_bytes(resolution_source),
            'review_item_id': review_item_id,
            'security_scope_id_bytes': _optional_exact_bytes(security_scope_id),
            'selected_citation_child_hmac': selected_citation_child_hmac,
        },
        schema_version='rag-approval-provenance:v1',
        settings=settings,
    )


def _fingerprint(
    value: object,
    *,
    schema_version: str,
    settings: Settings,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        value,
        secret=secret,
        schema_version=schema_version,
        policy_version=SERVING_EVIDENCE_POLICY_VERSION,
    )


def _optional_exact_bytes(value: str | None) -> dict[str, int | str] | None:
    return exact_utf8_bytes(value) if value is not None else None


def _require_permission(value: object) -> PermissionLevel:
    permission = known_permission(value)
    if permission is None:
        raise ValueError('canonical evidence permission is unknown')
    return permission


def _require_serving_kind(value: object) -> ServingKind:
    if value not in {'raw_chunk', 'trusted_knowledge'}:
        raise ValueError('serving kind is invalid')
    return cast(ServingKind, value)


def _validate_raw_envelope(envelope: RawServingVersionEnvelope) -> None:
    for value in (
        envelope.source_row_id,
        envelope.document_id,
        envelope.document_version_id,
        envelope.current_document_version_id,
        envelope.document_chunk_id,
        envelope.parser_run_id,
    ):
        require_positive_int(value)
    for value in (
        envelope.serving_document_id,
        envelope.public_source_id,
        envelope.server_content_signature_schema,
        envelope.server_content_signature,
        envelope.parser_policy_version,
        envelope.parser_version,
        envelope.chunk_policy_version,
    ):
        require_exact_nonblank(value)
    if envelope.external_revision is not None:
        require_exact_nonblank(envelope.external_revision)
    require_lower_hex_64(envelope.model_content_hmac)
    require_lower_hex_64(envelope.canonical_citation_projection_hmac)
    _require_permission(envelope.effective_permission)


def _validate_trusted_envelope(envelope: TrustedServingVersionEnvelope) -> None:
    require_exact_nonblank(envelope.serving_document_id)
    if envelope.knowledge_type not in {
        'decision_record',
        'history_event',
        'timeline_event',
        'todo',
    }:
        raise ValueError('trusted knowledge type is invalid')
    require_positive_int(envelope.knowledge_id)
    require_lower_hex_64(envelope.model_content_hmac)
    require_lower_hex_64(envelope.canonical_citation_projection_hmac)
    _require_permission(envelope.effective_permission)


def _provenance_payload(
    provenance: ExplicitApprovalProvenance | LegacyHumanProvenance,
) -> dict[str, object]:
    if isinstance(provenance, ExplicitApprovalProvenance):
        _validate_explicit_provenance(provenance)
        return {
            'branch': provenance.branch,
            'approval_link_id': provenance.approval_link_id,
            'review_item_id': provenance.review_item_id,
            'security_scope_id': exact_utf8_bytes(provenance.security_scope_id),
            'promotion_effect_kind': exact_utf8_bytes(provenance.promotion_effect_kind),
            'resolution_source': provenance.resolution_source,
            'claim_fingerprint': provenance.claim_fingerprint,
            'approval_permission_level': provenance.approval_permission_level,
            'approval_fingerprint_key_version': exact_utf8_bytes(
                provenance.approval_fingerprint_key_version
            ),
            'approval_fingerprint_key_material_verifier': (
                provenance.approval_fingerprint_key_material_verifier
            ),
            'evidence_links': [
                _evidence_link_identity_payload(identity)
                for identity in provenance.evidence_links
            ],
            'selected_citation_child': _selected_child_payload(
                provenance.selected_citation_child
            ),
        }
    if not isinstance(provenance, LegacyHumanProvenance):
        raise TypeError('trusted provenance type is unsupported')
    require_lower_hex_64(provenance.legacy_evidence_pairs_hmac)
    _require_permission(provenance.legacy_review_item_permission_level)
    require_positive_int(provenance.legacy_source_review_item_id)
    if (
        provenance.branch != 'legacy_human_base'
        or provenance.legacy_binding != 'review_item'
    ):
        raise ValueError('legacy provenance branch is invalid')
    return {
        'branch': provenance.branch,
        'legacy_binding': provenance.legacy_binding,
        'legacy_evidence_pairs_hmac': provenance.legacy_evidence_pairs_hmac,
        'legacy_review_item_permission_level': (
            provenance.legacy_review_item_permission_level
        ),
        'legacy_source_review_item_id': provenance.legacy_source_review_item_id,
    }


def _validate_explicit_provenance(provenance: ExplicitApprovalProvenance) -> None:
    if provenance.branch != 'explicit_approval':
        raise ValueError('explicit provenance branch is invalid')
    require_positive_int(provenance.approval_link_id)
    require_positive_int(provenance.review_item_id)
    require_exact_nonblank(provenance.security_scope_id)
    require_exact_nonblank(provenance.promotion_effect_kind)
    if provenance.resolution_source not in {'human', 'auto_policy'}:
        raise ValueError('approval resolution source is invalid')
    require_lower_hex_64(provenance.claim_fingerprint)
    _require_permission(provenance.approval_permission_level)
    require_exact_nonblank(provenance.approval_fingerprint_key_version)
    require_lower_hex_64(provenance.approval_fingerprint_key_material_verifier)
    if not provenance.evidence_links:
        raise ValueError('explicit provenance evidence set is empty')
    expected = tuple(
        sorted(
            provenance.evidence_links,
            key=lambda value: (
                value.canonical_source_kind,
                value.canonical_source_id,
                value.canonical_version_or_signature,
                value.evidence_hash,
                value.trusted_knowledge_evidence_link_id,
            ),
        )
    )
    if provenance.evidence_links != expected or len(set(expected)) != len(expected):
        raise ValueError('explicit provenance evidence order is invalid')
    for identity in provenance.evidence_links:
        _validate_evidence_link_identity(identity)
        if identity.approval_link_id != provenance.approval_link_id:
            raise ValueError('explicit provenance evidence parent is inconsistent')
    _validate_selected_child(provenance.selected_citation_child)
    if provenance.selected_citation_child.trusted_knowledge_evidence_link_id not in {
        identity.trusted_knowledge_evidence_link_id
        for identity in provenance.evidence_links
    }:
        raise ValueError('selected citation child is outside the evidence set')


def _validate_evidence_link_identity(identity: TrustedEvidenceLinkIdentity) -> None:
    require_positive_int(identity.trusted_knowledge_evidence_link_id)
    require_positive_int(identity.approval_link_id)
    require_exact_nonblank(identity.canonical_source_kind)
    require_exact_nonblank(identity.canonical_source_id)
    require_exact_nonblank(identity.canonical_version_or_signature)
    require_lower_hex_64(identity.evidence_hash)
    require_exact_nonblank(identity.fingerprint_key_version)
    require_lower_hex_64(identity.fingerprint_key_material_verifier)


def _validate_selected_child(child: SelectedCitationChild) -> None:
    require_positive_int(child.trusted_knowledge_evidence_link_id)
    require_positive_int(child.source_row_id)
    require_exact_nonblank(child.canonical_source_type)
    require_exact_nonblank(child.canonical_version_or_signature)
    require_nonnegative_int(child.review_item_source_pair_ordinal)


def _evidence_link_identity_payload(
    identity: TrustedEvidenceLinkIdentity,
) -> dict[str, object]:
    _validate_evidence_link_identity(identity)
    return {
        'trusted_knowledge_evidence_link_id': (
            identity.trusted_knowledge_evidence_link_id
        ),
        'approval_link_id': identity.approval_link_id,
        'canonical_source_kind': exact_utf8_bytes(identity.canonical_source_kind),
        'canonical_source_id': exact_utf8_bytes(identity.canonical_source_id),
        'canonical_version_or_signature': exact_utf8_bytes(
            identity.canonical_version_or_signature
        ),
        'evidence_hash': identity.evidence_hash,
        'fingerprint_key_version': exact_utf8_bytes(identity.fingerprint_key_version),
        'fingerprint_key_material_verifier': (
            identity.fingerprint_key_material_verifier
        ),
    }


def _selected_child_payload(child: SelectedCitationChild) -> dict[str, object]:
    _validate_selected_child(child)
    return {
        'trusted_knowledge_evidence_link_id': (
            child.trusted_knowledge_evidence_link_id
        ),
        'source_row_id': child.source_row_id,
        'canonical_source_type': exact_utf8_bytes(child.canonical_source_type),
        'canonical_version_or_signature': exact_utf8_bytes(
            child.canonical_version_or_signature
        ),
        'review_item_source_pair_ordinal': (child.review_item_source_pair_ordinal),
    }
