from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import (
    RagPublicCitationUrlValidator,
    SecurityScope,
    exact_utf8_bytes,
)
from backend.app.core.config import Settings
from backend.app.rag.retrieval import EvidenceSlot, EvidenceSlotId
from backend.app.rag.serving_contracts import (
    CanonicalServingProjection,
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    PermissionLevel,
    RawChunkProvenance,
    ServingEvidence,
    ServingEvidenceIdentity,
    SupportMode,
    build_canonical_citation_projection_hmac,
    build_model_content_hmac,
    build_serving_identity_hmac,
    build_serving_version_fingerprint,
    known_permission,
    require_exact_nonblank,
    require_lower_hex_64,
    require_nonnegative_int,
    require_positive_int,
)
from backend.app.rag.trusted_evidence import ServingEvidenceResolver

DependencyRole = Literal['selected_citation', 'unselected_model_influence']
_SLOT_IDS: tuple[EvidenceSlotId, ...] = ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8')


class CanonicalProjectionResolverPort(Protocol):
    def resolve_projection_candidate(
        self, *, db: object, identity: ServingEvidenceIdentity, scope: SecurityScope
    ) -> CanonicalServingProjection | None: ...


@dataclass(frozen=True, slots=True)
class V1EvidenceProjection:
    citations: tuple[dict[str, object], ...]
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    search_results: tuple[dict[str, object], ...]
    projection_hmac: str


@dataclass(frozen=True, slots=True)
class PreparedModelInfluenceObservation:
    slot_id: EvidenceSlotId
    support_mode: SupportMode
    lookup_identity: ServingEvidenceIdentity
    effective_permission: PermissionLevel
    serving_identity_hmac: str
    serving_version_fingerprint: str
    model_content_hmac: str
    canonical_citation_projection_hmac: str
    approval_provenance_hmac: str | None
    evidence_link_set_hmac: str | None
    observation_hmac: str


@dataclass(frozen=True, slots=True)
class ModelInfluenceDependencySnapshot:
    observation: PreparedModelInfluenceObservation
    dependency_role: DependencyRole
    fresh_lookup_identity: ServingEvidenceIdentity
    dependency_hmac: str


@dataclass(frozen=True, slots=True)
class ProjectionFence:
    prepared_corpus_generation: int
    prepared_index_generation: int | None
    current_corpus_generation: int
    current_index_generation: int | None
    prepared_readiness_hmac: str | None = None
    current_readiness_hmac: str | None = None
    prepared_hidden_membership_hmac: str | None = None
    current_hidden_membership_hmac: str | None = None

    @classmethod
    def generations_only(
        cls, corpus_generation: int, index_generation: int | None
    ) -> ProjectionFence:
        return cls(
            prepared_corpus_generation=corpus_generation,
            prepared_index_generation=index_generation,
            current_corpus_generation=corpus_generation,
            current_index_generation=index_generation,
        )

    def valid(self, *, require_hidden: bool = False) -> bool:
        try:
            require_nonnegative_int(self.prepared_corpus_generation)
            require_nonnegative_int(self.current_corpus_generation)
            for value in (
                self.prepared_index_generation,
                self.current_index_generation,
            ):
                if value is not None:
                    require_nonnegative_int(value)
            for value in (
                self.prepared_readiness_hmac,
                self.current_readiness_hmac,
                self.prepared_hidden_membership_hmac,
                self.current_hidden_membership_hmac,
            ):
                if value is not None:
                    require_lower_hex_64(value)
        except ValueError:
            return False
        return bool(
            self.prepared_corpus_generation == self.current_corpus_generation
            and self.prepared_index_generation == self.current_index_generation
            and self.prepared_readiness_hmac == self.current_readiness_hmac
            and self.prepared_hidden_membership_hmac
            == self.current_hidden_membership_hmac
            and (
                self.prepared_index_generation is None
                or self.prepared_readiness_hmac is not None
            )
            and (not require_hidden or self.prepared_hidden_membership_hmac is not None)
        )


class CanonicalEvidenceProjector:
    """Assemble immutable public projection bytes inside one active transaction."""

    def __init__(
        self,
        *,
        db: object,
        settings: Settings,
        resolver: CanonicalProjectionResolverPort | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._resolver = resolver or ServingEvidenceResolver(settings=settings)

    def project_search(
        self,
        slots: tuple[EvidenceSlot, ...],
        *,
        scope: SecurityScope,
        fence: ProjectionFence,
    ) -> V1EvidenceProjection:
        self._require_transaction()
        if not fence.valid(require_hidden=True):
            return _empty_search_projection(self._settings)
        try:
            fresh = self._resolve_slots(slots, scope=scope)
            if fresh is None:
                return _empty_search_projection(self._settings)
            results: list[dict[str, object]] = []
            result_hmacs: list[str] = []
            for slot, row in zip(slots, fresh, strict=True):
                citation, citation_hmac = _citation(slot, row, settings=self._settings)
                result = {
                    'id': row.public_result_id,
                    'source_id': row.evidence.public_source_id,
                    'text': row.evidence.model_content,
                    'source_snippet': row.source_snippet,
                    'source_url': row.source_url,
                    'source_type': row.evidence.public_source_type,
                    'permission_level': row.evidence.effective_permission,
                    'relevance_score': slot.relevance_score,
                    'matched_terms': list(slot.matched_terms),
                    'citation': citation,
                    'parser_status': row.parser_status,
                    'parser_status_reason': row.parser_status_reason,
                    'revision_id': row.revision_id,
                }
                result_hmacs.append(
                    _search_result_hmac(
                        row=row,
                        result=result,
                        citation_hmac=citation_hmac,
                        settings=self._settings,
                    )
                )
                results.append(result)
            projection_hmac = build_v1_search_result_set_projection_hmac(
                tuple(result_hmacs), settings=self._settings
            )
            return V1EvidenceProjection((), (), (), (), tuple(results), projection_hmac)
        except (TypeError, UnicodeError, ValueError):
            return _empty_search_projection(self._settings)

    def project_selected(
        self,
        slots: tuple[EvidenceSlot, ...],
        *,
        selected_slot_ids: tuple[EvidenceSlotId, ...],
        scope: SecurityScope,
        fence: ProjectionFence,
    ) -> V1EvidenceProjection:
        self._require_transaction()
        if not fence.valid(require_hidden=True) or not _selected_subset(
            slots, selected_slot_ids
        ):
            return _empty_selected_projection(self._settings)
        try:
            fresh = self._resolve_slots(slots, scope=scope)
            if fresh is None:
                return _empty_selected_projection(self._settings)
            selected = set(selected_slot_ids)
            projected = [
                (slot, row)
                for slot, row in zip(slots, fresh, strict=True)
                if slot.slot_id in selected
            ]
            projected.sort(key=lambda value: selected_slot_ids.index(value[0].slot_id))
            citations: list[dict[str, object]] = []
            citation_hmacs: list[str] = []
            for slot, row in projected:
                citation, citation_hmac = _citation(slot, row, settings=self._settings)
                citations.append(citation)
                citation_hmacs.append(citation_hmac)
            source_ids = tuple(row.evidence.public_source_id for _, row in projected)
            source_links = tuple(row.source_url for _, row in projected)
            source_snippets = tuple(row.source_snippet for _, row in projected)
            projection_hmac = build_v1_selected_evidence_projection_hmac(
                citation_hmacs=tuple(citation_hmacs),
                source_ids=source_ids,
                source_links=source_links,
                source_snippets=source_snippets,
                settings=self._settings,
            )
            return V1EvidenceProjection(
                tuple(citations),
                source_ids,
                source_links,
                source_snippets,
                (),
                projection_hmac,
            )
        except (TypeError, UnicodeError, ValueError):
            return _empty_selected_projection(self._settings)

    def prepare_model_influence(
        self,
        slots: tuple[EvidenceSlot, ...],
        *,
        scope: SecurityScope,
        prepared_corpus_generation: int,
        prepared_index_generation: int | None,
        rendered_input_hmac: str,
    ) -> tuple[PreparedModelInfluenceObservation, ...]:
        self._require_transaction()
        try:
            fresh = self._resolve_slots(slots, scope=scope)
        except (TypeError, UnicodeError, ValueError):
            return ()
        if fresh is None:
            return ()
        entries = tuple(
            (
                row.identity,
                slot.slot_id,
                slot.support_mode,
                row.approval_provenance_hmac,
                row.evidence_link_set_hmac,
            )
            for slot, row in zip(slots, fresh, strict=True)
        )
        observation_hmac = build_prepared_model_influence_observation_hmac(
            entries=entries,
            prepared_corpus_generation=prepared_corpus_generation,
            prepared_index_generation=prepared_index_generation,
            rendered_input_hmac=rendered_input_hmac,
            settings=self._settings,
        )
        return tuple(
            PreparedModelInfluenceObservation(
                slot_id=slot.slot_id,
                support_mode=slot.support_mode,
                lookup_identity=row.identity,
                effective_permission=row.evidence.effective_permission,
                serving_identity_hmac=row.evidence.serving_identity_hmac,
                serving_version_fingerprint=row.evidence.serving_version_fingerprint,
                model_content_hmac=row.evidence.model_content_hmac,
                canonical_citation_projection_hmac=(
                    row.evidence.canonical_citation_projection_hmac
                ),
                approval_provenance_hmac=row.approval_provenance_hmac,
                evidence_link_set_hmac=row.evidence_link_set_hmac,
                observation_hmac=observation_hmac,
            )
            for slot, row in zip(slots, fresh, strict=True)
        )

    def finalize_model_influence_dependencies(
        self,
        observations: tuple[PreparedModelInfluenceObservation, ...],
        selected_slot_ids: tuple[EvidenceSlotId, ...],
        *,
        scope: SecurityScope,
        fence: ProjectionFence,
    ) -> tuple[ModelInfluenceDependencySnapshot, ...]:
        self._require_transaction()
        if not fence.valid(require_hidden=True) or not _selected_observation_subset(
            observations, selected_slot_ids
        ):
            return ()
        selected = set(selected_slot_ids)
        dependencies: list[ModelInfluenceDependencySnapshot] = []
        try:
            for observation in observations:
                row = self._resolver.resolve_projection_candidate(
                    db=self._db, identity=observation.lookup_identity, scope=scope
                )
                if row is None or not _observation_matches(observation, row):
                    return ()
                role: DependencyRole = (
                    'selected_citation'
                    if observation.slot_id in selected
                    else 'unselected_model_influence'
                )
                dependencies.append(
                    ModelInfluenceDependencySnapshot(
                        observation=observation,
                        dependency_role=role,
                        fresh_lookup_identity=row.identity,
                        dependency_hmac=build_model_influence_dependency_hmac(
                            row=row,
                            slot_id=observation.slot_id,
                            support_mode=observation.support_mode,
                            dependency_role=role,
                            settings=self._settings,
                        ),
                    )
                )
        except (TypeError, UnicodeError, ValueError):
            return ()
        return tuple(dependencies)

    def _resolve_slots(
        self, slots: tuple[EvidenceSlot, ...], *, scope: SecurityScope
    ) -> tuple[CanonicalServingProjection, ...] | None:
        if not _valid_slots(slots):
            return None
        rows: list[CanonicalServingProjection] = []
        for slot in slots:
            identity = _identity_from_evidence(slot.evidence)
            row = self._resolver.resolve_projection_candidate(
                db=self._db, identity=identity, scope=scope
            )
            if row is None or row.identity != identity:
                return None
            _validate_row(slot, row, settings=self._settings)
            rows.append(row)
        return tuple(rows)

    def _require_transaction(self) -> None:
        in_transaction = getattr(self._db, 'in_transaction', None)
        if not callable(in_transaction) or in_transaction() is not True:
            raise RuntimeError('canonical projection requires an active transaction')


def build_v1_citation_projection_hmac(
    *,
    row: CanonicalServingProjection,
    relevance_score: float,
    matched_terms: tuple[str, ...],
    settings: Settings,
) -> str:
    _validate_public_fields(row)
    return _fp(
        {
            'source_id_bytes': exact_utf8_bytes(row.evidence.public_source_id),
            'source_url_bytes': exact_utf8_bytes(row.source_url),
            'source_type_bytes': _optional_bytes(row.evidence.public_source_type),
            'permission_level': _permission(row.evidence.effective_permission),
            'source_snippet_bytes': exact_utf8_bytes(row.source_snippet),
            'relevance_score_binary64_be_hex': _finite_binary64_be_hex(relevance_score),
            'matched_terms_bytes': _matched_terms_bytes(matched_terms),
        },
        schema='rag-v1-evidence-projection:citation:v1',
        policy='rag-v1-evidence-projection:v1',
        settings=settings,
    )


def build_v1_selected_evidence_projection_hmac(
    *,
    citation_hmacs: tuple[str, ...],
    source_ids: tuple[str, ...],
    source_links: tuple[str, ...],
    source_snippets: tuple[str, ...],
    settings: Settings,
) -> str:
    lengths = {
        len(citation_hmacs),
        len(source_ids),
        len(source_links),
        len(source_snippets),
    }
    if len(lengths) != 1:
        raise ValueError('selected projection arrays do not correspond')
    for value in citation_hmacs:
        require_lower_hex_64(value)
    return _fp(
        {
            'citations': [
                {'ordinal': ordinal, 'citation_projection_hmac': value}
                for ordinal, value in enumerate(citation_hmacs)
            ],
            'source_ids_bytes': [
                exact_utf8_bytes(require_exact_nonblank(value)) for value in source_ids
            ],
            'source_links_bytes': [
                exact_utf8_bytes(RagPublicCitationUrlValidator().validate(value))
                for value in source_links
            ],
            'source_snippets_bytes': [
                exact_utf8_bytes(require_exact_nonblank(value))
                for value in source_snippets
            ],
        },
        schema='rag-v1-evidence-projection:selected-set:v1',
        policy='rag-v1-evidence-projection:v1',
        settings=settings,
    )


def build_v1_search_result_set_projection_hmac(
    result_hmacs: tuple[str, ...], *, settings: Settings
) -> str:
    for value in result_hmacs:
        require_lower_hex_64(value)
    return _fp(
        {
            'results': [
                {'ordinal': ordinal, 'search_result_projection_hmac': value}
                for ordinal, value in enumerate(result_hmacs)
            ]
        },
        schema='rag-v1-evidence-projection:search-set:v1',
        policy='rag-v1-evidence-projection:v1',
        settings=settings,
    )


def build_hidden_membership_hmac(
    *,
    denied_known_member_identity_hmacs: tuple[str, ...],
    public_hidden_count: int,
    capped: bool,
    top_candidate_window_hmac: str,
    settings: Settings,
) -> str:
    if (
        type(capped) is not bool
        or type(public_hidden_count) is not int
        or not 0 <= public_hidden_count <= 20
    ):
        raise ValueError('hidden membership count is invalid')
    if public_hidden_count != len(denied_known_member_identity_hmacs) and not (
        capped
        and public_hidden_count == 20
        and len(denied_known_member_identity_hmacs) == 20
    ):
        raise ValueError('hidden membership does not match public count')
    require_lower_hex_64(top_candidate_window_hmac)
    for value in denied_known_member_identity_hmacs:
        require_lower_hex_64(value)
    return _fp(
        {
            'capped': capped,
            'denied_known_member_identity_hmacs': list(
                denied_known_member_identity_hmacs
            ),
            'public_hidden_count': public_hidden_count,
            'top_candidate_window_hmac': top_candidate_window_hmac,
        },
        schema='rag-hidden-membership:v1',
        policy='rag-retrieval-bounds:v1',
        settings=settings,
    )


def build_prepared_model_influence_observation_hmac(
    *,
    entries: tuple[
        tuple[
            ServingEvidenceIdentity, EvidenceSlotId, SupportMode, str | None, str | None
        ],
        ...,
    ],
    prepared_corpus_generation: int,
    prepared_index_generation: int | None,
    rendered_input_hmac: str,
    settings: Settings,
) -> str:
    require_nonnegative_int(prepared_corpus_generation)
    if prepared_index_generation is not None:
        require_nonnegative_int(prepared_index_generation)
    require_lower_hex_64(rendered_input_hmac)
    payload_entries = []
    for ordinal, (
        identity,
        slot_id,
        support_mode,
        approval_hmac,
        link_hmac,
    ) in enumerate(entries):
        _validate_slot_id(slot_id)
        _validate_support_mode(support_mode)
        for value in (approval_hmac, link_hmac):
            if value is not None:
                require_lower_hex_64(value)
        payload_entries.append(
            {
                'approval_provenance_hmac': approval_hmac,
                'canonical_citation_projection_hmac': require_lower_hex_64(
                    identity.canonical_citation_projection_hmac
                ),
                'effective_permission': _permission(identity.effective_permission),
                'evidence_link_set_hmac': link_hmac,
                'model_content_hmac': require_lower_hex_64(identity.model_content_hmac),
                'ordinal': ordinal,
                'serving_identity_hmac': _serving_identity_hmac(
                    identity, settings=settings
                ),
                'serving_version_fingerprint': require_lower_hex_64(
                    identity.serving_version_fingerprint
                ),
                'slot_id': slot_id,
                'support_mode': support_mode,
            }
        )
    return _fp(
        {
            'entries': payload_entries,
            'prepared_corpus_generation': prepared_corpus_generation,
            'prepared_index_generation': prepared_index_generation,
            'rendered_input_hmac': rendered_input_hmac,
        },
        schema='rag-prepared-model-influence-observation:v1',
        policy='rag-answer:v2',
        settings=settings,
    )


def build_model_influence_dependency_hmac(
    *,
    row: CanonicalServingProjection,
    slot_id: EvidenceSlotId,
    support_mode: SupportMode,
    dependency_role: DependencyRole,
    settings: Settings,
) -> str:
    _validate_slot_id(slot_id)
    _validate_support_mode(support_mode)
    if dependency_role not in {'selected_citation', 'unselected_model_influence'}:
        raise ValueError('model influence role is invalid')
    return _fp(
        {
            'approval_provenance_hmac': row.approval_provenance_hmac,
            'canonical_citation_projection_hmac': require_lower_hex_64(
                row.evidence.canonical_citation_projection_hmac
            ),
            'dependency_role': dependency_role,
            'effective_permission': _permission(row.evidence.effective_permission),
            'evidence_link_set_hmac': row.evidence_link_set_hmac,
            'model_content_hmac': require_lower_hex_64(row.evidence.model_content_hmac),
            'serving_identity_hmac': require_lower_hex_64(
                row.evidence.serving_identity_hmac
            ),
            'serving_version_fingerprint': require_lower_hex_64(
                row.evidence.serving_version_fingerprint
            ),
            'slot_id': slot_id,
            'support_mode': support_mode,
        },
        schema='rag-model-influence-dependency:v1',
        policy='rag-serving-evidence:v1',
        settings=settings,
    )


def build_model_influence_set_hmac(
    *,
    dependency_hmacs: tuple[str, ...],
    strictest_permission: PermissionLevel,
    settings: Settings,
) -> str:
    if not 1 <= len(dependency_hmacs) <= 8:
        raise ValueError('model influence set must contain one to eight entries')
    for value in dependency_hmacs:
        require_lower_hex_64(value)
    return _fp(
        {
            'dependencies': [
                {'dependency_hmac': value, 'ordinal': ordinal}
                for ordinal, value in enumerate(dependency_hmacs)
            ],
            'strictest_permission': _permission(strictest_permission),
        },
        schema='rag-model-influence-set:v1',
        policy='rag-serving-evidence:v1',
        settings=settings,
    )


def _identity_from_evidence(evidence: ServingEvidence) -> ServingEvidenceIdentity:
    return ServingEvidenceIdentity(
        serving_document_id=evidence.serving_document_id,
        serving_kind=evidence.serving_kind,
        public_source_id=evidence.public_source_id,
        public_source_type=evidence.public_source_type,
        effective_permission=evidence.effective_permission,
        model_content_hmac=evidence.model_content_hmac,
        canonical_citation_projection_hmac=evidence.canonical_citation_projection_hmac,
        serving_version_fingerprint=evidence.serving_version_fingerprint,
        version_envelope=evidence.version_envelope,
    )


def _validate_row(
    slot: EvidenceSlot, row: CanonicalServingProjection, *, settings: Settings
) -> None:
    evidence = row.evidence
    identity = row.identity
    if (
        evidence.support_mode != slot.support_mode
        or _identity_from_evidence(evidence) != identity
        or evidence.serving_identity_hmac
        != build_serving_identity_hmac(
            serving_document_id=evidence.serving_document_id,
            serving_kind=evidence.serving_kind,
            settings=settings,
        )
        or evidence.model_content_hmac
        != build_model_content_hmac(
            serving_kind=evidence.serving_kind,
            model_content=evidence.model_content,
            settings=settings,
        )
        or evidence.serving_version_fingerprint
        != build_serving_version_fingerprint(
            evidence.version_envelope, settings=settings
        )
        or (
            isinstance(evidence.provenance, RawChunkProvenance)
            and (
                row.approval_provenance_hmac is not None
                or row.evidence_link_set_hmac is not None
            )
        )
        or (
            isinstance(evidence.provenance, ExplicitApprovalProvenance)
            and (
                row.approval_provenance_hmac is None
                or row.evidence_link_set_hmac is None
            )
        )
        or (
            isinstance(evidence.provenance, LegacyHumanProvenance)
            and (
                row.approval_provenance_hmac is None
                or row.evidence_link_set_hmac is not None
            )
        )
    ):
        raise ValueError('canonical support mode drifted')
    _validate_public_fields(row)
    if (
        build_canonical_citation_projection_hmac(
            public_source_id=row.evidence.public_source_id,
            public_source_type=row.evidence.public_source_type,
            source_url=row.source_url,
            source_snippet=row.source_snippet,
            effective_permission=row.evidence.effective_permission,
            settings=settings,
        )
        != row.evidence.canonical_citation_projection_hmac
    ):
        raise ValueError('canonical citation projection drifted')
    _finite_binary64_be_hex(slot.relevance_score)
    _matched_terms_bytes(slot.matched_terms)


def _validate_public_fields(row: CanonicalServingProjection) -> None:
    require_positive_int(row.public_result_id)
    require_exact_nonblank(row.evidence.public_source_id)
    if row.evidence.public_source_type is not None:
        require_exact_nonblank(row.evidence.public_source_type)
    RagPublicCitationUrlValidator().validate(row.source_url)
    require_exact_nonblank(row.source_snippet)
    require_exact_nonblank(row.evidence.model_content)
    _permission(row.evidence.effective_permission)
    for value in (row.parser_status, row.parser_status_reason, row.revision_id):
        if value is not None:
            exact_utf8_bytes(value)
    for value in (row.approval_provenance_hmac, row.evidence_link_set_hmac):
        if value is not None:
            require_lower_hex_64(value)


def _citation(
    slot: EvidenceSlot, row: CanonicalServingProjection, *, settings: Settings
) -> tuple[dict[str, object], str]:
    citation = {
        'source_id': row.evidence.public_source_id,
        'source_url': row.source_url,
        'source_type': row.evidence.public_source_type,
        'permission_level': row.evidence.effective_permission,
        'source_snippet': row.source_snippet,
        'relevance_score': slot.relevance_score,
        'matched_terms': list(slot.matched_terms),
    }
    return citation, build_v1_citation_projection_hmac(
        row=row,
        relevance_score=slot.relevance_score,
        matched_terms=slot.matched_terms,
        settings=settings,
    )


def _search_result_hmac(
    *,
    row: CanonicalServingProjection,
    result: dict[str, object],
    citation_hmac: str,
    settings: Settings,
) -> str:
    return _fp(
        {
            'id': require_positive_int(result['id']),
            'source_id_bytes': exact_utf8_bytes(cast(str, result['source_id'])),
            'text_bytes': exact_utf8_bytes(cast(str, result['text'])),
            'source_snippet_bytes': exact_utf8_bytes(
                cast(str, result['source_snippet'])
            ),
            'source_url_bytes': exact_utf8_bytes(cast(str, result['source_url'])),
            'source_type_bytes': _optional_bytes(
                cast(str | None, result['source_type'])
            ),
            'permission_level': _permission(result['permission_level']),
            'relevance_score_binary64_be_hex': _finite_binary64_be_hex(
                result['relevance_score']
            ),
            'matched_terms_bytes': _matched_terms_bytes(
                tuple(cast(list[str], result['matched_terms']))
            ),
            'citation_projection_hmac': require_lower_hex_64(citation_hmac),
            'parser_status_bytes': _optional_bytes(row.parser_status),
            'parser_status_reason_bytes': _optional_bytes(row.parser_status_reason),
            'revision_id_bytes': _optional_bytes(row.revision_id),
        },
        schema='rag-v1-evidence-projection:search-result:v1',
        policy='rag-v1-evidence-projection:v1',
        settings=settings,
    )


def _valid_slots(slots: object) -> bool:
    return bool(
        type(slots) is tuple
        and len(slots) <= 8
        and all(
            type(slot) is EvidenceSlot and slot.slot_id == _SLOT_IDS[index]
            for index, slot in enumerate(cast(tuple[EvidenceSlot, ...], slots))
        )
    )


def _selected_subset(
    slots: tuple[EvidenceSlot, ...], selected: tuple[EvidenceSlotId, ...]
) -> bool:
    if (
        type(selected) is not tuple
        or not selected
        or len(set(selected)) != len(selected)
    ):
        return False
    available = {slot.slot_id for slot in slots}
    return all(value in available for value in selected)


def _selected_observation_subset(
    observations: tuple[PreparedModelInfluenceObservation, ...],
    selected: tuple[EvidenceSlotId, ...],
) -> bool:
    if type(observations) is not tuple or not observations:
        return False
    slots = tuple(value.slot_id for value in observations)
    return (
        slots == _SLOT_IDS[: len(slots)]
        and bool(selected)
        and len(set(selected)) == len(selected)
        and set(selected) <= set(slots)
    )


def _observation_matches(
    observation: PreparedModelInfluenceObservation, row: CanonicalServingProjection
) -> bool:
    return bool(
        observation.lookup_identity == row.identity
        and observation.effective_permission == row.evidence.effective_permission
        and observation.serving_identity_hmac == row.evidence.serving_identity_hmac
        and observation.serving_version_fingerprint
        == row.evidence.serving_version_fingerprint
        and observation.model_content_hmac == row.evidence.model_content_hmac
        and observation.canonical_citation_projection_hmac
        == row.evidence.canonical_citation_projection_hmac
        and observation.approval_provenance_hmac == row.approval_provenance_hmac
        and observation.evidence_link_set_hmac == row.evidence_link_set_hmac
    )


def _empty_selected_projection(settings: Settings) -> V1EvidenceProjection:
    return V1EvidenceProjection(
        (),
        (),
        (),
        (),
        (),
        build_v1_selected_evidence_projection_hmac(
            citation_hmacs=(),
            source_ids=(),
            source_links=(),
            source_snippets=(),
            settings=settings,
        ),
    )


def _empty_search_projection(settings: Settings) -> V1EvidenceProjection:
    return V1EvidenceProjection(
        (),
        (),
        (),
        (),
        (),
        build_v1_search_result_set_projection_hmac((), settings=settings),
    )


def _finite_binary64_be_hex(value: object) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('projection score must be a finite binary64 float')
    return struct.pack('>d', value).hex()


def _matched_terms_bytes(values: object) -> list[dict[str, int | str]]:
    if type(values) is not tuple:
        raise ValueError('matched terms must be immutable')
    return [exact_utf8_bytes(require_exact_nonblank(value)) for value in values]


def _optional_bytes(value: str | None) -> dict[str, int | str] | None:
    return exact_utf8_bytes(value) if value is not None else None


def _permission(value: object) -> PermissionLevel:
    permission = known_permission(value)
    if permission is None:
        raise ValueError('projection permission is invalid')
    return permission


def _validate_slot_id(value: object) -> EvidenceSlotId:
    if value not in _SLOT_IDS:
        raise ValueError('evidence slot id is invalid')
    return cast(EvidenceSlotId, value)


def _validate_support_mode(value: object) -> SupportMode:
    if value not in {'trusted_fact', 'source_observation'}:
        raise ValueError('support mode is invalid')
    return cast(SupportMode, value)


def _serving_identity_hmac(
    identity: ServingEvidenceIdentity, *, settings: Settings
) -> str:
    return build_serving_identity_hmac(
        serving_document_id=identity.serving_document_id,
        serving_kind=identity.serving_kind,
        settings=settings,
    )


def _fp(value: object, *, schema: str, policy: str, settings: Settings) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        value,
        secret=secret,
        schema_version=schema,
        policy_version=policy,
    )
