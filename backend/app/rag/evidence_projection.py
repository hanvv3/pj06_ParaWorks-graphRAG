from __future__ import annotations

import math
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from sqlalchemy.exc import SQLAlchemyError

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
from backend.app.knowledge.trusted_serving_eligibility import (
    canonical_knowledge_document_id,
)
from backend.app.rag.retrieval import EvidenceSlot, EvidenceSlotId
from backend.app.rag.serving_contracts import (
    CanonicalServingProjection,
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    PermissionLevel,
    RawChunkProvenance,
    RawServingVersionEnvelope,
    SelectedCitationChild,
    ServingEvidence,
    ServingEvidenceIdentity,
    SupportMode,
    TrustedEvidenceLinkIdentity,
    TrustedServingVersionEnvelope,
    build_approval_provenance_hmac,
    build_canonical_citation_projection_hmac,
    build_evidence_link_set_hmac,
    build_model_content_hmac,
    build_selected_citation_child_hmac,
    build_serving_identity_hmac,
    build_serving_version_fingerprint,
    known_permission,
    require_exact_nonblank,
    require_lower_hex_64,
    require_nonnegative_int,
    require_positive_int,
    strictest_permission,
)
from backend.app.rag.trusted_evidence import ServingEvidenceResolver

DependencyRole = Literal['selected_citation', 'unselected_model_influence']
_SLOT_IDS: tuple[EvidenceSlotId, ...] = ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8')
_RAW_PUBLIC_SOURCE_TYPES = {'gmail', 'gmail_attachment', 'drive', 'calendar'}


class CanonicalProjectionResolverPort(Protocol):
    def resolve_projection_candidate_strict(
        self, *, db: object, identity: ServingEvidenceIdentity, scope: SecurityScope
    ) -> CanonicalServingProjection | None: ...


class CanonicalProjectionInfrastructureError(RuntimeError):
    """Canonical projection could not establish a trustworthy read."""


class CanonicalProjectionTransactionError(CanonicalProjectionInfrastructureError):
    """Projection lost its exact active transaction fence."""


class FrozenDict(tuple, Mapping[str, object]):
    """Attribute-free immutable mapping stored only in tuple payload bytes."""

    __slots__ = ()

    def __new__(cls, values: Mapping[str, object]) -> FrozenDict:
        return tuple.__new__(cls, tuple(values.items()))

    def __getitem__(self, key: str) -> object:
        for current_key, value in tuple.__iter__(self):
            if current_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in tuple.__iter__(self))

    def __len__(self) -> int:
        return tuple.__len__(self)

    def __eq__(self, other: object) -> bool:
        return bool(
            isinstance(other, Mapping)
            and dict(self.items()) == dict(other.items())
        )

    __hash__ = None

    @staticmethod
    def _blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError('frozen projection record cannot be mutated')

    __setitem__ = _blocked
    __delitem__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked
    __ior__ = _blocked

    def copy(self) -> FrozenDict:
        return self

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenDict:
        del memo
        return self

    def __reduce__(self):
        return (FrozenDict, (dict(self.items()),))

    def __or__(self, other: object) -> FrozenDict:
        if not isinstance(other, Mapping):
            return NotImplemented
        return FrozenDict(dict(self) | other)

    def __ror__(self, other: object) -> FrozenDict:
        if not isinstance(other, Mapping):
            return NotImplemented
        return FrozenDict(dict(other) | dict(self))


@dataclass(frozen=True, slots=True)
class V1EvidenceProjection:
    citations: tuple[FrozenDict, ...]
    source_ids: tuple[str, ...]
    source_links: tuple[str, ...]
    source_snippets: tuple[str, ...]
    search_results: tuple[FrozenDict, ...]
    projection_hmac: str


@dataclass(frozen=True, slots=True)
class PreparedModelInfluenceObservation:
    ordinal: int
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
class PreparedModelInfluenceSet:
    observations: tuple[PreparedModelInfluenceObservation, ...]
    prepared_corpus_generation: int
    prepared_index_generation: int | None
    prepared_readiness_hmac: str | None
    rendered_input_hmac: str
    aggregate_observation_hmac: str

    def __iter__(self):
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> PreparedModelInfluenceObservation:
        return self.observations[index]


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


@dataclass(frozen=True, slots=True)
class HiddenMembershipSnapshot:
    actual_hidden_count: int
    public_hidden_count: int
    capped: bool
    membership_hmac: str


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
        self._transaction_token: object | None = None

    def project_search(
        self,
        slots: tuple[EvidenceSlot, ...],
        *,
        scope: SecurityScope,
        fence: ProjectionFence,
    ) -> V1EvidenceProjection:
        self._begin_transaction_fence()
        if len(slots) > 5 or not fence.valid(require_hidden=True):
            return self._checked_projection(_empty_search_projection(self._settings))
        try:
            fresh = self._resolve_slots(slots, scope=scope)
            if fresh is None:
                return self._checked_projection(_empty_search_projection(self._settings))
            results: list[FrozenDict] = []
            result_hmacs: list[str] = []
            for slot, row in zip(slots, fresh, strict=True):
                citation, citation_hmac = _citation(slot, row, settings=self._settings)
                result = _deep_freeze({
                    'id': row.public_result_id,
                    'source_id': row.evidence.public_source_id,
                    'text': row.evidence.model_content,
                    'source_snippet': row.source_snippet,
                    'source_url': row.source_url,
                    'source_type': row.evidence.public_source_type,
                    'permission_level': row.evidence.effective_permission,
                    'relevance_score': slot.relevance_score,
                    'matched_terms': slot.matched_terms,
                    'citation': citation,
                    'parser_status': row.parser_status,
                    'parser_status_reason': row.parser_status_reason,
                    'revision_id': row.revision_id,
                })
                assert type(result) is FrozenDict
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
            return self._checked_projection(
                V1EvidenceProjection((), (), (), (), tuple(results), projection_hmac)
            )
        except (TypeError, UnicodeError, ValueError):
            return self._checked_projection(_empty_search_projection(self._settings))

    def project_selected(
        self,
        slots: tuple[EvidenceSlot, ...],
        *,
        selected_slot_ids: tuple[EvidenceSlotId, ...],
        scope: SecurityScope,
        fence: ProjectionFence,
    ) -> V1EvidenceProjection:
        self._begin_transaction_fence()
        if not fence.valid(require_hidden=True) or not _selected_subset(
            slots, selected_slot_ids
        ):
            return self._checked_projection(_empty_selected_projection(self._settings))
        try:
            fresh = self._resolve_slots(slots, scope=scope)
            if fresh is None:
                return self._checked_projection(
                    _empty_selected_projection(self._settings)
                )
            selected = set(selected_slot_ids)
            projected = [
                (slot, row)
                for slot, row in zip(slots, fresh, strict=True)
                if slot.slot_id in selected
            ]
            projected.sort(key=lambda value: selected_slot_ids.index(value[0].slot_id))
            citations: list[FrozenDict] = []
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
            return self._checked_projection(
                V1EvidenceProjection(
                    tuple(citations),
                    source_ids,
                    source_links,
                    source_snippets,
                    (),
                    projection_hmac,
                )
            )
        except (TypeError, UnicodeError, ValueError):
            return self._checked_projection(_empty_selected_projection(self._settings))

    def prepare_model_influence(
        self,
        slots: tuple[EvidenceSlot, ...],
        *,
        scope: SecurityScope,
        prepared_corpus_generation: int,
        prepared_index_generation: int | None,
        prepared_readiness_hmac: str | None,
        rendered_input_hmac: str,
    ) -> PreparedModelInfluenceSet:
        self._begin_transaction_fence()
        if type(slots) is not tuple or not 1 <= len(slots) <= 8:
            raise ValueError('prepared influence requires one to eight observations')
        try:
            fresh = self._resolve_slots(slots, scope=scope)
        except (TypeError, UnicodeError, ValueError):
            return self._checked_prepared(_empty_prepared_influence_set())
        if fresh is None:
            return self._checked_prepared(_empty_prepared_influence_set())
        observations = tuple(
            PreparedModelInfluenceObservation(
                ordinal=ordinal,
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
                observation_hmac=_prepared_observation_child_hmac(
                    ordinal=ordinal,
                    slot_id=slot.slot_id,
                    support_mode=slot.support_mode,
                    row=row,
                    settings=self._settings,
                ),
            )
            for ordinal, (slot, row) in enumerate(zip(slots, fresh, strict=True))
        )
        aggregate = _build_prepared_model_influence_set_hmac_v2(
            observations=observations,
            prepared_corpus_generation=prepared_corpus_generation,
            prepared_index_generation=prepared_index_generation,
            prepared_readiness_hmac=prepared_readiness_hmac,
            rendered_input_hmac=rendered_input_hmac,
            settings=self._settings,
        )
        return self._checked_prepared(
            PreparedModelInfluenceSet(
                observations=observations,
                prepared_corpus_generation=prepared_corpus_generation,
                prepared_index_generation=prepared_index_generation,
                prepared_readiness_hmac=prepared_readiness_hmac,
                rendered_input_hmac=rendered_input_hmac,
                aggregate_observation_hmac=aggregate,
            )
        )

    def finalize_model_influence_dependencies(
        self,
        prepared: PreparedModelInfluenceSet,
        selected_slot_ids: tuple[EvidenceSlotId, ...],
        *,
        scope: SecurityScope,
        fence: ProjectionFence,
    ) -> tuple[ModelInfluenceDependencySnapshot, ...]:
        self._begin_transaction_fence()
        if type(prepared) is not PreparedModelInfluenceSet or not fence.valid(
            require_hidden=True
        ):
            return self._checked_dependencies(())
        observations = prepared.observations
        if not _valid_prepared_influence_set(
            prepared, fence=fence, settings=self._settings
        ):
            return self._checked_dependencies(())
        dependencies: list[ModelInfluenceDependencySnapshot] = []
        fresh_rows: list[CanonicalServingProjection] = []
        try:
            for observation in observations:
                row = self._strict_resolve(
                    db=self._db, identity=observation.lookup_identity, scope=scope
                )
                if row is not None:
                    _validate_row(
                        EvidenceSlot(
                            slot_id=observation.slot_id,
                            support_mode=observation.support_mode,
                            evidence=row.evidence,
                            relevance_score=0.0,
                            matched_terms=(),
                        ),
                        row,
                        settings=self._settings,
                    )
                if (
                    row is None
                    or not _observation_matches(observation, row)
                    or observation.observation_hmac
                    != _prepared_observation_child_hmac(
                        ordinal=observation.ordinal,
                        slot_id=observation.slot_id,
                        support_mode=observation.support_mode,
                        row=row,
                        settings=self._settings,
                    )
                ):
                    return self._checked_dependencies(())
                fresh_rows.append(row)
            if not _selected_observation_subset(observations, selected_slot_ids):
                return self._checked_dependencies(())
            selected = set(selected_slot_ids)
            for observation, row in zip(observations, fresh_rows, strict=True):
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
            return self._checked_dependencies(())
        return self._checked_dependencies(tuple(dependencies))

    def _resolve_slots(
        self, slots: tuple[EvidenceSlot, ...], *, scope: SecurityScope
    ) -> tuple[CanonicalServingProjection, ...] | None:
        if not _valid_slots(slots):
            return None
        rows: list[CanonicalServingProjection] = []
        for slot in slots:
            identity = _identity_from_evidence(slot.evidence)
            row = self._strict_resolve(
                db=self._db, identity=identity, scope=scope
            )
            if row is None:
                return None
            _validate_row(slot, row, settings=self._settings)
            if row.identity != identity:
                return None
            self._require_transaction()
            rows.append(row)
        self._require_transaction()
        return tuple(rows)

    def _strict_resolve(
        self,
        *,
        db: object,
        identity: ServingEvidenceIdentity,
        scope: SecurityScope,
    ) -> CanonicalServingProjection | None:
        self._require_transaction()
        try:
            row = self._resolver.resolve_projection_candidate_strict(
                db=db, identity=identity, scope=scope
            )
        except (ConnectionError, OSError, SQLAlchemyError) as exc:
            raise CanonicalProjectionInfrastructureError(
                'canonical projection read failed'
            ) from exc
        self._require_transaction()
        return row

    def _checked_projection(
        self, projection: V1EvidenceProjection
    ) -> V1EvidenceProjection:
        self._require_transaction()
        return projection

    def _checked_prepared(
        self, prepared: PreparedModelInfluenceSet
    ) -> PreparedModelInfluenceSet:
        self._require_transaction()
        return prepared

    def _checked_dependencies(
        self, dependencies: tuple[ModelInfluenceDependencySnapshot, ...]
    ) -> tuple[ModelInfluenceDependencySnapshot, ...]:
        self._require_transaction()
        return dependencies

    def _require_transaction(self) -> None:
        in_transaction = getattr(self._db, 'in_transaction', None)
        if not callable(in_transaction) or in_transaction() is not True:
            raise CanonicalProjectionTransactionError(
                'canonical projection requires an active transaction'
            )
        get_transaction = getattr(self._db, 'get_transaction', None)
        if callable(get_transaction):
            current_token = get_transaction()
            if (
                current_token is None
                or self._transaction_token is None
                or current_token is not self._transaction_token
            ):
                raise CanonicalProjectionTransactionError(
                    'canonical projection transaction identity changed'
                )

    def _begin_transaction_fence(self) -> None:
        in_transaction = getattr(self._db, 'in_transaction', None)
        if not callable(in_transaction) or in_transaction() is not True:
            raise CanonicalProjectionTransactionError(
                'canonical projection requires an active transaction'
            )
        get_transaction = getattr(self._db, 'get_transaction', None)
        token = get_transaction() if callable(get_transaction) else self._db
        if token is None:
            raise CanonicalProjectionTransactionError(
                'canonical projection transaction identity is unavailable'
            )
        self._transaction_token = token
        self._require_transaction()


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


def derive_hidden_membership(
    *,
    ordered_hidden_member_identity_hmacs: tuple[str, ...],
    top_candidate_window_hmac: str,
    settings: Settings,
) -> HiddenMembershipSnapshot:
    if (
        type(ordered_hidden_member_identity_hmacs) is not tuple
        or len(ordered_hidden_member_identity_hmacs) > 50
    ):
        raise ValueError('hidden membership window is invalid')
    actual_hidden_count = len(ordered_hidden_member_identity_hmacs)
    public_hidden_count = min(actual_hidden_count, 20)
    capped = actual_hidden_count > 20
    require_lower_hex_64(top_candidate_window_hmac)
    for value in ordered_hidden_member_identity_hmacs:
        require_lower_hex_64(value)
    retained_member_identity_hmacs = ordered_hidden_member_identity_hmacs[:20]
    membership_hmac = _fp(
        {
            'capped': capped,
            'denied_known_member_identity_hmacs': list(
                retained_member_identity_hmacs
            ),
            'public_hidden_count': public_hidden_count,
            'top_candidate_window_hmac': top_candidate_window_hmac,
        },
        schema='rag-hidden-membership:v1',
        policy='rag-retrieval-bounds:v1',
        settings=settings,
    )
    return HiddenMembershipSnapshot(
        actual_hidden_count=actual_hidden_count,
        public_hidden_count=public_hidden_count,
        capped=capped,
        membership_hmac=membership_hmac,
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
    for ordinal, entry in enumerate(entries):
        if type(entry) is not tuple or len(entry) != 5:
            raise ValueError('prepared influence v1 entry is invalid')
        identity, slot_id, support_mode, approval_hmac, link_hmac = entry
        if type(identity) is not ServingEvidenceIdentity:
            raise ValueError('prepared influence identity is invalid')
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
                'model_content_hmac': require_lower_hex_64(
                    identity.model_content_hmac
                ),
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


def _build_prepared_model_influence_set_hmac_v2(
    *,
    observations: tuple[PreparedModelInfluenceObservation, ...],
    prepared_corpus_generation: int,
    prepared_index_generation: int | None,
    prepared_readiness_hmac: str | None,
    rendered_input_hmac: str,
    settings: Settings,
) -> str:
    require_nonnegative_int(prepared_corpus_generation)
    if prepared_index_generation is not None:
        require_nonnegative_int(prepared_index_generation)
        require_lower_hex_64(prepared_readiness_hmac)
    elif prepared_readiness_hmac is not None:
        require_lower_hex_64(prepared_readiness_hmac)
    require_lower_hex_64(rendered_input_hmac)
    payload_entries: list[dict[str, object]] = []
    for ordinal, observation in enumerate(observations):
        if (
            type(observation) is not PreparedModelInfluenceObservation
            or observation.ordinal != ordinal
        ):
            raise ValueError('prepared influence v2 observation is invalid')
        _validate_slot_id(observation.slot_id)
        _validate_support_mode(observation.support_mode)
        payload_entries.append(
            {
                'observation_hmac': require_lower_hex_64(
                    observation.observation_hmac
                ),
                'ordinal': ordinal,
                'slot_id': observation.slot_id,
            }
        )
    return _fp(
        {
            'entries': payload_entries,
            'prepared_corpus_generation': prepared_corpus_generation,
            'prepared_index_generation': prepared_index_generation,
            'prepared_readiness_hmac': prepared_readiness_hmac,
            'rendered_input_hmac': rendered_input_hmac,
        },
        schema='rag-prepared-model-influence-set:v2',
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
    observation = PreparedModelInfluenceObservation(
        ordinal=_SLOT_IDS.index(slot_id),
        slot_id=slot_id,
        support_mode=support_mode,
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
        observation_hmac='0' * 64,
    )
    return _model_influence_dependency_hmac_from_observation(
        observation, dependency_role=dependency_role, settings=settings
    )


def build_model_influence_set_hmac(
    *,
    dependencies: tuple[ModelInfluenceDependencySnapshot, ...],
    settings: Settings,
) -> str:
    if type(dependencies) is not tuple or not 1 <= len(dependencies) <= 8:
        raise ValueError('model influence set must contain one to eight entries')
    dependency_hmacs: list[str] = []
    permissions: list[PermissionLevel] = []
    for ordinal, dependency in enumerate(dependencies):
        if (
            type(dependency) is not ModelInfluenceDependencySnapshot
            or dependency.observation.ordinal != ordinal
            or dependency.observation.slot_id != _SLOT_IDS[ordinal]
        ):
            raise ValueError('model influence dependency order is invalid')
        dependency_hmacs.append(require_lower_hex_64(dependency.dependency_hmac))
        if dependency.dependency_hmac != _model_influence_dependency_hmac_from_observation(
            dependency.observation,
            dependency_role=dependency.dependency_role,
            settings=settings,
        ):
            raise ValueError('model influence dependency HMAC is invalid')
        if (
            type(dependency.fresh_lookup_identity) is not ServingEvidenceIdentity
            or dependency.fresh_lookup_identity != dependency.observation.lookup_identity
        ):
            raise ValueError('model influence fresh identity is invalid')
        permissions.append(_permission(dependency.observation.effective_permission))
    computed_strictest = strictest_permission(tuple(permissions))
    if computed_strictest is None:
        raise ValueError('model influence permission is invalid')
    return _fp(
        {
            'dependencies': [
                {
                    'dependency_hmac': value,
                    'fresh_lookup_identity_hmac': _prepared_lookup_identity_hmac(
                        dependencies[ordinal].fresh_lookup_identity,
                        settings=settings,
                    ),
                    'ordinal': ordinal,
                }
                for ordinal, value in enumerate(dependency_hmacs)
            ],
            'strictest_permission': computed_strictest,
        },
        schema='rag-model-influence-set:v2',
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


def _validate_identity_authority(
    identity: ServingEvidenceIdentity, *, settings: Settings
) -> None:
    if type(identity) is not ServingEvidenceIdentity:
        raise ValueError('canonical serving identity type is invalid')
    for value in (
        identity.serving_document_id,
        identity.public_source_id,
        identity.public_source_type,
    ):
        require_exact_nonblank(value)
    if (
        type(identity.serving_kind) is not str
        or identity.serving_kind not in {'raw_chunk', 'trusted_knowledge'}
    ):
        raise ValueError('canonical serving identity kind is invalid')
    _permission(identity.effective_permission)
    require_lower_hex_64(identity.model_content_hmac)
    require_lower_hex_64(identity.canonical_citation_projection_hmac)
    require_lower_hex_64(identity.serving_version_fingerprint)

    envelope = identity.version_envelope
    if identity.serving_kind == 'raw_chunk':
        if type(envelope) is not RawServingVersionEnvelope:
            raise ValueError('canonical raw identity envelope is invalid')
        _validate_raw_identity_envelope(envelope)
        if (
            identity.serving_document_id != f'chunk:{envelope.document_chunk_id}'
            or identity.serving_document_id != envelope.serving_document_id
            or identity.public_source_id != envelope.public_source_id
            or identity.public_source_type not in _RAW_PUBLIC_SOURCE_TYPES
        ):
            raise ValueError('canonical raw identity mapping is invalid')
    else:
        if type(envelope) is not TrustedServingVersionEnvelope:
            raise ValueError('canonical trusted identity envelope is invalid')
        _validate_trusted_identity_envelope(envelope)
        serving_document_id = canonical_knowledge_document_id(
            envelope.knowledge_type, envelope.knowledge_id
        )
        if (
            identity.serving_document_id != serving_document_id
            or identity.public_source_id != serving_document_id
            or envelope.serving_document_id != serving_document_id
            or identity.public_source_type != envelope.knowledge_type
        ):
            raise ValueError('canonical trusted identity mapping is invalid')

    if (
        identity.effective_permission != envelope.effective_permission
        or identity.model_content_hmac != envelope.model_content_hmac
        or identity.canonical_citation_projection_hmac
        != envelope.canonical_citation_projection_hmac
        or identity.serving_version_fingerprint
        != build_serving_version_fingerprint(envelope, settings=settings)
    ):
        raise ValueError('canonical serving identity HMAC drifted')


def _validate_raw_identity_envelope(envelope: RawServingVersionEnvelope) -> None:
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
    _permission(envelope.effective_permission)
    if envelope.serving_document_id != f'chunk:{envelope.document_chunk_id}':
        raise ValueError('canonical raw envelope document is invalid')


def _validate_trusted_identity_envelope(
    envelope: TrustedServingVersionEnvelope,
) -> None:
    require_exact_nonblank(envelope.serving_document_id)
    if (
        type(envelope.knowledge_type) is not str
        or envelope.knowledge_type
        not in {'decision_record', 'history_event', 'timeline_event', 'todo'}
    ):
        raise ValueError('canonical trusted envelope type is invalid')
    require_positive_int(envelope.knowledge_id)
    require_lower_hex_64(envelope.model_content_hmac)
    require_lower_hex_64(envelope.canonical_citation_projection_hmac)
    _permission(envelope.effective_permission)
    if type(envelope.provenance) not in {
        ExplicitApprovalProvenance,
        LegacyHumanProvenance,
    }:
        raise ValueError('canonical trusted envelope provenance is invalid')
    if envelope.serving_document_id != canonical_knowledge_document_id(
        envelope.knowledge_type, envelope.knowledge_id
    ):
        raise ValueError('canonical trusted envelope document is invalid')


def _validate_row(
    slot: EvidenceSlot, row: CanonicalServingProjection, *, settings: Settings
) -> None:
    evidence = row.evidence
    identity = row.identity
    if (
        type(row) is not CanonicalServingProjection
        or type(evidence) is not ServingEvidence
        or type(identity) is not ServingEvidenceIdentity
    ):
        raise ValueError('canonical projection type is invalid')
    _validate_identity_authority(identity, settings=settings)
    if evidence.serving_kind == 'raw_chunk':
        envelope = evidence.version_envelope
        branch_valid = bool(
            type(evidence.serving_kind) is str
            and evidence.serving_kind == 'raw_chunk'
            and type(evidence.support_mode) is str
            and evidence.support_mode == 'source_observation'
            and type(evidence.public_source_type) is str
            and evidence.public_source_type in _RAW_PUBLIC_SOURCE_TYPES
            and type(envelope) is RawServingVersionEnvelope
            and type(evidence.provenance) is RawChunkProvenance
            and type(evidence.provenance.branch) is str
            and evidence.provenance.branch == 'raw_chunk'
            and evidence.provenance.raw_version is envelope
            and type(envelope.document_chunk_id) is int
            and envelope.document_chunk_id > 0
            and evidence.serving_document_id
            == f'chunk:{envelope.document_chunk_id}'
            and envelope.serving_document_id == evidence.serving_document_id
            and envelope.public_source_id == evidence.public_source_id
            and envelope.model_content_hmac == evidence.model_content_hmac
            and envelope.canonical_citation_projection_hmac
            == evidence.canonical_citation_projection_hmac
            and envelope.effective_permission == evidence.effective_permission
            and row.approval_provenance_hmac is None
            and row.evidence_link_set_hmac is None
            and type(row.evidence_link_hmacs) is tuple
            and not row.evidence_link_hmacs
        )
    elif evidence.serving_kind == 'trusted_knowledge':
        envelope = evidence.version_envelope
        branch_valid = bool(
            type(evidence.serving_kind) is str
            and evidence.serving_kind == 'trusted_knowledge'
            and type(evidence.support_mode) is str
            and evidence.support_mode == 'trusted_fact'
            and type(envelope) is TrustedServingVersionEnvelope
            and type(envelope.knowledge_type) is str
            and type(envelope.knowledge_id) is int
            and envelope.knowledge_id > 0
            and envelope.serving_document_id
            == evidence.public_source_id
            == evidence.serving_document_id
            == canonical_knowledge_document_id(
                envelope.knowledge_type, envelope.knowledge_id
            )
            and evidence.public_source_type == envelope.knowledge_type
            and row.public_result_id == envelope.knowledge_id
            and envelope.model_content_hmac == evidence.model_content_hmac
            and envelope.canonical_citation_projection_hmac
            == evidence.canonical_citation_projection_hmac
            and envelope.effective_permission == evidence.effective_permission
            and type(evidence.provenance)
            in {ExplicitApprovalProvenance, LegacyHumanProvenance}
            and evidence.provenance is envelope.provenance
            and row.approval_provenance_hmac is not None
            and _trusted_provenance_hmac_is_exact(row, settings=settings)
            and (
                type(evidence.provenance) is ExplicitApprovalProvenance
                and row.evidence_link_set_hmac is not None
                or type(evidence.provenance) is LegacyHumanProvenance
                and row.evidence_link_set_hmac is None
            )
        )
    else:
        branch_valid = False
    if (
        not branch_valid
        or evidence.support_mode != slot.support_mode
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


def _trusted_provenance_hmac_is_exact(
    row: CanonicalServingProjection, *, settings: Settings
) -> bool:
    provenance = row.evidence.provenance
    if type(provenance) is ExplicitApprovalProvenance:
        if (
            type(provenance.evidence_links) is not tuple
            or not provenance.evidence_links
            or not all(
                type(value) is TrustedEvidenceLinkIdentity
                and value.approval_link_id == provenance.approval_link_id
                for value in provenance.evidence_links
            )
            or type(provenance.selected_citation_child) is not SelectedCitationChild
            or not _selected_child_matches_exact_link(provenance)
            or type(row.evidence_link_hmacs) is not tuple
            or len(row.evidence_link_hmacs) != len(provenance.evidence_links)
        ):
            return False
        expected_link_set = build_evidence_link_set_hmac(
            approval_link_id=provenance.approval_link_id,
            ordered_link_hmacs=tuple(
                (identity.trusted_knowledge_evidence_link_id, link_hmac)
                for identity, link_hmac in zip(
                    provenance.evidence_links,
                    row.evidence_link_hmacs,
                    strict=True,
                )
            ),
            settings=settings,
        )
        if row.evidence_link_set_hmac != expected_link_set:
            return False
        expected_approval = build_approval_provenance_hmac(
            branch='explicit_approval',
            approval_fingerprint_key_material_verifier=(
                provenance.approval_fingerprint_key_material_verifier
            ),
            approval_fingerprint_key_version=(
                provenance.approval_fingerprint_key_version
            ),
            approval_link_id=provenance.approval_link_id,
            approval_permission=provenance.approval_permission_level,
            claim_fingerprint=provenance.claim_fingerprint,
            evidence_link_set_hmac=expected_link_set,
            legacy_evidence_pairs_hmac=None,
            promotion_effect_kind=provenance.promotion_effect_kind,
            resolution_source=provenance.resolution_source,
            review_item_id=provenance.review_item_id,
            security_scope_id=provenance.security_scope_id,
            selected_citation_child_hmac=build_selected_citation_child_hmac(
                provenance.selected_citation_child, settings=settings
            ),
            settings=settings,
        )
        return row.approval_provenance_hmac == expected_approval
    if type(provenance) is LegacyHumanProvenance:
        if (
            row.evidence_link_set_hmac is not None
            or type(row.evidence_link_hmacs) is not tuple
            or row.evidence_link_hmacs
        ):
            return False
        expected_approval = build_approval_provenance_hmac(
            branch='legacy_human_base',
            approval_fingerprint_key_material_verifier=None,
            approval_fingerprint_key_version=None,
            approval_link_id=None,
            approval_permission=provenance.legacy_review_item_permission_level,
            claim_fingerprint=None,
            evidence_link_set_hmac=None,
            legacy_evidence_pairs_hmac=provenance.legacy_evidence_pairs_hmac,
            promotion_effect_kind=None,
            resolution_source=None,
            review_item_id=provenance.legacy_source_review_item_id,
            security_scope_id=None,
            selected_citation_child_hmac=None,
            settings=settings,
        )
        return row.approval_provenance_hmac == expected_approval
    return False


def _selected_child_matches_exact_link(
    provenance: ExplicitApprovalProvenance,
) -> bool:
    selected = provenance.selected_citation_child
    matches = tuple(
        identity
        for identity in provenance.evidence_links
        if identity.trusted_knowledge_evidence_link_id
        == selected.trusted_knowledge_evidence_link_id
    )
    if len(matches) != 1:
        return False
    link = matches[0]
    canonical_source_id = link.canonical_source_id
    if (
        type(canonical_source_id) is not str
        or not canonical_source_id.isascii()
        or not canonical_source_id.isdecimal()
    ):
        return False
    source_row_id = int(canonical_source_id)
    return bool(
        source_row_id > 0
        and str(source_row_id) == canonical_source_id
        and type(selected.source_row_id) is int
        and selected.source_row_id == source_row_id
        and type(selected.canonical_source_type) is str
        and selected.canonical_source_type == link.canonical_source_kind
        and type(selected.canonical_version_or_signature) is str
        and selected.canonical_version_or_signature
        == link.canonical_version_or_signature
    )


def _citation(
    slot: EvidenceSlot, row: CanonicalServingProjection, *, settings: Settings
) -> tuple[FrozenDict, str]:
    citation = _deep_freeze({
        'source_id': row.evidence.public_source_id,
        'source_url': row.source_url,
        'source_type': row.evidence.public_source_type,
        'permission_level': row.evidence.effective_permission,
        'source_snippet': row.source_snippet,
        'relevance_score': slot.relevance_score,
        'matched_terms': slot.matched_terms,
    })
    assert type(citation) is FrozenDict
    return citation, build_v1_citation_projection_hmac(
        row=row,
        relevance_score=slot.relevance_score,
        matched_terms=slot.matched_terms,
        settings=settings,
    )


def _search_result_hmac(
    *,
    row: CanonicalServingProjection,
    result: FrozenDict,
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
                cast(tuple[str, ...], result['matched_terms'])
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
            type(slot) is EvidenceSlot
            and type(slot.slot_id) is str
            and slot.slot_id == _SLOT_IDS[index]
            and type(slot.support_mode) is str
            and slot.support_mode in {'trusted_fact', 'source_observation'}
            for index, slot in enumerate(cast(tuple[EvidenceSlot, ...], slots))
        )
    )


def _selected_subset(
    slots: tuple[EvidenceSlot, ...], selected: tuple[EvidenceSlotId, ...]
) -> bool:
    if (
        type(selected) is not tuple
        or not selected
        or not all(type(value) is str and value in _SLOT_IDS for value in selected)
        or len(set(selected)) != len(selected)
    ):
        return False
    available = {slot.slot_id for slot in slots}
    return all(value in available for value in selected)


def _selected_observation_subset(
    observations: tuple[PreparedModelInfluenceObservation, ...],
    selected: tuple[EvidenceSlotId, ...],
) -> bool:
    if (
        type(observations) is not tuple
        or type(selected) is not tuple
        or not observations
        or not all(type(value) is str and value in _SLOT_IDS for value in selected)
        or not all(type(value) is PreparedModelInfluenceObservation for value in observations)
    ):
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


def _prepared_observation_child_hmac(
    *,
    ordinal: int,
    slot_id: EvidenceSlotId,
    support_mode: SupportMode,
    row: CanonicalServingProjection,
    settings: Settings,
) -> str:
    return _prepared_observation_child_hmac_from_values(
        ordinal=ordinal,
        slot_id=slot_id,
        support_mode=support_mode,
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
        settings=settings,
    )


def _prepared_observation_child_hmac_from_values(
    *,
    ordinal: int,
    slot_id: EvidenceSlotId,
    support_mode: SupportMode,
    lookup_identity: ServingEvidenceIdentity,
    effective_permission: PermissionLevel,
    serving_identity_hmac: str,
    serving_version_fingerprint: str,
    model_content_hmac: str,
    canonical_citation_projection_hmac: str,
    approval_provenance_hmac: str | None,
    evidence_link_set_hmac: str | None,
    settings: Settings,
) -> str:
    require_nonnegative_int(ordinal)
    _validate_slot_id(slot_id)
    _validate_support_mode(support_mode)
    return _fp(
        {
            'approval_provenance_hmac': approval_provenance_hmac,
            'canonical_citation_projection_hmac': require_lower_hex_64(
                canonical_citation_projection_hmac
            ),
            'effective_permission': _permission(effective_permission),
            'evidence_link_set_hmac': evidence_link_set_hmac,
            'lookup_identity_hmac': _prepared_lookup_identity_hmac(
                lookup_identity, settings=settings
            ),
            'model_content_hmac': require_lower_hex_64(model_content_hmac),
            'ordinal': ordinal,
            'serving_identity_hmac': require_lower_hex_64(serving_identity_hmac),
            'serving_version_fingerprint': require_lower_hex_64(
                serving_version_fingerprint
            ),
            'slot_id': slot_id,
            'support_mode': support_mode,
        },
        schema='rag-prepared-model-influence-observation:child:v2',
        policy='rag-answer:v2',
        settings=settings,
    )


def _model_influence_dependency_hmac_from_observation(
    observation: PreparedModelInfluenceObservation,
    *,
    dependency_role: DependencyRole,
    settings: Settings,
) -> str:
    if type(observation) is not PreparedModelInfluenceObservation:
        raise ValueError('model influence observation type is invalid')
    _validate_slot_id(observation.slot_id)
    _validate_support_mode(observation.support_mode)
    if dependency_role not in {'selected_citation', 'unselected_model_influence'}:
        raise ValueError('model influence role is invalid')
    for value in (
        observation.approval_provenance_hmac,
        observation.evidence_link_set_hmac,
    ):
        if value is not None:
            require_lower_hex_64(value)
    return _fp(
        {
            'approval_provenance_hmac': observation.approval_provenance_hmac,
            'canonical_citation_projection_hmac': require_lower_hex_64(
                observation.canonical_citation_projection_hmac
            ),
            'dependency_role': dependency_role,
            'effective_permission': _permission(observation.effective_permission),
            'evidence_link_set_hmac': observation.evidence_link_set_hmac,
            'model_content_hmac': require_lower_hex_64(
                observation.model_content_hmac
            ),
            'serving_identity_hmac': require_lower_hex_64(
                observation.serving_identity_hmac
            ),
            'serving_version_fingerprint': require_lower_hex_64(
                observation.serving_version_fingerprint
            ),
            'slot_id': observation.slot_id,
            'support_mode': observation.support_mode,
        },
        schema='rag-model-influence-dependency:v1',
        policy='rag-serving-evidence:v1',
        settings=settings,
    )


def _valid_prepared_influence_set(
    prepared: PreparedModelInfluenceSet,
    *,
    fence: ProjectionFence,
    settings: Settings,
) -> bool:
    try:
        observations = prepared.observations
        if (
            type(observations) is not tuple
            or not 1 <= len(observations) <= 8
            or prepared.prepared_corpus_generation != fence.prepared_corpus_generation
            or prepared.prepared_index_generation != fence.prepared_index_generation
            or prepared.prepared_readiness_hmac != fence.prepared_readiness_hmac
            or type(prepared.rendered_input_hmac) is not str
        ):
            return False
        for ordinal, observation in enumerate(observations):
            if (
                type(observation) is not PreparedModelInfluenceObservation
                or observation.ordinal != ordinal
                or observation.slot_id != _SLOT_IDS[ordinal]
                or type(observation.lookup_identity) is not ServingEvidenceIdentity
            ):
                return False
            if observation.observation_hmac != _prepared_observation_child_hmac_from_values(
                ordinal=ordinal,
                slot_id=observation.slot_id,
                support_mode=observation.support_mode,
                lookup_identity=observation.lookup_identity,
                effective_permission=observation.effective_permission,
                serving_identity_hmac=observation.serving_identity_hmac,
                serving_version_fingerprint=observation.serving_version_fingerprint,
                model_content_hmac=observation.model_content_hmac,
                canonical_citation_projection_hmac=(
                    observation.canonical_citation_projection_hmac
                ),
                approval_provenance_hmac=observation.approval_provenance_hmac,
                evidence_link_set_hmac=observation.evidence_link_set_hmac,
                settings=settings,
            ):
                return False
        expected_aggregate = _build_prepared_model_influence_set_hmac_v2(
            observations=observations,
            prepared_corpus_generation=prepared.prepared_corpus_generation,
            prepared_index_generation=prepared.prepared_index_generation,
            prepared_readiness_hmac=prepared.prepared_readiness_hmac,
            rendered_input_hmac=prepared.rendered_input_hmac,
            settings=settings,
        )
        return prepared.aggregate_observation_hmac == expected_aggregate
    except (TypeError, UnicodeError, ValueError):
        return False


def _empty_prepared_influence_set() -> PreparedModelInfluenceSet:
    return PreparedModelInfluenceSet((), 0, None, None, '', '')


def _prepared_lookup_identity_hmac(
    identity: ServingEvidenceIdentity, *, settings: Settings
) -> str:
    if type(identity) is not ServingEvidenceIdentity:
        raise ValueError('prepared lookup identity type is invalid')
    return _fp(
        {
            'canonical_citation_projection_hmac': require_lower_hex_64(
                identity.canonical_citation_projection_hmac
            ),
            'effective_permission': _permission(identity.effective_permission),
            'model_content_hmac': require_lower_hex_64(identity.model_content_hmac),
            'public_source_id_bytes': exact_utf8_bytes(
                require_exact_nonblank(identity.public_source_id)
            ),
            'public_source_type_bytes': _optional_bytes(identity.public_source_type),
            'serving_document_id_bytes': exact_utf8_bytes(
                require_exact_nonblank(identity.serving_document_id)
            ),
            'serving_identity_hmac': build_serving_identity_hmac(
                serving_document_id=identity.serving_document_id,
                serving_kind=identity.serving_kind,
                settings=settings,
            ),
            'serving_version_fingerprint': require_lower_hex_64(
                identity.serving_version_fingerprint
            ),
            'version_envelope_fingerprint': build_serving_version_fingerprint(
                identity.version_envelope, settings=settings
            ),
        },
        schema='rag-prepared-model-influence-lookup-identity:v1',
        policy='rag-answer:v2',
        settings=settings,
    )


def _deep_freeze(value: object) -> object:
    if type(value) is FrozenDict:
        return value
    if isinstance(value, dict):
        return FrozenDict({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def projection_record_to_transport(record: FrozenDict) -> dict[str, object]:
    """Copy one committed immutable record into mutable transport-only values."""
    if type(record) is not FrozenDict:
        raise TypeError('projection transport requires an exact frozen record')
    return {
        key: _transport_copy(value)
        for key, value in record.items()
    }


def _transport_copy(value: object) -> object:
    if type(value) is FrozenDict:
        return {
            key: _transport_copy(item)
            for key, item in cast(FrozenDict, value).items()
        }
    if type(value) is tuple:
        return [_transport_copy(item) for item in cast(tuple[object, ...], value)]
    return value


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
    if type(value) is not str or value not in _SLOT_IDS:
        raise ValueError('evidence slot id is invalid')
    return cast(EvidenceSlotId, value)


def _validate_support_mode(value: object) -> SupportMode:
    if type(value) is not str or value not in {'trusted_fact', 'source_observation'}:
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
