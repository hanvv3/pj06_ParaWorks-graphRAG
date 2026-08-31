from __future__ import annotations

import copy
import operator
import pickle
from dataclasses import FrozenInstanceError, replace

import pytest
from sqlalchemy.orm import Session

from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.models import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    CanonicalProjectionInfrastructureError,
    CanonicalProjectionTransactionError,
    FrozenDict,
    HiddenMembershipSnapshot,
    PreparedModelInfluenceSet,
    ProjectionFence,
    V1EvidenceProjection,
    build_model_influence_dependency_hmac,
    build_model_influence_set_hmac,
    build_prepared_model_influence_observation_hmac,
    build_v1_citation_projection_hmac,
    build_v1_search_result_set_projection_hmac,
    build_v1_selected_evidence_projection_hmac,
    derive_hidden_membership,
    projection_record_to_transport,
)
from backend.app.rag.retrieval import RetrievalCandidate, rank_evidence_slots
from backend.app.rag.serving_contracts import (
    CanonicalServingProjection,
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    RawChunkProvenance,
    RawServingVersionEnvelope,
    SelectedCitationChild,
    ServingEvidence,
    ServingEvidenceIdentity,
    TrustedEvidenceLinkIdentity,
    TrustedServingVersionEnvelope,
    build_approval_provenance_hmac,
    build_canonical_citation_projection_hmac,
    build_evidence_link_set_hmac,
    build_model_content_hmac,
    build_selected_citation_child_hmac,
    build_serving_identity_hmac,
    build_serving_version_fingerprint,
)
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url='sqlite://',
        agent_runtime_fingerprint_secret='task-8-projection-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='task8-v1',
    )


def _scope() -> SecurityScope:
    return SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='task8-actor',
        workspace_scope_id='workspace-a',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public', 'internal', 'restricted'),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )


def _projection(ordinal: int, *, trusted: bool = False) -> CanonicalServingProjection:
    prefix = 'history_event' if trusted else 'chunk'
    serving_kind = 'trusted_knowledge' if trusted else 'raw_chunk'
    support_mode = 'trusted_fact' if trusted else 'source_observation'
    document_id = f'{prefix}:{ordinal}'
    signature = f'{ordinal:x}'.zfill(64)
    model_content = f'canonical content {ordinal}'
    source_url = f'https://mail.example.test/messages/{ordinal}'
    source_snippet = f'canonical snippet {ordinal}'
    public_source_id = document_id if trusted else f'gmail:{ordinal}'
    public_source_type = 'history_event' if trusted else 'gmail'
    model_hmac = build_model_content_hmac(
        serving_kind=serving_kind,
        model_content=model_content,
        settings=_settings(),  # type: ignore[arg-type]
    )
    citation_hmac = build_canonical_citation_projection_hmac(
        public_source_id=public_source_id,
        public_source_type=public_source_type,
        source_url=source_url,
        source_snippet=source_snippet,
        effective_permission='internal',
        settings=_settings(),
    )
    provenance = LegacyHumanProvenance(
        branch='legacy_human_base',
        legacy_binding='review_item',
        legacy_evidence_pairs_hmac=signature,
        legacy_review_item_permission_level='internal',
        legacy_source_review_item_id=ordinal,
    )
    envelope = (
        TrustedServingVersionEnvelope(
            serving_document_id=document_id,
            knowledge_type='history_event',
            knowledge_id=ordinal,
            model_content_hmac=model_hmac,
            canonical_citation_projection_hmac=citation_hmac,
            effective_permission='internal',
            provenance=provenance,
        )
        if trusted
        else RawServingVersionEnvelope(
            serving_document_id=document_id,
            source_row_id=ordinal,
            public_source_id=f'gmail:{ordinal}',
            document_id=ordinal,
            document_version_id=ordinal,
            current_document_version_id=ordinal,
            document_chunk_id=ordinal,
            parser_run_id=ordinal,
            external_revision=f'rev-{ordinal}',
            server_content_signature_schema='server-source-content:v1',
            server_content_signature=signature,
            parser_policy_version='parser-policy:v1',
            parser_version='parser:v1',
            chunk_policy_version='chunk:v1',
            model_content_hmac=model_hmac,
            canonical_citation_projection_hmac=citation_hmac,
            effective_permission='internal',
        )
    )
    version_hmac = build_serving_version_fingerprint(envelope, settings=_settings())
    identity = ServingEvidenceIdentity(
        serving_document_id=document_id,
        serving_kind=serving_kind,  # type: ignore[arg-type]
        public_source_id=public_source_id,
        public_source_type=public_source_type,
        effective_permission='internal',
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        serving_version_fingerprint=version_hmac,
        version_envelope=envelope,
    )
    evidence = ServingEvidence(
        serving_document_id=document_id,
        serving_kind=serving_kind,  # type: ignore[arg-type]
        public_source_id=public_source_id,
        public_source_type=public_source_type,
        support_mode=support_mode,  # type: ignore[arg-type]
        model_content=model_content,
        title=f'title {ordinal}',
        effective_permission='internal',
        serving_identity_hmac=build_serving_identity_hmac(
            serving_document_id=document_id,
            serving_kind=serving_kind,
            settings=_settings(),  # type: ignore[arg-type]
        ),
        serving_version_fingerprint=version_hmac,
        model_content_hmac=model_hmac,
        canonical_citation_projection_hmac=citation_hmac,
        version_envelope=envelope,
        provenance=(
            provenance
            if trusted
            else RawChunkProvenance(branch='raw_chunk', raw_version=envelope)  # type: ignore[arg-type]
        ),
    )
    approval_hmac = (
        build_approval_provenance_hmac(
            branch='legacy_human_base',
            approval_fingerprint_key_material_verifier=None,
            approval_fingerprint_key_version=None,
            approval_link_id=None,
            approval_permission='internal',
            claim_fingerprint=None,
            evidence_link_set_hmac=None,
            legacy_evidence_pairs_hmac=signature,
            promotion_effect_kind=None,
            resolution_source=None,
            review_item_id=ordinal,
            security_scope_id=None,
            selected_citation_child_hmac=None,
            settings=_settings(),
        )
        if trusted
        else None
    )
    return CanonicalServingProjection(
        identity=identity,
        evidence=evidence,
        public_result_id=ordinal,
        source_url=source_url,
        source_snippet=source_snippet,
        parser_status='parsed',
        parser_status_reason=None,
        revision_id=f'rev-{ordinal}',
        approval_provenance_hmac=approval_hmac,
        evidence_link_set_hmac=None,
    )


def _explicit_projection(
    *,
    canonical_source_id: str = '1',
    selected_link_id: int = 101,
    selected_source_row_id: int = 1,
    selected_source_type: str = 'gmail',
    selected_version: str = 'revision-1',
) -> CanonicalServingProjection:
    row = _projection(1, trusted=True)
    identity = TrustedEvidenceLinkIdentity(
        trusted_knowledge_evidence_link_id=101,
        approval_link_id=201,
        canonical_source_kind='gmail',
        canonical_source_id=canonical_source_id,
        canonical_version_or_signature='revision-1',
        evidence_hash='a' * 64,
        fingerprint_key_version='task8-key-v1',
        fingerprint_key_material_verifier='b' * 64,
    )
    selected_child = SelectedCitationChild(
        trusted_knowledge_evidence_link_id=selected_link_id,
        source_row_id=selected_source_row_id,
        canonical_source_type=selected_source_type,
        canonical_version_or_signature=selected_version,
        review_item_source_pair_ordinal=0,
    )
    provenance = ExplicitApprovalProvenance(
        branch='explicit_approval',
        approval_link_id=201,
        review_item_id=1,
        security_scope_id='workspace-a',
        promotion_effect_kind='trusted_knowledge',
        resolution_source='human',
        claim_fingerprint='c' * 64,
        approval_permission_level='internal',
        approval_fingerprint_key_version='task8-key-v1',
        approval_fingerprint_key_material_verifier='d' * 64,
        evidence_links=(identity,),
        selected_citation_child=selected_child,
    )
    evidence_link_hmac = 'e' * 64
    link_set_hmac = build_evidence_link_set_hmac(
        approval_link_id=201,
        ordered_link_hmacs=((101, evidence_link_hmac),),
        settings=_settings(),
    )
    approval_hmac = build_approval_provenance_hmac(
        branch='explicit_approval',
        approval_fingerprint_key_material_verifier=(
            provenance.approval_fingerprint_key_material_verifier
        ),
        approval_fingerprint_key_version=provenance.approval_fingerprint_key_version,
        approval_link_id=provenance.approval_link_id,
        approval_permission=provenance.approval_permission_level,
        claim_fingerprint=provenance.claim_fingerprint,
        evidence_link_set_hmac=link_set_hmac,
        legacy_evidence_pairs_hmac=None,
        promotion_effect_kind=provenance.promotion_effect_kind,
        resolution_source=provenance.resolution_source,
        review_item_id=provenance.review_item_id,
        security_scope_id=provenance.security_scope_id,
        selected_citation_child_hmac=build_selected_citation_child_hmac(
            selected_child, settings=_settings()
        ),
        settings=_settings(),
    )
    envelope = replace(row.evidence.version_envelope, provenance=provenance)
    version_hmac = build_serving_version_fingerprint(envelope, settings=_settings())
    return replace(
        row,
        identity=replace(
            row.identity,
            serving_version_fingerprint=version_hmac,
            version_envelope=envelope,
        ),
        evidence=replace(
            row.evidence,
            serving_version_fingerprint=version_hmac,
            version_envelope=envelope,
            provenance=provenance,
        ),
        approval_provenance_hmac=approval_hmac,
        evidence_link_set_hmac=link_set_hmac,
        evidence_link_hmacs=(evidence_link_hmac,),
    )


class _Transaction:
    def __init__(self) -> None:
        self.active = True
        self.token = object()

    def in_transaction(self) -> bool:
        return self.active

    def get_transaction(self) -> object | None:
        return self.token if self.active else None


class _Resolver:
    def __init__(self, rows: dict[str, CanonicalServingProjection]) -> None:
        self.rows = rows
        self.seen: list[str] = []

    def resolve_projection_candidate(self, *, db, identity, scope):
        assert db.in_transaction()
        assert scope == _scope()
        self.seen.append(identity.serving_document_id)
        return self.rows.get(identity.serving_document_id)

    def resolve_projection_candidate_strict(self, *, db, identity, scope):
        return self.resolve_projection_candidate(db=db, identity=identity, scope=scope)


def _candidate(row: CanonicalServingProjection, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        evidence=row.evidence,
        relevance_score=score,
        matched_terms=('alpha', f'term-{row.public_result_id}'),
    )


def _project_single_row(row: CanonicalServingProjection) -> V1EvidenceProjection:
    projector = CanonicalEvidenceProjector(
        db=_Transaction(),
        settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row}),
    )
    return projector.project_selected(
        rank_evidence_slots((_candidate(row, 0.9),)),
        selected_slot_ids=('E1',),
        scope=_scope(),
        fence=_fence(hidden_hmac=_hidden().membership_hmac),
    )


def _fence(*, hidden_hmac: str) -> ProjectionFence:
    return ProjectionFence(
        prepared_corpus_generation=7,
        prepared_index_generation=4,
        current_corpus_generation=7,
        current_index_generation=4,
        prepared_readiness_hmac='a' * 64,
        current_readiness_hmac='a' * 64,
        prepared_hidden_membership_hmac=hidden_hmac,
        current_hidden_membership_hmac=hidden_hmac,
    )


def _hidden(*, members: tuple[str, ...] = ()) -> HiddenMembershipSnapshot:
    return derive_hidden_membership(
        ordered_hidden_member_identity_hmacs=members,
        top_candidate_window_hmac='b' * 64,
        settings=_settings(),
    )


def build_hidden_membership_hmac(
    *,
    denied_known_member_identity_hmacs: tuple[str, ...],
    public_hidden_count: int,
    capped: bool,
    top_candidate_window_hmac: str,
    settings: Settings,
) -> str:
    full_members = (
        denied_known_member_identity_hmacs + ('e' * 64,)
        if capped
        else denied_known_member_identity_hmacs
    )
    snapshot = derive_hidden_membership(
        ordered_hidden_member_identity_hmacs=full_members,
        top_candidate_window_hmac=top_candidate_window_hmac,
        settings=settings,
    )
    assert snapshot.public_hidden_count == public_hidden_count
    assert snapshot.capped is capped
    return snapshot.membership_hmac


def _prepare_rows(
    rows: tuple[CanonicalServingProjection, ...],
    *,
    rendered_input_hmac: str,
) -> tuple[CanonicalEvidenceProjector, PreparedModelInfluenceSet]:
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row for row in rows}),
    )
    prepared = projector.prepare_model_influence(
        rank_evidence_slots(tuple(_candidate(row, 0.9) for row in rows)),
        scope=_scope(), prepared_corpus_generation=1, prepared_index_generation=None,
        prepared_readiness_hmac=None, rendered_input_hmac=rendered_input_hmac,
    )
    return projector, prepared


def test_slotting_is_trusted_first_bounded_contiguous_and_whole_tail_dropped() -> None:
    raw = [_projection(value) for value in range(1, 8)]
    trusted = [_projection(value, trusted=True) for value in range(8, 11)]
    candidates = tuple(
        _candidate(row, 1.0 - index / 100) for index, row in enumerate((*raw, *trusted))
    )

    slots = rank_evidence_slots(candidates, max_serialized_content_chars=40)

    assert [slot.evidence.serving_kind for slot in slots] == [
        'trusted_knowledge',
        'trusted_knowledge',
    ]
    assert [slot.slot_id for slot in slots] == ['E1', 'E2']
    with pytest.raises(FrozenInstanceError):
        slots[0].slot_id = 'E8'  # type: ignore[misc]


def test_projection_uses_fresh_canonical_bytes_and_selected_is_exact_subset() -> None:
    stale = _projection(1)
    fresh = stale
    resolver = _Resolver({stale.evidence.serving_document_id: fresh})
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=resolver
    )
    slots = rank_evidence_slots((_candidate(stale, 0.9),))
    hidden = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=(),
        public_hidden_count=0,
        capped=False,
        top_candidate_window_hmac='b' * 64,
        settings=_settings(),
    )

    projected = projector.project_selected(
        slots,
        selected_slot_ids=('E1',),
        scope=_scope(),
        fence=_fence(hidden_hmac=hidden),
    )
    invalid = projector.project_selected(
        slots,
        selected_slot_ids=('E2',),
        scope=_scope(),
        fence=_fence(hidden_hmac=hidden),
    )

    assert projected.source_links == (fresh.source_url,)
    assert projected.source_snippets == (fresh.source_snippet,)
    assert projected.citations[0]['source_id'] == fresh.evidence.public_source_id
    assert invalid.citations == ()


def test_search_projects_all_visible_slots_but_answer_only_selected_slots() -> None:
    rows = tuple(_projection(value) for value in range(1, 4))
    resolver = _Resolver({row.evidence.serving_document_id: row for row in rows})
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=resolver
    )
    slots = rank_evidence_slots(
        tuple(_candidate(row, 1.0 - value / 10) for value, row in enumerate(rows))
    )
    hidden = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=(),
        public_hidden_count=0,
        capped=False,
        top_candidate_window_hmac='b' * 64,
        settings=_settings(),
    )

    search = projector.project_search(
        slots, scope=_scope(), fence=_fence(hidden_hmac=hidden)
    )
    selected = projector.project_selected(
        slots,
        selected_slot_ids=('E2',),
        scope=_scope(),
        fence=_fence(hidden_hmac=hidden),
    )

    assert len(search.search_results) == 3
    assert selected.source_ids == (rows[1].evidence.public_source_id,)
    assert len(selected.citations) == 1


def test_prepare_has_no_role_and_finalization_keeps_every_provider_visible_slot() -> (
    None
):
    rows = tuple(_projection(value) for value in range(1, 4))
    resolver = _Resolver({row.evidence.serving_document_id: row for row in rows})
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=resolver
    )
    slots = rank_evidence_slots(tuple(_candidate(row, 0.9) for row in rows))

    observations = projector.prepare_model_influence(
        slots,
        scope=_scope(),
        prepared_corpus_generation=7,
        prepared_index_generation=4,
        prepared_readiness_hmac='a' * 64,
        rendered_input_hmac='c' * 64,
    )
    dependencies = projector.finalize_model_influence_dependencies(
        observations,
        selected_slot_ids=('E2',),
        scope=_scope(),
        fence=_fence(
            hidden_hmac=build_hidden_membership_hmac(
                denied_known_member_identity_hmacs=(),
                public_hidden_count=0,
                capped=False,
                top_candidate_window_hmac='d' * 64,
                settings=_settings(),
            )
        ),
    )

    assert len(observations) == len(dependencies) == 3
    assert not hasattr(observations[0], 'dependency_role')
    assert [value.dependency_role for value in dependencies] == [
        'unselected_model_influence',
        'selected_citation',
        'unselected_model_influence',
    ]


@pytest.mark.parametrize('drift', ['corpus', 'index', 'readiness', 'hidden', 'source'])
def test_any_generation_hidden_or_canonical_drift_redacts_whole_projection(
    drift: str,
) -> None:
    rows = tuple(_projection(value) for value in range(1, 3))
    current = {row.evidence.serving_document_id: row for row in rows}
    resolver = _Resolver(current)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=resolver
    )
    slots = rank_evidence_slots(tuple(_candidate(row, 0.9) for row in rows))
    hidden = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=('d' * 64,),
        public_hidden_count=1,
        capped=False,
        top_candidate_window_hmac='b' * 64,
        settings=_settings(),
    )
    fence = _fence(hidden_hmac=hidden)
    if drift == 'corpus':
        fence = replace(fence, current_corpus_generation=8)
    elif drift == 'index':
        fence = replace(fence, current_index_generation=5)
    elif drift == 'readiness':
        fence = replace(fence, current_readiness_hmac='e' * 64)
    elif drift == 'hidden':
        fence = replace(fence, current_hidden_membership_hmac='e' * 64)
    else:
        current[rows[1].evidence.serving_document_id] = replace(
            rows[1], source_snippet='changed after provider'
        )

    result = projector.project_selected(
        slots, selected_slot_ids=('E1', 'E2'), scope=_scope(), fence=fence
    )

    assert result.citations == ()
    assert result.source_ids == ()
    assert result.search_results == ()


def test_projection_rejects_invalid_score_unicode_and_url_as_whole_set() -> None:
    row = _projection(1)
    bad_url = replace(row, source_url='javascript:alert(1)')
    resolver = _Resolver({row.evidence.serving_document_id: bad_url})
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=resolver
    )
    slots = rank_evidence_slots((_candidate(row, 0.9),))

    assert (
        projector.project_search(
            slots,
            scope=_scope(),
            fence=replace(
                ProjectionFence.generations_only(1, None),
                prepared_hidden_membership_hmac='a' * 64,
                current_hidden_membership_hmac='a' * 64,
            ),
        ).search_results
        == ()
    )


def test_hmac_domains_bind_order_bits_nullable_keys_empty_and_role() -> None:
    settings = _settings()
    row = _projection(1)
    citation = build_v1_citation_projection_hmac(
        row=row,
        relevance_score=-0.0,
        matched_terms=('alpha', 'beta'),
        settings=settings,
    )
    plus_zero = build_v1_citation_projection_hmac(
        row=row, relevance_score=0.0, matched_terms=('alpha', 'beta'), settings=settings
    )
    reordered = build_v1_citation_projection_hmac(
        row=row,
        relevance_score=-0.0,
        matched_terms=('beta', 'alpha'),
        settings=settings,
    )
    selected = build_v1_selected_evidence_projection_hmac(
        citation_hmacs=(citation,),
        source_ids=(row.evidence.public_source_id,),
        source_links=(row.source_url,),
        source_snippets=(row.source_snippet,),
        settings=settings,
    )
    empty_search = build_v1_search_result_set_projection_hmac((), settings=settings)
    hidden = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=(),
        public_hidden_count=0,
        capped=False,
        top_candidate_window_hmac='a' * 64,
        settings=settings,
    )
    projector, prepared_set = _prepare_rows((row,), rendered_input_hmac='b' * 64)
    prepared = prepared_set.aggregate_observation_hmac
    selected_dep = build_model_influence_dependency_hmac(
        row=row,
        slot_id='E1',
        support_mode='source_observation',
        dependency_role='selected_citation',
        settings=settings,
    )
    unselected_dep = build_model_influence_dependency_hmac(
        row=row,
        slot_id='E1',
        support_mode='source_observation',
        dependency_role='unselected_model_influence',
        settings=settings,
    )
    dependencies = projector.finalize_model_influence_dependencies(
        prepared_set, selected_slot_ids=('E1',), scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )
    influence_set = build_model_influence_set_hmac(
        dependencies=dependencies, settings=settings,
    )

    values = {
        citation,
        plus_zero,
        reordered,
        selected,
        empty_search,
        hidden,
        prepared,
        selected_dep,
        unselected_dep,
        influence_set,
    }
    assert len(values) == 10
    assert all(len(value) == 64 for value in values)
    assert (
        citation,
        plus_zero,
        reordered,
        selected,
        empty_search,
        hidden,
        prepared,
        selected_dep,
        unselected_dep,
        influence_set,
    ) == (
        'd7dc5e9addd4191fd046ceb97ed366c85da6ea9c6fe78c9b8d432aaaf2b06704',
        '67bebd026afb9aeaad61f73a61df64f89ddce4b34cc82dfb6e1247867cb24c2c',
        '19f0e53ea270ff067ffc4ec2d9c3de21d645eab202f881d35aeace9814d1c57f',
        '790b585220f3c9147d20e593dab74cadf75d16424b3b0850eb313e99a77a5d36',
        '4f08fe7bb2aa66b637ee74824cf3ab5cb0802f96baac70bb3331b074236bf70b',
        '106c275260bfbb72e8b0b3a2fbfa86b4507401c71f186a5307bd8eacc74d5539',
        'c4ba54ec47b9f0da6e5e194ca0f5173a2d8aea312deb2e2629adf9155b34b3f1',
        '885d56f3d313746c850614c65c92fd38a9e5e29a3d4d6bacda927a0599a702d3',
        '8f0c0ccc7dc401027a4ead4bdab6b01a16c8e35b134ad980e14637f26092124e',
        '6021d66c5bef567f91a76f1d1063a313d9f24416ce8e0e09d5a226ae5c8a5f4b',
    )


def test_hmac_domains_change_on_order_nullable_presence_slot_and_strictest_permission() -> (
    None
):
    settings = _settings()
    first = _projection(1)
    second = _projection(2)
    first_citation = build_v1_citation_projection_hmac(
        row=first, relevance_score=0.9, matched_terms=('alpha',), settings=settings
    )
    second_citation = build_v1_citation_projection_hmac(
        row=second, relevance_score=0.8, matched_terms=('beta',), settings=settings
    )
    selected_forward = build_v1_selected_evidence_projection_hmac(
        citation_hmacs=(first_citation, second_citation),
        source_ids=(first.evidence.public_source_id, second.evidence.public_source_id),
        source_links=(first.source_url, second.source_url),
        source_snippets=(first.source_snippet, second.source_snippet),
        settings=settings,
    )
    selected_reverse = build_v1_selected_evidence_projection_hmac(
        citation_hmacs=(second_citation, first_citation),
        source_ids=(second.evidence.public_source_id, first.evidence.public_source_id),
        source_links=(second.source_url, first.source_url),
        source_snippets=(second.source_snippet, first.source_snippet),
        settings=settings,
    )
    search_forward = build_v1_search_result_set_projection_hmac(
        (first_citation, second_citation), settings=settings
    )
    search_reverse = build_v1_search_result_set_projection_hmac(
        (second_citation, first_citation), settings=settings
    )
    hidden_forward = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=('a' * 64, 'b' * 64),
        public_hidden_count=2,
        capped=False,
        top_candidate_window_hmac='c' * 64,
        settings=settings,
    )
    hidden_reverse = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=('b' * 64, 'a' * 64),
        public_hidden_count=2,
        capped=False,
        top_candidate_window_hmac='c' * 64,
        settings=settings,
    )
    projector, prepared_set = _prepare_rows(
        (first, second), rendered_input_hmac='d' * 64
    )
    legacy_entries = (
        (first.identity, 'E1', 'source_observation', None, None),
        (second.identity, 'E2', 'source_observation', None, None),
    )
    prepared_null = build_prepared_model_influence_observation_hmac(
        entries=legacy_entries,
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        rendered_input_hmac='d' * 64,
        settings=settings,
    )
    prepared_present = build_prepared_model_influence_observation_hmac(
        entries=(
            (first.identity, 'E1', 'source_observation', 'e' * 64, None),
            legacy_entries[1],
        ),
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        rendered_input_hmac='d' * 64,
        settings=settings,
    )
    dep_e1 = build_model_influence_dependency_hmac(
        row=first,
        slot_id='E1',
        support_mode='source_observation',
        dependency_role='selected_citation',
        settings=settings,
    )
    dep_e2 = build_model_influence_dependency_hmac(
        row=first,
        slot_id='E2',
        support_mode='source_observation',
        dependency_role='selected_citation',
        settings=settings,
    )
    dependencies = projector.finalize_model_influence_dependencies(
        prepared_set, selected_slot_ids=('E1',), scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )
    set_internal = build_model_influence_set_hmac(
        dependencies=dependencies, settings=settings
    )

    assert selected_forward != selected_reverse
    assert search_forward != search_reverse
    assert hidden_forward != hidden_reverse
    assert prepared_null != prepared_present
    assert dep_e1 != dep_e2
    assert len(set_internal) == 64
    with pytest.raises(ValueError):
        build_model_influence_set_hmac(
            dependencies=tuple(reversed(dependencies)), settings=settings
        )
    with pytest.raises(ValueError):
        build_model_influence_set_hmac(
            dependencies=(
                replace(
                    dependencies[0],
                    fresh_lookup_identity=replace(
                        dependencies[0].fresh_lookup_identity,
                        serving_document_id='forged:identity',
                    ),
                ),
                *dependencies[1:],
            ),
            settings=settings,
        )
    with pytest.raises(TypeError):
        build_model_influence_set_hmac(  # type: ignore[call-arg]
            dependencies=dependencies,
            strictest_permission='public',
            settings=settings,
        )


@pytest.mark.parametrize(
    'mutation', ['unicode', 'score', 'permission', 'version', 'provenance']
)
def test_invalid_or_drifted_canonical_field_redacts_entire_selected_answer(
    mutation: str,
) -> None:
    row = _projection(1)
    current = row
    slot = rank_evidence_slots((_candidate(row, 0.9),))[0]
    if mutation == 'unicode':
        current = replace(row, source_snippet='invalid\ud800')
    elif mutation == 'score':
        slot = replace(slot, relevance_score=float('nan'))
    elif mutation == 'permission':
        current = replace(
            row,
            evidence=replace(row.evidence, effective_permission='restricted'),
        )
    elif mutation == 'version':
        current = replace(
            row,
            identity=replace(row.identity, serving_version_fingerprint='f' * 64),
        )
    else:
        current = replace(row, approval_provenance_hmac='f' * 64)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(),
        settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: current}),
    )
    hidden = build_hidden_membership_hmac(
        denied_known_member_identity_hmacs=(),
        public_hidden_count=0,
        capped=False,
        top_candidate_window_hmac='a' * 64,
        settings=_settings(),
    )

    result = projector.project_selected(
        (slot,),
        selected_slot_ids=('E1',),
        scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=hidden,
            current_hidden_membership_hmac=hidden,
        ),
    )

    assert result.citations == ()
    assert result.source_ids == ()


def test_projector_requires_active_transaction_and_never_exposes_denied_ids() -> None:
    transaction = _Transaction()
    transaction.active = False
    projector = CanonicalEvidenceProjector(
        db=transaction, settings=_settings(), resolver=_Resolver({})
    )
    hidden = _hidden(
        members=tuple(f'{value:x}'.zfill(64) for value in range(21)),
    ).membership_hmac

    with pytest.raises(RuntimeError, match='active transaction'):
        projector.project_search((), scope=_scope(), fence=_fence(hidden_hmac=hidden))
    assert 'f' * 64 not in repr(hidden)


def test_default_resolver_projects_fresh_raw_and_legacy_trusted_rows(
    db_session: Session,
) -> None:
    settings = _settings()
    signature = '9' * 64
    source = Source(
        source_type='gmail',
        source_id='gmail:task8',
        source_url='https://mail.example.test/messages/task8',
        title='Task 8 source',
        author='owner@example.test',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db_session.add(source)
    db_session.flush()
    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id, version='v1', body='Fresh raw bytes'
    )
    db_session.add(version)
    db_session.flush()
    parser = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id='raw-revision',
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db_session.add(parser)
    db_session.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        parser_run_id=parser.id,
        chunk_index=0,
        text='Fresh raw bytes',
        source_snippet='Fresh raw bytes',
        permission_level='internal',
        metadata_={},
    )
    db_session.add(chunk)
    document.current_document_version_id = version.id
    review = ReviewItem(
        item_type='history_event',
        payload={'title': 'Trusted'},
        source_links=['https://knowledge.example.test/review/1'],
        source_snippets=['Fresh trusted citation'],
        confidence_score=0.99,
        permission_level='internal',
        status='approved',
        resolution_source='human',
    )
    db_session.add(review)
    db_session.flush()
    history = HistoryEvent(
        project_key='project-a',
        title='Trusted',
        reason='Fresh trusted bytes',
        source_links=list(review.source_links),
        source_snippets=list(review.source_snippets),
        confidence_score=0.99,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=review.id,
    )
    db_session.add(history)
    db_session.commit()
    raw = CanonicalSourceObservationResolver(
        db=db_session, settings=settings
    ).resolve_for_index(chunk.id)
    trusted = TrustedServingEnvelopeResolver(
        db=db_session, settings=settings
    ).resolve_for_index('history_event', history.id)
    assert raw is not None and trusted is not None
    slots = rank_evidence_slots(
        (
            RetrievalCandidate(trusted.evidence, 0.95, ('trusted',)),
            RetrievalCandidate(raw.evidence, 0.90, ('raw',)),
        )
    )
    projector = CanonicalEvidenceProjector(db=db_session, settings=settings)
    result = projector.project_search(
        slots,
        scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac='a' * 64,
            current_hidden_membership_hmac='a' * 64,
        ),
    )

    assert [value['source_snippet'] for value in result.search_results] == [
        'Fresh trusted citation',
        'Fresh raw bytes',
    ]
    assert [value['source_url'] for value in result.search_results] == [
        'https://knowledge.example.test/review/1',
        'https://mail.example.test/messages/task8',
    ]


@pytest.mark.parametrize(
    'mutation',
    [
        'drop', 'only_e2', 'reorder', 'renumber', 'child_hmac', 'aggregate',
        'rendered', 'generation', 'support', 'raw_to_trusted',
    ],
)
def test_finalizer_authenticates_the_complete_prepared_influence_set(mutation: str) -> None:
    rows = tuple(_projection(value) for value in range(1, 4))
    resolver = _Resolver({row.evidence.serving_document_id: row for row in rows})
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=resolver
    )
    prepared = projector.prepare_model_influence(
        rank_evidence_slots(tuple(_candidate(row, 0.9) for row in rows)),
        scope=_scope(), prepared_corpus_generation=7, prepared_index_generation=4,
        prepared_readiness_hmac='a' * 64, rendered_input_hmac='c' * 64,
    )
    assert type(prepared) is PreparedModelInfluenceSet
    if mutation == 'drop':
        prepared = replace(prepared, observations=prepared.observations[:-1])
    elif mutation == 'only_e2':
        prepared = replace(prepared, observations=(prepared.observations[1],))
    elif mutation == 'reorder':
        prepared = replace(prepared, observations=(prepared.observations[1], prepared.observations[0], prepared.observations[2]))
    elif mutation == 'renumber':
        prepared = replace(prepared, observations=(replace(prepared.observations[0], slot_id='E2'), *prepared.observations[1:]))
    elif mutation == 'child_hmac':
        prepared = replace(prepared, observations=(replace(prepared.observations[0], observation_hmac='f' * 64), *prepared.observations[1:]))
    elif mutation == 'aggregate':
        prepared = replace(prepared, aggregate_observation_hmac='f' * 64)
    elif mutation == 'rendered':
        prepared = replace(prepared, rendered_input_hmac='f' * 64)
    elif mutation == 'generation':
        prepared = replace(prepared, prepared_corpus_generation=8)
    elif mutation == 'support':
        prepared = replace(prepared, observations=(replace(prepared.observations[0], support_mode='trusted_fact'), *prepared.observations[1:]))
    else:
        forged_identity = replace(
            prepared.observations[0].lookup_identity,
            serving_kind='trusted_knowledge',
        )
        prepared = replace(
            prepared,
            observations=(
                replace(prepared.observations[0], lookup_identity=forged_identity),
                *prepared.observations[1:],
            ),
        )

    resolution_count = len(resolver.seen)
    dependencies = projector.finalize_model_influence_dependencies(
        prepared, selected_slot_ids=('E2',), scope=_scope(),
        fence=_fence(hidden_hmac=_hidden().membership_hmac),
    )

    assert dependencies == ()
    assert len(resolver.seen) == resolution_count


def test_projection_dtos_are_recursively_immutable_and_copy_does_not_open_alias() -> None:
    row = _projection(1)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row}),
    )
    projection = projector.project_search(
        rank_evidence_slots((_candidate(row, 0.9),)), scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )
    result = projection.search_results[0]
    assert type(result) is FrozenDict
    assert type(result['citation']) is FrozenDict
    assert type(result['matched_terms']) is tuple
    original_hmac = projection.projection_hmac
    for operation in (
        lambda: result.__setitem__('source_id', 'forged'),
        lambda: result.update({'source_id': 'forged'}),
        lambda: result.setdefault('x', 1),
        lambda: result.pop('source_id'),
        lambda: result.clear(),
        lambda: result.__ior__({'source_id': 'forged'}),
        lambda: dict.__setitem__(result, 'source_id', 'forged'),
        lambda: setattr(result, '_items', (('source_id', 'forged'),)),
        lambda: delattr(result, '_items'),
        lambda: operator.setitem(result['matched_terms'], 0, 'forged'),
        lambda: result.copy().update({'source_id': 'forged'}),
    ):
        with pytest.raises((AttributeError, TypeError)):
            operation()
    assert projection.projection_hmac == original_hmac
    assert result['source_id'] == row.evidence.public_source_id


def test_frozen_projection_mapping_has_no_attribute_or_serialization_mutation_path() -> None:
    row = _projection(1)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row}),
    )
    result = projector.project_search(
        rank_evidence_slots((_candidate(row, 0.9),)), scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    ).search_results[0]

    assert FrozenDict.__slots__ == ()
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(result, '_items', (('source_id', 'forged'),))
    assert tuple.__getitem__(result, 0) != ('source_id', 'forged')
    for cloned in (copy.copy(result), copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
        assert cloned == result
        with pytest.raises(TypeError):
            cloned.update({'source_id': 'forged'})
        assert cloned['source_id'] == row.evidence.public_source_id
    transport = projection_record_to_transport(result)
    transport['source_id'] = 'transport-only'
    assert result['source_id'] == row.evidence.public_source_id


def test_legacy_v1_prepared_and_dependency_hmac_goldens_remain_exact() -> None:
    row = _projection(1)
    prepared_v1 = build_prepared_model_influence_observation_hmac(
        entries=((row.identity, 'E1', 'source_observation', None, None),),
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        rendered_input_hmac='b' * 64,
        settings=_settings(),
    )
    selected_dependency_v1 = build_model_influence_dependency_hmac(
        row=row, slot_id='E1', support_mode='source_observation',
        dependency_role='selected_citation', settings=_settings(),
    )

    assert prepared_v1 == '940b88c5ea09bc54dc30b7dfacaf6c5ce96a8d6cd10dac823e3e4fadd6c5a696'
    assert selected_dependency_v1 == '885d56f3d313746c850614c65c92fd38a9e5e29a3d4d6bacda927a0599a702d3'
    _, prepared_v2 = _prepare_rows((row,), rendered_input_hmac='b' * 64)
    assert prepared_v2.aggregate_observation_hmac != prepared_v1
    assert replace(prepared_v2, rendered_input_hmac='c' * 64) != prepared_v2


def test_finalizer_revalidates_exact_fresh_branch_provenance() -> None:
    row = _projection(1)
    current = {row.evidence.serving_document_id: row}
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=_Resolver(current)
    )
    prepared = projector.prepare_model_influence(
        rank_evidence_slots((_candidate(row, 0.9),)),
        scope=_scope(), prepared_corpus_generation=7, prepared_index_generation=4,
        prepared_readiness_hmac='a' * 64, rendered_input_hmac='c' * 64,
    )
    current[row.evidence.serving_document_id] = replace(
        row,
        evidence=replace(
            row.evidence,
            provenance=_projection(2, trusted=True).evidence.provenance,
        ),
    )

    assert projector.finalize_model_influence_dependencies(
        prepared, selected_slot_ids=('E1',), scope=_scope(),
        fence=_fence(hidden_hmac=_hidden().membership_hmac),
    ) == ()


def test_real_ordered_search_result_projection_has_exact_golden_hmac() -> None:
    rows = (_projection(1), _projection(2))
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row for row in rows}),
    )
    projection = projector.project_search(
        rank_evidence_slots(
            (_candidate(rows[0], 0.91), _candidate(rows[1], 0.73))
        ),
        scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )

    assert len(projection.search_results) == 2
    assert projection.projection_hmac == (
        '1ee1340576bbe8a328103cd505f17dab5fc6fb3eb4afb2f4487005663e75ee0c'
    )


def test_hidden_membership_is_derived_from_actual_count_and_exact_retained_prefix() -> None:
    uncapped = _hidden(members=('a' * 64, 'b' * 64))
    full_hidden = tuple(f'{value:x}'.zfill(64) for value in range(21))
    capped = _hidden(members=full_hidden)

    assert (uncapped.public_hidden_count, uncapped.capped) == (2, False)
    assert (capped.public_hidden_count, capped.capped) == (20, True)
    assert capped.actual_hidden_count == 21
    assert not hasattr(capped, 'denied_ids')
    shifted_last_twenty = _hidden(members=full_hidden[1:])
    assert shifted_last_twenty.membership_hmac != capped.membership_hmac
    with pytest.raises(TypeError):
        derive_hidden_membership(  # type: ignore[call-arg]
            actual_hidden_count=21,
            ordered_hidden_member_identity_hmacs=full_hidden,
            top_candidate_window_hmac='b' * 64,
            settings=_settings(),
        )
    for members in ((True,), ('A' * 64,), tuple('a' * 64 for _ in range(51))):
        with pytest.raises(ValueError):
            _hidden(members=members)  # type: ignore[arg-type]


class _InfrastructureFailingResolver(_Resolver):
    def resolve_projection_candidate_strict(self, *, db, identity, scope):
        raise OSError('database connection lost')


def test_strict_projection_read_failure_is_typed_and_never_returns_redacted_dto() -> None:
    row = _projection(1)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=_InfrastructureFailingResolver({})
    )
    with pytest.raises(CanonicalProjectionInfrastructureError):
        projector.project_search(
            rank_evidence_slots((_candidate(row, 0.9),)), scope=_scope(),
            fence=replace(
                ProjectionFence.generations_only(1, None),
                prepared_hidden_membership_hmac=_hidden().membership_hmac,
                current_hidden_membership_hmac=_hidden().membership_hmac,
            ),
        )


def test_search_rejects_six_visible_slots_instead_of_projecting_eight() -> None:
    rows = tuple(_projection(value) for value in range(1, 7))
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row for row in rows}),
    )
    result = projector.project_search(
        rank_evidence_slots(tuple(_candidate(row, 0.9) for row in rows)),
        scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )
    assert result.search_results == ()


@pytest.mark.parametrize('mutation', ['raw_support', 'raw_type', 'raw_provenance', 'trusted_support', 'trusted_id', 'trusted_type'])
def test_exact_raw_and_trusted_branch_mappings_fail_closed(mutation: str) -> None:
    row = _projection(2, trusted=mutation.startswith('trusted'))
    evidence = row.evidence
    if mutation in {'raw_support', 'trusted_support'}:
        evidence = replace(evidence, support_mode='trusted_fact' if mutation == 'raw_support' else 'source_observation')
    elif mutation == 'raw_type':
        evidence = replace(evidence, public_source_type='slack')
    elif mutation == 'raw_provenance':
        evidence = replace(evidence, provenance=_projection(2, trusted=True).evidence.provenance)
    elif mutation == 'trusted_id':
        evidence = replace(evidence, public_source_id='forged')
    else:
        evidence = replace(evidence, public_source_type='todo')
    current = replace(row, evidence=evidence)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: current}),
    )
    result = projector.project_selected(
        rank_evidence_slots((_candidate(row, 0.9),)), selected_slot_ids=('E1',),
        scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )
    assert result.citations == ()


@pytest.mark.parametrize(
    'mutation',
    [
        'raw_envelope_source', 'raw_envelope_serving', 'raw_kind_subclass',
        'trusted_envelope_serving', 'trusted_approval',
    ],
)
def test_self_consistent_corrupt_cross_field_authority_is_rejected(mutation: str) -> None:
    trusted = mutation.startswith('trusted')
    row = _projection(2, trusted=trusted)
    evidence = row.evidence
    identity = row.identity
    if mutation == 'trusted_approval':
        current = replace(row, approval_provenance_hmac='f' * 64)
    elif mutation == 'raw_kind_subclass':
        evidence = replace(evidence, serving_kind=_SlotStringSubclass('raw_chunk'))
        identity = replace(identity, serving_kind=_SlotStringSubclass('raw_chunk'))
        current = replace(row, identity=identity, evidence=evidence)
    else:
        envelope = evidence.version_envelope
        if mutation == 'raw_envelope_source':
            envelope = replace(envelope, public_source_id='gmail:forged')
        else:
            envelope = replace(envelope, serving_document_id='forged:serving')
        version_hmac = build_serving_version_fingerprint(envelope, settings=_settings())
        identity = replace(
            identity,
            serving_version_fingerprint=version_hmac,
            version_envelope=envelope,
        )
        evidence = replace(
            evidence,
            serving_version_fingerprint=version_hmac,
            version_envelope=envelope,
            provenance=(
                replace(evidence.provenance, raw_version=envelope)
                if not trusted
                else evidence.provenance
            ),
        )
        if trusted:
            evidence = replace(evidence, provenance=envelope.provenance)
        current = replace(row, identity=identity, evidence=evidence)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: current}),
    )

    result = projector.project_selected(
        rank_evidence_slots((_candidate(row, 0.9),)), selected_slot_ids=('E1',),
        scope=_scope(),
        fence=replace(
            ProjectionFence.generations_only(1, None),
            prepared_hidden_membership_hmac=_hidden().membership_hmac,
            current_hidden_membership_hmac=_hidden().membership_hmac,
        ),
    )
    assert result.citations == ()


@pytest.mark.parametrize(
    'mutation',
    ['support_subclass', 'raw_third_branch', 'raw_canonical_id', 'trusted_canonical_id'],
)
def test_branch_literals_and_canonical_serving_ids_are_exact(mutation: str) -> None:
    trusted = mutation.startswith('trusted')
    row = _projection(1, trusted=trusted)
    evidence = row.evidence
    identity = row.identity
    envelope = evidence.version_envelope
    if mutation == 'support_subclass':
        evidence = replace(
            evidence, support_mode=_SlotStringSubclass('source_observation')
        )
    elif mutation == 'raw_third_branch':
        evidence = replace(
            evidence,
            provenance=replace(evidence.provenance, branch='third_branch'),
        )
    else:
        forged_id = 'forged:1'
        forged_citation_hmac = (
            build_canonical_citation_projection_hmac(
                public_source_id=forged_id,
                public_source_type=evidence.public_source_type,
                source_url=row.source_url,
                source_snippet=row.source_snippet,
                effective_permission=evidence.effective_permission,
                settings=_settings(),
            )
            if trusted
            else evidence.canonical_citation_projection_hmac
        )
        envelope = replace(envelope, serving_document_id=forged_id)
        if trusted:
            envelope = replace(
                envelope,
                canonical_citation_projection_hmac=forged_citation_hmac,
            )
        evidence = replace(
            evidence,
            serving_document_id=forged_id,
            public_source_id=(forged_id if trusted else evidence.public_source_id),
            canonical_citation_projection_hmac=forged_citation_hmac,
        )
        identity = replace(
            identity,
            serving_document_id=forged_id,
            public_source_id=(forged_id if trusted else identity.public_source_id),
            canonical_citation_projection_hmac=forged_citation_hmac,
        )
    if envelope is not evidence.version_envelope:
        version_hmac = build_serving_version_fingerprint(envelope, settings=_settings())
        evidence = replace(
            evidence,
            serving_version_fingerprint=version_hmac,
            version_envelope=envelope,
            provenance=(
                envelope.provenance
                if trusted
                else replace(evidence.provenance, raw_version=envelope)
            ),
            serving_identity_hmac=build_serving_identity_hmac(
                serving_document_id=evidence.serving_document_id,
                serving_kind=evidence.serving_kind,
                settings=_settings(),
            ),
        )
        identity = replace(
            identity,
            serving_version_fingerprint=version_hmac,
            version_envelope=envelope,
        )
    current = replace(row, identity=identity, evidence=evidence)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({current.evidence.serving_document_id: current}),
    )
    slots = rank_evidence_slots((_candidate(current, 0.9),))
    if mutation == 'support_subclass':
        slots = (replace(slots[0], support_mode='source_observation'),)
    result = projector.project_selected(
        slots,
        selected_slot_ids=('E1',), scope=_scope(),
        fence=_fence(hidden_hmac=_hidden().membership_hmac),
    )
    assert result.citations == ()


@pytest.mark.parametrize(
    'row',
    [
        _explicit_projection(selected_source_row_id=2),
        _explicit_projection(selected_source_type='drive'),
        _explicit_projection(selected_version='revision-2'),
        _explicit_projection(canonical_source_id='01'),
    ],
    ids=['source-row', 'source-type', 'version', 'noncanonical-source-id'],
)
def test_explicit_selected_child_must_match_exactly_one_evidence_link(
    row: CanonicalServingProjection,
) -> None:
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row}),
    )
    result = projector.project_selected(
        rank_evidence_slots((_candidate(row, 0.9),)),
        selected_slot_ids=('E1',), scope=_scope(),
        fence=_fence(hidden_hmac=_hidden().membership_hmac),
    )
    assert result.citations == ()


def test_explicit_selected_child_missing_or_duplicate_link_fails_closed() -> None:
    row = _explicit_projection()
    envelope = row.evidence.version_envelope
    provenance = envelope.provenance
    assert type(provenance) is ExplicitApprovalProvenance
    missing = replace(
        provenance,
        evidence_links=(
            replace(
                provenance.evidence_links[0],
                trusted_knowledge_evidence_link_id=102,
            ),
        ),
    )
    with pytest.raises(ValueError, match='outside the evidence set'):
        build_serving_version_fingerprint(
            replace(envelope, provenance=missing), settings=_settings()
        )

    duplicate = replace(
        provenance.evidence_links[0],
        canonical_source_id='2',
    )
    duplicate_provenance = replace(
        provenance,
        evidence_links=(provenance.evidence_links[0], duplicate),
    )
    duplicate_envelope = replace(envelope, provenance=duplicate_provenance)
    version_hmac = build_serving_version_fingerprint(
        duplicate_envelope, settings=_settings()
    )
    duplicate_row = replace(
        row,
        identity=replace(
            row.identity,
            serving_version_fingerprint=version_hmac,
            version_envelope=duplicate_envelope,
        ),
        evidence=replace(
            row.evidence,
            serving_version_fingerprint=version_hmac,
            version_envelope=duplicate_envelope,
            provenance=duplicate_provenance,
        ),
        evidence_link_hmacs=('e' * 64, 'f' * 64),
    )
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver(
            {duplicate_row.evidence.serving_document_id: duplicate_row}
        ),
    )
    assert projector.project_selected(
        rank_evidence_slots((_candidate(duplicate_row, 0.9),)),
        selected_slot_ids=('E1',), scope=_scope(),
        fence=_fence(hidden_hmac=_hidden().membership_hmac),
    ).citations == ()


class _SlotStringSubclass(str):
    pass


class _IdentityIntSubclass(int):
    pass


class _IdentityEqualityImpostor:
    def __init__(self, expected: object) -> None:
        self._expected = expected

    def __eq__(self, other: object) -> bool:
        return other == self._expected


class _SlotEqualityImpostor:
    def __hash__(self) -> int:
        return hash('E1')

    def __eq__(self, other: object) -> bool:
        return other == 'E1'


@pytest.mark.parametrize('selected', [(_SlotStringSubclass('E1'),), (_SlotEqualityImpostor(),), (1,)])
def test_selected_slot_ids_require_exact_builtin_literal_strings(selected: tuple[object, ...]) -> None:
    row = _projection(1)
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(),
        resolver=_Resolver({row.evidence.serving_document_id: row}),
    )
    result = projector.project_selected(
        rank_evidence_slots((_candidate(row, 0.9),)),
        selected_slot_ids=selected,  # type: ignore[arg-type]
        scope=_scope(), fence=_fence(hidden_hmac=_hidden().membership_hmac),
    )
    assert result.citations == ()


_IDENTITY_STRING_FIELDS = (
    'serving_document_id',
    'serving_kind',
    'public_source_id',
    'public_source_type',
    'effective_permission',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'serving_version_fingerprint',
)
_RAW_ENVELOPE_STRING_FIELDS = (
    'serving_document_id',
    'public_source_id',
    'external_revision',
    'server_content_signature_schema',
    'server_content_signature',
    'parser_policy_version',
    'parser_version',
    'chunk_policy_version',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'effective_permission',
)
_RAW_ENVELOPE_INT_FIELDS = (
    'source_row_id',
    'document_id',
    'document_version_id',
    'current_document_version_id',
    'document_chunk_id',
    'parser_run_id',
)
_TRUSTED_ENVELOPE_STRING_FIELDS = (
    'serving_document_id',
    'knowledge_type',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'effective_permission',
)


@pytest.mark.parametrize('trusted', [False, True], ids=['raw', 'trusted'])
@pytest.mark.parametrize('field', _IDENTITY_STRING_FIELDS)
def test_serving_identity_requires_exact_builtin_string_constituents(
    trusted: bool, field: str
) -> None:
    row = _projection(1, trusted=trusted)
    original = getattr(row.identity, field)
    assert type(original) is str
    current = replace(
        row,
        identity=replace(
            row.identity,
            **{field: _SlotStringSubclass(original)},
        ),
    )

    assert _project_single_row(current).citations == ()


@pytest.mark.parametrize('field', _RAW_ENVELOPE_STRING_FIELDS)
def test_raw_identity_envelope_requires_exact_builtin_string_constituents(
    field: str,
) -> None:
    row = _projection(1)
    envelope = row.identity.version_envelope
    assert type(envelope) is RawServingVersionEnvelope
    original = getattr(envelope, field)
    assert type(original) is str
    current = replace(
        row,
        identity=replace(
            row.identity,
            version_envelope=replace(
                envelope,
                **{field: _SlotStringSubclass(original)},
            ),
        ),
    )

    assert _project_single_row(current).citations == ()


@pytest.mark.parametrize('field', _RAW_ENVELOPE_INT_FIELDS)
def test_raw_identity_envelope_requires_exact_builtin_integer_constituents(
    field: str,
) -> None:
    row = _projection(1)
    envelope = row.identity.version_envelope
    assert type(envelope) is RawServingVersionEnvelope
    original = getattr(envelope, field)
    assert type(original) is int
    current = replace(
        row,
        identity=replace(
            row.identity,
            version_envelope=replace(
                envelope,
                **{field: _IdentityIntSubclass(original)},
            ),
        ),
    )

    assert _project_single_row(current).citations == ()


@pytest.mark.parametrize('field', _TRUSTED_ENVELOPE_STRING_FIELDS)
def test_trusted_identity_envelope_requires_exact_builtin_string_constituents(
    field: str,
) -> None:
    row = _projection(1, trusted=True)
    envelope = row.identity.version_envelope
    assert type(envelope) is TrustedServingVersionEnvelope
    original = getattr(envelope, field)
    assert type(original) is str
    current = replace(
        row,
        identity=replace(
            row.identity,
            version_envelope=replace(
                envelope,
                **{field: _SlotStringSubclass(original)},
            ),
        ),
    )

    assert _project_single_row(current).citations == ()


@pytest.mark.parametrize(
    ('trusted', 'field'),
    [(False, 'document_chunk_id'), (True, 'knowledge_id')],
    ids=['raw-bool', 'trusted-bool'],
)
def test_identity_envelope_rejects_bool_integer_impostors(
    trusted: bool, field: str
) -> None:
    row = _projection(1, trusted=trusted)
    envelope = row.identity.version_envelope
    current = replace(
        row,
        identity=replace(
            row.identity,
            version_envelope=replace(envelope, **{field: True}),
        ),
    )

    assert _project_single_row(current).citations == ()


def test_serving_identity_rejects_equality_impostor_with_matching_hmac_bytes() -> None:
    row = _projection(1, trusted=True)
    current = replace(
        row,
        identity=replace(
            row.identity,
            serving_kind=_IdentityEqualityImpostor('trusted_knowledge'),
        ),
    )

    assert _project_single_row(current).citations == ()


def test_prepare_rejects_zero_observations() -> None:
    projector = CanonicalEvidenceProjector(
        db=_Transaction(), settings=_settings(), resolver=_Resolver({})
    )
    with pytest.raises(ValueError, match='one to eight'):
        projector.prepare_model_influence(
            (), scope=_scope(), prepared_corpus_generation=7,
            prepared_index_generation=4, prepared_readiness_hmac='a' * 64,
            rendered_input_hmac='c' * 64,
        )


class _TransactionEndingResolver(_Resolver):
    def resolve_projection_candidate_strict(self, *, db, identity, scope):
        row = super().resolve_projection_candidate_strict(
            db=db, identity=identity, scope=scope
        )
        db.active = False
        return row


class _TransactionReplacingResolver(_Resolver):
    def resolve_projection_candidate_strict(self, *, db, identity, scope):
        row = super().resolve_projection_candidate_strict(
            db=db, identity=identity, scope=scope
        )
        db.token = object()
        return row


def test_resolver_ending_transaction_mid_read_raises_typed_failure_without_dto() -> None:
    row = _projection(1)
    transaction = _Transaction()
    projector = CanonicalEvidenceProjector(
        db=transaction, settings=_settings(),
        resolver=_TransactionEndingResolver({row.evidence.serving_document_id: row}),
    )
    with pytest.raises(CanonicalProjectionTransactionError):
        projector.project_search(
            rank_evidence_slots((_candidate(row, 0.9),)), scope=_scope(),
            fence=replace(
                ProjectionFence.generations_only(1, None),
                prepared_hidden_membership_hmac=_hidden().membership_hmac,
                current_hidden_membership_hmac=_hidden().membership_hmac,
            ),
        )


def test_resolver_replacing_transaction_mid_read_raises_typed_failure() -> None:
    row = _projection(1)
    transaction = _Transaction()
    projector = CanonicalEvidenceProjector(
        db=transaction, settings=_settings(),
        resolver=_TransactionReplacingResolver({row.evidence.serving_document_id: row}),
    )
    with pytest.raises(CanonicalProjectionTransactionError):
        projector.project_selected(
            rank_evidence_slots((_candidate(row, 0.9),)),
            selected_slot_ids=('E1',), scope=_scope(),
            fence=_fence(hidden_hmac=_hidden().membership_hmac),
        )
