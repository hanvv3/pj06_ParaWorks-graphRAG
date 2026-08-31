from __future__ import annotations

import operator
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
    FrozenDict,
    HiddenMembershipSnapshot,
    PreparedModelInfluenceSet,
    ProjectionFence,
    build_model_influence_dependency_hmac,
    build_model_influence_set_hmac,
    build_prepared_model_influence_observation_hmac,
    build_v1_citation_projection_hmac,
    build_v1_search_result_set_projection_hmac,
    build_v1_selected_evidence_projection_hmac,
    derive_hidden_membership,
)
from backend.app.rag.retrieval import RetrievalCandidate, rank_evidence_slots
from backend.app.rag.serving_contracts import (
    CanonicalServingProjection,
    LegacyHumanProvenance,
    RawChunkProvenance,
    RawServingVersionEnvelope,
    ServingEvidence,
    ServingEvidenceIdentity,
    TrustedServingVersionEnvelope,
    build_approval_provenance_hmac,
    build_canonical_citation_projection_hmac,
    build_model_content_hmac,
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


class _Transaction:
    def __init__(self) -> None:
        self.active = True

    def in_transaction(self) -> bool:
        return self.active


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


def _hidden(*, actual: int = 0, members: tuple[str, ...] = ()) -> HiddenMembershipSnapshot:
    return derive_hidden_membership(
        actual_hidden_count=actual,
        ordered_retained_member_identity_hmacs=members,
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
    actual = 21 if capped else public_hidden_count
    snapshot = derive_hidden_membership(
        actual_hidden_count=actual,
        ordered_retained_member_identity_hmacs=denied_known_member_identity_hmacs,
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
        'dd9a269e31b72099f0ef9294b48cfe278cf59ec4eec1422aa139d3caca08bf9c',
        '11fa414024896800554cb9c267ace86075fe82f2b081b16cceed0d63273b1a36',
        'b00a621382c0f5020bfdc82c4d116bbb09e756fd4e53d8a957280bf266293156',
        'c77e5e82d672ece0d4265298dd32e81468ccc3e997fd3adeaf2a39f38d8975c7',
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
    prepared_null = prepared_set.aggregate_observation_hmac
    present_observations = (
        replace(prepared_set.observations[0], approval_provenance_hmac='e' * 64),
        prepared_set.observations[1],
    )
    prepared_present = build_prepared_model_influence_observation_hmac(
        observations=present_observations,
        prepared_corpus_generation=1,
        prepared_index_generation=None,
        prepared_readiness_hmac=None,
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
        actual=21,
        members=tuple(f'{value:x}'.zfill(64) for value in range(20)),
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
        with pytest.raises(TypeError):
            operation()
    assert projection.projection_hmac == original_hmac
    assert result['source_id'] == row.evidence.public_source_id


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
    uncapped = _hidden(actual=2, members=('a' * 64, 'b' * 64))
    capped = _hidden(actual=21, members=tuple(f'{value:x}'.zfill(64) for value in range(20)))

    assert (uncapped.public_hidden_count, uncapped.capped) == (2, False)
    assert (capped.public_hidden_count, capped.capped) == (20, True)
    assert not hasattr(capped, 'denied_ids')
    for actual, members in ((1, ()), (19, tuple('a' * 64 for _ in range(18))), (21, tuple('a' * 64 for _ in range(19))), (0, ('a' * 64,))):
        with pytest.raises(ValueError):
            _hidden(actual=actual, members=members)


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
