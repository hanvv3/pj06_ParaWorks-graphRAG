from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.models import (
    AutoReviewValidation,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)
from backend.app.rag.serving_contracts import (
    ExplicitApprovalProvenance,
    LegacyHumanProvenance,
    RawServingVersionEnvelope,
    TrustedServingVersionEnvelope,
    build_approval_provenance_hmac,
    build_canonical_citation_projection_hmac,
    build_evidence_link_set_hmac,
    build_legacy_evidence_pairs_hmac,
    build_selected_citation_child_hmac,
    build_serving_version_fingerprint,
)
from backend.app.rag.trusted_evidence import (
    ServingEvidenceResolver,
    TrustedEvidenceAuthorizer,
    TrustedServingEnvelopeResolver,
)


def _settings(*, secret: str = 'task-4-trusted-secret-at-least-32-bytes') -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret=secret,
        agent_runtime_fingerprint_key_version='task4-trusted-v1',
    )


def _scope(
    *,
    source_ids: tuple[int, ...] = (),
    project_keys: tuple[str, ...] = (),
    permissions: tuple[str, ...] = ('public', 'internal', 'restricted'),
    workspace: str = 'workspace-a',
) -> SecurityScope:
    source_constraints = tuple(f'source_pk:{value}' for value in source_ids)
    project_constraints = tuple(f'project_key:{value}' for value in project_keys)
    constrained = bool(source_constraints or project_constraints)
    return SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='trusted-actor',
        workspace_scope_id=workspace,
        resource_scope_mode='constrained' if constrained else 'all_current_scope',
        project_constraints=project_constraints,
        source_constraints=source_constraints,
        allowed_permission_levels=permissions,
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )


def _seed_source(
    db: Session,
    *,
    ordinal: int,
    permission: str = 'internal',
) -> tuple[Source, DocumentChunk]:
    signature = f'{ordinal:x}'.zfill(64)
    text = f'Exact evidence number {ordinal}'
    source = Source(
        source_type='gmail',
        source_id=f'gmail:trusted-{ordinal}',
        source_url=f'https://mail.example.test/messages/{ordinal}',
        title=f'Source {ordinal}',
        author='owner@example.test',
        permission_level=permission,
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db.add(source)
    db.flush()
    document = Document(source_id=source.id, title=source.title, current_version='v1')
    db.add(document)
    db.flush()
    version = DocumentVersion(document_id=document.id, version='v1', body=text)
    db.add(version)
    db.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id=f'revision-{ordinal}',
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        parser_run_id=parser_run.id,
        chunk_index=0,
        text=text,
        source_snippet=text,
        permission_level=permission,
        metadata_={},
    )
    db.add(chunk)
    document.current_document_version_id = version.id
    db.flush()
    return source, chunk


def _seed_item(
    db: Session,
    *,
    source: Source,
    chunk: DocumentChunk,
    resolution_source: str,
    permission: str | None = None,
    pair_prefix: bool = False,
) -> ReviewItem:
    links = ['https://mail.example.test/messages/unselected'] if pair_prefix else []
    snippets = ['Different exact evidence'] if pair_prefix else []
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Trusted history'},
        source_links=[*links, source.source_url],
        source_snippets=[*snippets, chunk.source_snippet],
        confidence_score=0.99,
        permission_level=permission or source.permission_level,
        status='approved',
        candidate_contract_version='c5-v1',
        resolution_source=resolution_source,
        resolution_policy_version='auto-review-policy:v1',
        workflow_thread_id=(
            f'workflow-{source.id}' if resolution_source == 'auto_policy' else None
        ),
    )
    db.add(item)
    db.flush()
    if resolution_source == 'auto_policy':
        validation = AutoReviewValidation(
            review_item_id=item.id,
            validation_call_id=10_000 + item.id,
            workflow_thread_id=item.workflow_thread_id,
            validation_key=f'validation-{item.id}',
            evidence_version_hash='a' * 64,
            candidate_generation_fingerprint='b' * 64,
            status='completed',
            validator_provider='openai',
            validator_model='gpt-test',
            reasoning_effort='medium',
            validator_prompt_version='auto-review-validation:v2',
            validator_output_contract_version='candidate-validation-batch:v1',
            policy_version='auto-review-policy:v1',
            fingerprint_key_version='task4-trusted-v1',
            fingerprint_key_material_verifier='c' * 64,
            cost_policy_version='auto-review-cost:v1',
            confirmed_validation_cost_ceiling_usd=Decimal('0.01'),
            claim_results=[],
            minimum_entailment_score=Decimal('0.99'),
            uncertainty_codes=[],
            conflict_codes=[],
            policy_decision='auto_approve',
            policy_reason_codes=['direct_fact_supported'],
            input_tokens=1,
            output_tokens=1,
            estimated_cost_usd=Decimal('0'),
            cache_hit=False,
        )
        db.add(validation)
        db.flush()
        item.auto_validation_id = validation.id
    return item


def _seed_link(
    db: Session,
    *,
    target: HistoryEvent,
    item: ReviewItem,
    source: Source,
    resolution_source: str,
    permission: str | None = None,
) -> TrustedKnowledgeApprovalLink:
    settings = _settings()
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    link = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=target.id,
        review_item_id=item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='primary',
        resolution_source=resolution_source,
        claim_fingerprint=f'{item.id:x}'.zfill(64),
        permission_level=permission or item.permission_level,
        fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
        fingerprint_key_material_verifier=verifier,
        active=True,
    )
    db.add(link)
    db.flush()
    db.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=link.id,
            canonical_source_kind=source.source_type,
            canonical_source_id=str(source.id),
            canonical_version_or_signature=source.server_content_signature,
            evidence_hash=f'{100 + item.id:x}'.zfill(64),
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=verifier,
        )
    )
    db.flush()
    return link


def _seed_target(
    db: Session,
    *,
    source_review_item_id: int | None = None,
    permission: str = 'internal',
) -> HistoryEvent:
    target = HistoryEvent(
        project_key='project-a',
        title='Trusted history',
        reason='Canonical approved knowledge',
        source_links=[],
        source_snippets=[],
        confidence_score=0.99,
        permission_level=permission,
        review_status='approved',
        source_review_item_id=source_review_item_id,
    )
    db.add(target)
    db.flush()
    return target


def test_scope_aware_selection_authorizes_every_child_before_human_auto_priority(
    db_session: Session,
) -> None:
    auto_source, auto_chunk = _seed_source(db_session, ordinal=1)
    human_source, human_chunk = _seed_source(
        db_session,
        ordinal=2,
        permission='restricted',
    )
    target = _seed_target(db_session, permission='public')
    auto_item = _seed_item(
        db_session,
        source=auto_source,
        chunk=auto_chunk,
        resolution_source='auto_policy',
    )
    human_item = _seed_item(
        db_session,
        source=human_source,
        chunk=human_chunk,
        resolution_source='human',
        permission='restricted',
    )
    auto_link = _seed_link(
        db_session,
        target=target,
        item=auto_item,
        source=auto_source,
        resolution_source='auto_policy',
    )
    human_link = _seed_link(
        db_session,
        target=target,
        item=human_item,
        source=human_source,
        resolution_source='human',
        permission='restricted',
    )
    db_session.commit()

    resolver = ServingEvidenceResolver(settings=_settings())
    actor_limited = resolver.resolve_trusted_candidate(
        db=db_session,
        knowledge_type='history_event',
        knowledge_id=target.id,
        scope=_scope(
            source_ids=(auto_source.id,),
            project_keys=('project-a',),
            permissions=('public', 'internal', 'restricted'),
        ),
    )
    all_scope = resolver.resolve_trusted_candidate(
        db=db_session,
        knowledge_type='history_event',
        knowledge_id=target.id,
        scope=_scope(),
    )

    assert actor_limited is not None
    assert isinstance(actor_limited.provenance, ExplicitApprovalProvenance)
    assert actor_limited.provenance.approval_link_id == auto_link.id
    assert actor_limited.provenance.approval_link_id != human_link.id
    assert actor_limited.provenance.evidence_links[0].canonical_source_id == str(
        auto_source.id
    )
    assert all(
        identity.canonical_source_id != str(human_source.id)
        for identity in actor_limited.provenance.evidence_links
    )
    assert actor_limited.effective_permission == 'restricted'
    assert actor_limited.public_source_type == 'history_event'
    assert actor_limited.public_source_id == f'history_event:{target.id}'
    assert all_scope is not None
    assert isinstance(all_scope.provenance, ExplicitApprovalProvenance)
    assert all_scope.provenance.approval_link_id == human_link.id


def test_explicit_supporting_link_requires_every_child_in_actor_source_scope(
    db_session: Session,
) -> None:
    source_a, chunk_a = _seed_source(db_session, ordinal=1)
    source_b, chunk_b = _seed_source(db_session, ordinal=2)
    target = _seed_target(db_session)
    item = _seed_item(
        db_session,
        source=source_a,
        chunk=chunk_a,
        resolution_source='human',
    )
    item.source_links.append(source_b.source_url)
    item.source_snippets.append(chunk_b.source_snippet)
    link = _seed_link(
        db_session,
        target=target,
        item=item,
        source=source_a,
        resolution_source='human',
    )
    settings = _settings()
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=link.id,
            canonical_source_kind=source_b.source_type,
            canonical_source_id=str(source_b.id),
            canonical_version_or_signature=source_b.server_content_signature,
            evidence_hash='f' * 64,
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
        )
    )
    db_session.commit()
    resolver = ServingEvidenceResolver(settings=settings)

    partial_scope = resolver.resolve_trusted_candidate(
        db=db_session,
        knowledge_type='history_event',
        knowledge_id=target.id,
        scope=_scope(source_ids=(source_a.id,)),
    )
    complete_scope = resolver.resolve_trusted_candidate(
        db=db_session,
        knowledge_type='history_event',
        knowledge_id=target.id,
        scope=_scope(source_ids=(source_a.id, source_b.id)),
    )

    assert partial_scope is None
    assert complete_scope is not None
    assert isinstance(complete_scope.provenance, ExplicitApprovalProvenance)
    assert tuple(
        identity.canonical_source_id
        for identity in complete_scope.provenance.evidence_links
    ) == (str(source_a.id), str(source_b.id))


def test_any_stale_active_link_makes_the_whole_trusted_candidate_ineligible(
    db_session: Session,
) -> None:
    source_a, chunk_a = _seed_source(db_session, ordinal=1)
    source_b, chunk_b = _seed_source(db_session, ordinal=2)
    target = _seed_target(db_session)
    item_a = _seed_item(
        db_session,
        source=source_a,
        chunk=chunk_a,
        resolution_source='human',
    )
    item_b = _seed_item(
        db_session,
        source=source_b,
        chunk=chunk_b,
        resolution_source='human',
    )
    _seed_link(
        db_session,
        target=target,
        item=item_a,
        source=source_a,
        resolution_source='human',
    )
    stale_link = _seed_link(
        db_session,
        target=target,
        item=item_b,
        source=source_b,
        resolution_source='human',
    )
    stale_child = next(
        child
        for child in db_session.query(TrustedKnowledgeEvidenceLink).all()
        if child.approval_link_id == stale_link.id
    )
    stale_child.canonical_version_or_signature = 'f' * 64
    db_session.commit()

    resolved = TrustedServingEnvelopeResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index('history_event', target.id)

    assert resolved is None


def test_explicit_envelope_orders_links_and_binds_lowest_exact_review_pair(
    db_session: Session,
) -> None:
    source, chunk = _seed_source(db_session, ordinal=1)
    target = _seed_target(db_session)
    item = _seed_item(
        db_session,
        source=source,
        chunk=chunk,
        resolution_source='human',
        pair_prefix=True,
    )
    link = _seed_link(
        db_session,
        target=target,
        item=item,
        source=source,
        resolution_source='human',
    )
    db_session.commit()

    envelope = TrustedServingEnvelopeResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index('history_event', target.id)

    assert envelope is not None
    assert envelope.identity.serving_document_id == f'history_event:{target.id}'
    assert envelope.identity.serving_kind == 'trusted_knowledge'
    assert envelope.identity.public_source_id == f'history_event:{target.id}'
    assert envelope.identity.public_source_type == 'history_event'
    assert isinstance(envelope.identity.version_envelope, TrustedServingVersionEnvelope)
    assert envelope.identity.version_envelope is envelope.trusted_version
    assert envelope.evidence.version_envelope is envelope.trusted_version
    assert isinstance(envelope.evidence.provenance, ExplicitApprovalProvenance)
    assert envelope.evidence.provenance.approval_link_id == link.id
    assert envelope.evidence.provenance.selected_citation_child.source_row_id == source.id
    assert (
        envelope.evidence.provenance.selected_citation_child.review_item_source_pair_ordinal
        == 1
    )
    assert envelope.evidence.support_mode == 'trusted_fact'


def test_explicit_branch_rejects_review_pair_or_current_source_drift(
    db_session: Session,
) -> None:
    source, chunk = _seed_source(db_session, ordinal=1)
    target = _seed_target(db_session)
    item = _seed_item(
        db_session,
        source=source,
        chunk=chunk,
        resolution_source='human',
    )
    _seed_link(
        db_session,
        target=target,
        item=item,
        source=source,
        resolution_source='human',
    )
    item.source_snippets[0] = 'drifted snippet bytes'
    db_session.commit()

    resolved = TrustedServingEnvelopeResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index('history_event', target.id)

    assert resolved is None


def test_trusted_authorizer_classifies_project_source_and_permission_independently(
    db_session: Session,
) -> None:
    source, chunk = _seed_source(db_session, ordinal=1, permission='restricted')
    target = _seed_target(db_session, permission='public')
    item = _seed_item(
        db_session,
        source=source,
        chunk=chunk,
        resolution_source='human',
    )
    _seed_link(
        db_session,
        target=target,
        item=item,
        source=source,
        resolution_source='human',
    )
    db_session.commit()
    envelope = TrustedServingEnvelopeResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index('history_event', target.id)
    assert envelope is not None
    authorizer = TrustedEvidenceAuthorizer(db=db_session)

    visible = authorizer.classify_access(
        _scope(
            source_ids=(source.id,),
            project_keys=('project-a',),
            permissions=('restricted',),
        ),
        envelope,
    )
    denied = authorizer.classify_access(
        _scope(
            source_ids=(source.id,),
            project_keys=('project-a',),
            permissions=('public', 'internal'),
        ),
        envelope,
    )
    outside = authorizer.classify_access(
        _scope(source_ids=(source.id + 1,), permissions=('restricted',)),
        envelope,
    )

    assert visible.resource_scope == 'in_scope'
    assert visible.permission_visibility == 'visible'
    assert denied.resource_scope == 'in_scope'
    assert denied.permission_visibility == 'denied_known'
    assert outside.resource_scope == 'out_of_scope'


def test_legacy_human_branch_requires_one_exact_linked_review_item(
    db_session: Session,
) -> None:
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Legacy'},
        source_links=['https://legacy.example.test/evidence'],
        source_snippets=['Exact legacy evidence'],
        confidence_score=0.9,
        permission_level='restricted',
        status='approved',
        candidate_contract_version=None,
        resolution_source=None,
    )
    db_session.add(item)
    db_session.flush()
    target = HistoryEvent(
        project_key='project-a',
        title='Legacy history',
        reason='Exact legacy evidence',
        source_links=list(item.source_links),
        source_snippets=list(item.source_snippets),
        confidence_score=0.9,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=item.id,
    )
    db_session.add(target)
    db_session.commit()

    envelope = TrustedServingEnvelopeResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index('history_event', target.id)

    assert envelope is not None
    assert isinstance(envelope.evidence.provenance, LegacyHumanProvenance)
    assert envelope.evidence.provenance.legacy_source_review_item_id == item.id
    assert envelope.evidence.effective_permission == 'restricted'
    assert envelope.evidence.provenance.legacy_evidence_pairs_hmac == (
        build_legacy_evidence_pairs_hmac(
            knowledge_type='history_event',
            knowledge_id=target.id,
            knowledge_review_status='approved',
            knowledge_permission='internal',
            review_item=item,
            settings=_settings(),
        )
    )


def test_legacy_unbound_c5_or_mismatched_arrays_are_not_d_serving_evidence(
    db_session: Session,
) -> None:
    unbound = _seed_target(db_session, source_review_item_id=None)
    source, chunk = _seed_source(db_session, ordinal=1)
    c5_item = _seed_item(
        db_session,
        source=source,
        chunk=chunk,
        resolution_source='human',
    )
    c5_target = _seed_target(db_session, source_review_item_id=c5_item.id)
    legacy_item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Legacy mismatch'},
        source_links=['https://legacy.example.test/evidence'],
        source_snippets=['snippet'],
        confidence_score=0.9,
        permission_level='internal',
        status='approved',
        candidate_contract_version=None,
        resolution_source='human',
    )
    db_session.add(legacy_item)
    db_session.flush()
    mismatch = _seed_target(db_session, source_review_item_id=legacy_item.id)
    mismatch.source_links = list(legacy_item.source_links)
    mismatch.source_snippets = ['different']
    db_session.commit()
    resolver = TrustedServingEnvelopeResolver(db=db_session, settings=_settings())

    assert resolver.resolve_for_index('history_event', unbound.id) is None
    assert resolver.resolve_for_index('history_event', c5_target.id) is None
    assert resolver.resolve_for_index('history_event', mismatch.id) is None


def test_decision_storage_alias_is_canonicalized_before_public_and_internal_identity(
    db_session: Session,
) -> None:
    source, chunk = _seed_source(db_session, ordinal=1)
    target = _seed_target(db_session)
    item = _seed_item(
        db_session,
        source=source,
        chunk=chunk,
        resolution_source='human',
    )
    link = _seed_link(
        db_session,
        target=target,
        item=item,
        source=source,
        resolution_source='human',
    )
    link.knowledge_type = 'decision'
    target_id = target.id
    from backend.app.models import DecisionRecord

    db_session.delete(target)
    decision = DecisionRecord(
        id=target_id,
        project_key='project-a',
        title='Decision',
        decision_summary='Canonical decision',
        source_links=[],
        source_snippets=[],
        confidence_score=0.9,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=None,
    )
    db_session.add(decision)
    db_session.commit()

    envelope = TrustedServingEnvelopeResolver(
        db=db_session,
        settings=_settings(),
    ).resolve_for_index('decision', decision.id)

    assert envelope is not None
    assert envelope.identity.serving_document_id == f'decision_record:{decision.id}'
    assert envelope.identity.public_source_id == f'decision_record:{decision.id}'
    assert envelope.identity.public_source_type == 'decision_record'


def test_selected_child_and_legacy_provenance_hmacs_have_golden_exact_envelopes() -> None:
    from backend.app.rag.serving_contracts import SelectedCitationChild

    selected = SelectedCitationChild(
        trusted_knowledge_evidence_link_id=7,
        source_row_id=11,
        canonical_source_type='gmail',
        canonical_version_or_signature='가-é-🧭',
        review_item_source_pair_ordinal=2,
    )
    selected_hmac = build_selected_citation_child_hmac(
        selected,
        settings=_settings(),
    )
    provenance_hmac = build_approval_provenance_hmac(
        branch='explicit_approval',
        approval_fingerprint_key_material_verifier='a' * 64,
        approval_fingerprint_key_version='key-é',
        approval_link_id=5,
        approval_permission='restricted',
        claim_fingerprint='b' * 64,
        evidence_link_set_hmac='c' * 64,
        legacy_evidence_pairs_hmac=None,
        promotion_effect_kind='primary',
        resolution_source='human',
        review_item_id=3,
        security_scope_id='workspace-🧭',
        selected_citation_child_hmac=selected_hmac,
        settings=_settings(),
    )

    assert selected_hmac == 'a9f66891c1a865806fb54ce06ffac798caacd173b3c793e4019967229c983b5b'
    assert provenance_hmac == '9798e9d0c8cc7df78dcd593cdcf234469b2babdd5ab7b84c46016e176def03dd'
    assert build_selected_citation_child_hmac(
        replace(selected, review_item_source_pair_ordinal=3),
        settings=_settings(),
    ) != selected_hmac
    with pytest.raises(ValueError, match='incomplete'):
        build_approval_provenance_hmac(
            branch='explicit_approval',
            approval_fingerprint_key_material_verifier='a' * 64,
            approval_fingerprint_key_version='key-é',
            approval_link_id=5,
            approval_permission='restricted',
            claim_fingerprint='b' * 64,
            evidence_link_set_hmac='c' * 64,
            legacy_evidence_pairs_hmac=None,
            promotion_effect_kind='primary',
            resolution_source='human',
            review_item_id=3,
            security_scope_id='workspace-🧭',
            selected_citation_child_hmac=None,
            settings=_settings(),
        )


def test_raw_trusted_citation_and_evidence_set_hmacs_bind_exact_shape_order_null_and_type() -> None:
    raw = RawServingVersionEnvelope(
        serving_document_id='chunk:6',
        source_row_id=1,
        public_source_id='gmail:bytes-é-🧭',
        document_id=2,
        document_version_id=3,
        current_document_version_id=3,
        document_chunk_id=6,
        parser_run_id=4,
        external_revision=None,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='a' * 64,
        parser_policy_version='server-source-parser-policy:v1',
        parser_version='source-event-paragraph-parser:v1',
        chunk_policy_version='paragraph-chunks:1200:v1',
        model_content_hmac='b' * 64,
        canonical_citation_projection_hmac='c' * 64,
        effective_permission='internal',
    )
    legacy = LegacyHumanProvenance(
        branch='legacy_human_base',
        legacy_binding='review_item',
        legacy_evidence_pairs_hmac='d' * 64,
        legacy_review_item_permission_level='internal',
        legacy_source_review_item_id=4,
    )
    trusted = TrustedServingVersionEnvelope(
        serving_document_id='history_event:9',
        knowledge_type='history_event',
        knowledge_id=9,
        model_content_hmac='b' * 64,
        canonical_citation_projection_hmac='c' * 64,
        effective_permission='restricted',
        provenance=legacy,
    )

    raw_hmac = build_serving_version_fingerprint(raw, settings=_settings())
    trusted_hmac = build_serving_version_fingerprint(
        trusted,
        settings=_settings(),
    )
    citation_hmac = build_canonical_citation_projection_hmac(
        public_source_id='history_event:9',
        public_source_type='history_event',
        source_url='https://example.test/가?x=é',
        source_snippet='snippet 🧭',
        effective_permission='restricted',
        settings=_settings(),
    )
    evidence_set_hmac = build_evidence_link_set_hmac(
        approval_link_id=5,
        ordered_link_hmacs=((7, 'a' * 64), (8, 'b' * 64)),
        settings=_settings(),
    )

    assert raw_hmac == '6144d916c44c7c8a40e33463438b98b24d4d0ba8109f2f07e05353854195a304'
    assert trusted_hmac == 'a0965a51e710c6dd0ef8a5fa111c5f24c8ef752c644453a3f80045d655233e6e'
    assert citation_hmac == '2955776ce162daf343662d6e98782ee5ea3dfbc9856bf88d5f408a31a44d6295'
    assert evidence_set_hmac == '558bbaecddca77faeaf9779c6110168de03216f16e48bdd404349502d2db4b9a'
    assert build_serving_version_fingerprint(
        replace(raw, external_revision='revision-1'),
        settings=_settings(),
    ) != raw_hmac
    assert build_evidence_link_set_hmac(
        approval_link_id=5,
        ordered_link_hmacs=((8, 'b' * 64), (7, 'a' * 64)),
        settings=_settings(),
    ) == '88afef5ab87c98a63a8bd49dca7524f67d366c2bfec95d41d3b67279dd50ac82'
    with pytest.raises(ValueError, match='positive'):
        build_serving_version_fingerprint(
            replace(raw, source_row_id=True),
            settings=_settings(),
        )
