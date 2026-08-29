from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.models import (
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRevocationAssessment,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    AutoReviewValidation,
    AutoReviewValidationCall,
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
    TrustedKnowledgeFingerprint,
    VectorIndexState,
)
from backend.app.rag.indexing import PreviewVectorIndexWriter
from backend.app.review.auto_review_revoke import (
    SourceInvalidationRevokeContext,
)


def _seed_explicit_history(
    db: Session,
    *,
    resolution_source: str,
    current_signature: str = 'a' * 64,
    evidence_signature: str = 'a' * 64,
) -> tuple[HistoryEvent, ReviewItem, Source, TrustedKnowledgeApprovalLink]:
    settings = Settings(database_url='sqlite://')
    key_verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    if (
        db.scalar(
            select(AutoReviewRuntimeKeyState).where(
                AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
            )
        )
        is None
    ):
        db.add(
            AutoReviewRuntimeKeyState(
                component='auto_review_trust_promotion',
                fingerprint_key_version=(
                    settings.agent_runtime_fingerprint_key_version
                ),
                fingerprint_key_material_verifier=key_verifier,
                generation=1,
                ready=True,
            )
        )
    source = Source(
        source_type='gmail',
        source_id=f'gmail:{resolution_source}:{evidence_signature[:8]}',
        source_url='https://gmail.mock/evidence',
        title='Current evidence',
        author='owner@example.com',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=current_signature,
    )
    item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Trusted history', 'summary': 'Exact current evidence'},
        source_links=[source.source_url],
        source_snippets=['Exact current evidence'],
        confidence_score=0.99,
        permission_level='internal',
        status='approved',
        candidate_contract_version='c5-v1',
        resolution_source=resolution_source,
        resolution_policy_version='auto-review-policy:v1',
        workflow_thread_id=(
            'workflow-c5' if resolution_source == 'auto_policy' else None
        ),
    )
    db.add_all([source, item])
    db.flush()
    if resolution_source == 'auto_policy':
        validation_call_id = 1
        if db.get_bind().dialect.name == 'postgresql':
            call = AutoReviewValidationCall(
                workflow_thread_id='workflow-c5',
                batch_fingerprint='b' * 64,
                status='failed',
                candidate_count=1,
                max_provider_attempts=1,
                provider_attempt_count=0,
                attempt_started_at=None,
                reserved_input_tokens=0,
                reserved_output_tokens=0,
                reserved_cost_usd=Decimal('0'),
                charged_input_tokens=0,
                charged_output_tokens=0,
                charged_cost_usd=Decimal('0'),
                prepared_content_hmac='a' * 64,
                serialized_character_count=1,
                framed_input_token_count=1,
                max_output_tokens=1,
                token_estimator_version='test:v1',
                tokenizer_encoding='test',
                reply_priming_tokens=0,
                framing_safety_tokens=0,
                input_usd_per_1m=Decimal('0'),
                output_usd_per_1m=Decimal('0'),
                cost_policy_version='auto-review-cost:v1',
                provider_safety_state_version=1,
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier=key_verifier,
                workflow_extraction_cost_ceiling_usd=Decimal('0'),
                workflow_validation_cost_ceiling_usd=Decimal('0'),
                workflow_total_cost_ceiling_usd=Decimal('0'),
                provider_timeout_seconds=1,
                provider_send_start_window_seconds=1,
                provider_attempt_lease_seconds=1,
                provider_commit_grace_seconds=1,
                budget_overrun=False,
                budget_overrun_cost_usd=Decimal('0'),
                terminal_at=datetime.now(UTC),
            )
            db.add(call)
            db.flush([call])
            validation_call_id = call.id
        validation = AutoReviewValidation(
            review_item_id=item.id,
            validation_call_id=validation_call_id,
            workflow_thread_id='workflow-c5',
            validation_key='validation-key',
            evidence_version_hash='e' * 64,
            candidate_generation_fingerprint='c' * 64,
            status='completed',
            validator_provider='openai',
            validator_model='gpt-5.6-terra',
            reasoning_effort='medium',
            validator_prompt_version='auto-review-validation:v1',
            validator_output_contract_version='candidate-validation-batch:v1',
            policy_version='auto-review-policy:v1',
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier=key_verifier,
            cost_policy_version='auto-review-cost:v1',
            confirmed_validation_cost_ceiling_usd=Decimal('0.010000'),
            claim_results=[],
            minimum_entailment_score=Decimal('0.9900'),
            uncertainty_codes=[],
            conflict_codes=[],
            policy_decision='auto_approve',
            policy_reason_codes=['direct_fact_supported'],
            input_tokens=10,
            output_tokens=5,
            estimated_cost_usd=Decimal('0.000100'),
            cache_hit=False,
            completed_at=datetime.now(UTC),
        )
        db.add(validation)
        db.flush()
        item.auto_validation_id = validation.id
    history = HistoryEvent(
        project_key='project-a',
        title='Trusted history',
        reason='Exact current evidence',
        source_links=[source.source_url],
        source_snippets=['Exact current evidence'],
        confidence_score=0.99,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=item.id,
    )
    db.add(history)
    db.flush()
    link = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=history.id,
        review_item_id=item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='primary',
        resolution_source=resolution_source,
        claim_fingerprint='d' * 64,
        permission_level='internal',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier=key_verifier,
        active=True,
    )
    db.add(link)
    db.flush()
    db.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=link.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(source.id),
            canonical_version_or_signature=evidence_signature,
            evidence_hash='e' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier=key_verifier,
        )
    )
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v1',
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body='Exact current evidence',
    )
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
        document_version_label=version.version,
        revision_id='gmail-revision-1',
        content_signature=current_signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=current_signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    db.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text='Exact current evidence',
            source_snippet='Exact current evidence',
            permission_level='internal',
            metadata_={},
        )
    )
    document.current_document_version_id = version.id
    db.commit()
    return history, item, source, link


def test_post_c5_human_only_and_pre_c5_legacy_human_serving_semantics_are_compatible(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )

    explicit, _, _, _ = _seed_explicit_history(db_session, resolution_source='human')
    legacy_item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Legacy human history'},
        source_links=['https://legacy.mock/evidence'],
        source_snippets=['Legacy evidence'],
        confidence_score=0.8,
        permission_level='restricted',
        status='approved',
        resolution_source='human',
    )
    db_session.add(legacy_item)
    db_session.flush()
    legacy = HistoryEvent(
        project_key='project-legacy',
        title='Legacy human history',
        reason='Legacy evidence',
        source_links=legacy_item.source_links,
        source_snippets=legacy_item.source_snippets,
        confidence_score=0.8,
        permission_level='restricted',
        review_status='approved',
        source_review_item_id=legacy_item.id,
    )
    db_session.add(legacy)
    db_session.commit()
    service = TrustedServingEligibilityService(db_session)

    assert service.for_knowledge('history_event', explicit.id).eligible is True
    legacy_result = service.for_knowledge('history_event', legacy.id)
    assert legacy_result.eligible is True
    assert legacy_result.effective_permission == 'restricted'


def test_current_verified_parser_revision_is_valid_explicit_evidence_identity(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )

    history, _, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    evidence = db_session.scalar(
        select(TrustedKnowledgeEvidenceLink).where(
            TrustedKnowledgeEvidenceLink.approval_link_id == link.id
        )
    )
    evidence.canonical_version_or_signature = 'gmail-revision-41'
    parser_run = (
        db_session.query(DocumentParserRun).filter_by(source_id=source.id).one()
    )
    parser_run.revision_id = 'gmail-revision-41'
    db_session.commit()

    result = TrustedServingEligibilityService(db_session).for_knowledge(
        'history_event', history.id
    )

    assert result.eligible is True
    assert result.effective_permission == 'internal'


def test_auto_trusted_vector_is_excluded_before_ranking_while_reconciliation_is_pending(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )

    history, _, _, _ = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='b' * 64,
    )

    result = TrustedServingEligibilityService(db_session).for_knowledge(
        'history_event', history.id
    )

    assert result.eligible is False
    assert result.effective_permission is None


def test_direct_signature_does_not_bypass_current_pointer_in_serving_or_readiness(
    client,
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    history, _, source, _ = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
    )
    document = db_session.query(Document).filter_by(source_id=source.id).one()
    document.current_document_version_id = None
    db_session.commit()

    serving = TrustedServingEligibilityService(db_session).for_knowledge(
        'history_event', history.id
    )
    status = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    ).status(limit=100)
    knowledge = client.get(
        '/api/v1/knowledge',
        headers={'X-Demo-User': 'viewer'},
    )
    search = client.post(
        '/api/v1/search',
        headers={'X-Demo-User': 'viewer'},
        json={'query': 'Trusted history'},
    )
    ask = client.post(
        '/api/v1/ask',
        headers={'X-Demo-User': 'viewer'},
        json={'question': 'Trusted history'},
    )

    assert serving.eligible is False
    assert serving.effective_permission is None
    assert status.stale_count == 1
    assert status.readiness is False
    assert knowledge.status_code == 200
    assert knowledge.json()['history_events'] == []
    assert search.status_code == 200
    assert search.json()['results'] == []
    assert ask.status_code == 200
    assert ask.json()['source_ids'] == []


def test_any_critical_or_remediation_audit_quarantines_only_its_auto_effect_before_revoke_cleanup(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )

    history, item, _, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    db_session.add(
        AutoReviewPostAudit(
            review_item_id=item.id,
            promotion_decision_id=1,
            sample_cohort='manual',
            status='remediation_required',
            outcome='incorrect',
            system_resolution_code='revoke_pending',
            remediation_code='exact_revoke_required',
            auditor_subject_hmac='a' * 64,
            auditor_fingerprint_key_version='v1',
            auditor_fingerprint_key_material_verifier='b' * 64,
            audit_reason='Incorrect evidence',
            audited_at=datetime.now(UTC),
        )
    )
    db_session.commit()

    result = TrustedServingEligibilityService(db_session).for_knowledge(
        'history_event', history.id
    )

    assert result.eligible is False
    assert result.effective_permission is None


def test_public_to_internal_reconciliation_narrows_target_and_vector_without_embedding(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    history, item, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    history.permission_level = 'public'
    item.permission_level = 'public'
    link.permission_level = 'public'
    source.permission_level = 'internal'
    fingerprint = TrustedKnowledgeFingerprint(
        knowledge_type='history_event',
        knowledge_id=history.id,
        security_scope_id='workspace-a',
        scope_resolution='exact',
        project_scope_hmac='a' * 64,
        normalized_title_bucket_hmac='b' * 64,
        normalized_claim_fingerprint='c' * 64,
        fingerprint_key_version=link.fingerprint_key_version,
        fingerprint_key_material_verifier=(link.fingerprint_key_material_verifier),
        permission_level='public',
        review_status='approved',
    )
    index_state = VectorIndexState(
        document_id=f'history_event:{history.id}',
        embedding_model='fake:8',
        embedding_dimensions=8,
        content_hash='0' * 64,
        status='indexed',
    )
    db_session.add_all([fingerprint, index_state])
    writer = PreviewVectorIndexWriter()
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=writer,
    ).reconcile_source_ids([source.id])

    assert result.reconciled_count == 1
    assert result.revoked_count == 0
    assert history.permission_level == 'internal'
    assert item.permission_level == 'internal'
    assert link.permission_level == 'internal'
    assert fingerprint.permission_level == 'internal'
    assert index_state.content_hash != '0' * 64
    assert writer.permission_narrowings == [(('history_event:1',), 'internal')]


def test_committed_changed_state_dto_is_frozen_and_reconcile_is_the_public_handoff(
    db_session: Session,
) -> None:
    from dataclasses import FrozenInstanceError

    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
        CommittedSourceStateChange,
    )

    history, item, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    history.permission_level = 'public'
    item.permission_level = 'public'
    link.permission_level = 'public'
    source.permission_level = 'internal'
    db_session.commit()
    changed = CommittedSourceStateChange(
        source_id=source.id,
        content_changed=False,
        permission_changed=True,
        parser_policy_changed=False,
        primary_code='permission_changed',
    )
    with pytest.raises(FrozenInstanceError):
        changed.permission_changed = False  # type: ignore[misc]

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    ).reconcile([changed])

    assert result.reconciled_count == 1
    assert history.permission_level == 'internal'


def test_reconcile_uses_authoritative_flags_when_primary_code_is_forged_unchanged(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
        CommittedSourceStateChange,
    )

    history, item, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    history.permission_level = 'public'
    item.permission_level = 'public'
    link.permission_level = 'public'
    source.permission_level = 'internal'
    db_session.commit()

    changed = CommittedSourceStateChange(
        source_id=source.id,
        content_changed=False,
        permission_changed=True,
        parser_policy_changed=False,
        primary_code='unchanged',
    )

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    ).reconcile([changed])

    assert result.reconciled_count == 1
    assert history.permission_level == 'internal'


@pytest.mark.parametrize('primary_code', [None, '', 'x' * 65, object()])
def test_committed_changed_state_rejects_unbounded_primary_code(
    primary_code: object,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        CommittedSourceStateChange,
    )

    with pytest.raises(ValueError, match='primary_code is inconsistent'):
        CommittedSourceStateChange(
            source_id=1,
            content_changed=False,
            permission_changed=False,
            parser_policy_changed=False,
            primary_code=primary_code,  # type: ignore[arg-type]
        )


def test_restricted_unknown_absent_or_superseded_source_revokes_exact_auto_effect(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    history, item, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    source.permission_level = 'restricted'
    writer = PreviewVectorIndexWriter()
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=writer,
    ).reconcile_source_ids([source.id])

    db_session.refresh(item)
    db_session.refresh(link)
    db_session.refresh(history)
    assert result.revoked_count == 1
    assert item.status == 'revoked'
    assert link.active is False
    assert history.review_status == 'revoked'
    assert writer.deletes == [('history_event:1',)]


def test_source_reconciliation_recovery_is_bounded_idempotent_and_restart_safe(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _, _, source, _ = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='b' * 64,
    )
    service = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    )

    status = service.status(limit=1)
    first = service.recover_stale_sources(limit=1)
    replay = service.recover_stale_sources(limit=1)

    assert status.stale_count == 1
    assert first.reconciled_count == 1
    assert first.remaining_count == 0
    assert replay.reconciled_count == 0
    assert replay.remaining_count == 0


def test_relational_stale_scan_finds_late_row_after_high_cardinality_fresh_prefix(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _, _, source, link = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='a' * 64,
    )
    for ordinal in range(450):
        db_session.add(
            TrustedKnowledgeEvidenceLink(
                approval_link_id=link.id,
                canonical_source_kind='gmail',
                canonical_source_id=str(source.id),
                canonical_version_or_signature='a' * 64,
                evidence_hash=f'{ordinal:064x}',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='b' * 64,
            )
        )
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=link.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(source.id),
            canonical_version_or_signature='c' * 64,
            evidence_hash='d' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
        )
    )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
    ).status(limit=1)

    assert result.stale_count == 1
    assert result.remaining_count == 1
    assert result.readiness is False


def test_relational_stale_scan_has_constant_query_count_for_large_source(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _, _, source, link = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='a' * 64,
    )
    for ordinal in range(450):
        db_session.add(
            TrustedKnowledgeEvidenceLink(
                approval_link_id=link.id,
                canonical_source_kind='gmail',
                canonical_source_id=str(source.id),
                canonical_version_or_signature='a' * 64,
                evidence_hash=f'{ordinal:064x}',
                fingerprint_key_version=link.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    link.fingerprint_key_material_verifier
                ),
            )
        )
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=link.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(source.id),
            canonical_version_or_signature='c' * 64,
            evidence_hash='f' * 64,
            fingerprint_key_version=link.fingerprint_key_version,
            fingerprint_key_material_verifier=(link.fingerprint_key_material_verifier),
        )
    )
    db_session.commit()
    select_count = 0

    def count_selects(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal select_count
        if statement.lstrip().upper().startswith('SELECT'):
            select_count += 1

    bind = db_session.get_bind()
    event.listen(bind, 'before_cursor_execute', count_selects)
    try:
        result = AutoReviewSourceReconciliationService(
            db_session,
            settings=Settings(database_url='sqlite://'),
        ).status(limit=1)
    finally:
        event.remove(bind, 'before_cursor_execute', count_selects)

    assert result.stale_count == 1
    assert select_count <= 12


def test_recovery_reaches_late_stale_auto_evidence_after_irrelevant_prefix(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    history, item, source, auto_link = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='a' * 64,
    )
    inactive_item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Inactive prefix'},
        source_links=[source.source_url],
        source_snippets=['Inactive prefix evidence'],
        confidence_score=0.99,
        permission_level='internal',
        status='revoked',
        resolution_source='auto_policy',
    )
    human_item = ReviewItem(
        item_type='history_event',
        payload={'title': 'Human prefix'},
        source_links=[source.source_url],
        source_snippets=['Human prefix evidence'],
        confidence_score=0.99,
        permission_level='internal',
        status='approved',
        resolution_source='human',
    )
    db_session.add_all([inactive_item, human_item])
    db_session.flush()
    inactive_history = HistoryEvent(
        project_key='project-a',
        title='Inactive prefix',
        reason='Inactive prefix evidence',
        source_links=inactive_item.source_links,
        source_snippets=inactive_item.source_snippets,
        confidence_score=0.99,
        permission_level='internal',
        review_status='revoked',
        source_review_item_id=inactive_item.id,
    )
    human_history = HistoryEvent(
        project_key='project-a',
        title='Human prefix',
        reason='Human prefix evidence',
        source_links=human_item.source_links,
        source_snippets=human_item.source_snippets,
        confidence_score=0.99,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=human_item.id,
    )
    db_session.add_all([inactive_history, human_history])
    db_session.flush()
    inactive_link = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=inactive_history.id,
        review_item_id=inactive_item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='primary',
        resolution_source='auto_policy',
        claim_fingerprint='1' * 64,
        permission_level='internal',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='b' * 64,
        active=False,
    )
    human_link = TrustedKnowledgeApprovalLink(
        knowledge_type='history_event',
        knowledge_id=human_history.id,
        review_item_id=human_item.id,
        security_scope_id='workspace-a',
        promotion_effect_kind='primary',
        resolution_source='human',
        claim_fingerprint='2' * 64,
        permission_level='internal',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='b' * 64,
        active=True,
    )
    db_session.add_all([inactive_link, human_link])
    db_session.flush()
    prefix_links = (
        [(inactive_link, 'inactive')] * 40
        + [(human_link, 'human')] * 40
        + [(auto_link, 'current')] * 40
    )
    for ordinal, (link, category) in enumerate(prefix_links):
        db_session.add(
            TrustedKnowledgeEvidenceLink(
                approval_link_id=link.id,
                canonical_source_kind='gmail',
                canonical_source_id=str(source.id),
                canonical_version_or_signature='a' * 64,
                evidence_hash=f'{category}:{ordinal}'.encode().hex().ljust(64, '0'),
                fingerprint_key_version=link.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    link.fingerprint_key_material_verifier
                ),
            )
        )
    db_session.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=auto_link.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(source.id),
            canonical_version_or_signature='c' * 64,
            evidence_hash='f' * 64,
            fingerprint_key_version=auto_link.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                auto_link.fingerprint_key_material_verifier
            ),
        )
    )
    db_session.commit()

    writer = PreviewVectorIndexWriter()
    first = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=writer,
    ).recover_stale_sources(limit=1)
    replay_after_restart = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=writer,
    ).recover_stale_sources(limit=1)

    db_session.refresh(history)
    db_session.refresh(item)
    db_session.refresh(auto_link)
    assert first.stale_count == 1
    assert first.reconciled_count == 1
    assert first.revoked_count == 1
    assert first.remaining_count == 0
    assert first.readiness is True
    assert history.review_status == 'revoked'
    assert item.status == 'revoked'
    assert auto_link.active is False
    assert writer.deletes == [(f'history_event:{history.id}',)]
    assert replay_after_restart.reconciled_count == 0
    assert replay_after_restart.remaining_count == 0
    assert replay_after_restart.readiness is True


def test_recovery_reaches_permission_drift_after_one_hundred_current_links(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _, _, source, _ = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='a' * 64,
    )
    drift_target: DecisionRecord | None = None
    for ordinal in range(101):
        permission = 'public' if ordinal == 100 else 'internal'
        item_type = 'decision_record' if ordinal == 100 else 'history_event'
        item = ReviewItem(
            item_type=item_type,
            payload={'title': f'Permission candidate {ordinal}'},
            source_links=[source.source_url],
            source_snippets=[f'Permission evidence {ordinal}'],
            confidence_score=0.99,
            permission_level=permission,
            status='approved',
            resolution_source='auto_policy',
        )
        db_session.add(item)
        db_session.flush()
        if ordinal == 100:
            target = DecisionRecord(
                project_key='project-a',
                title=f'Permission candidate {ordinal}',
                decision_summary=f'Permission evidence {ordinal}',
                source_links=item.source_links,
                source_snippets=item.source_snippets,
                confidence_score=0.99,
                permission_level=permission,
                review_status='approved',
                source_review_item_id=item.id,
            )
            knowledge_type = 'decision'
        else:
            target = HistoryEvent(
                project_key='project-a',
                title=f'Permission candidate {ordinal}',
                reason=f'Permission evidence {ordinal}',
                source_links=item.source_links,
                source_snippets=item.source_snippets,
                confidence_score=0.99,
                permission_level=permission,
                review_status='approved',
                source_review_item_id=item.id,
            )
            knowledge_type = 'history_event'
        db_session.add(target)
        db_session.flush()
        link = TrustedKnowledgeApprovalLink(
            knowledge_type=knowledge_type,
            knowledge_id=target.id,
            review_item_id=item.id,
            security_scope_id='workspace-a',
            promotion_effect_kind='primary',
            resolution_source='auto_policy',
            claim_fingerprint=f'{ordinal + 10:064x}',
            permission_level=permission,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier=(
                fingerprint_key_material_verifier(
                    Settings(database_url='sqlite://').agent_runtime_fingerprint_secret
                )
            ),
            active=True,
        )
        db_session.add(link)
        db_session.flush()
        db_session.add(
            TrustedKnowledgeEvidenceLink(
                approval_link_id=link.id,
                canonical_source_kind='gmail',
                canonical_source_id=str(source.id),
                canonical_version_or_signature='a' * 64,
                evidence_hash=f'{ordinal + 1000:064x}',
                fingerprint_key_version=link.fingerprint_key_version,
                fingerprint_key_material_verifier=(
                    link.fingerprint_key_material_verifier
                ),
            )
        )
        if ordinal == 100:
            drift_target = target
    db_session.commit()
    assert drift_target is not None

    first = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    ).recover_stale_sources(limit=1)
    replay_after_restart = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    ).recover_stale_sources(limit=1)

    db_session.refresh(drift_target)
    assert first.stale_count == 1
    assert first.narrowed_count == 1
    assert first.remaining_count == 0
    assert drift_target.permission_level == 'internal'
    assert replay_after_restart.reconciled_count == 0
    assert replay_after_restart.remaining_count == 0


def test_recovery_initial_scan_database_failure_is_bounded() -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    engine = create_engine('sqlite://')
    try:
        with sessionmaker(bind=engine)() as db:
            result = AutoReviewSourceReconciliationService(
                db,
                settings=Settings(database_url='sqlite://'),
            ).recover_stale_sources(limit=1)
    finally:
        engine.dispose()

    assert result.failure_count == 1
    assert result.remaining_count == 1
    assert result.readiness is False


def test_recovery_final_scan_database_failure_is_bounded(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _, _, source, _ = _seed_explicit_history(
        db_session,
        resolution_source='auto_policy',
        current_signature='a' * 64,
        evidence_signature='b' * 64,
    )
    service = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    )
    original_scan = service._stale_source_ids
    scan_count = 0

    def fail_final_scan(*, limit: int) -> list[int]:
        nonlocal scan_count
        scan_count += 1
        if scan_count == 1:
            return original_scan(limit=limit)
        raise OperationalError('SELECT bounded stale work', {}, RuntimeError('boom'))

    monkeypatch.setattr(service, '_stale_source_ids', fail_final_scan)

    result = service.recover_stale_sources(limit=1)

    assert result.stale_count == 1
    assert result.reconciled_count == 1
    assert result.failure_count == 1
    assert result.remaining_count == 1
    assert result.readiness is False
    assert source.id > 0


def test_repair_initial_scan_database_failure_is_bounded() -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    engine = create_engine('sqlite://')
    try:
        with sessionmaker(bind=engine)() as db:
            result = AutoReviewSourceReconciliationService(
                db,
                settings=Settings(database_url='sqlite://'),
            ).repair_current_document_versions(limit=1)
    finally:
        engine.dispose()

    assert result.failure_count == 1
    assert result.remaining_count == 1
    assert result.readiness is False


def test_repair_row_database_failure_is_counted_and_later_rows_continue(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    sources = [
        Source(
            source_type='drive',
            source_id=f'drive:repair-failure:{ordinal}',
            source_url=f'https://drive.mock/repair-failure/{ordinal}',
            title=f'Repair failure {ordinal}',
            permission_level='internal',
            raw_metadata={},
        )
        for ordinal in range(2)
    ]
    db_session.add_all(sources)
    db_session.flush()
    documents = [
        Document(
            source_id=source.id,
            title=source.title,
            current_version='display-only',
        )
        for source in sources
    ]
    db_session.add_all(documents)
    db_session.commit()
    service = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
    )

    def fail_first_row(*, document_id: int, source_id: int) -> str:
        if document_id == documents[0].id:
            raise OperationalError('SELECT repair row', {}, RuntimeError('boom'))
        assert source_id == sources[1].id
        return 'repaired'

    monkeypatch.setattr(service, '_repair_current_document_version', fail_first_row)

    result = service.repair_current_document_versions(limit=2)

    assert result.repaired_count == 1
    assert result.failure_count == 1
    assert result.remaining_count == 2
    assert result.readiness is False


def test_crash_after_source_document_commit_before_reconciliation_is_fail_closed_and_recoverable(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    history, item, source, link = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    history.permission_level = 'public'
    item.permission_level = 'public'
    link.permission_level = 'public'
    source.permission_level = 'internal'
    db_session.commit()
    writer = PreviewVectorIndexWriter()
    service = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=writer,
    )

    serving = TrustedServingEligibilityService(db_session).for_knowledge(
        'history_event', history.id
    )
    status = service.status(limit=100)
    recovered = service.recover_stale_sources(limit=100)

    assert serving.eligible is True
    assert serving.effective_permission == 'internal'
    assert status.stale_count == 1
    assert recovered.reconciled_count == 1
    assert recovered.remaining_count == 0
    assert history.permission_level == 'internal'
    assert writer.permission_narrowings == [(('history_event:1',), 'internal')]


def test_selected_pending_audit_is_system_invalidated_and_no_longer_blocks_rollout_gate(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _, item, source, _ = _seed_explicit_history(
        db_session, resolution_source='auto_policy'
    )
    verifier = fingerprint_key_material_verifier(
        Settings(database_url='sqlite://').agent_runtime_fingerprint_secret
    )
    rollout = AutoReviewRolloutState(
        security_scope_id='workspace-a',
        policy_version='auto-review-policy:v1',
        pending_mandatory_audit_count=1,
    )
    db_session.add(rollout)
    db_session.flush([rollout])
    decision = AutoReviewPromotionDecision(
        review_item_id=item.id,
        security_scope_id=rollout.security_scope_id,
        policy_version=rollout.policy_version,
        rollout_authorization_generation=1,
        promotion_ordinal=1,
        requested_percentage=0,
        stored_percentage=0,
        authorized_percentage=0,
        enforce_selection_fingerprint='a' * 64,
        audit_selection_fingerprint='b' * 64,
        selection_result='first_50',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier=verifier,
    )
    db_session.add(decision)
    db_session.flush([decision])
    audit = AutoReviewPostAudit(
        review_item_id=item.id,
        promotion_decision_id=decision.id,
        sample_cohort='first_50',
        status='pending',
        outcome=None,
    )
    db_session.add(audit)
    source.permission_level = 'restricted'
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session,
        settings=Settings(database_url='sqlite://'),
        vector_writer=PreviewVectorIndexWriter(),
    ).reconcile_source_ids([source.id])

    db_session.refresh(audit)
    db_session.refresh(rollout)
    assert result.revoked_count == 1
    assert audit.status == 'completed'
    assert audit.outcome is None
    assert audit.system_resolution_code == 'source_invalidated_before_audit'
    assert rollout.pending_mandatory_audit_count == 0
    assert rollout.confirmed_mandatory_audit_count == 0
    assert rollout.invalidated_before_audit_count == 1
    assert db_session.query(AutoReviewRevocationAssessment).count() == 0


def test_source_invalidation_revoke_writes_no_human_revocation_assessment() -> None:
    with pytest.raises(TypeError, match='minted only by reconciliation'):
        SourceInvalidationRevokeContext(
            session_identity=1,
            review_item_id=1,
            canonical_source_id='1',
        )


def _seed_pointer_repair_candidate(
    db: Session,
    *,
    ordinal: int,
    parser_overrides: dict[str, str] | None = None,
    persist_chunk: bool = True,
) -> tuple[Document, DocumentVersion]:
    signature = f'{ordinal:064x}'
    source = Source(
        source_type='drive',
        source_id=f'drive:pointer-repair:{ordinal}',
        source_url=f'https://drive.mock/pointer-repair/{ordinal}',
        title=f'Pointer repair {ordinal}',
        permission_level='internal',
        raw_metadata={'mime_type': 'text/plain'},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db.add(source)
    db.flush()
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='display-only',
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version='v1',
        body='exact server body',
    )
    db.add(version)
    db.flush()
    parser_values = {
        'parser_name': 'server_drive_source_event',
        'mime_type': 'text/plain',
        'parser_policy_version': SERVER_PARSER_POLICY_VERSION,
        'parser_version': SERVER_PARSER_VERSION,
        'chunk_policy_version': SERVER_CHUNK_POLICY_VERSION,
    }
    parser_values.update(parser_overrides or {})
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name=parser_values['parser_name'],
        parser_status='parsed',
        mime_type=parser_values['mime_type'],
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        content_signature=signature,
        parser_policy_version=parser_values['parser_policy_version'],
        parser_version=parser_values['parser_version'],
        chunk_policy_version=parser_values['chunk_policy_version'],
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    if persist_chunk:
        db.add(
            DocumentChunk(
                version_id=version.id,
                source_id=source.id,
                parser_run_id=parser_run.id,
                chunk_index=0,
                text='exact server body',
                source_snippet='exact server body',
                permission_level='internal',
                metadata_={},
            )
        )
    return document, version


def _seed_runtime_key(db: Session, settings: Settings) -> None:
    db.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=(settings.agent_runtime_fingerprint_key_version),
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
            generation=1,
            ready=True,
        )
    )


def test_current_pointer_repair_accepts_exact_server_registry_identity(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    settings = Settings(database_url='sqlite://')
    _seed_runtime_key(db_session, settings)
    exact_document, exact_version = _seed_pointer_repair_candidate(
        db_session,
        ordinal=1,
    )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session, settings=settings
    ).repair_current_document_versions(limit=1)

    db_session.refresh(exact_document)
    assert result.repaired_count == 1
    assert result.ambiguous_count == 0
    assert result.remaining_count == 0
    assert result.readiness is True
    assert exact_document.current_document_version_id == exact_version.id


@pytest.mark.parametrize(
    ('field_name', 'wrong_value'),
    [
        ('parser_name', 'plain_text'),
        ('mime_type', 'application/pdf'),
        ('parser_policy_version', 'parser-policy:v1'),
        ('parser_version', 'plain-text:v1'),
        ('chunk_policy_version', 'chunk-policy:v1'),
    ],
)
def test_current_pointer_repair_rejects_non_registry_parser_identity(
    db_session: Session,
    field_name: str,
    wrong_value: str,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    settings = Settings(database_url='sqlite://')
    _seed_runtime_key(db_session, settings)
    document, _ = _seed_pointer_repair_candidate(
        db_session,
        ordinal=2,
        parser_overrides={field_name: wrong_value},
    )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session, settings=settings
    ).repair_current_document_versions(limit=1)

    db_session.refresh(document)
    assert document.current_document_version_id is None
    assert result.repaired_count == 0
    assert result.ambiguous_count == 1
    assert result.remaining_count == 1
    assert result.readiness is False


def test_current_pointer_repair_requires_exact_run_chunk_relation(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    settings = Settings(database_url='sqlite://')
    _seed_runtime_key(db_session, settings)
    document, _ = _seed_pointer_repair_candidate(
        db_session,
        ordinal=3,
        persist_chunk=False,
    )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session, settings=settings
    ).repair_current_document_versions(limit=1)

    db_session.refresh(document)
    assert document.current_document_version_id is None
    assert result.ambiguous_count == 1
    assert result.readiness is False


def test_current_pointer_repair_never_guesses_between_exact_relational_versions(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    settings = Settings(database_url='sqlite://')
    _seed_runtime_key(db_session, settings)
    document, _ = _seed_pointer_repair_candidate(db_session, ordinal=7)
    source = db_session.get(Source, document.source_id)
    assert source is not None
    second_version = DocumentVersion(
        document_id=document.id,
        version='v2',
        body='second exact server body',
    )
    db_session.add(second_version)
    db_session.flush()
    second_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=second_version.id,
        source_id=source.id,
        parser_name='server_drive_source_event',
        parser_status='parsed',
        mime_type='text/plain',
        document_version_label=second_version.version,
        content_signature=source.server_content_signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=source.server_content_signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db_session.add(second_run)
    db_session.flush()
    db_session.add(
        DocumentChunk(
            version_id=second_version.id,
            source_id=source.id,
            parser_run_id=second_run.id,
            chunk_index=0,
            text='second exact server body',
            source_snippet='second exact server body',
            permission_level='internal',
            metadata_={},
        )
    )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session, settings=settings
    ).repair_current_document_versions(limit=1)

    db_session.refresh(document)
    assert document.current_document_version_id is None
    assert result.ambiguous_count == 1
    assert result.remaining_count == 1
    assert result.readiness is False


def test_current_pointer_repair_limit_reports_unscanned_continuation(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    settings = Settings(database_url='sqlite://')
    _seed_runtime_key(db_session, settings)
    first, _ = _seed_pointer_repair_candidate(db_session, ordinal=4)
    second, _ = _seed_pointer_repair_candidate(db_session, ordinal=5)
    db_session.commit()
    service = AutoReviewSourceReconciliationService(db_session, settings=settings)

    first_result = service.repair_current_document_versions(limit=1)
    second_result = service.repair_current_document_versions(limit=1)

    db_session.refresh(first)
    db_session.refresh(second)
    assert first_result.repaired_count == 1
    assert first_result.remaining_count >= 1
    assert first_result.readiness is False
    assert second_result.repaired_count == 1
    assert second_result.remaining_count == 0
    assert second_result.readiness is True


def test_status_and_cli_exit_include_ambiguous_pointer_work(
    db_session: Session,
) -> None:
    from backend.app.admin.auto_review_source_reconciliation import (
        command_exit_code,
    )
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    _seed_pointer_repair_candidate(
        db_session,
        ordinal=6,
        parser_overrides={'parser_name': 'legacy_drive_parser'},
    )
    db_session.commit()

    status = AutoReviewSourceReconciliationService(
        db_session, settings=Settings(database_url='sqlite://')
    ).status(limit=1)

    assert status.stale_count == 0
    assert status.ambiguous_count == 1
    assert status.remaining_count >= 1
    assert status.readiness is False
    assert command_exit_code(status) == 3
