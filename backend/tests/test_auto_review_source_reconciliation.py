from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.core.config import Settings
from backend.app.models import (
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewRevocationAssessment,
    AutoReviewRolloutState,
    AutoReviewRuntimeKeyState,
    AutoReviewValidation,
    AutoReviewValidationCall,
    Document,
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
    if db.scalar(
        select(AutoReviewRuntimeKeyState).where(
            AutoReviewRuntimeKeyState.component
            == 'auto_review_trust_promotion'
        )
    ) is None:
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
    db.commit()
    return history, item, source, link


def test_post_c5_human_only_and_pre_c5_legacy_human_serving_semantics_are_compatible(
    db_session: Session,
) -> None:
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
    )

    explicit, _, _, _ = _seed_explicit_history(
        db_session, resolution_source='human'
    )
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
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v41',
    )
    db_session.add(document)
    db_session.flush([document])
    version = DocumentVersion(
        document_id=document.id,
        version='v41',
        body='Exact current evidence',
    )
    db_session.add(version)
    db_session.flush([version])
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='plain_text',
        parser_status='parsed',
        revision_id='gmail-revision-41',
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=source.server_content_signature,
        parser_policy_version='parser-policy:v1',
        parser_version='plain-text:v1',
        chunk_policy_version='chunk-policy:v1',
    )
    db_session.add(parser_run)
    db_session.flush([parser_run])
    document.current_document_version_id = version.id
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
        fingerprint_key_material_verifier=(
            link.fingerprint_key_material_verifier
        ),
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
            raise OperationalError(
                'SELECT repair row', {}, RuntimeError('boom')
            )
        assert source_id == sources[1].id
        return 'repaired'

    monkeypatch.setattr(
        service, '_repair_current_document_version', fail_first_row
    )

    result = service.repair_current_document_versions(limit=2)

    assert result.repaired_count == 1
    assert result.failure_count == 1
    assert result.remaining_count == 1
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
    assert writer.permission_narrowings == [
        (('history_event:1',), 'internal')
    ]


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


def test_current_pointer_repair_is_exact_and_ambiguity_remains_fail_closed(
    db_session: Session,
) -> None:
    from backend.app.review.auto_review_source_reconciliation import (
        AutoReviewSourceReconciliationService,
    )

    settings = Settings(database_url='sqlite://')
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version=(
                settings.agent_runtime_fingerprint_key_version
            ),
            fingerprint_key_material_verifier=verifier,
            generation=1,
            ready=True,
        )
    )
    exact_source = Source(
        source_type='drive',
        source_id='drive:repair-exact',
        source_url='https://drive.mock/repair-exact',
        title='Exact repair',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='1' * 64,
    )
    ambiguous_source = Source(
        source_type='drive',
        source_id='drive:repair-ambiguous',
        source_url='https://drive.mock/repair-ambiguous',
        title='Ambiguous repair',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='2' * 64,
    )
    db_session.add_all([exact_source, ambiguous_source])
    db_session.flush()
    exact_document = Document(
        source_id=exact_source.id,
        title=exact_source.title,
        current_version='display-only',
    )
    ambiguous_document = Document(
        source_id=ambiguous_source.id,
        title=ambiguous_source.title,
        current_version='display-only',
    )
    db_session.add_all([exact_document, ambiguous_document])
    db_session.flush()
    exact_version = DocumentVersion(
        document_id=exact_document.id, version='v1', body='exact'
    )
    ambiguous_versions = [
        DocumentVersion(
            document_id=ambiguous_document.id,
            version=f'v{index}',
            body=f'ambiguous {index}',
        )
        for index in (1, 2)
    ]
    db_session.add_all([exact_version, *ambiguous_versions])
    db_session.flush()
    for document, source, version in [
        (exact_document, exact_source, exact_version),
        *(
            (ambiguous_document, ambiguous_source, version)
            for version in ambiguous_versions
        ),
    ]:
        db_session.add(
            DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='plain_text',
                parser_status='parsed',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature=source.server_content_signature,
                parser_policy_version='parser-policy:v1',
                parser_version='plain-text:v1',
                chunk_policy_version='chunk-policy:v1',
            )
        )
    db_session.commit()

    result = AutoReviewSourceReconciliationService(
        db_session, settings=settings
    ).repair_current_document_versions(limit=2)

    db_session.refresh(exact_document)
    db_session.refresh(ambiguous_document)
    assert result.repaired_count == 1
    assert result.ambiguous_count == 1
    assert result.readiness is False
    assert exact_document.current_document_version_id == exact_version.id
    assert ambiguous_document.current_document_version_id is None
