from collections.abc import Generator
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.base import Base
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
    AutoReviewExtractionCall,
    AutoReviewPostAudit,
    AutoReviewProviderSafetyEvent,
    AutoReviewRevocationAssessment,
    AutoReviewRolloutControlEvent,
    Document,
    DocumentChunk,
    DocumentParserRun,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    SyncJob,
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)

    with session_local() as session:
        yield session

    Base.metadata.drop_all(engine)


def test_source_raw_metadata_tracks_in_place_mutation(db_session: Session) -> None:
    source = Source(
        source_type='drive',
        source_id='drive-1',
        source_url='https://example.test/drive-1',
        title='Source',
        permission_level='internal',
        raw_metadata={'labels': ['alpha']},
    )
    db_session.add(source)
    db_session.commit()

    source.raw_metadata['synced'] = True
    db_session.commit()
    db_session.refresh(source)

    assert source.raw_metadata['synced'] is True


def test_review_item_source_snippets_tracks_in_place_mutation(db_session: Session) -> None:
    review_item = ReviewItem(
        item_type='todo',
        payload={'title': 'Follow up'},
        source_links=[],
        source_snippets=['initial snippet'],
        confidence_score=0.91,
        permission_level='internal',
    )
    db_session.add(review_item)
    db_session.commit()

    review_item.source_snippets.append('new snippet')
    db_session.commit()
    db_session.refresh(review_item)

    assert review_item.source_snippets == ['initial snippet', 'new snippet']


def test_sync_job_updated_at_refreshes_on_update(db_session: Session) -> None:
    old_updated_at = datetime(2020, 1, 1)
    sync_job = SyncJob(
        job_id='sync-1',
        connector_type='drive',
        updated_at=old_updated_at,
    )
    db_session.add(sync_job)
    db_session.commit()

    sync_job.status = 'running'
    sync_job.progress_pct = 25
    db_session.commit()
    db_session.refresh(sync_job)

    assert sync_job.updated_at > old_updated_at


def test_c5_models_expose_bounded_identity_columns_and_named_ownership() -> None:
    assert AgentRun.__table__.c.generation_provider.type.length == 120
    assert ReviewItem.__table__.c.candidate_contract_version.nullable is True
    assert ReviewItem.__table__.c.revoked_by_subject_hmac.type.length == 64
    assert Source.__table__.c.server_content_signature.type.length == 64
    assert Document.__table__.c.current_document_version_id.nullable is True
    assert DocumentParserRun.__table__.c.parser_policy_version.nullable is True
    assert DocumentChunk.__table__.c.parser_run_id.nullable is True

    review_ref_constraints = {
        constraint.name for constraint in ReviewItemEvidenceRef.__table__.constraints
    }
    assert 'fk_review_item_evidence_refs_same_review_item_workflow' in review_ref_constraints
    assert 'fk_review_item_evidence_refs_same_evidence_workflow' in review_ref_constraints
    assert 'fk_review_items_auto_validation_same_item' in {
        constraint.name for constraint in ReviewItem.__table__.constraints
    }
    assert 'uq_agent_workflow_evidence_ref_id_workflow' in {
        constraint.name
        for constraint in AgentWorkflowEvidenceRef.__table__.constraints
    }


def test_v21_request_persists_exact_extraction_generation_snapshot() -> None:
    columns = AgentWorkflowRequest.__table__.c

    assert columns.auto_review_extraction_provider.type.length == 120
    assert columns.auto_review_extraction_model.type.length == 120
    assert columns.auto_review_extraction_reasoning_effort.type.length == 32
    assert columns.auto_review_extraction_route_version.type.length == 64
    assert all(
        columns[name].nullable
        for name in (
            'auto_review_extraction_provider',
            'auto_review_extraction_model',
            'auto_review_extraction_reasoning_effort',
            'auto_review_extraction_route_version',
        )
    )


def test_claimed_extraction_call_rejects_terminal_charge_fields(
    db_session: Session,
) -> None:
    thread = AgentWorkflowThread(
        thread_id='extraction-charge-thread',
        workflow_name='company-memory-review',
        graph_version='company-memory-review-v2.1-auto-review',
        checkpoint_thread_id='checkpoint-extraction-charge',
        checkpoint_store='sqlite',
        owner_subject_id='owner',
        security_scope_id='scope',
        input_hash='a' * 64,
        evidence_version_hash='b' * 64,
        status='created',
    )
    run = AgentRun(
        agent_name='timeline_agent',
        prompt_version='timeline:c5-v1',
        status='claimed',
        source_window='window',
        cache_key='extraction-charge-cache',
        model_name='gpt-5.4-mini-2026-03-17',
        permission_level='internal',
        workflow_thread_id=thread.thread_id,
        generation_provider='openai',
        generation_reasoning_effort='none',
        generation_route_version='auto-review-extraction-route:v1',
        generation_output_contract_version='timeline-candidate:c5-v1',
    )
    db_session.add_all([thread, run])
    db_session.flush()
    db_session.add(
        AutoReviewExtractionCall(
            agent_run_id=run.id,
            workflow_thread_id=thread.thread_id,
            agent_name=run.agent_name,
            extraction_plan_hmac='c' * 64,
            provider='openai',
            model='gpt-5.4-mini-2026-03-17',
            reasoning_effort='none',
            route_version='auto-review-extraction-route:v1',
            prompt_version='timeline:c5-v1',
            output_contract_version='timeline-candidate:c5-v1',
            extraction_registry_version='registry:v1',
            cost_policy_version='cost:v1',
            provider_safety_state_version=1,
            token_estimator_version='estimator:v1',
            tokenizer_encoding='o200k_base',
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            prepared_content_hmac='d' * 64,
            prepared_character_count=100,
            framed_input_token_cap=1000,
            total_output_token_cap=100,
            max_candidates_per_agent=1,
            input_usd_per_1m=Decimal('1.000000'),
            output_usd_per_1m=Decimal('2.000000'),
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='e' * 64,
            provider_timeout_seconds=30,
            provider_send_start_window_seconds=5,
            provider_attempt_lease_seconds=60,
            provider_commit_grace_seconds=5,
            workflow_extraction_cost_ceiling_usd=Decimal('1.000000'),
            workflow_total_cost_ceiling_usd=Decimal('2.000000'),
            status='claimed',
            provider_attempt_count=0,
            reserved_input_tokens=100,
            reserved_output_tokens=100,
            reserved_cost_usd=Decimal('0.100000'),
            charged_input_tokens=1,
            charged_output_tokens=0,
            charged_cost_usd=Decimal('0.000001'),
            budget_overrun=False,
            budget_overrun_cost_usd=Decimal('0.000000'),
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_extraction_call_rejects_more_than_one_candidate_cap(
    db_session: Session,
) -> None:
    thread = AgentWorkflowThread(
        thread_id='extraction-cap-thread',
        workflow_name='company-memory-review',
        graph_version='company-memory-review-v2.1-auto-review',
        checkpoint_thread_id='checkpoint-extraction-cap',
        checkpoint_store='sqlite',
        owner_subject_id='owner',
        security_scope_id='scope',
        input_hash='a' * 64,
        evidence_version_hash='b' * 64,
        status='created',
    )
    run = AgentRun(
        agent_name='timeline_agent',
        prompt_version='timeline:c5-v1',
        status='claimed',
        source_window='window',
        cache_key='extraction-cap-cache',
        model_name='gpt-5.4-mini-2026-03-17',
        permission_level='internal',
        workflow_thread_id=thread.thread_id,
        generation_provider='openai',
        generation_reasoning_effort='none',
        generation_route_version='auto-review-extraction-route:v1',
        generation_output_contract_version='timeline-candidate:c5-v1',
    )
    db_session.add_all([thread, run])
    db_session.flush()
    call = AutoReviewExtractionCall(
        agent_run_id=run.id,
        workflow_thread_id=thread.thread_id,
        agent_name=run.agent_name,
        extraction_plan_hmac='c' * 64,
        provider='openai',
        model='gpt-5.4-mini-2026-03-17',
        reasoning_effort='none',
        route_version='auto-review-extraction-route:v1',
        prompt_version='timeline:c5-v1',
        output_contract_version='timeline-candidate:c5-v1',
        extraction_registry_version='registry:v1',
        cost_policy_version='cost:v1',
        provider_safety_state_version=1,
        token_estimator_version='estimator:v1',
        tokenizer_encoding='o200k_base',
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        prepared_content_hmac='d' * 64,
        prepared_character_count=100,
        framed_input_token_cap=1000,
        total_output_token_cap=100,
        max_candidates_per_agent=2,
        input_usd_per_1m=Decimal('1.000000'),
        output_usd_per_1m=Decimal('2.000000'),
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='e' * 64,
        provider_timeout_seconds=30,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=60,
        provider_commit_grace_seconds=5,
        workflow_extraction_cost_ceiling_usd=Decimal('1.000000'),
        workflow_total_cost_ceiling_usd=Decimal('2.000000'),
        status='claimed',
        provider_attempt_count=0,
        reserved_input_tokens=100,
        reserved_output_tokens=100,
        reserved_cost_usd=Decimal('0.100000'),
        charged_cost_usd=Decimal('0.000000'),
        budget_overrun=False,
        budget_overrun_cost_usd=Decimal('0.000000'),
    )
    db_session.add(call)

    with pytest.raises(IntegrityError):
        db_session.commit()


@pytest.mark.parametrize('signature', ['A' * 64, 'g' * 64])
def test_source_authority_rejects_non_lowercase_or_non_hex_signature(
    db_session: Session,
    signature: str,
) -> None:
    db_session.add(
        Source(
            source_type='drive',
            source_id='invalid-server-signature',
            source_url='https://example.test/source',
            title='Source',
            permission_level='internal',
            raw_metadata={},
            server_content_signature_schema='server-source-content:v1',
            server_content_signature=signature,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_pending_audit_rejects_human_terminal_fields(db_session: Session) -> None:
    db_session.add(
        AutoReviewPostAudit(
            review_item_id=100,
            promotion_decision_id=200,
            sample_cohort='first_50',
            status='pending',
            outcome='confirmed',
            auditor_subject_hmac='a' * 64,
            auditor_fingerprint_key_version='v1',
            auditor_fingerprint_key_material_verifier='b' * 64,
            audit_reason='confirmed by a human',
            audited_at=datetime.now(),
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


@pytest.mark.parametrize(
    'values',
    [
        {
            'status': 'completed',
            'system_resolution_code': 'source_invalidated_before_audit',
            'audit_reason': 'must be null for system resolution',
            'audited_at': datetime.now(),
        },
        {
            'status': 'completed',
            'outcome': 'confirmed',
            'audit_reason': 'missing auditor identity',
            'audited_at': datetime.now(),
        },
        {
            'status': 'completed',
            'system_resolution_code': 'wrong-code',
            'audited_at': datetime.now(),
        },
        {
            'status': 'remediation_required',
            'outcome': 'incorrect',
            'auditor_subject_hmac': 'a' * 64,
            'auditor_fingerprint_key_version': 'v1',
            'auditor_fingerprint_key_material_verifier': 'b' * 64,
            'audit_reason': 'reason',
            'audited_at': datetime.now(),
        },
    ],
)
def test_audit_rejects_non_exact_terminal_field_combinations(
    db_session: Session,
    values: dict,
) -> None:
    db_session.add(
        AutoReviewPostAudit(
            review_item_id=101,
            promotion_decision_id=201,
            sample_cohort='first_50',
            **values,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_c5_append_only_and_authority_checks_are_present_in_metadata() -> None:
    checks = {
        constraint.name
        for table in (
            AutoReviewExtractionCall.__table__,
            AutoReviewPostAudit.__table__,
            AutoReviewRevocationAssessment.__table__,
            AutoReviewProviderSafetyEvent.__table__,
            AutoReviewRolloutControlEvent.__table__,
        )
        for constraint in table.constraints
        if constraint.name is not None
    }
    assert {
        'ck_auto_review_extraction_calls_terminal_result',
        'ck_auto_review_post_audits_terminal_outcome',
        'ck_auto_review_revocation_assessments_reason_code',
        'ck_auto_review_provider_safety_events_kind',
        'ck_auto_review_rollout_control_events_kind',
    } <= checks


def test_c5_model_metadata_can_create_complete_sqlite_schema() -> None:
    engine = create_engine('sqlite:///:memory:')

    Base.metadata.create_all(engine)

    assert 'auto_review_validations' in inspect(engine).get_table_names()
