from collections.abc import Generator
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.base import Base
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
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
