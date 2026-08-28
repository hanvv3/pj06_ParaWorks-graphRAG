import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    AssistantConversation,
    AssistantMessage,
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
    AutoReviewAuditCorrection,
    AutoReviewExtractionCall,
    AutoReviewPostAudit,
    AutoReviewPromotionDecision,
    AutoReviewProviderSafetyEvent,
    AutoReviewProviderSafetyState,
    AutoReviewRevocationAssessment,
    AutoReviewRolloutControlEvent,
    AutoReviewRolloutState,
    AutoReviewValidation,
    AutoReviewValidationCall,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    ReviewItem,
    ReviewItemEvidenceRef,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)

REVISION = '7c5a2e9f4b10'
PREVIOUS_REVISION = '2f6a8b9c0d1e'

AUTO_REVIEW_TABLES = {
    'auto_review_runtime_key_states',
    'auto_review_provider_safety_states',
    'auto_review_provider_safety_events',
    'trusted_knowledge_fingerprint_projection_states',
    'trusted_knowledge_fingerprints',
    'review_item_evidence_refs',
    'auto_review_extraction_calls',
    'auto_review_validation_calls',
    'auto_review_validations',
    'trusted_knowledge_approval_links',
    'trusted_knowledge_evidence_links',
    'assistant_message_evidence_dependencies',
    'assistant_message_knowledge_evidence_refs',
    'auto_review_rollout_states',
    'auto_review_rollout_control_events',
    'auto_review_promotion_decisions',
    'auto_review_post_audits',
    'auto_review_revocation_assessments',
    'auto_review_audit_corrections',
    'vector_serving_tombstones',
}


@pytest.fixture
def migration_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = tmp_path / 'auto-review-migration.db'
    database_url = f'sqlite:///{database_path.as_posix()}'
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    config = Config('alembic.ini')
    yield config, database_url
    get_settings.cache_clear()


def _run(operation: Callable, config: Config, revision: str) -> None:
    get_settings.cache_clear()
    operation(config, revision)
    get_settings.cache_clear()


def _postgres_engine(monkeypatch: pytest.MonkeyPatch):
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.fail(
            'PARAWORKS_TEST_POSTGRES_URL is required for Task 2 PostgreSQL guards'
        )
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        get_settings.cache_clear()
    return create_engine(database_url)


def _add_workflow_candidate(
    db: Session,
    *,
    suffix: str,
    agent_name: str = 'timeline_agent',
) -> tuple[AgentWorkflowThread, AgentRun, ReviewItem]:
    thread = AgentWorkflowThread(
        thread_id=f'ownership-{suffix}',
        workflow_name='company-memory-review',
        graph_version='company-memory-review-v2.1-auto-review',
        checkpoint_thread_id=f'ownership-checkpoint-{suffix}',
        checkpoint_store='postgres',
        owner_subject_id='owner',
        security_scope_id=f'scope-{suffix}',
        input_hash='a' * 64,
        evidence_version_hash='b' * 64,
        status='created',
    )
    run = AgentRun(
        agent_name=agent_name,
        prompt_version='timeline:c5-v1',
        status='complete',
        source_window='window',
        cache_key=f'ownership-cache-{suffix}',
        model_name='gpt-5.4-mini-2026-03-17',
        permission_level='internal',
        workflow_thread_id=thread.thread_id,
        generation_provider='openai',
        generation_reasoning_effort='none',
        generation_route_version='auto-review-extraction-route:v1',
        generation_output_contract_version='timeline-candidate:c5-v1',
    )
    evidence = AgentWorkflowEvidenceRef(
        workflow_thread_id=thread.thread_id,
        ordinal=1,
        canonical_source_type='drive',
        canonical_table='sources',
        canonical_row_id=1,
        content_signature=f'signature-{suffix}',
        permission_level_snapshot='internal',
        content_fingerprint='c' * 64,
    )
    db.add_all([thread, run, evidence])
    db.flush()
    item = ReviewItem(
        item_type='timeline_event',
        payload={},
        source_links=[],
        source_snippets=[],
        confidence_score=0.99,
        permission_level='internal',
        workflow_thread_id=thread.thread_id,
        candidate_key=f'candidate-{suffix}',
        agent_run_id=run.id,
        candidate_contract_version='c5-v1',
    )
    db.add(item)
    db.flush()
    db.add(
        ReviewItemEvidenceRef(
            review_item_id=item.id,
            workflow_thread_id=thread.thread_id,
            workflow_evidence_ref_id=evidence.id,
            candidate_slot_ordinal=1,
            message_content_fingerprint='d' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='e' * 64,
        )
    )
    db.flush()
    return thread, run, item


def _add_claimed_validation(
    db: Session,
    *,
    thread: AgentWorkflowThread,
    item: ReviewItem,
    suffix: str,
    validation_key: str,
) -> AutoReviewValidation:
    call = AutoReviewValidationCall(
        workflow_thread_id=thread.thread_id,
        batch_fingerprint=f'{suffix:0<64}'[:64],
        status='claimed',
        candidate_count=1,
        reserved_input_tokens=100,
        reserved_output_tokens=100,
        reserved_cost_usd=Decimal('0.100000'),
        charged_cost_usd=Decimal('0.000000'),
        prepared_content_hmac='f' * 64,
        serialized_character_count=100,
        framed_input_token_count=50,
        max_output_tokens=100,
        token_estimator_version='estimator:v1',
        tokenizer_encoding='o200k_base',
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        input_usd_per_1m=Decimal('1.000000'),
        output_usd_per_1m=Decimal('2.000000'),
        cost_policy_version='cost:v1',
        provider_safety_state_version=1,
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='1' * 64,
        workflow_extraction_cost_ceiling_usd=Decimal('1.000000'),
        workflow_validation_cost_ceiling_usd=Decimal('1.000000'),
        workflow_total_cost_ceiling_usd=Decimal('2.000000'),
        provider_timeout_seconds=30,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=60,
        provider_commit_grace_seconds=5,
        budget_overrun=False,
        budget_overrun_cost_usd=Decimal('0.000000'),
    )
    db.add(call)
    db.flush()
    validation = AutoReviewValidation(
        review_item_id=item.id,
        validation_call_id=call.id,
        workflow_thread_id=thread.thread_id,
        validation_key=validation_key,
        evidence_version_hash='2' * 64,
        candidate_generation_fingerprint='3' * 64,
        status='claimed',
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        reasoning_effort='medium',
        validator_prompt_version='validator:v1',
        validator_output_contract_version='validator-output:v1',
        policy_version='policy:v1',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='4' * 64,
        cost_policy_version='cost:v1',
        confirmed_validation_cost_ceiling_usd=Decimal('1.000000'),
        claim_results=[],
        uncertainty_codes=[],
        conflict_codes=[],
        policy_reason_codes=[],
    )
    db.add(validation)
    db.flush()
    return validation


def _claimed_extraction_call(
    *,
    run_id: int,
    workflow_thread_id: str,
    agent_name: str,
    suffix: str,
):
    return AutoReviewExtractionCall(
        agent_run_id=run_id,
        workflow_thread_id=workflow_thread_id,
        agent_name=agent_name,
        extraction_plan_hmac=f'{suffix:0<64}'[:64],
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
        prepared_content_hmac='c' * 64,
        prepared_character_count=100,
        framed_input_token_cap=1000,
        total_output_token_cap=100,
        max_candidates_per_agent=1,
        input_usd_per_1m=Decimal('1.000000'),
        output_usd_per_1m=Decimal('2.000000'),
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='d' * 64,
        provider_timeout_seconds=30,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=60,
        provider_commit_grace_seconds=5,
        workflow_extraction_cost_ceiling_usd=Decimal('1.000000'),
        workflow_total_cost_ceiling_usd=Decimal('2.000000'),
        status='claimed',
        reserved_input_tokens=100,
        reserved_output_tokens=100,
        reserved_cost_usd=Decimal('0.100000'),
        charged_cost_usd=Decimal('0.000000'),
        budget_overrun=False,
        budget_overrun_cost_usd=Decimal('0.000000'),
    )


def test_auto_review_migration_upgrades_fresh_schema_and_writes_boundary(
    migration_database,
) -> None:
    config, database_url = migration_database

    _run(command.upgrade, config, 'head')

    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) >= AUTO_REVIEW_TABLES
    with engine.connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == REVISION
        marker = connection.execute(
            text(
                "SELECT package_name, package_version, schema_revision "
                "FROM agent_runtime_schema_versions "
                "WHERE component='auto_review_trust_promotion'"
            )
        ).one()
    assert marker == ('paraworks', 'c5-v1', 1)


def test_auto_review_migration_has_exact_revision_chain() -> None:
    migration = Path(
        'backend/migrations/versions/'
        '7c5a2e9f4b10_add_auto_review_trust_promotion.py'
    ).read_text(encoding='utf-8')

    assert "revision = '7c5a2e9f4b10'" in migration
    assert "down_revision = '2f6a8b9c0d1e'" in migration


def test_auto_review_migration_preserves_legacy_rows_and_connector_signature_only(
    migration_database,
) -> None:
    config, database_url = migration_database
    _run(command.upgrade, config, PREVIOUS_REVISION)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO sources "
                "(source_type, source_id, source_url, title, permission_level, "
                "raw_metadata, created_at) VALUES "
                "('drive', 'legacy-source', 'https://example.test/source', "
                "'Legacy', 'internal', "
                "'{\"content_signature\":\"connector-only\"}', CURRENT_TIMESTAMP)"
            )
        )
        source_id = connection.scalar(
            text("SELECT id FROM sources WHERE source_id='legacy-source'")
        )

    _run(command.upgrade, config, 'head')

    with engine.connect() as connection:
        row = connection.execute(
            text(
                'SELECT connector_content_signature, '
                'server_content_signature_schema, server_content_signature '
                'FROM sources WHERE id=:source_id'
            ),
            {'source_id': source_id},
        ).one()
    assert row == ('connector-only', None, None)


def test_migration_backfills_only_unique_verified_parser_signature_pointer(
    migration_database,
) -> None:
    config, database_url = migration_database
    _run(command.upgrade, config, PREVIOUS_REVISION)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        source_id = connection.scalar(
            text(
                'INSERT INTO sources '
                '(source_type, source_id, source_url, title, permission_level, '
                'raw_metadata, server_content_signature_schema, '
                'server_content_signature, created_at) VALUES '
                "('drive', 'verified-pointer', 'https://example.test/pointer', "
                "'Pointer', 'internal', '{}', 'server-source-content:v1', "
                ':signature, CURRENT_TIMESTAMP) RETURNING id'
            ),
            {'signature': 'a' * 64},
        )
        document_id = connection.scalar(
            text(
                'INSERT INTO documents (source_id, title, current_version) '
                "VALUES (:source_id, 'Document', 'v1') RETURNING id"
            ),
            {'source_id': source_id},
        )
        version_ids = [
            connection.scalar(
                text(
                    'INSERT INTO document_versions (document_id, version, body) '
                    "VALUES (:document_id, 'v1', :body) RETURNING id"
                ),
                {'document_id': document_id, 'body': body},
            )
            for body in ('ambiguous-old', 'verified-current')
        ]
        connection.execute(
            text(
                'INSERT INTO document_parser_runs '
                '(document_id, document_version_id, source_id, parser_name, '
                'parser_status, mime_type, document_version_label, revision_id, '
                'content_signature, server_content_signature_schema, '
                'server_content_signature, parser_policy_version, parser_version, '
                'chunk_policy_version, chunk_count, started_at, finished_at, metadata) '
                'VALUES (:document_id, :version_id, :source_id, :parser_name, '
                "'complete', '', 'v1', '', '', 'server-source-content:v1', "
                ":signature, 'parser-policy:v1', 'parser:v1', 'chunk:v1', 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{}')"
            ),
            {
                'document_id': document_id,
                'version_id': version_ids[1],
                'source_id': source_id,
                'parser_name': 'verified-parser',
                'signature': 'a' * 64,
            },
        )
        ambiguous_source_id = connection.scalar(
            text(
                'INSERT INTO sources '
                '(source_type, source_id, source_url, title, permission_level, '
                'raw_metadata, server_content_signature_schema, '
                'server_content_signature, created_at) VALUES '
                "('drive', 'ambiguous-pointer', 'https://example.test/ambiguous', "
                "'Ambiguous', 'internal', '{}', 'server-source-content:v1', "
                ':signature, CURRENT_TIMESTAMP) RETURNING id'
            ),
            {'signature': 'b' * 64},
        )
        ambiguous_document_id = connection.scalar(
            text(
                'INSERT INTO documents (source_id, title, current_version) '
                "VALUES (:source_id, 'Ambiguous document', 'v1') RETURNING id"
            ),
            {'source_id': ambiguous_source_id},
        )
        ambiguous_versions = [
            connection.scalar(
                text(
                    'INSERT INTO document_versions (document_id, version, body) '
                    "VALUES (:document_id, 'v1', :body) RETURNING id"
                ),
                {'document_id': ambiguous_document_id, 'body': body},
            )
            for body in ('match-one', 'match-two')
        ]
        for ordinal, version_id in enumerate(ambiguous_versions, start=1):
            connection.execute(
                text(
                    'INSERT INTO document_parser_runs '
                    '(document_id, document_version_id, source_id, parser_name, '
                    'parser_status, mime_type, document_version_label, revision_id, '
                    'content_signature, server_content_signature_schema, '
                    'server_content_signature, parser_policy_version, parser_version, '
                    'chunk_policy_version, chunk_count, started_at, finished_at, metadata) '
                    'VALUES (:document_id, :version_id, :source_id, :parser_name, '
                    "'complete', '', 'v1', '', '', 'server-source-content:v1', "
                    ":signature, 'parser-policy:v1', 'parser:v1', 'chunk:v1', 0, "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{}')"
                ),
                {
                    'document_id': ambiguous_document_id,
                    'version_id': version_id,
                    'source_id': ambiguous_source_id,
                    'parser_name': f'ambiguous-parser-{ordinal}',
                    'signature': 'b' * 64,
                },
            )

    _run(command.upgrade, config, 'head')

    with engine.connect() as connection:
        pointer = connection.scalar(
            text(
                'SELECT current_document_version_id FROM documents WHERE id=:id'
            ),
            {'id': document_id},
        )
        ambiguous_pointer = connection.scalar(
            text(
                'SELECT current_document_version_id FROM documents WHERE id=:id'
            ),
            {'id': ambiguous_document_id},
        )
    assert pointer == version_ids[1]
    assert ambiguous_pointer is None


def test_empty_auto_review_schema_downgrades_to_runtime_foundation(
    migration_database,
) -> None:
    config, database_url = migration_database
    _run(command.upgrade, config, 'head')

    _run(command.downgrade, config, PREVIOUS_REVISION)

    engine = create_engine(database_url)
    assert AUTO_REVIEW_TABLES.isdisjoint(inspect(engine).get_table_names())
    with engine.connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == PREVIOUS_REVISION
        assert connection.scalar(
            text(
                "SELECT COUNT(*) FROM agent_runtime_schema_versions "
                "WHERE component='auto_review_trust_promotion'"
            )
        ) == 0


def test_populated_auto_review_schema_refuses_destructive_downgrade(
    migration_database,
) -> None:
    config, database_url = migration_database
    _run(command.upgrade, config, 'head')
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO auto_review_runtime_key_states '
                '(component, fingerprint_key_version, '
                'fingerprint_key_material_verifier, generation, ready, updated_at) '
                "VALUES ('auto_review_trust_promotion', 'v1', :verifier, 1, 0, CURRENT_TIMESTAMP)"
            ),
            {'verifier': 'a' * 64},
        )

    with pytest.raises(Exception, match=r'retained C\.5 state'):
        _run(command.downgrade, config, PREVIOUS_REVISION)

    with engine.connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == REVISION


def test_auto_review_schema_exposes_named_provenance_constraints(
    migration_database,
) -> None:
    config, database_url = migration_database
    _run(command.upgrade, config, 'head')
    inspector = inspect(create_engine(database_url))

    review_ref_uniques = {
        item['name']
        for item in inspector.get_unique_constraints('review_item_evidence_refs')
    }
    review_ref_fks = {
        item['name']
        for item in inspector.get_foreign_keys('review_item_evidence_refs')
    }
    assert {
        'uq_review_item_evidence_ref_source',
        'uq_review_item_evidence_ref_slot',
    } <= review_ref_uniques
    assert {
        'fk_review_item_evidence_refs_same_review_item_workflow',
        'fk_review_item_evidence_refs_same_evidence_workflow',
    } <= review_ref_fks
    assert 'fk_review_items_auto_validation_same_item' in {
        item['name'] for item in inspector.get_foreign_keys('review_items')
    }


def test_repeated_upgrade_to_head_is_idempotent(migration_database) -> None:
    config, database_url = migration_database
    _run(command.upgrade, config, 'head')
    _run(command.upgrade, config, 'head')

    with create_engine(database_url).connect() as connection:
        assert connection.scalar(
            text(
                "SELECT COUNT(*) FROM agent_runtime_schema_versions "
                "WHERE component='auto_review_trust_promotion'"
            )
        ) == 1


def test_postgresql_cutover_guards_reject_old_writer_and_accept_bound_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    workflow_id = f'c5-pg-{suffix}'
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO agent_workflow_threads '
                '(thread_id, workflow_name, graph_version, checkpoint_thread_id, '
                'checkpoint_store, owner_subject_id, security_scope_id, input_hash, '
                'evidence_version_hash, status, state_version, created_at, updated_at) '
                "VALUES (:thread_id, 'company-memory-review', "
                "'company-memory-review-v2.0', :checkpoint, 'postgres', 'owner', "
                "'default', :input_hash, :evidence_hash, 'created', 0, "
                'CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)'
            ),
            {
                'thread_id': workflow_id,
                'checkpoint': f'checkpoint-{suffix}',
                'input_hash': 'a' * 64,
                'evidence_hash': 'b' * 64,
            },
        )
        run_id = connection.scalar(
            text(
                'INSERT INTO agent_runs '
                '(agent_name, prompt_version, status, source_window, cache_key, '
                'model_name, input_tokens, output_tokens, total_tokens, '
                'estimated_cost_usd, permission_level, metadata, workflow_thread_id, '
                'started_at, generation_provider, generation_reasoning_effort, '
                'generation_route_version, generation_output_contract_version) '
                "VALUES ('timeline_agent', 'timeline:c5-v1', 'complete', 'window', "
                ":cache_key, 'gpt-5.4-mini-2026-03-17', 0, 0, 0, 0, 'internal', "
                "'{}', :thread_id, CURRENT_TIMESTAMP, 'openai', 'none', "
                "'auto-review-extraction-route:v1', 'timeline-candidate:c5-v1') "
                'RETURNING id'
            ),
            {'cache_key': f'cache-{suffix}', 'thread_id': workflow_id},
        )
        evidence_ref_id = connection.scalar(
            text(
                'INSERT INTO agent_workflow_evidence_refs '
                '(workflow_thread_id, ordinal, canonical_source_type, canonical_table, '
                'canonical_row_id, content_signature, permission_level_snapshot, '
                'content_fingerprint) VALUES '
                "(:thread_id, 1, 'drive', 'sources', 1, :signature, 'internal', "
                ':fingerprint) RETURNING id'
            ),
            {
                'thread_id': workflow_id,
                'signature': f'signature-{suffix}',
                'fingerprint': 'c' * 64,
            },
        )

    old_writer = engine.connect()
    old_tx = old_writer.begin()
    old_writer.execute(
        text(
            'INSERT INTO review_items '
            '(item_type, payload, source_links, source_snippets, confidence_score, '
            'permission_level, status, workflow_thread_id, candidate_key, created_at) '
            "VALUES ('timeline_event', '{}', '[]', '[]', 0.99, 'internal', "
            "'pending_review', :thread_id, :candidate_key, CURRENT_TIMESTAMP)"
        ),
        {'thread_id': workflow_id, 'candidate_key': f'old-{suffix}'},
    )
    with pytest.raises(DBAPIError, match='exact provenance'):
        old_tx.commit()
    old_writer.close()

    with engine.begin() as connection:
        review_item_id = connection.scalar(
            text(
                'INSERT INTO review_items '
                '(item_type, payload, source_links, source_snippets, confidence_score, '
                'permission_level, status, workflow_thread_id, candidate_key, '
                'agent_run_id, candidate_contract_version, created_at) '
                "VALUES ('timeline_event', '{}', '[]', '[]', 0.99, 'internal', "
                "'pending_review', :thread_id, :candidate_key, :run_id, 'c5-v1', "
                'CURRENT_TIMESTAMP) RETURNING id'
            ),
            {
                'thread_id': workflow_id,
                'candidate_key': f'bound-{suffix}',
                'run_id': run_id,
            },
        )
        connection.execute(
            text(
                'INSERT INTO review_item_evidence_refs '
                '(review_item_id, workflow_thread_id, workflow_evidence_ref_id, '
                'candidate_slot_ordinal, message_content_fingerprint, '
                'fingerprint_key_version, fingerprint_key_material_verifier, created_at) '
                'VALUES (:review_item_id, :thread_id, :evidence_ref_id, 1, '
                ':message_hmac, :key_version, :verifier, CURRENT_TIMESTAMP)'
            ),
            {
                'review_item_id': review_item_id,
                'thread_id': workflow_id,
                'evidence_ref_id': evidence_ref_id,
                'message_hmac': 'd' * 64,
                'key_version': 'pg-test-v1',
                'verifier': 'e' * 64,
            },
        )

    with engine.connect() as connection:
        tx = connection.begin()
        with pytest.raises(DBAPIError, match='schema boundary is immutable'):
            connection.execute(
                text(
                    "DELETE FROM agent_runtime_schema_versions "
                    "WHERE component='auto_review_trust_promotion'"
                )
            )
        tx.rollback()

    with engine.connect() as connection:
        tx = connection.begin()
        with pytest.raises(DBAPIError, match='schema boundary is immutable'):
            connection.execute(
                text(
                    "UPDATE agent_runtime_schema_versions SET schema_revision=2 "
                    "WHERE component='auto_review_trust_promotion'"
                )
            )
        tx.rollback()

    with engine.connect() as connection:
        tx = connection.begin()
        connection.execute(
            text(
                'DELETE FROM review_item_evidence_refs '
                'WHERE review_item_id=:review_item_id'
            ),
            {'review_item_id': review_item_id},
        )
        with pytest.raises(DBAPIError, match='bound ReviewItem evidence is immutable'):
            tx.commit()
        connection.close()

    for assignment in (
        "workflow_thread_id='reassigned-workflow'",
        'agent_run_id=NULL',
        'candidate_contract_version=NULL',
    ):
        with engine.connect() as connection:
            tx = connection.begin()
            with pytest.raises(DBAPIError, match='C.5 ReviewItem ownership is immutable'):
                connection.execute(
                    text(
                        f'UPDATE review_items SET {assignment} WHERE id=:review_item_id'
                    ),
                    {'review_item_id': review_item_id},
                )
            tx.rollback()

    with engine.begin() as connection:
        unattached_id = connection.scalar(
            text(
                'INSERT INTO review_items '
                '(item_type, payload, source_links, source_snippets, confidence_score, '
                'permission_level, status, created_at) VALUES '
                "('timeline_event', '{}', '[]', '[]', 0.99, 'internal', "
                "'pending_review', CURRENT_TIMESTAMP) RETURNING id"
            )
        )
    with engine.connect() as connection:
        tx = connection.begin()
        connection.execute(
            text(
                'UPDATE review_items SET workflow_thread_id=:thread_id, '
                "candidate_key=:candidate_key, agent_run_id=:run_id, "
                "candidate_contract_version='c5-v1' WHERE id=:review_item_id"
            ),
            {
                'thread_id': workflow_id,
                'candidate_key': f'attach-{suffix}',
                'run_id': run_id,
                'review_item_id': unattached_id,
            },
        )
        with pytest.raises(DBAPIError, match='exact provenance'):
            tx.commit()
        connection.close()


def test_postgresql_same_owner_composite_keys_reject_cross_workflow_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:10]
    with Session(engine) as db:
        thread_a, _, item_a = _add_workflow_candidate(db, suffix=f'{suffix}-a')
        thread_b, _, item_b = _add_workflow_candidate(db, suffix=f'{suffix}-b')
        shared_key = '5' * 64
        validation_a = _add_claimed_validation(
            db,
            thread=thread_a,
            item=item_a,
            suffix=f'{suffix}-a',
            validation_key=shared_key,
        )
        validation_b = _add_claimed_validation(
            db,
            thread=thread_b,
            item=item_b,
            suffix=f'{suffix}-b',
            validation_key=shared_key,
        )
        db.commit()
        item_a_id = item_a.id
        item_b_id = item_b.id
        validation_a_id = validation_a.id
        validation_b_id = validation_b.id
        thread_a_id = thread_a.thread_id
        thread_b_id = thread_b.thread_id

    with Session(engine) as db:
        item_a = db.get(ReviewItem, item_a_id)
        assert item_a is not None
        item_a.auto_validation_id = validation_b_id
        with pytest.raises(DBAPIError, match='fk_review_items_auto_validation_same_item'):
            db.commit()

    with Session(engine) as db:
        evidence_b = db.query(AgentWorkflowEvidenceRef).filter_by(
            workflow_thread_id=thread_b_id
        ).one()
        db.add(
            ReviewItemEvidenceRef(
                review_item_id=item_a_id,
                workflow_thread_id=thread_b_id,
                workflow_evidence_ref_id=evidence_b.id,
                candidate_slot_ordinal=2,
                message_content_fingerprint='6' * 64,
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='7' * 64,
            )
        )
        with pytest.raises(
            DBAPIError,
            match='fk_review_item_evidence_refs_same_review_item_workflow',
        ):
            db.commit()

    with engine.connect() as connection:
        tx = connection.begin()
        call_a_id = connection.scalar(
            text(
                'SELECT validation_call_id FROM auto_review_validations WHERE id=:id'
            ),
            {'id': validation_a_id},
        )
        with pytest.raises(
            DBAPIError, match='fk_auto_review_validation_call_same_workflow'
        ):
            connection.execute(
                text(
                    'INSERT INTO auto_review_validations '
                    '(review_item_id, validation_call_id, workflow_thread_id, '
                    'validation_key, evidence_version_hash, '
                    'candidate_generation_fingerprint, status, validator_provider, '
                    'validator_model, reasoning_effort, validator_prompt_version, '
                    'validator_output_contract_version, policy_version, '
                    'fingerprint_key_version, fingerprint_key_material_verifier, '
                    'cost_policy_version, confirmed_validation_cost_ceiling_usd, '
                    'claim_results, uncertainty_codes, conflict_codes, '
                    'policy_reason_codes, input_tokens, output_tokens, '
                    'estimated_cost_usd, cache_hit, created_at) VALUES '
                    '(:item_id, :call_id, :thread_id, :validation_key, '
                    ':evidence_hash, :candidate_hash, \'claimed\', \'openai\', '
                    "'gpt-5.6-terra', 'medium', 'validator:v1', "
                    "'validator-output:v1', 'policy:v1', 'v1', :verifier, "
                    "'cost:v1', 1.000000, '[]', '[]', '[]', '[]', 0, 0, 0, "
                    'false, CURRENT_TIMESTAMP)'
                ),
                {
                    'item_id': item_b_id,
                    'call_id': call_a_id,
                    'thread_id': thread_b_id,
                    'validation_key': '8' * 64,
                    'evidence_hash': '9' * 64,
                    'candidate_hash': 'a' * 64,
                    'verifier': 'b' * 64,
                },
            )
        tx.rollback()
        connection.close()

    with engine.connect() as connection:
        count = connection.scalar(
            text(
                'SELECT count(*) FROM auto_review_validations '
                'WHERE validation_key=:key AND workflow_thread_id IN (:thread_a, :thread_b)'
            ),
            {'key': shared_key, 'thread_a': thread_a_id, 'thread_b': thread_b_id},
        )
    assert count == 2


def test_postgresql_cutover_exclusive_ddl_waits_for_preexisting_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    writer = engine.connect()
    writer_tx = writer.begin()
    writer.execute(
        text(
            'INSERT INTO review_items '
            '(item_type, payload, source_links, source_snippets, confidence_score, '
            'permission_level, status, created_at) VALUES '
            "('timeline_event', '{}', '[]', '[]', 0.5, 'internal', "
            "'pending_review', CURRENT_TIMESTAMP)"
        )
    )

    def acquire_cutover_lock() -> None:
        with engine.begin() as connection:
            connection.execute(text("SET LOCAL lock_timeout = '5s'"))
            connection.execute(
                text('LOCK TABLE review_items IN ACCESS EXCLUSIVE MODE')
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(acquire_cutover_lock)
        time.sleep(0.2)
        assert future.done() is False
        writer_tx.commit()
        writer.close()
        future.result(timeout=5)


@pytest.mark.parametrize(
    ('use_other_workflow', 'agent_name'),
    [(True, 'timeline_agent'), (False, 'history_agent')],
)
def test_postgresql_extraction_call_requires_exact_agent_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    use_other_workflow: bool,
    agent_name: str,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:10]
    with Session(engine) as db:
        thread_a, run_a, _ = _add_workflow_candidate(db, suffix=f'{suffix}-a')
        thread_b, _, _ = _add_workflow_candidate(db, suffix=f'{suffix}-b')
        db.commit()
        run_id = run_a.id
        workflow_thread_id = (
            thread_b.thread_id if use_other_workflow else thread_a.thread_id
        )

    with Session(engine) as db:
        db.add(
            _claimed_extraction_call(
                run_id=run_id,
                workflow_thread_id=workflow_thread_id,
                agent_name=agent_name,
                suffix=suffix,
            )
        )
        with pytest.raises(
            DBAPIError, match='fk_auto_review_extraction_call_agent_run_identity'
        ):
            db.commit()


def test_postgresql_provider_events_require_gapless_atomic_backpointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with engine.begin() as connection:
        state_id = connection.scalar(
            text(
                'INSERT INTO auto_review_provider_safety_states '
                '(purpose, provider, model, reasoning_effort, state_version, '
                'authorized_cost_policy_version, token_estimator_version, '
                'tokenizer_encoding, reply_priming_tokens, framing_safety_tokens, '
                'input_usd_per_1m, output_usd_per_1m, breaker_open, overrun_count, '
                'authorized_at, last_event_sequence, created_at, updated_at) VALUES '
                "('validation', 'openai', :model, 'medium', 1, 'cost:v1', "
                "'estimator:v1', 'o200k_base', 16, 512, 2.000000, 12.000000, "
                'false, 0, CURRENT_TIMESTAMP, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) '
                'RETURNING id'
            ),
            {'model': f'terra-{suffix}'},
        )
        event_id = connection.scalar(
            text(
                'INSERT INTO auto_review_provider_safety_events '
                '(provider_safety_state_id, purpose, provider, model, '
                'reasoning_effort, event_sequence, event_kind, prior_state_version, '
                'new_state_version, cost_policy_version, token_estimator_version, '
                'tokenizer_encoding, reply_priming_tokens, framing_safety_tokens, '
                'input_usd_per_1m, output_usd_per_1m, prior_breaker_open, '
                'new_breaker_open, actor_subject_hmac, fingerprint_key_version, '
                'fingerprint_key_material_verifier, created_at) VALUES '
                "(:state_id, 'validation', 'openai', :model, 'medium', 1, "
                "'initial_authorized', 0, 1, 'cost:v1', 'estimator:v1', "
                "'o200k_base', 16, 512, 2.000000, 12.000000, false, false, "
                ':actor, :key_version, :verifier, CURRENT_TIMESTAMP) RETURNING id'
            ),
            {
                'state_id': state_id,
                'model': f'terra-{suffix}',
                'actor': 'a' * 64,
                'key_version': 'pg-test-v1',
                'verifier': 'b' * 64,
            },
        )
        connection.execute(
            text(
                'UPDATE auto_review_provider_safety_states SET '
                'last_event_sequence=1, last_event_id=:event_id WHERE id=:state_id'
            ),
            {'event_id': event_id, 'state_id': state_id},
        )

    with engine.connect() as connection:
        tx = connection.begin()
        with pytest.raises(DBAPIError, match='append-only'):
            connection.execute(
                text(
                    'UPDATE auto_review_provider_safety_events '
                    "SET reason_code='rewritten' WHERE id=:event_id"
                ),
                {'event_id': event_id},
            )
        tx.rollback()

    connection = engine.connect()
    tx = connection.begin()
    connection.execute(
        text(
            'INSERT INTO auto_review_provider_safety_events '
            '(provider_safety_state_id, purpose, provider, model, reasoning_effort, '
            'event_sequence, event_kind, prior_state_version, new_state_version, '
            'cost_policy_version, token_estimator_version, tokenizer_encoding, '
            'reply_priming_tokens, framing_safety_tokens, input_usd_per_1m, '
            'output_usd_per_1m, prior_breaker_open, new_breaker_open, '
            'actor_subject_hmac, fingerprint_key_version, '
            'fingerprint_key_material_verifier, created_at) VALUES '
            "(:state_id, 'validation', 'openai', :model, 'medium', 3, "
            "'breaker_cleared', 1, 2, 'cost:v2', 'estimator:v2', 'o200k_base', "
            '16, 512, 2.000000, 12.000000, false, false, :actor, '
            ':key_version, :verifier, CURRENT_TIMESTAMP)'
        ),
        {
            'state_id': state_id,
            'model': f'terra-{suffix}',
            'actor': 'c' * 64,
            'key_version': 'pg-test-v1',
            'verifier': 'b' * 64,
        },
    )
    with pytest.raises(DBAPIError, match='gapless'):
        tx.commit()
    connection.close()


def test_postgresql_provider_state_requires_initial_authorization_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        db.add(
            AutoReviewProviderSafetyState(
                purpose='extraction',
                provider='openai',
                model=f'provider-no-event-{suffix}',
                reasoning_effort='none',
                state_version=1,
                authorized_cost_policy_version='cost:v1',
                token_estimator_version='estimator:v1',
                tokenizer_encoding='o200k_base',
                reply_priming_tokens=16,
                framing_safety_tokens=512,
                input_usd_per_1m=Decimal('1.000000'),
                output_usd_per_1m=Decimal('2.000000'),
                authorized_at=datetime.now(UTC),
            )
        )

        with pytest.raises(DBAPIError, match='initial authorization event'):
            db.commit()


def test_postgresql_rollout_control_fields_cannot_change_without_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        state = AutoReviewRolloutState(
            security_scope_id=f'scope-{suffix}',
            policy_version='policy:v1',
        )
        db.add(state)
        db.commit()

        state.state_version = 1
        state.control_epoch = 1
        state.max_authorized_percentage = 10
        state.authorization_generation = 1
        state.authorization_at = datetime.now(UTC)

        with pytest.raises(DBAPIError, match='control event'):
            db.commit()


def test_postgresql_completed_validation_call_requires_exact_terminal_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        db.add(
            AutoReviewValidationCall(
                workflow_thread_id=f'validation-no-child-{suffix}',
                batch_fingerprint='a' * 64,
                status='completed',
                candidate_count=1,
                provider_attempt_count=1,
                attempt_started_at=datetime.now(UTC),
                reserved_input_tokens=100,
                reserved_output_tokens=100,
                reserved_cost_usd=Decimal('0.100000'),
                charged_input_tokens=50,
                charged_output_tokens=20,
                charged_cost_usd=Decimal('0.050000'),
                prepared_content_hmac='b' * 64,
                serialized_character_count=100,
                framed_input_token_count=50,
                max_output_tokens=100,
                token_estimator_version='estimator:v1',
                tokenizer_encoding='o200k_base',
                reply_priming_tokens=16,
                framing_safety_tokens=512,
                input_usd_per_1m=Decimal('1.000000'),
                output_usd_per_1m=Decimal('2.000000'),
                cost_policy_version='cost:v1',
                provider_safety_state_version=1,
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='c' * 64,
                workflow_extraction_cost_ceiling_usd=Decimal('1.000000'),
                workflow_validation_cost_ceiling_usd=Decimal('1.000000'),
                workflow_total_cost_ceiling_usd=Decimal('2.000000'),
                provider_timeout_seconds=30,
                provider_send_start_window_seconds=5,
                provider_attempt_lease_seconds=60,
                provider_commit_grace_seconds=5,
                budget_overrun=False,
                budget_overrun_cost_usd=Decimal('0.000000'),
                terminal_at=datetime.now(UTC),
            )
        )

        with pytest.raises(DBAPIError, match='validation terminal children'):
            db.commit()


def test_postgresql_validation_children_cannot_outpace_or_mutate_terminal_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:10]
    with Session(engine) as db:
        thread, _, item = _add_workflow_candidate(db, suffix=suffix)
        validation = _add_claimed_validation(
            db,
            thread=thread,
            item=item,
            suffix=suffix,
            validation_key='c' * 64,
        )
        db.commit()
        call_id = validation.validation_call_id
        validation_id = validation.id

    with Session(engine) as db:
        validation = db.get(AutoReviewValidation, validation_id)
        assert validation is not None
        validation.status = 'completed'
        validation.completed_at = datetime.now(UTC)
        with pytest.raises(DBAPIError, match='validation terminal children'):
            db.commit()

    with Session(engine) as db:
        call = db.get(AutoReviewValidationCall, call_id)
        validation = db.get(AutoReviewValidation, validation_id)
        assert call is not None and validation is not None
        call.status = 'completed'
        call.provider_attempt_count = 1
        call.attempt_started_at = datetime.now(UTC)
        call.charged_input_tokens = 10
        call.charged_output_tokens = 5
        call.charged_cost_usd = Decimal('0.010000')
        call.terminal_at = datetime.now(UTC)
        validation.status = 'completed'
        validation.completed_at = datetime.now(UTC)
        db.commit()

    with Session(engine) as db:
        validation = db.get(AutoReviewValidation, validation_id)
        assert validation is not None
        validation.policy_reason_codes = ['rewrite']
        with pytest.raises(DBAPIError, match='validation child is immutable'):
            db.commit()


def test_postgresql_raw_assistant_dependency_requires_exact_current_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        source = Source(
            source_type='drive',
            source_id=f'raw-lineage-{suffix}',
            source_url='https://example.test/raw',
            title='Raw source',
            permission_level='internal',
            raw_metadata={},
            server_content_signature_schema='server-source-content:v1',
            server_content_signature='a' * 64,
        )
        document = Document(source=source, title='Raw document')
        old_version = DocumentVersion(document=document, version='v1', body='old')
        current_version = DocumentVersion(
            document=document, version='v1', body='current'
        )
        db.add_all([source, document, old_version, current_version])
        db.flush()
        document.current_document_version_id = current_version.id
        parser_run = DocumentParserRun(
            document_id=document.id,
            document_version_id=old_version.id,
            source_id=source.id,
            parser_name='drive-parser',
            parser_status='complete',
            server_content_signature_schema='server-source-content:v1',
            server_content_signature='a' * 64,
            parser_policy_version='parser-policy:v1',
            parser_version='parser:v1',
            chunk_policy_version='chunk:v1',
        )
        db.add(parser_run)
        db.flush()
        chunk = DocumentChunk(
            version_id=old_version.id,
            source_id=source.id,
            parser_run_id=parser_run.id,
            chunk_index=0,
            text='old',
            source_snippet='old',
            permission_level='internal',
            metadata_={},
        )
        conversation = AssistantConversation(user_id=f'user-{suffix}')
        message = AssistantMessage(
            conversation=conversation,
            role='assistant',
            content='answer',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            evidence_contract_version='assistant-evidence:v1',
            serving_dependency_count=1,
            metadata_={},
        )
        db.add_all([chunk, message])
        db.flush()
        db.add(
            AssistantMessageEvidenceDependency(
                assistant_message_id=message.id,
                candidate_ordinal=1,
                serving_document_id=f'raw:{suffix}',
                dependency_kind='raw_chunk',
                dependency_set_hmac='d' * 64,
                serving_content_hash='e' * 64,
                permission_level='internal',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='f' * 64,
                document_chunk_id=chunk.id,
                document_version_id=old_version.id,
                source_id=source.id,
                parser_run_id=parser_run.id,
                current_document_version_id=old_version.id,
                server_content_signature_schema='server-source-content:v1',
                server_content_signature='a' * 64,
                parser_policy_version='parser-policy:v1',
                parser_version='parser:v1',
                chunk_policy_version='chunk:v1',
                legacy_human_base=False,
            )
        )

        with pytest.raises(DBAPIError, match='raw dependency is not current'):
            db.commit()


def test_postgresql_trusted_dependency_requires_complete_active_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        review_item = ReviewItem(
            item_type='timeline_event',
            payload={},
            source_links=[],
            source_snippets=[],
            confidence_score=0.99,
            permission_level='internal',
            status='approved',
            resolution_source='human',
        )
        db.add(review_item)
        db.flush()
        approval = TrustedKnowledgeApprovalLink(
            knowledge_type='timeline_event',
            knowledge_id=901,
            review_item_id=review_item.id,
            security_scope_id=f'scope-{suffix}',
            promotion_effect_kind='created',
            resolution_source='human',
            claim_fingerprint='a' * 64,
            permission_level='internal',
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
            active=True,
        )
        db.add(approval)
        db.flush()
        evidence_one = TrustedKnowledgeEvidenceLink(
            approval_link_id=approval.id,
            canonical_source_kind='drive',
            canonical_source_id=f'source-one-{suffix}',
            canonical_version_or_signature='v1',
            evidence_hash='c' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
        )
        evidence_two = TrustedKnowledgeEvidenceLink(
            approval_link_id=approval.id,
            canonical_source_kind='drive',
            canonical_source_id=f'source-two-{suffix}',
            canonical_version_or_signature='v1',
            evidence_hash='d' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
        )
        conversation = AssistantConversation(user_id=f'user-{suffix}')
        message = AssistantMessage(
            conversation=conversation,
            role='assistant',
            content='answer',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            evidence_contract_version='assistant-evidence:v1',
            serving_dependency_count=1,
            metadata_={},
        )
        db.add_all([evidence_one, evidence_two, message])
        db.flush()
        dependency = AssistantMessageEvidenceDependency(
            assistant_message_id=message.id,
            candidate_ordinal=1,
            serving_document_id=f'trusted:{suffix}',
            dependency_kind='trusted_knowledge',
            dependency_set_hmac='e' * 64,
            serving_content_hash='f' * 64,
            permission_level='internal',
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
            knowledge_type='timeline_event',
            knowledge_id=901,
            approval_link_id=approval.id,
            legacy_human_base=False,
        )
        db.add(dependency)
        db.flush()
        db.add(
            AssistantMessageKnowledgeEvidenceRef(
                dependency_id=dependency.id,
                assistant_message_id=message.id,
                approval_link_id=approval.id,
                trusted_knowledge_evidence_link_id=evidence_one.id,
            )
        )

        with pytest.raises(DBAPIError, match='complete active approval effect'):
            db.commit()


def test_postgresql_linkless_legacy_dependency_requires_relational_human_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        item = ReviewItem(
            item_type='timeline_event',
            payload={},
            source_links=[],
            source_snippets=[],
            confidence_score=0.99,
            permission_level='internal',
            status='approved',
            resolution_source='auto_review',
        )
        conversation = AssistantConversation(user_id=f'user-{suffix}')
        message = AssistantMessage(
            conversation=conversation,
            role='assistant',
            content='legacy answer',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            evidence_contract_version='assistant-evidence:v1',
            serving_dependency_count=1,
            metadata_={},
        )
        db.add_all([item, message])
        db.flush()
        db.add(
            AssistantMessageEvidenceDependency(
                assistant_message_id=message.id,
                candidate_ordinal=1,
                serving_document_id=f'legacy:{suffix}',
                dependency_kind='trusted_knowledge',
                dependency_set_hmac='a' * 64,
                serving_content_hash='b' * 64,
                permission_level='internal',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='c' * 64,
                knowledge_type='timeline_event',
                knowledge_id=999999,
                legacy_human_base=True,
                legacy_source_review_item_id=item.id,
            )
        )

        with pytest.raises(DBAPIError, match='legacy human proof'):
            db.commit()


def _add_promotion(
    db: Session,
    *,
    suffix: str,
    outcome_item_status: str = 'approved',
) -> tuple[ReviewItem, AutoReviewPromotionDecision]:
    rollout = AutoReviewRolloutState(
        security_scope_id=f'audit-scope-{suffix}',
        policy_version='policy:v1',
    )
    item = ReviewItem(
        item_type='timeline_event',
        payload={},
        source_links=[],
        source_snippets=[],
        confidence_score=0.99,
        permission_level='internal',
        status=outcome_item_status,
        resolution_source='auto_review',
    )
    db.add_all([rollout, item])
    db.flush()
    decision = AutoReviewPromotionDecision(
        review_item_id=item.id,
        security_scope_id=rollout.security_scope_id,
        policy_version=rollout.policy_version,
        rollout_authorization_generation=0,
        promotion_ordinal=1,
        requested_percentage=0,
        stored_percentage=0,
        authorized_percentage=0,
        enforce_selection_fingerprint='a' * 64,
        audit_selection_fingerprint='b' * 64,
        selection_result='first_50',
        fingerprint_key_version='v1',
        fingerprint_key_material_verifier='c' * 64,
    )
    db.add(decision)
    db.flush()
    return item, decision


def test_postgresql_promotion_and_audit_require_exact_relational_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:10]
    with Session(engine) as db:
        rollout = AutoReviewRolloutState(
            security_scope_id=f'owner-scope-{suffix}',
            policy_version='policy:v1',
        )
        item = ReviewItem(
            item_type='timeline_event',
            payload={},
            source_links=[],
            source_snippets=[],
            confidence_score=0.99,
            permission_level='internal',
        )
        db.add_all([rollout, item])
        db.commit()
        item_id = item.id

    with Session(engine) as db:
        db.add(
            AutoReviewPromotionDecision(
                review_item_id=item_id,
                security_scope_id=f'wrong-scope-{suffix}',
                policy_version='policy:v1',
                rollout_authorization_generation=0,
                promotion_ordinal=1,
                requested_percentage=0,
                stored_percentage=0,
                authorized_percentage=0,
                enforce_selection_fingerprint='a' * 64,
                audit_selection_fingerprint='b' * 64,
                selection_result='not_selected',
                fingerprint_key_version='v1',
                fingerprint_key_material_verifier='c' * 64,
            )
        )
        with pytest.raises(
            DBAPIError, match='fk_auto_review_promotion_rollout_scope'
        ):
            db.commit()

    with Session(engine) as db:
        item_a, decision_a = _add_promotion(db, suffix=f'{suffix}-a')
        _, decision_b = _add_promotion(db, suffix=f'{suffix}-b')
        db.commit()
        item_a_id = item_a.id
        decision_a_id = decision_a.id
        decision_b_id = decision_b.id

    with Session(engine) as db:
        db.add(
            AutoReviewPostAudit(
                review_item_id=item_a_id,
                promotion_decision_id=decision_b_id,
                sample_cohort='first_50',
                status='pending',
            )
        )
        with pytest.raises(DBAPIError, match='fk_auto_review_post_audit_same_promotion'):
            db.commit()

    assert decision_a_id != decision_b_id


def test_postgresql_trusted_provenance_rows_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        item = ReviewItem(
            item_type='timeline_event',
            payload={},
            source_links=[],
            source_snippets=[],
            confidence_score=0.99,
            permission_level='internal',
            status='approved',
            resolution_source='human',
        )
        db.add(item)
        db.flush()
        approval = TrustedKnowledgeApprovalLink(
            knowledge_type='timeline_event',
            knowledge_id=902,
            review_item_id=item.id,
            security_scope_id=f'scope-{suffix}',
            promotion_effect_kind='created',
            resolution_source='human',
            claim_fingerprint='d' * 64,
            permission_level='internal',
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='e' * 64,
            active=True,
        )
        db.add(approval)
        db.flush()
        evidence = TrustedKnowledgeEvidenceLink(
            approval_link_id=approval.id,
            canonical_source_kind='drive',
            canonical_source_id=f'source-{suffix}',
            canonical_version_or_signature='v1',
            evidence_hash='f' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='e' * 64,
        )
        conversation = AssistantConversation(user_id=f'immutable-user-{suffix}')
        message = AssistantMessage(
            conversation=conversation,
            role='assistant',
            content='answer',
            citations=[],
            source_ids=[],
            source_links=[],
            source_snippets=[],
            permission_level='internal',
            evidence_contract_version='assistant-evidence:v1',
            serving_dependency_count=1,
            metadata_={},
        )
        db.add_all([evidence, message])
        db.flush()
        dependency = AssistantMessageEvidenceDependency(
            assistant_message_id=message.id,
            candidate_ordinal=1,
            serving_document_id=f'trusted-immutable:{suffix}',
            dependency_kind='trusted_knowledge',
            dependency_set_hmac='1' * 64,
            serving_content_hash='2' * 64,
            permission_level='internal',
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='e' * 64,
            knowledge_type='timeline_event',
            knowledge_id=902,
            approval_link_id=approval.id,
            legacy_human_base=False,
        )
        db.add(dependency)
        db.flush()
        child = AssistantMessageKnowledgeEvidenceRef(
            dependency_id=dependency.id,
            assistant_message_id=message.id,
            approval_link_id=approval.id,
            trusted_knowledge_evidence_link_id=evidence.id,
        )
        db.add(child)
        db.commit()
        dependency_id = dependency.id
        child_id = child.id

        evidence.evidence_hash = '0' * 64
        with pytest.raises(
            DBAPIError, match='trusted provenance is immutable|append-only'
        ):
            db.commit()

    with Session(engine) as db:
        dependency = db.get(AssistantMessageEvidenceDependency, dependency_id)
        assert dependency is not None
        dependency.permission_level = 'restricted'
        with pytest.raises(DBAPIError, match='append-only'):
            db.commit()

    with Session(engine) as db:
        child = db.get(AssistantMessageKnowledgeEvidenceRef, child_id)
        assert child is not None
        db.delete(child)
        with pytest.raises(DBAPIError, match='append-only'):
            db.commit()


def test_postgresql_first_commit_revoke_snapshot_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    with Session(engine) as db:
        item = ReviewItem(
            item_type='timeline_event',
            payload={},
            source_links=[],
            source_snippets=[],
            confidence_score=0.99,
            permission_level='internal',
            status='approved',
        )
        db.add(item)
        db.commit()
        item.status = 'revoked'
        item.revoked_at = datetime.now(UTC)
        item.revoked_by_subject_hmac = 'a' * 64
        item.revoked_by_fingerprint_key_version = 'v1'
        item.revoked_by_fingerprint_key_material_verifier = 'b' * 64
        item.revoke_knowledge_remained_trusted = False
        item.revoke_document_count = 1
        db.commit()

        item.revoke_document_count = 2
        with pytest.raises(DBAPIError, match='revoke snapshot is immutable'):
            db.commit()


def test_postgresql_revoked_insert_requires_complete_first_commit_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    with Session(engine) as db:
        db.add(
            ReviewItem(
                item_type='timeline_event',
                payload={},
                source_links=[],
                source_snippets=[],
                confidence_score=0.99,
                permission_level='internal',
                status='revoked',
            )
        )

        with pytest.raises(DBAPIError, match='complete revoke snapshot'):
            db.commit()


def test_postgresql_parser_identity_must_match_source_at_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        source = Source(
            source_type='drive',
            source_id=f'parser-authority-{suffix}',
            source_url='https://example.test/parser',
            title='Parser source',
            permission_level='internal',
            raw_metadata={},
            server_content_signature_schema='server-source-content:v1',
            server_content_signature='a' * 64,
        )
        document = Document(source=source, title='Document')
        version = DocumentVersion(document=document, version='v1', body='body')
        db.add_all([source, document, version])
        db.flush()
        db.add(
            DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name='drive-parser',
                parser_status='complete',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature='b' * 64,
                parser_policy_version='parser-policy:v1',
                parser_version='parser:v1',
                chunk_policy_version='chunk:v1',
            )
        )

        with pytest.raises(DBAPIError, match='parser authority mismatch'):
            db.commit()


def test_postgresql_c5_chunk_lineage_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        source = Source(
            source_type='drive',
            source_id=f'chunk-lineage-{suffix}',
            source_url='https://example.test/chunk',
            title='Chunk source',
            permission_level='internal',
            raw_metadata={},
            server_content_signature_schema='server-source-content:v1',
            server_content_signature='a' * 64,
        )
        document = Document(source=source, title='Document')
        version = DocumentVersion(document=document, version='v1', body='body')
        db.add_all([source, document, version])
        db.flush()
        runs = [
            DocumentParserRun(
                document_id=document.id,
                document_version_id=version.id,
                source_id=source.id,
                parser_name=f'drive-parser-{ordinal}',
                parser_status='complete',
                server_content_signature_schema='server-source-content:v1',
                server_content_signature='a' * 64,
                parser_policy_version='parser-policy:v1',
                parser_version='parser:v1',
                chunk_policy_version='chunk:v1',
            )
            for ordinal in (1, 2)
        ]
        db.add_all(runs)
        db.flush()
        chunk = DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=runs[0].id,
            chunk_index=0,
            text='body',
            source_snippet='body',
            permission_level='internal',
            metadata_={},
        )
        db.add(chunk)
        db.commit()

        chunk.parser_run_id = runs[1].id
        with pytest.raises(DBAPIError, match='chunk lineage is immutable'):
            db.commit()


def test_postgresql_terminal_audit_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        item, decision = _add_promotion(db, suffix=suffix)
        audit = AutoReviewPostAudit(
            review_item_id=item.id,
            promotion_decision_id=decision.id,
            sample_cohort='first_50',
            status='completed',
            outcome='confirmed',
            auditor_subject_hmac='d' * 64,
            auditor_fingerprint_key_version='v1',
            auditor_fingerprint_key_material_verifier='e' * 64,
            audit_reason='confirmed by a human',
            audited_at=datetime.now(UTC),
        )
        db.add(audit)
        db.commit()

        audit.audit_reason = 'rewritten reason'
        with pytest.raises(DBAPIError, match='terminal audit is immutable'):
            db.commit()


def test_postgresql_correction_requires_confirmed_parent_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        item, decision = _add_promotion(db, suffix=suffix)
        audit = AutoReviewPostAudit(
            review_item_id=item.id,
            promotion_decision_id=decision.id,
            sample_cohort='first_50',
            status='completed',
            outcome='incorrect',
            auditor_subject_hmac='d' * 64,
            auditor_fingerprint_key_version='v1',
            auditor_fingerprint_key_material_verifier='e' * 64,
            audit_reason='incorrect content',
            audited_at=datetime.now(UTC),
        )
        assessment = AutoReviewRevocationAssessment(
            review_item_id=item.id,
            reason_code='incorrect_content',
            actor_subject_hmac='f' * 64,
            actor_fingerprint_key_version='v1',
            actor_fingerprint_key_material_verifier='0' * 64,
        )
        db.add_all([audit, assessment])
        db.flush()
        db.add(
            AutoReviewAuditCorrection(
                post_audit_id=audit.id,
                review_item_id=item.id,
                assessment_id=assessment.id,
                effective_outcome='incorrect',
                status='remediation_required',
                system_resolution_code='revoke_pending',
                actor_subject_hmac='1' * 64,
                actor_fingerprint_key_version='v1',
                actor_fingerprint_key_material_verifier='2' * 64,
            )
        )

        with pytest.raises(DBAPIError, match='confirmed parent audit'):
            db.commit()


def test_postgresql_provider_event_rejects_wrong_prior_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        state = AutoReviewProviderSafetyState(
            purpose='validation',
            provider='openai',
            model=f'provider-prior-{suffix}',
            reasoning_effort='medium',
            state_version=1,
            authorized_cost_policy_version='cost:v1',
            token_estimator_version='estimator:v1',
            tokenizer_encoding='o200k_base',
            reply_priming_tokens=16,
            framing_safety_tokens=512,
            input_usd_per_1m=Decimal('1.000000'),
            output_usd_per_1m=Decimal('2.000000'),
            authorized_at=datetime.now(UTC),
        )
        db.add(state)
        db.flush()
        initial = AutoReviewProviderSafetyEvent(
            provider_safety_state_id=state.id,
            purpose=state.purpose,
            provider=state.provider,
            model=state.model,
            reasoning_effort=state.reasoning_effort,
            event_sequence=1,
            event_kind='initial_authorized',
            prior_state_version=0,
            new_state_version=1,
            cost_policy_version=state.authorized_cost_policy_version,
            token_estimator_version=state.token_estimator_version,
            tokenizer_encoding=state.tokenizer_encoding,
            reply_priming_tokens=state.reply_priming_tokens,
            framing_safety_tokens=state.framing_safety_tokens,
            input_usd_per_1m=state.input_usd_per_1m,
            output_usd_per_1m=state.output_usd_per_1m,
            prior_breaker_open=False,
            new_breaker_open=False,
            actor_subject_hmac='a' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
        )
        db.add(initial)
        db.flush()
        state.last_event_sequence = 1
        state.last_event_id = initial.id
        db.commit()

        overrun = AutoReviewProviderSafetyEvent(
            provider_safety_state_id=state.id,
            purpose=state.purpose,
            provider=state.provider,
            model=state.model,
            reasoning_effort=state.reasoning_effort,
            event_sequence=2,
            event_kind='budget_overrun',
            prior_state_version=1,
            new_state_version=2,
            cost_policy_version=state.authorized_cost_policy_version,
            token_estimator_version=state.token_estimator_version,
            tokenizer_encoding=state.tokenizer_encoding,
            reply_priming_tokens=state.reply_priming_tokens,
            framing_safety_tokens=state.framing_safety_tokens,
            input_usd_per_1m=state.input_usd_per_1m,
            output_usd_per_1m=state.output_usd_per_1m,
            prior_breaker_open=True,
            new_breaker_open=True,
            call_hmac='c' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='b' * 64,
        )
        db.add(overrun)
        db.flush()
        state.state_version = 2
        state.breaker_open = True
        state.overrun_count = 1
        state.last_event_sequence = 2
        state.last_event_id = overrun.id

        with pytest.raises(DBAPIError, match='prior provider snapshot'):
            db.commit()


def test_postgresql_rollout_event_rejects_wrong_prior_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _postgres_engine(monkeypatch)
    suffix = uuid4().hex[:12]
    with Session(engine) as db:
        state = AutoReviewRolloutState(
            security_scope_id=f'rollout-prior-{suffix}',
            policy_version='policy:v1',
        )
        db.add(state)
        db.commit()
        event = AutoReviewRolloutControlEvent(
            rollout_state_id=state.id,
            security_scope_id=state.security_scope_id,
            policy_version=state.policy_version,
            event_sequence=1,
            event_kind='percentage_authorized',
            prior_state_version=0,
            new_state_version=1,
            prior_control_epoch=0,
            new_control_epoch=1,
            prior_max_authorized_percentage=10,
            new_max_authorized_percentage=10,
            prior_breaker_open=False,
            new_breaker_open=False,
            prior_authorization_generation=0,
            new_authorization_generation=1,
            actor_subject_hmac='d' * 64,
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='e' * 64,
        )
        db.add(event)
        db.flush()
        state.state_version = 1
        state.control_epoch = 1
        state.max_authorized_percentage = 10
        state.authorization_generation = 1
        state.authorization_at = datetime.now(UTC)
        state.last_event_sequence = 1
        state.last_event_id = event.id

        with pytest.raises(DBAPIError, match='prior rollout snapshot'):
            db.commit()
