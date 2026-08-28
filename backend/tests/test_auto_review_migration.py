import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError

from backend.app.core.config import get_settings

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
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('set PARAWORKS_TEST_POSTGRES_URL to an isolated PostgreSQL test database')
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        get_settings.cache_clear()

    engine = create_engine(database_url)
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


def test_postgresql_provider_events_require_gapless_atomic_backpointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('set PARAWORKS_TEST_POSTGRES_URL to an isolated PostgreSQL test database')
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        get_settings.cache_clear()
    engine = create_engine(database_url)
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
