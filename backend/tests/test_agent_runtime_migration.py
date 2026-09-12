import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Engine

from backend.app.core.config import get_settings

REVISION = 'b5e6f7a8b9c0'
RUNTIME_REVISION = '2f6a8b9c0d1e'
PREVIOUS_REVISION = 'b4b6d9f4d3e1'

WORKFLOW_TABLE_COLUMNS = {
    'agent_workflow_threads': {
        'thread_id',
        'workflow_name',
        'graph_version',
        'checkpoint_thread_id',
        'checkpoint_store',
        'owner_subject_id',
        'security_scope_id',
        'client_request_id',
        'input_hash',
        'evidence_version_hash',
        'status',
        'state_version',
        'lease_token',
        'lease_expires_at',
        'checkpoint_confirmed_at',
        'cancelled_at',
        'cancelled_by_subject_id',
        'created_at',
        'updated_at',
        'completed_at',
        'expires_at',
    },
    'agent_workflow_requests': {
        'workflow_thread_id',
        'input_schema_version',
        'request_kind',
        'agent_names',
        'selection_policy_version',
        'input_hash',
        'fingerprint_key_version',
    },
    'agent_workflow_evidence_refs': {
        'id',
        'workflow_thread_id',
        'ordinal',
        'canonical_source_type',
        'canonical_table',
        'canonical_row_id',
        'document_version_id',
        'external_revision',
        'content_signature',
        'permission_level_snapshot',
        'content_fingerprint',
    },
    'agent_runtime_schema_versions': {
        'id',
        'component',
        'package_name',
        'package_version',
        'schema_revision',
        'applied_at',
    },
}

LEGACY_NEW_COLUMNS = {
    'review_items': {
        'workflow_thread_id',
        'candidate_key',
        'predecessor_review_item_id',
    },
    'agent_runs': {'workflow_thread_id', 'effect_key'},
    'decision_records': {'source_review_item_id'},
    'history_events': {'source_review_item_id'},
    'timeline_events': {'source_review_item_id'},
    'todos': {'source_review_item_id'},
}

EXPECTED_INDEXES = {
    'agent_workflow_threads': {
        'ix_agent_workflow_threads_expires_at',
        'ix_agent_workflow_threads_owner_subject_id',
        'ix_agent_workflow_threads_status',
        'ix_agent_workflow_threads_workflow_name',
        'uq_agent_workflow_thread_client_request',
    },
    'agent_workflow_evidence_refs': {
        'ix_agent_workflow_evidence_refs_workflow_thread_id',
    },
    'review_items': {'uq_review_items_workflow_candidate'},
    'agent_runs': {'uq_agent_runs_workflow_effect'},
    'decision_records': {'uq_decision_records_source_review_item'},
    'history_events': {'uq_history_events_source_review_item'},
    'timeline_events': {'uq_timeline_events_source_review_item'},
    'todos': {'uq_todos_source_review_item'},
}

LANGGRAPH_CHECKPOINT_TABLES = {
    'checkpoints',
    'checkpoint_blobs',
    'checkpoint_writes',
    'checkpoint_migrations',
}


@pytest.fixture
def migration_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Config, str]:
    database_path = tmp_path / 'agent-runtime-migration.db'
    database_url = f'sqlite:///{database_path.as_posix()}'
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    config = Config('alembic.ini')
    yield config, database_url
    get_settings.cache_clear()


def _run_alembic(
    operation: Callable[[Config, str], None],
    config: Config,
    revision: str,
) -> None:
    get_settings.cache_clear()
    try:
        operation(config, revision)
    finally:
        get_settings.cache_clear()


def _create_legacy_schema(database_url: str) -> Engine:
    engine = create_engine(database_url)
    metadata = MetaData()
    for table_name in LEGACY_NEW_COLUMNS:
        Table(
            table_name,
            metadata,
            Column('id', Integer, primary_key=True),
            Column('legacy_value', String(64), nullable=False),
        )
    metadata.create_all(bind=engine)
    with engine.begin() as connection:
        for table in metadata.sorted_tables:
            connection.execute(
                table.insert().values(id=1, legacy_value=f'{table.name}-legacy')
            )
    return engine


def _assert_runtime_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    assert WORKFLOW_TABLE_COLUMNS.keys() <= table_names
    assert LANGGRAPH_CHECKPOINT_TABLES.isdisjoint(table_names)

    for table_name, expected_columns in WORKFLOW_TABLE_COLUMNS.items():
        column_names = {
            column['name'] for column in inspector.get_columns(table_name)
        }
        assert expected_columns <= column_names

    for table_name, expected_columns in LEGACY_NEW_COLUMNS.items():
        column_names = {
            column['name'] for column in inspector.get_columns(table_name)
        }
        assert expected_columns <= column_names

    for table_name, expected_indexes in EXPECTED_INDEXES.items():
        index_names = {index['name'] for index in inspector.get_indexes(table_name)}
        assert expected_indexes <= index_names

    evidence_constraints = {
        constraint['name']
        for constraint in inspector.get_unique_constraints(
            'agent_workflow_evidence_refs'
        )
    }
    assert 'uq_agent_workflow_evidence_ref_ordinal' in evidence_constraints


def _assert_revision(engine: Engine, revision: str) -> None:
    with engine.connect() as connection:
        current_revision = connection.scalar(text('SELECT version_num FROM alembic_version'))
    assert current_revision == revision


def _assert_legacy_rows_remain(engine: Engine) -> None:
    with engine.connect() as connection:
        for table_name in LEGACY_NEW_COLUMNS:
            legacy_value = connection.scalar(
                text(f'SELECT legacy_value FROM {table_name} WHERE id = 1')
            )
            assert legacy_value == f'{table_name}-legacy'


def test_agent_runtime_migration_upgrades_fresh_schema(
    migration_database: tuple[Config, str],
) -> None:
    config, database_url = migration_database

    _run_alembic(command.upgrade, config, 'head')

    engine = create_engine(database_url)
    _assert_revision(engine, REVISION)
    _assert_runtime_schema(engine)


def test_runtime_foundation_revision_remains_independently_upgradeable(
    migration_database: tuple[Config, str],
) -> None:
    config, database_url = migration_database

    _run_alembic(command.upgrade, config, RUNTIME_REVISION)

    engine = create_engine(database_url)
    _assert_revision(engine, RUNTIME_REVISION)
    _assert_runtime_schema(engine)


def test_agent_runtime_migration_preserves_application_loggers(
    migration_database: tuple[Config, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _database_url = migration_database
    application_logger = logging.getLogger('AssistantTool')
    monkeypatch.setattr(application_logger, 'disabled', False)

    _run_alembic(command.upgrade, config, 'head')

    assert application_logger.disabled is False


def test_agent_runtime_migration_upgrades_existing_schema_idempotently(
    migration_database: tuple[Config, str],
) -> None:
    # This intentionally partial fixture has no Assistant tables; exercise its
    # original migration boundary. Full-schema head coverage is tested separately.
    config, database_url = migration_database
    engine = _create_legacy_schema(database_url)
    _run_alembic(command.stamp, config, PREVIOUS_REVISION)

    _run_alembic(command.upgrade, config, 'a4d5e6f7b8c9')

    _assert_revision(engine, 'a4d5e6f7b8c9')
    _assert_runtime_schema(engine)
    _assert_legacy_rows_remain(engine)

    _run_alembic(command.upgrade, config, 'a4d5e6f7b8c9')

    _assert_revision(engine, 'a4d5e6f7b8c9')
    _assert_runtime_schema(engine)
    _assert_legacy_rows_remain(engine)


def test_agent_runtime_migration_downgrade_removes_only_runtime_foundation(
    migration_database: tuple[Config, str],
) -> None:
    # This intentionally partial fixture has no Assistant tables; exercise its
    # original migration boundary. Full-schema head coverage is tested separately.
    config, database_url = migration_database
    engine = _create_legacy_schema(database_url)
    _run_alembic(command.stamp, config, PREVIOUS_REVISION)
    _run_alembic(command.upgrade, config, 'a4d5e6f7b8c9')
    _assert_revision(engine, 'a4d5e6f7b8c9')

    _run_alembic(command.downgrade, config, PREVIOUS_REVISION)

    _assert_revision(engine, PREVIOUS_REVISION)
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    assert WORKFLOW_TABLE_COLUMNS.keys().isdisjoint(table_names)
    for table_name, removed_columns in LEGACY_NEW_COLUMNS.items():
        column_names = {
            column['name'] for column in inspector.get_columns(table_name)
        }
        assert removed_columns.isdisjoint(column_names)
        index_names = {index['name'] for index in inspector.get_indexes(table_name)}
        assert EXPECTED_INDEXES[table_name].isdisjoint(index_names)
    _assert_legacy_rows_remain(engine)
