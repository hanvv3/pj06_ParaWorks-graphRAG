from __future__ import annotations

import importlib.util
import inspect as pyinspect
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from backend.app.core.config import get_settings

REVISION = 'e2b3c4d5f6a7'
PREVIOUS_REVISION = 'd1a2b3c4e5f6'
MIGRATION_PATH = Path(
    'backend/migrations/versions/e2b3c4d5f6a7_add_rag_runtime_safety.py'
)
NEW_TABLES = {
    'agent_run_cost_components',
    'rag_provider_safety_authorities',
    'rag_provider_readiness',
    'rag_provider_safety_transitions',
    'rag_advisory_lock_key_registry',
}
AGENT_RUN_COLUMNS = {
    'run_contract_version',
    'run_record_phase',
    'total_charged_cost_usd',
    'projection_owner_fence_hmac',
}


@pytest.fixture
def sqlite_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Config, str]]:
    database_url = f'sqlite:///{(tmp_path / "rag-runtime.db").as_posix()}'
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', database_url)
    get_settings.cache_clear()
    try:
        yield Config('alembic.ini'), database_url
    finally:
        get_settings.cache_clear()


def _load_migration_module():
    assert MIGRATION_PATH.is_file()
    spec = importlib.util.spec_from_file_location('rag_v2_task11_migration', MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_second_d_revision_has_exact_chain_and_postgresql_guard_installers() -> None:
    migration = _load_migration_module()
    assert migration.revision == REVISION
    assert migration.down_revision == PREVIOUS_REVISION
    assert callable(migration._install_postgresql_runtime_guards)
    assert callable(migration._install_postgresql_generation_guards)


def test_sqlite_head_upgrade_is_additive_and_leaves_historical_rows_null(
    sqlite_migration: tuple[Config, str],
) -> None:
    config, database_url = sqlite_migration
    command.upgrade(config, PREVIOUS_REVISION)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO agent_runs (agent_name,prompt_version,status,source_window,"
                "cache_key,model_name,input_tokens,output_tokens,total_tokens,"
                "estimated_cost_usd,permission_level,metadata,started_at) VALUES "
                "('legacy','rag-answer:v1','complete','legacy','legacy','deterministic',"
                "0,0,0,0.0,'internal','{}',CURRENT_TIMESTAMP)"
            )
        )

    command.upgrade(config, 'head')

    schema = inspect(engine)
    assert set(schema.get_table_names()) >= NEW_TABLES
    with engine.connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == REVISION
        row = connection.execute(
            text(
                'SELECT run_contract_version, run_record_phase, '
                'total_charged_cost_usd, projection_owner_fence_hmac '
                'FROM agent_runs'
            )
        ).one()
    assert tuple(row) == (None, None, None, None)


def test_alembic_exposes_one_head() -> None:
    config = Config('alembic.ini')
    from alembic.script import ScriptDirectory

    assert ScriptDirectory.from_config(config).get_heads() == [REVISION]


def test_postgresql_guard_installers_declare_deferred_lifecycle_and_generation_guards() -> None:
    migration = _load_migration_module()
    runtime_source = pyinspect.getsource(migration._install_postgresql_runtime_guards)
    generation_source = pyinspect.getsource(
        migration._install_postgresql_generation_guards
    )

    assert 'CREATE CONSTRAINT TRIGGER rag_agent_run_v2_costs_guard' in runtime_source
    assert 'DEFERRABLE INITIALLY DEFERRED' in runtime_source
    assert 'rag_provider_safety_whole_set_guard' in runtime_source
    assert 'rag_provider_safety_transition_append_only' in runtime_source
    assert 'rag_advisory_lock_registry_append_only' in runtime_source
    assert 'pg_advisory_xact_lock' not in runtime_source

    assert 'rag_mark_corpus_generation_mutation' in generation_source
    assert 'rag_mark_vector_generation_mutation' in generation_source
    assert 'rag_require_corpus_generation_mutation' in generation_source
    assert 'rag_require_vector_generation_mutation' in generation_source
    assert 'rag_lexical_serving_projections' in generation_source
    assert 'vector_index_states' in generation_source


def test_sqlite_revision_declares_exact_new_columns_indexes_and_foreign_keys(
    sqlite_migration: tuple[Config, str],
) -> None:
    config, database_url = sqlite_migration
    command.upgrade(config, 'head')
    schema = inspect(create_engine(database_url))
    agent_columns = {item['name']: item for item in schema.get_columns('agent_runs')}
    assert set(agent_columns) >= AGENT_RUN_COLUMNS
    assert all(agent_columns[name]['nullable'] for name in AGENT_RUN_COLUMNS)
    assert agent_columns['total_charged_cost_usd']['type'].precision == 24
    assert agent_columns['total_charged_cost_usd']['type'].scale == 6

    component_fks = {
        (item['referred_table'], tuple(item['constrained_columns']))
        for item in schema.get_foreign_keys('agent_run_cost_components')
    }
    assert ('agent_runs', ('agent_run_id',)) in component_fks
    assistant_indexes = {
        item['name'] for item in schema.get_indexes('assistant_messages')
    }
    assert 'ix_assistant_messages_linked_agent_run_id' in assistant_indexes


def test_upgrade_replaces_dependency_kind_checks_for_legacy_v1_only_rows(
    sqlite_migration: tuple[Config, str],
) -> None:
    migration = _load_migration_module()
    assert '_replace_dependency_kind_checks(forward=True)' in pyinspect.getsource(
        migration.upgrade
    )
    assert '_replace_dependency_kind_checks(forward=False)' in pyinspect.getsource(
        migration.downgrade
    )

    config, database_url = sqlite_migration
    command.upgrade(config, PREVIOUS_REVISION)
    command.upgrade(config, 'head')

    checks = {
        item['name']: item['sqltext']
        for item in inspect(create_engine(database_url)).get_check_constraints(
            'assistant_message_evidence_dependencies'
        )
    }
    assert 'legacy_unbound' in checks['ck_assistant_message_dependency_kind']
    assert 'legacy_unbound' in checks['ck_assistant_message_dependency_exact_kind']


def test_sqlite_exact_downgrade_removes_only_second_revision(
    sqlite_migration: tuple[Config, str],
) -> None:
    config, database_url = sqlite_migration
    command.upgrade(config, 'head')
    command.downgrade(config, PREVIOUS_REVISION)

    engine = create_engine(database_url)
    schema = inspect(engine)
    assert NEW_TABLES.isdisjoint(schema.get_table_names())
    assert AGENT_RUN_COLUMNS.isdisjoint(
        {item['name'] for item in schema.get_columns('agent_runs')}
    )
    assert (
        engine.connect().scalar(text('SELECT version_num FROM alembic_version'))
        == PREVIOUS_REVISION
    )
