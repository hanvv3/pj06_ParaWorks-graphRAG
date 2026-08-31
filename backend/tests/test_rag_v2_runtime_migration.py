from __future__ import annotations

import importlib.util
import inspect as pyinspect
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError

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


def test_postgresql_runtime_installer_emits_complete_relational_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    statements: list[str] = []
    bind = type('Bind', (), {'dialect': type('Dialect', (), {'name': 'postgresql'})()})()
    monkeypatch.setattr(migration.op, 'get_bind', lambda: bind)
    monkeypatch.setattr(migration.op, 'execute', lambda statement: statements.append(str(statement)))

    migration._install_postgresql_runtime_guards()
    emitted = '\n'.join(statements)

    assert 'rag_validate_agent_run_v2_costs_for' in emitted
    assert 'OLD.agent_run_id IS DISTINCT FROM NEW.agent_run_id' in emitted
    assert 'PERFORM rag_validate_agent_run_v2_costs_for(OLD.agent_run_id)' in emitted
    assert 'PERFORM rag_validate_agent_run_v2_costs_for(NEW.agent_run_id)' in emitted
    assert "parent_phase = 'admission'" in emitted
    assert "parent_phase = 'cost_finalized_pending_projection'" in emitted
    assert "parent_phase = 'final'" in emitted
    assert "parent_phase = 'admission_only'" in emitted
    assert "metadata ->> 'outcome'" in emitted
    assert 'abandoned_count <> 0 OR terminal_count = 2' in emitted
    assert 'parent_projection_fence IS NULL OR' in emitted
    assert 'not_attempted_count <> 0 OR dispatching_count <> 0' in emitted
    assert 'abandoned_count = 0 OR terminal_count + abandoned_count <> 2' in emitted
    assert 'transition.prior_state_version IS DISTINCT FROM' in emitted
    assert 'prior_transition.new_state_version' in emitted
    assert 'transition.new_state_version IS DISTINCT FROM readiness.state_version' in emitted
    assert 'transition.envelope_digest <> authority.envelope_digest' in emitted
    assert 'transition.reviewed_transition_reference_hmac IS DISTINCT FROM' in emitted
    assert 'readiness.reviewed_gate_reference_hmac' in emitted
    assert 'provider readiness family identity is immutable' in emitted
    assert 'rag_validate_assistant_integrity_for' in emitted
    assert 'serving_dependency_count' in emitted
    assert 'fingerprint_key_version IS DISTINCT FROM' in emitted
    assert 'rag_assistant_integrity_guard_linked_run' in emitted
    assert "linked.run_contract_version <> 'rag-run:v2'" in emitted
    assert "linked.metadata ->> 'rag_result_hmac' IS DISTINCT FROM" in emitted


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


@pytest.fixture
def postgres_runtime_migration(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    database_url = os.getenv('PARAWORKS_TEST_POSTGRES_URL')
    if not database_url:
        pytest.skip('PARAWORKS_TEST_POSTGRES_URL is unavailable for Task 11 guards')
    parsed = make_url(database_url)
    if parsed.host != '127.0.0.1' or parsed.port != 55432:
        pytest.fail('Task 11 PostgreSQL checks require disposable 127.0.0.1:55432')
    admin = create_engine(database_url)
    schema_name = f'rag_task11_{uuid4().hex}'
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema_name}'))
    query = dict(parsed.query)
    query['options'] = f'-csearch_path={schema_name},public'
    isolated_url = parsed.set(query=query).render_as_string(hide_password=False)
    monkeypatch.setenv('PARAWORKS_DEMO_MODE', 'false')
    monkeypatch.setenv('PARAWORKS_DATABASE_URL', isolated_url)
    get_settings.cache_clear()
    command.upgrade(Config('alembic.ini'), 'head')
    engine = create_engine(isolated_url)
    try:
        yield engine
    finally:
        engine.dispose()
        get_settings.cache_clear()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema_name} CASCADE'))
        admin.dispose()


def _pg_parent(
    connection,
    *,
    phase: str = 'admission',
    status: str = 'running',
    outcome: str | None = None,
    completed_at: datetime | None = None,
    fence: str | None = None,
) -> int:
    return connection.scalar(
        text(
            'INSERT INTO agent_runs '
            '(agent_name,prompt_version,status,source_window,cache_key,model_name,'
            'input_tokens,output_tokens,total_tokens,estimated_cost_usd,permission_level,'
            'metadata,started_at,completed_at,run_contract_version,run_record_phase,'
            'total_charged_cost_usd,projection_owner_fence_hmac) VALUES '
            "('rag_orchestrator_agent','rag-answer:v2',:status,'runtime-test','runtime-test',"
            "'rag-v2-admission',0,0,0,0.0,'restricted',CAST(:metadata AS json),"
            "CURRENT_TIMESTAMP,:completed_at,'rag-run:v2',:phase,0.000000,:fence) "
            'RETURNING id'
        ),
        {
            'status': status,
            'metadata': '{}' if outcome is None else '{"outcome":"' + outcome + '"}',
            'completed_at': completed_at,
            'phase': phase,
            'fence': fence,
        },
    )


def _pg_component(connection, parent_id: int, component: str, state: str) -> int:
    ordinal = 0 if component == 'query_embedding' else 1
    model = 'text-embedding-3-small' if ordinal == 0 else 'gpt-5.4-mini-2026-03-17'
    config = 'rag-query-embedding-config:v1' if ordinal == 0 else 'rag-answer-model-config:v1'
    cost = 'rag-query-embedding-cost:v1' if ordinal == 0 else 'rag-answer-cost:v1'
    estimator = (
        'openai-cl100k-text-embedding-3-small:v1'
        if ordinal == 0
        else 'openai-o200k-rag-answer:v1'
    )
    return connection.scalar(
        text(
            'INSERT INTO agent_run_cost_components '
            '(agent_run_id,component,component_ordinal,dispatch_state,attempted,'
            'dispatch_count,reserved_input_tokens,reserved_output_tokens,'
            'actual_input_tokens,actual_output_tokens,reserved_cost_usd,charged_cost_usd,'
            'charge_basis,overrun,provider,model,authorized_model_config_version,'
            'authorized_model_config_snapshot_hmac,authorized_cost_policy_version,'
            'authorized_token_estimator_version,authorized_policy_snapshot_hmac,'
            'terminal_outcome,created_at,updated_at) VALUES '
            '(:parent,:component,:ordinal,:state,false,0,0,0,NULL,NULL,0.000000,'
            "0.000000,'zero',false,'openai',:model,:config,:config_hmac,:cost,"
            ':estimator,:policy_hmac,NULL,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP) RETURNING id'
        ),
        {
            'parent': parent_id,
            'component': component,
            'ordinal': ordinal,
            'state': state,
            'model': model,
            'config': config,
            'config_hmac': 'a' * 64,
            'cost': cost,
            'estimator': estimator,
            'policy_hmac': 'b' * 64,
        },
    )


def test_postgresql_runtime_relational_guards_reject_confirmed_bypasses(
    postgres_runtime_migration: Engine,
) -> None:
    engine = postgres_runtime_migration
    with engine.begin() as connection:
        source_parent = _pg_parent(connection)
        query_id = _pg_component(connection, source_parent, 'query_embedding', 'not_attempted')
        _pg_component(connection, source_parent, 'answer_generation', 'not_attempted')

    with pytest.raises(IntegrityError), engine.begin() as connection:
        target_parent = _pg_parent(connection)
        _pg_component(connection, target_parent, 'answer_generation', 'not_attempted')
        connection.execute(
            text('UPDATE agent_run_cost_components SET agent_run_id=:target WHERE id=:child'),
            {'target': target_parent, 'child': query_id},
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        invalid_final = _pg_parent(
            connection,
            phase='final',
            status='complete',
            outcome='supported_answer',
            completed_at=datetime.now(UTC),
        )
        _pg_component(connection, invalid_final, 'query_embedding', 'not_attempted')
        _pg_component(connection, invalid_final, 'answer_generation', 'not_attempted')

    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO rag_provider_safety_authorities '
                '(id,authority_uuid,designated_environment_id,global_safety_generation,'
                'envelope_digest,fingerprint_key_version,fingerprint_key_material_verifier,'
                'created_at,updated_at) VALUES '
                "(1,'00000000-0000-0000-0000-000000000001','test',1,:digest,'key',"
                ':verifier,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)'
            ),
            {'digest': 'c' * 64, 'verifier': 'd' * 64},
        )
        readiness_ids: list[int] = []
        for component, model, config, cost, estimator, review in (
            ('query_embedding', 'text-embedding-3-small', 'rag-query-embedding-config:v1', 'rag-query-embedding-cost:v1', 'openai-cl100k-text-embedding-3-small:v1', '1' * 64),
            ('answer_generation', 'gpt-5.4-mini-2026-03-17', 'rag-answer-model-config:v1', 'rag-answer-cost:v1', 'openai-o200k-rag-answer:v1', '2' * 64),
        ):
            readiness_ids.append(
                connection.scalar(
                    text(
                        'INSERT INTO rag_provider_readiness '
                        '(authority_id,component,provider,model,reasoning_or_config_identity,'
                        'active,authorized_model_config_version,'
                        'authorized_model_config_snapshot_hmac,authorized_cost_policy_version,'
                        'authorized_token_estimator_version,authorized_fingerprint_key_version,'
                        'authorized_fingerprint_key_material_verifier,'
                        'authorized_policy_snapshot_hmac,state,state_version,'
                        'family_safety_generation,reviewed_gate_reference_hmac,created_at,updated_at) '
                        "VALUES (1,:component,'openai',:model,:config,true,:config,:config_hmac,"
                        ":cost,:estimator,'key',:verifier,:policy,'ready',1,0,:review,"
                        'CURRENT_TIMESTAMP,CURRENT_TIMESTAMP) RETURNING id'
                    ),
                    {'component': component, 'model': model, 'config': config, 'config_hmac': 'e' * 64, 'cost': cost, 'estimator': estimator, 'verifier': 'd' * 64, 'policy': 'f' * 64, 'review': review},
                )
            )
        for generation, readiness_id, review in ((0, readiness_ids[0], '1' * 64), (1, readiness_ids[1], '2' * 64)):
            connection.execute(
                text(
                    'INSERT INTO rag_provider_safety_transitions '
                    '(authority_id,readiness_id,global_safety_generation,transition_kind,'
                    'prior_state,new_state,prior_state_version,new_state_version,'
                    'prior_family_safety_generation,new_family_safety_generation,'
                    'envelope_digest,reviewed_transition_reference_hmac,created_at) VALUES '
                    "(1,:readiness,:generation,'bootstrap',NULL,'ready',NULL,1,NULL,0,"
                    ':digest,:review,CURRENT_TIMESTAMP)'
                ),
                {'readiness': readiness_id, 'generation': generation, 'digest': 'c' * 64, 'review': review},
            )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE rag_provider_readiness SET state='rebind_required', "
                'state_version=2,family_safety_generation=1 WHERE id=:id'
            ),
            {'id': readiness_ids[0]},
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        linked = _pg_parent(
            connection,
            phase='final',
            status='complete',
            outcome='supported_answer',
            completed_at=datetime.now(UTC),
        )
        _pg_component(connection, linked, 'query_embedding', 'terminal')
        _pg_component(connection, linked, 'answer_generation', 'terminal')
        connection.execute(
            text('UPDATE agent_runs SET metadata=CAST(:metadata AS json) WHERE id=:id'),
            {
                'id': linked,
                'metadata': '{"outcome":"supported_answer","rag_result_hmac":"'
                + 'a' * 64
                + '"}',
            },
        )
        conversation_id = connection.scalar(
            text(
                "INSERT INTO assistant_conversations (user_id,title,created_at,updated_at) "
                "VALUES ('owner','title',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP) RETURNING id"
            )
        )
        connection.execute(
            text(
                'INSERT INTO assistant_messages '
                '(conversation_id,role,content,citations,source_ids,source_links,'
                'source_snippets,hidden_match_count,evidence_contract_version,'
                'serving_dependency_count,content_write_mode,content_hmac_schema_version,'
                'assistant_message_content_hmac,content_hmac_key_version,'
                'content_hmac_key_material_verifier,content_origin,content_origin_hmac,'
                'rag_result_hmac,linked_agent_run_id,dependency_set_hmac_schema_version,'
                'dependency_set_hmac,parent_selected_evidence_projection_hmac,'
                'model_influence_set_hmac,metadata,created_at) VALUES '
                "(:conversation,'assistant','answer','[]','[]','[]','[]',0,"
                "'assistant-evidence:v1',1,'rag_v2_exact',"
                "'assistant-message-content-hmac:v1',:hmac,'key',:hmac,"
                "'rag_assembled',:hmac,:hmac,:run,'assistant-dependency-set-hmac:v2',"
                ':hmac,:hmac,:hmac,CAST(\'{}\' AS json),CURRENT_TIMESTAMP)'
            ),
            {'conversation': conversation_id, 'run': linked, 'hmac': 'a' * 64},
        )
