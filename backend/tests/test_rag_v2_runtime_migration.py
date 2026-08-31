from __future__ import annotations

import importlib.util
import inspect as pyinspect
import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError

from backend.app.agent_runtime.rag_advisory_locks import (
    RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
    load_registered_advisory_capability,
    register_advisory_identity_db,
)
from backend.app.agent_runtime.rag_provider_safety import (
    RagProviderSafetyError,
    RagProviderSafetyReviewAuthority,
    RagProviderSafetyService,
)
from backend.app.core.config import get_settings
from backend.tests.test_rag_v2_costs import _snapshot

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
    spec = importlib.util.spec_from_file_location(
        'rag_v2_task11_migration', MIGRATION_PATH
    )
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
                'INSERT INTO agent_runs (agent_name,prompt_version,status,source_window,'
                'cache_key,model_name,input_tokens,output_tokens,total_tokens,'
                'estimated_cost_usd,permission_level,metadata,started_at) VALUES '
                "('legacy','rag-answer:v1','complete','legacy','legacy','deterministic',"
                "0,0,0,0.0,'internal','{}',CURRENT_TIMESTAMP)"
            )
        )

    command.upgrade(config, 'head')

    schema = inspect(engine)
    assert set(schema.get_table_names()) >= NEW_TABLES
    with engine.connect() as connection:
        assert (
            connection.scalar(text('SELECT version_num FROM alembic_version'))
            == REVISION
        )
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


def test_postgresql_guard_installers_declare_deferred_lifecycle_and_generation_guards() -> (
    None
):
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
    assert 'authority-wide generation-0 bootstrap' in runtime_source
    assert 'bootstrap iff generation zero' in runtime_source
    assert 'first ordinary family mutation must bind bootstrap' in runtime_source
    assert 'provider supersession requires a new active family' in runtime_source
    assert 'readiness_count <> 2 + supersession_count' in runtime_source
    assert (
        'NEW.family_safety_generation <= OLD.family_safety_generation'
        in runtime_source
    )
    assert 'OLD.active IS TRUE AND NEW.active IS FALSE' in runtime_source
    assert 'targeted family transition state mismatch' in runtime_source
    assert 'untouched bootstrap family drift' in runtime_source
    assert 'new_state_version IS DISTINCT FROM 1' in runtime_source

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
    bind = type(
        'Bind', (), {'dialect': type('Dialect', (), {'name': 'postgresql'})()}
    )()
    monkeypatch.setattr(migration.op, 'get_bind', lambda: bind)
    monkeypatch.setattr(
        migration.op, 'execute', lambda statement: statements.append(str(statement))
    )

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
    assert (
        'transition.new_state_version IS DISTINCT FROM readiness.state_version'
        in emitted
    )
    assert 'transition.envelope_digest <> authority.envelope_digest' in emitted
    assert 'transition.reviewed_transition_reference_hmac IS DISTINCT FROM' in emitted
    assert 'readiness.reviewed_gate_reference_hmac' in emitted
    assert 'provider readiness family identity is immutable' in emitted
    assert 'rag_validate_assistant_integrity_for' in emitted
    assert 'serving_dependency_count' in emitted
    assert 'fingerprint_key_version IS DISTINCT FROM' in emitted
    assert 'rag_assistant_integrity_guard_linked_run' in emitted
    assert emitted.rfind(
        'CREATE CONSTRAINT TRIGGER rag_assistant_integrity_guard_linked_run'
    ) > emitted.rfind('DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_linked_run')
    assert "linked.run_contract_version IS DISTINCT FROM 'rag-run:v2'" in emitted
    assert "dependency_serving_scope IS DISTINCT FROM 'rag_v2'" in emitted
    assert "dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only'" in emitted
    assert "dependency_role IS DISTINCT FROM 'selected_citation'" in emitted
    assert "linked.metadata ->> 'rag_result_hmac' IS DISTINCT FROM" in emitted


def test_postgresql_runtime_downgrade_drops_linked_run_trigger_before_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    statements: list[str] = []
    bind = type(
        'Bind', (), {'dialect': type('Dialect', (), {'name': 'postgresql'})()}
    )()
    monkeypatch.setattr(migration.op, 'get_bind', lambda: bind)
    monkeypatch.setattr(
        migration.op, 'execute', lambda statement: statements.append(str(statement))
    )

    migration._drop_postgresql_runtime_guards()
    emitted = '\n'.join(statements)
    trigger_drop = emitted.index(
        'DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_linked_run'
    )
    function_drop = emitted.index(
        'DROP FUNCTION IF EXISTS rag_validate_assistant_integrity()'
    )
    assert trigger_drop < function_drop


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


def test_postgresql_provider_safety_service_commits_one_whole_set_history_per_generation(
    postgres_runtime_migration: Engine,
    tmp_path: Path,
) -> None:
    engine = postgres_runtime_migration
    with engine.connect() as connection:
        assert register_advisory_identity_db(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        ) is None
        connection.commit()
        capability = load_registered_advisory_capability(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
        connection.rollback()
    service = RagProviderSafetyService(
        latch_path=tmp_path / 'provider-safety.json',
        identity_secret=b'task12-postgres-provider-safety-secret',
        designated_environment_id='task12-postgres',
        advisory_capability=capability,
    )
    snapshots = (_snapshot('query_embedding'), _snapshot('answer_generation'))
    with engine.connect() as connection:
        service.bootstrap(
            connection,
            snapshots,
            reviewed_transition_reference_hmac='9' * 64,
        )
        review = RagProviderSafetyReviewAuthority(
            identity_secret=b'task12-postgres-provider-safety-secret'
        )
        command = review.issue(
            service.review_context(connection, 'answer_generation'),
            operation='mark_rebind_required',
            successor=None,
            actor_subject_hmac='4' * 64,
            reviewed_gate_reference_hmac='1' * 64,
            historical_block_acknowledged=False,
        )
        service.mark_rebind_required(connection, command)
        rebound = replace(
            snapshots[1],
            authorized_model_config_snapshot_hmac='d' * 64,
            authorized_policy_snapshot_hmac='e' * 64,
        )
        command = review.issue(
            service.review_context(connection, 'answer_generation'),
            operation='rebind',
            successor=rebound,
            actor_subject_hmac='4' * 64,
            reviewed_gate_reference_hmac='2' * 64,
            historical_block_acknowledged=False,
        )
        service.reviewed_rebind(connection, command, rebound)
        successor = replace(
            rebound,
            model='gpt-5.6-luna',
            reasoning_or_config_identity='reasoning:low',
            authorized_policy_snapshot_hmac='f' * 64,
        )
        command = review.issue(
            service.review_context(connection, 'answer_generation'),
            operation='supersession',
            successor=successor,
            actor_subject_hmac='4' * 64,
            reviewed_gate_reference_hmac='3' * 64,
            historical_block_acknowledged=False,
        )
        service.reviewed_supersession(connection, command, successor)
        assert (
            service.require_ready(
                connection, 'answer_generation', successor
            ).global_safety_generation
            == 3
        )
        assert connection.execute(
            text(
                'SELECT global_safety_generation FROM '
                'rag_provider_safety_transitions ORDER BY global_safety_generation'
            )
        ).scalars().all() == [0, 1, 2, 3]


def test_postgresql_external_write_failure_leaves_db_only_fail_stop_generation(
    postgres_runtime_migration: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = postgres_runtime_migration
    with engine.connect() as connection:
        register_advisory_identity_db(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
        connection.commit()
        capability = load_registered_advisory_capability(
            connection,
            RAG_PROVIDER_SAFETY_AUTHORITY_LOCK_ID,
            identity_namespace='static',
        )
        connection.rollback()
    path = tmp_path / 'provider-safety-remediation.json'
    service = RagProviderSafetyService(
        latch_path=path,
        identity_secret=b'task12-postgres-provider-safety-secret',
        designated_environment_id='task12-postgres',
        advisory_capability=capability,
    )
    snapshots = (_snapshot('query_embedding'), _snapshot('answer_generation'))
    with engine.connect() as connection:
        service.bootstrap(
            connection,
            snapshots,
            reviewed_transition_reference_hmac='9' * 64,
        )
        monkeypatch.setattr(
            service._authority,
            '_replace_unlocked',
            lambda _value: (_ for _ in ()).throw(
                OSError('simulated external write failure')
            ),
        )
        with pytest.raises(RagProviderSafetyError, match='external transition failed'):
            service.block_remediation(
                connection,
                'answer_generation',
                category='provider_response_identity_invalid',
                agent_run_id=23,
                input_tokens=1,
                output_tokens=1,
                cost_usd=Decimal('0.000001'),
            )
        restarted = RagProviderSafetyService(
            latch_path=path,
            identity_secret=b'task12-postgres-provider-safety-secret',
            designated_environment_id='task12-postgres',
            advisory_capability=capability,
        )
        provider_calls: list[str] = []
        with pytest.raises(RagProviderSafetyError, match='drift'):
            restarted.require_ready(
                connection, 'answer_generation', snapshots[1]
            )
        assert provider_calls == []


def _pg_component(connection, parent_id: int, component: str, state: str) -> int:
    ordinal = 0 if component == 'query_embedding' else 1
    model = 'text-embedding-3-small' if ordinal == 0 else 'gpt-5.4-mini-2026-03-17'
    config = (
        'rag-query-embedding-config:v1'
        if ordinal == 0
        else 'rag-answer-model-config:v1'
    )
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


def _pg_conversation(connection) -> int:
    return connection.scalar(
        text(
            'INSERT INTO assistant_conversations (user_id,title,created_at,updated_at) '
            "VALUES ('owner','title',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP) RETURNING id"
        )
    )


def _pg_integrity_message(
    connection,
    *,
    conversation_id: int,
    origin: str,
    linked_run_id: int | None,
    dependency_count: int,
) -> int:
    rag_v2 = origin != 'legacy_evidence'
    return connection.scalar(
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
            "'assistant-evidence:v1',:dependency_count,:mode,"
            "'assistant-message-content-hmac:v1',:hmac,'key',:hmac,"
            ':origin,:hmac,:rag_result,:linked_run,'
            "'assistant-dependency-set-hmac:v2',:hmac,:hmac,:influence,"
            "CAST('{}' AS json),CURRENT_TIMESTAMP) RETURNING id"
        ),
        {
            'conversation': conversation_id,
            'dependency_count': dependency_count,
            'mode': 'rag_v2_exact' if rag_v2 else 'legacy_trimmed',
            'hmac': 'a' * 64,
            'origin': origin,
            'rag_result': 'a' * 64 if rag_v2 else None,
            'linked_run': linked_run_id,
            'influence': 'a' * 64 if rag_v2 else None,
        },
    )


def _pg_dependency(
    connection,
    *,
    message_id: int,
    ordinal: int,
    parent_origin: str,
    historical_null_scope: bool,
) -> None:
    legacy = parent_origin == 'legacy_evidence'
    dependency_kind = 'legacy_unbound' if legacy else 'trusted_knowledge'
    scope = (
        None if historical_null_scope else ('legacy_v1_only' if legacy else 'rag_v2')
    )
    role = None if historical_null_scope else 'selected_citation'
    connection.execute(
        text(
            'INSERT INTO assistant_message_evidence_dependencies '
            '(assistant_message_id,candidate_ordinal,serving_document_id,dependency_kind,'
            'dependency_set_hmac,serving_content_hash,permission_level,'
            'fingerprint_key_version,fingerprint_key_material_verifier,knowledge_type,'
            'knowledge_id,legacy_human_base,dependency_serving_scope,dependency_role,'
            'dependency_child_hmac,legacy_dependency_identity_hmac,model_content_hmac,'
            'canonical_citation_projection_hmac,selected_v1_citation_projection_hmac,'
            'serving_identity_hmac,serving_version_fingerprint,support_mode,created_at) '
            'VALUES (:message,:ordinal,:document,:kind,:hmac,:content_hash,'
            "'internal','key',:hmac,:knowledge_type,:knowledge_id,:legacy_human,"
            ':scope,:role,:child_hmac,:legacy_hmac,:model_hmac,:citation_hmac,'
            ':selected_hmac,:serving_hmac,:version_hmac,:support,CURRENT_TIMESTAMP)'
        ),
        {
            'message': message_id,
            'ordinal': ordinal,
            'document': f'dependency:{parent_origin}:{ordinal}',
            'kind': dependency_kind,
            'hmac': 'a' * 64,
            'content_hash': 'b' * 64,
            'knowledge_type': None if legacy else 'decision',
            'knowledge_id': None if legacy else ordinal + 1,
            'legacy_human': not legacy,
            'scope': scope,
            'role': role,
            'child_hmac': None if historical_null_scope else 'c' * 64,
            'legacy_hmac': ('d' * 64 if legacy and not historical_null_scope else None),
            'model_hmac': None if historical_null_scope else 'e' * 64,
            'citation_hmac': None if historical_null_scope else 'f' * 64,
            'selected_hmac': None if historical_null_scope else '1' * 64,
            'serving_hmac': (
                '2' * 64 if not legacy and not historical_null_scope else None
            ),
            'version_hmac': (
                '3' * 64 if not legacy and not historical_null_scope else None
            ),
            'support': None if legacy or historical_null_scope else 'trusted_fact',
        },
    )


def test_postgresql_runtime_relational_guards_reject_confirmed_bypasses(
    postgres_runtime_migration: Engine,
) -> None:
    engine = postgres_runtime_migration
    with engine.connect() as connection:
        assert (
            connection.scalar(
                text(
                    'SELECT count(*) FROM pg_trigger WHERE tgname = '
                    "'rag_assistant_integrity_guard_linked_run' AND NOT tgisinternal"
                )
            )
            == 1
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        legacy_run = connection.scalar(
            text(
                'INSERT INTO agent_runs '
                '(agent_name,prompt_version,status,source_window,cache_key,model_name,'
                'input_tokens,output_tokens,total_tokens,estimated_cost_usd,permission_level,'
                'metadata,started_at,completed_at) VALUES '
                "('legacy','rag-answer:v1','complete','legacy','legacy','deterministic',"
                "0,0,0,0.0,'internal',CAST(:metadata AS json),CURRENT_TIMESTAMP,"
                'CURRENT_TIMESTAMP) RETURNING id'
            ),
            {'metadata': '{"rag_result_hmac":"' + 'a' * 64 + '"}'},
        )
        conversation_id = _pg_conversation(connection)
        connection.execute(
            text(
                'INSERT INTO assistant_messages '
                '(conversation_id,role,content,citations,source_ids,source_links,'
                'source_snippets,hidden_match_count,evidence_contract_version,'
                'serving_dependency_count,content_write_mode,content_hmac_schema_version,'
                'assistant_message_content_hmac,content_hmac_key_version,'
                'content_hmac_key_material_verifier,content_origin,content_origin_hmac,'
                'rag_result_hmac,linked_agent_run_id,metadata,created_at) VALUES '
                "(:conversation,'assistant','canned','[]','[]','[]','[]',0,'none-v1',0,"
                "'rag_v2_exact','assistant-message-content-hmac:v1',:hmac,'key',:hmac,"
                "'rag_canned',:hmac,:hmac,:run,CAST('{}' AS json),CURRENT_TIMESTAMP)"
            ),
            {'conversation': conversation_id, 'hmac': 'a' * 64, 'run': legacy_run},
        )

    for origin in ('rag_assembled', 'legacy_evidence'):
        with pytest.raises(IntegrityError), engine.begin() as connection:
            linked_run_id = None
            if origin == 'rag_assembled':
                linked_run_id = _pg_parent(
                    connection,
                    phase='final',
                    status='complete',
                    outcome='supported_answer',
                    completed_at=datetime.now(UTC),
                )
                _pg_component(connection, linked_run_id, 'query_embedding', 'terminal')
                _pg_component(
                    connection, linked_run_id, 'answer_generation', 'terminal'
                )
                connection.execute(
                    text(
                        'UPDATE agent_runs SET metadata=CAST(:metadata AS json) WHERE id=:id'
                    ),
                    {
                        'id': linked_run_id,
                        'metadata': '{"outcome":"supported_answer",'
                        '"rag_result_hmac":"' + 'a' * 64 + '"}',
                    },
                )
            message_id = _pg_integrity_message(
                connection,
                conversation_id=_pg_conversation(connection),
                origin=origin,
                linked_run_id=linked_run_id,
                dependency_count=2,
            )
            _pg_dependency(
                connection,
                message_id=message_id,
                ordinal=0,
                parent_origin=origin,
                historical_null_scope=False,
            )
            _pg_dependency(
                connection,
                message_id=message_id,
                ordinal=1,
                parent_origin=origin,
                historical_null_scope=True,
            )

    with engine.begin() as connection:
        source_parent = _pg_parent(connection)
        query_id = _pg_component(
            connection, source_parent, 'query_embedding', 'not_attempted'
        )
        _pg_component(connection, source_parent, 'answer_generation', 'not_attempted')

    with pytest.raises(IntegrityError), engine.begin() as connection:
        target_parent = _pg_parent(connection)
        _pg_component(connection, target_parent, 'answer_generation', 'not_attempted')
        connection.execute(
            text(
                'UPDATE agent_run_cost_components SET agent_run_id=:target WHERE id=:child'
            ),
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
                "(1,'00000000-0000-0000-0000-000000000001','test',0,:digest,'key',"
                ':verifier,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)'
            ),
            {'digest': 'c' * 64, 'verifier': 'd' * 64},
        )
        readiness_ids: list[int] = []
        for component, model, config, cost, estimator in (
            (
                'query_embedding',
                'text-embedding-3-small',
                'rag-query-embedding-config:v1',
                'rag-query-embedding-cost:v1',
                'openai-cl100k-text-embedding-3-small:v1',
            ),
            (
                'answer_generation',
                'gpt-5.4-mini-2026-03-17',
                'rag-answer-model-config:v1',
                'rag-answer-cost:v1',
                'openai-o200k-rag-answer:v1',
            ),
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
                    {
                        'component': component,
                        'model': model,
                        'config': config,
                        'config_hmac': 'e' * 64,
                        'cost': cost,
                        'estimator': estimator,
                        'verifier': 'd' * 64,
                        'policy': 'f' * 64,
                        'review': 'c' * 64,
                    },
                )
            )
        connection.execute(
            text(
                'INSERT INTO rag_provider_safety_transitions '
                '(authority_id,readiness_id,global_safety_generation,transition_kind,'
                'prior_state,new_state,prior_state_version,new_state_version,'
                'prior_family_safety_generation,new_family_safety_generation,'
                'envelope_digest,reviewed_transition_reference_hmac,created_at) VALUES '
                "(1,NULL,0,'bootstrap',NULL,NULL,NULL,NULL,NULL,NULL,"
                ':digest,:review,CURRENT_TIMESTAMP)'
            ),
            {'digest': 'c' * 64, 'review': 'c' * 64},
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
        connection.execute(
            text(
                'UPDATE rag_provider_safety_authorities SET '
                "global_safety_generation=1,envelope_digest=:digest WHERE id=1"
            ),
            {'digest': 'b' * 64},
        )
        connection.execute(
            text(
                "UPDATE rag_provider_readiness SET state='rebind_required', "
                'state_version=2,family_safety_generation=1,'
                'reviewed_gate_reference_hmac=:review WHERE id=:id'
            ),
            {'id': readiness_ids[0], 'review': 'e' * 64},
        )
        connection.execute(
            text(
                'INSERT INTO rag_provider_safety_transitions '
                '(authority_id,readiness_id,global_safety_generation,transition_kind,'
                'prior_state,new_state,prior_state_version,new_state_version,'
                'prior_family_safety_generation,new_family_safety_generation,'
                'envelope_digest,reviewed_transition_reference_hmac,created_at) VALUES '
                "(1,:id,1,'rebind_required',NULL,'rebind_required',NULL,2,NULL,1,"
                ':digest,:review,CURRENT_TIMESTAMP)'
            ),
            {'id': readiness_ids[0], 'digest': 'b' * 64, 'review': 'e' * 64},
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
                'INSERT INTO assistant_conversations (user_id,title,created_at,updated_at) '
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
                ":hmac,:hmac,:hmac,CAST('{}' AS json),CURRENT_TIMESTAMP)"
            ),
            {'conversation': conversation_id, 'run': linked, 'hmac': 'a' * 64},
        )
