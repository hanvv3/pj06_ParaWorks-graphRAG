from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

import backend.app.models  # noqa: F401
from backend.app.db.base import Base

revision = 'e2b3c4d5f6a7'
down_revision = 'd1a2b3c4e5f6'
branch_labels = None
depends_on = None

_NEW_TABLES = (
    'agent_run_cost_components',
    'rag_provider_safety_authorities',
    'rag_provider_readiness',
    'rag_provider_safety_transitions',
    'rag_advisory_lock_key_registry',
)
_AGENT_RUN_COLUMNS = (
    'run_contract_version',
    'run_record_phase',
    'total_charged_cost_usd',
    'projection_owner_fence_hmac',
)
_ASSISTANT_COLUMNS = (
    'content_write_mode',
    'content_hmac_schema_version',
    'assistant_message_content_hmac',
    'content_hmac_key_version',
    'content_hmac_key_material_verifier',
    'content_origin',
    'content_origin_hmac',
    'rag_result_hmac',
    'linked_agent_run_id',
    'dependency_set_hmac_schema_version',
    'dependency_set_hmac',
    'parent_selected_evidence_projection_hmac',
    'model_influence_set_hmac',
)
_DEPENDENCY_COLUMNS = (
    'dependency_serving_scope',
    'dependency_role',
    'dependency_child_hmac',
    'approval_provenance_hmac',
    'evidence_link_set_hmac',
    'legacy_dependency_identity_hmac',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'selected_v1_citation_projection_hmac',
    'serving_identity_hmac',
    'serving_version_fingerprint',
    'support_mode',
)
_DEPENDENCY_KIND_CHECKS = (
    'ck_assistant_message_dependency_kind',
    'ck_assistant_message_dependency_exact_kind',
)
_LEGACY_DEPENDENCY_KIND_SQL = {
    'ck_assistant_message_dependency_kind': (
        "dependency_kind IN ('raw_chunk', 'trusted_knowledge')"
    ),
    'ck_assistant_message_dependency_exact_kind': (
        "(dependency_kind = 'raw_chunk' AND document_chunk_id IS NOT NULL AND "
        'document_version_id IS NOT NULL AND source_id IS NOT NULL AND '
        'parser_run_id IS NOT NULL AND server_content_signature_schema = '
        "'server-source-content:v1' AND length(server_content_signature) = 64 "
        'AND knowledge_type IS NULL AND knowledge_id IS NULL AND '
        'approval_link_id IS NULL AND legacy_human_base = false) OR '
        "(dependency_kind = 'trusted_knowledge' AND document_chunk_id IS NULL "
        'AND document_version_id IS NULL AND source_id IS NULL AND '
        'parser_run_id IS NULL AND server_content_signature_schema IS NULL AND '
        'server_content_signature IS NULL AND knowledge_type IS NOT NULL AND '
        'knowledge_id IS NOT NULL AND ((approval_link_id IS NOT NULL AND '
        'legacy_human_base = false) OR (approval_link_id IS NULL AND '
        'legacy_human_base = true)))'
    ),
}


def upgrade() -> None:
    _add_columns('agent_runs', _AGENT_RUN_COLUMNS)
    _add_columns('assistant_messages', _ASSISTANT_COLUMNS)
    _add_columns('assistant_message_evidence_dependencies', _DEPENDENCY_COLUMNS)
    _replace_dependency_kind_checks(forward=True)
    _create_additive_checks()
    for table_name in _NEW_TABLES:
        Base.metadata.create_all(
            op.get_bind(), tables=[Base.metadata.tables[table_name]], checkfirst=True
        )
    _create_linked_agent_run_index()
    _install_postgresql_runtime_guards()
    _install_postgresql_generation_guards()


def downgrade() -> None:
    _drop_postgresql_generation_guards()
    _drop_postgresql_runtime_guards()
    _drop_linked_agent_run_index()
    for table_name in reversed(_NEW_TABLES):
        Base.metadata.tables[table_name].drop(op.get_bind(), checkfirst=True)
    _drop_additive_checks()
    _replace_dependency_kind_checks(forward=False)
    _drop_columns('assistant_message_evidence_dependencies', _DEPENDENCY_COLUMNS)
    _drop_columns('assistant_messages', _ASSISTANT_COLUMNS)
    _drop_columns('agent_runs', _AGENT_RUN_COLUMNS)


def _add_columns(table_name: str, names: tuple[str, ...]) -> None:
    if table_name not in _table_names():
        return
    existing = _column_names(table_name)
    table = Base.metadata.tables[table_name]
    for name in names:
        if name not in existing:
            op.add_column(table_name, table.c[name]._copy())


def _drop_columns(table_name: str, names: tuple[str, ...]) -> None:
    if table_name not in _table_names():
        return
    existing = _column_names(table_name)
    for name in reversed(names):
        if name in existing:
            with op.batch_alter_table(table_name) as batch:
                batch.drop_column(name)


def _replace_dependency_kind_checks(*, forward: bool) -> None:
    table_name = 'assistant_message_evidence_dependencies'
    if table_name not in _table_names():
        return
    existing = _constraint_names(table_name, 'check')
    table = Base.metadata.tables[table_name]
    model_checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, sa.CheckConstraint)
    }
    replacements = model_checks if forward else _LEGACY_DEPENDENCY_KIND_SQL
    with op.batch_alter_table(table_name) as batch:
        for name in _DEPENDENCY_KIND_CHECKS:
            if name in existing:
                batch.drop_constraint(name, type_='check')
        for name in _DEPENDENCY_KIND_CHECKS:
            batch.create_check_constraint(name, replacements[name])


def _create_additive_checks() -> None:
    checks = {
        'agent_runs': (
            'ck_agent_runs_rag_v2_parent_shape',
            'ck_agent_runs_rag_v2_lifecycle',
        ),
        'assistant_messages': (
            'ck_assistant_messages_dependency_set_schema',
            'ck_assistant_messages_content_write_mode',
            'ck_assistant_messages_content_origin',
            'ck_assistant_messages_content_integrity',
            'ck_assistant_messages_content_origin_xor',
        ),
        'assistant_message_evidence_dependencies': (
            'ck_assistant_message_dependency_v2_scope_role',
            'ck_assistant_message_dependency_v2_hmacs',
            'ck_assistant_message_dependency_v2_support',
        ),
    }
    for table_name, names in checks.items():
        if table_name not in _table_names():
            continue
        existing = _constraint_names(table_name, 'check')
        columns = _column_names(table_name)
        table = Base.metadata.tables[table_name]
        model_checks = {
            constraint.name: str(constraint.sqltext)
            for constraint in table.constraints
            if isinstance(constraint, sa.CheckConstraint)
        }
        for name in names:
            if (
                name == 'ck_agent_runs_rag_v2_lifecycle'
                and not {'status', 'completed_at'} <= columns
            ):
                continue
            if name not in existing:
                with op.batch_alter_table(table_name) as batch:
                    batch.create_check_constraint(name, model_checks[name])


def _drop_additive_checks() -> None:
    checks = {
        'assistant_message_evidence_dependencies': (
            'ck_assistant_message_dependency_v2_support',
            'ck_assistant_message_dependency_v2_hmacs',
            'ck_assistant_message_dependency_v2_scope_role',
        ),
        'assistant_messages': (
            'ck_assistant_messages_content_origin_xor',
            'ck_assistant_messages_content_integrity',
            'ck_assistant_messages_content_origin',
            'ck_assistant_messages_content_write_mode',
            'ck_assistant_messages_dependency_set_schema',
        ),
        'agent_runs': (
            'ck_agent_runs_rag_v2_lifecycle',
            'ck_agent_runs_rag_v2_parent_shape',
        ),
    }
    for table_name, names in checks.items():
        if table_name not in _table_names():
            continue
        for name in names:
            if name in _constraint_names(table_name, 'check'):
                with op.batch_alter_table(table_name) as batch:
                    batch.drop_constraint(name, type_='check')


def _create_linked_agent_run_index() -> None:
    if 'assistant_messages' not in _table_names():
        return
    indexes = {item['name'] for item in inspect(op.get_bind()).get_indexes('assistant_messages')}
    if 'ix_assistant_messages_linked_agent_run_id' not in indexes:
        op.create_index(
            'ix_assistant_messages_linked_agent_run_id',
            'assistant_messages',
            ['linked_agent_run_id'],
        )


def _drop_linked_agent_run_index() -> None:
    if 'assistant_messages' not in _table_names():
        return
    indexes = {item['name'] for item in inspect(op.get_bind()).get_indexes('assistant_messages')}
    if 'ix_assistant_messages_linked_agent_run_id' in indexes:
        op.drop_index(
            'ix_assistant_messages_linked_agent_run_id',
            table_name='assistant_messages',
        )


def _install_postgresql_runtime_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION rag_validate_agent_run_v2_costs()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
          target_run_id bigint;
          parent_version text;
          parent_total numeric(24,6);
          child_count integer;
          child_components integer;
          child_total numeric(24,6);
        BEGIN
          IF TG_TABLE_NAME = 'agent_runs' THEN
            target_run_id := COALESCE(NEW.id, OLD.id);
          ELSE
            target_run_id := COALESCE(NEW.agent_run_id, OLD.agent_run_id);
          END IF;
          SELECT run_contract_version, total_charged_cost_usd
            INTO parent_version, parent_total
            FROM agent_runs WHERE id = target_run_id;
          SELECT count(*),
                 count(DISTINCT component),
                 COALESCE(sum(charged_cost_usd), 0.000000)
            INTO child_count, child_components, child_total
            FROM agent_run_cost_components WHERE agent_run_id = target_run_id;
          IF parent_version = 'rag-run:v2' THEN
            IF parent_total IS NULL OR child_count <> 2 OR child_components <> 2 OR
               NOT EXISTS (SELECT 1 FROM agent_run_cost_components
                           WHERE agent_run_id = target_run_id
                             AND component = 'query_embedding'
                             AND component_ordinal = 0) OR
               NOT EXISTS (SELECT 1 FROM agent_run_cost_components
                           WHERE agent_run_id = target_run_id
                             AND component = 'answer_generation'
                             AND component_ordinal = 1) OR
               child_total <> parent_total THEN
              RAISE EXCEPTION 'rag-run:v2 requires exact two balanced cost children';
            END IF;
          ELSIF child_count <> 0 THEN
            RAISE EXCEPTION 'legacy AgentRun cannot own D cost children';
          END IF;
          RETURN NULL;
        END $$;
        DROP TRIGGER IF EXISTS rag_agent_run_v2_costs_guard_parent ON agent_runs;
        CREATE CONSTRAINT TRIGGER rag_agent_run_v2_costs_guard_parent
          AFTER INSERT OR UPDATE ON agent_runs DEFERRABLE INITIALLY DEFERRED
          FOR EACH ROW EXECUTE FUNCTION rag_validate_agent_run_v2_costs();
        DROP TRIGGER IF EXISTS rag_agent_run_v2_costs_guard_child
          ON agent_run_cost_components;
        CREATE CONSTRAINT TRIGGER rag_agent_run_v2_costs_guard_child
          AFTER INSERT OR UPDATE OR DELETE ON agent_run_cost_components
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_agent_run_v2_costs();
    """))
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION rag_refuse_terminal_cost_component_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.dispatch_state IN ('terminal', 'abandoned_unknown') THEN
            RAISE EXCEPTION 'terminal RAG cost components are immutable';
          END IF;
          RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END $$;
        DROP TRIGGER IF EXISTS rag_terminal_cost_component_immutable
          ON agent_run_cost_components;
        CREATE TRIGGER rag_terminal_cost_component_immutable
          BEFORE UPDATE OR DELETE ON agent_run_cost_components
          FOR EACH ROW EXECUTE FUNCTION rag_refuse_terminal_cost_component_mutation();
    """))
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION rag_validate_provider_safety_whole_set()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
          authority_count integer;
          authority_generation bigint;
          active_count integer;
          transition_count bigint;
          min_generation bigint;
          max_generation bigint;
        BEGIN
          SELECT count(*), max(global_safety_generation)
            INTO authority_count, authority_generation
            FROM rag_provider_safety_authorities;
          IF authority_count = 0 THEN
            IF EXISTS (SELECT 1 FROM rag_provider_readiness) OR
               EXISTS (SELECT 1 FROM rag_provider_safety_transitions) THEN
              RAISE EXCEPTION 'provider safety children require singleton authority';
            END IF;
            RETURN NULL;
          END IF;
          IF authority_count <> 1 OR
             NOT EXISTS (SELECT 1 FROM rag_provider_safety_authorities WHERE id = 1) THEN
            RAISE EXCEPTION 'provider safety authority must be exact singleton';
          END IF;
          SELECT count(*) INTO active_count FROM rag_provider_readiness
            WHERE active = true;
          IF active_count <> 2 OR
             NOT EXISTS (SELECT 1 FROM rag_provider_readiness
                         WHERE active = true AND component = 'query_embedding') OR
             NOT EXISTS (SELECT 1 FROM rag_provider_readiness
                         WHERE active = true AND component = 'answer_generation') THEN
            RAISE EXCEPTION 'provider safety requires exact two active families';
          END IF;
          SELECT count(*), min(global_safety_generation), max(global_safety_generation)
            INTO transition_count, min_generation, max_generation
            FROM rag_provider_safety_transitions WHERE authority_id = 1;
          IF transition_count <> authority_generation + 1 OR min_generation <> 0 OR
             max_generation <> authority_generation THEN
            RAISE EXCEPTION 'provider safety transition generations must be gapless';
          END IF;
          RETURN NULL;
        END $$;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_authority
          ON rag_provider_safety_authorities;
        CREATE CONSTRAINT TRIGGER rag_provider_safety_whole_set_guard_authority
          AFTER INSERT OR UPDATE OR DELETE ON rag_provider_safety_authorities
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_provider_safety_whole_set();
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_readiness
          ON rag_provider_readiness;
        CREATE CONSTRAINT TRIGGER rag_provider_safety_whole_set_guard_readiness
          AFTER INSERT OR UPDATE OR DELETE ON rag_provider_readiness
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_provider_safety_whole_set();
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_transition
          ON rag_provider_safety_transitions;
        CREATE CONSTRAINT TRIGGER rag_provider_safety_whole_set_guard_transition
          AFTER INSERT ON rag_provider_safety_transitions
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_provider_safety_whole_set();
    """))
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION rag_refuse_append_only_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'RAG authority history is append-only';
        END $$;
        DROP TRIGGER IF EXISTS rag_provider_safety_transition_append_only
          ON rag_provider_safety_transitions;
        CREATE TRIGGER rag_provider_safety_transition_append_only
          BEFORE UPDATE OR DELETE ON rag_provider_safety_transitions
          FOR EACH ROW EXECUTE FUNCTION rag_refuse_append_only_mutation();
        DROP TRIGGER IF EXISTS rag_advisory_lock_registry_append_only
          ON rag_advisory_lock_key_registry;
        CREATE TRIGGER rag_advisory_lock_registry_append_only
          BEFORE UPDATE OR DELETE ON rag_advisory_lock_key_registry
          FOR EACH ROW EXECUTE FUNCTION rag_refuse_append_only_mutation();
    """))


def _drop_postgresql_runtime_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute(sa.text("""
        DROP TRIGGER IF EXISTS rag_advisory_lock_registry_append_only
          ON rag_advisory_lock_key_registry;
        DROP TRIGGER IF EXISTS rag_provider_safety_transition_append_only
          ON rag_provider_safety_transitions;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_transition
          ON rag_provider_safety_transitions;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_readiness
          ON rag_provider_readiness;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_authority
          ON rag_provider_safety_authorities;
        DROP TRIGGER IF EXISTS rag_terminal_cost_component_immutable
          ON agent_run_cost_components;
        DROP TRIGGER IF EXISTS rag_agent_run_v2_costs_guard_child
          ON agent_run_cost_components;
        DROP TRIGGER IF EXISTS rag_agent_run_v2_costs_guard_parent ON agent_runs;
        DROP FUNCTION IF EXISTS rag_refuse_append_only_mutation();
        DROP FUNCTION IF EXISTS rag_validate_provider_safety_whole_set();
        DROP FUNCTION IF EXISTS rag_refuse_terminal_cost_component_mutation();
        DROP FUNCTION IF EXISTS rag_validate_agent_run_v2_costs();
    """))


def _install_postgresql_generation_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION rag_mark_corpus_generation_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'UPDATE' AND NEW.corpus_generation <> OLD.corpus_generation THEN
            PERFORM set_config('paraworks.rag_corpus_generation_tx',
                               txid_current()::text, true);
          END IF;
          RETURN NEW;
        END $$;
        CREATE OR REPLACE FUNCTION rag_mark_vector_generation_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'UPDATE' AND
             NEW.vector_index_generation <> OLD.vector_index_generation THEN
            PERFORM set_config('paraworks.rag_vector_generation_tx',
                               txid_current()::text, true);
          END IF;
          RETURN NEW;
        END $$;
        DROP TRIGGER IF EXISTS rag_mark_corpus_generation_mutation
          ON rag_serving_corpus_generations;
        CREATE TRIGGER rag_mark_corpus_generation_mutation
          AFTER UPDATE OF corpus_generation ON rag_serving_corpus_generations
          FOR EACH ROW EXECUTE FUNCTION rag_mark_corpus_generation_mutation();
        DROP TRIGGER IF EXISTS rag_mark_vector_generation_mutation
          ON rag_serving_corpus_generations;
        CREATE TRIGGER rag_mark_vector_generation_mutation
          AFTER UPDATE OF vector_index_generation ON rag_serving_corpus_generations
          FOR EACH ROW EXECUTE FUNCTION rag_mark_vector_generation_mutation();
    """))
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION rag_require_corpus_generation_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF current_setting('paraworks.rag_corpus_generation_tx', true)
             IS DISTINCT FROM txid_current()::text THEN
            RAISE EXCEPTION 'canonical serving mutation requires corpus generation';
          END IF;
          RETURN NULL;
        END $$;
        CREATE OR REPLACE FUNCTION rag_require_vector_generation_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_TABLE_NAME = 'vector_index_states' AND
             COALESCE(NEW.serving_kind, OLD.serving_kind) IS NULL THEN
            RETURN NULL;
          END IF;
          IF current_setting('paraworks.rag_vector_generation_tx', true)
             IS DISTINCT FROM txid_current()::text THEN
            RAISE EXCEPTION 'D vector mutation requires vector generation';
          END IF;
          RETURN NULL;
        END $$;
    """))
    for table_name in (
        'sources',
        'documents',
        'document_versions',
        'document_chunks',
        'review_items',
        'decision_records',
        'history_events',
        'timeline_events',
        'todos',
        'trusted_knowledge_approval_links',
        'trusted_knowledge_evidence_links',
        'rag_lexical_serving_projections',
    ):
        op.execute(sa.text(f"""
            DROP TRIGGER IF EXISTS rag_require_corpus_generation_mutation
              ON {table_name};
            CREATE CONSTRAINT TRIGGER rag_require_corpus_generation_mutation
              AFTER INSERT OR UPDATE OR DELETE ON {table_name}
              DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
              EXECUTE FUNCTION rag_require_corpus_generation_mutation();
        """))
    for table_name in ('vector_index_states', 'vector_serving_tombstones'):
        op.execute(sa.text(f"""
            DROP TRIGGER IF EXISTS rag_require_vector_generation_mutation
              ON {table_name};
            CREATE CONSTRAINT TRIGGER rag_require_vector_generation_mutation
              AFTER INSERT OR UPDATE OR DELETE ON {table_name}
              DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
              EXECUTE FUNCTION rag_require_vector_generation_mutation();
        """))


def _drop_postgresql_generation_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    for table_name in ('vector_serving_tombstones', 'vector_index_states'):
        op.execute(sa.text(
            f'DROP TRIGGER IF EXISTS rag_require_vector_generation_mutation ON {table_name}'
        ))
    for table_name in (
        'rag_lexical_serving_projections',
        'trusted_knowledge_evidence_links',
        'trusted_knowledge_approval_links',
        'todos',
        'timeline_events',
        'history_events',
        'decision_records',
        'review_items',
        'document_chunks',
        'document_versions',
        'documents',
        'sources',
    ):
        op.execute(sa.text(
            f'DROP TRIGGER IF EXISTS rag_require_corpus_generation_mutation ON {table_name}'
        ))
    op.execute(sa.text("""
        DROP TRIGGER IF EXISTS rag_mark_vector_generation_mutation
          ON rag_serving_corpus_generations;
        DROP TRIGGER IF EXISTS rag_mark_corpus_generation_mutation
          ON rag_serving_corpus_generations;
        DROP FUNCTION IF EXISTS rag_require_vector_generation_mutation();
        DROP FUNCTION IF EXISTS rag_require_corpus_generation_mutation();
        DROP FUNCTION IF EXISTS rag_mark_vector_generation_mutation();
        DROP FUNCTION IF EXISTS rag_mark_corpus_generation_mutation();
    """))


def _constraint_names(table_name: str, kind: str) -> set[str | None]:
    schema = inspect(op.get_bind())
    getter = {'check': schema.get_check_constraints}[kind]
    return {item['name'] for item in getter(table_name)}


def _table_names() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    return {item['name'] for item in inspect(op.get_bind()).get_columns(table_name)}
