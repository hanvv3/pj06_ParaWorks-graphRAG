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
        CREATE OR REPLACE FUNCTION rag_validate_agent_run_v2_costs_for(
          target_run_id bigint
        ) RETURNS void LANGUAGE plpgsql AS $$
        DECLARE
          parent_version text;
          parent_phase text;
          parent_status text;
          parent_outcome text;
          parent_completed_at timestamptz;
          parent_projection_fence text;
          parent_total numeric(24,6);
          child_count integer;
          child_components integer;
          child_total numeric(24,6);
          not_attempted_count integer;
          dispatching_count integer;
          terminal_count integer;
          abandoned_count integer;
        BEGIN
          IF target_run_id IS NULL THEN RETURN; END IF;
          SELECT run_contract_version, run_record_phase, status,
                 metadata ->> 'outcome', completed_at,
                 projection_owner_fence_hmac, total_charged_cost_usd
            INTO parent_version, parent_phase, parent_status, parent_outcome,
                 parent_completed_at, parent_projection_fence, parent_total
            FROM agent_runs WHERE id = target_run_id;
          SELECT count(*), count(DISTINCT component),
                 COALESCE(sum(charged_cost_usd), 0.000000),
                 count(*) FILTER (WHERE dispatch_state = 'not_attempted'),
                 count(*) FILTER (WHERE dispatch_state = 'dispatching'),
                 count(*) FILTER (WHERE dispatch_state = 'terminal'),
                 count(*) FILTER (WHERE dispatch_state = 'abandoned_unknown')
            INTO child_count, child_components, child_total,
                 not_attempted_count, dispatching_count, terminal_count,
                 abandoned_count
            FROM agent_run_cost_components WHERE agent_run_id = target_run_id;
          IF parent_version IS NULL THEN
            IF child_count <> 0 THEN
              RAISE EXCEPTION 'legacy or missing AgentRun cannot own D cost children';
            END IF;
            RETURN;
          END IF;
          IF parent_version <> 'rag-run:v2' OR parent_total IS NULL OR
             child_count <> 2 OR child_components <> 2 OR
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
          IF parent_phase = 'admission' THEN
            IF parent_status <> 'running' OR parent_outcome IS NOT NULL OR
               parent_completed_at IS NOT NULL OR parent_projection_fence IS NOT NULL OR
               abandoned_count <> 0 OR terminal_count = 2 THEN
              RAISE EXCEPTION 'invalid rag-run:v2 admission lifecycle';
            END IF;
          ELSIF parent_phase = 'cost_finalized_pending_projection' THEN
            IF parent_status <> 'running' OR parent_outcome IS NULL OR
               parent_completed_at IS NOT NULL OR parent_projection_fence IS NULL OR
               terminal_count <> 2 THEN
              RAISE EXCEPTION 'invalid rag-run:v2 pending-projection lifecycle';
            END IF;
          ELSIF parent_phase = 'final' THEN
            IF parent_status NOT IN ('complete', 'failed') OR parent_outcome IS NULL OR
               parent_completed_at IS NULL OR terminal_count <> 2 OR
               not_attempted_count <> 0 OR dispatching_count <> 0 OR
               abandoned_count <> 0 THEN
              RAISE EXCEPTION 'invalid rag-run:v2 final lifecycle';
            END IF;
          ELSIF parent_phase = 'admission_only' THEN
            IF parent_status <> 'failed' OR parent_outcome <> 'abandoned_unknown' OR
               parent_completed_at IS NULL OR parent_projection_fence IS NOT NULL OR
               not_attempted_count <> 0 OR dispatching_count <> 0 OR
               abandoned_count = 0 OR terminal_count + abandoned_count <> 2 THEN
              RAISE EXCEPTION 'invalid rag-run:v2 admission-only lifecycle';
            END IF;
          ELSE
            RAISE EXCEPTION 'invalid rag-run:v2 phase';
          END IF;
        END $$;
        CREATE OR REPLACE FUNCTION rag_validate_agent_run_v2_costs()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_TABLE_NAME = 'agent_runs' THEN
            PERFORM rag_validate_agent_run_v2_costs_for(COALESCE(NEW.id, OLD.id));
          ELSE
            IF TG_OP IN ('UPDATE', 'DELETE') THEN
              PERFORM rag_validate_agent_run_v2_costs_for(OLD.agent_run_id);
            END IF;
            IF TG_OP IN ('INSERT', 'UPDATE') AND
               (TG_OP <> 'UPDATE' OR OLD.agent_run_id IS DISTINCT FROM NEW.agent_run_id) THEN
              PERFORM rag_validate_agent_run_v2_costs_for(NEW.agent_run_id);
            ELSIF TG_OP = 'UPDATE' THEN
              PERFORM rag_validate_agent_run_v2_costs_for(NEW.agent_run_id);
            END IF;
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
          authority rag_provider_safety_authorities%ROWTYPE;
          readiness rag_provider_readiness%ROWTYPE;
          transition rag_provider_safety_transitions%ROWTYPE;
          prior_transition rag_provider_safety_transitions%ROWTYPE;
          authority_count integer;
          active_count integer;
          readiness_count integer;
          supersession_count integer;
          transition_count bigint;
          min_generation bigint;
          max_generation bigint;
        BEGIN
          SELECT count(*) INTO authority_count FROM rag_provider_safety_authorities;
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
          SELECT * INTO STRICT authority
            FROM rag_provider_safety_authorities WHERE id = 1;
          SELECT count(*), count(*) FILTER (WHERE active = true)
            INTO readiness_count, active_count FROM rag_provider_readiness;
          SELECT count(*) INTO supersession_count
            FROM rag_provider_safety_transitions
            WHERE transition_kind = 'supersession';
          IF active_count <> 2 OR
             readiness_count <> 2 + supersession_count OR
             NOT EXISTS (SELECT 1 FROM rag_provider_readiness
                         WHERE active = true AND component = 'query_embedding') OR
             NOT EXISTS (SELECT 1 FROM rag_provider_readiness
                         WHERE active = true AND component = 'answer_generation') THEN
            RAISE EXCEPTION 'provider safety requires exact two active families';
          END IF;
          SELECT count(*), min(global_safety_generation), max(global_safety_generation)
            INTO transition_count, min_generation, max_generation
            FROM rag_provider_safety_transitions WHERE authority_id = 1;
          IF transition_count <> authority.global_safety_generation + 1 OR
             min_generation <> 0 OR
             max_generation <> authority.global_safety_generation THEN
            RAISE EXCEPTION 'provider safety transition generations must be gapless';
          END IF;
          SELECT * INTO STRICT transition FROM rag_provider_safety_transitions
            WHERE authority_id = 1
              AND global_safety_generation = authority.global_safety_generation;
          IF transition.envelope_digest <> authority.envelope_digest THEN
            RAISE EXCEPTION 'provider authority mutation requires exact transition';
          END IF;
          -- bootstrap iff generation zero; authority-wide generation-0 bootstrap is targetless.
          IF (authority.global_safety_generation = 0) <>
             (transition.transition_kind = 'bootstrap') OR
             (authority.global_safety_generation = 0 AND
             (transition.transition_kind <> 'bootstrap' OR
              transition.readiness_id IS NOT NULL OR
              transition.prior_state IS NOT NULL OR transition.new_state IS NOT NULL)) THEN
            RAISE EXCEPTION 'provider generation-0 bootstrap is invalid';
          END IF;
          FOR readiness IN SELECT * FROM rag_provider_readiness LOOP
            SELECT * INTO transition FROM rag_provider_safety_transitions
              WHERE readiness_id = readiness.id
              ORDER BY global_safety_generation DESC LIMIT 1;
            IF NOT FOUND THEN
              IF readiness.state <> 'ready' OR readiness.state_version <> 1 OR
                 readiness.family_safety_generation <> 0 OR
                 readiness.reviewed_gate_reference_hmac IS DISTINCT FROM
                   (SELECT reviewed_transition_reference_hmac
                      FROM rag_provider_safety_transitions
                     WHERE authority_id = 1 AND global_safety_generation = 0) THEN
                RAISE EXCEPTION 'untouched bootstrap family drift';
              END IF;
              CONTINUE;
            END IF;
            IF transition.authority_id <> readiness.authority_id OR
               transition.new_state IS DISTINCT FROM readiness.state OR
               transition.new_state_version IS DISTINCT FROM readiness.state_version OR
               transition.new_family_safety_generation IS DISTINCT FROM
                 readiness.family_safety_generation OR
               transition.reviewed_transition_reference_hmac IS DISTINCT FROM
                 readiness.reviewed_gate_reference_hmac THEN
              RAISE EXCEPTION 'targeted family transition state mismatch';
            END IF;
            IF (transition.prior_state_version IS NULL AND
                transition.new_state_version IS DISTINCT FROM 1) OR
               (transition.prior_state_version IS NOT NULL AND
                transition.new_state_version IS DISTINCT FROM
                  transition.prior_state_version + 1) OR
               transition.new_family_safety_generation IS DISTINCT FROM
                 transition.global_safety_generation THEN
              RAISE EXCEPTION 'targeted family transition monotonicity mismatch';
            END IF;
            SELECT * INTO prior_transition FROM rag_provider_safety_transitions
              WHERE readiness_id = readiness.id
                AND global_safety_generation < transition.global_safety_generation
              ORDER BY global_safety_generation DESC LIMIT 1;
            IF FOUND THEN
              IF transition.prior_state IS DISTINCT FROM prior_transition.new_state OR
                 transition.prior_state_version IS DISTINCT FROM
                   prior_transition.new_state_version OR
                 transition.prior_family_safety_generation IS DISTINCT FROM
                   prior_transition.new_family_safety_generation THEN
                RAISE EXCEPTION 'provider transition prior snapshot mismatch';
              END IF;
            ELSIF transition.transition_kind = 'supersession' THEN
              IF transition.prior_state IS NOT NULL OR
                 transition.prior_state_version IS NOT NULL OR
                 transition.prior_family_safety_generation IS NOT NULL OR
                 readiness.active IS NOT TRUE OR
                 transition.new_state IS DISTINCT FROM 'ready' OR
                 transition.new_state_version IS DISTINCT FROM 1 THEN
                RAISE EXCEPTION 'provider supersession requires a new active family';
              END IF;
            ELSIF transition.prior_state IS DISTINCT FROM 'ready' OR
                  transition.prior_state_version IS DISTINCT FROM 1 OR
                  transition.prior_family_safety_generation IS DISTINCT FROM 0 THEN
              RAISE EXCEPTION 'first ordinary family mutation must bind bootstrap';
            END IF;
          END LOOP;
          RETURN NULL;
        END $$;
        CREATE OR REPLACE FUNCTION rag_require_audited_provider_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            RAISE EXCEPTION 'provider safety current rows cannot be deleted';
          END IF;
          IF TG_TABLE_NAME = 'rag_provider_safety_authorities' THEN
            IF OLD.authority_uuid IS DISTINCT FROM NEW.authority_uuid OR
               OLD.designated_environment_id IS DISTINCT FROM
                 NEW.designated_environment_id OR
               NEW.global_safety_generation <> OLD.global_safety_generation + 1 OR
               NEW.envelope_digest IS NOT DISTINCT FROM OLD.envelope_digest THEN
              RAISE EXCEPTION 'provider authority update requires audited generation';
            END IF;
          ELSE
            IF OLD.authority_id IS DISTINCT FROM NEW.authority_id OR
               OLD.component IS DISTINCT FROM NEW.component OR
               OLD.provider IS DISTINCT FROM NEW.provider OR
               OLD.model IS DISTINCT FROM NEW.model OR
               OLD.reasoning_or_config_identity IS DISTINCT FROM
                 NEW.reasoning_or_config_identity THEN
              RAISE EXCEPTION 'provider readiness family identity is immutable';
            END IF;
            IF OLD.active IS TRUE AND NEW.active IS FALSE AND
               NEW.state IS NOT DISTINCT FROM OLD.state AND
               NEW.state_version IS NOT DISTINCT FROM OLD.state_version AND
               NEW.family_safety_generation IS NOT DISTINCT FROM
                 OLD.family_safety_generation AND
               NEW.authorized_model_config_version IS NOT DISTINCT FROM
                 OLD.authorized_model_config_version AND
               NEW.authorized_model_config_snapshot_hmac IS NOT DISTINCT FROM
                 OLD.authorized_model_config_snapshot_hmac AND
               NEW.authorized_cost_policy_version IS NOT DISTINCT FROM
                 OLD.authorized_cost_policy_version AND
               NEW.authorized_token_estimator_version IS NOT DISTINCT FROM
                 OLD.authorized_token_estimator_version AND
               NEW.authorized_fingerprint_key_version IS NOT DISTINCT FROM
                 OLD.authorized_fingerprint_key_version AND
               NEW.authorized_fingerprint_key_material_verifier IS NOT DISTINCT FROM
                 OLD.authorized_fingerprint_key_material_verifier AND
               NEW.authorized_policy_snapshot_hmac IS NOT DISTINCT FROM
                 OLD.authorized_policy_snapshot_hmac AND
               NEW.reviewed_gate_reference_hmac IS NOT DISTINCT FROM
                 OLD.reviewed_gate_reference_hmac AND
               NEW.first_blocker_category IS NOT DISTINCT FROM
                 OLD.first_blocker_category AND
               NEW.first_blocker_agent_run_hmac IS NOT DISTINCT FROM
                 OLD.first_blocker_agent_run_hmac AND
               NEW.first_blocker_observed_at IS NOT DISTINCT FROM
                 OLD.first_blocker_observed_at AND
               NEW.overrun_agent_run_id IS NOT DISTINCT FROM
                 OLD.overrun_agent_run_id AND
               NEW.overrun_input_tokens IS NOT DISTINCT FROM
                 OLD.overrun_input_tokens AND
               NEW.overrun_output_tokens IS NOT DISTINCT FROM
                 OLD.overrun_output_tokens AND
               NEW.overrun_cost_usd IS NOT DISTINCT FROM OLD.overrun_cost_usd AND
               NEW.overrun_observed_at IS NOT DISTINCT FROM
                 OLD.overrun_observed_at AND
               NEW.reset_by IS NOT DISTINCT FROM OLD.reset_by AND
               NEW.reset_at IS NOT DISTINCT FROM OLD.reset_at THEN
              RETURN NEW;
            END IF;
            IF NEW.active IS DISTINCT FROM OLD.active OR
               NEW.state_version <> OLD.state_version + 1 OR
               NEW.family_safety_generation <= OLD.family_safety_generation THEN
              RAISE EXCEPTION 'provider readiness update requires audited generation';
            END IF;
          END IF;
          RETURN NEW;
        END $$;
        DROP TRIGGER IF EXISTS rag_provider_safety_authority_audited_mutation
          ON rag_provider_safety_authorities;
        CREATE TRIGGER rag_provider_safety_authority_audited_mutation
          BEFORE UPDATE OR DELETE ON rag_provider_safety_authorities
          FOR EACH ROW EXECUTE FUNCTION rag_require_audited_provider_mutation();
        DROP TRIGGER IF EXISTS rag_provider_readiness_audited_mutation
          ON rag_provider_readiness;
        CREATE TRIGGER rag_provider_readiness_audited_mutation
          BEFORE UPDATE OR DELETE ON rag_provider_readiness
          FOR EACH ROW EXECUTE FUNCTION rag_require_audited_provider_mutation();
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
        CREATE OR REPLACE FUNCTION rag_validate_assistant_integrity_for(
          target_message_id bigint
        ) RETURNS void LANGUAGE plpgsql AS $$
        DECLARE
          parent assistant_messages%ROWTYPE;
          linked agent_runs%ROWTYPE;
          child_count integer;
          ordinal_count integer;
          minimum_ordinal integer;
          maximum_ordinal integer;
          selected_count integer;
          scope_mismatch_count integer;
          hmac_mismatch_count integer;
        BEGIN
          IF target_message_id IS NULL THEN RETURN; END IF;
          SELECT * INTO parent FROM assistant_messages WHERE id = target_message_id;
          IF NOT FOUND THEN
            IF EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies
                       WHERE assistant_message_id = target_message_id) THEN
              RAISE EXCEPTION 'Assistant dependencies require a parent message';
            END IF;
            RETURN;
          END IF;
          IF parent.content_write_mode IS NULL THEN RETURN; END IF;
          IF parent.content_write_mode = 'rag_v2_exact' THEN
            SELECT * INTO linked FROM agent_runs WHERE id = parent.linked_agent_run_id;
            IF NOT FOUND OR
               linked.run_contract_version IS DISTINCT FROM 'rag-run:v2' OR
               linked.run_record_phase IS DISTINCT FROM 'final' OR
               linked.status NOT IN ('complete', 'failed') OR
               linked.completed_at IS NULL OR
               linked.metadata ->> 'rag_result_hmac' IS DISTINCT FROM
                 parent.rag_result_hmac THEN
              RAISE EXCEPTION 'RAG Assistant message requires linked final rag-run:v2';
            END IF;
          END IF;
          SELECT count(*), count(DISTINCT candidate_ordinal),
                 min(candidate_ordinal), max(candidate_ordinal),
                 count(*) FILTER (WHERE dependency_role = 'selected_citation'),
                 count(*) FILTER (WHERE
                   (parent.content_origin = 'rag_assembled' AND
                    dependency_serving_scope IS DISTINCT FROM 'rag_v2') OR
                   (parent.content_origin = 'legacy_evidence' AND
                    (dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only' OR
                     dependency_role IS DISTINCT FROM 'selected_citation'))),
                 count(*) FILTER (WHERE
                   dependency_set_hmac IS DISTINCT FROM parent.dependency_set_hmac OR
                   fingerprint_key_version IS DISTINCT FROM
                     parent.content_hmac_key_version OR
                   fingerprint_key_material_verifier IS DISTINCT FROM
                     parent.content_hmac_key_material_verifier)
            INTO child_count, ordinal_count, minimum_ordinal, maximum_ordinal,
                 selected_count, scope_mismatch_count, hmac_mismatch_count
            FROM assistant_message_evidence_dependencies
            WHERE assistant_message_id = target_message_id;
          IF parent.content_origin = 'rag_canned' THEN
            IF child_count <> 0 OR parent.serving_dependency_count <> 0 THEN
              RAISE EXCEPTION 'canned Assistant message cannot own dependencies';
            END IF;
          ELSIF parent.content_origin IN ('rag_assembled', 'legacy_evidence') THEN
            IF child_count <> parent.serving_dependency_count OR child_count <= 0 OR
               ordinal_count <> child_count OR minimum_ordinal <> 0 OR
               maximum_ordinal <> child_count - 1 OR selected_count <= 0 OR
               scope_mismatch_count <> 0 OR hmac_mismatch_count <> 0 THEN
              RAISE EXCEPTION 'Assistant dependency whole-set mismatch';
            END IF;
          ELSE
            RAISE EXCEPTION 'unknown Assistant content origin';
          END IF;
        END $$;
        CREATE OR REPLACE FUNCTION rag_validate_assistant_integrity()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
          dependent_message_id bigint;
        BEGIN
          IF TG_TABLE_NAME = 'agent_runs' THEN
            FOR dependent_message_id IN
              SELECT id FROM assistant_messages
              WHERE linked_agent_run_id = COALESCE(NEW.id, OLD.id)
            LOOP
              PERFORM rag_validate_assistant_integrity_for(dependent_message_id);
            END LOOP;
          ELSIF TG_TABLE_NAME = 'assistant_messages' THEN
            PERFORM rag_validate_assistant_integrity_for(COALESCE(NEW.id, OLD.id));
          ELSE
            IF TG_OP IN ('UPDATE', 'DELETE') THEN
              PERFORM rag_validate_assistant_integrity_for(OLD.assistant_message_id);
            END IF;
            IF TG_OP IN ('INSERT', 'UPDATE') AND
               (TG_OP <> 'UPDATE' OR OLD.assistant_message_id IS DISTINCT FROM
                 NEW.assistant_message_id) THEN
              PERFORM rag_validate_assistant_integrity_for(NEW.assistant_message_id);
            ELSIF TG_OP = 'UPDATE' THEN
              PERFORM rag_validate_assistant_integrity_for(NEW.assistant_message_id);
            END IF;
          END IF;
          RETURN NULL;
        END $$;
        DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_parent
          ON assistant_messages;
        CREATE CONSTRAINT TRIGGER rag_assistant_integrity_guard_parent
          AFTER INSERT OR UPDATE OR DELETE ON assistant_messages
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_assistant_integrity();
        DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_child
          ON assistant_message_evidence_dependencies;
        CREATE CONSTRAINT TRIGGER rag_assistant_integrity_guard_child
          AFTER INSERT OR UPDATE OR DELETE ON assistant_message_evidence_dependencies
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_assistant_integrity();
        CREATE CONSTRAINT TRIGGER rag_assistant_integrity_guard_linked_run
          AFTER UPDATE OR DELETE ON agent_runs
          DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
          EXECUTE FUNCTION rag_validate_assistant_integrity();
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
        DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_linked_run
          ON agent_runs;
        DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_child
          ON assistant_message_evidence_dependencies;
        DROP TRIGGER IF EXISTS rag_assistant_integrity_guard_parent
          ON assistant_messages;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_transition
          ON rag_provider_safety_transitions;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_readiness
          ON rag_provider_readiness;
        DROP TRIGGER IF EXISTS rag_provider_safety_whole_set_guard_authority
          ON rag_provider_safety_authorities;
        DROP TRIGGER IF EXISTS rag_provider_readiness_audited_mutation
          ON rag_provider_readiness;
        DROP TRIGGER IF EXISTS rag_provider_safety_authority_audited_mutation
          ON rag_provider_safety_authorities;
        DROP TRIGGER IF EXISTS rag_terminal_cost_component_immutable
          ON agent_run_cost_components;
        DROP TRIGGER IF EXISTS rag_agent_run_v2_costs_guard_child
          ON agent_run_cost_components;
        DROP TRIGGER IF EXISTS rag_agent_run_v2_costs_guard_parent ON agent_runs;
        DROP FUNCTION IF EXISTS rag_refuse_append_only_mutation();
        DROP FUNCTION IF EXISTS rag_validate_assistant_integrity();
        DROP FUNCTION IF EXISTS rag_validate_assistant_integrity_for(bigint);
        DROP FUNCTION IF EXISTS rag_validate_provider_safety_whole_set();
        DROP FUNCTION IF EXISTS rag_require_audited_provider_mutation();
        DROP FUNCTION IF EXISTS rag_refuse_terminal_cost_component_mutation();
        DROP FUNCTION IF EXISTS rag_validate_agent_run_v2_costs();
        DROP FUNCTION IF EXISTS rag_validate_agent_run_v2_costs_for(bigint);
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
