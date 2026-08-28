"""add auto review trust promotion persistence

Revision ID: 7c5a2e9f4b10
Revises: 2f6a8b9c0d1e
Create Date: 2026-08-28 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

import backend.app.models  # noqa: F401
from backend.app.db.base import Base

revision = '7c5a2e9f4b10'
down_revision = '2f6a8b9c0d1e'
branch_labels = None
depends_on = None

BOUNDARY_COMPONENT = 'auto_review_trust_promotion'

C5_TABLE_NAMES = (
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
)

ADDED_COLUMNS = {
    'agent_workflow_requests': (
        'fingerprint_key_material_verifier',
        'auto_review_mode',
        'auto_review_validator_provider',
        'auto_review_validator_model',
        'auto_review_reasoning_effort',
        'auto_review_validator_prompt_version',
        'auto_review_validator_output_contract_version',
        'auto_review_policy_version',
        'auto_review_cost_policy_version',
        'auto_review_extraction_cost_policy_version',
        'auto_review_token_estimator_version',
        'auto_review_extraction_token_estimator_version',
        'auto_review_tokenizer_encoding',
        'auto_review_reply_priming_tokens',
        'auto_review_framing_safety_tokens',
        'auto_review_max_input_tokens',
        'auto_review_max_output_tokens',
        'auto_review_max_candidates_per_batch',
        'auto_review_max_batches_per_workflow',
        'auto_review_max_candidates_per_workflow',
        'auto_review_max_provider_attempts',
        'auto_review_provider_timeout_seconds',
        'auto_review_provider_send_start_window_seconds',
        'auto_review_provider_attempt_lease_seconds',
        'auto_review_provider_commit_grace_seconds',
        'auto_review_validator_input_usd_per_1m',
        'auto_review_validator_output_usd_per_1m',
        'auto_review_extraction_input_usd_per_1m',
        'auto_review_extraction_output_usd_per_1m',
        'auto_review_enforce_percentage',
        'authorized_percentage_at_launch',
        'rollout_authorization_generation',
        'validation_provider_safety_state_version',
        'extraction_provider_safety_snapshot_set_hmac',
        'rollout_control_epoch',
        'selected_extraction_agent_count',
        'extraction_plan_set_hmac',
        'extraction_max_input_chars_per_agent',
        'extraction_max_input_tokens_per_agent',
        'extraction_max_output_tokens_per_agent',
        'extraction_max_candidates_per_agent',
        'confirmed_extraction_cost_ceiling_usd',
        'confirmed_validation_cost_ceiling_usd',
        'confirmed_total_cost_ceiling_usd',
        'auto_review_budget_limit_usd',
    ),
    'agent_runs': (
        'generation_provider',
        'generation_reasoning_effort',
        'generation_route_version',
        'generation_output_contract_version',
    ),
    'assistant_messages': (
        'evidence_contract_version',
        'serving_dependency_count',
    ),
    'sources': (
        'server_content_signature_schema',
        'server_content_signature',
        'connector_content_signature',
    ),
    'documents': ('current_document_version_id',),
    'document_parser_runs': (
        'server_content_signature_schema',
        'server_content_signature',
        'parser_policy_version',
        'parser_version',
        'chunk_policy_version',
    ),
    'document_chunks': ('parser_run_id',),
    'review_items': (
        'agent_run_id',
        'candidate_contract_version',
        'resolution_source',
        'resolution_policy_version',
        'auto_validation_id',
        'revoked_at',
        'revoked_by_subject_hmac',
        'revoked_by_fingerprint_key_version',
        'revoked_by_fingerprint_key_material_verifier',
        'revoke_knowledge_remained_trusted',
        'revoke_document_count',
    ),
}


def upgrade() -> None:
    _add_model_columns()
    _add_existing_table_constraints_before_new_tables()
    _backfill_connector_evidence_only()
    _backfill_unambiguous_current_document_versions()
    _create_c5_tables()
    _add_existing_table_constraints_after_new_tables()
    _install_postgresql_guards()
    _insert_boundary_marker()


def downgrade() -> None:
    _refuse_retained_c5_state()
    _drop_postgresql_guards()
    _delete_boundary_marker()
    _drop_existing_table_constraints_after_new_tables()
    _drop_c5_tables()
    _drop_existing_table_constraints_before_new_tables()
    _drop_model_columns()


def _add_model_columns() -> None:
    for table_name, column_names in ADDED_COLUMNS.items():
        if table_name not in _table_names():
            continue
        existing = _column_names(table_name)
        model_table = Base.metadata.tables[table_name]
        for column_name in column_names:
            if column_name not in existing:
                op.add_column(table_name, model_table.c[column_name]._copy())


def _add_existing_table_constraints_before_new_tables() -> None:
    _create_unique(
        'agent_workflow_evidence_refs',
        'uq_agent_workflow_evidence_ref_id_workflow',
        ['id', 'workflow_thread_id'],
    )
    _create_unique(
        'agent_runs',
        'uq_agent_runs_id_workflow',
        ['id', 'workflow_thread_id'],
    )
    _create_unique(
        'agent_runs',
        'uq_agent_runs_id_workflow_agent',
        ['id', 'workflow_thread_id', 'agent_name'],
    )
    _create_unique(
        'review_items',
        'uq_review_items_id_workflow',
        ['id', 'workflow_thread_id'],
    )
    _create_unique(
        'document_versions',
        'uq_document_versions_id_document',
        ['id', 'document_id'],
    )
    _create_unique(
        'document_parser_runs',
        'uq_document_parser_runs_identity',
        ['id', 'document_version_id', 'source_id'],
    )
    _create_unique(
        'document_chunks',
        'uq_document_chunks_exact_identity',
        ['id', 'version_id', 'source_id', 'parser_run_id'],
    )
    _create_unique(
        'document_chunks',
        'uq_document_chunks_parser_run_index',
        ['parser_run_id', 'chunk_index'],
    )
    _create_fk(
        'review_items',
        'fk_review_items_agent_run_same_workflow',
        ['agent_run_id', 'workflow_thread_id'],
        'agent_runs',
        ['id', 'workflow_thread_id'],
    )
    _create_fk(
        'documents',
        'fk_documents_current_version_same_document',
        ['current_document_version_id', 'id'],
        'document_versions',
        ['id', 'document_id'],
        deferrable=True,
        initially='DEFERRED',
    )
    _create_fk(
        'document_chunks',
        'fk_document_chunks_parser_run_identity',
        ['parser_run_id', 'version_id', 'source_id'],
        'document_parser_runs',
        ['id', 'document_version_id', 'source_id'],
    )
    _create_check(
        'sources',
        'ck_sources_server_content_signature_authority',
        '(server_content_signature IS NULL AND '
        'server_content_signature_schema IS NULL) OR '
        "(server_content_signature_schema = 'server-source-content:v1' AND "
        'length(server_content_signature) = 64)',
    )
    _create_check(
        'document_parser_runs',
        'ck_document_parser_runs_c5_identity',
        '(server_content_signature IS NULL AND '
        'server_content_signature_schema IS NULL AND '
        'parser_policy_version IS NULL AND parser_version IS NULL AND '
        'chunk_policy_version IS NULL) OR '
        "(server_content_signature_schema = 'server-source-content:v1' AND "
        'length(server_content_signature) = 64 AND '
        'parser_policy_version IS NOT NULL AND parser_version IS NOT NULL AND '
        'chunk_policy_version IS NOT NULL)',
    )
    _create_check(
        'assistant_messages',
        'ck_assistant_messages_evidence_contract',
        "evidence_contract_version IS NULL OR evidence_contract_version IN "
        "('none-v1', 'assistant-evidence:v1')",
    )
    _create_check(
        'assistant_messages',
        'ck_assistant_messages_serving_dependency_count',
        'serving_dependency_count IS NULL OR serving_dependency_count >= 0',
    )


def _add_existing_table_constraints_after_new_tables() -> None:
    _create_fk(
        'review_items',
        'fk_review_items_auto_validation_same_item',
        ['id', 'auto_validation_id'],
        'auto_review_validations',
        ['review_item_id', 'id'],
        deferrable=True,
        initially='DEFERRED',
    )


def _create_c5_tables() -> None:
    required_parents = {
        'agent_runs',
        'agent_workflow_evidence_refs',
        'assistant_messages',
        'document_chunks',
        'review_items',
    }
    if not required_parents <= _table_names():
        return
    tables = [Base.metadata.tables[name] for name in C5_TABLE_NAMES]
    Base.metadata.create_all(op.get_bind(), tables=tables, checkfirst=True)


def _backfill_connector_evidence_only() -> None:
    bind = op.get_bind()
    if 'sources' not in _table_names():
        return
    if bind.dialect.name == 'postgresql':
        bind.execute(
            sa.text(
                "UPDATE sources SET connector_content_signature = "
                "raw_metadata ->> 'content_signature' "
                "WHERE connector_content_signature IS NULL AND "
                "raw_metadata ->> 'content_signature' IS NOT NULL"
            )
        )
    elif bind.dialect.name == 'sqlite':
        bind.execute(
            sa.text(
                "UPDATE sources SET connector_content_signature = "
                "json_extract(raw_metadata, '$.content_signature') "
                "WHERE connector_content_signature IS NULL AND "
                "json_extract(raw_metadata, '$.content_signature') IS NOT NULL"
            )
        )


def _backfill_unambiguous_current_document_versions() -> None:
    if {'documents', 'document_versions'} <= _table_names():
        op.get_bind().execute(
            sa.text(
                'UPDATE documents SET current_document_version_id = '
                '(SELECT MIN(document_versions.id) FROM document_versions '
                'WHERE document_versions.document_id = documents.id) '
                'WHERE current_document_version_id IS NULL AND '
                '(SELECT COUNT(*) FROM document_versions '
                'WHERE document_versions.document_id = documents.id) = 1'
            )
        )


def _insert_boundary_marker() -> None:
    bind = op.get_bind()
    if not set(C5_TABLE_NAMES) <= _table_names():
        return
    exists = bind.scalar(
        sa.text(
            'SELECT COUNT(*) FROM agent_runtime_schema_versions '
            'WHERE component=:component'
        ),
        {'component': BOUNDARY_COMPONENT},
    )
    if not exists:
        bind.execute(
            sa.text(
                'INSERT INTO agent_runtime_schema_versions '
                '(component, package_name, package_version, schema_revision, applied_at) '
                "VALUES (:component, 'paraworks', 'c5-v1', 1, CURRENT_TIMESTAMP)"
            ),
            {'component': BOUNDARY_COMPONENT},
        )


def _delete_boundary_marker() -> None:
    if 'agent_runtime_schema_versions' in _table_names():
        op.get_bind().execute(
            sa.text(
                'DELETE FROM agent_runtime_schema_versions WHERE component=:component'
            ),
            {'component': BOUNDARY_COMPONENT},
        )


def _install_postgresql_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    if not set(C5_TABLE_NAMES) <= _table_names():
        return
    for statement in _postgresql_guard_statements():
        op.execute(sa.text(statement))


def _postgresql_guard_statements() -> tuple[str, ...]:
    return (
        """
        CREATE OR REPLACE FUNCTION enforce_review_item_post_c5_evidence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.workflow_thread_id IS NOT NULL THEN
            IF NEW.candidate_contract_version IS DISTINCT FROM 'c5-v1'
               OR NEW.agent_run_id IS NULL
               OR NOT EXISTS (
                 SELECT 1 FROM review_item_evidence_refs ref
                 WHERE ref.review_item_id = NEW.id
                   AND ref.workflow_thread_id = NEW.workflow_thread_id
               ) THEN
              RAISE EXCEPTION 'post-C.5 workflow ReviewItem requires exact provenance';
            END IF;
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_review_item_post_c5_evidence_guard
        AFTER INSERT OR UPDATE OF workflow_thread_id, agent_run_id,
          candidate_contract_version ON review_items
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_review_item_post_c5_evidence()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_review_item_evidence_child_guard()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE parent_id integer := COALESCE(NEW.review_item_id, OLD.review_item_id);
        BEGIN
          IF TG_OP = 'DELETE' AND EXISTS (
             SELECT 1 FROM review_items item WHERE item.id = parent_id
             AND item.candidate_contract_version = 'c5-v1') THEN
            RAISE EXCEPTION 'bound ReviewItem evidence is immutable';
          END IF;
          IF TG_OP = 'UPDATE' AND (
             NEW.review_item_id IS DISTINCT FROM OLD.review_item_id OR
             NEW.workflow_thread_id IS DISTINCT FROM OLD.workflow_thread_id OR
             NEW.workflow_evidence_ref_id IS DISTINCT FROM OLD.workflow_evidence_ref_id OR
             NEW.candidate_slot_ordinal IS DISTINCT FROM OLD.candidate_slot_ordinal OR
             NEW.message_content_fingerprint IS DISTINCT FROM OLD.message_content_fingerprint OR
             NEW.fingerprint_key_version IS DISTINCT FROM OLD.fingerprint_key_version OR
             NEW.fingerprint_key_material_verifier IS DISTINCT FROM
               OLD.fingerprint_key_material_verifier) THEN
            RAISE EXCEPTION 'bound ReviewItem evidence is immutable';
          END IF;
          IF EXISTS (SELECT 1 FROM review_items item WHERE item.id = parent_id
                     AND item.candidate_contract_version = 'c5-v1')
             AND NOT EXISTS (SELECT 1 FROM review_item_evidence_refs ref
                             WHERE ref.review_item_id = parent_id) THEN
            RAISE EXCEPTION 'C.5 ReviewItem must retain evidence';
          END IF;
          RETURN COALESCE(NEW, OLD);
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_review_item_evidence_child_guard
        AFTER UPDATE OR DELETE ON review_item_evidence_refs
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_review_item_evidence_child_guard()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_review_item_c5_identity_immutable()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.candidate_contract_version = 'c5-v1' AND (
             NEW.workflow_thread_id IS DISTINCT FROM OLD.workflow_thread_id OR
             NEW.agent_run_id IS DISTINCT FROM OLD.agent_run_id OR
             NEW.candidate_contract_version IS DISTINCT FROM OLD.candidate_contract_version)
          THEN RAISE EXCEPTION 'C.5 ReviewItem ownership is immutable'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_review_item_c5_identity_immutable
        BEFORE UPDATE ON review_items FOR EACH ROW
        EXECUTE FUNCTION enforce_review_item_c5_identity_immutable()
        """,
        """
        CREATE OR REPLACE FUNCTION protect_auto_review_schema_boundary()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.component = 'auto_review_trust_promotion' THEN
            RAISE EXCEPTION 'auto-review schema boundary is immutable';
          END IF;
          RETURN COALESCE(NEW, OLD);
        END $$
        """,
        """
        CREATE TRIGGER trg_auto_review_schema_boundary_immutable
        BEFORE UPDATE OR DELETE ON agent_runtime_schema_versions FOR EACH ROW
        EXECUTE FUNCTION protect_auto_review_schema_boundary()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_assistant_message_evidence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE dependency_count integer;
        BEGIN
          SELECT COUNT(*) INTO dependency_count
          FROM assistant_message_evidence_dependencies dep
          WHERE dep.assistant_message_id = NEW.id;
          IF NEW.evidence_contract_version IS NULL OR
             NEW.serving_dependency_count IS NULL OR
             NEW.serving_dependency_count <> dependency_count OR
             (NEW.evidence_contract_version = 'assistant-evidence:v1'
                AND dependency_count < 1) OR
             (NEW.evidence_contract_version = 'none-v1' AND dependency_count <> 0)
          THEN RAISE EXCEPTION 'assistant evidence dependency contract mismatch'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_assistant_message_evidence_guard
        AFTER INSERT OR UPDATE OF evidence_contract_version, serving_dependency_count
        ON assistant_messages DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_assistant_message_evidence()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_assistant_dependency_child_count()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE message_id integer := COALESCE(
          NEW.assistant_message_id, OLD.assistant_message_id);
        DECLARE stored_count integer;
        DECLARE actual_count integer;
        DECLARE contract_version varchar;
        BEGIN
          SELECT serving_dependency_count, evidence_contract_version
            INTO stored_count, contract_version
          FROM assistant_messages WHERE id = message_id;
          SELECT COUNT(*) INTO actual_count
          FROM assistant_message_evidence_dependencies
          WHERE assistant_message_id = message_id;
          IF stored_count IS DISTINCT FROM actual_count OR
             (contract_version = 'assistant-evidence:v1' AND actual_count < 1) OR
             (contract_version = 'none-v1' AND actual_count <> 0)
          THEN RAISE EXCEPTION 'assistant evidence dependency contract mismatch'; END IF;
          RETURN COALESCE(NEW, OLD);
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_assistant_dependency_child_count_guard
        AFTER INSERT OR UPDATE OR DELETE ON assistant_message_evidence_dependencies
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_assistant_dependency_child_count()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_provider_safety_event_sequence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE parent_sequence integer;
        DECLARE parent_event_id integer;
        DECLARE parent_state_version integer;
        DECLARE prior_sequence integer;
        BEGIN
          SELECT last_event_sequence, last_event_id, state_version
            INTO parent_sequence, parent_event_id, parent_state_version
          FROM auto_review_provider_safety_states
          WHERE id = NEW.provider_safety_state_id;
          SELECT COALESCE(MAX(event_sequence), 0) INTO prior_sequence
          FROM auto_review_provider_safety_events
          WHERE provider_safety_state_id = NEW.provider_safety_state_id
            AND id <> NEW.id;
          IF NEW.event_sequence <> prior_sequence + 1 OR
             parent_sequence <> NEW.event_sequence OR
             parent_event_id <> NEW.id OR
             parent_state_version <> NEW.new_state_version
          THEN RAISE EXCEPTION 'provider safety events require gapless atomic backpointer';
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_provider_safety_event_sequence_guard
        AFTER INSERT ON auto_review_provider_safety_events
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_provider_safety_event_sequence()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_rollout_control_event_sequence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE parent_sequence integer;
        DECLARE parent_event_id integer;
        DECLARE parent_state_version integer;
        DECLARE parent_control_epoch integer;
        DECLARE prior_sequence integer;
        BEGIN
          SELECT last_event_sequence, last_event_id, state_version, control_epoch
            INTO parent_sequence, parent_event_id, parent_state_version,
                 parent_control_epoch
          FROM auto_review_rollout_states WHERE id = NEW.rollout_state_id;
          SELECT COALESCE(MAX(event_sequence), 0) INTO prior_sequence
          FROM auto_review_rollout_control_events
          WHERE rollout_state_id = NEW.rollout_state_id AND id <> NEW.id;
          IF NEW.event_sequence <> prior_sequence + 1 OR
             parent_sequence <> NEW.event_sequence OR parent_event_id <> NEW.id OR
             parent_state_version <> NEW.new_state_version OR
             parent_control_epoch <> NEW.new_control_epoch
          THEN RAISE EXCEPTION 'rollout events require gapless atomic backpointer';
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_rollout_control_event_sequence_guard
        AFTER INSERT ON auto_review_rollout_control_events
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_rollout_control_event_sequence()
        """,
        """
        CREATE OR REPLACE FUNCTION reject_c5_immutable_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'C.5 audit/evidence row is append-only'; END $$
        """,
        """
        CREATE TRIGGER trg_provider_safety_event_append_only
        BEFORE UPDATE OR DELETE ON auto_review_provider_safety_events
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE TRIGGER trg_rollout_control_event_append_only
        BEFORE UPDATE OR DELETE ON auto_review_rollout_control_events
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE TRIGGER trg_promotion_decision_immutable
        BEFORE UPDATE OR DELETE ON auto_review_promotion_decisions
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE TRIGGER trg_revocation_assessment_immutable
        BEFORE UPDATE OR DELETE ON auto_review_revocation_assessments
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE TRIGGER trg_assistant_dependency_immutable
        BEFORE UPDATE OR DELETE ON assistant_message_evidence_dependencies
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE TRIGGER trg_assistant_dependency_child_immutable
        BEFORE UPDATE OR DELETE ON assistant_message_knowledge_evidence_refs
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_assistant_message_immutable()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.evidence_contract_version IS NOT NULL AND (
             TG_OP = 'DELETE' OR NEW.content IS DISTINCT FROM OLD.content OR
             NEW.citations IS DISTINCT FROM OLD.citations OR
             NEW.source_ids IS DISTINCT FROM OLD.source_ids OR
             NEW.source_links IS DISTINCT FROM OLD.source_links OR
             NEW.source_snippets IS DISTINCT FROM OLD.source_snippets OR
             NEW.permission_level IS DISTINCT FROM OLD.permission_level OR
             NEW.evidence_contract_version IS DISTINCT FROM OLD.evidence_contract_version OR
             NEW.serving_dependency_count IS DISTINCT FROM OLD.serving_dependency_count)
          THEN RAISE EXCEPTION 'committed assistant evidence is immutable'; END IF;
          RETURN COALESCE(NEW, OLD);
        END $$
        """,
        """
        CREATE TRIGGER trg_assistant_message_committed_immutable
        BEFORE UPDATE OR DELETE ON assistant_messages FOR EACH ROW
        EXECUTE FUNCTION enforce_assistant_message_immutable()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_audit_correction_monotonic()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            RAISE EXCEPTION 'auto-review audit correction is immutable';
          END IF;
          IF NEW.post_audit_id IS DISTINCT FROM OLD.post_audit_id OR
             NEW.review_item_id IS DISTINCT FROM OLD.review_item_id OR
             NEW.assessment_id IS DISTINCT FROM OLD.assessment_id OR
             NEW.effective_outcome IS DISTINCT FROM OLD.effective_outcome OR
             NEW.actor_subject_hmac IS DISTINCT FROM OLD.actor_subject_hmac OR
             NEW.actor_fingerprint_key_version IS DISTINCT FROM
               OLD.actor_fingerprint_key_version OR
             NEW.actor_fingerprint_key_material_verifier IS DISTINCT FROM
               OLD.actor_fingerprint_key_material_verifier OR
             NEW.created_at IS DISTINCT FROM OLD.created_at OR
             NOT ((OLD.status = 'remediation_required' AND
                   OLD.system_resolution_code = 'revoke_pending' AND
                   NEW.status = 'remediation_required' AND
                   NEW.system_resolution_code = 'revoke_failed' AND
                   NEW.completed_at IS NULL) OR
                  (OLD.status = 'remediation_required' AND
                   NEW.status = 'completed' AND
                   NEW.system_resolution_code = 'revoked' AND
                   NEW.completed_at IS NOT NULL))
          THEN RAISE EXCEPTION 'auto-review audit correction transition is not monotonic';
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_auto_review_audit_correction_monotonic
        BEFORE UPDATE OR DELETE ON auto_review_audit_corrections FOR EACH ROW
        EXECUTE FUNCTION enforce_audit_correction_monotonic()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_c5_parser_identity_immutable()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.server_content_signature IS NOT NULL AND (
             NEW.server_content_signature_schema IS DISTINCT FROM
               OLD.server_content_signature_schema OR
             NEW.server_content_signature IS DISTINCT FROM OLD.server_content_signature OR
             NEW.parser_policy_version IS DISTINCT FROM OLD.parser_policy_version OR
             NEW.parser_version IS DISTINCT FROM OLD.parser_version OR
             NEW.chunk_policy_version IS DISTINCT FROM OLD.chunk_policy_version)
          THEN RAISE EXCEPTION 'C.5 parser identity is immutable'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_document_parser_run_c5_identity_immutable
        BEFORE UPDATE ON document_parser_runs FOR EACH ROW
        EXECUTE FUNCTION enforce_c5_parser_identity_immutable()
        """,
    )


def _drop_postgresql_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    for table_name, trigger_name in (
        ('document_parser_runs', 'trg_document_parser_run_c5_identity_immutable'),
        ('auto_review_audit_corrections', 'trg_auto_review_audit_correction_monotonic'),
        ('assistant_messages', 'trg_assistant_message_committed_immutable'),
        ('assistant_message_knowledge_evidence_refs', 'trg_assistant_dependency_child_immutable'),
        ('assistant_message_evidence_dependencies', 'trg_assistant_dependency_immutable'),
        ('auto_review_revocation_assessments', 'trg_revocation_assessment_immutable'),
        ('auto_review_promotion_decisions', 'trg_promotion_decision_immutable'),
        ('auto_review_rollout_control_events', 'trg_rollout_control_event_append_only'),
        ('auto_review_provider_safety_events', 'trg_provider_safety_event_append_only'),
        ('auto_review_rollout_control_events', 'trg_rollout_control_event_sequence_guard'),
        ('auto_review_provider_safety_events', 'trg_provider_safety_event_sequence_guard'),
        ('assistant_message_evidence_dependencies', 'trg_assistant_dependency_child_count_guard'),
        ('assistant_messages', 'trg_assistant_message_evidence_guard'),
        ('agent_runtime_schema_versions', 'trg_auto_review_schema_boundary_immutable'),
        ('review_items', 'trg_review_item_c5_identity_immutable'),
        ('review_item_evidence_refs', 'trg_review_item_evidence_child_guard'),
        ('review_items', 'trg_review_item_post_c5_evidence_guard'),
    ):
        op.execute(sa.text(f'DROP TRIGGER IF EXISTS {trigger_name} ON {table_name}'))
    for function_name in (
        'enforce_c5_parser_identity_immutable',
        'enforce_audit_correction_monotonic',
        'enforce_assistant_message_immutable',
        'reject_c5_immutable_mutation',
        'enforce_rollout_control_event_sequence',
        'enforce_provider_safety_event_sequence',
        'enforce_assistant_dependency_child_count',
        'enforce_assistant_message_evidence',
        'protect_auto_review_schema_boundary',
        'enforce_review_item_c5_identity_immutable',
        'enforce_review_item_evidence_child_guard',
        'enforce_review_item_post_c5_evidence',
    ):
        op.execute(sa.text(f'DROP FUNCTION IF EXISTS {function_name}()'))


def _refuse_retained_c5_state() -> None:
    bind = op.get_bind()
    tables = _table_names()
    for table_name in C5_TABLE_NAMES:
        if table_name in tables and bind.scalar(
            sa.text(f'SELECT COUNT(*) FROM {table_name}')
        ):
            raise RuntimeError(
                f'retained C.5 state in {table_name}; schema downgrade refused'
            )
    predicates = {
        'agent_workflow_threads': "graph_version = 'company-memory-review-v2.1-auto-review'",
        'agent_workflow_requests': ' OR '.join(
            f'{name} IS NOT NULL' for name in ADDED_COLUMNS['agent_workflow_requests']
        ),
        'review_items': ' OR '.join(
            f'{name} IS NOT NULL' for name in ADDED_COLUMNS['review_items']
        ),
        'agent_runs': ' OR '.join(
            f'{name} IS NOT NULL' for name in ADDED_COLUMNS['agent_runs']
        ),
        'assistant_messages': ' OR '.join(
            f'{name} IS NOT NULL' for name in ADDED_COLUMNS['assistant_messages']
        ),
        'sources': ' OR '.join(
            f'{name} IS NOT NULL' for name in ADDED_COLUMNS['sources']
        ),
        'documents': 'current_document_version_id IS NOT NULL',
        'document_parser_runs': ' OR '.join(
            f'{name} IS NOT NULL' for name in ADDED_COLUMNS['document_parser_runs']
        ),
        'document_chunks': 'parser_run_id IS NOT NULL',
    }
    for table_name, predicate in predicates.items():
        if table_name == 'review_items' and 'status' in _column_names(table_name):
            predicate += " OR status = 'revoked'"
        if table_name in tables and bind.scalar(
            sa.text(f'SELECT COUNT(*) FROM {table_name} WHERE {predicate}')
        ):
            raise RuntimeError(
                f'retained C.5 state in {table_name}; schema downgrade refused'
            )


def _drop_existing_table_constraints_after_new_tables() -> None:
    _drop_fk('review_items', 'fk_review_items_auto_validation_same_item')


def _drop_c5_tables() -> None:
    existing = _table_names()
    tables = [
        Base.metadata.tables[name]
        for name in C5_TABLE_NAMES
        if name in existing
    ]
    Base.metadata.drop_all(op.get_bind(), tables=tables, checkfirst=True)


def _drop_existing_table_constraints_before_new_tables() -> None:
    for table_name, constraint_name in (
        ('assistant_messages', 'ck_assistant_messages_serving_dependency_count'),
        ('assistant_messages', 'ck_assistant_messages_evidence_contract'),
        ('document_parser_runs', 'ck_document_parser_runs_c5_identity'),
        ('sources', 'ck_sources_server_content_signature_authority'),
    ):
        _drop_check(table_name, constraint_name)
    for table_name, constraint_name in (
        ('document_chunks', 'fk_document_chunks_parser_run_identity'),
        ('documents', 'fk_documents_current_version_same_document'),
        ('review_items', 'fk_review_items_agent_run_same_workflow'),
    ):
        _drop_fk(table_name, constraint_name)
    for table_name, constraint_name in (
        ('document_chunks', 'uq_document_chunks_parser_run_index'),
        ('document_chunks', 'uq_document_chunks_exact_identity'),
        ('document_parser_runs', 'uq_document_parser_runs_identity'),
        ('document_versions', 'uq_document_versions_id_document'),
        ('review_items', 'uq_review_items_id_workflow'),
        ('agent_runs', 'uq_agent_runs_id_workflow_agent'),
        ('agent_runs', 'uq_agent_runs_id_workflow'),
        ('agent_workflow_evidence_refs', 'uq_agent_workflow_evidence_ref_id_workflow'),
    ):
        _drop_unique(table_name, constraint_name)


def _drop_model_columns() -> None:
    for table_name, column_names in reversed(tuple(ADDED_COLUMNS.items())):
        if table_name not in _table_names():
            continue
        existing = _column_names(table_name)
        for column_name in reversed(column_names):
            if column_name in existing:
                with op.batch_alter_table(table_name) as batch:
                    batch.drop_column(column_name)


def _create_unique(table_name: str, name: str, columns: list[str]) -> None:
    if table_name not in _table_names():
        return
    if name in _constraint_names(table_name, 'unique'):
        return
    with op.batch_alter_table(table_name) as batch:
        batch.create_unique_constraint(name, columns)


def _create_fk(
    table_name: str,
    name: str,
    local_columns: list[str],
    remote_table: str,
    remote_columns: list[str],
    *,
    deferrable: bool | None = None,
    initially: str | None = None,
) -> None:
    if table_name not in _table_names() or remote_table not in _table_names():
        return
    if name in _constraint_names(table_name, 'foreign_key'):
        return
    with op.batch_alter_table(table_name) as batch:
        batch.create_foreign_key(
            name,
            remote_table,
            local_columns,
            remote_columns,
            deferrable=deferrable,
            initially=initially,
        )


def _create_check(table_name: str, name: str, condition: str) -> None:
    if table_name not in _table_names():
        return
    if name in _constraint_names(table_name, 'check'):
        return
    with op.batch_alter_table(table_name) as batch:
        batch.create_check_constraint(name, condition)


def _drop_unique(table_name: str, name: str) -> None:
    if name in _constraint_names(table_name, 'unique'):
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(name, type_='unique')


def _drop_fk(table_name: str, name: str) -> None:
    if name in _constraint_names(table_name, 'foreign_key'):
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(name, type_='foreignkey')


def _drop_check(table_name: str, name: str) -> None:
    if name in _constraint_names(table_name, 'check'):
        with op.batch_alter_table(table_name) as batch:
            batch.drop_constraint(name, type_='check')


def _constraint_names(table_name: str, kind: str) -> set[str | None]:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return set()
    getters = {
        'unique': inspector.get_unique_constraints,
        'foreign_key': inspector.get_foreign_keys,
        'check': inspector.get_check_constraints,
    }
    return {item['name'] for item in getters[kind](table_name)}


def _table_names() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    return {
        item['name'] for item in inspect(op.get_bind()).get_columns(table_name)
    }
