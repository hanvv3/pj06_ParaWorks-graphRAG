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
from backend.app.admin.auto_review_retained_state import (
    C5_ADDED_COLUMNS,
    C5_TABLE_NAMES,
    has_retained_c5_state,
)
from backend.app.db.base import Base

revision = '7c5a2e9f4b10'
down_revision = '2f6a8b9c0d1e'
branch_labels = None
depends_on = None

BOUNDARY_COMPONENT = 'auto_review_trust_promotion'

_LOWER_HEX_64_REMAINDER = 'server_content_signature'
for _character in '0123456789abcdef':
    _LOWER_HEX_64_REMAINDER = (
        f"replace({_LOWER_HEX_64_REMAINDER}, '{_character}', '')"
    )

ADDED_COLUMNS = C5_ADDED_COLUMNS


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
        'length(server_content_signature) = 64 AND '
        f'{_LOWER_HEX_64_REMAINDER} = \'\')',
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
        f'{_LOWER_HEX_64_REMAINDER} = \'\' AND '
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
        if {'sources', 'document_parser_runs'} <= _table_names():
            op.get_bind().execute(
                sa.text(
                    'UPDATE documents SET current_document_version_id = '
                    '(SELECT MIN(run.document_version_id) '
                    'FROM document_parser_runs run JOIN sources source '
                    'ON source.id = run.source_id '
                    'WHERE run.document_id = documents.id '
                    'AND run.server_content_signature_schema = '
                    "'server-source-content:v1' "
                    'AND run.server_content_signature = '
                    'source.server_content_signature '
                    'AND source.server_content_signature_schema = '
                    "'server-source-content:v1' "
                    'AND run.parser_policy_version IS NOT NULL '
                    'AND run.parser_version IS NOT NULL '
                    'AND run.chunk_policy_version IS NOT NULL) '
                    'WHERE current_document_version_id IS NULL AND '
                    '(SELECT COUNT(*) FROM document_parser_runs run '
                    'JOIN sources source ON source.id = run.source_id '
                    'WHERE run.document_id = documents.id '
                    'AND run.server_content_signature_schema = '
                    "'server-source-content:v1' "
                    'AND run.server_content_signature = '
                    'source.server_content_signature '
                    'AND source.server_content_signature_schema = '
                    "'server-source-content:v1' "
                    'AND run.parser_policy_version IS NOT NULL '
                    'AND run.parser_version IS NOT NULL '
                    'AND run.chunk_policy_version IS NOT NULL) = 1'
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
        CREATE OR REPLACE FUNCTION enforce_validation_call_terminal_children()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE new_row jsonb := COALESCE(to_jsonb(NEW), '{}'::jsonb);
        DECLARE old_row jsonb := COALESCE(to_jsonb(OLD), '{}'::jsonb);
        DECLARE call_id integer;
        DECLARE parent auto_review_validation_calls%ROWTYPE;
        DECLARE child_count integer;
        DECLARE claimed_count integer;
        DECLARE completed_count integer;
        DECLARE failed_count integer;
        BEGIN
          call_id := COALESCE(
            (new_row ->> 'validation_call_id')::integer,
            (old_row ->> 'validation_call_id')::integer,
            (new_row ->> 'id')::integer,
            (old_row ->> 'id')::integer);
          SELECT * INTO parent FROM auto_review_validation_calls WHERE id = call_id;
          IF NOT FOUND THEN RETURN COALESCE(NEW, OLD); END IF;
          SELECT COUNT(*), COUNT(*) FILTER (WHERE status = 'claimed'),
                 COUNT(*) FILTER (WHERE status = 'completed'),
                 COUNT(*) FILTER (WHERE status = 'failed')
            INTO child_count, claimed_count, completed_count, failed_count
          FROM auto_review_validations WHERE validation_call_id = call_id;
          IF child_count <> parent.candidate_count OR child_count NOT BETWEEN 1 AND 4 OR
             (parent.status = 'claimed' AND claimed_count <> child_count) OR
             (parent.status = 'completed' AND completed_count <> child_count) OR
             (parent.status = 'failed' AND failed_count <> child_count)
          THEN RAISE EXCEPTION 'validation terminal children mismatch'; END IF;
          RETURN COALESCE(NEW, OLD);
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_validation_call_terminal_children_guard
        AFTER INSERT OR UPDATE ON auto_review_validation_calls
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_validation_call_terminal_children()
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_validation_child_terminal_parent_guard
        AFTER INSERT OR UPDATE OR DELETE ON auto_review_validations
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_validation_call_terminal_children()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_validation_child_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE parent_status varchar;
        BEGIN
          SELECT status INTO parent_status FROM auto_review_validation_calls
          WHERE id = OLD.validation_call_id;
          IF TG_OP = 'DELETE' OR OLD.status <> 'claimed' OR
             parent_status <> 'claimed' OR
             NEW.review_item_id IS DISTINCT FROM OLD.review_item_id OR
             NEW.validation_call_id IS DISTINCT FROM OLD.validation_call_id OR
             NEW.workflow_thread_id IS DISTINCT FROM OLD.workflow_thread_id OR
             NEW.validation_key IS DISTINCT FROM OLD.validation_key OR
             NEW.evidence_version_hash IS DISTINCT FROM OLD.evidence_version_hash OR
             NEW.candidate_generation_fingerprint IS DISTINCT FROM
               OLD.candidate_generation_fingerprint
          THEN RAISE EXCEPTION 'validation child is immutable'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_validation_child_mutation_guard
        BEFORE UPDATE OR DELETE ON auto_review_validations FOR EACH ROW
        EXECUTE FUNCTION enforce_validation_child_mutation()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_assistant_dependency_exactness()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE new_row jsonb := COALESCE(to_jsonb(NEW), '{}'::jsonb);
        DECLARE old_row jsonb := COALESCE(to_jsonb(OLD), '{}'::jsonb);
        DECLARE dependency_id integer;
        DECLARE dep assistant_message_evidence_dependencies%ROWTYPE;
        DECLARE expected_count integer;
        DECLARE actual_count integer;
        DECLARE human_proven boolean := false;
        BEGIN
          dependency_id := COALESCE(
            (new_row ->> 'dependency_id')::integer,
            (old_row ->> 'dependency_id')::integer,
            (new_row ->> 'id')::integer,
            (old_row ->> 'id')::integer);
          SELECT * INTO dep FROM assistant_message_evidence_dependencies
          WHERE id = dependency_id;
          IF NOT FOUND THEN RETURN COALESCE(NEW, OLD); END IF;
          IF dep.dependency_kind = 'raw_chunk' THEN
            IF NOT EXISTS (
              SELECT 1 FROM document_chunks chunk
              JOIN document_parser_runs parser ON parser.id = chunk.parser_run_id
              JOIN document_versions version ON version.id = chunk.version_id
              JOIN documents document ON document.id = version.document_id
              JOIN sources source ON source.id = chunk.source_id
              WHERE chunk.id = dep.document_chunk_id
                AND chunk.version_id = dep.document_version_id
                AND chunk.source_id = dep.source_id
                AND chunk.parser_run_id = dep.parser_run_id
                AND document.current_document_version_id = dep.document_version_id
                AND dep.current_document_version_id = dep.document_version_id
                AND parser.document_id = document.id
                AND parser.document_version_id = dep.document_version_id
                AND parser.source_id = dep.source_id
                AND parser.server_content_signature_schema =
                    dep.server_content_signature_schema
                AND parser.server_content_signature = dep.server_content_signature
                AND parser.parser_policy_version = dep.parser_policy_version
                AND parser.parser_version = dep.parser_version
                AND parser.chunk_policy_version = dep.chunk_policy_version
                AND source.server_content_signature_schema =
                    dep.server_content_signature_schema
                AND source.server_content_signature = dep.server_content_signature
            ) THEN RAISE EXCEPTION 'raw dependency is not current'; END IF;
          ELSIF dep.approval_link_id IS NOT NULL THEN
            SELECT COUNT(*) INTO expected_count FROM trusted_knowledge_evidence_links
            WHERE approval_link_id = dep.approval_link_id;
            SELECT COUNT(*) INTO actual_count
            FROM assistant_message_knowledge_evidence_refs ref
            WHERE ref.dependency_id = dep.id;
            IF expected_count < 1 OR actual_count <> expected_count OR NOT EXISTS (
              SELECT 1 FROM trusted_knowledge_approval_links approval
              WHERE approval.id = dep.approval_link_id
                AND approval.knowledge_type = dep.knowledge_type
                AND approval.knowledge_id = dep.knowledge_id
                AND approval.active = true AND approval.revoked_at IS NULL
            ) THEN RAISE EXCEPTION 'complete active approval effect required'; END IF;
          ELSE
            SELECT EXISTS (
              SELECT 1 FROM review_items item
              WHERE item.id = dep.legacy_source_review_item_id
                AND item.status = 'approved' AND item.resolution_source = 'human'
            ) INTO human_proven;
            IF human_proven AND dep.knowledge_type = 'timeline_event' THEN
              SELECT EXISTS (SELECT 1 FROM timeline_events target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'history_event' THEN
              SELECT EXISTS (SELECT 1 FROM history_events target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'decision_record' THEN
              SELECT EXISTS (SELECT 1 FROM decision_records target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSIF human_proven AND dep.knowledge_type = 'todo' THEN
              SELECT EXISTS (SELECT 1 FROM todos target
                WHERE target.id = dep.knowledge_id AND
                  target.source_review_item_id = dep.legacy_source_review_item_id)
                INTO human_proven;
            ELSE human_proven := false;
            END IF;
            IF NOT human_proven THEN RAISE EXCEPTION 'legacy human proof required';
            END IF;
          END IF;
          RETURN COALESCE(NEW, OLD);
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_assistant_dependency_exactness_guard
        AFTER INSERT OR UPDATE ON assistant_message_evidence_dependencies
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_assistant_dependency_exactness()
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_assistant_dependency_child_exactness_guard
        AFTER INSERT OR UPDATE OR DELETE ON assistant_message_knowledge_evidence_refs
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_assistant_dependency_exactness()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_provider_safety_event_sequence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE parent_sequence integer;
        DECLARE parent_event_id integer;
        DECLARE parent_state_version integer;
        DECLARE prior_sequence integer;
        DECLARE prior_event auto_review_provider_safety_events%ROWTYPE;
        BEGIN
          SELECT last_event_sequence, last_event_id, state_version
            INTO parent_sequence, parent_event_id, parent_state_version
          FROM auto_review_provider_safety_states
          WHERE id = NEW.provider_safety_state_id;
          SELECT COALESCE(MAX(event_sequence), 0) INTO prior_sequence
          FROM auto_review_provider_safety_events
          WHERE provider_safety_state_id = NEW.provider_safety_state_id
            AND id <> NEW.id;
          IF NEW.event_sequence = 1 THEN
            IF NEW.event_kind <> 'initial_authorized' OR
               NEW.prior_state_version <> 0 OR NEW.prior_breaker_open <> false
            THEN RAISE EXCEPTION 'prior provider snapshot mismatch'; END IF;
          ELSE
            SELECT * INTO prior_event FROM auto_review_provider_safety_events
            WHERE provider_safety_state_id = NEW.provider_safety_state_id
              AND event_sequence = NEW.event_sequence - 1;
            IF NOT FOUND THEN
              RAISE EXCEPTION 'provider safety events require gapless atomic backpointer';
            END IF;
            IF NEW.event_kind = 'initial_authorized' OR
               prior_event.new_state_version <> NEW.prior_state_version OR
               prior_event.new_breaker_open <> NEW.prior_breaker_open
            THEN RAISE EXCEPTION 'prior provider snapshot mismatch'; END IF;
          END IF;
          IF (NEW.event_kind = 'budget_overrun' AND NEW.new_breaker_open <> true) OR
             (NEW.event_kind = 'breaker_cleared' AND
               (NEW.new_breaker_open <> false OR
                NEW.cost_policy_version = prior_event.cost_policy_version)) OR
             (NEW.event_kind = 'initial_authorized' AND
               NEW.new_breaker_open <> false)
          THEN RAISE EXCEPTION 'provider event transition mismatch'; END IF;
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
        CREATE OR REPLACE FUNCTION enforce_provider_safety_state_event()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE aggregate auto_review_provider_safety_states%ROWTYPE;
        DECLARE event auto_review_provider_safety_events%ROWTYPE;
        DECLARE protected_changed boolean;
        BEGIN
          SELECT * INTO aggregate FROM auto_review_provider_safety_states
          WHERE id = NEW.id;
          SELECT * INTO event FROM auto_review_provider_safety_events candidate
          WHERE candidate.id = aggregate.last_event_id
            AND candidate.provider_safety_state_id = aggregate.id
            AND candidate.event_sequence = aggregate.last_event_sequence;
          IF aggregate.last_event_id IS NULL OR
             aggregate.last_event_sequence = 0 OR NOT FOUND OR
             event.purpose <> aggregate.purpose OR
             event.provider <> aggregate.provider OR
             event.model <> aggregate.model OR
             event.reasoning_effort <> aggregate.reasoning_effort OR
             event.new_state_version <> aggregate.state_version OR
             event.cost_policy_version <>
                aggregate.authorized_cost_policy_version
             OR event.token_estimator_version <>
                aggregate.token_estimator_version
             OR event.tokenizer_encoding <> aggregate.tokenizer_encoding OR
             event.reply_priming_tokens <> aggregate.reply_priming_tokens OR
             event.framing_safety_tokens <> aggregate.framing_safety_tokens OR
             event.input_usd_per_1m <> aggregate.input_usd_per_1m OR
             event.output_usd_per_1m <> aggregate.output_usd_per_1m OR
             event.new_breaker_open <> aggregate.breaker_open
          THEN RAISE EXCEPTION 'provider initial authorization event required';
          END IF;

          IF TG_OP = 'INSERT' OR
             (OLD.last_event_sequence = 0 AND
              aggregate.last_event_sequence = 1 AND
              aggregate.state_version = OLD.state_version)
          THEN
            IF event.event_kind <> 'initial_authorized' OR
               event.prior_state_version <> 0 OR
               event.new_state_version <> 1 OR
               aggregate.state_version <> 1 OR
               aggregate.overrun_count <> 0 OR
               aggregate.last_overrun_cost_usd IS NOT NULL OR
               aggregate.last_overrun_at IS NOT NULL OR
               aggregate.breaker_open OR
               aggregate.breaker_reason_code IS NOT NULL OR
               event.reason_code IS DISTINCT FROM
                 aggregate.breaker_reason_code OR
               event.regression_gate_reference IS DISTINCT FROM
                 aggregate.regression_gate_reference OR
               aggregate.authorized_at IS DISTINCT FROM event.created_at OR
               aggregate.cleared_at IS NOT NULL
            THEN RAISE EXCEPTION 'provider initial authorization event required';
            END IF;
            RETURN NEW;
          END IF;

          protected_changed :=
            aggregate.purpose IS DISTINCT FROM OLD.purpose OR
            aggregate.provider IS DISTINCT FROM OLD.provider OR
            aggregate.model IS DISTINCT FROM OLD.model OR
            aggregate.reasoning_effort IS DISTINCT FROM OLD.reasoning_effort OR
            aggregate.state_version IS DISTINCT FROM OLD.state_version OR
            aggregate.authorized_cost_policy_version IS DISTINCT FROM
              OLD.authorized_cost_policy_version OR
            aggregate.token_estimator_version IS DISTINCT FROM
              OLD.token_estimator_version OR
            aggregate.tokenizer_encoding IS DISTINCT FROM OLD.tokenizer_encoding OR
            aggregate.reply_priming_tokens IS DISTINCT FROM
              OLD.reply_priming_tokens OR
            aggregate.framing_safety_tokens IS DISTINCT FROM
              OLD.framing_safety_tokens OR
            aggregate.input_usd_per_1m IS DISTINCT FROM OLD.input_usd_per_1m OR
            aggregate.output_usd_per_1m IS DISTINCT FROM OLD.output_usd_per_1m OR
            aggregate.breaker_open IS DISTINCT FROM OLD.breaker_open OR
            aggregate.breaker_reason_code IS DISTINCT FROM
              OLD.breaker_reason_code OR
            aggregate.overrun_count IS DISTINCT FROM OLD.overrun_count OR
            aggregate.last_overrun_cost_usd IS DISTINCT FROM
              OLD.last_overrun_cost_usd OR
            aggregate.last_overrun_at IS DISTINCT FROM OLD.last_overrun_at OR
            aggregate.regression_gate_reference IS DISTINCT FROM
              OLD.regression_gate_reference OR
            aggregate.authorized_at IS DISTINCT FROM OLD.authorized_at OR
            aggregate.cleared_at IS DISTINCT FROM OLD.cleared_at OR
            aggregate.last_event_sequence IS DISTINCT FROM
              OLD.last_event_sequence OR
            aggregate.last_event_id IS DISTINCT FROM OLD.last_event_id;

          IF protected_changed THEN
            IF aggregate.last_event_sequence <> OLD.last_event_sequence + 1 OR
               aggregate.last_event_id IS NOT DISTINCT FROM OLD.last_event_id OR
               aggregate.state_version <> OLD.state_version + 1 OR
               event.prior_state_version <> OLD.state_version OR
               event.new_state_version <> aggregate.state_version OR
               event.prior_breaker_open <> OLD.breaker_open OR
               event.reason_code IS DISTINCT FROM
                 aggregate.breaker_reason_code OR
               event.regression_gate_reference IS DISTINCT FROM
                 aggregate.regression_gate_reference
            THEN RAISE EXCEPTION 'provider safety event required'; END IF;

            IF event.event_kind = 'budget_overrun' THEN
              IF NOT aggregate.breaker_open OR
                 aggregate.overrun_count <> OLD.overrun_count + 1 OR
                 aggregate.last_overrun_cost_usd IS NULL OR
                 aggregate.last_overrun_at IS DISTINCT FROM event.created_at OR
                 aggregate.authorized_at IS DISTINCT FROM OLD.authorized_at OR
                 aggregate.cleared_at IS DISTINCT FROM OLD.cleared_at
              THEN RAISE EXCEPTION 'provider safety event required'; END IF;
            ELSIF event.event_kind = 'breaker_cleared' THEN
              IF aggregate.breaker_open OR
                 aggregate.overrun_count <> OLD.overrun_count OR
                 aggregate.last_overrun_cost_usd IS DISTINCT FROM
                   OLD.last_overrun_cost_usd OR
                 aggregate.last_overrun_at IS DISTINCT FROM OLD.last_overrun_at OR
                 aggregate.authorized_at IS DISTINCT FROM OLD.authorized_at OR
                 aggregate.cleared_at IS DISTINCT FROM event.created_at
              THEN RAISE EXCEPTION 'provider safety event required'; END IF;
            ELSE
              RAISE EXCEPTION 'provider safety event required';
            END IF;
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE CONSTRAINT TRIGGER trg_provider_safety_state_event_guard
        AFTER INSERT OR UPDATE ON auto_review_provider_safety_states
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
        EXECUTE FUNCTION enforce_provider_safety_state_event()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_rollout_control_event_sequence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE parent_sequence integer;
        DECLARE parent_event_id integer;
        DECLARE parent_state_version integer;
        DECLARE parent_control_epoch integer;
        DECLARE prior_sequence integer;
        DECLARE prior_event auto_review_rollout_control_events%ROWTYPE;
        BEGIN
          SELECT last_event_sequence, last_event_id, state_version, control_epoch
            INTO parent_sequence, parent_event_id, parent_state_version,
                 parent_control_epoch
          FROM auto_review_rollout_states WHERE id = NEW.rollout_state_id;
          SELECT COALESCE(MAX(event_sequence), 0) INTO prior_sequence
          FROM auto_review_rollout_control_events
          WHERE rollout_state_id = NEW.rollout_state_id AND id <> NEW.id;
          IF NEW.event_sequence = 1 THEN
            IF NEW.prior_state_version < 0 OR NEW.prior_control_epoch <> 0 OR
               NEW.prior_max_authorized_percentage <> 0 OR
               NEW.prior_breaker_open <> false OR
               NEW.prior_authorization_generation <> 0
            THEN RAISE EXCEPTION 'prior rollout snapshot mismatch'; END IF;
          ELSE
            SELECT * INTO prior_event FROM auto_review_rollout_control_events
            WHERE rollout_state_id = NEW.rollout_state_id
              AND event_sequence = NEW.event_sequence - 1;
            IF NOT FOUND THEN
              RAISE EXCEPTION 'rollout events require gapless atomic backpointer';
            END IF;
            IF prior_event.new_state_version > NEW.prior_state_version OR
               prior_event.new_control_epoch <> NEW.prior_control_epoch OR
               prior_event.new_max_authorized_percentage <>
                 NEW.prior_max_authorized_percentage OR
               prior_event.new_breaker_open <> NEW.prior_breaker_open OR
               prior_event.new_authorization_generation <>
                 NEW.prior_authorization_generation
            THEN RAISE EXCEPTION 'prior rollout snapshot mismatch'; END IF;
          END IF;
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
        CREATE OR REPLACE FUNCTION enforce_rollout_state_control_event()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE metric_changed boolean;
        DECLARE control_changed boolean;
        BEGIN
          IF NEW.security_scope_id IS DISTINCT FROM OLD.security_scope_id OR
             NEW.policy_version IS DISTINCT FROM OLD.policy_version
          THEN RAISE EXCEPTION 'rollout identity is immutable'; END IF;

          IF NEW.corrected_critical_count < OLD.corrected_critical_count THEN
            RAISE EXCEPTION 'corrected critical count is monotonic';
          END IF;
          IF NEW.shadow_predicted_count < OLD.shadow_predicted_count OR
             NEW.shadow_completed_count < OLD.shadow_completed_count OR
             NEW.shadow_supported_count < OLD.shadow_supported_count OR
             NEW.enforce_promotion_ordinal < OLD.enforce_promotion_ordinal OR
             NEW.post_audit_selected_count < OLD.post_audit_selected_count OR
             NEW.post_audit_completed_count < OLD.post_audit_completed_count OR
             NEW.post_audit_critical_count < OLD.post_audit_critical_count OR
             NEW.confirmed_mandatory_audit_count <
               OLD.confirmed_mandatory_audit_count OR
             NEW.invalidated_before_audit_count <
               OLD.invalidated_before_audit_count OR
             NEW.authorization_generation < OLD.authorization_generation
          THEN RAISE EXCEPTION 'rollout cumulative counter is monotonic'; END IF;

          metric_changed :=
            NEW.shadow_predicted_count IS DISTINCT FROM
              OLD.shadow_predicted_count OR
            NEW.shadow_completed_count IS DISTINCT FROM
              OLD.shadow_completed_count OR
            NEW.shadow_supported_count IS DISTINCT FROM
              OLD.shadow_supported_count OR
            NEW.enforce_promotion_ordinal IS DISTINCT FROM
              OLD.enforce_promotion_ordinal OR
            NEW.post_audit_selected_count IS DISTINCT FROM
              OLD.post_audit_selected_count OR
            NEW.post_audit_completed_count IS DISTINCT FROM
              OLD.post_audit_completed_count OR
            NEW.post_audit_critical_count IS DISTINCT FROM
              OLD.post_audit_critical_count OR
            NEW.confirmed_mandatory_audit_count IS DISTINCT FROM
              OLD.confirmed_mandatory_audit_count OR
            NEW.pending_mandatory_audit_count IS DISTINCT FROM
              OLD.pending_mandatory_audit_count OR
            NEW.invalidated_before_audit_count IS DISTINCT FROM
              OLD.invalidated_before_audit_count OR
            NEW.corrected_critical_count IS DISTINCT FROM
              OLD.corrected_critical_count;
          control_changed :=
            NEW.control_epoch IS DISTINCT FROM OLD.control_epoch OR
            NEW.max_authorized_percentage IS DISTINCT FROM
               OLD.max_authorized_percentage OR
            NEW.authorization_generation IS DISTINCT FROM
               OLD.authorization_generation OR
            NEW.authorization_at IS DISTINCT FROM OLD.authorization_at OR
            NEW.breaker_open IS DISTINCT FROM OLD.breaker_open OR
            NEW.breaker_reason_code IS DISTINCT FROM OLD.breaker_reason_code OR
            NEW.breaker_opened_at IS DISTINCT FROM OLD.breaker_opened_at OR
            NEW.regression_gate_reference IS DISTINCT FROM
              OLD.regression_gate_reference;

          IF metric_changed OR control_changed THEN
            IF NEW.state_version <> OLD.state_version + 1
            THEN RAISE EXCEPTION 'rollout metric state version required'; END IF;
          ELSIF NEW.state_version IS DISTINCT FROM OLD.state_version THEN
            RAISE EXCEPTION 'rollout state version requires mutation';
          END IF;

          IF control_changed THEN
            IF EXISTS (
              SELECT 1 FROM auto_review_rollout_control_events event
              WHERE event.id = NEW.last_event_id
                AND event.rollout_state_id = NEW.id
            ) AND NOT EXISTS (
              SELECT 1 FROM auto_review_rollout_control_events event
              WHERE event.id = NEW.last_event_id
                AND event.rollout_state_id = NEW.id
                AND event.prior_state_version = OLD.state_version
                AND event.prior_control_epoch = OLD.control_epoch
                AND event.prior_max_authorized_percentage =
                  OLD.max_authorized_percentage
                AND event.prior_authorization_generation =
                  OLD.authorization_generation
                AND event.prior_breaker_open = OLD.breaker_open
            ) THEN RAISE EXCEPTION 'prior rollout snapshot mismatch'; END IF;
            IF NEW.control_epoch <> OLD.control_epoch + 1 OR
               NEW.last_event_sequence <> OLD.last_event_sequence + 1 OR
               NEW.last_event_id IS NULL OR
               NEW.last_event_id IS NOT DISTINCT FROM OLD.last_event_id OR NOT EXISTS (
                 SELECT 1 FROM auto_review_rollout_control_events event
                 WHERE event.id = NEW.last_event_id
                   AND event.rollout_state_id = NEW.id
                   AND event.event_sequence = NEW.last_event_sequence
                   AND event.prior_state_version = OLD.state_version
                   AND event.new_state_version = NEW.state_version
                   AND event.prior_control_epoch = OLD.control_epoch
                   AND event.new_control_epoch = NEW.control_epoch
                   AND event.prior_max_authorized_percentage =
                     OLD.max_authorized_percentage
                   AND event.new_max_authorized_percentage =
                     NEW.max_authorized_percentage
                   AND event.prior_authorization_generation =
                     OLD.authorization_generation
                   AND event.new_authorization_generation =
                     NEW.authorization_generation
                   AND event.prior_breaker_open = OLD.breaker_open
                   AND event.new_breaker_open = NEW.breaker_open
                   AND event.reason_code IS NOT DISTINCT FROM
                     NEW.breaker_reason_code
                   AND event.regression_gate_reference IS NOT DISTINCT FROM
                     NEW.regression_gate_reference
               )
            THEN RAISE EXCEPTION 'rollout control event required'; END IF;
          ELSIF NEW.control_epoch IS DISTINCT FROM OLD.control_epoch OR
                NEW.last_event_sequence IS DISTINCT FROM
                  OLD.last_event_sequence OR
                NEW.last_event_id IS DISTINCT FROM OLD.last_event_id
          THEN RAISE EXCEPTION 'rollout control event required';
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_rollout_state_control_event_guard
        BEFORE UPDATE ON auto_review_rollout_states FOR EACH ROW
        EXECUTE FUNCTION enforce_rollout_state_control_event()
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
        CREATE TRIGGER trg_trusted_evidence_link_immutable
        BEFORE UPDATE OR DELETE ON trusted_knowledge_evidence_links
        FOR EACH ROW EXECUTE FUNCTION reject_c5_immutable_mutation()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_trusted_approval_link_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' OR
             NEW.knowledge_type IS DISTINCT FROM OLD.knowledge_type OR
             NEW.knowledge_id IS DISTINCT FROM OLD.knowledge_id OR
             NEW.review_item_id IS DISTINCT FROM OLD.review_item_id OR
             NEW.security_scope_id IS DISTINCT FROM OLD.security_scope_id OR
             NEW.promotion_effect_kind IS DISTINCT FROM OLD.promotion_effect_kind OR
             NEW.resolution_source IS DISTINCT FROM OLD.resolution_source OR
             NEW.claim_fingerprint IS DISTINCT FROM OLD.claim_fingerprint OR
             NEW.permission_level IS DISTINCT FROM OLD.permission_level OR
             NEW.fingerprint_key_version IS DISTINCT FROM
               OLD.fingerprint_key_version OR
             NEW.fingerprint_key_material_verifier IS DISTINCT FROM
               OLD.fingerprint_key_material_verifier OR
             NEW.created_at IS DISTINCT FROM OLD.created_at OR
             NOT ((NEW.active = OLD.active AND
                   NEW.revoked_at IS NOT DISTINCT FROM OLD.revoked_at) OR
                  (OLD.active = true AND OLD.revoked_at IS NULL AND
                   NEW.active = false AND NEW.revoked_at IS NOT NULL))
          THEN RAISE EXCEPTION 'trusted provenance is immutable'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_trusted_approval_link_mutation_guard
        BEFORE UPDATE OR DELETE ON trusted_knowledge_approval_links FOR EACH ROW
        EXECUTE FUNCTION enforce_trusted_approval_link_mutation()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_review_item_revoke_snapshot()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'UPDATE' AND OLD.revoked_at IS NOT NULL THEN
            IF NEW.status IS DISTINCT FROM OLD.status OR
               NEW.revoked_at IS DISTINCT FROM OLD.revoked_at OR
               NEW.revoked_by_subject_hmac IS DISTINCT FROM
                 OLD.revoked_by_subject_hmac OR
               NEW.revoked_by_fingerprint_key_version IS DISTINCT FROM
                 OLD.revoked_by_fingerprint_key_version OR
               NEW.revoked_by_fingerprint_key_material_verifier IS DISTINCT FROM
                 OLD.revoked_by_fingerprint_key_material_verifier OR
               NEW.revoke_knowledge_remained_trusted IS DISTINCT FROM
                 OLD.revoke_knowledge_remained_trusted OR
               NEW.revoke_document_count IS DISTINCT FROM OLD.revoke_document_count
            THEN RAISE EXCEPTION 'revoke snapshot is immutable'; END IF;
          ELSIF NEW.status = 'revoked' OR NEW.revoked_at IS NOT NULL OR
                NEW.revoked_by_subject_hmac IS NOT NULL OR
                NEW.revoked_by_fingerprint_key_version IS NOT NULL OR
                NEW.revoked_by_fingerprint_key_material_verifier IS NOT NULL OR
                NEW.revoke_knowledge_remained_trusted IS NOT NULL OR
                NEW.revoke_document_count IS NOT NULL
          THEN
            IF NEW.status <> 'revoked' OR NEW.revoked_at IS NULL OR
               length(NEW.revoked_by_subject_hmac) <> 64 OR
               NEW.revoked_by_fingerprint_key_version IS NULL OR
               length(NEW.revoked_by_fingerprint_key_material_verifier) <> 64 OR
               NEW.revoke_knowledge_remained_trusted IS NULL OR
               NEW.revoke_document_count IS NULL OR NEW.revoke_document_count < 0
            THEN RAISE EXCEPTION 'complete revoke snapshot required'; END IF;
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_review_item_revoke_snapshot_guard
        BEFORE INSERT OR UPDATE ON review_items FOR EACH ROW
        EXECUTE FUNCTION enforce_review_item_revoke_snapshot()
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
        CREATE OR REPLACE FUNCTION enforce_auto_review_post_audit_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' OR OLD.status <> 'pending' OR
             NEW.review_item_id IS DISTINCT FROM OLD.review_item_id OR
             NEW.promotion_decision_id IS DISTINCT FROM OLD.promotion_decision_id OR
             NEW.sample_cohort IS DISTINCT FROM OLD.sample_cohort OR
             NEW.created_at IS DISTINCT FROM OLD.created_at
          THEN RAISE EXCEPTION 'terminal audit is immutable'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_auto_review_post_audit_mutation_guard
        BEFORE UPDATE OR DELETE ON auto_review_post_audits FOR EACH ROW
        EXECUTE FUNCTION enforce_auto_review_post_audit_mutation()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_correction_confirmed_parent()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM auto_review_post_audits audit
            WHERE audit.id = NEW.post_audit_id
              AND audit.review_item_id = NEW.review_item_id
              AND audit.status = 'completed' AND audit.outcome = 'confirmed'
          ) THEN RAISE EXCEPTION 'correction requires confirmed parent audit';
          END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_auto_review_correction_confirmed_parent
        BEFORE INSERT ON auto_review_audit_corrections FOR EACH ROW
        EXECUTE FUNCTION enforce_correction_confirmed_parent()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_c5_parser_identity_immutable()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'UPDATE' AND (
             NEW.document_id IS DISTINCT FROM OLD.document_id OR
             NEW.document_version_id IS DISTINCT FROM OLD.document_version_id OR
             NEW.source_id IS DISTINCT FROM OLD.source_id OR
             NEW.parser_name IS DISTINCT FROM OLD.parser_name OR
             NEW.content_signature IS DISTINCT FROM OLD.content_signature OR
             NEW.server_content_signature_schema IS DISTINCT FROM
               OLD.server_content_signature_schema OR
             NEW.server_content_signature IS DISTINCT FROM OLD.server_content_signature OR
             NEW.parser_policy_version IS DISTINCT FROM OLD.parser_policy_version OR
             NEW.parser_version IS DISTINCT FROM OLD.parser_version OR
             NEW.chunk_policy_version IS DISTINCT FROM OLD.chunk_policy_version)
          THEN RAISE EXCEPTION 'C.5 parser identity is immutable'; END IF;
          IF NEW.server_content_signature IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM document_versions version
            JOIN documents document ON document.id = version.document_id
            JOIN sources source ON source.id = NEW.source_id
            WHERE version.id = NEW.document_version_id
              AND document.id = NEW.document_id
              AND document.source_id = NEW.source_id
              AND source.server_content_signature_schema =
                NEW.server_content_signature_schema
              AND source.server_content_signature = NEW.server_content_signature
          ) THEN RAISE EXCEPTION 'parser authority mismatch'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_document_parser_run_c5_identity_immutable
        BEFORE INSERT OR UPDATE ON document_parser_runs FOR EACH ROW
        EXECUTE FUNCTION enforce_c5_parser_identity_immutable()
        """,
        """
        CREATE OR REPLACE FUNCTION enforce_c5_chunk_lineage_immutable()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.parser_run_id IS DISTINCT FROM OLD.parser_run_id OR
             NEW.version_id IS DISTINCT FROM OLD.version_id OR
             NEW.source_id IS DISTINCT FROM OLD.source_id OR
             NEW.chunk_index IS DISTINCT FROM OLD.chunk_index
          THEN RAISE EXCEPTION 'C.5 chunk lineage is immutable'; END IF;
          RETURN NEW;
        END $$
        """,
        """
        CREATE TRIGGER trg_document_chunk_c5_lineage_immutable
        BEFORE UPDATE ON document_chunks FOR EACH ROW
        EXECUTE FUNCTION enforce_c5_chunk_lineage_immutable()
        """,
    )


def _drop_postgresql_guards() -> None:
    if op.get_bind().dialect.name != 'postgresql':
        return
    for table_name, trigger_name in (
        ('document_chunks', 'trg_document_chunk_c5_lineage_immutable'),
        ('document_parser_runs', 'trg_document_parser_run_c5_identity_immutable'),
        ('auto_review_audit_corrections', 'trg_auto_review_correction_confirmed_parent'),
        ('auto_review_audit_corrections', 'trg_auto_review_audit_correction_monotonic'),
        ('auto_review_post_audits', 'trg_auto_review_post_audit_mutation_guard'),
        ('review_items', 'trg_review_item_revoke_snapshot_guard'),
        ('trusted_knowledge_approval_links', 'trg_trusted_approval_link_mutation_guard'),
        ('trusted_knowledge_evidence_links', 'trg_trusted_evidence_link_immutable'),
        ('assistant_messages', 'trg_assistant_message_committed_immutable'),
        ('assistant_message_knowledge_evidence_refs', 'trg_assistant_dependency_child_exactness_guard'),
        ('assistant_message_evidence_dependencies', 'trg_assistant_dependency_exactness_guard'),
        ('assistant_message_knowledge_evidence_refs', 'trg_assistant_dependency_child_immutable'),
        ('assistant_message_evidence_dependencies', 'trg_assistant_dependency_immutable'),
        ('auto_review_validations', 'trg_validation_child_mutation_guard'),
        ('auto_review_validations', 'trg_validation_child_terminal_parent_guard'),
        ('auto_review_validation_calls', 'trg_validation_call_terminal_children_guard'),
        ('auto_review_revocation_assessments', 'trg_revocation_assessment_immutable'),
        ('auto_review_promotion_decisions', 'trg_promotion_decision_immutable'),
        ('auto_review_rollout_control_events', 'trg_rollout_control_event_append_only'),
        ('auto_review_provider_safety_events', 'trg_provider_safety_event_append_only'),
        ('auto_review_rollout_control_events', 'trg_rollout_control_event_sequence_guard'),
        ('auto_review_rollout_states', 'trg_rollout_state_control_event_guard'),
        ('auto_review_provider_safety_events', 'trg_provider_safety_event_sequence_guard'),
        ('auto_review_provider_safety_states', 'trg_provider_safety_state_event_guard'),
        ('assistant_message_evidence_dependencies', 'trg_assistant_dependency_child_count_guard'),
        ('assistant_messages', 'trg_assistant_message_evidence_guard'),
        ('agent_runtime_schema_versions', 'trg_auto_review_schema_boundary_immutable'),
        ('review_items', 'trg_review_item_c5_identity_immutable'),
        ('review_item_evidence_refs', 'trg_review_item_evidence_child_guard'),
        ('review_items', 'trg_review_item_post_c5_evidence_guard'),
    ):
        op.execute(sa.text(f'DROP TRIGGER IF EXISTS {trigger_name} ON {table_name}'))
    for function_name in (
        'enforce_c5_chunk_lineage_immutable',
        'enforce_c5_parser_identity_immutable',
        'enforce_correction_confirmed_parent',
        'enforce_auto_review_post_audit_mutation',
        'enforce_audit_correction_monotonic',
        'enforce_review_item_revoke_snapshot',
        'enforce_trusted_approval_link_mutation',
        'enforce_assistant_message_immutable',
        'reject_c5_immutable_mutation',
        'enforce_rollout_state_control_event',
        'enforce_rollout_control_event_sequence',
        'enforce_provider_safety_state_event',
        'enforce_provider_safety_event_sequence',
        'enforce_assistant_dependency_exactness',
        'enforce_validation_child_mutation',
        'enforce_validation_call_terminal_children',
        'enforce_assistant_dependency_child_count',
        'enforce_assistant_message_evidence',
        'protect_auto_review_schema_boundary',
        'enforce_review_item_c5_identity_immutable',
        'enforce_review_item_evidence_child_guard',
        'enforce_review_item_post_c5_evidence',
    ):
        op.execute(sa.text(f'DROP FUNCTION IF EXISTS {function_name}()'))


def _refuse_retained_c5_state() -> None:
    if has_retained_c5_state(
        op.get_bind(), require_complete_schema=False
    ):
        raise RuntimeError('retained C.5 state; schema downgrade refused')


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
