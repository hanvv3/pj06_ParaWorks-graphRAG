"""Harden published legacy-v3 ownership and PostgreSQL staging provenance."""

import sqlalchemy as sa
from alembic import op

from backend.migrations.versions import (
    b5e6f7a8b9c0_add_legacy_v3_integrity as predecessor,
)

revision = 'c6f7a8b9c0d1'
down_revision = 'b5e6f7a8b9c0'
branch_labels = None
depends_on = None

# The predecessor is an applied, frozen migration. Reusing its frozen constants
# here avoids importing mutable application code into the migration chain.
NEW_CHECKS = predecessor.NEW_CHECKS
SQLITE_QUERY = predecessor.SQLITE_QUERY


PG_STAGING = """CREATE OR REPLACE FUNCTION enforce_assistant_legacy_v3_staging()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $$
DECLARE mutable_columns text[] := ARRAY[
 'dependency_set_hmac', 'dependency_serving_scope', 'dependency_role',
 'dependency_child_hmac', 'legacy_dependency_identity_hmac', 'model_content_hmac',
 'canonical_citation_projection_hmac', 'selected_v1_citation_projection_hmac',
 'approval_provenance_hmac', 'evidence_link_set_hmac'];
BEGIN
 IF TG_OP='UPDATE' AND OLD.dependency_serving_scope IS NULL
    AND NEW.dependency_serving_scope='legacy_v1_only'
    AND (to_jsonb(NEW)-mutable_columns)=(to_jsonb(OLD)-mutable_columns)
    AND EXISTS (
      SELECT 1
      FROM assistant_legacy_v3_parent_staging parent_stage
      JOIN assistant_legacy_v3_dependency_staging child_stage
        ON child_stage.parent_id=parent_stage.parent_id
       AND child_stage.transaction_id=parent_stage.transaction_id
      WHERE parent_stage.parent_id=OLD.assistant_message_id
        AND child_stage.parent_id=OLD.assistant_message_id
        AND child_stage.dependency_id=OLD.id
        AND parent_stage.transaction_id=txid_current()
        AND child_stage.transaction_id=txid_current())
 THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'C.5 audit/evidence row is append-only';
END $$;"""


PG_REGISTRATION = """
CREATE TABLE assistant_legacy_v3_parent_staging (
  parent_id bigint PRIMARY KEY REFERENCES assistant_messages(id) ON DELETE CASCADE,
  transaction_id bigint NOT NULL CHECK (transaction_id > 0)
);
CREATE TABLE assistant_legacy_v3_dependency_staging (
  dependency_id bigint PRIMARY KEY
    REFERENCES assistant_message_evidence_dependencies(id) ON DELETE CASCADE,
  parent_id bigint NOT NULL
    REFERENCES assistant_legacy_v3_parent_staging(parent_id) ON DELETE CASCADE,
  transaction_id bigint NOT NULL CHECK (transaction_id > 0),
  UNIQUE (parent_id, dependency_id),
  UNIQUE (parent_id, dependency_id, transaction_id)
);
ALTER TABLE assistant_legacy_v3_parent_staging ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_legacy_v3_parent_staging FORCE ROW LEVEL SECURITY;
ALTER TABLE assistant_legacy_v3_dependency_staging ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_legacy_v3_dependency_staging FORCE ROW LEVEL SECURITY;
CREATE POLICY legacy_v3_parent_trigger_only
  ON assistant_legacy_v3_parent_staging
  USING (pg_trigger_depth() > 0) WITH CHECK (pg_trigger_depth() > 0);
CREATE POLICY legacy_v3_dependency_trigger_only
  ON assistant_legacy_v3_dependency_staging
  USING (pg_trigger_depth() > 0) WITH CHECK (pg_trigger_depth() > 0);
REVOKE ALL ON assistant_legacy_v3_parent_staging FROM PUBLIC;
REVOKE ALL ON assistant_legacy_v3_dependency_staging FROM PUBLIC;

CREATE FUNCTION register_assistant_legacy_v3_parent_insert()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $$
BEGIN
  IF NEW.role='assistant'
     AND NEW.content_write_mode IS NULL
     AND NEW.evidence_contract_version='assistant-evidence:v1'
     AND NEW.serving_dependency_count > 0
     AND NEW.linked_agent_run_id IS NULL THEN
    INSERT INTO assistant_legacy_v3_parent_staging(parent_id, transaction_id)
    VALUES (NEW.id, txid_current());
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER legacy_v3_register_parent
AFTER INSERT ON assistant_messages
FOR EACH ROW EXECUTE FUNCTION register_assistant_legacy_v3_parent_insert();

CREATE FUNCTION register_assistant_legacy_v3_dependency_insert()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $$
BEGIN
  IF NEW.dependency_serving_scope IS NULL THEN
    INSERT INTO assistant_legacy_v3_dependency_staging(
      dependency_id, parent_id, transaction_id)
    SELECT NEW.id, NEW.assistant_message_id, parent_stage.transaction_id
    FROM assistant_legacy_v3_parent_staging parent_stage
    WHERE parent_stage.parent_id=NEW.assistant_message_id
      AND parent_stage.transaction_id=txid_current();
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER legacy_v3_register_dependency
AFTER INSERT ON assistant_message_evidence_dependencies
FOR EACH ROW EXECUTE FUNCTION register_assistant_legacy_v3_dependency_insert();

CREATE FUNCTION require_consumed_assistant_legacy_v3_staging()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $$
BEGIN
  IF TG_TABLE_NAME='assistant_legacy_v3_parent_staging' AND EXISTS (
    SELECT 1 FROM assistant_legacy_v3_parent_staging staged
    WHERE staged.parent_id=NEW.parent_id
      AND staged.transaction_id=NEW.transaction_id) THEN
    RAISE EXCEPTION 'legacy v3 parent staging must publish before commit';
  ELSIF TG_TABLE_NAME='assistant_legacy_v3_dependency_staging' AND EXISTS (
    SELECT 1 FROM assistant_legacy_v3_dependency_staging staged
    WHERE staged.dependency_id=NEW.dependency_id
      AND staged.parent_id=NEW.parent_id
      AND staged.transaction_id=NEW.transaction_id) THEN
    RAISE EXCEPTION 'legacy v3 dependency staging must publish before commit';
  END IF;
  RETURN NEW;
END $$;
CREATE CONSTRAINT TRIGGER legacy_v3_parent_staging_consumed
AFTER INSERT ON assistant_legacy_v3_parent_staging
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION require_consumed_assistant_legacy_v3_staging();
CREATE CONSTRAINT TRIGGER legacy_v3_dependency_staging_consumed
AFTER INSERT ON assistant_legacy_v3_dependency_staging
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION require_consumed_assistant_legacy_v3_staging();

CREATE FUNCTION consume_assistant_legacy_v3_parent_staging()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $$
DECLARE expected_children integer;
BEGIN
  IF OLD.dependency_set_hmac_schema_version IS DISTINCT FROM
       'assistant-dependency-set-hmac:v3'
     AND NEW.dependency_set_hmac_schema_version=
       'assistant-dependency-set-hmac:v3' THEN
    SELECT count(*) INTO expected_children
    FROM assistant_legacy_v3_dependency_staging child_stage
    WHERE child_stage.parent_id=NEW.id
      AND child_stage.transaction_id=txid_current();
    IF NOT EXISTS (
      SELECT 1 FROM assistant_legacy_v3_parent_staging parent_stage
      WHERE parent_stage.parent_id=NEW.id
        AND parent_stage.transaction_id=txid_current())
      OR expected_children IS DISTINCT FROM NEW.serving_dependency_count THEN
      RAISE EXCEPTION 'legacy v3 publication requires exact INSERT registrations';
    END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER legacy_v3_require_parent_staging
BEFORE UPDATE ON assistant_messages
FOR EACH ROW EXECUTE FUNCTION consume_assistant_legacy_v3_parent_staging();

CREATE FUNCTION clear_assistant_legacy_v3_parent_staging()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $$
BEGIN
  IF OLD.dependency_set_hmac_schema_version IS DISTINCT FROM
       'assistant-dependency-set-hmac:v3'
     AND NEW.dependency_set_hmac_schema_version=
       'assistant-dependency-set-hmac:v3' THEN
    DELETE FROM assistant_legacy_v3_parent_staging
    WHERE parent_id=NEW.id AND transaction_id=txid_current();
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER legacy_v3_clear_parent_staging
AFTER UPDATE ON assistant_messages
FOR EACH ROW EXECUTE FUNCTION clear_assistant_legacy_v3_parent_staging();
"""


def _sqlite_statements():
    tables = {
        'assistant_messages': ('NEW.id', 'OLD.id'),
        'assistant_message_evidence_dependencies': (
            'NEW.assistant_message_id',
            'OLD.assistant_message_id',
        ),
        'assistant_message_knowledge_evidence_refs': (
            'NEW.assistant_message_id',
            'OLD.assistant_message_id',
        ),
    }
    for table, (new, old) in tables.items():
        for action in ('INSERT', 'UPDATE', 'DELETE'):
            targets = (
                [new]
                if action == 'INSERT'
                else [old]
                if action == 'DELETE'
                else [old, new]
            )
            conditions = ' OR '.join(
                f'EXISTS ({SQLITE_QUERY.format(target=target)})'
                for target in targets
            )
            if table == 'assistant_messages' and action == 'DELETE':
                conditions += " OR (OLD.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND (EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=OLD.id) OR EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r WHERE r.assistant_message_id=OLD.id)))"
            if table == 'assistant_messages' and action == 'UPDATE':
                conditions += " OR (OLD.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND NEW.dependency_set_hmac_schema_version IS DISTINCT FROM 'assistant-dependency-set-hmac:v3')"
            if (
                table == 'assistant_message_knowledge_evidence_refs'
                and action in {'INSERT', 'UPDATE'}
            ):
                conditions += " OR (((EXISTS (SELECT 1 FROM assistant_messages claimed WHERE claimed.id=NEW.assistant_message_id AND claimed.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3')) OR (EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies actual JOIN assistant_messages owner ON owner.id=actual.assistant_message_id WHERE actual.id=NEW.dependency_id AND owner.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3'))) AND NOT EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies actual JOIN assistant_messages owner ON owner.id=actual.assistant_message_id JOIN trusted_knowledge_evidence_links evidence ON evidence.id=NEW.trusted_knowledge_evidence_link_id WHERE actual.id=NEW.dependency_id AND actual.assistant_message_id=NEW.assistant_message_id AND owner.id=NEW.assistant_message_id AND owner.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND actual.approval_link_id IS NOT NULL AND actual.approval_link_id=NEW.approval_link_id AND evidence.approval_link_id=actual.approval_link_id))"
            yield f"CREATE TRIGGER IF NOT EXISTS legacy_v3_{table}_{action.lower()} AFTER {action} ON {table} BEGIN SELECT CASE WHEN {conditions} THEN RAISE(ABORT,'legacy v3 published integrity mismatch') END; END"


def _drop_sqlite_guards():
    for table in (
        'assistant_messages',
        'assistant_message_evidence_dependencies',
        'assistant_message_knowledge_evidence_refs',
    ):
        for action in ('insert', 'update', 'delete'):
            op.execute(sa.text(f'DROP TRIGGER IF EXISTS legacy_v3_{table}_{action}'))


def _drop_postgres_registration():
    for trigger, table in (
        ('legacy_v3_clear_parent_staging', 'assistant_messages'),
        ('legacy_v3_require_parent_staging', 'assistant_messages'),
        ('legacy_v3_register_dependency', 'assistant_message_evidence_dependencies'),
        ('legacy_v3_register_parent', 'assistant_messages'),
    ):
        op.execute(sa.text(f'DROP TRIGGER IF EXISTS {trigger} ON {table}'))
    op.execute(
        sa.text(
            'DROP FUNCTION IF EXISTS clear_assistant_legacy_v3_parent_staging()'
        )
    )
    op.execute(
        sa.text(
            'DROP FUNCTION IF EXISTS consume_assistant_legacy_v3_parent_staging()'
        )
    )
    op.execute(
        sa.text(
            'DROP FUNCTION IF EXISTS register_assistant_legacy_v3_dependency_insert()'
        )
    )
    op.execute(
        sa.text('DROP FUNCTION IF EXISTS register_assistant_legacy_v3_parent_insert()')
    )
    op.execute(
        sa.text(
            'DROP FUNCTION IF EXISTS require_consumed_assistant_legacy_v3_staging() CASCADE'
        )
    )
    op.execute(sa.text('DROP TABLE assistant_legacy_v3_dependency_staging'))
    op.execute(sa.text('DROP TABLE assistant_legacy_v3_parent_staging'))


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == 'sqlite':
        _drop_sqlite_guards()
        for statement in _sqlite_statements():
            op.execute(sa.text(statement))
    elif bind.dialect.name == 'postgresql':
        op.execute(sa.text(PG_REGISTRATION))
        op.execute(sa.text(PG_STAGING))


def downgrade():
    bind = op.get_bind()
    if bind.scalar(
        sa.text(
            "SELECT count(*) FROM assistant_messages WHERE dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3'"
        )
    ):
        raise ValueError('cannot downgrade while legacy v3 messages exist')
    if bind.dialect.name == 'sqlite':
        _drop_sqlite_guards()
        for statement in predecessor._sqlite_statements():
            op.execute(sa.text(statement))
    elif bind.dialect.name == 'postgresql':
        _drop_postgres_registration()
        op.execute(sa.text(predecessor.PG_STAGING))
