"""Bind legacy-v3 PostgreSQL staging authority to canonical trigger relations."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from backend.migrations.versions import (
    c6f7a8b9c0d1_harden_legacy_v3_database_guards as predecessor,
)

revision = 'd7a8b9c0d1e2'
down_revision = 'c6f7a8b9c0d1'
branch_labels = None
depends_on = None


FUNCTIONS = (
    'enforce_assistant_legacy_v3_staging',
    'register_assistant_legacy_v3_parent_insert',
    'register_assistant_legacy_v3_dependency_insert',
    'require_consumed_assistant_legacy_v3_parent_staging',
    'require_consumed_assistant_legacy_v3_dependency_staging',
    'consume_assistant_legacy_v3_parent_staging',
    'clear_assistant_legacy_v3_parent_staging',
)

TRIGGERS = (
    (
        'trg_assistant_dependency_immutable',
        'assistant_message_evidence_dependencies',
    ),
    ('legacy_v3_register_parent', 'assistant_messages'),
    (
        'legacy_v3_register_dependency',
        'assistant_message_evidence_dependencies',
    ),
    (
        'legacy_v3_parent_staging_consumed',
        'assistant_legacy_v3_parent_staging',
    ),
    (
        'legacy_v3_dependency_staging_consumed',
        'assistant_legacy_v3_dependency_staging',
    ),
    ('legacy_v3_require_parent_staging', 'assistant_messages'),
    ('legacy_v3_clear_parent_staging', 'assistant_messages'),
)


def _quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _trusted_schema(bind) -> str:
    schema = bind.scalar(sa.text('SELECT pg_catalog.current_schema()'))
    if not isinstance(schema, str) or not schema or '\x00' in schema:
        raise ValueError('trusted PostgreSQL application schema is required')
    if schema in {'pg_catalog', 'information_schema'} or schema.startswith(
        ('pg_temp_', 'pg_toast_temp_')
    ):
        raise ValueError('temporary/system schema cannot own application authority')
    return schema


def _qualified(schema: str, name: str) -> str:
    return f'{_quote_identifier(schema)}.{_quote_identifier(name)}'


def _regclass(schema: str, name: str) -> str:
    qualified = _qualified(schema, name).replace("'", "''")
    return f"'{qualified}'::pg_catalog.regclass"


def _drop_authority_sql(schema: str) -> tuple[str, ...]:
    statements = [
        f'DROP TRIGGER IF EXISTS {_quote_identifier(trigger)} ON '
        f'{_qualified(schema, table)}'
        for trigger, table in TRIGGERS
    ]
    # c6 used one shared deferred function. CASCADE also retires any foreign
    # trigger that was attached while its default PUBLIC grant existed.
    function_names = (*FUNCTIONS, 'require_consumed_assistant_legacy_v3_staging')
    statements.extend(
        f'DROP FUNCTION IF EXISTS {_qualified(schema, function)}() CASCADE'
        for function in function_names
    )
    return tuple(statements)


def _hardened_sql(schema: str) -> str:
    path = f'pg_catalog, {_quote_identifier(schema)}, pg_temp'
    messages = _qualified(schema, 'assistant_messages')
    dependencies = _qualified(
        schema, 'assistant_message_evidence_dependencies'
    )
    parent_staging = _qualified(schema, 'assistant_legacy_v3_parent_staging')
    dependency_staging = _qualified(
        schema, 'assistant_legacy_v3_dependency_staging'
    )
    enforce = _qualified(schema, 'enforce_assistant_legacy_v3_staging')
    register_parent = _qualified(
        schema, 'register_assistant_legacy_v3_parent_insert'
    )
    register_dependency = _qualified(
        schema, 'register_assistant_legacy_v3_dependency_insert'
    )
    require_parent = _qualified(
        schema, 'require_consumed_assistant_legacy_v3_parent_staging'
    )
    require_dependency = _qualified(
        schema, 'require_consumed_assistant_legacy_v3_dependency_staging'
    )
    consume_parent = _qualified(
        schema, 'consume_assistant_legacy_v3_parent_staging'
    )
    clear_parent = _qualified(
        schema, 'clear_assistant_legacy_v3_parent_staging'
    )

    return f"""
CREATE FUNCTION {enforce}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
DECLARE mutable_columns pg_catalog.text[] := ARRAY[
 'dependency_set_hmac', 'dependency_serving_scope', 'dependency_role',
 'dependency_child_hmac', 'legacy_dependency_identity_hmac', 'model_content_hmac',
 'canonical_citation_projection_hmac', 'selected_v1_citation_projection_hmac',
 'approval_provenance_hmac', 'evidence_link_set_hmac'];
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_message_evidence_dependencies')}
    OR TG_OP NOT IN ('UPDATE', 'DELETE') THEN
   RAISE EXCEPTION 'legacy v3 staging called from unauthorized trigger relation';
 END IF;
 IF TG_OP='UPDATE' AND OLD.dependency_serving_scope IS NULL
    AND NEW.dependency_serving_scope='legacy_v1_only'
    AND (pg_catalog.to_jsonb(NEW)-mutable_columns)=
        (pg_catalog.to_jsonb(OLD)-mutable_columns)
    AND EXISTS (
      SELECT 1
      FROM {parent_staging} parent_stage
      JOIN {dependency_staging} child_stage
        ON child_stage.parent_id=parent_stage.parent_id
       AND child_stage.transaction_id=parent_stage.transaction_id
      WHERE parent_stage.parent_id=OLD.assistant_message_id
        AND child_stage.parent_id=OLD.assistant_message_id
        AND child_stage.dependency_id=OLD.id
        AND parent_stage.transaction_id=pg_catalog.txid_current()
        AND child_stage.transaction_id=pg_catalog.txid_current())
 THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'C.5 audit/evidence row is append-only';
END $function$;
REVOKE ALL ON FUNCTION {enforce}() FROM PUBLIC;
CREATE TRIGGER trg_assistant_dependency_immutable
BEFORE UPDATE OR DELETE ON {dependencies}
FOR EACH ROW EXECUTE FUNCTION {enforce}();

CREATE FUNCTION {register_parent}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_messages')}
    OR TG_OP IS DISTINCT FROM 'INSERT' THEN
   RAISE EXCEPTION 'legacy v3 parent registration called from unauthorized trigger relation';
 END IF;
 IF NEW.role='assistant'
    AND NEW.content_write_mode IS NULL
    AND NEW.evidence_contract_version='assistant-evidence:v1'
    AND NEW.serving_dependency_count > 0
    AND NEW.linked_agent_run_id IS NULL THEN
   INSERT INTO {parent_staging}(parent_id, transaction_id)
   VALUES (NEW.id, pg_catalog.txid_current());
 END IF;
 RETURN NEW;
END $function$;
REVOKE ALL ON FUNCTION {register_parent}() FROM PUBLIC;
CREATE TRIGGER legacy_v3_register_parent
AFTER INSERT ON {messages}
FOR EACH ROW EXECUTE FUNCTION {register_parent}();

CREATE FUNCTION {register_dependency}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_message_evidence_dependencies')}
    OR TG_OP IS DISTINCT FROM 'INSERT' THEN
   RAISE EXCEPTION 'legacy v3 dependency registration called from unauthorized trigger relation';
 END IF;
 IF NEW.dependency_serving_scope IS NULL THEN
   INSERT INTO {dependency_staging}(dependency_id, parent_id, transaction_id)
   SELECT NEW.id, NEW.assistant_message_id, parent_stage.transaction_id
   FROM {parent_staging} parent_stage
   WHERE parent_stage.parent_id=NEW.assistant_message_id
     AND parent_stage.transaction_id=pg_catalog.txid_current();
 END IF;
 RETURN NEW;
END $function$;
REVOKE ALL ON FUNCTION {register_dependency}() FROM PUBLIC;
CREATE TRIGGER legacy_v3_register_dependency
AFTER INSERT ON {dependencies}
FOR EACH ROW EXECUTE FUNCTION {register_dependency}();

CREATE FUNCTION {require_parent}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_legacy_v3_parent_staging')}
    OR TG_OP IS DISTINCT FROM 'INSERT' THEN
   RAISE EXCEPTION 'legacy v3 parent consumer called from unauthorized trigger relation';
 END IF;
 IF EXISTS (
   SELECT 1 FROM {parent_staging} staged
   WHERE staged.parent_id=NEW.parent_id
     AND staged.transaction_id=NEW.transaction_id) THEN
   RAISE EXCEPTION 'legacy v3 parent staging must publish before commit';
 END IF;
 RETURN NEW;
END $function$;
REVOKE ALL ON FUNCTION {require_parent}() FROM PUBLIC;
CREATE CONSTRAINT TRIGGER legacy_v3_parent_staging_consumed
AFTER INSERT ON {parent_staging}
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION {require_parent}();

CREATE FUNCTION {require_dependency}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_legacy_v3_dependency_staging')}
    OR TG_OP IS DISTINCT FROM 'INSERT' THEN
   RAISE EXCEPTION 'legacy v3 dependency consumer called from unauthorized trigger relation';
 END IF;
 IF EXISTS (
   SELECT 1 FROM {dependency_staging} staged
   WHERE staged.dependency_id=NEW.dependency_id
     AND staged.parent_id=NEW.parent_id
     AND staged.transaction_id=NEW.transaction_id) THEN
   RAISE EXCEPTION 'legacy v3 dependency staging must publish before commit';
 END IF;
 RETURN NEW;
END $function$;
REVOKE ALL ON FUNCTION {require_dependency}() FROM PUBLIC;
CREATE CONSTRAINT TRIGGER legacy_v3_dependency_staging_consumed
AFTER INSERT ON {dependency_staging}
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION {require_dependency}();

CREATE FUNCTION {consume_parent}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
DECLARE expected_children pg_catalog.int4;
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_messages')}
    OR TG_OP IS DISTINCT FROM 'UPDATE' THEN
   RAISE EXCEPTION 'legacy v3 parent consumer called from unauthorized trigger relation';
 END IF;
 IF OLD.dependency_set_hmac_schema_version IS DISTINCT FROM
      'assistant-dependency-set-hmac:v3'
    AND NEW.dependency_set_hmac_schema_version=
      'assistant-dependency-set-hmac:v3' THEN
   SELECT pg_catalog.count(*) INTO expected_children
   FROM {dependency_staging} child_stage
   WHERE child_stage.parent_id=NEW.id
     AND child_stage.transaction_id=pg_catalog.txid_current();
   IF NOT EXISTS (
     SELECT 1 FROM {parent_staging} parent_stage
     WHERE parent_stage.parent_id=NEW.id
       AND parent_stage.transaction_id=pg_catalog.txid_current())
     OR expected_children IS DISTINCT FROM NEW.serving_dependency_count THEN
     RAISE EXCEPTION 'legacy v3 publication requires exact INSERT registrations';
   END IF;
 END IF;
 RETURN NEW;
END $function$;
REVOKE ALL ON FUNCTION {consume_parent}() FROM PUBLIC;
CREATE TRIGGER legacy_v3_require_parent_staging
BEFORE UPDATE ON {messages}
FOR EACH ROW EXECUTE FUNCTION {consume_parent}();

CREATE FUNCTION {clear_parent}()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER
SET search_path = {path} AS $function$
BEGIN
 IF TG_RELID IS DISTINCT FROM {_regclass(schema, 'assistant_messages')}
    OR TG_OP IS DISTINCT FROM 'UPDATE' THEN
   RAISE EXCEPTION 'legacy v3 parent clearing called from unauthorized trigger relation';
 END IF;
 IF OLD.dependency_set_hmac_schema_version IS DISTINCT FROM
      'assistant-dependency-set-hmac:v3'
    AND NEW.dependency_set_hmac_schema_version=
      'assistant-dependency-set-hmac:v3' THEN
   DELETE FROM {parent_staging}
   WHERE parent_id=NEW.id
     AND transaction_id=pg_catalog.txid_current();
 END IF;
 RETURN NEW;
END $function$;
REVOKE ALL ON FUNCTION {clear_parent}() FROM PUBLIC;
CREATE TRIGGER legacy_v3_clear_parent_staging
AFTER UPDATE ON {messages}
FOR EACH ROW EXECUTE FUNCTION {clear_parent}();
"""


def _drop_authority(schema: str) -> None:
    for statement in _drop_authority_sql(schema):
        op.execute(sa.text(statement))


def _v3_count(bind, schema: str | None = None) -> int:
    table = (
        _qualified(schema, 'assistant_messages')
        if schema is not None
        else 'assistant_messages'
    )
    return int(
        bind.scalar(
            sa.text(
                f"SELECT count(*) FROM {table} "
                "WHERE dependency_set_hmac_schema_version="
                "'assistant-dependency-set-hmac:v3'"
            )
        )
        or 0
    )


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return
    schema = _trusted_schema(bind)
    _drop_authority(schema)
    op.execute(sa.text(_hardened_sql(schema)))


def downgrade():
    bind = op.get_bind()
    schema = _trusted_schema(bind) if bind.dialect.name == 'postgresql' else None
    if _v3_count(bind, schema):
        raise ValueError('cannot downgrade while legacy v3 messages exist')
    if bind.dialect.name != 'postgresql':
        return
    assert schema is not None
    _drop_authority(schema)
    op.execute(
        sa.text(
            f'DROP TABLE {_qualified(schema, "assistant_legacy_v3_dependency_staging")}'
        )
    )
    op.execute(
        sa.text(
            f'DROP TABLE {_qualified(schema, "assistant_legacy_v3_parent_staging")}'
        )
    )
    # Downgrading intentionally restores the frozen c6 implementation, which
    # is a known unsafe intermediate and must not be used as a running release.
    op.execute(
        sa.text(
            'SET LOCAL search_path = '
            f'{_quote_identifier(schema)}, pg_catalog, pg_temp'
        )
    )
    op.execute(sa.text(predecessor.PG_REGISTRATION))
    op.execute(sa.text(predecessor.PG_STAGING))
