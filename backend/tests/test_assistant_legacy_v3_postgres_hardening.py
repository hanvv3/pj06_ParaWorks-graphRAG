"""Static PostgreSQL authority contracts for the legacy-v3 migration head."""

import importlib
from types import SimpleNamespace

import pytest
from sqlalchemy import text

MIGRATION = (
    'backend.migrations.versions.'
    'd7a8b9c0d1e2_secure_legacy_v3_postgres_authority'
)
SCHEMA = 'paraworks_app'
FUNCTION_TABLE_OP = {
    'enforce_assistant_legacy_v3_staging': (
        'assistant_message_evidence_dependencies',
        "TG_OP NOT IN ('UPDATE', 'DELETE')",
    ),
    'register_assistant_legacy_v3_parent_insert': (
        'assistant_messages',
        "TG_OP IS DISTINCT FROM 'INSERT'",
    ),
    'register_assistant_legacy_v3_dependency_insert': (
        'assistant_message_evidence_dependencies',
        "TG_OP IS DISTINCT FROM 'INSERT'",
    ),
    'require_consumed_assistant_legacy_v3_parent_staging': (
        'assistant_legacy_v3_parent_staging',
        "TG_OP IS DISTINCT FROM 'INSERT'",
    ),
    'require_consumed_assistant_legacy_v3_dependency_staging': (
        'assistant_legacy_v3_dependency_staging',
        "TG_OP IS DISTINCT FROM 'INSERT'",
    ),
    'consume_assistant_legacy_v3_parent_staging': (
        'assistant_messages',
        "TG_OP IS DISTINCT FROM 'UPDATE'",
    ),
    'clear_assistant_legacy_v3_parent_staging': (
        'assistant_messages',
        "TG_OP IS DISTINCT FROM 'UPDATE'",
    ),
}


def _upgrade_sql(monkeypatch):
    migration = importlib.import_module(MIGRATION)
    emitted: list[str] = []

    def scalar(statement):
        return SCHEMA if 'current_schema' in str(statement) else 0

    monkeypatch.setattr(
        migration,
        'op',
        SimpleNamespace(
            get_bind=lambda: SimpleNamespace(
                dialect=SimpleNamespace(name='postgresql'),
                scalar=scalar,
            ),
            execute=lambda statement: emitted.append(str(statement)),
        ),
    )
    migration.upgrade()
    return migration, emitted


def _function_body(sql: str, name: str) -> str:
    marker = f'CREATE FUNCTION "{SCHEMA}"."{name}"()'
    return sql.split(marker, 1)[1].split('END $function$;', 1)[0]


def test_trusted_schema_is_quoted_and_temporary_or_system_names_are_refused():
    migration = importlib.import_module(MIGRATION)
    quoted = migration._hardened_sql('tenant"blue')

    assert '"tenant""blue"."assistant_messages"' in quoted
    assert (
        "'\"tenant\"\"blue\".\"assistant_messages\"'::pg_catalog.regclass"
        in quoted
    )
    for value in (None, '', 'pg_catalog', 'information_schema', 'pg_temp_4'):
        bind = SimpleNamespace(scalar=lambda _statement, value=value: value)
        with pytest.raises(ValueError):
            migration._trusted_schema(bind)


@pytest.mark.parametrize(
    'schema',
    ('public', 'tenant"blue', "tenant'blue", 'tenant blue', '회사'),
)
def test_trusted_schema_keeps_supported_quoted_names(schema):
    migration = importlib.import_module(MIGRATION)
    bind = SimpleNamespace(scalar=lambda _statement: schema)

    assert migration._trusted_schema(bind) == schema
    sql = migration._hardened_sql(schema)
    function_body = sql.split(' AS $function$', 1)[1].split('$function$', 1)[0]
    assert function_body.rstrip().endswith('END')
    assert not text(sql).compile().params


@pytest.mark.parametrize(
    'schema',
    ('tenant:blue', ':tenant', 'tenant$function$blue'),
)
def test_trusted_schema_rejects_names_unsafe_for_generated_sql(schema):
    migration = importlib.import_module(MIGRATION)
    bind = SimpleNamespace(scalar=lambda _statement: schema)

    with pytest.raises(ValueError) as exc_info:
        migration._trusted_schema(bind)

    assert exc_info.value.args == (
        'unsupported PostgreSQL application schema spelling',
    )


@pytest.mark.parametrize('operation', ('upgrade', 'downgrade'))
@pytest.mark.parametrize('schema', (':tenant', 'tenant$function$blue'))
def test_unsafe_schema_is_refused_before_any_generated_sql(
    monkeypatch,
    operation,
    schema,
):
    migration = importlib.import_module(MIGRATION)
    emitted: list[str] = []
    bind = SimpleNamespace(
        dialect=SimpleNamespace(name='postgresql'),
        scalar=lambda _statement: schema,
    )
    monkeypatch.setattr(
        migration,
        'op',
        SimpleNamespace(
            get_bind=lambda: bind,
            execute=lambda statement: emitted.append(str(statement)),
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        getattr(migration, operation)()

    assert exc_info.value.args == (
        'unsupported PostgreSQL application schema spelling',
    )
    assert emitted == []


def test_definer_functions_reject_foreign_trigger_relations_and_operations(monkeypatch):
    _, emitted = _upgrade_sql(monkeypatch)
    sql = '\n'.join(emitted)
    relation_prefix = f"'\"{SCHEMA}\".\""

    assert 'SET search_path FROM CURRENT' not in sql
    assert sql.count(
        f'SET search_path = pg_catalog, "{SCHEMA}", pg_temp'
    ) == len(FUNCTION_TABLE_OP)
    for function, (table, operation_guard) in FUNCTION_TABLE_OP.items():
        body = _function_body(sql, function)
        assert (
            f"TG_RELID IS DISTINCT FROM {relation_prefix}{table}\"'::pg_catalog.regclass"
            in body
        )
        assert operation_guard in body
        first_row_reference = min(
            position
            for marker in ('NEW.', 'OLD.')
            if (position := body.find(marker)) >= 0
        )
        assert body.index('TG_RELID') < first_row_reference
        assert body.index(operation_guard) < first_row_reference
        assert (
            f'REVOKE ALL ON FUNCTION "{SCHEMA}"."{function}"() FROM PUBLIC;'
            in sql
        )


def test_definer_replacement_removes_old_grants_and_qualifies_authority(monkeypatch):
    _, emitted = _upgrade_sql(monkeypatch)
    sql = '\n'.join(emitted)

    for function in FUNCTION_TABLE_OP:
        assert f'DROP FUNCTION IF EXISTS "{SCHEMA}"."{function}"()' in sql
    assert f'FROM "{SCHEMA}"."assistant_legacy_v3_parent_staging"' in sql
    assert f'JOIN "{SCHEMA}"."assistant_legacy_v3_dependency_staging"' in sql
    assert f'DELETE FROM "{SCHEMA}"."assistant_legacy_v3_parent_staging"' in sql
    assert (
        f'BEFORE UPDATE OR DELETE ON "{SCHEMA}".'
        '"assistant_message_evidence_dependencies"' in sql
    )
    assert f'AFTER INSERT ON "{SCHEMA}"."assistant_messages"' in sql


def test_deferred_consumers_are_row_specific_and_downgrade_restores_c6(monkeypatch):
    migration, emitted = _upgrade_sql(monkeypatch)
    sql = '\n'.join(emitted)
    parent = _function_body(
        sql, 'require_consumed_assistant_legacy_v3_parent_staging'
    )
    dependency = _function_body(
        sql, 'require_consumed_assistant_legacy_v3_dependency_staging'
    )

    assert 'NEW.dependency_id' not in parent
    assert 'NEW.parent_id' in parent and 'NEW.transaction_id' in parent
    assert 'NEW.dependency_id' in dependency
    assert sql.count('DEFERRABLE INITIALLY DEFERRED') == 2
    assert (
        f'CREATE FUNCTION "{SCHEMA}".'
        '"require_consumed_assistant_legacy_v3_staging"()' not in sql
    )

    emitted.clear()
    migration.downgrade()
    downgraded = '\n'.join(emitted)
    assert migration.predecessor.PG_REGISTRATION in downgraded
    assert migration.predecessor.PG_STAGING in downgraded
    assert downgraded.index(
        f'DROP TABLE "{SCHEMA}"."assistant_legacy_v3_dependency_staging"'
    ) < downgraded.index(
        f'DROP TABLE "{SCHEMA}"."assistant_legacy_v3_parent_staging"'
    )
