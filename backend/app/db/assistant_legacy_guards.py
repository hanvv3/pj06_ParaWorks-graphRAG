"""Published legacy-v3 structural guards (SQLite immediate, PostgreSQL deferred)."""

from sqlalchemy import event, inspect


def invalid_v3_sql(dialect):
    array_len = 'jsonb_array_length' if dialect == 'postgresql' else 'json_array_length'

    def length(field):
        return (
            f'{array_len}(p.{field}::jsonb)'
            if dialect == 'postgresql'
            else f'{array_len}(p.{field})'
        )

    derived = (
        "p.metadata::jsonb -> 'evidence_derived' = 'true'::jsonb"
        if dialect == 'postgresql'
        else "json_type(p.metadata, '$.evidence_derived') = 'true'"
    )

    def count(where=''):
        return f'(SELECT count(*) FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id {where})'

    selected = count("AND d.dependency_role='selected_citation'")
    hashes = [
        'dependency_set_hmac',
        'dependency_child_hmac',
        'model_content_hmac',
        'canonical_citation_projection_hmac',
        'legacy_dependency_identity_hmac',
    ]

    def invalid_hash(column):
        return f'length({column})<>64 OR ' + (
            f"{column} !~ '^[0-9a-f]{{64}}$'"
            if dialect == 'postgresql'
            else f"{column} GLOB '*[^0-9a-f]*'"
        )

    bad_hash = ' OR '.join(
        f'd.{name} IS NULL OR {invalid_hash(f"d.{name}")}' for name in hashes
    )
    optional_hash = ' OR '.join(
        f'(d.{name} IS NOT NULL AND ({invalid_hash(f"d.{name}")}))'
        for name in (
            'selected_v1_citation_projection_hmac',
            'approval_provenance_hmac',
            'evidence_link_set_hmac',
        )
    )
    parent_hash = ' OR '.join(
        f'p.{name} IS NULL OR {invalid_hash(f"p.{name}")}'
        for name in (
            'assistant_message_content_hmac',
            'content_origin_hmac',
            'dependency_set_hmac',
            'parent_selected_evidence_projection_hmac',
        )
    )
    return f"""
    {parent_hash} OR
    p.serving_dependency_count IS NULL OR p.serving_dependency_count <= 0 OR
    {count()} <> p.serving_dependency_count OR
    (SELECT min(candidate_ordinal) FROM assistant_message_evidence_dependencies WHERE assistant_message_id=p.id) <> 0 OR
    (SELECT max(candidate_ordinal) FROM assistant_message_evidence_dependencies WHERE assistant_message_id=p.id) <> p.serving_dependency_count-1 OR
    {selected} <> {length('citations')} OR
    ({selected}=0 AND (NOT COALESCE(({derived}), false) OR
       {length('source_ids')}<>0 OR {length('source_links')}<>0 OR {length('source_snippets')}<>0)) OR
    EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND (
       d.dependency_serving_scope IS DISTINCT FROM 'legacy_v1_only' OR
       d.dependency_role NOT IN ('selected_citation','legacy_evidence_influence') OR
       d.dependency_set_hmac IS DISTINCT FROM p.dependency_set_hmac OR
       d.fingerprint_key_version IS DISTINCT FROM p.content_hmac_key_version OR
       d.fingerprint_key_material_verifier IS DISTINCT FROM p.content_hmac_key_material_verifier OR
       d.serving_identity_hmac IS NOT NULL OR d.serving_version_fingerprint IS NOT NULL OR d.support_mode IS NOT NULL OR
       {bad_hash} OR {optional_hash} OR
       (d.approval_link_id IS NOT NULL AND (
          NOT EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=d.id) OR
          (SELECT count(*) FROM assistant_message_knowledge_evidence_refs r WHERE r.dependency_id=d.id) <>
          (SELECT count(*) FROM trusted_knowledge_evidence_links e WHERE e.approval_link_id=d.approval_link_id))) OR
       EXISTS (SELECT 1 FROM assistant_message_knowledge_evidence_refs r
          LEFT JOIN trusted_knowledge_evidence_links e ON e.id=r.trusted_knowledge_evidence_link_id
          WHERE r.dependency_id=d.id AND (r.assistant_message_id IS DISTINCT FROM p.id OR
          r.approval_link_id IS DISTINCT FROM d.approval_link_id OR e.id IS NULL OR
          e.approval_link_id IS DISTINCT FROM d.approval_link_id))
    ))
    """


def sqlite_guard_statements():
    invalid = invalid_v3_sql('sqlite')
    query = f"SELECT 1 FROM assistant_messages p WHERE p.id={{target}} AND ((p.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND ({invalid})) OR (p.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v2' AND p.content_origin='legacy_evidence' AND EXISTS (SELECT 1 FROM assistant_message_evidence_dependencies d WHERE d.assistant_message_id=p.id AND (d.dependency_kind<>'legacy_unbound' OR d.dependency_role<>'selected_citation' OR d.dependency_serving_scope<>'legacy_v1_only'))))"
    result = []
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
                f'EXISTS ({query.format(target=target)})' for target in targets
            )
            if table == 'assistant_messages' and action == 'UPDATE':
                conditions += " OR (OLD.dependency_set_hmac_schema_version='assistant-dependency-set-hmac:v3' AND NEW.dependency_set_hmac_schema_version IS DISTINCT FROM 'assistant-dependency-set-hmac:v3')"
            result.append(
                f"CREATE TRIGGER IF NOT EXISTS legacy_v3_{table}_{action.lower()} AFTER {action} ON {table} BEGIN SELECT CASE WHEN {conditions} THEN RAISE(ABORT,'legacy v3 published integrity mismatch') END; END"
            )
    return tuple(result)


def register_sqlite_guards(metadata):
    @event.listens_for(metadata, 'after_create')
    def install(target, connection, **kwargs):
        if (
            connection.dialect.name == 'sqlite'
            and 'assistant_messages' in target.tables
            and not inspect(connection).has_table('alembic_version')
        ):
            for statement in sqlite_guard_statements():
                connection.exec_driver_sql(statement)
