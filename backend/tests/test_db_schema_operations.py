import warnings
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning

from backend.app.db.base import Base
from scripts.checks.check_db_schema import check_schema


def test_alembic_operational_files_exist() -> None:
    assert Path('alembic.ini').is_file()
    assert Path('backend/migrations/env.py').is_file()
    assert Path('backend/migrations/script.py.mako').is_file()
    assert Path('backend/migrations/versions/0001_create_current_schema.py').is_file()
    assert Path(
        'backend/migrations/versions/'
        '2f6a8b9c0d1e_add_agent_workflow_runtime_foundation.py'
    ).is_file()
    assert Path(
        'backend/migrations/versions/'
        '7c5a2e9f4b10_add_auto_review_trust_promotion.py'
    ).is_file()


def test_project_key_migration_is_idempotent_with_current_schema_baseline() -> None:
    migration = Path('backend/migrations/versions/5f8d874023d7_add_project_key_to_knowledge_models.py').read_text(
        encoding='utf-8',
    )

    assert '_add_column_if_missing' in migration
    assert '_create_index_if_missing' in migration
    assert "op.add_column('decision_records'" not in migration


def test_projects_table_migration_is_idempotent_with_current_schema_baseline() -> None:
    migration = Path('backend/migrations/versions/9451b1f116b5_add_projects_table.py').read_text(
        encoding='utf-8',
    )

    assert '_create_projects_table_if_missing' in migration
    assert '_create_index_if_missing' in migration
    assert "op.create_table('projects'" not in migration


def test_schema_checker_reports_missing_model_tables() -> None:
    engine = create_engine('sqlite:///:memory:')

    result = check_schema(engine, include_native_pgvector=False)

    assert not result.ok
    assert 'auth_users' in result.missing_tables
    assert 'assistant_conversations' in result.missing_tables


def test_schema_checker_accepts_sqlalchemy_metadata_tables() -> None:
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(bind=engine)

    result = check_schema(engine, include_native_pgvector=False)

    assert result.ok
    assert result.missing_tables == []
    assert result.missing_columns == {}


class FakePostgresEngine:
    dialect = type('Dialect', (), {'name': 'postgresql'})()


class FakeSchemaInspector:
    def get_table_names(self) -> list[str]:
        return [*Base.metadata.tables, 'rag_vector_documents']

    def get_columns(self, table_name: str) -> list[dict[str, str]]:
        if table_name == 'rag_vector_documents':
            return [
                {'name': 'document_id'},
                {'name': 'text'},
                {'name': 'source_url'},
                {'name': 'source_snippet'},
                {'name': 'permission_level'},
                {'name': 'metadata_json'},
                {'name': 'embedding'},
                {'name': 'updated_at'},
            ]
        return [{'name': column.name} for column in Base.metadata.tables[table_name].columns]


def test_schema_checker_reports_pgvector_embedding_dimension_mismatch(monkeypatch) -> None:
    engine = FakePostgresEngine()

    monkeypatch.setattr('scripts.checks.check_db_schema.inspect', lambda _engine: FakeSchemaInspector())
    monkeypatch.setattr(
        'scripts.checks.check_db_schema._postgres_column_type',
        lambda _engine, _table_name, _column_name: 'vector(3072)',
    )

    result = check_schema(
        engine,
        include_native_pgvector=True,
        expected_embedding_dimensions=1536,
    )

    assert not result.ok
    assert (
        'rag_vector_documents.embedding type is vector(3072), expected vector(1536)'
        in result.native_pgvector_errors
    )


def test_schema_checker_uses_default_pgvector_dimension_when_not_overridden(monkeypatch) -> None:
    engine = FakePostgresEngine()

    monkeypatch.setattr('scripts.checks.check_db_schema.inspect', lambda _engine: FakeSchemaInspector())
    monkeypatch.setattr(
        'scripts.checks.check_db_schema._postgres_column_type',
        lambda _engine, _table_name, _column_name: 'vector(3072)',
    )

    result = check_schema(engine, include_native_pgvector=True)

    assert not result.ok
    assert (
        'rag_vector_documents.embedding type is vector(3072), expected vector(1536)'
        in result.native_pgvector_errors
    )


class WarningPgvectorInspector(FakeSchemaInspector):
    def get_columns(self, table_name: str) -> list[dict[str, str]]:
        if table_name == 'rag_vector_documents':
            warnings.warn(
                "Did not recognize type 'vector' of column 'embedding'",
                SAWarning,
                stacklevel=2,
            )
        return super().get_columns(table_name)


def test_schema_checker_suppresses_expected_pgvector_reflection_warning(monkeypatch) -> None:
    engine = FakePostgresEngine()

    monkeypatch.setattr('scripts.checks.check_db_schema.inspect', lambda _engine: WarningPgvectorInspector())
    monkeypatch.setattr(
        'scripts.checks.check_db_schema._postgres_column_type',
        lambda _engine, _table_name, _column_name: 'vector(1536)',
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        result = check_schema(
            engine,
            include_native_pgvector=True,
            expected_embedding_dimensions=1536,
        )

    assert result.ok
    assert not [
        item
        for item in caught_warnings
        if "Did not recognize type 'vector'" in str(item.message)
    ]
