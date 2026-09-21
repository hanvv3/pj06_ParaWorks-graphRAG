from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import SAWarning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.app.models  # noqa: E402,F401
from backend.app.core.config import get_settings  # noqa: E402
from backend.app.db.base import Base  # noqa: E402
from backend.app.rag.pgvector_store import PgVectorConfig  # noqa: E402

DEFAULT_PGVECTOR_CONFIG = PgVectorConfig()
NATIVE_PGVECTOR_TABLE = DEFAULT_PGVECTOR_CONFIG.table_name
DEFAULT_EMBEDDING_DIMENSIONS = DEFAULT_PGVECTOR_CONFIG.embedding_dimensions
NATIVE_PGVECTOR_COLUMNS = {
    'document_id',
    'text',
    'source_url',
    'source_snippet',
    'permission_level',
    'metadata_json',
    'embedding',
    'updated_at',
}


@dataclass(frozen=True)
class SchemaCheckResult:
    missing_tables: list[str] = field(default_factory=list)
    missing_columns: dict[str, list[str]] = field(default_factory=dict)
    native_pgvector_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_tables and not self.missing_columns and not self.native_pgvector_errors


def check_schema(
    engine: Engine,
    *,
    include_native_pgvector: bool | None = None,
    expected_embedding_dimensions: int | None = None,
) -> SchemaCheckResult:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    missing_tables: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    native_pgvector_errors: list[str] = []

    for table_name, table in sorted(Base.metadata.tables.items()):
        if table_name not in existing_tables:
            missing_tables.append(table_name)
            continue
        existing_columns = {column['name'] for column in inspector.get_columns(table_name)}
        expected_columns = {column.name for column in table.columns}
        missing = sorted(expected_columns - existing_columns)
        if missing:
            missing_columns[table_name] = missing

    should_check_pgvector = (
        include_native_pgvector
        if include_native_pgvector is not None
        else engine.dialect.name == 'postgresql'
    )
    if should_check_pgvector:
        native_pgvector_errors.extend(
            _check_native_pgvector_table(
                engine,
                existing_tables=existing_tables,
                expected_embedding_dimensions=expected_embedding_dimensions,
            )
        )

    return SchemaCheckResult(
        missing_tables=missing_tables,
        missing_columns=missing_columns,
        native_pgvector_errors=native_pgvector_errors,
    )


def _check_native_pgvector_table(
    engine: Engine,
    *,
    existing_tables: set[str],
    expected_embedding_dimensions: int | None,
) -> list[str]:
    if NATIVE_PGVECTOR_TABLE not in existing_tables:
        return [f'{NATIVE_PGVECTOR_TABLE} table is missing']

    inspector = inspect(engine)
    # pgvector 타입은 SQLAlchemy 기본 리플렉션에서 경고가 나므로, 검사용 경로에서만 숨긴다.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r"Did not recognize type 'vector' of column 'embedding'",
            category=SAWarning,
        )
        existing_columns = {
            column['name'] for column in inspector.get_columns(NATIVE_PGVECTOR_TABLE)
        }
    missing_columns = sorted(NATIVE_PGVECTOR_COLUMNS - existing_columns)
    errors = [
        f'{NATIVE_PGVECTOR_TABLE}.{column_name} column is missing'
        for column_name in missing_columns
    ]

    expected_dimensions = expected_embedding_dimensions or DEFAULT_EMBEDDING_DIMENSIONS
    if engine.dialect.name == 'postgresql':
        actual_type = _postgres_column_type(engine, NATIVE_PGVECTOR_TABLE, 'embedding')
        expected_type = f'vector({expected_dimensions})'
        if actual_type is None:
            errors.append(f'{NATIVE_PGVECTOR_TABLE}.embedding type could not be inspected')
        elif actual_type != expected_type:
            errors.append(
                f'{NATIVE_PGVECTOR_TABLE}.embedding type is {actual_type}, expected {expected_type}'
            )

    return errors


def _postgres_column_type(engine: Engine, table_name: str, column_name: str) -> str | None:
    query = text(
        """
        SELECT format_type(attribute.atttypid, attribute.atttypmod) AS column_type
        FROM pg_attribute AS attribute
        JOIN pg_class AS class ON class.oid = attribute.attrelid
        JOIN pg_namespace AS namespace ON namespace.oid = class.relnamespace
        WHERE namespace.nspname = 'public'
          AND class.relname = :table_name
          AND attribute.attname = :column_name
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped
        """
    )
    with engine.connect() as connection:
        return connection.execute(
            query,
            {'table_name': table_name, 'column_name': column_name},
        ).scalar_one_or_none()


def _print_result(result: SchemaCheckResult) -> None:
    if result.ok:
        print('DB schema check passed.')
        return

    print('DB schema check failed.')
    if result.missing_tables:
        print('Missing tables:')
        for table_name in result.missing_tables:
            print(f'- {table_name}')
    if result.missing_columns:
        print('Missing columns:')
        for table_name, columns in result.missing_columns.items():
            print(f'- {table_name}: {", ".join(columns)}')
    if result.native_pgvector_errors:
        print('Native pgvector errors:')
        for error in result.native_pgvector_errors:
            print(f'- {error}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate ParaWorks database schema.')
    parser.add_argument('--database-url', default=None)
    parser.add_argument('--skip-native-pgvector', action='store_true')
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.resolved_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)
    result = check_schema(
        engine,
        include_native_pgvector=False if args.skip_native_pgvector else None,
        expected_embedding_dimensions=settings.openai_embedding_dimensions,
    )
    _print_result(result)
    return 0 if result.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
