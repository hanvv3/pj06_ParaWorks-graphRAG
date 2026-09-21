from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.rag.pgvector_store import PgVectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate the local PostgreSQL + pgvector dev database.')
    parser.add_argument(
        '--database-url',
        default=os.getenv('DATABASE_URL'),
        help='SQLAlchemy database URL. Defaults to DATABASE_URL.',
    )
    parser.add_argument(
        '--ensure-vector-schema',
        action='store_true',
        help='Create the pgvector extension and rag_vector_documents table if missing.',
    )
    parser.add_argument(
        '--expect-app-schema',
        action='store_true',
        help='Also require app tables created by backend.app.db.init_db.',
    )
    args = parser.parse_args()

    if not args.database_url:
        print('DATABASE_URL is required.', file=sys.stderr)
        return 2

    try:
        engine = create_engine(args.database_url, pool_pre_ping=True)
        if args.ensure_vector_schema:
            with Session(engine) as session:
                PgVectorStore(session=session).ensure_schema()
                session.commit()

        with engine.connect() as connection:
            connection.execute(text('SELECT 1'))
            vector_version = connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            vector_table = connection.scalar(text("SELECT to_regclass('public.rag_vector_documents')"))
            state_table = connection.scalar(text("SELECT to_regclass('public.vector_index_states')"))

        missing = []
        if vector_version is None:
            missing.append('pgvector extension')
        if vector_table is None:
            missing.append('rag_vector_documents table')
        if args.expect_app_schema and state_table is None:
            missing.append('vector_index_states table')

        if missing:
            print(f'pgvector dev database is missing: {", ".join(missing)}', file=sys.stderr)
            return 1

        print('pgvector dev database OK')
        print(f'  vector extension: {vector_version}')
        print(f'  vector table: {vector_table}')
        if args.expect_app_schema:
            print(f'  index state table: {state_table}')
        return 0
    except SQLAlchemyError as exc:
        print('Could not connect to the pgvector dev database.', file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
