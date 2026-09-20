"""Create current ParaWorks schema.

Revision ID: 0001_create_current_schema
Revises:
Create Date: 2026-05-13
"""
from __future__ import annotations

from alembic import op
from sqlalchemy.orm import Session

import backend.app.models  # noqa: F401
from backend.app.db.base import Base
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore

revision = '0001_create_current_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, checkfirst=False)
    if bind.dialect.name == 'postgresql':
        session = Session(bind=bind)
        try:
            store = PgVectorStore(
                session=session,
                config=PgVectorConfig(embedding_dimensions=1536),
            )
            for statement in store.schema_sql():
                op.execute(statement)
        finally:
            session.close()


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute('DROP TABLE IF EXISTS rag_vector_documents')
    Base.metadata.drop_all(bind=bind)
