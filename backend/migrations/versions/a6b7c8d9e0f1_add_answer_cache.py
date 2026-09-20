"""Add independent default-off answer cache; no runtime integration.

Revision ID: a6b7c8d9e0f1
Revises: d7a8b9c0d1e2
"""

import sqlalchemy as sa
from alembic import op

revision = 'a6b7c8d9e0f1'
down_revision = 'd7a8b9c0d1e2'
branch_labels = None
depends_on = None


def upgrade():
    # The historical baseline builds current metadata on fresh databases.
    # Enumerate the current schema, not has_table's search_path fallback.
    if 'rag_answer_cache_entries' in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        'rag_answer_cache_entries',
        sa.Column('key_hmac', sa.String(64), primary_key=True),
        sa.Column('scope_hmac', sa.String(64), nullable=False),
        sa.Column('dependencies_json', sa.Text(), nullable=False),
        sa.Column('answer_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.BigInteger(), nullable=False),
        sa.Column('expires_at', sa.BigInteger(), nullable=False),
        sa.Column('value_hmac', sa.String(64), nullable=False),
        sa.CheckConstraint(
            'created_at >= 0 AND expires_at > created_at AND expires_at <= created_at + 86400',
            name='ck_answer_cache_retention',
        ),
        sa.CheckConstraint(
            'length(key_hmac) = 64 AND length(scope_hmac) = 64 AND length(value_hmac) = 64',
            name='ck_answer_cache_hmac',
        ),
    )
    op.create_index(
        'ix_answer_cache_expiry', 'rag_answer_cache_entries', ['expires_at']
    )


def downgrade():
    op.drop_index('ix_answer_cache_expiry', table_name='rag_answer_cache_entries')
    op.drop_table('rag_answer_cache_entries')
