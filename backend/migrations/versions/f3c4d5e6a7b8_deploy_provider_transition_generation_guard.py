"""deploy the provider transition generation guard

Revision ID: f3c4d5e6a7b8
Revises: e2b3c4d5f6a7
Create Date: 2026-09-01 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = 'f3c4d5e6a7b8'
down_revision = 'e2b3c4d5f6a7'
branch_labels = None
depends_on = None

TABLE_NAME = 'rag_provider_safety_transitions'
CONSTRAINT_NAME = 'ck_rag_provider_safety_transition_bootstrap_generation'
CONSTRAINT_SQL = (
    "(global_safety_generation = 0 AND transition_kind = 'bootstrap' AND "
    'readiness_id IS NULL) OR '
    "(global_safety_generation > 0 AND transition_kind <> 'bootstrap')"
)
E2_CONSTRAINT_SQL = (
    "(global_safety_generation = 0 AND transition_kind = 'bootstrap' AND "
    'readiness_id IS NULL) OR '
    "(global_safety_generation > 0 AND transition_kind <> 'bootstrap' AND "
    'readiness_id IS NOT NULL)'
)


def upgrade() -> None:
    _replace_constraint(CONSTRAINT_SQL)


def downgrade() -> None:
    _replace_constraint(E2_CONSTRAINT_SQL)


def _replace_constraint(expression: str) -> None:
    bind = op.get_bind()
    if TABLE_NAME not in inspect(bind).get_table_names():
        return
    existing = {
        item['name'] for item in inspect(bind).get_check_constraints(TABLE_NAME)
    }
    if bind.dialect.name == 'sqlite':
        with op.batch_alter_table(TABLE_NAME, recreate='always') as batch:
            if CONSTRAINT_NAME in existing:
                batch.drop_constraint(CONSTRAINT_NAME, type_='check')
            batch.create_check_constraint(CONSTRAINT_NAME, expression)
        return
    if CONSTRAINT_NAME in existing:
        op.drop_constraint(CONSTRAINT_NAME, TABLE_NAME, type_='check')
    op.create_check_constraint(CONSTRAINT_NAME, TABLE_NAME, expression)
