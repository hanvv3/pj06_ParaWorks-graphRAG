"""harden extraction call lifecycle

Revision ID: 9d7f3a1c6e20
Revises: 7c5a2e9f4b10
Create Date: 2026-08-28 12:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = '9d7f3a1c6e20'
down_revision = '7c5a2e9f4b10'
branch_labels = None
depends_on = None

_TABLE = 'auto_review_extraction_calls'
_CONSTRAINT = 'ck_auto_review_extraction_calls_terminal_charge'

_HARDENED = (
    "(status = 'claimed' AND terminal_at IS NULL AND "
    '((provider_attempt_count = 0 AND attempt_started_at IS NULL) OR '
    '(provider_attempt_count = 1 AND attempt_started_at IS NOT NULL)) AND '
    'charged_input_tokens IS NULL AND charged_output_tokens IS NULL AND '
    'charged_cost_usd = 0 AND budget_overrun = false) OR '
    "(status = 'completed' AND provider_attempt_count = 1 AND "
    'attempt_started_at IS NOT NULL AND terminal_at IS NOT NULL AND '
    'charged_input_tokens IS NOT NULL AND charged_input_tokens >= 0 AND '
    'charged_output_tokens IS NOT NULL AND charged_output_tokens >= 0) OR '
    "(status = 'failed' AND terminal_at IS NOT NULL AND "
    '((provider_attempt_count = 0 AND attempt_started_at IS NULL AND '
    'charged_input_tokens = 0 AND charged_output_tokens = 0 AND '
    'charged_cost_usd = 0 AND budget_overrun = false) OR '
    '(provider_attempt_count = 1 AND attempt_started_at IS NOT NULL AND '
    'charged_input_tokens IS NOT NULL AND charged_input_tokens >= 0 AND '
    'charged_output_tokens IS NOT NULL AND charged_output_tokens >= 0)))'
)

_TASK2 = (
    "(status = 'claimed' AND terminal_at IS NULL AND "
    'provider_attempt_count = 0 AND attempt_started_at IS NULL AND '
    'charged_input_tokens IS NULL AND charged_output_tokens IS NULL AND '
    'charged_cost_usd = 0 AND budget_overrun = false) OR '
    "(status = 'completed' AND provider_attempt_count = 1 AND "
    'attempt_started_at IS NOT NULL AND terminal_at IS NOT NULL AND '
    'charged_input_tokens IS NOT NULL AND charged_input_tokens >= 0 AND '
    'charged_output_tokens IS NOT NULL AND charged_output_tokens >= 0) OR '
    "(status = 'failed' AND terminal_at IS NOT NULL AND "
    '((provider_attempt_count = 0 AND attempt_started_at IS NULL AND '
    'charged_input_tokens = 0 AND charged_output_tokens = 0 AND '
    'charged_cost_usd = 0 AND budget_overrun = false) OR '
    '(provider_attempt_count = 1 AND attempt_started_at IS NOT NULL AND '
    'charged_input_tokens IS NOT NULL AND charged_input_tokens >= 0 AND '
    'charged_output_tokens IS NOT NULL AND charged_output_tokens >= 0)))'
)


def upgrade() -> None:
    if not _table_exists():
        return
    _replace_constraint(_HARDENED)


def downgrade() -> None:
    if not _table_exists():
        return
    connection = op.get_bind()
    retained = connection.scalar(sa.text(f'SELECT count(*) FROM {_TABLE}'))
    if retained:
        raise RuntimeError(
            'refusing extraction lifecycle downgrade while C.5 calls are retained'
        )
    _replace_constraint(_TASK2)


def _replace_constraint(expression: str) -> None:
    if op.get_bind().dialect.name == 'sqlite':
        with op.batch_alter_table(_TABLE, recreate='always') as batch:
            batch.drop_constraint(_CONSTRAINT, type_='check')
            batch.create_check_constraint(_CONSTRAINT, expression)
        return
    op.drop_constraint(_CONSTRAINT, _TABLE, type_='check')
    op.create_check_constraint(_CONSTRAINT, _TABLE, expression)


def _table_exists() -> bool:
    return _TABLE in inspect(op.get_bind()).get_table_names()
