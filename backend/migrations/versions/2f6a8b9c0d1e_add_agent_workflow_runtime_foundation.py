"""add agent workflow runtime foundation

Revision ID: 2f6a8b9c0d1e
Revises: b4b6d9f4d3e1
Create Date: 2026-08-26 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = '2f6a8b9c0d1e'
down_revision = 'b4b6d9f4d3e1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    _create_workflow_tables()
    _create_workflow_indexes()
    _add_idempotency_and_provenance_columns()
    _create_idempotency_and_provenance_indexes()


def downgrade() -> None:
    _drop_idempotency_and_provenance_indexes()
    _drop_idempotency_and_provenance_columns()
    _drop_index_if_exists(
        'ix_agent_workflow_evidence_refs_workflow_thread_id',
        'agent_workflow_evidence_refs',
    )
    for index_name in (
        'uq_agent_workflow_thread_client_request',
        'ix_agent_workflow_threads_expires_at',
        'ix_agent_workflow_threads_status',
        'ix_agent_workflow_threads_owner_subject_id',
        'ix_agent_workflow_threads_workflow_name',
    ):
        _drop_index_if_exists(index_name, 'agent_workflow_threads')
    for table_name in (
        'agent_runtime_schema_versions',
        'agent_workflow_evidence_refs',
        'agent_workflow_requests',
        'agent_workflow_threads',
    ):
        _drop_table_if_exists(table_name)


def _create_workflow_tables() -> None:
    _create_table_if_missing(
        'agent_workflow_threads',
        sa.Column('thread_id', sa.String(length=64), nullable=False),
        sa.Column('workflow_name', sa.String(length=64), nullable=False),
        sa.Column('graph_version', sa.String(length=64), nullable=False),
        sa.Column('checkpoint_thread_id', sa.String(length=128), nullable=False),
        sa.Column('checkpoint_store', sa.String(length=32), nullable=False),
        sa.Column('owner_subject_id', sa.String(length=128), nullable=False),
        sa.Column('security_scope_id', sa.String(length=128), nullable=False),
        sa.Column('client_request_id', sa.String(length=128), nullable=True),
        sa.Column('input_hash', sa.String(length=64), nullable=False),
        sa.Column('evidence_version_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('state_version', sa.Integer(), nullable=False),
        sa.Column('lease_token', sa.String(length=64), nullable=True),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'checkpoint_confirmed_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_by_subject_id', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('thread_id'),
        sa.UniqueConstraint('checkpoint_thread_id'),
    )
    _create_table_if_missing(
        'agent_workflow_requests',
        sa.Column('workflow_thread_id', sa.String(length=64), nullable=False),
        sa.Column('input_schema_version', sa.String(length=32), nullable=False),
        sa.Column('request_kind', sa.String(length=64), nullable=False),
        sa.Column('agent_names', sa.JSON(), nullable=False),
        sa.Column('selection_policy_version', sa.String(length=64), nullable=False),
        sa.Column('input_hash', sa.String(length=64), nullable=False),
        sa.Column('fingerprint_key_version', sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint('workflow_thread_id'),
    )
    _create_table_if_missing(
        'agent_workflow_evidence_refs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('workflow_thread_id', sa.String(length=64), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('canonical_source_type', sa.String(length=32), nullable=False),
        sa.Column('canonical_table', sa.String(length=64), nullable=False),
        sa.Column('canonical_row_id', sa.Integer(), nullable=False),
        sa.Column('document_version_id', sa.Integer(), nullable=True),
        sa.Column('external_revision', sa.String(length=255), nullable=True),
        sa.Column('content_signature', sa.String(length=128), nullable=False),
        sa.Column(
            'permission_level_snapshot',
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column('content_fingerprint', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'workflow_thread_id',
            'ordinal',
            name='uq_agent_workflow_evidence_ref_ordinal',
        ),
    )
    _create_table_if_missing(
        'agent_runtime_schema_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('component', sa.String(length=64), nullable=False),
        sa.Column('package_name', sa.String(length=128), nullable=False),
        sa.Column('package_version', sa.String(length=32), nullable=False),
        sa.Column('schema_revision', sa.Integer(), nullable=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('component'),
    )


def _create_workflow_indexes() -> None:
    for index_name, column_name in (
        ('ix_agent_workflow_threads_workflow_name', 'workflow_name'),
        ('ix_agent_workflow_threads_owner_subject_id', 'owner_subject_id'),
        ('ix_agent_workflow_threads_status', 'status'),
        ('ix_agent_workflow_threads_expires_at', 'expires_at'),
    ):
        _create_index_if_missing(
            index_name,
            'agent_workflow_threads',
            [column_name],
            unique=False,
        )
    _create_index_if_missing(
        'uq_agent_workflow_thread_client_request',
        'agent_workflow_threads',
        [
            'security_scope_id',
            'workflow_name',
            'owner_subject_id',
            'client_request_id',
        ],
        unique=True,
        where='client_request_id IS NOT NULL',
    )
    _create_index_if_missing(
        'ix_agent_workflow_evidence_refs_workflow_thread_id',
        'agent_workflow_evidence_refs',
        ['workflow_thread_id'],
        unique=False,
    )


def _add_idempotency_and_provenance_columns() -> None:
    for column in (
        sa.Column('workflow_thread_id', sa.String(length=64), nullable=True),
        sa.Column('candidate_key', sa.String(length=128), nullable=True),
        sa.Column('predecessor_review_item_id', sa.Integer(), nullable=True),
    ):
        _add_column_if_missing('review_items', column)
    for column in (
        sa.Column('workflow_thread_id', sa.String(length=64), nullable=True),
        sa.Column('effect_key', sa.String(length=128), nullable=True),
    ):
        _add_column_if_missing('agent_runs', column)
    for table_name in (
        'decision_records',
        'history_events',
        'timeline_events',
        'todos',
    ):
        _add_column_if_missing(
            table_name,
            sa.Column('source_review_item_id', sa.Integer(), nullable=True),
        )


def _create_idempotency_and_provenance_indexes() -> None:
    _create_index_if_missing(
        'uq_review_items_workflow_candidate',
        'review_items',
        ['workflow_thread_id', 'candidate_key'],
        unique=True,
        where='workflow_thread_id IS NOT NULL AND candidate_key IS NOT NULL',
    )
    _create_index_if_missing(
        'uq_agent_runs_workflow_effect',
        'agent_runs',
        ['workflow_thread_id', 'effect_key'],
        unique=True,
        where='workflow_thread_id IS NOT NULL AND effect_key IS NOT NULL',
    )
    for index_name, table_name in (
        ('uq_decision_records_source_review_item', 'decision_records'),
        ('uq_history_events_source_review_item', 'history_events'),
        ('uq_timeline_events_source_review_item', 'timeline_events'),
        ('uq_todos_source_review_item', 'todos'),
    ):
        _create_index_if_missing(
            index_name,
            table_name,
            ['source_review_item_id'],
            unique=True,
            where='source_review_item_id IS NOT NULL',
        )


def _drop_idempotency_and_provenance_indexes() -> None:
    for index_name, table_name in (
        ('uq_todos_source_review_item', 'todos'),
        ('uq_timeline_events_source_review_item', 'timeline_events'),
        ('uq_history_events_source_review_item', 'history_events'),
        ('uq_decision_records_source_review_item', 'decision_records'),
        ('uq_agent_runs_workflow_effect', 'agent_runs'),
        ('uq_review_items_workflow_candidate', 'review_items'),
    ):
        _drop_index_if_exists(index_name, table_name)


def _drop_idempotency_and_provenance_columns() -> None:
    for table_name in (
        'todos',
        'timeline_events',
        'history_events',
        'decision_records',
    ):
        _drop_column_if_exists(table_name, 'source_review_item_id')
    for column_name in ('effect_key', 'workflow_thread_id'):
        _drop_column_if_exists('agent_runs', column_name)
    for column_name in (
        'predecessor_review_item_id',
        'candidate_key',
        'workflow_thread_id',
    ):
        _drop_column_if_exists('review_items', column_name)


def _create_table_if_missing(table_name: str, *columns, **kwargs) -> None:
    if table_name not in inspect(op.get_bind()).get_table_names():
        op.create_table(table_name, *columns, **kwargs)


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    column_names = {
        item['name'] for item in inspect(op.get_bind()).get_columns(table_name)
    }
    if column.name not in column_names:
        op.add_column(table_name, column)


def _create_index_if_missing(
    index_name: str,
    table_name: str,
    columns: list[str],
    *,
    unique: bool,
    where: str | None = None,
) -> None:
    index_names = {
        item['name'] for item in inspect(op.get_bind()).get_indexes(table_name)
    }
    if index_name in index_names:
        return
    dialect_options = {}
    if where is not None:
        dialect_options = {
            'postgresql_where': sa.text(where),
            'sqlite_where': sa.text(where),
        }
    op.create_index(
        index_name,
        table_name,
        columns,
        unique=unique,
        **dialect_options,
    )


def _drop_index_if_exists(index_name: str, table_name: str) -> None:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return
    index_names = {item['name'] for item in inspector.get_indexes(table_name)}
    if index_name in index_names:
        op.drop_index(op.f(index_name), table_name=table_name)


def _drop_column_if_exists(table_name: str, column_name: str) -> None:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return
    column_names = {item['name'] for item in inspector.get_columns(table_name)}
    if column_name in column_names:
        op.drop_column(table_name, column_name)


def _drop_table_if_exists(table_name: str) -> None:
    if table_name in inspect(op.get_bind()).get_table_names():
        op.drop_table(table_name)
