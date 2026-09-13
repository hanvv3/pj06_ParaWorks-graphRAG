from __future__ import annotations

from pathlib import Path

from sqlalchemy import MetaData

from backend.app.db.base import Base

EXPECTED_TABLES = {
    'rag_live_gate_authorizations',
    'rag_live_gate_cases',
    'rag_live_gate_dispatches',
    'rag_live_gate_ledgers',
    'rag_live_gate_quality_reports',
    'rag_live_gate_transitions',
}
FORBIDDEN_COLUMN_PARTS = {
    'query',
    'evidence',
    'snippet',
    'output',
    'prompt',
    'response',
    'model_text',
}


def test_release_metadata_contains_exact_six_default_schema_tables() -> None:
    from backend.app.rag.release_schema import (
        RAG_RELEASE_TABLE_NAMES,
        build_rag_release_metadata,
    )

    metadata = build_rag_release_metadata()
    assert isinstance(metadata, MetaData)
    assert metadata.schema is None
    assert set(metadata.tables) == EXPECTED_TABLES
    assert frozenset(EXPECTED_TABLES) == RAG_RELEASE_TABLE_NAMES
    assert not (set(Base.metadata.tables) & EXPECTED_TABLES)


def test_release_schema_has_no_plaintext_or_application_foreign_keys() -> None:
    from backend.app.rag.release_schema import build_rag_release_metadata

    metadata = build_rag_release_metadata()
    for table in metadata.tables.values():
        assert table.schema is None
        for column in table.columns:
            assert not any(part in column.name for part in FORBIDDEN_COLUMN_PARTS)
        for foreign_key in table.foreign_keys:
            assert foreign_key.column.table.name in EXPECTED_TABLES


def test_release_tables_are_absent_from_application_migrations() -> None:
    migration_root = Path('backend/alembic/versions')
    migration_text = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in migration_root.glob('*.py')
    )
    assert not any(name in migration_text for name in EXPECTED_TABLES)


def test_release_tables_have_composite_scope_and_append_only_payload_shape() -> None:
    from backend.app.rag.release_schema import build_rag_release_metadata

    metadata = build_rag_release_metadata()
    for table in metadata.tables.values():
        assert {'ledger_uuid', 'ledger_epoch'} <= set(table.columns.keys())

    ledgers = metadata.tables['rag_live_gate_ledgers']
    assert tuple(column.name for column in ledgers.primary_key.columns) == (
        'ledger_uuid',
        'ledger_epoch',
    )
    assert {
        'bootstrap_review_envelope_hmac',
        'bootstrap_review_nonce_hmac',
        'designated_environment_id_hmac',
        'designated_host_id_hmac',
        'validation_database_identity_hmac',
        'validation_database_identity_uuid',
        'validation_database_oid',
    } <= set(ledgers.columns.keys())

    transitions = metadata.tables['rag_live_gate_transitions']
    assert tuple(column.name for column in transitions.primary_key.columns) == (
        'ledger_uuid',
        'ledger_epoch',
        'generation',
    )
    assert not transitions.c.payload_canonical_bytes.nullable
    assert transitions.c.payload_canonical_bytes.type.python_type is bytes

    reports = metadata.tables['rag_live_gate_quality_reports']
    assert tuple(column.name for column in reports.primary_key.columns) == (
        'ledger_uuid',
        'ledger_epoch',
        'approval_id_hmac',
    )
    assert not reports.c.payload_canonical_bytes.nullable
