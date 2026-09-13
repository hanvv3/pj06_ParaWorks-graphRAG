from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, create_mock_engine, inspect
from sqlalchemy.schema import CreateIndex

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.rag.release_authority import (
    RagReleaseAuthority,
    RagReleaseAuthorityError,
    ValidationDatabaseIdentity,
)
from backend.app.rag.release_schema import (
    RAG_RELEASE_SCHEMA_VERSION,
    assert_rag_release_physical_contract,
    build_rag_release_metadata,
    release_tables,
)

_SECRET = b'task23-remediation-runtime-key-material-32-bytes'


def _arguments(tmp_path: Path) -> dict[str, object]:
    return {
        'marker_path': tmp_path / 'release' / 'marker.json',
        'provider_safety_latch_path': tmp_path / 'provider' / 'state.json',
        'identity_secret': _SECRET,
        'fingerprint_key_version': 'v1',
        'designated_environment_id': 'live-validation',
        'designated_host_id': 'immutable-host',
        'repository_roots': (),
        'database_backup_roots': (),
    }


def test_arbitrary_provider_file_or_forgeable_peer_cannot_authorize_release(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    DurableFileAuthority(arguments['provider_safety_latch_path']).write(  # type: ignore[arg-type]
        {'provider': 'owner-only-but-not-task22'}
    )
    with pytest.raises(ValueError, match='pinned provider'):
        RagReleaseAuthority(**arguments, provider_safety_release_peer=object())
    assert not Path(arguments['marker_path']).exists()  # type: ignore[arg-type]


def test_public_release_mutation_is_postgresql_only_even_with_supplied_identity(
    tmp_path: Path,
) -> None:
    from backend.app.admin.rag_provider_safety import RagProviderSafetyReleasePeer

    peer = object.__new__(RagProviderSafetyReleasePeer)
    authority = RagReleaseAuthority(
        **_arguments(tmp_path), provider_safety_release_peer=peer
    )
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection, pytest.raises(
        RagReleaseAuthorityError, match='requires PostgreSQL'
    ):
        authority.initialize(
            connection,
            database_identity=ValidationDatabaseIdentity('forged', 1),
            review_envelope_hmac='1' * 64,
            review_nonce_hmac='2' * 64,
        )
    assert not authority.marker_path.exists()
    with engine.connect() as connection:
        assert not inspect(connection).get_table_names()


def test_frozen_postgresql_ddl_contains_claim_owner_identity_guards_and_version() -> None:
    statements: list[str] = []

    def record(sql, *multiparams, **params) -> None:
        del multiparams, params
        statements.append(str(sql.compile(dialect=engine.dialect)))

    engine = create_mock_engine('postgresql+psycopg://', record)
    metadata = build_rag_release_metadata()
    metadata.create_all(engine)
    rendered = '\n'.join(statements)
    assert 'rag_release_guard_database_identity' in rendered
    assert 'rag_release_guard_authorization' in rendered
    assert RAG_RELEASE_SCHEMA_VERSION in rendered
    owner = next(
        constraint
        for constraint in release_tables(metadata).authorizations.constraints
        if constraint.name == 'ck_rag_release_authorization_owner'
    )
    assert 'aborted_corpus_drift' in str(owner.sqltext)
    assert 'aborted_provider_safety' in str(owner.sqltext)
    claimed = next(
        index
        for index in release_tables(metadata).cases.indexes
        if index.name == 'uq_rag_release_one_claimed_case'
    )
    compiled = str(CreateIndex(claimed).compile(dialect=engine.dialect))
    assert 'UNIQUE INDEX uq_rag_release_one_claimed_case' in compiled
    assert "WHERE state = 'claimed'" in compiled


def test_physical_contract_inspector_never_treats_sqlite_as_operational_proof() -> None:
    engine = create_engine('sqlite+pysqlite:///:memory:')
    build_rag_release_metadata().create_all(engine)
    with engine.connect() as connection, pytest.raises(
        ValueError, match='requires PostgreSQL'
    ):
        assert_rag_release_physical_contract(connection)
