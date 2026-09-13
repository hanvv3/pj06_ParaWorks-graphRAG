from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables

_SECRET = b'task23-release-runtime-key-material-32-bytes'


def _provider_authority(path: Path) -> None:
    DurableFileAuthority(path).write({'provider': 'test-only'})


def _service(tmp_path: Path, **overrides):
    from backend.app.rag.release_authority import RagReleaseAuthority

    provider_path = tmp_path / 'provider' / 'safety.json'
    release_path = tmp_path / 'release' / 'ledger.json'
    if not provider_path.exists():
        _provider_authority(provider_path)
    values = {
        'marker_path': release_path,
        'provider_safety_latch_path': provider_path,
        'identity_secret': _SECRET,
        'fingerprint_key_version': 'v1',
        'designated_environment_id': 'release-validation',
        'designated_host_id': 'immutable-host-01',
        'repository_roots': (Path.cwd(),),
        'database_backup_roots': (tmp_path / 'db-backups',),
    }
    values.update(overrides)
    return RagReleaseAuthority(**values), release_path, provider_path


def _identity(name: str = 'validation_test', oid: int = 41):
    from backend.app.rag.release_authority import ValidationDatabaseIdentity

    return ValidationDatabaseIdentity(database_name=name, database_oid=oid)


def _review_args(seed: str = '1') -> dict[str, str]:
    return {
        'review_envelope_hmac': seed * 64,
        'review_nonce_hmac': str((int(seed, 16) + 1) % 16)[-1] * 64,
    }


def test_initialize_writes_canonical_hmac_marker_and_generation_zero_db_peer(
    tmp_path: Path,
) -> None:
    authority, marker_path, _provider_path = _service(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')

    with engine.begin() as connection:
        snapshot = authority.initialize(
            connection, database_identity=_identity(), **_review_args()
        )

    assert snapshot.ledger_epoch == 1
    assert snapshot.generation == 0
    assert snapshot.last_transition_digest is None
    assert snapshot.predecessor_marker_digest is None
    assert snapshot.rebootstrap_reason_hmac is None
    assert snapshot.designated_environment_id_hmac != 'release-validation'
    assert snapshot.designated_host_id_hmac != 'immutable-host-01'
    raw = marker_path.read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw))
    assert set(json.loads(raw)) == {'hmac_sha256', 'signed_payload'}
    assert b'immutable-host-01' not in raw
    assert b'release-validation' not in raw

    metadata = build_rag_release_metadata()
    tables = release_tables(metadata)
    with engine.connect() as connection:
        metadata.reflect(bind=connection)
        row = connection.execute(select(tables.ledgers)).mappings().one()
        assert row['ledger_uuid'] == str(snapshot.ledger_uuid)
        assert row['ledger_epoch'] == 1
        assert row['generation'] == 0
        assert row['designated_host_id_hmac'] == snapshot.designated_host_id_hmac
        assert connection.scalar(select(func.count()).select_from(tables.authorizations)) == 0


def test_initialize_refuses_second_init_and_marker_first_crash_is_fail_stop(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_authority import RagReleaseAuthorityError

    authority, marker_path, _provider_path = _service(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        authority.initialize(connection, database_identity=_identity(), **_review_args())
    with engine.begin() as connection, pytest.raises(RagReleaseAuthorityError, match='already'):
        authority.initialize(connection, database_identity=_identity(), **_review_args())

    other, other_marker, _provider_path = _service(
        tmp_path / 'crash',
        after_marker_replace=lambda: (_ for _ in ()).throw(RuntimeError('crash')),
    )
    other_engine = create_engine('sqlite+pysqlite:///:memory:')
    with other_engine.begin() as connection, pytest.raises(RuntimeError, match='crash'):
        other.initialize(
            connection,
            database_identity=_identity('other', 42),
            **_review_args('3'),
        )
    assert other_marker.exists()
    with other_engine.connect() as connection:
        assert not build_rag_release_metadata().tables.keys() <= set(
            __import__('sqlalchemy').inspect(connection).get_table_names()
        )
    with other_engine.begin() as connection, pytest.raises(RagReleaseAuthorityError):
        other.initialize(
            connection,
            database_identity=_identity('other', 42),
            **_review_args('3'),
        )
    assert marker_path.exists()


def test_inspection_rejects_marker_tamper_database_identity_and_host_drift(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_authority import RagReleaseAuthorityError

    authority, marker_path, provider_path = _service(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        authority.initialize(connection, database_identity=_identity(), **_review_args())
    with engine.connect() as connection:
        assert authority.inspect(connection, database_identity=_identity()).generation == 0
        with pytest.raises(RagReleaseAuthorityError, match='database'):
            authority.inspect(connection, database_identity=_identity('restored', 42))

    drifted, _marker, _provider = _service(
        tmp_path,
        provider_safety_latch_path=provider_path,
        designated_host_id='different-host',
    )
    with engine.connect() as connection, pytest.raises(RagReleaseAuthorityError, match='marker'):
        drifted.inspect(connection, database_identity=_identity())

    parsed = json.loads(marker_path.read_text(encoding='utf-8'))
    parsed['signed_payload']['body']['generation'] = 8
    marker_path.write_bytes(canonical_json_bytes(parsed))
    with engine.connect() as connection, pytest.raises(RagReleaseAuthorityError, match='marker'):
        authority.inspect(connection, database_identity=_identity())


def test_inspection_rejects_noncanonical_marker_bytes(tmp_path: Path) -> None:
    from backend.app.rag.release_authority import RagReleaseAuthorityError

    authority, marker_path, _provider_path = _service(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        authority.initialize(connection, database_identity=_identity(), **_review_args())
    parsed = json.loads(marker_path.read_text(encoding='utf-8'))
    marker_path.write_text(json.dumps(parsed, indent=2), encoding='utf-8')
    with engine.connect() as connection, pytest.raises(
        RagReleaseAuthorityError, match='noncanonical'
    ):
        authority.inspect(connection, database_identity=_identity())


@pytest.mark.parametrize('alias_kind', ['same-data', 'same-sidecar', 'ancestor'])
def test_four_authority_leaves_reject_cross_aliases(
    tmp_path: Path, alias_kind: str
) -> None:
    from backend.app.rag.release_authority import (
        ExternalAuthorityPathSetValidator,
        RagReleaseAuthorityError,
    )

    provider = (tmp_path / 'provider.json').absolute()
    release = (tmp_path / 'release.json').absolute()
    if alias_kind == 'same-data':
        release = provider
    elif alias_kind == 'same-sidecar':
        release = Path(str(provider) + '.lock')
    else:
        release = provider / 'nested.json'
    with pytest.raises(RagReleaseAuthorityError, match='distinct'):
        ExternalAuthorityPathSetValidator.validate(
            provider_path=provider,
            release_path=release,
            repository_roots=(),
            database_backup_roots=(),
        )


def test_release_marker_must_be_outside_repository_and_backup_roots(
    tmp_path: Path,
) -> None:
    from backend.app.rag.release_authority import (
        ExternalAuthorityPathSetValidator,
        RagReleaseAuthorityError,
    )

    provider = (tmp_path / 'provider.json').absolute()
    release = (tmp_path / 'repo' / 'release.json').absolute()
    with pytest.raises(RagReleaseAuthorityError, match='outside'):
        ExternalAuthorityPathSetValidator.validate(
            provider_path=provider,
            release_path=release,
            repository_roots=(tmp_path / 'repo',),
            database_backup_roots=(),
        )
    with pytest.raises(RagReleaseAuthorityError, match='outside'):
        ExternalAuthorityPathSetValidator.validate(
            provider_path=provider,
            release_path=release,
            repository_roots=(),
            database_backup_roots=(tmp_path / 'repo',),
        )


def test_rebootstrap_preserves_old_epoch_and_disaster_uses_fresh_ledger(
    tmp_path: Path,
) -> None:
    authority, marker_path, provider_path = _service(tmp_path)
    engine = create_engine('sqlite+pysqlite:///:memory:')
    with engine.begin() as connection:
        first = authority.initialize(
            connection, database_identity=_identity(), **_review_args()
        )
    reason = 'a' * 64
    with engine.begin() as connection, pytest.raises(
        __import__(
            'backend.app.rag.release_authority', fromlist=['RagReleaseAuthorityError']
        ).RagReleaseAuthorityError,
        match='healthy',
    ):
        authority.rebootstrap(
            connection,
            database_identity=_identity(),
            rebootstrap_reason_hmac=reason,
            **_review_args('3'),
        )
    tables = release_tables(build_rag_release_metadata())
    with engine.begin() as connection:
        connection.execute(
            tables.ledgers.update()
            .where(
                tables.ledgers.c.ledger_uuid == str(first.ledger_uuid),
                tables.ledgers.c.ledger_epoch == 1,
            )
            .values(generation=1, last_transition_digest='c' * 64)
        )
    with engine.begin() as connection:
        second = authority.rebootstrap(
            connection,
            database_identity=_identity(),
            rebootstrap_reason_hmac=reason,
            **_review_args('3'),
        )
    assert second.ledger_uuid == first.ledger_uuid
    assert second.ledger_epoch == 2
    assert second.generation == 0
    assert second.predecessor_marker_digest == first.marker_file_digest
    assert second.rebootstrap_reason_hmac == reason
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(tables.ledgers)) == 2

    marker_path.write_bytes(b'corrupt')
    disaster, _path, _provider = _service(
        tmp_path,
        provider_safety_latch_path=provider_path,
    )
    with engine.begin() as connection:
        third = disaster.disaster_initialize(
            connection,
            database_identity=_identity(),
            rebootstrap_reason_hmac='b' * 64,
            **_review_args('5'),
        )
    assert third.ledger_uuid not in {first.ledger_uuid, second.ledger_uuid}
    assert third.ledger_epoch == 1
    assert third.predecessor_marker_digest is None
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(tables.ledgers)) == 3
