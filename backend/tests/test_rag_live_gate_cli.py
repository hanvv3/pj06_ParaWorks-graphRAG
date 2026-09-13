from __future__ import annotations

import hashlib
import hmac
import io
import json
import socket
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.core.config import Settings

_REVIEW_KEY = b'task23-distinct-external-review-key-32-bytes'
_RUNTIME_KEY = b'task23-release-runtime-key-material-32-bytes'
_PLAN_HMAC = '1' * 64
_KEY_ID = 'rag-release-review-v1'


def _review_verifier() -> str:
    return hmac.new(
        _REVIEW_KEY,
        b'paraworks:rag-release-admin-review-key-verifier:v1\x00',
        hashlib.sha256,
    ).hexdigest()


def _target(tmp_path: Path):
    from backend.app.admin.rag_live_gate import RagReleaseAdminTarget

    provider = tmp_path / 'provider' / 'state.json'
    if not provider.exists():
        DurableFileAuthority(provider).write({'provider': 'test-only'})
    return RagReleaseAdminTarget.build(
        database_url='postgresql+psycopg://release:private@127.0.0.1/validation',
        marker_path=tmp_path / 'release' / 'marker.json',
        provider_safety_latch_path=provider,
        designated_environment_id='release-validation',
        designated_host_id='host-one',
        review_secret=_REVIEW_KEY,
    )


def _review_bytes(
    target,
    operation: str,
    *,
    expected_context: dict[str, object] | None = None,
    reason_hmac: str | None = None,
    domain: bytes = b'paraworks:rag-release-admin-review:v1\x00',
    nonce: str | None = None,
) -> bytes:
    signed = {
        'actor_subject_hmac': '2' * 64,
        'expected_context': expected_context,
        'implementation_plan_reference_hmac': _PLAN_HMAC,
        'nonce': nonce or str(uuid4()),
        'operation': operation,
        'rebootstrap_reason_hmac': reason_hmac,
        'review_authority_key_id': _KEY_ID,
        'schema_version': 'rag-release-admin-review:v1',
        'target': target.review_identity,
    }
    return canonical_json_bytes(
        {
            'hmac_sha256': hmac.new(
                _REVIEW_KEY,
                domain + canonical_json_bytes(signed),
                hashlib.sha256,
            ).hexdigest(),
            'signed_payload': signed,
        }
    )


def _admin(tmp_path: Path):
    from backend.app.admin.rag_live_gate import RagLiveGateAdminService
    from backend.app.rag.release_authority import RagReleaseAuthority

    target = _target(tmp_path)
    authority = RagReleaseAuthority(
        marker_path=target.marker_path,
        provider_safety_latch_path=target.provider_safety_latch_path,
        identity_secret=_RUNTIME_KEY,
        fingerprint_key_version='v1',
        designated_environment_id=target.designated_environment_id,
        designated_host_id=target.designated_host_id,
        repository_roots=(),
        database_backup_roots=(),
    )
    engine = create_engine('sqlite+pysqlite:///:memory:')
    admin = RagLiveGateAdminService(
        target=target,
        connection_factory=engine.connect,
        authority=authority,
        review_secret=_REVIEW_KEY,
        review_key_id=_KEY_ID,
        implementation_plan_reference_hmac=_PLAN_HMAC,
        review_key_registry={_KEY_ID: _review_verifier()},
        database_identity_factory=lambda _connection: __import__(
            'backend.app.rag.release_authority',
            fromlist=['ValidationDatabaseIdentity'],
        ).ValidationDatabaseIdentity('validation', 61),
    )
    return engine, target, admin


def test_release_review_domain_registry_operation_target_and_context_are_exact(
    tmp_path: Path,
) -> None:
    from backend.app.admin.rag_live_gate import (
        RagReleaseReviewError,
        verify_release_review_envelope,
    )

    target = _target(tmp_path)
    raw = _review_bytes(target, 'release-ledger-init')
    verified = verify_release_review_envelope(
        raw,
        expected_operation='release-ledger-init',
        expected_target=target,
        expected_context=None,
        review_secret=_REVIEW_KEY,
        review_key_id=_KEY_ID,
        implementation_plan_reference_hmac=_PLAN_HMAC,
        review_key_registry={_KEY_ID: _review_verifier()},
    )
    assert verified.operation == 'release-ledger-init'
    assert verified.rebootstrap_reason_hmac is None
    with pytest.raises(RagReleaseReviewError):
        verify_release_review_envelope(
            _review_bytes(
                target,
                'release-ledger-init',
                domain=b'paraworks:provider-safety-admin-review:v1\x00',
            ),
            expected_operation='release-ledger-init',
            expected_target=target,
            expected_context=None,
            review_secret=_REVIEW_KEY,
            review_key_id=_KEY_ID,
            implementation_plan_reference_hmac=_PLAN_HMAC,
            review_key_registry={_KEY_ID: _review_verifier()},
        )
    with pytest.raises(RagReleaseReviewError, match='registry'):
        verify_release_review_envelope(
            raw,
            expected_operation='release-ledger-init',
            expected_target=target,
            expected_context=None,
            review_secret=_REVIEW_KEY,
            review_key_id=_KEY_ID,
            implementation_plan_reference_hmac=_PLAN_HMAC,
            review_key_registry={},
        )
    import backend.app.admin.rag_live_gate as live_gate

    assert not any('sign' in name.lower() for name in live_gate.__all__)


def test_init_review_rejects_reason_and_rebootstrap_requires_reason_and_context(
    tmp_path: Path,
) -> None:
    from backend.app.admin.rag_live_gate import (
        RagReleaseReviewError,
        verify_release_review_envelope,
    )

    target = _target(tmp_path)
    common = {
        'expected_target': target,
        'review_secret': _REVIEW_KEY,
        'review_key_id': _KEY_ID,
        'implementation_plan_reference_hmac': _PLAN_HMAC,
        'review_key_registry': {_KEY_ID: _review_verifier()},
    }
    with pytest.raises(RagReleaseReviewError, match='reason'):
        verify_release_review_envelope(
            _review_bytes(target, 'release-ledger-init', reason_hmac='3' * 64),
            expected_operation='release-ledger-init',
            expected_context=None,
            **common,
        )
    context = {'marker_state': 'generation_mismatch', 'snapshot_hmac': '4' * 64}
    with pytest.raises(RagReleaseReviewError, match='reason'):
        verify_release_review_envelope(
            _review_bytes(
                target,
                'release-ledger-rebootstrap',
                expected_context=context,
            ),
            expected_operation='release-ledger-rebootstrap',
            expected_context=context,
            **common,
        )
    with pytest.raises(RagReleaseReviewError, match='context'):
        verify_release_review_envelope(
            _review_bytes(
                target,
                'release-ledger-rebootstrap',
                expected_context={'marker_state': 'other', 'snapshot_hmac': '4' * 64},
                reason_hmac='5' * 64,
            ),
            expected_operation='release-ledger-rebootstrap',
            expected_context=context,
            **common,
        )


def test_admin_init_consumes_nonce_once_and_status_is_read_only(tmp_path: Path) -> None:
    from backend.app.admin.rag_live_gate import RagReleaseReviewError

    _engine, target, admin = _admin(tmp_path)
    before = admin.status()
    assert before.authority_present is False
    assert not target.marker_path.exists()
    raw = _review_bytes(target, 'release-ledger-init')
    result = admin.initialize(raw)
    assert result.operation == 'release-ledger-init'
    assert result.generation == 0
    with pytest.raises(RagReleaseReviewError):
        admin.initialize(raw)
    assert admin.status().authority_present is True


def test_reviewed_rebootstrap_and_disaster_require_fresh_exact_context(
    tmp_path: Path,
) -> None:
    from backend.app.admin.rag_live_gate import RagReleaseReviewError
    from backend.app.rag.release_schema import (
        build_rag_release_metadata,
        release_tables,
    )

    engine, target, admin = _admin(tmp_path)
    admin.initialize(_review_bytes(target, 'release-ledger-init'))
    tables = release_tables(build_rag_release_metadata())
    with engine.begin() as connection:
        connection.execute(
            tables.ledgers.update()
            .where(tables.ledgers.c.ledger_epoch == 1)
            .values(generation=1, last_transition_digest='a' * 64)
        )
        context = admin._recovery_context(connection)
    reviewed = _review_bytes(
        target,
        'release-ledger-rebootstrap',
        expected_context=context,
        reason_hmac='b' * 64,
    )
    assert admin.rebootstrap(reviewed).ledger_epoch == 2
    with pytest.raises(RagReleaseReviewError, match='context'):
        admin.rebootstrap(reviewed)

    target.marker_path.write_bytes(b'corrupt-release-marker')
    with engine.connect() as connection:
        disaster_context = admin._recovery_context(connection)
    disaster_review = _review_bytes(
        target,
        'release-ledger-disaster-init',
        expected_context=disaster_context,
        reason_hmac='c' * 64,
    )
    result = admin.disaster_initialize(disaster_review)
    assert (result.ledger_epoch, result.generation) == (1, 0)
    with pytest.raises(RagReleaseReviewError, match='context'):
        admin.disaster_initialize(disaster_review)


def test_cli_exposes_only_four_commands_and_sanitized_aggregate_output(
    tmp_path: Path,
) -> None:
    from backend.app.admin.rag_live_gate import CliOutcome, _run_cli

    _engine, target, admin = _admin(tmp_path)
    status = _run_cli(
        ['status'],
        stdin=io.BytesIO(b'must-not-be-read'),
        service_factory=lambda: admin,
    )
    assert status.exit_code == 0
    assert status.payload == {
        'authority_present': False,
        'generation': None,
        'ledger_epoch': None,
        'ok': True,
        'state': 'absent',
    }
    initialized = _run_cli(
        ['release-ledger-init'],
        stdin=io.BytesIO(_review_bytes(target, 'release-ledger-init')),
        service_factory=lambda: admin,
    )
    assert initialized.exit_code == 0
    rendered = canonical_json_bytes(initialized.payload)
    for forbidden in (
        b'private',
        str(target.marker_path).encode(),
        b'host-one',
        b'release-validation',
        b'111111111111',
        b'222222222222',
    ):
        assert forbidden not in rendered
    assert _run_cli(
        ['preview'], stdin=io.BytesIO(), service_factory=lambda: admin
    ).exit_code == 2

    invalid = json.loads(_review_bytes(target, 'release-ledger-rebootstrap'))
    invalid['signed_payload']['actor_subject_hmac'] = 'not-a-hmac'
    invalid['hmac_sha256'] = hmac.new(
        _REVIEW_KEY,
        b'paraworks:rag-release-admin-review:v1\x00'
        + canonical_json_bytes(invalid['signed_payload']),
        hashlib.sha256,
    ).hexdigest()
    refused = _run_cli(
        ['release-ledger-rebootstrap'],
        stdin=io.BytesIO(canonical_json_bytes(invalid)),
        service_factory=lambda: admin,
    )
    assert refused == CliOutcome(2, {'code': 'review_refused', 'ok': False})


def test_admin_commands_are_provider_and_network_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.agent_runtime.rag_provider_transport import (
        _DirectOpenAIProviderClient,
    )
    from backend.app.rag.embeddings import OpenAIEmbeddingModel

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError('provider/network forbidden')

    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.setattr(_DirectOpenAIProviderClient, 'send', forbidden)
    monkeypatch.setattr(OpenAIEmbeddingModel, 'embed_many', forbidden)
    from backend.app.rag.release_schema import (
        build_rag_release_metadata,
        release_tables,
    )

    engine, target, admin = _admin(tmp_path)
    admin.status()
    admin.initialize(_review_bytes(target, 'release-ledger-init'))
    admin.status()
    tables = release_tables(build_rag_release_metadata())
    with engine.begin() as connection:
        connection.execute(
            tables.ledgers.update()
            .where(tables.ledgers.c.ledger_epoch == 1)
            .values(generation=1, last_transition_digest='a' * 64)
        )
        context = admin._recovery_context(connection)
    admin.rebootstrap(
        _review_bytes(
            target,
            'release-ledger-rebootstrap',
            expected_context=context,
            reason_hmac='b' * 64,
        )
    )
    target.marker_path.write_bytes(b'corrupt-release-marker')
    with engine.connect() as connection:
        context = admin._recovery_context(connection)
    admin.disaster_initialize(
        _review_bytes(
            target,
            'release-ledger-disaster-init',
            expected_context=context,
            reason_hmac='c' * 64,
        )
    )
    admin.status()


def test_release_settings_are_explicit_and_host_identity_is_frozen(tmp_path: Path) -> None:
    values = {
        'paraworks_release_ledger_authority_path': str(
            (tmp_path / 'release.json').absolute()
        ),
        'paraworks_rag_live_validation_host_id': 'host-one',
        'paraworks_release_review_key_path': str((tmp_path / 'review.key').absolute()),
        'paraworks_release_review_key_id': _KEY_ID,
        'paraworks_release_implementation_plan_reference_hmac': _PLAN_HMAC,
    }
    settings = Settings(**values)
    assert settings.paraworks_rag_live_validation_host_id == 'host-one'
    with pytest.raises(ValidationError):
        settings.paraworks_rag_live_validation_host_id = 'host-two'


def test_database_locator_identity_ignores_credentials_and_normalizes_loopback() -> None:
    from backend.app.admin.rag_live_gate import _database_locator_identity

    first = _database_locator_identity(
        'postgresql+psycopg://app:first@localhost/release_test'
    )
    second = _database_locator_identity(
        'postgresql+psycopg://other:second@127.0.0.1:5432/release_test'
    )
    assert first == second
