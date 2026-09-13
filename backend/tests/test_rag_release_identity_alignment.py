from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, event, select, update

from backend.app.rag.release_authority import RagReleaseAuthorityError
from backend.app.rag.release_schema import build_rag_release_metadata, release_tables
from backend.tests.test_rag_live_gate_preview import fingerprint, utf8
from backend.tests.test_rag_release_authority import (
    _SECRET,
    _deterministic_non_product_database_seam,  # noqa: F401
    _identity,
    _review_args,
    _service,
)
from backend.tests.test_rag_release_reviewer import encoded


@pytest.mark.parametrize(
    'environment,host',
    [
        ('release-validation', 'immutable-host-01'),
        ('회사-é', '개발-e\u0301'),
        ('같은-이름', '같은-이름'),
    ],
)
def test_real_authority_uses_exact_structured_identity_and_shared_key_registry(
    tmp_path, environment, host
):
    authority, marker, _provider = _service(
        tmp_path, designated_environment_id=environment, designated_host_id=host
    )
    engine = create_engine('sqlite://')
    with engine.connect() as connection:
        initialized = authority.initialize(
            connection, database_identity=_identity(), **_review_args()
        )
        inspected = authority.inspect(connection, database_identity=_identity())
    assert initialized == inspected
    for field, value in (
        ('designated_environment_id', environment),
        ('designated_host_id', host),
    ):
        expected = fingerprint(
            {field + '_bytes': utf8(value)},
            'rag-live-' + field.replace('_', '-') + ':v1',
            secret=_SECRET,
        )
        assert getattr(inspected, field + '_hmac') == expected
    assert inspected.designated_environment_id_hmac != inspected.designated_host_id_hmac
    expected_verifier = hmac.new(
        _SECRET, b'paraworks:auto-review-key-material-verifier:v1', hashlib.sha256
    ).hexdigest()
    assert inspected.fingerprint_key_material_verifier == expected_verifier
    payload = json.loads(marker.read_bytes())['signed_payload']
    assert payload['fingerprint_key_material_verifier'] == expected_verifier


def test_same_key_release_and_provider_material_verifier_are_equal(tmp_path):
    from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier

    authority, _marker, _provider = _service(tmp_path)
    assert authority._key_verifier() == fingerprint_key_material_verifier(
        _SECRET.decode()
    )
    assert (
        authority._key_verifier()
        == hmac.new(
            _SECRET, b'paraworks:auto-review-key-material-verifier:v1', hashlib.sha256
        ).hexdigest()
    )


@pytest.mark.parametrize(
    'attack',
    [
        'legacy_environment',
        'legacy_host',
        'legacy_key',
        'field_swap',
        'wrong_field',
        'wrong_domain',
        'rotated_key',
    ],
)
def test_legacy_or_cross_domain_marker_and_matching_db_are_refused_without_mutation(
    tmp_path, attack
):
    authority, marker, _provider = _service(tmp_path)
    engine = create_engine('sqlite://')
    with engine.connect() as connection:
        authority.initialize(
            connection, database_identity=_identity(), **_review_args()
        )
    envelope = json.loads(marker.read_bytes())
    signed = envelope['signed_payload']
    body = signed['body']
    if attack in ('legacy_environment', 'legacy_host'):
        name, text = (
            ('designated_environment_id', 'release-validation')
            if attack == 'legacy_environment'
            else ('designated_host_id', 'immutable-host-01')
        )
        body[name + '_hmac'] = fingerprint(
            text, 'rag-live-' + name.replace('_', '-') + ':v1', secret=_SECRET
        )
    if attack == 'legacy_key':
        signed['fingerprint_key_material_verifier'] = hmac.new(
            _SECRET,
            b'paraworks:rag-release-key-material-verifier:v1\x00',
            hashlib.sha256,
        ).hexdigest()
    if attack == 'field_swap':
        body['designated_environment_id_hmac'], body['designated_host_id_hmac'] = (
            body['designated_host_id_hmac'],
            body['designated_environment_id_hmac'],
        )
    if attack == 'wrong_field':
        body['designated_environment_id_hmac'] = fingerprint(
            {'designated_host_id_bytes': utf8('release-validation')},
            'rag-live-designated-environment-id:v1',
            secret=_SECRET,
        )
    if attack == 'wrong_domain':
        body['designated_environment_id_hmac'] = fingerprint(
            {'designated_environment_id_bytes': utf8('release-validation')},
            'rag-live-designated-host-id:v1',
            secret=_SECRET,
        )
    if attack == 'rotated_key':
        signed['fingerprint_key_material_verifier'] = hmac.new(
            b'different-key-material-at-least-32-bytes',
            b'paraworks:auto-review-key-material-verifier:v1',
            hashlib.sha256,
        ).hexdigest()
    envelope['hmac_sha256'] = fingerprint(
        signed, 'rag-release-ledger-marker-envelope:v1', secret=_SECRET
    )
    raw = encoded(envelope)
    marker.write_bytes(raw)
    table = release_tables(build_rag_release_metadata()).ledgers
    digest = hashlib.sha256(
        b'paraworks:release-ledger-marker-file:v1\x00' + raw
    ).hexdigest()
    with engine.begin() as connection:
        connection.execute(
            update(table).values(
                designated_environment_id_hmac=body['designated_environment_id_hmac'],
                designated_host_id_hmac=body['designated_host_id_hmac'],
                fingerprint_key_material_verifier=signed[
                    'fingerprint_key_material_verifier'
                ],
                marker_file_digest=digest,
            )
        )
        before = dict(connection.execute(select(table)).mappings().one())
    writes = []

    def capture(_c, _cu, sql, *_):
        if sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            writes.append(sql)

    event.listen(engine, 'before_cursor_execute', capture)
    with engine.connect() as connection, pytest.raises(RagReleaseAuthorityError):
        authority.inspect(connection, database_identity=_identity())
    assert writes == [] and marker.read_bytes() == raw
    with engine.connect() as connection:
        assert dict(connection.execute(select(table)).mappings().one()) == before


def test_unicode_recomposition_does_not_alias_an_existing_authority(tmp_path):
    first, marker, _provider = _service(tmp_path, designated_host_id='host-e\u0301')
    engine = create_engine('sqlite://')
    with engine.connect() as connection:
        snapshot = first.initialize(
            connection, database_identity=_identity(), **_review_args()
        )
    before = marker.read_bytes()
    different, _marker, _provider = _service(tmp_path, designated_host_id='host-é')
    with engine.connect() as connection, pytest.raises(RagReleaseAuthorityError):
        different.inspect(connection, database_identity=_identity())
    assert marker.read_bytes() == before
    stale = replace(
        snapshot,
        designated_host_id_hmac=fingerprint(
            'host-e\u0301', 'rag-live-designated-host-id:v1', secret=_SECRET
        ),
    )
    with engine.connect() as connection, pytest.raises(RagReleaseAuthorityError):
        first._inspect_locked(connection, stale, database_identity=_identity())


@pytest.mark.parametrize(
    'field,value',
    [
        ('designated_environment_id', ''),
        ('designated_environment_id', True),
        ('designated_environment_id', b'bytes'),
        ('designated_host_id', None),
        ('designated_host_id', ' trailing '),
        ('designated_host_id', 'nul\x00host'),
        ('identity_secret', bytearray(b'x' * 32)),
        ('identity_secret', b'\xff' * 32),
    ],
)
def test_identity_inputs_refuse_without_creating_release_marker(tmp_path, field, value):
    with pytest.raises(ValueError):
        _service(tmp_path, **{field: value})
    assert not (tmp_path / 'release' / 'ledger.json').exists()
