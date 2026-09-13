"""Authority-owned SQL and exact images for a sealed provider incident."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select, text, update

from backend.app.agent_runtime.durable_file_authority import DurableFileAuthority
from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_provider_schema import _provider_tables

_COMPONENTS = {'answer_generation', 'query_embedding'}
_STATES = {'ready', 'rebind_required', 'blocked_overrun', 'blocked_remediation'}
_TRANSITIONS = {
    'rebind_required': ('ready', 'rebind_required'),
    'rebind': ('rebind_required', 'ready'),
    'reset': ({'blocked_overrun', 'blocked_remediation'}, 'ready'),
    'block_overrun': ('ready', 'blocked_overrun'),
    'block_remediation': ('ready', 'blocked_remediation'),
}

FrozenRow = tuple[tuple[str, object], ...]
FrozenRows = tuple[FrozenRow, ...]
FrozenImage = tuple[FrozenRows, FrozenRows, FrozenRows]


def _image(connection, tables) -> tuple[list[dict[str, object]], ...]:
    return tuple(
        [
            dict(row)
            for row in connection.execute(
                select(table).order_by(table.c.id).with_for_update()
            ).mappings()
        ]
        for table in tables
    )


def _comparable(images):
    # SQLite loses timezone metadata; only trusted native DB clocks normalize.
    return tuple(
        [
            {
                key: value.replace(tzinfo=UTC)
                if type(value) is datetime and value.tzinfo is None
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
        for rows in images
    )


def _freeze(images) -> FrozenImage:
    return tuple(tuple(tuple(row.items()) for row in rows) for rows in images)  # type: ignore[return-value]


def _thaw(images: FrozenImage) -> tuple[list[dict[str, object]], ...]:
    return tuple([dict(row) for row in rows] for rows in images)


def _same(left: FrozenImage, right: FrozenImage) -> bool:
    return _comparable(_thaw(left)) == _comparable(_thaw(right))


def _serializable(value: object) -> object:
    if type(value) is datetime:
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return {'datetime': normalized.astimezone(UTC).isoformat()}
    if type(value) is Decimal:
        return {'decimal': str(value)}
    if type(value) is bytes:
        return {'bytes': value.hex()}
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TypeError('provider incident image contains a non-native value')


def _image_bytes(images) -> bytes:
    normalized = [
        [{key: _serializable(value) for key, value in row.items()} for row in rows]
        for rows in _comparable(images)
    ]
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')


def _prepared_image_hmac(image: bytes, secret: bytes) -> str:
    return hmac.new(
        secret,
        b'paraworks:provider-incident-history-image:v1\x00' + image,
        hashlib.sha256,
    ).hexdigest()


def _lower_hmac(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _clock(value: object) -> datetime | None:
    if type(value) is not datetime:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _validate_history(images) -> None:
    """Validate the complete stored chain in physical identity order."""
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    authorities, readiness, history = images
    if len(authorities) != 1 or len(readiness) < 2 or not history:
        raise RagProviderSafetyError('provider incident history is inconsistent')
    authority = authorities[0]
    bootstrap = history[0]
    bootstrap_nullable = (
        'readiness_id',
        'prior_state',
        'new_state',
        'prior_state_version',
        'new_state_version',
        'prior_family_safety_generation',
        'new_family_safety_generation',
        'actor_subject_hmac',
        'agent_run_id',
    )
    if (
        authority['id'] != 1
        or any(
            type(row['id']) is not int or row['id'] <= 0
            for row in (*readiness, *history)
        )
        or [row['global_safety_generation'] for row in history]
        != list(range(authority['global_safety_generation'] + 1))
        or any(row['authority_id'] != 1 for row in readiness)
        or any(row['authority_id'] != 1 for row in history)
        or bootstrap['transition_kind'] != 'bootstrap'
        or any(bootstrap[key] is not None for key in bootstrap_nullable)
        or bootstrap['created_at'] is None
        or history[-1]['envelope_digest'] != authority['envelope_digest']
        or len({row['envelope_digest'] for row in history}) != len(history)
        or {row['component'] for row in readiness} != _COMPONENTS
    ):
        raise RagProviderSafetyError('provider incident history is inconsistent')
    if not _lower_hmac(bootstrap['reviewed_transition_reference_hmac']) or any(
        not _lower_hmac(row['envelope_digest'])
        or not _lower_hmac(row['reviewed_transition_reference_hmac'])
        or (
            row['actor_subject_hmac'] is not None
            and not _lower_hmac(row['actor_subject_hmac'])
        )
        for row in history
    ):
        raise RagProviderSafetyError('provider incident history is inconsistent')

    by_id = {row['id']: row for row in readiness}
    latest: dict[int, dict[str, object]] = {}
    last_clock = _clock(bootstrap['created_at'])
    for row in history[1:]:
        key = row['readiness_id']
        current_clock = _clock(row['created_at'])
        previous = latest.get(key)
        prior = (
            row['prior_state'],
            row['prior_state_version'],
            row['prior_family_safety_generation'],
        )
        prior_absent = prior == (None, None, None)
        expected_prior = (
            (None, None, None)
            if prior_absent
            else (
                ('ready', 1, 0)
                if previous is None
                else (
                    previous['new_state'],
                    previous['new_state_version'],
                    previous['new_family_safety_generation'],
                )
            )
        )
        kind = row['transition_kind']
        matrix = _TRANSITIONS.get(kind)
        valid_matrix = (
            kind == 'supersession'
            and prior_absent
            and row['new_state'] == 'ready'
            and row['new_state_version'] == 1
        ) or (
            matrix is not None
            and not prior_absent
            and (
                row['prior_state'] in matrix[0]
                if type(matrix[0]) is set
                else row['prior_state'] == matrix[0]
            )
            and row['new_state'] == matrix[1]
        )
        blocker = kind in {'block_overrun', 'block_remediation'}
        reviewed = kind in {'rebind_required', 'rebind', 'reset', 'supersession'}
        prior_version = row['prior_state_version']
        expected_version = (
            1
            if prior_absent
            else prior_version + 1
            if type(prior_version) is int
            else None
        )
        if (
            type(key) is not int
            or key not in by_id
            or by_id[key]['component'] not in _COMPONENTS
            or row['new_state'] not in _STATES
            or row['new_family_safety_generation'] != row['global_safety_generation']
            or prior != expected_prior
            or (kind == 'supersession') is not prior_absent
            or row['new_state_version'] != expected_version
            or not valid_matrix
            or blocker
            and (row['actor_subject_hmac'] is None or row['agent_run_id'] is None)
            or reviewed
            and (row['actor_subject_hmac'] is None or row['agent_run_id'] is not None)
            or current_clock is None
            or last_clock is None
            or current_clock < last_clock
        ):
            raise RagProviderSafetyError('provider incident history is inconsistent')
        latest[key] = row
        last_clock = current_clock

    for key, row in by_id.items():
        last = latest.get(key)
        expected = (
            ('ready', 1, 0, bootstrap['reviewed_transition_reference_hmac'])
            if last is None
            else (
                last['new_state'],
                last['new_state_version'],
                last['new_family_safety_generation'],
                last['reviewed_transition_reference_hmac'],
            )
        )
        if (
            row['state'],
            row['state_version'],
            row['family_safety_generation'],
            row['reviewed_gate_reference_hmac'],
        ) != expected:
            raise RagProviderSafetyError('provider incident history is inconsistent')


def _validate_digest_chain(images, body, service) -> None:
    """Reverse every reconstructible signed envelope and authenticate its digest."""
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    _authorities, readiness, history = images
    envelope = json.loads(canonical_json_bytes(dict(body['_envelope'])).decode('utf-8'))
    rows_by_readiness: dict[int, list[dict[str, object]]] = {}
    for row in history[1:]:
        rows_by_readiness.setdefault(row['readiness_id'], []).append(row)
    readiness_by_id = {row['id']: row for row in readiness}
    bootstrap_reference = history[0]['reviewed_transition_reference_hmac']
    for row in reversed(history[1:]):
        raw = canonical_json_bytes(envelope)
        digest = hashlib.sha256(
            b'paraworks:provider-safety-envelope-file:v1\x00' + raw
        ).hexdigest()
        if digest != row['envelope_digest']:
            raise RagProviderSafetyError('provider incident history digest differs')
        # Rebind overwrites policy material that the physical v1 history does not
        # retain; supersession changes the roster. Their current-side digest is
        # authenticated above and their semantic chain remains fully checked.
        if row['transition_kind'] in {'rebind', 'supersession'}:
            break
        stored = readiness_by_id[row['readiness_id']]
        signed = envelope['signed_payload']
        envelope_body = signed['body']
        target = next(
            record
            for record in envelope_body['family_records']
            if all(
                record[key] == stored[key]
                for key in (
                    'component',
                    'model',
                    'provider',
                    'reasoning_or_config_identity',
                )
            )
        )
        target['state'] = row['prior_state']
        target['state_version'] = row['prior_state_version']
        target['family_safety_generation'] = row['prior_family_safety_generation']
        predecessors = [
            item
            for item in rows_by_readiness[row['readiness_id']]
            if item['global_safety_generation'] < row['global_safety_generation']
        ]
        target['reviewed_transition_reference_hmac'] = (
            predecessors[-1]['reviewed_transition_reference_hmac']
            if predecessors
            else bootstrap_reference
        )
        if row['transition_kind'] in {'block_overrun', 'block_remediation'} and not any(
            item['transition_kind'] in {'block_overrun', 'block_remediation'}
            for item in predecessors
        ):
            target['first_blocker_agent_run_hmac'] = None
            target['first_blocker_category'] = None
            target['first_blocker_observed_at'] = None
        envelope_body['global_safety_generation'] = row['global_safety_generation'] - 1
        envelope['hmac_sha256'] = service._signature(signed)
    else:
        raw = canonical_json_bytes(envelope)
        digest = hashlib.sha256(
            b'paraworks:provider-safety-envelope-file:v1\x00' + raw
        ).hexdigest()
        if digest != history[0]['envelope_digest']:
            raise RagProviderSafetyError('provider incident history digest differs')


def _capture_prepared_image(connection, body, service) -> tuple[bytes, str]:
    """Bind the complete private SQL image when the incident capability is made."""
    images = _image(connection, _provider_tables())
    _validate_history(images)
    _validate_digest_chain(images, body, service)
    image = _image_bytes(images)
    return image, _prepared_image_hmac(image, service._secret)


def _raw_authority_bytes(path: Path) -> bytes:
    """Read an already locked authority without service or authority dispatch."""
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_nlink != 1
        ):
            raise OSError('provider authority file identity changed')
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        return b''.join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _ProviderIncidentOperation:
    connection: Any
    transaction: Any
    tables: tuple[Any, Any, Any]
    statements: tuple[Any, Any, Any]
    authority: DurableFileAuthority
    latch_before: bytes
    latch_after: bytes
    provider_before: FrozenImage
    provider_prospective: FrozenImage
    target_readiness_id: int


def _prepare_incident_operation(connection, prepared) -> _ProviderIncidentOperation:
    """Finish every service callback before the external-first mutation begins."""
    from backend.app.agent_runtime.rag_provider_safety import (
        _RELEASE_INCIDENT_PLAN_SEAL,
        RagProviderSafetyError,
        RagProviderSafetyService,
        _PreparedReleaseProviderIncident,
    )

    service = prepared.service
    if (
        type(prepared) is not _PreparedReleaseProviderIncident
        or prepared._seal is not _RELEASE_INCIDENT_PLAN_SEAL
        or type(service) is not RagProviderSafetyService
    ):
        raise RagProviderSafetyError('provider incident plan is invalid')
    old = service._read_unlocked()
    authority_before, rows_before = service._match_db_whole_set(
        connection, old, for_update=True
    )
    record = service._active_record(old, prepared.component)
    readiness_before = next(
        row
        for row in rows_before
        if row['active'] and row['component'] == prepared.component
    )
    if (
        old['envelope_digest'] != prepared.old_envelope_digest
        or old['global_safety_generation'] != prepared.old_global_generation
        or record['state'] != 'ready'
        or record['state_version'] != prepared.old_state_version
    ):
        raise RagProviderSafetyError('provider incident plan CAS failed')
    prospective = service._validate_envelope(
        dict(prepared.new_envelope),
        raw=canonical_json_bytes(dict(prepared.new_envelope)),
    )
    tables = _provider_tables()
    authority_table, readiness_table, history_table = tables
    before = _image(connection, tables)
    _validate_history(before)
    _validate_digest_chain(before, old, service)
    before_image = _image_bytes(before)
    if (
        not hmac.compare_digest(
            prepared.provider_before_image_hmac,
            _prepared_image_hmac(prepared.provider_before_image, service._secret),
        )
        or before_image != prepared.provider_before_image
    ):
        raise RagProviderSafetyError('provider incident prepared history changed')
    authority = before[0][0]
    targets = [
        row
        for row in before[1]
        if row['active'] and row['component'] == prepared.component
    ]
    if len(targets) != 1 or dict(authority_before) != authority:
        raise RagProviderSafetyError('provider incident roster is inconsistent')
    target = targets[0]
    if dict(readiness_before) != target:
        raise RagProviderSafetyError('provider incident readiness differs')
    cost = prepared.cost_usd
    if (
        type(cost) is not Decimal
        or not cost.is_finite()
        or cost < 0
        or cost >= Decimal('1e18')
        or cost.as_tuple().exponent < -6
    ):
        raise RagProviderSafetyError('provider incident cost cannot be stored exactly')
    if connection.dialect.name == 'postgresql':
        history_id = connection.execute(
            text(
                "SELECT nextval(pg_get_serial_sequence('rag_provider_safety_transitions', 'id'))"
            )
        ).scalar_one()
    else:
        history_id = max(row['id'] for row in before[2]) + 1
    if (
        type(history_id) is not int
        or history_id <= max(row['id'] for row in before[2])
        or any(row['id'] == history_id for row in before[2])
    ):
        raise RagProviderSafetyError('provider incident history identity is invalid')
    now = prepared.observed_at
    authority_values = {
        'global_safety_generation': prospective['global_safety_generation'],
        'envelope_digest': prospective['envelope_digest'],
        'updated_at': now,
    }
    readiness_values = {
        'state': prepared.state,
        'state_version': prepared.old_state_version + 1,
        'family_safety_generation': prospective['global_safety_generation'],
        'reviewed_gate_reference_hmac': prepared.reviewed_reference,
        'updated_at': now,
    }
    if target['overrun_agent_run_id'] is None:
        readiness_values.update(
            overrun_agent_run_id=prepared.agent_run_id,
            overrun_input_tokens=prepared.input_tokens,
            overrun_output_tokens=prepared.output_tokens,
            overrun_cost_usd=cost,
            overrun_observed_at=now,
        )
    transition = {
        'id': history_id,
        'authority_id': 1,
        'readiness_id': target['id'],
        'global_safety_generation': prospective['global_safety_generation'],
        'transition_kind': 'block_overrun'
        if prepared.state == 'blocked_overrun'
        else 'block_remediation',
        'prior_state': target['state'],
        'new_state': prepared.state,
        'prior_state_version': target['state_version'],
        'new_state_version': prepared.old_state_version + 1,
        'prior_family_safety_generation': target['family_safety_generation'],
        'new_family_safety_generation': prospective['global_safety_generation'],
        'envelope_digest': prospective['envelope_digest'],
        'reviewed_transition_reference_hmac': prepared.reviewed_reference,
        'actor_subject_hmac': prepared.actor_subject_hmac,
        'agent_run_id': prepared.agent_run_id,
        'created_at': now,
    }
    after = (
        [{**authority, **authority_values}],
        [
            {**row, **readiness_values} if row['id'] == target['id'] else dict(row)
            for row in before[1]
        ],
        [*before[2], transition],
    )
    _validate_history(after)
    _validate_digest_chain(after, prospective, service)
    service._match_database_rows(prospective, after[0][0], after[1])
    latch_before = canonical_json_bytes(dict(old['_envelope']))
    latch_after = canonical_json_bytes(dict(prepared.new_envelope))
    if (
        _raw_authority_bytes(service._latch_path) != latch_before
        or hashlib.sha256(
            b'paraworks:provider-safety-envelope-file:v1\x00' + latch_after
        ).hexdigest()
        != prepared.new_envelope_digest
    ):
        raise RagProviderSafetyError('provider incident envelope differs')
    transaction = connection.get_transaction()
    if transaction is None or not transaction.is_active:
        raise RagProviderSafetyError('provider incident transaction changed')
    statements = (
        update(authority_table)
        .where(
            authority_table.c.id == 1,
            authority_table.c.global_safety_generation
            == old['global_safety_generation'],
            authority_table.c.envelope_digest == old['envelope_digest'],
        )
        .values(**authority_values),
        update(readiness_table)
        .where(
            readiness_table.c.id == target['id'],
            readiness_table.c.active.is_(True),
            readiness_table.c.state_version == target['state_version'],
            readiness_table.c.family_safety_generation
            == target['family_safety_generation'],
        )
        .values(**readiness_values),
        insert(history_table).values(**transition),
    )
    # Sealed publication does not retain caller-installed instance dispatch
    # seams. Every such callback has completed above, while rollback and an
    # unchanged latch are still possible; later inspections fall back to the
    # exact class implementations.
    service_attributes = vars(service)
    for name in (
        '_apply_release_incident',
        '_file_digest_bytes',
        '_match_database_rows',
        '_match_db_whole_set',
        '_read_unlocked',
        '_signature',
        '_validate_envelope',
    ):
        service_attributes.pop(name, None)
    return _ProviderIncidentOperation(
        connection=connection,
        transaction=transaction,
        tables=tables,
        statements=statements,
        authority=service._authority,
        latch_before=latch_before,
        latch_after=latch_after,
        provider_before=_freeze(before),
        provider_prospective=_freeze(after),
        target_readiness_id=target['id'],
    )


def _require_transaction(operation: _ProviderIncidentOperation) -> None:
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    connection = operation.connection
    if (
        connection.closed
        or connection.get_transaction() is not operation.transaction
        or not operation.transaction.is_active
    ):
        raise RagProviderSafetyError('provider incident transaction changed')


def _execute_incident_operation(operation: _ProviderIncidentOperation):
    """Apply external-first, then use only private SQL and raw exact checks."""
    from backend.app.agent_runtime.rag_provider_safety import (
        _RELEASE_INCIDENT_PLAN_SEAL,
        RagProviderSafetyError,
        _AppliedReleaseProviderIncident,
    )

    _require_transaction(operation)
    if _raw_authority_bytes(operation.authority.path) != operation.latch_before:
        raise RagProviderSafetyError('provider incident before-image changed')
    envelope = json.loads(operation.latch_after.decode('utf-8'))
    DurableFileAuthority._replace_unlocked(operation.authority, envelope)
    if _raw_authority_bytes(operation.authority.path) != operation.latch_after:
        raise RagProviderSafetyError('provider incident envelope differs')
    try:
        _require_transaction(operation)
        current = _image(operation.connection, operation.tables)
        if _freeze(current) != operation.provider_before:
            raise RagProviderSafetyError('provider incident before-image changed')
        for statement in operation.statements:
            _require_transaction(operation)
            if operation.connection.execute(statement).rowcount != 1:
                raise RagProviderSafetyError('provider incident SQL CAS failed')
        _require_transaction(operation)
        actual = _image(operation.connection, operation.tables)
        if not _same(_freeze(actual), operation.provider_prospective):
            raise RagProviderSafetyError('provider incident SQL image differs')
        _require_transaction(operation)
    except Exception:
        operation.connection.rollback()
        raise RagProviderSafetyError(
            'external provider incident persisted but DB transition failed'
        ) from None
    return _AppliedReleaseProviderIncident(
        latch_before=operation.latch_before,
        latch_after=operation.latch_after,
        provider_before=operation.provider_before,
        provider_prospective=operation.provider_prospective,
        provider_actual=_freeze(actual),
        target_readiness_id=operation.target_readiness_id,
        _seal=_RELEASE_INCIDENT_PLAN_SEAL,
    )


def _incident_write(connection, prepared, old, new):
    """Compatibility refusal: incident execution is now an owned operation."""
    del connection, prepared, old, new
    raise RuntimeError('provider incident writes require the owned operation')
