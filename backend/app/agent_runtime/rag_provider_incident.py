"""Owned, complete SQL images for the sealed Task22→Task23 incident boundary."""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from sqlalchemy import insert, select, text, update

from backend.app.agent_runtime.rag_provider_schema import _provider_tables


def _image(connection, tables):
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


def _validate_history(images):
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    authorities, readiness, history = images
    if len(authorities) != 1 or not history:
        raise RagProviderSafetyError('provider incident history is inconsistent')
    authority = authorities[0]
    ordered = sorted(history, key=lambda row: row['global_safety_generation'])
    bootstrap = ordered[0]
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
        [row['global_safety_generation'] for row in ordered]
        != list(range(authority['global_safety_generation'] + 1))
        or any(row['authority_id'] != 1 for row in ordered)
        or bootstrap['transition_kind'] != 'bootstrap'
        or any(bootstrap[key] is not None for key in bootstrap_nullable)
        or ordered[-1]['envelope_digest'] != authority['envelope_digest']
    ):
        raise RagProviderSafetyError('provider incident history is inconsistent')

    def lower_hmac(value):
        return (
            type(value) is str
            and len(value) == 64
            and all(character in '0123456789abcdef' for character in value)
        )

    if not lower_hmac(bootstrap['reviewed_transition_reference_hmac']) or any(
        not lower_hmac(row['envelope_digest'])
        or not lower_hmac(row['reviewed_transition_reference_hmac'])
        or (
            row['actor_subject_hmac'] is not None
            and not lower_hmac(row['actor_subject_hmac'])
        )
        for row in ordered
    ):
        raise RagProviderSafetyError('provider incident history is inconsistent')
    by_id = {row['id']: row for row in readiness}
    latest = {}
    for row in ordered[1:]:
        key = row['readiness_id']
        previous = latest.get(key)
        prior = (
            row['prior_state'],
            row['prior_state_version'],
            row['prior_family_safety_generation'],
        )
        prior_absent = prior == (None, None, None)
        if previous is None:
            expected_prior = (None, None, None) if prior_absent else ('ready', 1, 0)
        else:
            expected_prior = (
                previous['new_state'],
                previous['new_state_version'],
                previous['new_family_safety_generation'],
            )
        expected_version = (
            1
            if prior_absent
            else (
                row['prior_state_version'] + 1
                if type(row['prior_state_version']) is int
                else None
            )
        )
        if (
            key not in by_id
            or row['new_family_safety_generation'] != row['global_safety_generation']
            or row['transition_kind'] == 'bootstrap'
            or prior != expected_prior
            or (row['transition_kind'] == 'supersession') is not prior_absent
            or row['new_state_version'] != expected_version
        ):
            raise RagProviderSafetyError('provider incident history is inconsistent')
        latest[key] = row
    for key, row in by_id.items():
        last = latest.get(key)
        expected = (
            (
                'ready',
                1,
                0,
                bootstrap['reviewed_transition_reference_hmac'],
            )
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


def _incident_write(connection, prepared, old, new):
    """Capture/validate before DML, return only schema-owned deterministic writes."""
    from backend.app.agent_runtime.rag_provider_safety import RagProviderSafetyError

    tables = _provider_tables()
    authority_table, readiness_table, history_table = tables
    before = _image(connection, tables)
    _validate_history(before)
    transaction = connection.get_transaction()
    authority = before[0][0]
    targets = [
        row
        for row in before[1]
        if row['active'] and row['component'] == prepared.component
    ]
    if len(targets) != 1:
        raise RagProviderSafetyError('provider incident roster is inconsistent')
    target = targets[0]
    cost = prepared.cost_usd
    if (
        type(cost) is not Decimal
        or not cost.is_finite()
        or cost < 0
        or cost >= Decimal('1e18')
        or cost.as_tuple().exponent < -6
    ):
        raise RagProviderSafetyError('provider incident cost cannot be stored exactly')
    # Native sequence allocation occurs before the first mutation and preserves
    # PostgreSQL SERIAL ownership (including ordinary admin writers afterwards).
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
        or history_id <= 0
        or any(row['id'] == history_id for row in before[2])
    ):
        raise RagProviderSafetyError('provider incident history identity is invalid')
    now = prepared.observed_at
    authority_values = {
        'global_safety_generation': new['global_safety_generation'],
        'envelope_digest': new['envelope_digest'],
        'updated_at': now,
    }
    readiness_values = {
        'state': prepared.state,
        'state_version': prepared.old_state_version + 1,
        'family_safety_generation': new['global_safety_generation'],
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
        'global_safety_generation': new['global_safety_generation'],
        'transition_kind': 'block_overrun'
        if prepared.state == 'blocked_overrun'
        else 'block_remediation',
        'prior_state': target['state'],
        'new_state': prepared.state,
        'prior_state_version': target['state_version'],
        'new_state_version': prepared.old_state_version + 1,
        'prior_family_safety_generation': target['family_safety_generation'],
        'new_family_safety_generation': new['global_safety_generation'],
        'envelope_digest': new['envelope_digest'],
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
    after[2].sort(key=lambda row: row['id'])
    _validate_history(after)
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

    def require_transaction():
        if (
            connection.closed
            or transaction is None
            or connection.get_transaction() is not transaction
            or not transaction.is_active
        ):
            raise RagProviderSafetyError('provider incident transaction changed')

    def check():
        require_transaction()
        current = _image(connection, tables)
        if _comparable(current) != _comparable(after):
            raise RagProviderSafetyError('provider incident SQL image differs')
        require_transaction()
        return current

    def execute():
        require_transaction()
        if _comparable(_image(connection, tables)) != _comparable(before):
            raise RagProviderSafetyError('provider incident before-image changed')
        for statement in statements:
            require_transaction()
            if connection.execute(statement).rowcount != 1:
                raise RagProviderSafetyError('provider incident SQL CAS failed')
        return check()

    return SimpleNamespace(
        execute=execute, authority_after=after[0][0], readiness_after=after[1]
    )
