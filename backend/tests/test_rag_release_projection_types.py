"""Independent actual-append oracles for Task24's literal type boundary."""

from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, IntEnum
from types import MappingProxyType
from uuid import UUID

import pytest
from sqlalchemy import event

from backend.app.models.agent_runs import AgentRun
from backend.app.models.rag_runtime import AgentRunCostComponent
from backend.app.rag.release_ledger import (
    RagReleaseLedgerError,
    release_runtime_mutation_hmac,
)
from backend.app.rag.release_review import (
    case_claim_manifest_hmac,
    prepare_case_claim_projection,
)
from backend.tests.test_rag_release_case_claim_projection import claim_changes
from backend.tests.test_rag_release_ledger_review_q import (
    _release_test_seam,  # noqa: F401
)
from backend.tests.test_rag_release_ledger_round4 import harness as harness


class StringAlias(str):
    pass


class CopyNormalizedString(str):
    def __deepcopy__(self, _memo):
        return str(self)


class IntegerAlias(int):
    pass


class FloatAlias(float):
    pass


class DecimalAlias(Decimal):
    pass


class DateTimeAlias(datetime):
    pass


class DictAlias(dict):
    pass


def alias(value, variant):
    if type(value) is str:
        if variant == 'subclass':
            return StringAlias(value)
        if variant == 'enum':
            return Enum('TextAlias', {'VALUE': value}, type=str).VALUE
        return bytes.fromhex(value) if len(value) == 64 else value.encode()
    if type(value) is bool:
        return (
            IntegerAlias(value)
            if variant == 'subclass'
            else IntEnum('BoolAlias', {'VALUE': int(value)}).VALUE
            if variant == 'enum'
            else int(value)
        )
    if type(value) is int:
        return (
            IntegerAlias(value)
            if variant == 'subclass'
            else IntEnum('NumberAlias', {'VALUE': value}).VALUE
            if variant == 'enum'
            else bool(value)
            if value in {0, 1}
            else float(value)
        )
    if type(value) is float:
        return FloatAlias(value) if variant == 'subclass' else Decimal(str(value))
    if type(value) is Decimal:
        return DecimalAlias(value) if variant == 'subclass' else float(value)
    if type(value) is datetime:
        return (
            DateTimeAlias.fromisoformat(value.isoformat())
            if variant == 'subclass'
            else value.date()
            if variant == 'enum'
            else value.isoformat()
        )
    if type(value) is dict:
        return (
            DictAlias(value)
            if variant == 'subclass'
            else MappingProxyType(value)
            if variant == 'enum'
            else {
                **value,
                'current_text_hmac': bytes.fromhex(value['current_text_hmac']),
            }
        )
    return False


def no_effects(harness, monkeypatch, action):
    """Inspect side effects even when the old implementation raises SQL errors."""
    from backend.app.rag.release_authority import _RagReleaseBarrierGuard

    kinds = (
        'authorization',
        'case',
        'agent_run',
        'cost_component',
        'dispatch',
        'provider_safety_authority',
        'provider_readiness',
        'quality_report',
    )
    before = {kind: harness.records(kind) for kind in kinds}
    marker = harness.authority.marker_path.read_bytes()
    generation = harness.snapshot.generation
    dml = []
    incidents = []
    original = _RagReleaseBarrierGuard.apply_provider_incident

    def incident(self, value):
        incidents.append(True)
        return original(self, value)

    def record(_connection, _cursor, statement, _params, _context, _many):
        if statement.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            dml.append(statement.split()[0])

    monkeypatch.setattr(_RagReleaseBarrierGuard, 'apply_provider_incident', incident)
    event.listen(harness.engine, 'before_cursor_execute', record)
    error = None
    try:
        try:
            action()
        except Exception as caught:
            error = caught
    finally:
        event.remove(harness.engine, 'before_cursor_execute', record)
    assert {kind: harness.records(kind) for kind in kinds} == before
    assert harness.authority.marker_path.read_bytes() == marker
    assert harness.snapshot.generation == generation
    assert incidents == []
    assert dml == [], 'invalid literal reached DML before refusal'
    assert type(error) is RagReleaseLedgerError, type(error).__name__


def append_changed_literal(harness, monkeypatch, row_index, field, transform):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    with harness.engine.connect() as connection:
        payload, mutations = harness.prepare(
            connection, kind, changes, case=kwargs['case']
        )
        plan = mutations._plans[row_index]
        old = changes[row_index][2][field]
        mutations._plans[row_index] = replace(
            plan, statement=plan.statement.values(**{field: transform(old)})
        )
        # Keep the legitimate original signature: append itself must validate
        # unnormalized literals, independently of any caller-side signer.
        no_effects(
            harness,
            monkeypatch,
            lambda: harness.ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=harness.database_identity,
                approved_case_claim=kwargs['approved_case_claim'],
            ),
        )


@pytest.mark.parametrize(
    'index,field',
    [
        (3, 'authorized_model_config_snapshot_hmac'),
        (4, 'authorized_policy_snapshot_hmac'),
    ],
)
def test_hmac_bytes_alias_is_rejected_by_actual_append_before_any_effect(
    harness, monkeypatch, index, field
):
    append_changed_literal(harness, monkeypatch, index, field, bytes.fromhex)


ROW_FIELDS = [(2, str(column.name)) for column in AgentRun.__table__.c] + [
    (index, str(column.name))
    for index in (3, 4)
    for column in AgentRunCostComponent.__table__.c
]


@pytest.mark.parametrize('index,field', ROW_FIELDS)
@pytest.mark.parametrize('variant', ['coercion', 'subclass', 'enum'])
def test_every_initial_column_rejects_coercive_types_before_sql(
    harness, monkeypatch, index, field, variant
):
    append_changed_literal(
        harness, monkeypatch, index, field, lambda value: alias(value, variant)
    )


@pytest.mark.parametrize(
    'variant', ['key_subclass', 'value_bytes', 'value_enum', 'dict_subclass']
)
def test_metadata_keys_and_nested_values_require_exact_json_types(
    harness, monkeypatch, variant
):
    def changed(value):
        result = deepcopy(value)
        if variant == 'key_subclass':
            result[StringAlias('current_text_hmac')] = result.pop('current_text_hmac')
        elif variant == 'value_bytes':
            result['current_text_hmac'] = bytes.fromhex(result['current_text_hmac'])
        elif variant == 'value_enum':
            result['surface'] = alias(result['surface'], 'enum')
        else:
            result = DictAlias(result)
        return result

    append_changed_literal(harness, monkeypatch, 2, 'metadata', changed)


@pytest.mark.parametrize(
    'field,transform',
    [
        ('manifest_ordinal', bool),
        ('embedding_reserved_cost_usd', float),
        ('ledger_uuid', UUID),
        ('case_id_hmac', StringAlias),
    ],
)
def test_case_identity_and_reserve_types_cannot_use_python_equality_aliases(
    harness, monkeypatch, field, transform
):
    append_changed_literal(harness, monkeypatch, 1, field, transform)


def test_nested_json_float_and_object_have_distinct_signed_bytes(harness, monkeypatch):
    _, changes, _ = claim_changes(harness, monkeypatch)
    parent = changes[2][2]
    numeric = {**parent, 'metadata': {'number': 0.001}}
    lookalike = {**parent, 'metadata': {'number': {'binary64_hex': (0.001).hex()}}}
    assert release_runtime_mutation_hmac(
        'agent_run', None, numeric, identity_secret=harness.secret
    ) != release_runtime_mutation_hmac(
        'agent_run', None, lookalike, identity_secret=harness.secret
    )


@pytest.mark.parametrize(
    'index,field',
    [
        (3, 'authorized_model_config_snapshot_hmac'),
        (4, 'authorized_policy_snapshot_hmac'),
        (2, 'input_tokens'),
    ],
)
def test_runtime_signer_rejects_alias_instead_of_producing_ambiguous_hmac(
    harness, monkeypatch, index, field
):
    _, changes, _ = claim_changes(harness, monkeypatch)
    row = deepcopy(changes[index][2])
    row[field] = alias(row[field], 'coercion')
    with pytest.raises(RagReleaseLedgerError):
        release_runtime_mutation_hmac(
            'agent_run' if index == 2 else 'cost_component',
            None,
            row,
            identity_secret=harness.secret,
        )


MANIFEST_FIELDS = (
    [
        ('manifest', field)
        for field in (
            'contract_version',
            'fixture_manifest_version',
            'runtime_contract_version',
            'assembly_version',
        )
    ]
    + [
        ('case', field)
        for field in (
            'ordinal',
            'case_id_hmac',
            'surface',
            'configured_backend',
            'current_text_hmac',
            'retrieval_query_hmac',
            'security_scope_fingerprint',
        )
    ]
    + [
        ('component', field)
        for field in (
            'reserved_input_tokens',
            'reserved_output_tokens',
            'reserved_cost_usd',
        )
    ]
    + [
        ('policy', field)
        for field in (
            'component',
            'provider',
            'model',
            'reasoning_or_config_identity',
            'authorized_model_config_version',
            'authorized_model_config_snapshot_hmac',
            'authorized_cost_policy_version',
            'authorized_token_estimator_version',
            'fingerprint_key_version',
            'fingerprint_key_material_verifier',
            'authorized_policy_snapshot_hmac',
        )
    ]
)


@pytest.mark.parametrize('level,field', MANIFEST_FIELDS)
@pytest.mark.parametrize('variant', ['coercion', 'subclass', 'enum'])
def test_every_manifest_scalar_type_rejects_before_canonicalization(
    harness, monkeypatch, level, field, variant
):
    manifest = harness.manifest
    case = manifest.cases[0]
    component = case.components[0]
    current = {
        'manifest': manifest,
        'case': case,
        'component': component,
        'policy': component.policy,
    }[level]
    changed = replace(current, **{field: alias(getattr(current, field), variant)})
    if level == 'policy':
        changed = replace(component, policy=changed)
    if level in {'policy', 'component'}:
        changed = replace(case, components=(changed, case.components[1]))
    if level != 'manifest':
        changed = replace(manifest, cases=(changed, *manifest.cases[1:]))
    no_effects(
        harness,
        monkeypatch,
        lambda: case_claim_manifest_hmac(changed, identity_secret=harness.secret),
    )


@pytest.mark.parametrize('level', ['manifest', 'case', 'component', 'policy'])
def test_manifest_contract_objects_cannot_be_subclasses(harness, monkeypatch, level):
    manifest = harness.manifest
    case = manifest.cases[0]
    component = case.components[0]
    current = {
        'manifest': manifest,
        'case': case,
        'component': component,
        'policy': component.policy,
    }[level]
    subtype = type('UnapprovedSubclass', (type(current),), {})
    changed = subtype(
        **{field.name: getattr(current, field.name) for field in fields(current)}
    )
    if level == 'policy':
        changed = replace(component, policy=changed)
    if level in {'policy', 'component'}:
        changed = replace(case, components=(changed, case.components[1]))
    if level != 'manifest':
        changed = replace(manifest, cases=(changed, *manifest.cases[1:]))
    no_effects(
        harness,
        monkeypatch,
        lambda: case_claim_manifest_hmac(changed, identity_secret=harness.secret),
    )


@pytest.mark.parametrize(
    'field', [field for level, field in MANIFEST_FIELDS if level == 'policy']
)
def test_policy_field_type_is_checked_before_dataclass_deepcopy(
    harness, monkeypatch, field
):
    manifest = harness.manifest
    case = manifest.cases[0]
    component = case.components[0]
    policy = replace(
        component.policy,
        **{field: CopyNormalizedString(getattr(component.policy, field))},
    )
    changed = replace(
        manifest,
        cases=(
            replace(
                case,
                components=(replace(component, policy=policy), case.components[1]),
            ),
            *manifest.cases[1:],
        ),
    )
    no_effects(
        harness,
        monkeypatch,
        lambda: case_claim_manifest_hmac(changed, identity_secret=harness.secret),
    )


@pytest.mark.parametrize(
    'field',
    [
        'ledger_uuid',
        'ledger_epoch',
        'approval_id_hmac',
        'approval_hmac',
        'approved_corpus_snapshot_hmac',
        'approved_provider_safety_snapshot_hmac',
        'provider_safety_envelope_digest',
        'validation_database_identity_hmac',
        'from_generation',
        'execution_process_instance_hmac',
        'execution_runner_fence_hmac',
        'case_id_hmac',
    ],
)
def test_issued_projection_binding_rejects_alias_before_issuance(
    harness, monkeypatch, field
):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    with harness.engine.connect() as connection:
        payload, _ = harness.prepare(connection, kind, changes, case=kwargs['case'])
        payload[field] = alias(payload[field], 'subclass')
        no_effects(
            harness,
            monkeypatch,
            lambda: prepare_case_claim_projection(
                connection,
                manifest=harness.manifest,
                payload=payload,
                identity_secret=harness.secret,
            ),
        )


@pytest.mark.parametrize(
    'index,field',
    [
        (2, 'started_at'),
        (3, 'created_at'),
        (3, 'updated_at'),
        (4, 'created_at'),
        (4, 'updated_at'),
    ],
)
@pytest.mark.parametrize(
    'zone',
    [
        timezone(timedelta(hours=1)),
        timezone(timedelta(0), 'unapproved-clock-zone'),
        'fold',
    ],
)
def test_initial_clock_requires_assembly_utc_before_sql(
    harness, monkeypatch, index, field, zone
):
    append_changed_literal(
        harness,
        monkeypatch,
        index,
        field,
        lambda value: (
            value.replace(fold=1) if zone == 'fold' else value.astimezone(zone)
        ),
    )


@pytest.mark.parametrize(
    'value',
    [
        Decimal('1'),
        b'abc',
        datetime(2026, 9, 13),
        (1, 2),
        MappingProxyType({'value': 1}),
        StringAlias('text'),
        IntegerAlias(1),
        FloatAlias(1.0),
        DictAlias({'value': 1}),
        UUID('11111111-1111-1111-1111-111111111111'),
    ],
)
def test_nested_json_non_native_aliases_cannot_be_signed(harness, monkeypatch, value):
    _, changes, _ = claim_changes(harness, monkeypatch)
    parent = {**changes[2][2], 'metadata': {'nested': [value]}}
    with pytest.raises(RagReleaseLedgerError):
        release_runtime_mutation_hmac(
            'agent_run', None, parent, identity_secret=harness.secret
        )


@pytest.mark.parametrize(
    'left,right',
    [(True, 1), (False, 0), (1, 1.0), (None, 'null'), ([], {}), ([1], {'0': 1})],
)
def test_native_json_types_have_distinct_signed_bytes(
    harness, monkeypatch, left, right
):
    _, changes, _ = claim_changes(harness, monkeypatch)
    parent = changes[2][2]

    def signed(value):
        return release_runtime_mutation_hmac(
            'agent_run',
            None,
            {**parent, 'metadata': {'nested': [value]}},
            identity_secret=harness.secret,
        )

    assert signed(left) != signed(right)


@pytest.mark.parametrize('level', ['cases', 'components'])
@pytest.mark.parametrize('container', [list, type('TupleAlias', (tuple,), {})])
def test_manifest_roster_container_type_is_exact(
    harness, monkeypatch, level, container
):
    manifest = harness.manifest
    if level == 'cases':
        changed = replace(manifest, cases=container(manifest.cases))
    else:
        case = manifest.cases[0]
        changed = replace(
            manifest,
            cases=(
                replace(case, components=container(case.components)),
                *manifest.cases[1:],
            ),
        )
    no_effects(
        harness,
        monkeypatch,
        lambda: case_claim_manifest_hmac(changed, identity_secret=harness.secret),
    )


def test_projection_subclass_cannot_impersonate_issued_capability(harness, monkeypatch):
    kind, changes, kwargs = claim_changes(harness, monkeypatch)
    legitimate = kwargs['approved_case_claim']
    subclass = type('UnapprovedProjection', (type(legitimate),), {})
    with harness.engine.connect() as connection:
        payload, mutations = harness.prepare(
            connection, kind, changes, case=kwargs['case']
        )
        no_effects(
            harness,
            monkeypatch,
            lambda: harness.ledger.append(
                connection,
                payload,
                actual_mutations=mutations,
                database_identity=harness.database_identity,
                approved_case_claim=object.__new__(subclass),
            ),
        )
