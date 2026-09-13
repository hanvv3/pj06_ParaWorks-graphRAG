from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import (
    DDL,
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    event,
    inspect,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CheckConstraint as SchemaCheckConstraint
from sqlalchemy.schema import UniqueConstraint as SchemaUniqueConstraint

RAG_RELEASE_TABLE_NAMES = frozenset(
    {
        'rag_live_gate_ledgers',
        'rag_live_gate_authorizations',
        'rag_live_gate_cases',
        'rag_live_gate_dispatches',
        'rag_live_gate_transitions',
        'rag_live_gate_quality_reports',
    }
)
RAG_RELEASE_SCHEMA_VERSION = 'paraworks:rag-release-schema:v1'

_HMAC = String(64)
_UUID = String(36)
_MONEY = Numeric(18, 6)

RELEASE_POSTGRES_IMMUTABILITY_DDL = (
    """
CREATE OR REPLACE FUNCTION paraworks_rag_release_guard_database_identity()
RETURNS trigger LANGUAGE plpgsql AS $release_database_identity$
BEGIN
  IF EXISTS (
    SELECT 1 FROM rag_live_gate_ledgers
    WHERE validation_database_identity_uuid <>
          NEW.validation_database_identity_uuid
       OR validation_database_oid <> NEW.validation_database_oid
       OR validation_database_identity_hmac <>
          NEW.validation_database_identity_hmac
       OR validation_database_locator_hmac <>
          NEW.validation_database_locator_hmac
  ) THEN
    RAISE EXCEPTION 'rag release validation database identity changed';
  END IF;
  RETURN NEW;
END
$release_database_identity$
""",
    """
CREATE TRIGGER rag_release_guard_database_identity
BEFORE INSERT ON rag_live_gate_ledgers
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_guard_database_identity()
""",
    """
CREATE OR REPLACE FUNCTION paraworks_rag_release_guard_ledger() RETURNS trigger
LANGUAGE plpgsql AS $release_guard$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'rag release ledger rows are append-only';
  END IF;
  IF ROW(OLD.ledger_uuid, OLD.ledger_epoch,
         OLD.predecessor_marker_digest, OLD.rebootstrap_reason_hmac,
         OLD.fingerprint_key_version, OLD.fingerprint_key_material_verifier,
         OLD.designated_environment_id_hmac, OLD.designated_host_id_hmac,
         OLD.validation_database_identity_hmac,
         OLD.validation_database_locator_hmac,
         OLD.validation_database_identity_uuid, OLD.validation_database_oid,
         OLD.bootstrap_review_envelope_hmac, OLD.bootstrap_review_nonce_hmac,
         OLD.bootstrap_operation)
     IS DISTINCT FROM
     ROW(NEW.ledger_uuid, NEW.ledger_epoch,
         NEW.predecessor_marker_digest, NEW.rebootstrap_reason_hmac,
         NEW.fingerprint_key_version, NEW.fingerprint_key_material_verifier,
         NEW.designated_environment_id_hmac, NEW.designated_host_id_hmac,
         NEW.validation_database_identity_hmac,
         NEW.validation_database_locator_hmac,
         NEW.validation_database_identity_uuid, NEW.validation_database_oid,
         NEW.bootstrap_review_envelope_hmac, NEW.bootstrap_review_nonce_hmac,
         NEW.bootstrap_operation) THEN
    RAISE EXCEPTION 'rag release ledger immutable identity changed';
  END IF;
  IF NEW.generation <> OLD.generation + 1 THEN
    RAISE EXCEPTION 'rag release ledger generation is not gapless';
  END IF;
  RETURN NEW;
END
$release_guard$
""",
    """
CREATE TRIGGER rag_release_guard_ledger
BEFORE UPDATE OR DELETE ON rag_live_gate_ledgers
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_guard_ledger()
""",
    """
CREATE OR REPLACE FUNCTION paraworks_rag_release_reject_mutation() RETURNS trigger
LANGUAGE plpgsql AS $release_immutable$
BEGIN
  RAISE EXCEPTION 'rag release append-only row mutation refused';
END
$release_immutable$
""",
    """
CREATE TRIGGER rag_release_guard_transition
BEFORE UPDATE OR DELETE ON rag_live_gate_transitions
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_reject_mutation()
""",
    """
CREATE TRIGGER rag_release_guard_quality_report
BEFORE UPDATE OR DELETE ON rag_live_gate_quality_reports
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_reject_mutation()
""",
    """
CREATE OR REPLACE FUNCTION paraworks_rag_release_guard_authorization() RETURNS trigger
LANGUAGE plpgsql AS $release_authorization$
BEGIN
  IF TG_OP = 'DELETE' OR
     ROW(OLD.ledger_uuid, OLD.ledger_epoch, OLD.approval_id_hmac,
         OLD.approval_hmac, OLD.base_generation,
         OLD.approved_corpus_snapshot_hmac,
         OLD.approved_provider_safety_snapshot_hmac,
         OLD.provider_safety_envelope_digest,
         OLD.validation_database_identity_hmac, OLD.manifest_hmac,
         OLD.baseline_hmac, OLD.reviewer_roster_hmac)
     IS DISTINCT FROM
     ROW(NEW.ledger_uuid, NEW.ledger_epoch, NEW.approval_id_hmac,
         NEW.approval_hmac, NEW.base_generation,
         NEW.approved_corpus_snapshot_hmac,
         NEW.approved_provider_safety_snapshot_hmac,
         NEW.provider_safety_envelope_digest,
         NEW.validation_database_identity_hmac, NEW.manifest_hmac,
         NEW.baseline_hmac, NEW.reviewer_roster_hmac) OR
     (ROW(OLD.execution_process_instance_hmac, OLD.execution_runner_fence_hmac)
       IS DISTINCT FROM
       ROW(NEW.execution_process_instance_hmac, NEW.execution_runner_fence_hmac)
      AND NOT (
        OLD.state = 'unused' AND NEW.state = 'started'
        AND OLD.execution_process_instance_hmac IS NULL
        AND OLD.execution_runner_fence_hmac IS NULL
        AND NEW.execution_process_instance_hmac IS NOT NULL
        AND NEW.execution_runner_fence_hmac IS NOT NULL
      )) OR
     NOT (
       (OLD.state = 'unused' AND NEW.state IN
        ('unused','started','aborted_corpus_drift','aborted_provider_safety'))
       OR (OLD.state = 'started' AND NEW.state IN
        ('started','complete','finished_failed','aborted_corpus_drift',
         'aborted_execution_crash','aborted_overrun','aborted_provider_safety'))
       OR (OLD.state NOT IN ('unused','started') AND NEW.state = OLD.state)
     ) OR
     NEW.case_claim_count < OLD.case_claim_count OR
     NEW.embedding_dispatch_count < OLD.embedding_dispatch_count OR
     NEW.generation_dispatch_count < OLD.generation_dispatch_count OR
     NEW.total_dispatch_count < OLD.total_dispatch_count OR
     NEW.reserved_cost_usd < OLD.reserved_cost_usd OR
     NEW.charged_cost_usd < OLD.charged_cost_usd OR
     (OLD.state = 'unused' AND NEW.state = 'unused' AND OLD IS DISTINCT FROM NEW) OR
     (OLD.state NOT IN ('unused','started') AND OLD IS DISTINCT FROM NEW) THEN
    RAISE EXCEPTION 'rag release authorization immutable snapshot changed';
  END IF;
  RETURN NEW;
END
$release_authorization$
""",
    """
CREATE TRIGGER rag_release_guard_authorization
BEFORE UPDATE OR DELETE ON rag_live_gate_authorizations
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_guard_authorization()
""",
    """
CREATE OR REPLACE FUNCTION paraworks_rag_release_guard_case() RETURNS trigger
LANGUAGE plpgsql AS $release_case$
BEGIN
  IF TG_OP = 'DELETE' OR
     ROW(OLD.ledger_uuid, OLD.ledger_epoch, OLD.approval_id_hmac,
         OLD.case_id_hmac, OLD.manifest_ordinal,
         OLD.runtime_agent_run_id_hmac, OLD.embedding_reserved_cost_usd,
         OLD.generation_reserved_cost_usd, OLD.total_reserved_cost_usd)
     IS DISTINCT FROM
     ROW(NEW.ledger_uuid, NEW.ledger_epoch, NEW.approval_id_hmac,
         NEW.case_id_hmac, NEW.manifest_ordinal,
         NEW.runtime_agent_run_id_hmac, NEW.embedding_reserved_cost_usd,
         NEW.generation_reserved_cost_usd, NEW.total_reserved_cost_usd) OR
     NOT (
       (OLD.state = 'claimed' AND NEW.state IN ('claimed','complete','failed'))
       OR (OLD.state IN ('complete','failed') AND NEW.state = OLD.state)
     ) OR
     (OLD.state IN ('complete','failed') AND OLD IS DISTINCT FROM NEW) OR
     (NEW.case_projection_hmac IS DISTINCT FROM OLD.case_projection_hmac
      AND NOT (
        OLD.state = 'claimed' AND NEW.state = 'complete'
        AND OLD.case_projection_hmac IS NULL
        AND NEW.case_projection_hmac IS NOT NULL
      )) THEN
    RAISE EXCEPTION 'rag release case immutable identity changed';
  END IF;
  RETURN NEW;
END
$release_case$
""",
    """
CREATE TRIGGER rag_release_guard_case
BEFORE UPDATE OR DELETE ON rag_live_gate_cases
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_guard_case()
""",
    """
CREATE OR REPLACE FUNCTION paraworks_rag_release_guard_dispatch() RETURNS trigger
LANGUAGE plpgsql AS $release_dispatch$
BEGIN
  IF TG_OP = 'DELETE' OR
     ROW(OLD.ledger_uuid, OLD.ledger_epoch, OLD.approval_id_hmac,
         OLD.case_id_hmac, OLD.component, OLD.dispatch_fence_hmac,
         OLD.reserved_cost_usd)
     IS DISTINCT FROM
     ROW(NEW.ledger_uuid, NEW.ledger_epoch, NEW.approval_id_hmac,
         NEW.case_id_hmac, NEW.component, NEW.dispatch_fence_hmac,
         NEW.reserved_cost_usd) OR
     NOT (
       (OLD.state = 'dispatching' AND NEW.state = 'terminal')
       OR (OLD.state = 'terminal' AND NEW.state = 'terminal')
     ) OR
     NEW.dispatch_count <> OLD.dispatch_count OR
     (OLD.state = 'terminal' AND OLD IS DISTINCT FROM NEW) THEN
    RAISE EXCEPTION 'rag release dispatch immutable identity changed';
  END IF;
  RETURN NEW;
END
$release_dispatch$
""",
    """
CREATE TRIGGER rag_release_guard_dispatch
BEFORE UPDATE OR DELETE ON rag_live_gate_dispatches
FOR EACH ROW EXECUTE FUNCTION paraworks_rag_release_guard_dispatch()
""",
)


@dataclass(frozen=True, slots=True)
class RagReleaseTables:
    ledgers: Table
    authorizations: Table
    cases: Table
    dispatches: Table
    transitions: Table
    quality_reports: Table


def _scoped_fk(target: str, *extra: str, name: str) -> ForeignKeyConstraint:
    local = ['ledger_uuid', 'ledger_epoch', *extra]
    remote = [
        f'rag_live_gate_{target}.ledger_uuid',
        f'rag_live_gate_{target}.ledger_epoch',
        *(f'rag_live_gate_{target}.{column}' for column in extra),
    ]
    return ForeignKeyConstraint(local, remote, ondelete='RESTRICT', name=name)


def build_rag_release_metadata() -> MetaData:
    """Build validation-only metadata; this module never imports application ORM."""
    metadata = MetaData()
    Table(
        'rag_live_gate_ledgers',
        metadata,
        Column('ledger_uuid', _UUID, primary_key=True),
        Column('ledger_epoch', Integer, primary_key=True),
        Column('generation', Integer, nullable=False),
        Column('last_transition_digest', _HMAC),
        Column('predecessor_marker_digest', _HMAC),
        Column('rebootstrap_reason_hmac', _HMAC),
        Column('marker_file_digest', _HMAC, nullable=False),
        Column('bootstrap_review_envelope_hmac', _HMAC, nullable=False),
        Column('bootstrap_review_nonce_hmac', _HMAC, nullable=False),
        Column('bootstrap_operation', String(48), nullable=False),
        Column('fingerprint_key_version', String(128), nullable=False),
        Column('fingerprint_key_material_verifier', _HMAC, nullable=False),
        Column('designated_environment_id_hmac', _HMAC, nullable=False),
        Column('designated_host_id_hmac', _HMAC, nullable=False),
        Column('validation_database_identity_hmac', _HMAC, nullable=False),
        Column('validation_database_locator_hmac', _HMAC, nullable=False),
        Column('validation_database_identity_uuid', _UUID, nullable=False),
        Column('validation_database_oid', Integer, nullable=False),
        CheckConstraint('ledger_epoch > 0', name='ck_rag_release_ledger_epoch'),
        CheckConstraint('generation >= 0', name='ck_rag_release_ledger_generation'),
        CheckConstraint(
            "(generation = 0 AND last_transition_digest IS NULL) OR "
            "(generation > 0 AND last_transition_digest IS NOT NULL)",
            name='ck_rag_release_ledger_digest_generation',
        ),
        CheckConstraint(
            'validation_database_oid > 0',
            name='ck_rag_release_ledger_database_oid',
        ),
        CheckConstraint(
            "bootstrap_operation IN ('release-ledger-init',"
            "'release-ledger-rebootstrap','release-ledger-disaster-init')",
            name='ck_rag_release_ledger_bootstrap_operation',
        ),
        UniqueConstraint(
            'bootstrap_review_envelope_hmac',
            name='uq_rag_release_bootstrap_review_envelope',
        ),
        UniqueConstraint(
            'bootstrap_review_nonce_hmac',
            name='uq_rag_release_bootstrap_review_nonce',
        ),
    )
    Table(
        'rag_live_gate_authorizations',
        metadata,
        Column('ledger_uuid', _UUID, primary_key=True),
        Column('ledger_epoch', Integer, primary_key=True),
        Column('approval_id_hmac', _HMAC, primary_key=True),
        Column('approval_hmac', _HMAC, nullable=False, unique=True),
        Column('base_generation', Integer, nullable=False),
        Column('state', String(40), nullable=False),
        Column('approved_corpus_snapshot_hmac', _HMAC, nullable=False),
        Column('approved_provider_safety_snapshot_hmac', _HMAC, nullable=False),
        Column('provider_safety_envelope_digest', _HMAC, nullable=False),
        Column('validation_database_identity_hmac', _HMAC, nullable=False),
        Column('manifest_hmac', _HMAC, nullable=False),
        Column('baseline_hmac', _HMAC, nullable=False),
        Column('reviewer_roster_hmac', _HMAC, nullable=False),
        Column('execution_process_instance_hmac', _HMAC),
        Column('execution_runner_fence_hmac', _HMAC),
        Column('case_claim_count', Integer, nullable=False, default=0),
        Column('embedding_dispatch_count', Integer, nullable=False, default=0),
        Column('generation_dispatch_count', Integer, nullable=False, default=0),
        Column('total_dispatch_count', Integer, nullable=False, default=0),
        Column('reserved_cost_usd', _MONEY, nullable=False, default=0),
        Column('charged_cost_usd', _MONEY, nullable=False, default=0),
        _scoped_fk('ledgers', name='fk_rag_release_authorization_ledger'),
        CheckConstraint('base_generation >= 0', name='ck_rag_release_authorization_base'),
        CheckConstraint(
            "state IN ('unused','started','complete','finished_failed',"
            "'aborted_corpus_drift','aborted_execution_crash','aborted_overrun',"
            "'aborted_provider_safety')",
            name='ck_rag_release_authorization_state',
        ),
        CheckConstraint(
            'case_claim_count BETWEEN 0 AND 30 AND '
            'embedding_dispatch_count BETWEEN 0 AND 10 AND '
            'generation_dispatch_count BETWEEN 0 AND 30 AND '
            'total_dispatch_count BETWEEN 0 AND 40',
            name='ck_rag_release_authorization_counts',
        ),
        CheckConstraint(
            'total_dispatch_count = embedding_dispatch_count + generation_dispatch_count',
            name='ck_rag_release_authorization_total_count',
        ),
        CheckConstraint(
            'reserved_cost_usd >= 0 AND charged_cost_usd >= 0',
            name='ck_rag_release_authorization_costs',
        ),
        CheckConstraint(
            "((execution_process_instance_hmac IS NULL) = "
            "(execution_runner_fence_hmac IS NULL)) AND "
            "((state = 'unused' AND execution_process_instance_hmac IS NULL) OR "
            "(state IN ('aborted_corpus_drift','aborted_provider_safety') AND "
            "(execution_process_instance_hmac IS NULL OR "
            " execution_process_instance_hmac IS NOT NULL)) OR "
            "(state NOT IN ('unused','aborted_corpus_drift',"
            "'aborted_provider_safety') AND "
            " execution_process_instance_hmac IS NOT NULL))",
            name='ck_rag_release_authorization_owner',
        ),
    )
    cases = Table(
        'rag_live_gate_cases',
        metadata,
        Column('ledger_uuid', _UUID, primary_key=True),
        Column('ledger_epoch', Integer, primary_key=True),
        Column('approval_id_hmac', _HMAC, primary_key=True),
        Column('case_id_hmac', _HMAC, primary_key=True),
        Column('manifest_ordinal', Integer, nullable=False),
        Column('case_projection_hmac', _HMAC),
        Column('state', String(16), nullable=False),
        Column('runtime_agent_run_id_hmac', _HMAC, nullable=False),
        Column('embedding_reserved_cost_usd', _MONEY, nullable=False),
        Column('generation_reserved_cost_usd', _MONEY, nullable=False),
        Column('total_reserved_cost_usd', _MONEY, nullable=False),
        _scoped_fk(
            'authorizations',
            'approval_id_hmac',
            name='fk_rag_release_case_authorization',
        ),
        UniqueConstraint(
            'ledger_uuid',
            'ledger_epoch',
            'approval_id_hmac',
            'manifest_ordinal',
            name='uq_rag_release_case_manifest_ordinal',
        ),
        CheckConstraint('manifest_ordinal BETWEEN 0 AND 29', name='ck_rag_release_case_ordinal'),
        CheckConstraint(
            "state IN ('claimed','complete','failed')",
            name='ck_rag_release_case_state',
        ),
        CheckConstraint(
            'embedding_reserved_cost_usd >= 0 AND '
            'generation_reserved_cost_usd >= 0 AND total_reserved_cost_usd >= 0',
            name='ck_rag_release_case_costs',
        ),
        CheckConstraint(
            'total_reserved_cost_usd = embedding_reserved_cost_usd + '
            'generation_reserved_cost_usd',
            name='ck_rag_release_case_cost_total',
        ),
    )
    Index(
        'uq_rag_release_one_claimed_case',
        cases.c.ledger_uuid,
        cases.c.ledger_epoch,
        cases.c.approval_id_hmac,
        unique=True,
        postgresql_where=cases.c.state == 'claimed',
        sqlite_where=cases.c.state == 'claimed',
    )
    Table(
        'rag_live_gate_dispatches',
        metadata,
        Column('ledger_uuid', _UUID, primary_key=True),
        Column('ledger_epoch', Integer, primary_key=True),
        Column('approval_id_hmac', _HMAC, primary_key=True),
        Column('case_id_hmac', _HMAC, primary_key=True),
        Column('component', String(32), primary_key=True),
        Column('dispatch_fence_hmac', _HMAC, nullable=False),
        Column('state', String(16), nullable=False),
        Column('dispatch_count', Integer, nullable=False),
        Column('reserved_cost_usd', _MONEY, nullable=False),
        Column('charged_cost_usd', _MONEY, nullable=False),
        Column('charge_basis', String(16), nullable=False),
        _scoped_fk(
            'cases',
            'approval_id_hmac',
            'case_id_hmac',
            name='fk_rag_release_dispatch_case',
        ),
        CheckConstraint(
            "component IN ('query_embedding','answer_generation')",
            name='ck_rag_release_dispatch_component',
        ),
        CheckConstraint(
            "state IN ('not_attempted','dispatching','terminal')",
            name='ck_rag_release_dispatch_state',
        ),
        CheckConstraint('dispatch_count IN (0,1)', name='ck_rag_release_dispatch_count'),
        CheckConstraint(
            "charge_basis IN ('zero','reserved','actual')",
            name='ck_rag_release_dispatch_charge_basis',
        ),
        CheckConstraint(
            'reserved_cost_usd >= 0 AND charged_cost_usd >= 0',
            name='ck_rag_release_dispatch_costs',
        ),
    )
    Table(
        'rag_live_gate_transitions',
        metadata,
        Column('ledger_uuid', _UUID, primary_key=True),
        Column('ledger_epoch', Integer, primary_key=True),
        Column('generation', Integer, primary_key=True),
        Column('transition_kind', String(64), nullable=False),
        Column('transition_digest', _HMAC, nullable=False, unique=True),
        Column('payload_canonical_bytes', LargeBinary, nullable=False),
        _scoped_fk('ledgers', name='fk_rag_release_transition_ledger'),
        CheckConstraint('generation > 0', name='ck_rag_release_transition_generation'),
    )
    Table(
        'rag_live_gate_quality_reports',
        metadata,
        Column('ledger_uuid', _UUID, primary_key=True),
        Column('ledger_epoch', Integer, primary_key=True),
        Column('approval_id_hmac', _HMAC, primary_key=True),
        Column('quality_report_hmac', _HMAC, nullable=False, unique=True),
        Column('manifest_hmac', _HMAC, nullable=False),
        Column('baseline_hmac', _HMAC, nullable=False),
        Column('reviewer_roster_hmac', _HMAC, nullable=False),
        Column('payload_canonical_bytes', LargeBinary, nullable=False),
        _scoped_fk(
            'authorizations',
            'approval_id_hmac',
            name='fk_rag_release_quality_authorization',
        ),
    )
    assert set(metadata.tables) == RAG_RELEASE_TABLE_NAMES
    assert all(table.schema is None for table in metadata.tables.values())
    for statement in RELEASE_POSTGRES_IMMUTABILITY_DDL:
        event.listen(
            metadata.tables['rag_live_gate_quality_reports'],
            'after_create',
            DDL(statement).execute_if(dialect='postgresql'),
        )
    for table_name in sorted(RAG_RELEASE_TABLE_NAMES):
        event.listen(
            metadata.tables['rag_live_gate_quality_reports'],
            'after_create',
            DDL(
                f"COMMENT ON TABLE {table_name} IS '{RAG_RELEASE_SCHEMA_VERSION}'"
            ).execute_if(dialect='postgresql'),
        )
    return metadata


def release_tables(metadata: MetaData) -> RagReleaseTables:
    if set(metadata.tables) != RAG_RELEASE_TABLE_NAMES:
        raise ValueError('release metadata must contain exactly six tables')
    return RagReleaseTables(
        ledgers=metadata.tables['rag_live_gate_ledgers'],
        authorizations=metadata.tables['rag_live_gate_authorizations'],
        cases=metadata.tables['rag_live_gate_cases'],
        dispatches=metadata.tables['rag_live_gate_dispatches'],
        transitions=metadata.tables['rag_live_gate_transitions'],
        quality_reports=metadata.tables['rag_live_gate_quality_reports'],
    )


_EXPECTED_RELEASE_TRIGGERS = {
    'rag_live_gate_ledgers': {
        'rag_release_guard_database_identity',
        'rag_release_guard_ledger',
    },
    'rag_live_gate_authorizations': {'rag_release_guard_authorization'},
    'rag_live_gate_cases': {'rag_release_guard_case'},
    'rag_live_gate_dispatches': {'rag_release_guard_dispatch'},
    'rag_live_gate_transitions': {'rag_release_guard_transition'},
    'rag_live_gate_quality_reports': {'rag_release_guard_quality_report'},
}


def _normalized_sql(value: object) -> str:
    return ''.join(str(value).lower().split()).replace('::text', '')


def assert_rag_release_physical_contract(connection: Connection) -> None:
    """Fail closed unless PostgreSQL exposes the frozen exact-six DDL contract."""
    if connection.dialect.name != 'postgresql':
        raise ValueError('release physical contract requires PostgreSQL')
    inspector = inspect(connection)
    live_names = {
        name
        for name in inspector.get_table_names(schema=None)
        if name.startswith('rag_live_gate_')
    }
    if live_names != set(RAG_RELEASE_TABLE_NAMES):
        raise ValueError('release physical table contract differs')
    expected_metadata = build_rag_release_metadata()
    dialect = postgresql.dialect()
    for name in sorted(RAG_RELEASE_TABLE_NAMES):
        expected = expected_metadata.tables[name]
        actual_columns = inspector.get_columns(name, schema=None)
        expected_columns = list(expected.columns)
        if len(actual_columns) != len(expected_columns):
            raise ValueError('release physical column contract differs')
        for actual, column in zip(actual_columns, expected_columns, strict=True):
            actual_type = actual['type'].compile(dialect=dialect).upper()
            expected_type = column.type.compile(dialect=dialect).upper()
            if (
                actual['name'] != column.name
                or actual_type != expected_type
                or bool(actual['nullable']) != bool(column.nullable)
            ):
                raise ValueError('release physical column contract differs')
        actual_pk = inspector.get_pk_constraint(name, schema=None)
        if tuple(actual_pk.get('constrained_columns') or ()) != tuple(
            column.name for column in expected.primary_key.columns
        ):
            raise ValueError('release physical primary-key contract differs')
        expected_fks = {
            (
                constraint.name,
                tuple(element.parent.name for element in constraint.elements),
                constraint.referred_table.name,
                tuple(element.column.name for element in constraint.elements),
                constraint.ondelete,
            )
            for constraint in expected.foreign_key_constraints
        }
        actual_fks = {
            (
                item.get('name'),
                tuple(item.get('constrained_columns') or ()),
                item.get('referred_table'),
                tuple(item.get('referred_columns') or ()),
                (item.get('options') or {}).get('ondelete'),
            )
            for item in inspector.get_foreign_keys(name, schema=None)
        }
        if actual_fks != expected_fks:
            raise ValueError('release physical foreign-key contract differs')
        expected_unique = {
            (constraint.name, tuple(column.name for column in constraint.columns))
            for constraint in expected.constraints
            if isinstance(constraint, SchemaUniqueConstraint)
        }
        actual_unique = {
            (item.get('name'), tuple(item.get('column_names') or ()))
            for item in inspector.get_unique_constraints(name, schema=None)
        }
        named_expected = {item for item in expected_unique if item[0] is not None}
        if (
            {columns for _name, columns in actual_unique}
            != {columns for _name, columns in expected_unique}
            or not named_expected <= actual_unique
        ):
            raise ValueError('release physical unique contract differs')
        expected_checks = {
            constraint.name: _normalized_sql(constraint.sqltext)
            for constraint in expected.constraints
            if isinstance(constraint, SchemaCheckConstraint)
        }
        actual_checks = {
            item.get('name'): _normalized_sql(item.get('sqltext'))
            for item in inspector.get_check_constraints(name, schema=None)
        }
        if set(actual_checks) != set(expected_checks) or any(
            expected_checks[key] not in actual_checks[key]
            and actual_checks[key] not in expected_checks[key]
            for key in expected_checks
        ):
            raise ValueError('release physical check contract differs')
        expected_indexes = {
            (
                index.name,
                tuple(column.name for column in index.columns),
                bool(index.unique),
                _normalized_sql(index.dialect_options['postgresql'].get('where')),
            )
            for index in expected.indexes
        }
        actual_indexes = {
            (
                item.get('name'),
                tuple(item.get('column_names') or ()),
                bool(item.get('unique')),
                _normalized_sql(
                    (item.get('dialect_options') or {}).get('postgresql_where')
                ),
            )
            for item in inspector.get_indexes(name, schema=None)
            if not item.get('duplicates_constraint')
        }
        if actual_indexes != expected_indexes:
            raise ValueError('release physical index contract differs')
        comment = connection.scalar(
            text('SELECT obj_description(to_regclass(:name), :catalog)'),
            {'catalog': 'pg_class', 'name': name},
        )
        if comment != RAG_RELEASE_SCHEMA_VERSION:
            raise ValueError('release physical schema version differs')
    trigger_rows = connection.execute(
        text(
            'SELECT c.relname, t.tgname, p.proname, t.tgenabled, '
            'pg_get_triggerdef(t.oid) FROM pg_trigger t '
            'JOIN pg_class c ON c.oid=t.tgrelid '
            'JOIN pg_namespace n ON n.oid=c.relnamespace '
            'JOIN pg_proc p ON p.oid=t.tgfoid '
            "WHERE NOT t.tgisinternal AND n.nspname=current_schema() "
            "AND c.relname LIKE 'rag_live_gate_%'"
        )
    ).all()
    actual_triggers: dict[str, set[str]] = {
        name: set() for name in RAG_RELEASE_TABLE_NAMES
    }
    expected_functions = {
        'rag_release_guard_database_identity': (
            'paraworks_rag_release_guard_database_identity',
            'before insert',
        ),
        'rag_release_guard_ledger': (
            'paraworks_rag_release_guard_ledger',
            'before update or delete',
        ),
        'rag_release_guard_authorization': (
            'paraworks_rag_release_guard_authorization',
            'before update or delete',
        ),
        'rag_release_guard_case': (
            'paraworks_rag_release_guard_case',
            'before update or delete',
        ),
        'rag_release_guard_dispatch': (
            'paraworks_rag_release_guard_dispatch',
            'before update or delete',
        ),
        'rag_release_guard_transition': (
            'paraworks_rag_release_reject_mutation',
            'before update or delete',
        ),
        'rag_release_guard_quality_report': (
            'paraworks_rag_release_reject_mutation',
            'before update or delete',
        ),
    }
    for table_name, trigger_name, function_name, enabled, definition in trigger_rows:
        actual_triggers.setdefault(table_name, set()).add(trigger_name)
        expected_function = expected_functions.get(trigger_name)
        normalized_definition = ' '.join(str(definition).lower().split())
        if (
            expected_function is None
            or function_name != expected_function[0]
            or enabled != 'O'
            or expected_function[1] not in normalized_definition
        ):
            raise ValueError('release physical trigger contract differs')
    if actual_triggers != _EXPECTED_RELEASE_TRIGGERS:
        raise ValueError('release physical trigger contract differs')
    expected_sources: dict[str, str] = {}
    for statement in RELEASE_POSTGRES_IMMUTABILITY_DDL:
        match = re.search(
            r'CREATE OR REPLACE FUNCTION\s+(\w+)\(\).*?AS \$\w+\$(.*?)\$\w+\$',
            statement,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            expected_sources[match.group(1)] = _normalized_sql(match.group(2))
    function_rows = connection.execute(
        text(
            'SELECT p.proname, p.prosrc, pg_get_function_result(p.oid), '
            'pg_get_function_arguments(p.oid), l.lanname '
            'FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace '
            'JOIN pg_language l ON l.oid=p.prolang '
            "WHERE n.nspname=current_schema() AND p.proname LIKE "
            "'paraworks_rag_release_%'"
        )
    ).all()
    actual_sources = {
        name: _normalized_sql(source)
        for name, source, result, arguments, language in function_rows
        if result == 'trigger' and arguments == '' and language == 'plpgsql'
    }
    if actual_sources != expected_sources:
        raise ValueError('release physical trigger function contract differs')
