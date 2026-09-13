from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import (
    DDL,
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    event,
)

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

_HMAC = String(64)
_UUID = String(36)
_MONEY = Numeric(18, 6)

RELEASE_POSTGRES_IMMUTABILITY_DDL = (
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
         OLD.validation_database_identity_uuid, OLD.validation_database_oid,
         OLD.bootstrap_review_envelope_hmac, OLD.bootstrap_review_nonce_hmac,
         OLD.bootstrap_operation)
     IS DISTINCT FROM
     ROW(NEW.ledger_uuid, NEW.ledger_epoch,
         NEW.predecessor_marker_digest, NEW.rebootstrap_reason_hmac,
         NEW.fingerprint_key_version, NEW.fingerprint_key_material_verifier,
         NEW.designated_environment_id_hmac, NEW.designated_host_id_hmac,
         NEW.validation_database_identity_hmac,
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
     (OLD.execution_process_instance_hmac IS NOT NULL AND
      ROW(OLD.execution_process_instance_hmac, OLD.execution_runner_fence_hmac)
      IS DISTINCT FROM
      ROW(NEW.execution_process_instance_hmac, NEW.execution_runner_fence_hmac)) THEN
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
         NEW.generation_reserved_cost_usd, NEW.total_reserved_cost_usd) THEN
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
         NEW.reserved_cost_usd) THEN
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
            "(state = 'unused' AND execution_process_instance_hmac IS NULL AND "
            "execution_runner_fence_hmac IS NULL) OR "
            "(state <> 'unused' AND execution_process_instance_hmac IS NOT NULL AND "
            "execution_runner_fence_hmac IS NOT NULL)",
            name='ck_rag_release_authorization_owner',
        ),
    )
    Table(
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
