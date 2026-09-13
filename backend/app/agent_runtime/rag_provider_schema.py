"""Private SQL descriptions for provider authority operations, never ORM aliases.

These factories do not create/alter database objects. Every invocation allocates
fresh native columns/types and has no caller default, compiler or processor.
Physical schema ownership remains with the existing runtime migration.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
)


def _provider_tables():
    metadata = MetaData()

    def integer(name, *, nullable=False, primary_key=False, big=False):
        return Column(
            name,
            BigInteger() if big else Integer(),
            nullable=nullable,
            primary_key=primary_key,
        )

    def string(name, length, *, nullable=False):
        return Column(name, String(length), nullable=nullable)

    def clock(name, *, nullable=False):
        return Column(name, DateTime(timezone=True), nullable=nullable)

    authority = Table(
        'rag_provider_safety_authorities',
        metadata,
        integer('id', primary_key=True),
        string('authority_uuid', 36),
        string('designated_environment_id', 128),
        integer('global_safety_generation', big=True),
        string('envelope_digest', 64),
        string('fingerprint_key_version', 64),
        string('fingerprint_key_material_verifier', 64),
        clock('created_at'),
        clock('updated_at'),
    )
    readiness = Table(
        'rag_provider_readiness',
        metadata,
        integer('id', primary_key=True),
        integer('authority_id'),
        string('component', 32),
        string('provider', 120),
        string('model', 120),
        string('reasoning_or_config_identity', 128),
        Column('active', Boolean(), nullable=False),
        string('authorized_model_config_version', 64),
        string('authorized_model_config_snapshot_hmac', 64),
        string('authorized_cost_policy_version', 64),
        string('authorized_token_estimator_version', 96),
        string('authorized_fingerprint_key_version', 64),
        string('authorized_fingerprint_key_material_verifier', 64),
        string('authorized_policy_snapshot_hmac', 64),
        string('state', 32),
        integer('state_version', big=True),
        integer('family_safety_generation', big=True),
        integer('overrun_agent_run_id', nullable=True),
        integer('overrun_input_tokens', big=True, nullable=True),
        integer('overrun_output_tokens', big=True, nullable=True),
        Column('overrun_cost_usd', Numeric(24, 6), nullable=True),
        clock('overrun_observed_at', nullable=True),
        string('reset_by', 120, nullable=True),
        clock('reset_at', nullable=True),
        string('reviewed_gate_reference_hmac', 64, nullable=True),
        clock('created_at'),
        clock('updated_at'),
    )
    history = Table(
        'rag_provider_safety_transitions',
        metadata,
        integer('id', primary_key=True),
        integer('authority_id'),
        integer('readiness_id', nullable=True),
        integer('global_safety_generation', big=True),
        string('transition_kind', 32),
        string('prior_state', 32, nullable=True),
        string('new_state', 32, nullable=True),
        integer('prior_state_version', big=True, nullable=True),
        integer('new_state_version', big=True, nullable=True),
        integer('prior_family_safety_generation', big=True, nullable=True),
        integer('new_family_safety_generation', big=True, nullable=True),
        string('envelope_digest', 64),
        string('reviewed_transition_reference_hmac', 64),
        string('actor_subject_hmac', 64, nullable=True),
        integer('agent_run_id', nullable=True),
        clock('created_at'),
    )
    return authority, readiness, history
