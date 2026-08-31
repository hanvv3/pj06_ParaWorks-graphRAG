from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


_HMAC64 = "length(%s) = 64"


class AgentRunCostComponent(Base):
    __tablename__ = 'agent_run_cost_components'
    __table_args__ = (
        UniqueConstraint(
            'agent_run_id',
            'component',
            name='uq_agent_run_cost_component',
        ),
        UniqueConstraint(
            'agent_run_id',
            'component_ordinal',
            name='uq_agent_run_cost_component_ordinal',
        ),
        CheckConstraint(
            "(component = 'query_embedding' AND component_ordinal = 0) OR "
            "(component = 'answer_generation' AND component_ordinal = 1)",
            name='ck_agent_run_cost_component_order',
        ),
        CheckConstraint(
            "dispatch_state IN ('not_attempted', 'dispatching', 'terminal', "
            "'abandoned_unknown') AND dispatch_count IN (0, 1)",
            name='ck_agent_run_cost_component_dispatch',
        ),
        CheckConstraint(
            'reserved_input_tokens >= 0 AND reserved_output_tokens >= 0 AND '
            '(actual_input_tokens IS NULL OR actual_input_tokens >= 0) AND '
            '(actual_output_tokens IS NULL OR actual_output_tokens >= 0) AND '
            'reserved_cost_usd >= 0 AND charged_cost_usd >= 0',
            name='ck_agent_run_cost_component_nonnegative',
        ),
        CheckConstraint(
            "charge_basis IN ('zero', 'actual', 'reserved')",
            name='ck_agent_run_cost_component_charge_basis',
        ),
        CheckConstraint(
            "provider = 'openai' AND ((component = 'query_embedding' AND "
            "model = 'text-embedding-3-small' AND "
            "authorized_model_config_version = 'rag-query-embedding-config:v1' AND "
            "authorized_cost_policy_version = 'rag-query-embedding-cost:v1' AND "
            "authorized_token_estimator_version = "
            "'openai-cl100k-text-embedding-3-small:v1') OR "
            "(component = 'answer_generation' AND "
            "model = 'gpt-5.4-mini-2026-03-17' AND "
            "authorized_model_config_version = 'rag-answer-model-config:v1' AND "
            "authorized_cost_policy_version = 'rag-answer-cost:v1' AND "
            "authorized_token_estimator_version = 'openai-o200k-rag-answer:v1'))",
            name='ck_agent_run_cost_component_target_identity',
        ),
        CheckConstraint(
            'length(authorized_model_config_snapshot_hmac) = 64 AND '
            'length(authorized_policy_snapshot_hmac) = 64 AND '
            '(dispatch_fence_hmac IS NULL OR length(dispatch_fence_hmac) = 64) AND '
            '(process_instance_hmac IS NULL OR length(process_instance_hmac) = 64)',
            name='ck_agent_run_cost_component_hmacs',
        ),
        CheckConstraint(
            "(dispatch_state = 'not_attempted' AND attempted = false AND "
            "dispatch_count = 0 AND actual_input_tokens IS NULL AND "
            "actual_output_tokens IS NULL AND charged_cost_usd = 0 AND "
            "charge_basis = 'zero' AND overrun = false AND "
            "dispatch_fence_hmac IS NULL AND process_instance_hmac IS NULL AND "
            "terminal_outcome IS NULL) OR "
            "(dispatch_state = 'dispatching' AND attempted = true AND "
            "dispatch_count = 1 AND actual_input_tokens IS NULL AND "
            "actual_output_tokens IS NULL AND charged_cost_usd = reserved_cost_usd AND "
            "charge_basis = 'reserved' AND overrun = false AND "
            "dispatch_fence_hmac IS NOT NULL AND process_instance_hmac IS NOT NULL AND "
            "terminal_outcome IS NULL) OR "
            "(dispatch_state = 'terminal' AND attempted = false AND dispatch_count = 0 AND "
            "reserved_input_tokens = 0 AND reserved_output_tokens = 0 AND "
            "actual_input_tokens IS NULL AND actual_output_tokens IS NULL AND "
            "reserved_cost_usd = 0 AND charged_cost_usd = 0 AND "
            "charge_basis = 'zero' AND overrun = false AND "
            "dispatch_fence_hmac IS NULL AND process_instance_hmac IS NULL AND "
            "terminal_outcome IS NULL) OR "
            "(dispatch_state = 'terminal' AND attempted = true AND dispatch_count = 1 AND "
            "dispatch_fence_hmac IS NOT NULL AND process_instance_hmac IS NOT NULL AND "
            "terminal_outcome IS NOT NULL AND ((charge_basis = 'actual' AND "
            "actual_input_tokens IS NOT NULL AND actual_output_tokens IS NOT NULL) OR "
            "(charge_basis = 'reserved' AND actual_input_tokens IS NULL AND "
            "actual_output_tokens IS NULL AND charged_cost_usd = reserved_cost_usd)) AND "
            "(overrun = false OR charge_basis = 'actual')) OR "
            "(dispatch_state = 'abandoned_unknown' AND attempted = true AND "
            "dispatch_count = 1 AND actual_input_tokens IS NULL AND "
            "actual_output_tokens IS NULL AND charged_cost_usd = reserved_cost_usd AND "
            "charge_basis = 'reserved' AND overrun = false AND "
            "dispatch_fence_hmac IS NOT NULL AND process_instance_hmac IS NOT NULL AND "
            "terminal_outcome = 'abandoned_unknown')",
            name='ck_agent_run_cost_component_state_shape',
        ),
        Index(
            'ix_agent_run_cost_components_parent_order',
            'agent_run_id',
            'component_ordinal',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey('agent_runs.id', ondelete='RESTRICT')
    )
    component: Mapped[str] = mapped_column(String(32))
    component_ordinal: Mapped[int] = mapped_column(Integer)
    dispatch_state: Mapped[str] = mapped_column(String(32))
    dispatch_fence_hmac: Mapped[str | None] = mapped_column(String(64))
    process_instance_hmac: Mapped[str | None] = mapped_column(String(64))
    attempted: Mapped[bool] = mapped_column(Boolean)
    dispatch_count: Mapped[int] = mapped_column(Integer)
    reserved_input_tokens: Mapped[int] = mapped_column(BigInteger)
    reserved_output_tokens: Mapped[int] = mapped_column(BigInteger)
    actual_input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    actual_output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    reserved_cost_usd: Mapped[Decimal] = mapped_column(Numeric(24, 6))
    charged_cost_usd: Mapped[Decimal] = mapped_column(Numeric(24, 6))
    charge_basis: Mapped[str] = mapped_column(String(16))
    overrun: Mapped[bool] = mapped_column(Boolean, default=False)
    provider: Mapped[str] = mapped_column(String(120))
    model: Mapped[str] = mapped_column(String(120))
    authorized_model_config_version: Mapped[str] = mapped_column(String(64))
    authorized_model_config_snapshot_hmac: Mapped[str] = mapped_column(String(64))
    authorized_cost_policy_version: Mapped[str] = mapped_column(String(64))
    authorized_token_estimator_version: Mapped[str] = mapped_column(String(96))
    authorized_policy_snapshot_hmac: Mapped[str] = mapped_column(String(64))
    terminal_outcome: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RagProviderSafetyAuthority(Base):
    __tablename__ = 'rag_provider_safety_authorities'
    __table_args__ = (
        CheckConstraint('id = 1', name='ck_rag_provider_safety_authority_singleton'),
        CheckConstraint(
            'global_safety_generation >= 0 AND length(envelope_digest) = 64 AND '
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_rag_provider_safety_authority_shape',
        ),
        UniqueConstraint('authority_uuid', name='uq_rag_provider_safety_authority_uuid'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    authority_uuid: Mapped[str] = mapped_column(String(36))
    designated_environment_id: Mapped[str] = mapped_column(String(128))
    global_safety_generation: Mapped[int] = mapped_column(BigInteger)
    envelope_digest: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RagProviderReadiness(Base):
    __tablename__ = 'rag_provider_readiness'
    __table_args__ = (
        UniqueConstraint(
            'component',
            'provider',
            'model',
            'reasoning_or_config_identity',
            name='uq_rag_provider_readiness_family',
        ),
        CheckConstraint(
            "component IN ('query_embedding', 'answer_generation') AND "
            "state IN ('ready', 'rebind_required', 'blocked_overrun', "
            "'blocked_remediation') AND state_version >= 1 AND "
            'family_safety_generation >= 0',
            name='ck_rag_provider_readiness_state',
        ),
        CheckConstraint(
            'length(authorized_model_config_snapshot_hmac) = 64 AND '
            'length(authorized_fingerprint_key_material_verifier) = 64 AND '
            'length(authorized_policy_snapshot_hmac) = 64 AND '
            '(reviewed_gate_reference_hmac IS NULL OR '
            'length(reviewed_gate_reference_hmac) = 64)',
            name='ck_rag_provider_readiness_hmacs',
        ),
        CheckConstraint(
            'CASE WHEN overrun_agent_run_id IS NULL AND '
            'overrun_input_tokens IS NULL AND overrun_output_tokens IS NULL AND '
            'overrun_cost_usd IS NULL AND overrun_observed_at IS NULL THEN true ELSE '
            'overrun_agent_run_id IS NOT NULL AND overrun_input_tokens IS NOT NULL AND '
            'overrun_output_tokens IS NOT NULL AND overrun_cost_usd IS NOT NULL AND '
            'overrun_observed_at IS NOT NULL AND overrun_input_tokens >= 0 AND '
            'overrun_output_tokens >= 0 AND overrun_cost_usd >= 0 END',
            name='ck_rag_provider_readiness_overrun',
        ),
        CheckConstraint(
            '(reset_by IS NULL AND reset_at IS NULL) OR '
            '(reset_by IS NOT NULL AND reset_at IS NOT NULL)',
            name='ck_rag_provider_readiness_reset_attribution',
        ),
        Index(
            'uq_rag_provider_readiness_active_component',
            'component',
            unique=True,
            postgresql_where=text('active = true'),
            sqlite_where=text('active = 1'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    authority_id: Mapped[int] = mapped_column(
        ForeignKey('rag_provider_safety_authorities.id', ondelete='RESTRICT'), default=1
    )
    component: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(120))
    model: Mapped[str] = mapped_column(String(120))
    reasoning_or_config_identity: Mapped[str] = mapped_column(String(128))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    authorized_model_config_version: Mapped[str] = mapped_column(String(64))
    authorized_model_config_snapshot_hmac: Mapped[str] = mapped_column(String(64))
    authorized_cost_policy_version: Mapped[str] = mapped_column(String(64))
    authorized_token_estimator_version: Mapped[str] = mapped_column(String(96))
    authorized_fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    authorized_fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    authorized_policy_snapshot_hmac: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(32))
    state_version: Mapped[int] = mapped_column(BigInteger)
    family_safety_generation: Mapped[int] = mapped_column(BigInteger)
    overrun_agent_run_id: Mapped[int | None] = mapped_column(
        ForeignKey('agent_runs.id', ondelete='RESTRICT')
    )
    overrun_input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    overrun_output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    overrun_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(24, 6))
    overrun_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reset_by: Mapped[str | None] = mapped_column(String(120))
    reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewed_gate_reference_hmac: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RagProviderSafetyTransition(Base):
    __tablename__ = 'rag_provider_safety_transitions'
    __table_args__ = (
        UniqueConstraint(
            'authority_id',
            'global_safety_generation',
            name='uq_rag_provider_safety_transition_generation',
        ),
        CheckConstraint(
            'global_safety_generation >= 0 AND '
            "transition_kind IN ('bootstrap', 'rebind_required', 'rebind', 'reset', "
            "'block_overrun', 'block_remediation', 'supersession') AND "
            'length(envelope_digest) = 64 AND '
            'length(reviewed_transition_reference_hmac) = 64',
            name='ck_rag_provider_safety_transition_shape',
        ),
        CheckConstraint(
            '(readiness_id IS NULL AND prior_state IS NULL AND new_state IS NULL AND '
            'prior_state_version IS NULL AND new_state_version IS NULL AND '
            'prior_family_safety_generation IS NULL AND '
            'new_family_safety_generation IS NULL) OR '
            '(readiness_id IS NOT NULL AND new_state IS NOT NULL AND '
            "new_state IN ('ready', 'rebind_required', 'blocked_overrun', "
            "'blocked_remediation') AND new_state_version IS NOT NULL AND "
            'new_state_version >= 1 AND new_family_safety_generation IS NOT NULL AND '
            'new_family_safety_generation >= 0 AND '
            '((prior_state IS NULL AND prior_state_version IS NULL AND '
            'prior_family_safety_generation IS NULL) OR '
            '(prior_state IS NOT NULL AND prior_state IN '
            "('ready', 'rebind_required', 'blocked_overrun', 'blocked_remediation') AND "
            'prior_state_version IS NOT NULL AND prior_state_version >= 1 AND '
            'prior_family_safety_generation IS NOT NULL AND '
            'prior_family_safety_generation >= 0)))',
            name='ck_rag_provider_safety_transition_family_snapshot',
        ),
        CheckConstraint(
            "(global_safety_generation = 0 AND transition_kind = 'bootstrap' AND "
            'readiness_id IS NULL) OR '
            "(global_safety_generation > 0 AND transition_kind <> 'bootstrap')",
            name='ck_rag_provider_safety_transition_bootstrap_generation',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    authority_id: Mapped[int] = mapped_column(
        ForeignKey('rag_provider_safety_authorities.id', ondelete='RESTRICT')
    )
    readiness_id: Mapped[int | None] = mapped_column(
        ForeignKey('rag_provider_readiness.id', ondelete='RESTRICT')
    )
    global_safety_generation: Mapped[int] = mapped_column(BigInteger)
    transition_kind: Mapped[str] = mapped_column(String(32))
    prior_state: Mapped[str | None] = mapped_column(String(32))
    new_state: Mapped[str | None] = mapped_column(String(32))
    prior_state_version: Mapped[int | None] = mapped_column(BigInteger)
    new_state_version: Mapped[int | None] = mapped_column(BigInteger)
    prior_family_safety_generation: Mapped[int | None] = mapped_column(BigInteger)
    new_family_safety_generation: Mapped[int | None] = mapped_column(BigInteger)
    envelope_digest: Mapped[str] = mapped_column(String(64))
    reviewed_transition_reference_hmac: Mapped[str] = mapped_column(String(64))
    actor_subject_hmac: Mapped[str | None] = mapped_column(String(64))
    agent_run_id: Mapped[int | None] = mapped_column(
        ForeignKey('agent_runs.id', ondelete='RESTRICT')
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RagAdvisoryLockKey(Base):
    __tablename__ = 'rag_advisory_lock_key_registry'
    __table_args__ = (
        UniqueConstraint(
            'lock_identity_digest', name='uq_rag_advisory_lock_identity_digest'
        ),
        CheckConstraint(
            'key1 >= -2147483648 AND key1 <= 2147483647 AND '
            'key2 >= -2147483648 AND key2 <= 2147483647 AND '
            'length(lock_identity_digest) = 64 AND '
            "identity_namespace IN ('static', 'dynamic') AND "
            'length(lock_identity_canonical_bytes) > 0',
            name='ck_rag_advisory_lock_key_shape',
        ),
    )

    key1: Mapped[int] = mapped_column(Integer, primary_key=True)
    key2: Mapped[int] = mapped_column(Integer, primary_key=True)
    identity_namespace: Mapped[str] = mapped_column(String(16))
    lock_identity_digest: Mapped[str] = mapped_column(String(64))
    lock_identity_canonical_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
