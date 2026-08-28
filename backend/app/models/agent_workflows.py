from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgentWorkflowThread(Base):
    __tablename__ = 'agent_workflow_threads'
    __table_args__ = (
        Index(
            'uq_agent_workflow_thread_client_request',
            'security_scope_id',
            'workflow_name',
            'owner_subject_id',
            'client_request_id',
            unique=True,
            postgresql_where=text('client_request_id IS NOT NULL'),
            sqlite_where=text('client_request_id IS NOT NULL'),
        ),
    )

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workflow_name: Mapped[str] = mapped_column(String(64), index=True)
    graph_version: Mapped[str] = mapped_column(String(64))
    checkpoint_thread_id: Mapped[str] = mapped_column(String(128), unique=True)
    checkpoint_store: Mapped[str] = mapped_column(String(32))
    owner_subject_id: Mapped[str] = mapped_column(String(128), index=True)
    security_scope_id: Mapped[str] = mapped_column(String(128), default='default')
    client_request_id: Mapped[str | None] = mapped_column(String(128))
    input_hash: Mapped[str] = mapped_column(String(64))
    evidence_version_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default='created', index=True)
    state_version: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checkpoint_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_by_subject_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )


class AgentWorkflowRequest(Base):
    __tablename__ = 'agent_workflow_requests'

    workflow_thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    input_schema_version: Mapped[str] = mapped_column(String(32))
    request_kind: Mapped[str] = mapped_column(
        String(64),
        default='review_source_versions',
    )
    agent_names: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
    )
    selection_policy_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(32))
    fingerprint_key_material_verifier: Mapped[str | None] = mapped_column(
        String(64)
    )
    auto_review_mode: Mapped[str | None] = mapped_column(String(32))
    auto_review_validator_provider: Mapped[str | None] = mapped_column(String(120))
    auto_review_validator_model: Mapped[str | None] = mapped_column(String(120))
    auto_review_reasoning_effort: Mapped[str | None] = mapped_column(String(32))
    auto_review_validator_prompt_version: Mapped[str | None] = mapped_column(
        String(64)
    )
    auto_review_validator_output_contract_version: Mapped[str | None] = (
        mapped_column(String(64))
    )
    auto_review_policy_version: Mapped[str | None] = mapped_column(String(64))
    auto_review_cost_policy_version: Mapped[str | None] = mapped_column(String(64))
    auto_review_extraction_cost_policy_version: Mapped[str | None] = mapped_column(
        String(64)
    )
    auto_review_token_estimator_version: Mapped[str | None] = mapped_column(
        String(64)
    )
    auto_review_extraction_token_estimator_version: Mapped[str | None] = (
        mapped_column(String(64))
    )
    auto_review_tokenizer_encoding: Mapped[str | None] = mapped_column(String(64))
    auto_review_reply_priming_tokens: Mapped[int | None] = mapped_column(Integer)
    auto_review_framing_safety_tokens: Mapped[int | None] = mapped_column(Integer)
    auto_review_max_input_tokens: Mapped[int | None] = mapped_column(Integer)
    auto_review_max_output_tokens: Mapped[int | None] = mapped_column(Integer)
    auto_review_max_candidates_per_batch: Mapped[int | None] = mapped_column(Integer)
    auto_review_max_batches_per_workflow: Mapped[int | None] = mapped_column(Integer)
    auto_review_max_candidates_per_workflow: Mapped[int | None] = mapped_column(
        Integer
    )
    auto_review_max_provider_attempts: Mapped[int | None] = mapped_column(Integer)
    auto_review_provider_timeout_seconds: Mapped[int | None] = mapped_column(Integer)
    auto_review_provider_send_start_window_seconds: Mapped[int | None] = (
        mapped_column(Integer)
    )
    auto_review_provider_attempt_lease_seconds: Mapped[int | None] = mapped_column(
        Integer
    )
    auto_review_provider_commit_grace_seconds: Mapped[int | None] = mapped_column(
        Integer
    )
    auto_review_validator_input_usd_per_1m: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    auto_review_validator_output_usd_per_1m: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    auto_review_extraction_input_usd_per_1m: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    auto_review_extraction_output_usd_per_1m: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    auto_review_enforce_percentage: Mapped[int | None] = mapped_column(Integer)
    authorized_percentage_at_launch: Mapped[int | None] = mapped_column(Integer)
    rollout_authorization_generation: Mapped[int | None] = mapped_column(Integer)
    validation_provider_safety_state_version: Mapped[int | None] = mapped_column(
        Integer
    )
    extraction_provider_safety_snapshot_set_hmac: Mapped[str | None] = (
        mapped_column(String(64))
    )
    rollout_control_epoch: Mapped[int | None] = mapped_column(Integer)
    selected_extraction_agent_count: Mapped[int | None] = mapped_column(Integer)
    extraction_plan_set_hmac: Mapped[str | None] = mapped_column(String(64))
    extraction_max_input_chars_per_agent: Mapped[int | None] = mapped_column(Integer)
    extraction_max_input_tokens_per_agent: Mapped[int | None] = mapped_column(
        Integer
    )
    extraction_max_output_tokens_per_agent: Mapped[int | None] = mapped_column(
        Integer
    )
    extraction_max_candidates_per_agent: Mapped[int | None] = mapped_column(
        Integer
    )
    confirmed_extraction_cost_ceiling_usd: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    confirmed_validation_cost_ceiling_usd: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    confirmed_total_cost_ceiling_usd: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )
    auto_review_budget_limit_usd: Mapped[float | None] = mapped_column(
        Numeric(12, 6)
    )


class AgentWorkflowEvidenceRef(Base):
    __tablename__ = 'agent_workflow_evidence_refs'
    __table_args__ = (
        UniqueConstraint(
            'workflow_thread_id',
            'ordinal',
            name='uq_agent_workflow_evidence_ref_ordinal',
        ),
        UniqueConstraint(
            'id',
            'workflow_thread_id',
            name='uq_agent_workflow_evidence_ref_id_workflow',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_thread_id: Mapped[str] = mapped_column(String(64), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    canonical_source_type: Mapped[str] = mapped_column(String(32))
    canonical_table: Mapped[str] = mapped_column(String(64))
    canonical_row_id: Mapped[int] = mapped_column(Integer)
    document_version_id: Mapped[int | None] = mapped_column(Integer)
    external_revision: Mapped[str | None] = mapped_column(String(255))
    content_signature: Mapped[str] = mapped_column(String(128))
    permission_level_snapshot: Mapped[str] = mapped_column(String(32))
    content_fingerprint: Mapped[str] = mapped_column(String(64))


class AgentRuntimeSchemaVersion(Base):
    __tablename__ = 'agent_runtime_schema_versions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(64), unique=True)
    package_name: Mapped[str] = mapped_column(String(128))
    package_version: Mapped[str] = mapped_column(String(32))
    schema_revision: Mapped[int] = mapped_column(Integer)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
