from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class AutoReviewRuntimeKeyState(Base):
    __tablename__ = 'auto_review_runtime_key_states'
    __table_args__ = (
        UniqueConstraint('component', name='uq_auto_review_runtime_key_component'),
        CheckConstraint(
            'generation >= 1', name='ck_auto_review_runtime_key_generation'
        ),
        CheckConstraint(
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_auto_review_runtime_key_material_verifier',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    generation: Mapped[int] = mapped_column(Integer)
    ready: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewProviderSafetyState(Base):
    __tablename__ = 'auto_review_provider_safety_states'
    __table_args__ = (
        UniqueConstraint(
            'purpose',
            'provider',
            'model',
            'reasoning_effort',
            name='uq_auto_review_provider_safety_identity',
        ),
        UniqueConstraint(
            'id',
            'purpose',
            'provider',
            'model',
            'reasoning_effort',
            name='uq_auto_review_provider_safety_parent_identity',
        ),
        UniqueConstraint(
            'id', name='uq_auto_review_provider_safety_state_id'
        ),
        CheckConstraint(
            "purpose IN ('extraction', 'validation')",
            name='ck_auto_review_provider_safety_purpose',
        ),
        CheckConstraint(
            'state_version >= 1',
            name='ck_auto_review_provider_safety_state_version',
        ),
        CheckConstraint(
            'overrun_count >= 0',
            name='ck_auto_review_provider_safety_overrun_count',
        ),
        CheckConstraint(
            '(last_event_id IS NULL AND last_event_sequence = 0) OR '
            '(last_event_id IS NOT NULL AND last_event_sequence > 0)',
            name='ck_auto_review_provider_safety_last_event',
        ),
        ForeignKeyConstraint(
            ['last_event_id', 'id'],
            [
                'auto_review_provider_safety_events.id',
                'auto_review_provider_safety_events.provider_safety_state_id',
            ],
            name='fk_auto_review_provider_safety_last_event',
            use_alter=True,
            deferrable=True,
            initially='DEFERRED',
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement='ignore_fk'
    )
    purpose: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(120))
    model: Mapped[str] = mapped_column(String(120))
    reasoning_effort: Mapped[str] = mapped_column(String(32))
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    authorized_cost_policy_version: Mapped[str] = mapped_column(String(64))
    token_estimator_version: Mapped[str] = mapped_column(String(64))
    tokenizer_encoding: Mapped[str] = mapped_column(String(64))
    reply_priming_tokens: Mapped[int] = mapped_column(Integer)
    framing_safety_tokens: Mapped[int] = mapped_column(Integer)
    input_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    output_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    breaker_open: Mapped[bool] = mapped_column(Boolean, default=False)
    breaker_reason_code: Mapped[str | None] = mapped_column(String(64))
    overrun_count: Mapped[int] = mapped_column(Integer, default=0)
    last_overrun_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    last_overrun_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    regression_gate_reference: Mapped[str | None] = mapped_column(String(120))
    authorized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_event_sequence: Mapped[int] = mapped_column(Integer, default=0)
    last_event_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewProviderSafetyEvent(Base):
    __tablename__ = 'auto_review_provider_safety_events'
    __table_args__ = (
        UniqueConstraint(
            'provider_safety_state_id',
            'event_sequence',
            name='uq_auto_review_provider_safety_event_sequence',
        ),
        UniqueConstraint(
            'id',
            'provider_safety_state_id',
            name='uq_auto_review_provider_safety_event_parent',
        ),
        ForeignKeyConstraint(
            [
                'provider_safety_state_id',
                'purpose',
                'provider',
                'model',
                'reasoning_effort',
            ],
            [
                'auto_review_provider_safety_states.id',
                'auto_review_provider_safety_states.purpose',
                'auto_review_provider_safety_states.provider',
                'auto_review_provider_safety_states.model',
                'auto_review_provider_safety_states.reasoning_effort',
            ],
            name='fk_auto_review_provider_safety_event_parent_identity',
            ondelete='RESTRICT',
        ),
        CheckConstraint(
            "event_kind IN ('initial_authorized', 'budget_overrun', "
            "'breaker_cleared')",
            name='ck_auto_review_provider_safety_events_kind',
        ),
        CheckConstraint(
            'event_sequence > 0 AND prior_state_version >= 0 AND '
            'new_state_version > prior_state_version',
            name='ck_auto_review_provider_safety_event_sequence_versions',
        ),
        CheckConstraint(
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_auto_review_provider_safety_event_key_material',
        ),
        CheckConstraint(
            "(event_kind = 'budget_overrun' AND length(call_hmac) = 64 AND "
            'actor_subject_hmac IS NULL) OR '
            "(event_kind IN ('initial_authorized', 'breaker_cleared') AND "
            'length(actor_subject_hmac) = 64 AND call_hmac IS NULL)',
            name='ck_auto_review_provider_safety_event_actor_or_call',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_safety_state_id: Mapped[int] = mapped_column(Integer)
    purpose: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(120))
    model: Mapped[str] = mapped_column(String(120))
    reasoning_effort: Mapped[str] = mapped_column(String(32))
    event_sequence: Mapped[int] = mapped_column(Integer)
    event_kind: Mapped[str] = mapped_column(String(32))
    prior_state_version: Mapped[int] = mapped_column(Integer)
    new_state_version: Mapped[int] = mapped_column(Integer)
    cost_policy_version: Mapped[str] = mapped_column(String(64))
    token_estimator_version: Mapped[str] = mapped_column(String(64))
    tokenizer_encoding: Mapped[str] = mapped_column(String(64))
    reply_priming_tokens: Mapped[int] = mapped_column(Integer)
    framing_safety_tokens: Mapped[int] = mapped_column(Integer)
    input_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    output_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    prior_breaker_open: Mapped[bool] = mapped_column(Boolean)
    new_breaker_open: Mapped[bool] = mapped_column(Boolean)
    reason_code: Mapped[str | None] = mapped_column(String(64))
    regression_gate_reference: Mapped[str | None] = mapped_column(String(120))
    actor_subject_hmac: Mapped[str | None] = mapped_column(String(64))
    call_hmac: Mapped[str | None] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrustedKnowledgeFingerprintProjectionState(Base):
    __tablename__ = 'trusted_knowledge_fingerprint_projection_states'
    __table_args__ = (
        UniqueConstraint(
            'component', name='uq_trusted_fingerprint_projection_component'
        ),
        CheckConstraint(
            'generation >= 1', name='ck_trusted_fingerprint_projection_generation'
        ),
        CheckConstraint(
            'source_active_count >= 0 AND projected_active_count >= 0',
            name='ck_trusted_fingerprint_projection_counts',
        ),
        CheckConstraint(
            '(source_checksum IS NULL OR length(source_checksum) = 64) AND '
            '(projected_checksum IS NULL OR length(projected_checksum) = 64)',
            name='ck_trusted_fingerprint_projection_checksums',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(64))
    projection_schema_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    generation: Mapped[int] = mapped_column(Integer)
    ready: Mapped[bool] = mapped_column(Boolean, default=False)
    source_active_count: Mapped[int] = mapped_column(Integer, default=0)
    projected_active_count: Mapped[int] = mapped_column(Integer, default=0)
    source_checksum: Mapped[str | None] = mapped_column(String(64))
    projected_checksum: Mapped[str | None] = mapped_column(String(64))
    rebuild_required: Mapped[bool] = mapped_column(Boolean, default=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrustedKnowledgeFingerprint(Base):
    __tablename__ = 'trusted_knowledge_fingerprints'
    __table_args__ = (
        UniqueConstraint(
            'knowledge_type',
            'knowledge_id',
            name='uq_trusted_knowledge_fingerprint_target',
        ),
        CheckConstraint(
            "scope_resolution IN ('exact', 'legacy_unknown')",
            name='ck_trusted_knowledge_fingerprint_scope_resolution',
        ),
        Index(
            'ix_trusted_knowledge_fingerprint_collision_lookup',
            'security_scope_id',
            'knowledge_type',
            'project_scope_hmac',
            'normalized_title_bucket_hmac',
            'review_status',
            'permission_level',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    knowledge_type: Mapped[str] = mapped_column(String(32))
    knowledge_id: Mapped[int] = mapped_column(Integer)
    security_scope_id: Mapped[str | None] = mapped_column(String(128))
    scope_resolution: Mapped[str] = mapped_column(String(32))
    project_scope_hmac: Mapped[str] = mapped_column(String(64))
    normalized_title_bucket_hmac: Mapped[str] = mapped_column(String(64))
    normalized_claim_fingerprint: Mapped[str | None] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    permission_level: Mapped[str] = mapped_column(String(32))
    review_status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ReviewItemEvidenceRef(Base):
    __tablename__ = 'review_item_evidence_refs'
    __table_args__ = (
        UniqueConstraint(
            'review_item_id',
            'workflow_evidence_ref_id',
            name='uq_review_item_evidence_ref_source',
        ),
        UniqueConstraint(
            'review_item_id',
            'candidate_slot_ordinal',
            name='uq_review_item_evidence_ref_slot',
        ),
        ForeignKeyConstraint(
            ['review_item_id', 'workflow_thread_id'],
            ['review_items.id', 'review_items.workflow_thread_id'],
            name='fk_review_item_evidence_refs_same_review_item_workflow',
        ),
        ForeignKeyConstraint(
            ['workflow_evidence_ref_id', 'workflow_thread_id'],
            [
                'agent_workflow_evidence_refs.id',
                'agent_workflow_evidence_refs.workflow_thread_id',
            ],
            name='fk_review_item_evidence_refs_same_evidence_workflow',
        ),
        CheckConstraint(
            'candidate_slot_ordinal >= 1',
            name='ck_review_item_evidence_ref_slot_positive',
        ),
        CheckConstraint(
            'length(message_content_fingerprint) = 64 AND '
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_review_item_evidence_ref_keyed_identity',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_item_id: Mapped[int] = mapped_column(Integer)
    workflow_thread_id: Mapped[str] = mapped_column(String(64))
    workflow_evidence_ref_id: Mapped[int] = mapped_column(Integer)
    candidate_slot_ordinal: Mapped[int] = mapped_column(Integer)
    message_content_fingerprint: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewExtractionCall(Base):
    __tablename__ = 'auto_review_extraction_calls'
    __table_args__ = (
        UniqueConstraint('agent_run_id', name='uq_auto_review_extraction_call_run'),
        UniqueConstraint(
            'workflow_thread_id',
            'agent_name',
            name='uq_auto_review_extraction_call_workflow_agent',
        ),
        UniqueConstraint(
            'workflow_thread_id',
            'agent_name',
            'extraction_plan_hmac',
            name='uq_auto_review_extraction_call_plan',
        ),
        ForeignKeyConstraint(
            ['agent_run_id', 'workflow_thread_id', 'agent_name'],
            ['agent_runs.id', 'agent_runs.workflow_thread_id', 'agent_runs.agent_name'],
            name='fk_auto_review_extraction_call_agent_run_identity',
        ),
        CheckConstraint(
            "status IN ('claimed', 'completed', 'failed')",
            name='ck_auto_review_extraction_calls_status',
        ),
        CheckConstraint(
            'max_provider_attempts = 1',
            name='ck_auto_review_extraction_calls_one_attempt',
        ),
        CheckConstraint(
            'provider_attempt_count IN (0, 1)',
            name='ck_auto_review_extraction_calls_attempt_count',
        ),
        CheckConstraint(
            'reserved_input_tokens >= 0 AND reserved_output_tokens >= 0 AND '
            'reserved_cost_usd >= 0 AND charged_cost_usd >= 0 AND '
            'budget_overrun_cost_usd >= 0',
            name='ck_auto_review_extraction_calls_nonnegative_charge',
        ),
        CheckConstraint(
            'result_candidate_count IS NULL OR result_candidate_count IN (0, 1)',
            name='ck_auto_review_extraction_calls_candidate_count',
        ),
        CheckConstraint(
            "(status != 'completed' AND result_kind IS NULL AND "
            'result_candidate_count IS NULL AND result_candidate_set_hmac IS NULL) '
            "OR (status = 'completed' AND result_kind = 'no_candidate' AND "
            'result_candidate_count = 0 AND length(result_candidate_set_hmac) = 64) '
            "OR (status = 'completed' AND result_kind = 'candidate' AND "
            'result_candidate_count = 1 AND length(result_candidate_set_hmac) = 64)',
            name='ck_auto_review_extraction_calls_terminal_result',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_run_id: Mapped[int] = mapped_column(Integer)
    workflow_thread_id: Mapped[str] = mapped_column(String(64))
    agent_name: Mapped[str] = mapped_column(String(64))
    extraction_plan_hmac: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(120))
    model: Mapped[str] = mapped_column(String(120))
    reasoning_effort: Mapped[str] = mapped_column(String(32))
    route_version: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    output_contract_version: Mapped[str] = mapped_column(String(64))
    extraction_registry_version: Mapped[str] = mapped_column(String(64))
    cost_policy_version: Mapped[str] = mapped_column(String(64))
    provider_safety_state_version: Mapped[int] = mapped_column(Integer)
    token_estimator_version: Mapped[str] = mapped_column(String(64))
    tokenizer_encoding: Mapped[str] = mapped_column(String(64))
    reply_priming_tokens: Mapped[int] = mapped_column(Integer)
    framing_safety_tokens: Mapped[int] = mapped_column(Integer)
    prepared_content_hmac: Mapped[str] = mapped_column(String(64))
    prepared_character_count: Mapped[int] = mapped_column(Integer)
    framed_input_token_cap: Mapped[int] = mapped_column(Integer)
    total_output_token_cap: Mapped[int] = mapped_column(Integer)
    max_candidates_per_agent: Mapped[int] = mapped_column(Integer, default=1)
    input_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    output_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    provider_timeout_seconds: Mapped[int] = mapped_column(Integer)
    provider_send_start_window_seconds: Mapped[int] = mapped_column(Integer)
    provider_attempt_lease_seconds: Mapped[int] = mapped_column(Integer)
    provider_commit_grace_seconds: Mapped[int] = mapped_column(Integer)
    workflow_extraction_cost_ceiling_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6)
    )
    workflow_total_cost_ceiling_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    status: Mapped[str] = mapped_column(String(32))
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    max_provider_attempts: Mapped[int] = mapped_column(Integer, default=1)
    provider_attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    attempt_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reserved_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reserved_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=0)
    charged_input_tokens: Mapped[int | None] = mapped_column(Integer)
    charged_output_tokens: Mapped[int | None] = mapped_column(Integer)
    charged_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=0)
    budget_overrun: Mapped[bool] = mapped_column(Boolean, default=False)
    budget_overrun_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), default=0
    )
    result_kind: Mapped[str | None] = mapped_column(String(32))
    result_candidate_count: Mapped[int | None] = mapped_column(Integer)
    result_candidate_set_hmac: Mapped[str | None] = mapped_column(String(64))
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AutoReviewValidationCall(Base):
    __tablename__ = 'auto_review_validation_calls'
    __table_args__ = (
        UniqueConstraint(
            'workflow_thread_id',
            'batch_fingerprint',
            name='uq_auto_review_validation_call_batch',
        ),
        UniqueConstraint(
            'id',
            'workflow_thread_id',
            name='uq_auto_review_validation_call_id_workflow',
        ),
        CheckConstraint(
            "status IN ('claimed', 'completed', 'failed')",
            name='ck_auto_review_validation_calls_status',
        ),
        CheckConstraint(
            'max_provider_attempts = 1',
            name='ck_auto_review_validation_calls_one_attempt',
        ),
        CheckConstraint(
            'provider_attempt_count IN (0, 1)',
            name='ck_auto_review_validation_calls_attempt_count',
        ),
        CheckConstraint(
            'candidate_count BETWEEN 1 AND 4 AND reserved_input_tokens >= 0 AND '
            'reserved_output_tokens >= 0 AND reserved_cost_usd >= 0 AND '
            'charged_cost_usd >= 0 AND '
            '(charged_input_tokens IS NULL OR charged_input_tokens >= 0) AND '
            '(charged_output_tokens IS NULL OR charged_output_tokens >= 0) AND '
            'budget_overrun_cost_usd >= 0',
            name='ck_auto_review_validation_calls_nonnegative_charge',
        ),
        CheckConstraint(
            "(status = 'claimed' AND terminal_at IS NULL AND "
            'charged_input_tokens IS NULL AND charged_output_tokens IS NULL AND '
            'charged_cost_usd = 0 AND budget_overrun = false) OR '
            "(status = 'completed' AND provider_attempt_count = 1 AND "
            'attempt_started_at IS NOT NULL AND terminal_at IS NOT NULL AND '
            'charged_input_tokens IS NOT NULL AND charged_output_tokens IS NOT NULL) '
            "OR (status = 'failed' AND terminal_at IS NOT NULL AND "
            '((provider_attempt_count = 0 AND attempt_started_at IS NULL AND '
            'charged_input_tokens = 0 AND charged_output_tokens = 0 AND '
            'charged_cost_usd = 0 AND budget_overrun = false) OR '
            '(provider_attempt_count = 1 AND attempt_started_at IS NOT NULL AND '
            'charged_input_tokens IS NOT NULL AND charged_output_tokens IS NOT NULL)))',
            name='ck_auto_review_validation_calls_terminal_charge',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workflow_thread_id: Mapped[str] = mapped_column(String(64))
    batch_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    candidate_count: Mapped[int] = mapped_column(Integer)
    max_provider_attempts: Mapped[int] = mapped_column(Integer, default=1)
    provider_attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    attempt_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reserved_input_tokens: Mapped[int] = mapped_column(Integer)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer)
    reserved_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    charged_input_tokens: Mapped[int | None] = mapped_column(Integer)
    charged_output_tokens: Mapped[int | None] = mapped_column(Integer)
    charged_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=0)
    prepared_content_hmac: Mapped[str] = mapped_column(String(64))
    serialized_character_count: Mapped[int] = mapped_column(Integer)
    framed_input_token_count: Mapped[int] = mapped_column(Integer)
    max_output_tokens: Mapped[int] = mapped_column(Integer)
    token_estimator_version: Mapped[str] = mapped_column(String(64))
    tokenizer_encoding: Mapped[str] = mapped_column(String(64))
    reply_priming_tokens: Mapped[int] = mapped_column(Integer)
    framing_safety_tokens: Mapped[int] = mapped_column(Integer)
    input_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    output_usd_per_1m: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    cost_policy_version: Mapped[str] = mapped_column(String(64))
    provider_safety_state_version: Mapped[int] = mapped_column(Integer)
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    workflow_extraction_cost_ceiling_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6)
    )
    workflow_validation_cost_ceiling_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6)
    )
    workflow_total_cost_ceiling_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    provider_timeout_seconds: Mapped[int] = mapped_column(Integer)
    provider_send_start_window_seconds: Mapped[int] = mapped_column(Integer)
    provider_attempt_lease_seconds: Mapped[int] = mapped_column(Integer)
    provider_commit_grace_seconds: Mapped[int] = mapped_column(Integer)
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    budget_overrun: Mapped[bool] = mapped_column(Boolean, default=False)
    budget_overrun_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), default=0
    )
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AutoReviewValidation(Base):
    __tablename__ = 'auto_review_validations'
    __table_args__ = (
        UniqueConstraint(
            'workflow_thread_id',
            'validation_key',
            name='uq_auto_review_validation_workflow_key',
        ),
        UniqueConstraint(
            'review_item_id',
            'id',
            name='uq_auto_review_validation_review_item_id',
        ),
        ForeignKeyConstraint(
            ['validation_call_id', 'workflow_thread_id'],
            [
                'auto_review_validation_calls.id',
                'auto_review_validation_calls.workflow_thread_id',
            ],
            name='fk_auto_review_validation_call_same_workflow',
        ),
        ForeignKeyConstraint(
            ['review_item_id', 'workflow_thread_id'],
            ['review_items.id', 'review_items.workflow_thread_id'],
            name='fk_auto_review_validation_review_item_same_workflow',
        ),
        CheckConstraint(
            "status IN ('claimed', 'completed', 'failed')",
            name='ck_auto_review_validations_status',
        ),
        CheckConstraint(
            'estimated_cost_usd >= 0 AND input_tokens >= 0 AND output_tokens >= 0',
            name='ck_auto_review_validations_nonnegative_cost',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_item_id: Mapped[int] = mapped_column(Integer)
    validation_call_id: Mapped[int] = mapped_column(Integer)
    workflow_thread_id: Mapped[str] = mapped_column(String(64))
    validation_key: Mapped[str] = mapped_column(String(64))
    evidence_version_hash: Mapped[str] = mapped_column(String(64))
    candidate_generation_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    validator_provider: Mapped[str] = mapped_column(String(120))
    validator_model: Mapped[str] = mapped_column(String(120))
    reasoning_effort: Mapped[str] = mapped_column(String(32))
    validator_prompt_version: Mapped[str] = mapped_column(String(64))
    validator_output_contract_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    cost_policy_version: Mapped[str] = mapped_column(String(64))
    confirmed_validation_cost_ceiling_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6)
    )
    claim_results: Mapped[list] = mapped_column(MutableList.as_mutable(JSON), default=list)
    minimum_entailment_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    uncertainty_codes: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON), default=list
    )
    conflict_codes: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON), default=list
    )
    policy_decision: Mapped[str | None] = mapped_column(String(32))
    policy_reason_codes: Mapped[list] = mapped_column(
        MutableList.as_mutable(JSON), default=list
    )
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    shadow_comparison_status: Mapped[str | None] = mapped_column(String(32))
    shadow_human_resolution: Mapped[str | None] = mapped_column(String(32))
    shadow_exclusion_code: Mapped[str | None] = mapped_column(String(64))
    shadow_compared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TrustedKnowledgeApprovalLink(Base):
    __tablename__ = 'trusted_knowledge_approval_links'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'knowledge_type',
            'knowledge_id',
            name='uq_trusted_knowledge_approval_link_target',
        ),
        UniqueConstraint(
            'knowledge_type',
            'knowledge_id',
            'review_item_id',
            'promotion_effect_kind',
            name='uq_trusted_knowledge_approval_effect',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    knowledge_type: Mapped[str] = mapped_column(String(32))
    knowledge_id: Mapped[int] = mapped_column(Integer)
    review_item_id: Mapped[int] = mapped_column(ForeignKey('review_items.id'))
    security_scope_id: Mapped[str] = mapped_column(String(128))
    promotion_effect_kind: Mapped[str] = mapped_column(String(32))
    resolution_source: Mapped[str] = mapped_column(String(32))
    claim_fingerprint: Mapped[str] = mapped_column(String(64))
    permission_level: Mapped[str] = mapped_column(String(32))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrustedKnowledgeEvidenceLink(Base):
    __tablename__ = 'trusted_knowledge_evidence_links'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'approval_link_id',
            name='uq_trusted_knowledge_evidence_link_parent',
        ),
        UniqueConstraint(
            'approval_link_id',
            'canonical_source_kind',
            'canonical_source_id',
            'canonical_version_or_signature',
            'evidence_hash',
            name='uq_trusted_knowledge_evidence_identity',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    approval_link_id: Mapped[int] = mapped_column(
        ForeignKey('trusted_knowledge_approval_links.id', ondelete='RESTRICT')
    )
    canonical_source_kind: Mapped[str] = mapped_column(String(32))
    canonical_source_id: Mapped[str] = mapped_column(String(128))
    canonical_version_or_signature: Mapped[str] = mapped_column(String(255))
    evidence_hash: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AssistantMessageEvidenceDependency(Base):
    __tablename__ = 'assistant_message_evidence_dependencies'
    __table_args__ = (
        UniqueConstraint(
            'assistant_message_id',
            'candidate_ordinal',
            name='uq_assistant_message_dependency_ordinal',
        ),
        UniqueConstraint(
            'assistant_message_id',
            'serving_document_id',
            name='uq_assistant_message_dependency_serving_document',
        ),
        UniqueConstraint(
            'id',
            'assistant_message_id',
            'approval_link_id',
            name='uq_assistant_message_dependency_effect',
        ),
        ForeignKeyConstraint(
            ['document_chunk_id', 'document_version_id', 'source_id', 'parser_run_id'],
            [
                'document_chunks.id',
                'document_chunks.version_id',
                'document_chunks.source_id',
                'document_chunks.parser_run_id',
            ],
            name='fk_assistant_message_dependency_raw_chunk_identity',
        ),
        ForeignKeyConstraint(
            ['approval_link_id', 'knowledge_type', 'knowledge_id'],
            [
                'trusted_knowledge_approval_links.id',
                'trusted_knowledge_approval_links.knowledge_type',
                'trusted_knowledge_approval_links.knowledge_id',
            ],
            name='fk_assistant_message_dependency_approval_target',
        ),
        CheckConstraint(
            "dependency_kind IN ('raw_chunk', 'trusted_knowledge')",
            name='ck_assistant_message_dependency_kind',
        ),
        CheckConstraint(
            'length(serving_content_hash) = 64',
            name='ck_assistant_message_dependency_content_hash',
        ),
        CheckConstraint(
            "permission_level IN ('public', 'internal', 'restricted')",
            name='ck_assistant_message_dependency_permission',
        ),
        CheckConstraint(
            "(dependency_kind = 'raw_chunk' AND document_chunk_id IS NOT NULL AND "
            'document_version_id IS NOT NULL AND source_id IS NOT NULL AND '
            "parser_run_id IS NOT NULL AND server_content_signature_schema = "
            "'server-source-content:v1' AND length(server_content_signature) = 64 "
            'AND knowledge_type IS NULL AND knowledge_id IS NULL AND '
            'approval_link_id IS NULL AND legacy_human_base = false) OR '
            "(dependency_kind = 'trusted_knowledge' AND document_chunk_id IS NULL "
            'AND document_version_id IS NULL AND source_id IS NULL AND '
            'parser_run_id IS NULL AND server_content_signature_schema IS NULL AND '
            'server_content_signature IS NULL AND knowledge_type IS NOT NULL AND '
            'knowledge_id IS NOT NULL AND ((approval_link_id IS NOT NULL AND '
            'legacy_human_base = false) OR (approval_link_id IS NULL AND '
            'legacy_human_base = true)))',
            name='ck_assistant_message_dependency_exact_kind',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assistant_message_id: Mapped[int] = mapped_column(
        ForeignKey('assistant_messages.id', ondelete='RESTRICT')
    )
    candidate_ordinal: Mapped[int] = mapped_column(Integer)
    serving_document_id: Mapped[str] = mapped_column(String(160))
    dependency_kind: Mapped[str] = mapped_column(String(32))
    dependency_set_hmac: Mapped[str] = mapped_column(String(64))
    serving_content_hash: Mapped[str] = mapped_column(String(64))
    permission_level: Mapped[str] = mapped_column(String(32))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    document_chunk_id: Mapped[int | None] = mapped_column(Integer)
    document_version_id: Mapped[int | None] = mapped_column(Integer)
    source_id: Mapped[int | None] = mapped_column(Integer)
    parser_run_id: Mapped[int | None] = mapped_column(Integer)
    current_document_version_id: Mapped[int | None] = mapped_column(Integer)
    server_content_signature_schema: Mapped[str | None] = mapped_column(String(64))
    server_content_signature: Mapped[str | None] = mapped_column(String(64))
    parser_policy_version: Mapped[str | None] = mapped_column(String(64))
    parser_version: Mapped[str | None] = mapped_column(String(64))
    chunk_policy_version: Mapped[str | None] = mapped_column(String(64))
    knowledge_type: Mapped[str | None] = mapped_column(String(32))
    knowledge_id: Mapped[int | None] = mapped_column(Integer)
    approval_link_id: Mapped[int | None] = mapped_column(Integer)
    legacy_human_base: Mapped[bool] = mapped_column(Boolean, default=False)
    legacy_source_review_item_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AssistantMessageKnowledgeEvidenceRef(Base):
    __tablename__ = 'assistant_message_knowledge_evidence_refs'
    __table_args__ = (
        UniqueConstraint(
            'dependency_id',
            'trusted_knowledge_evidence_link_id',
            name='uq_assistant_message_knowledge_evidence_ref',
        ),
        ForeignKeyConstraint(
            ['dependency_id', 'assistant_message_id', 'approval_link_id'],
            [
                'assistant_message_evidence_dependencies.id',
                'assistant_message_evidence_dependencies.assistant_message_id',
                'assistant_message_evidence_dependencies.approval_link_id',
            ],
            name='fk_assistant_knowledge_evidence_ref_same_dependency_effect',
        ),
        ForeignKeyConstraint(
            ['trusted_knowledge_evidence_link_id', 'approval_link_id'],
            [
                'trusted_knowledge_evidence_links.id',
                'trusted_knowledge_evidence_links.approval_link_id',
            ],
            name='fk_assistant_knowledge_evidence_ref_same_approval_effect',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dependency_id: Mapped[int] = mapped_column(Integer)
    assistant_message_id: Mapped[int] = mapped_column(Integer)
    approval_link_id: Mapped[int] = mapped_column(Integer)
    trusted_knowledge_evidence_link_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewRolloutState(Base):
    __tablename__ = 'auto_review_rollout_states'
    __table_args__ = (
        UniqueConstraint(
            'security_scope_id',
            'policy_version',
            name='uq_auto_review_rollout_scope_policy',
        ),
        UniqueConstraint(
            'id',
            'security_scope_id',
            'policy_version',
            name='uq_auto_review_rollout_parent_identity',
        ),
        UniqueConstraint('id', name='uq_auto_review_rollout_state_id'),
        ForeignKeyConstraint(
            ['last_event_id', 'id'],
            [
                'auto_review_rollout_control_events.id',
                'auto_review_rollout_control_events.rollout_state_id',
            ],
            name='fk_auto_review_rollout_last_event',
            use_alter=True,
            deferrable=True,
            initially='DEFERRED',
        ),
        CheckConstraint(
            'state_version >= 0 AND control_epoch >= 0',
            name='ck_auto_review_rollout_versions',
        ),
        CheckConstraint(
            'max_authorized_percentage IN (0, 10, 100)',
            name='ck_auto_review_rollout_authorized_percentage',
        ),
        CheckConstraint(
            'authorization_generation >= 0 AND shadow_predicted_count >= 0 AND '
            'shadow_completed_count >= 0 AND shadow_supported_count >= 0 AND '
            'enforce_promotion_ordinal >= 0 AND post_audit_selected_count >= 0 AND '
            'post_audit_completed_count >= 0 AND post_audit_critical_count >= 0 AND '
            'confirmed_mandatory_audit_count >= 0 AND '
            'pending_mandatory_audit_count >= 0 AND '
            'invalidated_before_audit_count >= 0 AND corrected_critical_count >= 0',
            name='ck_auto_review_rollout_nonnegative_counters',
        ),
        CheckConstraint(
            'confirmed_mandatory_audit_count <= 50 AND '
            'pending_mandatory_audit_count <= 50 AND '
            'confirmed_mandatory_audit_count + pending_mandatory_audit_count <= 50',
            name='ck_auto_review_rollout_mandatory_slots',
        ),
        CheckConstraint(
            '(last_event_id IS NULL AND last_event_sequence = 0) OR '
            '(last_event_id IS NOT NULL AND last_event_sequence > 0)',
            name='ck_auto_review_rollout_last_event',
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement='ignore_fk'
    )
    security_scope_id: Mapped[str] = mapped_column(String(128))
    policy_version: Mapped[str] = mapped_column(String(64))
    state_version: Mapped[int] = mapped_column(Integer, default=0)
    control_epoch: Mapped[int] = mapped_column(Integer, default=0)
    shadow_predicted_count: Mapped[int] = mapped_column(Integer, default=0)
    shadow_completed_count: Mapped[int] = mapped_column(Integer, default=0)
    shadow_supported_count: Mapped[int] = mapped_column(Integer, default=0)
    enforce_promotion_ordinal: Mapped[int] = mapped_column(Integer, default=0)
    post_audit_selected_count: Mapped[int] = mapped_column(Integer, default=0)
    post_audit_completed_count: Mapped[int] = mapped_column(Integer, default=0)
    post_audit_critical_count: Mapped[int] = mapped_column(Integer, default=0)
    confirmed_mandatory_audit_count: Mapped[int] = mapped_column(Integer, default=0)
    pending_mandatory_audit_count: Mapped[int] = mapped_column(Integer, default=0)
    invalidated_before_audit_count: Mapped[int] = mapped_column(Integer, default=0)
    corrected_critical_count: Mapped[int] = mapped_column(Integer, default=0)
    breaker_open: Mapped[bool] = mapped_column(Boolean, default=False)
    breaker_reason_code: Mapped[str | None] = mapped_column(String(64))
    breaker_opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    max_authorized_percentage: Mapped[int] = mapped_column(Integer, default=0)
    authorization_generation: Mapped[int] = mapped_column(Integer, default=0)
    authorization_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    regression_gate_reference: Mapped[str | None] = mapped_column(String(120))
    last_event_sequence: Mapped[int] = mapped_column(Integer, default=0)
    last_event_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewRolloutControlEvent(Base):
    __tablename__ = 'auto_review_rollout_control_events'
    __table_args__ = (
        UniqueConstraint(
            'rollout_state_id',
            'event_sequence',
            name='uq_auto_review_rollout_event_sequence',
        ),
        UniqueConstraint(
            'id',
            'rollout_state_id',
            name='uq_auto_review_rollout_event_parent',
        ),
        ForeignKeyConstraint(
            ['rollout_state_id', 'security_scope_id', 'policy_version'],
            [
                'auto_review_rollout_states.id',
                'auto_review_rollout_states.security_scope_id',
                'auto_review_rollout_states.policy_version',
            ],
            name='fk_auto_review_rollout_event_parent_identity',
            ondelete='RESTRICT',
        ),
        CheckConstraint(
            "event_kind IN ('percentage_authorized', 'breaker_opened', "
            "'breaker_closed', 'generation_invalidated')",
            name='ck_auto_review_rollout_control_events_kind',
        ),
        CheckConstraint(
            'event_sequence > 0 AND length(actor_subject_hmac) = 64 AND '
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_auto_review_rollout_event_actor_key',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rollout_state_id: Mapped[int] = mapped_column(Integer)
    security_scope_id: Mapped[str] = mapped_column(String(128))
    policy_version: Mapped[str] = mapped_column(String(64))
    event_sequence: Mapped[int] = mapped_column(Integer)
    event_kind: Mapped[str] = mapped_column(String(32))
    prior_state_version: Mapped[int] = mapped_column(Integer)
    new_state_version: Mapped[int] = mapped_column(Integer)
    prior_control_epoch: Mapped[int] = mapped_column(Integer)
    new_control_epoch: Mapped[int] = mapped_column(Integer)
    prior_max_authorized_percentage: Mapped[int] = mapped_column(Integer)
    new_max_authorized_percentage: Mapped[int] = mapped_column(Integer)
    prior_breaker_open: Mapped[bool] = mapped_column(Boolean)
    new_breaker_open: Mapped[bool] = mapped_column(Boolean)
    prior_authorization_generation: Mapped[int] = mapped_column(Integer)
    new_authorization_generation: Mapped[int] = mapped_column(Integer)
    reason_code: Mapped[str | None] = mapped_column(String(64))
    regression_gate_reference: Mapped[str | None] = mapped_column(String(120))
    actor_subject_hmac: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewPromotionDecision(Base):
    __tablename__ = 'auto_review_promotion_decisions'
    __table_args__ = (
        UniqueConstraint('review_item_id', name='uq_auto_review_promotion_review_item'),
        UniqueConstraint(
            'review_item_id', 'id', name='uq_auto_review_promotion_review_item_id'
        ),
        UniqueConstraint(
            'security_scope_id',
            'policy_version',
            'promotion_ordinal',
            name='uq_auto_review_promotion_scope_ordinal',
        ),
        ForeignKeyConstraint(
            ['security_scope_id', 'policy_version'],
            [
                'auto_review_rollout_states.security_scope_id',
                'auto_review_rollout_states.policy_version',
            ],
            name='fk_auto_review_promotion_rollout_scope',
        ),
        CheckConstraint(
            "selection_result IN ('first_50', 'sample_10', 'sample_2', "
            "'not_selected')",
            name='ck_auto_review_promotion_selection_result',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_item_id: Mapped[int] = mapped_column(ForeignKey('review_items.id'))
    security_scope_id: Mapped[str] = mapped_column(String(128))
    policy_version: Mapped[str] = mapped_column(String(64))
    rollout_authorization_generation: Mapped[int] = mapped_column(Integer)
    promotion_ordinal: Mapped[int] = mapped_column(Integer)
    requested_percentage: Mapped[int] = mapped_column(Integer)
    stored_percentage: Mapped[int] = mapped_column(Integer)
    authorized_percentage: Mapped[int] = mapped_column(Integer)
    enforce_selection_fingerprint: Mapped[str] = mapped_column(String(64))
    audit_selection_fingerprint: Mapped[str] = mapped_column(String(64))
    selection_result: Mapped[str] = mapped_column(String(32))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewPostAudit(Base):
    __tablename__ = 'auto_review_post_audits'
    __table_args__ = (
        UniqueConstraint('review_item_id', name='uq_auto_review_post_audit_item'),
        UniqueConstraint(
            'review_item_id', 'id', name='uq_auto_review_post_audit_item_id'
        ),
        UniqueConstraint(
            'promotion_decision_id', name='uq_auto_review_post_audit_decision'
        ),
        ForeignKeyConstraint(
            ['review_item_id', 'promotion_decision_id'],
            [
                'auto_review_promotion_decisions.review_item_id',
                'auto_review_promotion_decisions.id',
            ],
            name='fk_auto_review_post_audit_same_promotion',
        ),
        CheckConstraint(
            'audit_reason IS NULL OR length(audit_reason) BETWEEN 1 AND 500',
            name='ck_auto_review_post_audits_reason_length',
        ),
        CheckConstraint(
            "status IN ('pending', 'completed', 'remediation_required')",
            name='ck_auto_review_post_audits_status',
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('confirmed', 'incorrect', "
            "'permission_violation', 'source_version_violation', "
            "'policy_violation')",
            name='ck_auto_review_post_audits_outcome',
        ),
        CheckConstraint(
            "NOT (status = 'completed' AND outcome IS NULL) OR "
            "(system_resolution_code = 'source_invalidated_before_audit' AND "
            'audit_reason IS NULL AND auditor_subject_hmac IS NULL)',
            name='ck_auto_review_post_audits_terminal_outcome',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_item_id: Mapped[int] = mapped_column(Integer)
    promotion_decision_id: Mapped[int] = mapped_column(Integer)
    sample_cohort: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    outcome: Mapped[str | None] = mapped_column(String(32))
    system_resolution_code: Mapped[str | None] = mapped_column(String(64))
    remediation_code: Mapped[str | None] = mapped_column(String(64))
    auditor_subject_hmac: Mapped[str | None] = mapped_column(String(64))
    auditor_fingerprint_key_version: Mapped[str | None] = mapped_column(String(64))
    auditor_fingerprint_key_material_verifier: Mapped[str | None] = mapped_column(
        String(64)
    )
    audit_reason: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    audited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AutoReviewRevocationAssessment(Base):
    __tablename__ = 'auto_review_revocation_assessments'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'review_item_id',
            name='uq_auto_review_revocation_assessment_item_id',
        ),
        UniqueConstraint(
            'review_item_id', name='uq_auto_review_revocation_assessment_item'
        ),
        CheckConstraint(
            "reason_code IN ('business_withdrawal', 'incorrect_content', "
            "'permission_violation', 'wrong_source_version', 'policy_violation')",
            name='ck_auto_review_revocation_assessments_reason_code',
        ),
        CheckConstraint(
            'length(actor_subject_hmac) = 64 AND '
            'length(actor_fingerprint_key_material_verifier) = 64',
            name='ck_auto_review_revocation_assessment_actor_key',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_item_id: Mapped[int] = mapped_column(
        ForeignKey('review_items.id'), unique=True
    )
    reason_code: Mapped[str] = mapped_column(String(32))
    actor_subject_hmac: Mapped[str] = mapped_column(String(64))
    actor_fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    actor_fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AutoReviewAuditCorrection(Base):
    __tablename__ = 'auto_review_audit_corrections'
    __table_args__ = (
        UniqueConstraint('post_audit_id', name='uq_auto_review_correction_audit'),
        UniqueConstraint('review_item_id', name='uq_auto_review_correction_item'),
        UniqueConstraint('assessment_id', name='uq_auto_review_correction_assessment'),
        ForeignKeyConstraint(
            ['review_item_id', 'post_audit_id'],
            ['auto_review_post_audits.review_item_id', 'auto_review_post_audits.id'],
            name='fk_auto_review_correction_same_audit_item',
        ),
        ForeignKeyConstraint(
            ['assessment_id', 'review_item_id'],
            [
                'auto_review_revocation_assessments.id',
                'auto_review_revocation_assessments.review_item_id',
            ],
            name='fk_auto_review_correction_same_assessment_item',
        ),
        CheckConstraint(
            "effective_outcome IN ('incorrect', 'permission_violation', "
            "'source_version_violation', 'policy_violation')",
            name='ck_auto_review_audit_corrections_outcome',
        ),
        CheckConstraint(
            "status IN ('remediation_required', 'completed')",
            name='ck_auto_review_audit_corrections_status',
        ),
        CheckConstraint(
            "(status = 'remediation_required' AND system_resolution_code IN "
            "('revoke_pending', 'revoke_failed') AND completed_at IS NULL) OR "
            "(status = 'completed' AND system_resolution_code = 'revoked' AND "
            'completed_at IS NOT NULL)',
            name='ck_auto_review_audit_corrections_remediation',
        ),
        CheckConstraint(
            'length(actor_subject_hmac) = 64 AND '
            'length(actor_fingerprint_key_material_verifier) = 64',
            name='ck_auto_review_audit_correction_actor_key',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_audit_id: Mapped[int] = mapped_column(Integer)
    review_item_id: Mapped[int] = mapped_column(Integer)
    assessment_id: Mapped[int] = mapped_column(Integer)
    effective_outcome: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    system_resolution_code: Mapped[str] = mapped_column(String(32))
    actor_subject_hmac: Mapped[str] = mapped_column(String(64))
    actor_fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    actor_fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VectorServingTombstone(Base):
    __tablename__ = 'vector_serving_tombstones'
    __table_args__ = (
        UniqueConstraint('document_id', name='uq_vector_serving_tombstone_document'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(String(160))
    source_review_item_id: Mapped[int] = mapped_column(ForeignKey('review_items.id'))
    reason_code: Mapped[str] = mapped_column(String(64))
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
