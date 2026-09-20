from contextlib import nullcontext
from dataclasses import asdict, dataclass
from decimal import Decimal
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.agent_runtime.canonical_sources import (
    ResolvedSourceVersion,
    ReviewWorkflowPreflightError,
    build_keyed_fingerprint,
    resolve_source_versions,
)
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_versions import (
    SourceVersionRef,
    normalize_source_version_refs,
    review_batch_marker_is_v2,
)
from backend.app.models.agent_workflows import (
    AgentWorkflowEvidenceRef,
    AgentWorkflowRequest,
    AgentWorkflowThread,
)
from backend.app.models.source import Source
from backend.app.schemas.auto_review import COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
from backend.app.schemas.review_workflow import (
    COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
    COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
    COMPANY_MEMORY_REVIEW_WORKFLOW,
    COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    ReviewWorkflowRunRequest,
    normalize_agent_names,
)

EVIDENCE_VERSION_HASH_SCHEMA = 'review-evidence-versions:v1'
INPUT_HASH_POLICY = 'review-workflow-batch:v1'
CREATOR_LOCK_HASH_SCHEMA = 'review-creator-key:v1'
CREATOR_LOCK_HASH_POLICY = 'review-workflow-creator-lock:v1'

# SQLite/demo has no cross-process lock primitive. This lock only serializes
# preflight lookup/create operations within one Python process.
_SQLITE_PREFLIGHT_LOCK = RLock()


@dataclass(frozen=True)
class PreparedReviewRequest:
    source_refs: tuple[ResolvedSourceVersion, ...]
    agent_names: tuple[str, ...]
    input_hash: str
    evidence_version_hash: str
    selection_policy_version: str


@dataclass(frozen=True)
class V21PreparedReviewConfig:
    configured_auto_review_mode: str
    validator_provider: str
    validator_model: str
    validator_reasoning_effort: str
    validator_prompt_version: str
    validator_output_contract_version: str
    policy_version: str
    cost_policy_version: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    token_estimator_version: str
    tokenizer_encoding: str
    max_input_tokens_per_batch: int
    max_output_tokens_per_batch: int
    reply_priming_tokens: int
    framing_safety_tokens: int
    max_validation_batches_per_workflow: int
    max_validation_candidates_per_batch: int
    max_validation_candidates_per_workflow: int
    max_provider_attempts: int
    provider_timeout_seconds: int
    provider_send_start_window_seconds: int
    provider_attempt_lease_seconds: int
    provider_commit_grace_seconds: int
    validator_input_usd_per_1m: Decimal
    validator_output_usd_per_1m: Decimal
    enforce_percentage: int
    authorized_percentage_at_launch: int
    rollout_authorization_generation: int
    validation_provider_safety_state_version: int
    rollout_control_epoch: int
    extraction_plan_set_hmac: str
    extraction_provider_safety_snapshot_set_hmac: str
    confirmed_extraction_cost_ceiling_usd: Decimal
    confirmed_validation_cost_ceiling_usd: Decimal
    confirmed_total_cost_ceiling_usd: Decimal
    total_budget_limit_usd: Decimal
    extraction_plan_identities: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PreparedReviewRequestV21:
    source_refs: tuple[ResolvedSourceVersion, ...]
    agent_names: tuple[str, ...]
    input_hash: str
    evidence_version_hash: str
    selection_policy_version: str
    config: V21PreparedReviewConfig
    graph_version: str = COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21

    def stored_snapshot(self) -> dict[str, Any]:
        config = self.config
        extraction = _uniform_extraction_snapshot(config)
        return {
            'auto_review_mode': config.configured_auto_review_mode,
            'auto_review_validator_provider': config.validator_provider,
            'auto_review_validator_model': config.validator_model,
            'auto_review_reasoning_effort': config.validator_reasoning_effort,
            'auto_review_validator_prompt_version': config.validator_prompt_version,
            'auto_review_validator_output_contract_version': config.validator_output_contract_version,
            'auto_review_policy_version': config.policy_version,
            'auto_review_cost_policy_version': config.cost_policy_version,
            'auto_review_extraction_cost_policy_version': extraction[
                'cost_policy_version'
            ],
            'fingerprint_key_version': config.fingerprint_key_version,
            'fingerprint_key_material_verifier': config.fingerprint_key_material_verifier,
            'auto_review_token_estimator_version': config.token_estimator_version,
            'auto_review_extraction_token_estimator_version': extraction[
                'token_estimator_version'
            ],
            'auto_review_tokenizer_encoding': config.tokenizer_encoding,
            'auto_review_max_input_tokens': config.max_input_tokens_per_batch,
            'auto_review_max_output_tokens': config.max_output_tokens_per_batch,
            'auto_review_reply_priming_tokens': config.reply_priming_tokens,
            'auto_review_framing_safety_tokens': config.framing_safety_tokens,
            'auto_review_max_batches_per_workflow': config.max_validation_batches_per_workflow,
            'auto_review_max_candidates_per_batch': config.max_validation_candidates_per_batch,
            'auto_review_max_candidates_per_workflow': config.max_validation_candidates_per_workflow,
            'auto_review_max_provider_attempts': config.max_provider_attempts,
            'auto_review_provider_timeout_seconds': config.provider_timeout_seconds,
            'auto_review_provider_send_start_window_seconds': config.provider_send_start_window_seconds,
            'auto_review_provider_attempt_lease_seconds': config.provider_attempt_lease_seconds,
            'auto_review_provider_commit_grace_seconds': config.provider_commit_grace_seconds,
            'auto_review_validator_input_usd_per_1m': config.validator_input_usd_per_1m,
            'auto_review_validator_output_usd_per_1m': config.validator_output_usd_per_1m,
            'auto_review_extraction_input_usd_per_1m': Decimal(
                str(extraction['input_usd_per_1m'])
            ),
            'auto_review_extraction_output_usd_per_1m': Decimal(
                str(extraction['output_usd_per_1m'])
            ),
            'auto_review_extraction_provider': extraction['provider'],
            'auto_review_extraction_model': extraction['model'],
            'auto_review_extraction_reasoning_effort': extraction['reasoning_effort'],
            'auto_review_extraction_route_version': extraction['route_version'],
            'auto_review_enforce_percentage': config.enforce_percentage,
            'authorized_percentage_at_launch': config.authorized_percentage_at_launch,
            'rollout_authorization_generation': config.rollout_authorization_generation,
            'validation_provider_safety_state_version': config.validation_provider_safety_state_version,
            'rollout_control_epoch': config.rollout_control_epoch,
            'extraction_plan_set_hmac': config.extraction_plan_set_hmac,
            'extraction_provider_safety_snapshot_set_hmac': config.extraction_provider_safety_snapshot_set_hmac,
            'selected_extraction_agent_count': len(config.extraction_plan_identities),
            'extraction_max_input_chars_per_agent': extraction['max_input_chars'],
            'extraction_max_input_tokens_per_agent': extraction['max_input_tokens'],
            'extraction_max_output_tokens_per_agent': extraction['max_output_tokens'],
            'extraction_max_candidates_per_agent': extraction['max_candidates'],
            'confirmed_extraction_cost_ceiling_usd': config.confirmed_extraction_cost_ceiling_usd,
            'confirmed_validation_cost_ceiling_usd': config.confirmed_validation_cost_ceiling_usd,
            'confirmed_total_cost_ceiling_usd': config.confirmed_total_cost_ceiling_usd,
            'auto_review_budget_limit_usd': config.total_budget_limit_usd,
        }

    def matches_stored_snapshot(self, values: dict[str, Any]) -> bool:
        return values == self.stored_snapshot()


def _uniform_extraction_snapshot(config: V21PreparedReviewConfig) -> dict[str, Any]:
    if not config.extraction_plan_identities:
        raise ValueError('V2.1 requires a non-empty extraction plan set')
    fields = (
        'provider',
        'model',
        'reasoning_effort',
        'route_version',
        'cost_policy_version',
        'token_estimator_version',
        'tokenizer_encoding',
        'reply_priming_tokens',
        'framing_safety_tokens',
        'max_input_chars',
        'max_input_tokens',
        'max_output_tokens',
        'max_candidates',
        'max_provider_attempts',
        'input_usd_per_1m',
        'output_usd_per_1m',
        'timing',
    )
    first = config.extraction_plan_identities[0]
    if any(field not in first for field in fields):
        raise ValueError('V2.1 extraction plan identity is incomplete')
    expected = {field: first[field] for field in fields}
    for identity in config.extraction_plan_identities[1:]:
        if any(identity.get(field) != value for field, value in expected.items()):
            raise ValueError(
                'V2.1 extraction routes must share immutable policy values'
            )
    if tuple(expected['timing']) != (
        config.provider_timeout_seconds,
        config.provider_send_start_window_seconds,
        config.provider_attempt_lease_seconds,
        config.provider_commit_grace_seconds,
    ):
        raise ValueError('V2.1 extraction timing differs from prepared request')
    return first


def _json_identity(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, 'f')
    if isinstance(value, dict):
        return {key: _json_identity(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_identity(item) for item in value]
    return value


def build_prepared_review_identity_v21(
    *,
    source_refs: tuple[ResolvedSourceVersion, ...],
    agent_names: tuple[str, ...],
    security_scope_id: str,
    config: V21PreparedReviewConfig,
    fingerprint_secret: bytes,
    evidence_version_hash: str | None = None,
) -> PreparedReviewRequestV21:
    if config.configured_auto_review_mode not in {'shadow', 'enforce'}:
        raise ValueError('V2.1 stores only shadow or enforce mode')
    extraction = _uniform_extraction_snapshot(config)
    if tuple(
        sorted(item['agent_name'] for item in config.extraction_plan_identities)
    ) != tuple(sorted(agent_names)):
        raise ValueError('V2.1 extraction plan set differs from selected agents')
    if extraction['max_provider_attempts'] != config.max_provider_attempts:
        raise ValueError('V2.1 extraction attempt cap differs from prepared request')
    if (
        config.provider_timeout_seconds
        + config.provider_send_start_window_seconds
        + config.provider_commit_grace_seconds
        >= config.provider_attempt_lease_seconds
    ):
        raise ValueError('provider lease must exceed send, timeout, and commit grace')
    if evidence_version_hash is None:
        evidence_version_hash = keyed_fingerprint(
            [asdict(ref) for ref in source_refs],
            secret=fingerprint_secret,
            schema_version='review-evidence-versions:v1',
            policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        )
    identity = {
        'security_scope_id': security_scope_id,
        'workflow_name': COMPANY_MEMORY_REVIEW_WORKFLOW,
        'graph_version': COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21,
        'evidence_version_hash': evidence_version_hash,
        'agent_names': sorted(agent_names),
        'selection_policy_version': COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        'config': _json_identity(asdict(config)),
    }
    input_hash = keyed_fingerprint(
        identity,
        secret=fingerprint_secret,
        schema_version='company-memory-review-v2.1-input:v1',
        policy_version='review-workflow-batch:v2.1',
    )
    return PreparedReviewRequestV21(
        source_refs=source_refs,
        agent_names=tuple(sorted(agent_names)),
        input_hash=input_hash,
        evidence_version_hash=evidence_version_hash,
        selection_policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        config=config,
    )


@dataclass(frozen=True)
class WorkflowPreflightResult:
    thread: AgentWorkflowThread
    created: bool
    shared_reuse: bool


class ReviewThreadCreationIdentity(Protocol):
    """Internal creation identity; source/agent validation precedes this boundary."""

    client_request_id: str | None


def build_prepared_review_identity(
    *,
    source_refs: tuple[ResolvedSourceVersion, ...],
    agent_names: tuple[str, ...],
    settings: Settings,
) -> PreparedReviewRequest:
    evidence_version_hash = build_keyed_fingerprint(
        [asdict(ref) for ref in source_refs],
        settings=settings,
        schema_version=EVIDENCE_VERSION_HASH_SCHEMA,
        policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    )
    input_hash = build_keyed_fingerprint(
        {
            'security_scope_id': settings.agent_runtime_security_scope_id,
            'workflow_name': COMPANY_MEMORY_REVIEW_WORKFLOW,
            'graph_version': COMPANY_MEMORY_REVIEW_GRAPH_VERSION,
            'evidence_version_hash': evidence_version_hash,
            'agent_names': list(agent_names),
            'selection_policy_version': COMPANY_MEMORY_SELECTION_POLICY_VERSION,
        },
        settings=settings,
        schema_version=COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
        policy_version=INPUT_HASH_POLICY,
    )
    return PreparedReviewRequest(
        source_refs=source_refs,
        agent_names=agent_names,
        input_hash=input_hash,
        evidence_version_hash=evidence_version_hash,
        selection_policy_version=COMPANY_MEMORY_SELECTION_POLICY_VERSION,
    )


def prepare_review_request(
    db: Session,
    *,
    request: ReviewWorkflowRunRequest,
    actor: DemoUser,
    registry: AgentRegistry,
    settings: Settings,
    v21_config: V21PreparedReviewConfig | None = None,
) -> PreparedReviewRequest | PreparedReviewRequestV21:
    try:
        agent_names = normalize_agent_names(request.agent_names)
        source_refs = normalize_source_version_refs(request.source_refs)
    except ValueError as exc:
        raise ReviewWorkflowPreflightError('invalid_input', str(exc)) from None
    if not settings.agent_runtime_security_scope_id.strip():
        raise ReviewWorkflowPreflightError(
            'invalid_input',
            'server security scope is not configured',
        )
    for agent_name in agent_names:
        try:
            manifest = registry.get(agent_name)
        except KeyError:
            raise ReviewWorkflowPreflightError(
                'invalid_input',
                'requested agent is not registered',
            ) from None
        if manifest.name != agent_name:
            raise ReviewWorkflowPreflightError(
                'invalid_input',
                'requested agent manifest does not match its registry key',
            )

    resolved_refs = resolve_source_versions(
        db,
        refs=source_refs,
        actor=actor,
        settings=settings,
    )
    if v21_config is not None:
        secret, _ = fingerprint_secret_bytes(settings)
        return build_prepared_review_identity_v21(
            source_refs=resolved_refs,
            agent_names=agent_names,
            security_scope_id=settings.agent_runtime_security_scope_id,
            config=v21_config,
            fingerprint_secret=secret,
        )
    return build_prepared_review_identity(
        source_refs=resolved_refs,
        agent_names=agent_names,
        settings=settings,
    )


def advisory_key_from_hmac(value: str) -> int:
    unsigned = int(value[:16], 16)
    return unsigned - (1 << 64) if unsigned >= (1 << 63) else unsigned


def create_or_reuse_review_thread(
    db: Session,
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    request: ReviewThreadCreationIdentity,
    actor: DemoUser,
    settings: Settings,
) -> WorkflowPreflightResult:
    creator_thread = _find_creator_thread(
        db,
        actor=actor,
        client_request_id=request.client_request_id,
        settings=settings,
    )
    if creator_thread is not None:
        try:
            result = _creator_replay_or_conflict(
                db,
                thread=creator_thread,
                prepared=prepared,
                actor=actor,
                settings=settings,
            )
            db.commit()
            return result
        except Exception:
            db.rollback()
            raise

    # End the read-only creator lookup transaction before waiting on SQLite's
    # process lock or starting PostgreSQL's advisory-lock transaction.
    db.rollback()
    postgres = _dialect_name(db) == 'postgresql'
    lock_context = nullcontext() if postgres else _SQLITE_PREFLIGHT_LOCK
    with lock_context:
        try:
            if postgres:
                for advisory_key in _postgres_advisory_keys(
                    prepared=prepared,
                    request=request,
                    actor=actor,
                    settings=settings,
                ):
                    db.execute(
                        text('SELECT pg_advisory_xact_lock(:key)'),
                        {'key': advisory_key},
                    )

            creator_thread = _find_creator_thread(
                db,
                actor=actor,
                client_request_id=request.client_request_id,
                settings=settings,
            )
            if creator_thread is not None:
                result = _creator_replay_or_conflict(
                    db,
                    thread=creator_thread,
                    prepared=prepared,
                    actor=actor,
                    settings=settings,
                )
                db.commit()
                return result

            _ensure_prepared_is_current(
                db,
                prepared=prepared,
                actor=actor,
                settings=settings,
            )
            shared_thread = _find_shared_thread(
                db,
                prepared=prepared,
                settings=settings,
            )
            if shared_thread is not None:
                db.commit()
                return WorkflowPreflightResult(
                    thread=shared_thread,
                    created=False,
                    shared_reuse=True,
                )

            if settings.langgraph_review_v2_enabled:
                _ensure_v2_waterline(db, prepared=prepared)

            thread = _create_thread_rows(
                db,
                prepared=prepared,
                request=request,
                actor=actor,
                settings=settings,
                checkpoint_store='postgres' if postgres else 'memory',
            )
            db.commit()
            return WorkflowPreflightResult(
                thread=thread,
                created=True,
                shared_reuse=False,
            )
        except Exception:
            db.rollback()
            raise


def _find_creator_thread(
    db: Session,
    *,
    actor: DemoUser,
    client_request_id: str | None,
    settings: Settings,
) -> AgentWorkflowThread | None:
    if client_request_id is None:
        return None
    return db.scalars(
        select(AgentWorkflowThread).where(
            AgentWorkflowThread.security_scope_id
            == settings.agent_runtime_security_scope_id,
            AgentWorkflowThread.workflow_name == COMPANY_MEMORY_REVIEW_WORKFLOW,
            AgentWorkflowThread.owner_subject_id == actor.id,
            AgentWorkflowThread.client_request_id == client_request_id,
        )
    ).first()


def _postgres_advisory_keys(
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    request: ReviewThreadCreationIdentity,
    actor: DemoUser,
    settings: Settings,
) -> tuple[int, ...]:
    keys = {advisory_key_from_hmac(prepared.input_hash)}
    if request.client_request_id is not None:
        creator_hash = build_keyed_fingerprint(
            {
                'security_scope_id': settings.agent_runtime_security_scope_id,
                'workflow_name': COMPANY_MEMORY_REVIEW_WORKFLOW,
                'owner_subject_id': actor.id,
                'client_request_id': request.client_request_id,
            },
            settings=settings,
            schema_version=CREATOR_LOCK_HASH_SCHEMA,
            policy_version=CREATOR_LOCK_HASH_POLICY,
        )
        keys.add(advisory_key_from_hmac(creator_hash))
    return tuple(sorted(keys))


def _creator_replay_or_conflict(
    db: Session,
    *,
    thread: AgentWorkflowThread,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    actor: DemoUser,
    settings: Settings,
) -> WorkflowPreflightResult:
    if not review_thread_matches_prepared(
        db,
        thread=thread,
        prepared=prepared,
        settings=settings,
    ):
        raise ReviewWorkflowPreflightError(
            'idempotency_key_reused',
            'client request key was already used for a different request',
        )
    _ensure_prepared_is_current(
        db,
        prepared=prepared,
        actor=actor,
        settings=settings,
    )
    return WorkflowPreflightResult(
        thread=thread,
        created=False,
        shared_reuse=False,
    )


def _find_shared_thread(
    db: Session,
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    settings: Settings,
) -> AgentWorkflowThread | None:
    candidates = db.scalars(
        select(AgentWorkflowThread)
        .where(
            AgentWorkflowThread.security_scope_id
            == settings.agent_runtime_security_scope_id,
            AgentWorkflowThread.workflow_name == COMPANY_MEMORY_REVIEW_WORKFLOW,
            AgentWorkflowThread.graph_version
            == (
                COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
                if isinstance(prepared, PreparedReviewRequestV21)
                else COMPANY_MEMORY_REVIEW_GRAPH_VERSION
            ),
            AgentWorkflowThread.input_hash == prepared.input_hash,
            AgentWorkflowThread.evidence_version_hash == prepared.evidence_version_hash,
        )
        .order_by(AgentWorkflowThread.created_at, AgentWorkflowThread.thread_id)
    ).all()
    return next(
        (
            thread
            for thread in candidates
            if review_thread_matches_prepared(
                db,
                thread=thread,
                prepared=prepared,
                settings=settings,
            )
        ),
        None,
    )


def find_matching_review_thread(
    db: Session,
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    actor: DemoUser,
    settings: Settings,
) -> AgentWorkflowThread | None:
    _ensure_prepared_is_current(
        db,
        prepared=prepared,
        actor=actor,
        settings=settings,
    )
    return _find_shared_thread(db, prepared=prepared, settings=settings)


def review_thread_matches_prepared(
    db: Session,
    *,
    thread: AgentWorkflowThread,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    settings: Settings,
) -> bool:
    if isinstance(prepared, PreparedReviewRequestV21):
        return _v21_thread_matches_prepared(
            db, thread=thread, prepared=prepared, settings=settings
        )
    if (
        thread.security_scope_id != settings.agent_runtime_security_scope_id
        or thread.workflow_name != COMPANY_MEMORY_REVIEW_WORKFLOW
        or thread.graph_version != COMPANY_MEMORY_REVIEW_GRAPH_VERSION
        or thread.input_hash != prepared.input_hash
        or thread.evidence_version_hash != prepared.evidence_version_hash
    ):
        return False
    stored_request = db.scalars(
        select(AgentWorkflowRequest).where(
            AgentWorkflowRequest.workflow_thread_id == thread.thread_id
        )
    ).first()
    _, fingerprint_key_version = fingerprint_secret_bytes(settings)
    if stored_request is None or (
        stored_request.input_schema_version != COMPANY_MEMORY_INPUT_SCHEMA_VERSION
        or stored_request.request_kind != 'review_source_versions'
        or tuple(stored_request.agent_names) != prepared.agent_names
        or stored_request.selection_policy_version != prepared.selection_policy_version
        or stored_request.input_hash != prepared.input_hash
        or stored_request.fingerprint_key_version != fingerprint_key_version
    ):
        return False
    stored_refs = db.scalars(
        select(AgentWorkflowEvidenceRef)
        .where(AgentWorkflowEvidenceRef.workflow_thread_id == thread.thread_id)
        .order_by(AgentWorkflowEvidenceRef.ordinal)
    ).all()
    return tuple(_stored_evidence_values(ref) for ref in stored_refs) == tuple(
        _resolved_evidence_values(ordinal, ref)
        for ordinal, ref in enumerate(prepared.source_refs)
    )


def _v21_thread_matches_prepared(
    db: Session,
    *,
    thread: AgentWorkflowThread,
    prepared: PreparedReviewRequestV21,
    settings: Settings,
) -> bool:
    if (
        thread.security_scope_id != settings.agent_runtime_security_scope_id
        or thread.workflow_name != COMPANY_MEMORY_REVIEW_WORKFLOW
        or thread.graph_version != COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
        or thread.input_hash != prepared.input_hash
        or thread.evidence_version_hash != prepared.evidence_version_hash
    ):
        return False
    stored_request = db.get(AgentWorkflowRequest, thread.thread_id)
    if stored_request is None:
        return False
    actual = {
        field: getattr(stored_request, field) for field in prepared.stored_snapshot()
    }
    if not prepared.matches_stored_snapshot(actual):
        return False
    if (
        stored_request.request_kind != 'review_source_versions'
        or tuple(stored_request.agent_names) != prepared.agent_names
        or stored_request.selection_policy_version != prepared.selection_policy_version
        or stored_request.input_hash != prepared.input_hash
    ):
        return False
    stored_refs = db.scalars(
        select(AgentWorkflowEvidenceRef)
        .where(AgentWorkflowEvidenceRef.workflow_thread_id == thread.thread_id)
        .order_by(AgentWorkflowEvidenceRef.ordinal)
    ).all()
    return tuple(_stored_evidence_values(ref) for ref in stored_refs) == tuple(
        _resolved_evidence_values(ordinal, ref)
        for ordinal, ref in enumerate(prepared.source_refs)
    )


def _ensure_prepared_is_current(
    db: Session,
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    actor: DemoUser,
    settings: Settings,
) -> None:
    source_ids = [ref.canonical_row_id for ref in prepared.source_refs]
    sources = db.scalars(select(Source).where(Source.id.in_(source_ids))).all()
    by_id = {source.id: source for source in sources}
    if len(by_id) != len(source_ids) or any(
        by_id[source_id].permission_level not in actor.permission_levels
        for source_id in source_ids
        if source_id in by_id
    ):
        raise ReviewWorkflowPreflightError(
            'not_found',
            'source reference was not found',
        )
    current_refs = tuple(
        SourceVersionRef(
            source_type=ref.source_type,  # type: ignore[arg-type]
            source_id=by_id[ref.canonical_row_id].source_id,
            version_or_signature=ref.content_signature,
        )
        for ref in prepared.source_refs
    )
    current = resolve_source_versions(
        db,
        refs=current_refs,
        actor=actor,
        settings=settings,
    )
    if current != prepared.source_refs:
        raise ReviewWorkflowPreflightError(
            'evidence_changed',
            'source evidence changed; synchronize again',
        )


def _ensure_v2_waterline(
    db: Session,
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
) -> None:
    source_ids = [ref.canonical_row_id for ref in prepared.source_refs]
    sources = db.scalars(select(Source).where(Source.id.in_(source_ids))).all()
    by_id = {source.id: source for source in sources}
    if len(by_id) != len(source_ids) or any(
        not review_batch_marker_is_v2(
            by_id[ref.canonical_row_id],
            signature=ref.content_signature,
        )
        for ref in prepared.source_refs
        if ref.canonical_row_id in by_id
    ):
        raise ReviewWorkflowPreflightError(
            'evidence_changed',
            'source evidence changed; synchronize again',
        )


def _create_thread_rows(
    db: Session,
    *,
    prepared: PreparedReviewRequest | PreparedReviewRequestV21,
    request: ReviewThreadCreationIdentity,
    actor: DemoUser,
    settings: Settings,
    checkpoint_store: str,
) -> AgentWorkflowThread:
    thread_id = uuid4().hex
    _, fingerprint_key_version = fingerprint_secret_bytes(settings)
    is_v21 = isinstance(prepared, PreparedReviewRequestV21)
    thread = AgentWorkflowThread(
        thread_id=thread_id,
        workflow_name=COMPANY_MEMORY_REVIEW_WORKFLOW,
        graph_version=(
            COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21
            if is_v21
            else COMPANY_MEMORY_REVIEW_GRAPH_VERSION
        ),
        checkpoint_thread_id=(
            f'review-v21:{uuid4().hex}' if is_v21 else f'review-v2:{uuid4().hex}'
        ),
        checkpoint_store=checkpoint_store,
        owner_subject_id=actor.id,
        security_scope_id=settings.agent_runtime_security_scope_id,
        client_request_id=request.client_request_id,
        input_hash=prepared.input_hash,
        evidence_version_hash=prepared.evidence_version_hash,
        status='created',
    )
    snapshot = prepared.stored_snapshot() if is_v21 else {}
    fingerprint_key_version = str(
        snapshot.pop('fingerprint_key_version', fingerprint_key_version)
    )
    stored_request = AgentWorkflowRequest(
        workflow_thread_id=thread_id,
        input_schema_version=COMPANY_MEMORY_INPUT_SCHEMA_VERSION,
        request_kind='review_source_versions',
        agent_names=list(prepared.agent_names),
        selection_policy_version=prepared.selection_policy_version,
        input_hash=prepared.input_hash,
        fingerprint_key_version=fingerprint_key_version,
        **snapshot,
    )
    evidence_rows = [
        AgentWorkflowEvidenceRef(
            workflow_thread_id=thread_id,
            ordinal=ordinal,
            canonical_source_type=ref.source_type,
            canonical_table=ref.canonical_table,
            canonical_row_id=ref.canonical_row_id,
            document_version_id=ref.document_version_id,
            external_revision=ref.external_revision,
            content_signature=ref.content_signature,
            permission_level_snapshot=ref.permission_level,
            content_fingerprint=ref.content_fingerprint,
        )
        for ordinal, ref in enumerate(prepared.source_refs)
    ]
    db.add_all([thread, stored_request, *evidence_rows])
    db.flush()
    return thread


def _stored_evidence_values(ref: AgentWorkflowEvidenceRef) -> tuple[object, ...]:
    return (
        ref.ordinal,
        ref.canonical_source_type,
        ref.canonical_table,
        ref.canonical_row_id,
        ref.document_version_id,
        ref.external_revision,
        ref.content_signature,
        ref.permission_level_snapshot,
        ref.content_fingerprint,
    )


def _resolved_evidence_values(
    ordinal: int,
    ref: ResolvedSourceVersion,
) -> tuple[object, ...]:
    return (
        ordinal,
        ref.source_type,
        ref.canonical_table,
        ref.canonical_row_id,
        ref.document_version_id,
        ref.external_revision,
        ref.content_signature,
        ref.permission_level,
        ref.content_fingerprint,
    )


def _dialect_name(db: Session) -> str:
    return db.get_bind().dialect.name
