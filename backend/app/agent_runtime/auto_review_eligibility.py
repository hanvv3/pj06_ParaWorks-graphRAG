from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.auto_review_cost_policy import get_extraction_route
from backend.app.agent_runtime.auto_review_input_safety import (
    contains_high_risk_language,
    scan_auto_review_plaintext,
)
from backend.app.agent_runtime.auto_review_policy import (
    AutoReviewPolicyEngine,
    CandidateValidationRequest,
    EligibilityPolicyInput,
    TrustedTargetRef,
    ValidationClaimInput,
    ValidationEvidenceSlot,
)
from backend.app.agent_runtime.canonical_sources import (
    build_keyed_fingerprint,
    resolve_bound_workflow_evidence,
)
from backend.app.agent_runtime.fingerprints import fingerprint_secret_bytes
from backend.app.agent_runtime.registry import AgentRegistry
from backend.app.core.config import Settings
from backend.app.knowledge.promotion import build_promotion_preview
from backend.app.knowledge.trusted_fingerprint_projection import (
    find_visible_trusted_collisions,
    hidden_or_legacy_collision_exists,
    trusted_fingerprint_projection_ready,
)
from backend.app.knowledge.trusted_provenance import (
    TrustedProvenanceMismatch,
    build_trusted_candidate_fingerprint,
)
from backend.app.models import (
    AgentRun,
    AgentWorkflowEvidenceRef,
    AgentWorkflowThread,
    ReviewItem,
    ReviewItemEvidenceRef,
)
from backend.app.schemas.auto_review import (
    AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION,
    AUTO_REVIEW_VALIDATOR_MODEL,
    AUTO_REVIEW_VALIDATOR_PROVIDER,
    AutoReviewEligibilityDecision,
    AutoReviewPolicyReasonCode,
)

_CLAIM_FIELDS = {
    'timeline_event': ('title', 'result_summary'),
    'history_event': ('title', 'reason'),
}
_CANDIDATE_SYSTEM_PAYLOAD_KEYS = frozenset(
    {
        'title',
        'summary',
        'agent_name',
        'agent_run_id',
        'prompt_version',
        'cache_key',
        'estimated_cost_usd',
        'token_usage',
        'uncertainty_reason',
        'source_ids',
        'source_types',
        'source_authors',
    }
)


@dataclass(frozen=True, slots=True)
class AutoReviewEligibilityContext:
    evidence_state: str
    evidence_texts: tuple[str, ...]
    claim_fingerprint: str | None
    visible_targets: tuple[TrustedTargetRef, ...] = ()
    hidden_or_legacy_collision: bool = False
    lookup_available: bool = True
    fingerprint_projection_ready: bool = True
    generation_identity_ready: bool = True
    validator_identity_collision: bool = False
    registry_ready: bool = True
    budget_available: bool = True


@dataclass(frozen=True, slots=True)
class AutoReviewEvidenceMessage:
    workflow_evidence_ref_id: int
    stable_message_identity: str
    text: str
    permission_level: str


@dataclass(frozen=True, slots=True)
class AutoReviewEligibilityResult:
    decision: AutoReviewEligibilityDecision
    reason_codes: tuple[AutoReviewPolicyReasonCode, ...]
    claim_fingerprint: str | None = None
    duplicate_target: TrustedTargetRef | None = None
    validation_request: CandidateValidationRequest | None = None


class AutoReviewEligibilityService:
    def __init__(self, *, policy_engine: AutoReviewPolicyEngine | None = None) -> None:
        self._policy = policy_engine or AutoReviewPolicyEngine()

    def evaluate(
        self,
        item: ReviewItem,
        *,
        context: AutoReviewEligibilityContext,
    ) -> AutoReviewEligibilityResult:
        preview = build_promotion_preview(item)
        normalized = preview.get('normalized_payload')
        payload_complete = bool(
            preview.get('can_approve') and isinstance(normalized, dict)
        )
        project_selection_required = 'project_key' in tuple(
            preview.get('missing_required_fields') or ()
        )
        plaintext = _candidate_plaintext(item, context.evidence_texts)
        sensitive = any(
            not scan_auto_review_plaintext(value).allowed for value in plaintext
        )
        if sensitive:
            return AutoReviewEligibilityResult(
                decision='human_review',
                reason_codes=('sensitive_input_detected',),
                claim_fingerprint=context.claim_fingerprint,
            )
        uncertainty = item.payload.get('uncertainty_reason')
        high_risk = any(contains_high_risk_language(value) for value in plaintext)
        policy_result = self._policy.evaluate_eligibility(
            EligibilityPolicyInput(
                item_type=item.item_type,
                permission_level=item.permission_level,
                evidence_state=context.evidence_state,  # type: ignore[arg-type]
                payload_complete=payload_complete,
                uncertainty_present=isinstance(uncertainty, str)
                and bool(uncertainty.strip()),
                high_risk_language=high_risk,
                project_selection_required=project_selection_required,
                generation_identity_ready=context.generation_identity_ready,
                validator_identity_collision=context.validator_identity_collision,
                registry_ready=context.registry_ready,
                budget_available=context.budget_available,
                fingerprint_projection_ready=(
                    context.fingerprint_projection_ready
                ),
                claim_fingerprint=context.claim_fingerprint,
                visible_targets=context.visible_targets,
                hidden_or_legacy_collision=(
                    context.hidden_or_legacy_collision
                ),
                lookup_available=context.lookup_available,
            )
        )
        request = None
        if policy_result.validation_allowed:
            request = build_candidate_validation_request(
                item,
                evidence_texts=context.evidence_texts,
            )
            if request is None:
                return AutoReviewEligibilityResult(
                    decision='human_review',
                    reason_codes=('evidence_binding_mismatch',),
                    claim_fingerprint=context.claim_fingerprint,
                )
        return AutoReviewEligibilityResult(
            decision=policy_result.decision,
            reason_codes=policy_result.reason_codes,
            claim_fingerprint=context.claim_fingerprint,
            duplicate_target=policy_result.duplicate_target,
            validation_request=request,
        )

    def evaluate_from_database(
        self,
        db: Session,
        item: ReviewItem,
        *,
        settings: Settings,
        registry: AgentRegistry,
        visible_permission_levels: tuple[str, ...],
        evidence_messages: tuple[AutoReviewEvidenceMessage, ...],
        budget_available: bool,
    ) -> AutoReviewEligibilityResult:
        context = load_database_eligibility_context(
            db,
            item=item,
            settings=settings,
            registry=registry,
            visible_permission_levels=visible_permission_levels,
            evidence_messages=evidence_messages,
            budget_available=budget_available,
        )
        return self.evaluate(item, context=context)


def load_database_eligibility_context(
    db: Session,
    *,
    item: ReviewItem,
    settings: Settings,
    registry: AgentRegistry,
    visible_permission_levels: tuple[str, ...],
    evidence_messages: tuple[AutoReviewEvidenceMessage, ...],
    budget_available: bool,
) -> AutoReviewEligibilityContext:
    evidence_state, evidence_texts = _bound_evidence_state(
        db,
        item=item,
        settings=settings,
        visible_permission_levels=visible_permission_levels,
        evidence_messages=evidence_messages,
    )
    generation_ready, identity_collision, registry_ready = _generation_readiness(
        db,
        item=item,
        registry=registry,
    )
    claim_fingerprint: str | None = None
    visible_targets: tuple[TrustedTargetRef, ...] = ()
    hidden_collision = False
    lookup_available = True
    projection_ready = trusted_fingerprint_projection_ready(
        db,
        settings=settings,
    )
    thread = (
        db.get(AgentWorkflowThread, item.workflow_thread_id)
        if item.workflow_thread_id is not None
        else None
    )
    security_scope_id = (
        thread.security_scope_id.strip()
        if thread is not None
        and isinstance(thread.security_scope_id, str)
        and thread.security_scope_id.strip()
        else None
    )
    try:
        if security_scope_id is None:
            raise TrustedProvenanceMismatch('workflow scope is unavailable')
        identity = build_trusted_candidate_fingerprint(
            item=item,
            security_scope_id=security_scope_id,
            settings=settings,
        )
        claim_fingerprint = identity.normalized_claim_fingerprint
        if projection_ready:
            visible = find_visible_trusted_collisions(
                db,
                knowledge_type=identity.knowledge_type,
                security_scope_id=identity.security_scope_id,
                project_scope_hmac=identity.project_scope_hmac,
                normalized_title_bucket_hmac=(
                    identity.normalized_title_bucket_hmac
                ),
                visible_permission_levels=visible_permission_levels,
                candidate_permission_level=item.permission_level,
            )
            visible_targets = tuple(
                TrustedTargetRef(
                    knowledge_type=row.knowledge_type,  # type: ignore[arg-type]
                    knowledge_id=row.knowledge_id,
                    claim_fingerprint=row.normalized_claim_fingerprint,
                )
                for row in visible
            )
            hidden_collision = hidden_or_legacy_collision_exists(
                db,
                knowledge_type=identity.knowledge_type,
                security_scope_id=identity.security_scope_id,
                project_scope_hmac=identity.project_scope_hmac,
                normalized_title_bucket_hmac=(
                    identity.normalized_title_bucket_hmac
                ),
                visible_permission_levels=visible_permission_levels,
                candidate_permission_level=item.permission_level,
            )
    except (SQLAlchemyError, TrustedProvenanceMismatch, ValueError):
        lookup_available = False
    return AutoReviewEligibilityContext(
        evidence_state=evidence_state,
        evidence_texts=evidence_texts,
        claim_fingerprint=claim_fingerprint,
        visible_targets=visible_targets,
        hidden_or_legacy_collision=hidden_collision,
        lookup_available=lookup_available,
        fingerprint_projection_ready=projection_ready,
        generation_identity_ready=generation_ready,
        validator_identity_collision=identity_collision,
        registry_ready=registry_ready,
        budget_available=budget_available,
    )


def build_candidate_validation_request(
    item: ReviewItem,
    *,
    evidence_texts: tuple[str, ...],
) -> CandidateValidationRequest | None:
    """Build the ephemeral provider DTO only after deterministic safety checks."""
    fields = _CLAIM_FIELDS.get(item.item_type)
    if fields is None or not 1 <= len(evidence_texts) <= 12:
        return None
    normalized_evidence = tuple(text.strip() for text in evidence_texts)
    if (
        any(not text for text in normalized_evidence)
        or len(set(normalized_evidence)) != len(normalized_evidence)
    ):
        return None
    preview = build_promotion_preview(item)
    normalized = preview.get('normalized_payload')
    if not preview.get('can_approve') or not isinstance(normalized, dict):
        return None
    claim_values = tuple(normalized.get(field) for field in fields)
    if any(not isinstance(value, str) or not value.strip() for value in claim_values):
        return None
    plaintext = (*claim_values, *normalized_evidence)
    if any(not scan_auto_review_plaintext(text).allowed for text in plaintext):
        return None
    claims = tuple(
        ValidationClaimInput(field_key=field, text=value.strip())
        for field, value in zip(fields, claim_values, strict=True)
    )
    evidence_slots = tuple(
        ValidationEvidenceSlot(slot_id=f'E{ordinal:02d}', text=text)
        for ordinal, text in enumerate(normalized_evidence, start=1)
    )
    return CandidateValidationRequest(
        candidate_slot_id='C01',
        item_type=item.item_type,
        claims=claims,  # type: ignore[arg-type]
        evidence_slots=evidence_slots,
    )


def _candidate_plaintext(
    item: ReviewItem, evidence_texts: tuple[str, ...]
) -> tuple[str, ...]:
    preview = build_promotion_preview(item)
    normalized = preview.get('normalized_payload')
    fields = _CLAIM_FIELDS.get(item.item_type, ())
    claims = (
        tuple(
            value
            for field in fields
            if isinstance((value := normalized.get(field)), str)
        )
        if isinstance(normalized, dict)
        else ()
    )
    return (*claims, *evidence_texts)


def _bound_evidence_state(
    db: Session,
    *,
    item: ReviewItem,
    settings: Settings,
    visible_permission_levels: tuple[str, ...],
    evidence_messages: tuple[AutoReviewEvidenceMessage, ...],
) -> tuple[str, tuple[str, ...]]:
    if (
        item.id is None
        or item.workflow_thread_id is None
        or item.candidate_contract_version != 'c5-v1'
    ):
        return 'missing', ()
    if not _candidate_key_is_exact(item=item, settings=settings):
        return 'mismatch', ()
    bindings = tuple(
        db.scalars(
            select(ReviewItemEvidenceRef)
            .where(
                ReviewItemEvidenceRef.review_item_id == item.id,
                ReviewItemEvidenceRef.workflow_thread_id
                == item.workflow_thread_id,
            )
            .order_by(ReviewItemEvidenceRef.candidate_slot_ordinal)
        ).all()
    )
    if not bindings:
        return 'missing', ()
    if tuple(binding.candidate_slot_ordinal for binding in bindings) != tuple(
        range(1, len(bindings) + 1)
    ):
        return 'mismatch', ()
    ref_ids = tuple(binding.workflow_evidence_ref_id for binding in bindings)
    if len(set(ref_ids)) != len(ref_ids):
        return 'mismatch', ()
    refs = {
        ref.id: ref
        for ref in db.scalars(
            select(AgentWorkflowEvidenceRef).where(
                AgentWorkflowEvidenceRef.workflow_thread_id
                == item.workflow_thread_id,
                AgentWorkflowEvidenceRef.id.in_(ref_ids),
            )
        ).all()
    }
    if len(refs) != len(bindings):
        return 'mismatch', ()
    secret, key_version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    if any(
        binding.fingerprint_key_version != key_version
        or binding.fingerprint_key_material_verifier != verifier
        or len(binding.message_content_fingerprint) != 64
        for binding in bindings
    ):
        return 'mismatch', ()
    if not 1 <= len(evidence_messages) <= 12:
        return 'missing', ()
    grouped: dict[int, list[AutoReviewEvidenceMessage]] = {}
    for message in evidence_messages:
        grouped.setdefault(message.workflow_evidence_ref_id, []).append(message)
    if set(grouped) != set(ref_ids):
        return 'mismatch', ()
    ordered_messages: list[AutoReviewEvidenceMessage] = []
    for binding in bindings:
        ref = refs[binding.workflow_evidence_ref_id]
        messages = tuple(
            sorted(
                grouped[binding.workflow_evidence_ref_id],
                key=lambda message: message.stable_message_identity,
            )
        )
        identities = tuple(message.stable_message_identity for message in messages)
        if (
            not messages
            or any(
                not identity or not message.text.strip()
                for identity, message in zip(identities, messages, strict=True)
            )
            or len(set(identities)) != len(identities)
        ):
            return 'mismatch', ()
        permission = _strictest_permission(
            tuple(message.permission_level for message in messages)
        )
        if permission != ref.permission_level_snapshot:
            return 'mismatch', ()
        message_set_hmac = build_keyed_fingerprint(
            {
                'canonical_source_kind': ref.canonical_source_type,
                'canonical_source_id': ref.canonical_row_id,
                'canonical_version_or_signature': (
                    ref.external_revision or ref.content_signature
                ),
                'content_fingerprint': ref.content_fingerprint,
                'permission_level': permission,
                'fingerprint_key_version': key_version,
                'fingerprint_key_material_verifier': verifier,
                'messages': [
                    {
                        'stable_message_identity': message.stable_message_identity,
                        'text_fingerprint': build_keyed_fingerprint(
                            message.text,
                            settings=settings,
                            schema_version='candidate-message-content:v1',
                            policy_version='candidate-message-content:v1',
                        ),
                    }
                    for message in messages
                ],
            },
            settings=settings,
            schema_version='candidate-message-set:v1',
            policy_version='candidate-message-set:v1',
        )
        if message_set_hmac != binding.message_content_fingerprint:
            return 'mismatch', ()
        ordered_messages.extend(messages)
    resolutions = tuple(
        resolve_bound_workflow_evidence(
            db,
            ref=refs[binding.workflow_evidence_ref_id],
            visible_permission_levels=visible_permission_levels,
            settings=settings,
        )
        for binding in bindings
    )
    states = {resolution.state for resolution in resolutions}
    if 'mismatch' in states:
        return 'mismatch', ()
    if 'changed' in states:
        return 'changed', ()
    if 'missing' in states:
        return 'missing', ()
    current_permissions = tuple(
        resolution.resolved.permission_level
        for resolution in resolutions
        if resolution.resolved is not None
    )
    if not current_permissions or item.permission_level != _strictest_permission(
        current_permissions
    ):
        return 'changed', ()
    if any(
        message.permission_level not in visible_permission_levels
        for message in ordered_messages
    ):
        return 'missing', ()
    return 'exact', tuple(message.text for message in ordered_messages)


def _generation_readiness(
    db: Session,
    *,
    item: ReviewItem,
    registry: AgentRegistry,
) -> tuple[bool, bool, bool]:
    if item.agent_run_id is None or item.workflow_thread_id is None:
        return False, False, False
    run = db.get(AgentRun, item.agent_run_id)
    thread = db.get(AgentWorkflowThread, item.workflow_thread_id)
    if (
        run is None
        or thread is None
        or run.workflow_thread_id != item.workflow_thread_id
        or run.status != 'complete'
        or thread.graph_version != 'company-memory-review-v2.1-auto-review'
    ):
        return False, False, False
    if (
        item.payload.get('agent_name') != run.agent_name
        or item.payload.get('agent_run_id') != run.id
        or item.payload.get('prompt_version') != run.prompt_version
        or item.payload.get('cache_key') != run.effect_key
    ):
        return False, False, False
    collision = (run.generation_provider, run.model_name) == (
        AUTO_REVIEW_VALIDATOR_PROVIDER,
        AUTO_REVIEW_VALIDATOR_MODEL,
    )
    route = get_extraction_route(
        run.agent_name,
        extraction_cost_policy_version=AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION,
        provider=run.generation_provider or '',
        model=run.model_name,
        reasoning_effort=run.generation_reasoning_effort or '',
        route_version=run.generation_route_version or '',
        prompt_version=run.prompt_version,
        output_contract_version=run.generation_output_contract_version or '',
    )
    identity_ready = route is not None and not collision
    try:
        manifest = registry.get(run.agent_name)
    except KeyError:
        return identity_ready, collision, False
    registry_ready = bool(
        manifest.output_contract == 'AgentRunResult'
        and item.permission_level in manifest.supported_permissions
    )
    return identity_ready, collision, registry_ready


def _strictest_permission(levels: tuple[str, ...]) -> str:
    rank = {'public': 0, 'internal': 1, 'restricted': 2}
    return max(levels, key=lambda level: rank.get(level, 3))


def _candidate_key_is_exact(*, item: ReviewItem, settings: Settings) -> bool:
    if not isinstance(item.candidate_key, str) or len(item.candidate_key) != 64:
        return False
    effect_key = item.payload.get('cache_key')
    title = item.payload.get('title')
    summary = item.payload.get('summary')
    if not all(isinstance(value, str) and value for value in (effect_key, title, summary)):
        return False
    payload_fields = {
        key: value
        for key, value in item.payload.items()
        if key not in _CANDIDATE_SYSTEM_PAYLOAD_KEYS
    }
    expected = build_keyed_fingerprint(
        {
            'effect_key': effect_key,
            'candidate': {
                'item_type': item.item_type,
                'title': title,
                'summary': summary,
                'source_links': list(item.source_links),
                'source_snippets': list(item.source_snippets),
                'confidence_score': item.confidence_score,
                'permission_level': item.permission_level,
                'uncertainty_reason': item.payload.get('uncertainty_reason'),
                'payload_fields': payload_fields,
            },
        },
        settings=settings,
        schema_version='review-agent-candidate:v1',
        policy_version='review-agent-candidate-key:v1',
    )
    return expected == item.candidate_key
