from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from backend.app.schemas.auto_review import (
    AUTO_REVIEW_MIN_ENTAILMENT,
    AUTO_REVIEW_POLICY_VERSION,
    AUTO_REVIEW_REASONING_EFFORT,
    AUTO_REVIEW_VALIDATOR_MODEL,
    AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
    AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
    AUTO_REVIEW_VALIDATOR_PROVIDER,
    AutoReviewEligibilityDecision,
    AutoReviewPolicyDecision,
    AutoReviewPolicyReasonCode,
    CandidateValidationResult,
)

EvidenceState = Literal['exact', 'missing', 'changed', 'mismatch']
ReusableKnowledgeType = Literal['timeline_event', 'history_event']
ValidationClaimField = Literal['title', 'result_summary', 'reason']


@dataclass(frozen=True, slots=True)
class TrustedTargetRef:
    knowledge_type: ReusableKnowledgeType
    knowledge_id: int
    claim_fingerprint: str

    def __post_init__(self) -> None:
        if self.knowledge_id <= 0 or len(self.claim_fingerprint) != 64:
            raise ValueError('trusted target identity is invalid')


@dataclass(frozen=True, slots=True)
class ValidationClaimInput:
    field_key: ValidationClaimField
    text: str


@dataclass(frozen=True, slots=True)
class ValidationEvidenceSlot:
    slot_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CandidateValidationRequest:
    candidate_slot_id: str
    item_type: ReusableKnowledgeType
    claims: tuple[ValidationClaimInput, ValidationClaimInput]
    evidence_slots: tuple[ValidationEvidenceSlot, ...]

    def __post_init__(self) -> None:
        if self.candidate_slot_id not in {'C01', 'C02', 'C03', 'C04'}:
            raise ValueError('candidate slot is invalid')
        expected_claims = {
            'timeline_event': ('title', 'result_summary'),
            'history_event': ('title', 'reason'),
        }[self.item_type]
        if tuple(claim.field_key for claim in self.claims) != expected_claims:
            raise ValueError('candidate claim fields are invalid')
        if any(not claim.text.strip() for claim in self.claims):
            raise ValueError('candidate claim text is empty')
        if not 1 <= len(self.evidence_slots) <= 12:
            raise ValueError('candidate evidence slot count is invalid')
        expected_slots = tuple(
            f'E{ordinal:02d}' for ordinal in range(1, len(self.evidence_slots) + 1)
        )
        if tuple(slot.slot_id for slot in self.evidence_slots) != expected_slots:
            raise ValueError('candidate evidence slots are invalid')
        texts = tuple(slot.text.strip() for slot in self.evidence_slots)
        if any(not text for text in texts) or len(set(texts)) != len(texts):
            raise ValueError('candidate evidence text is empty or ambiguous')


@dataclass(frozen=True, slots=True)
class EligibilityPolicyInput:
    item_type: str
    permission_level: str
    evidence_state: EvidenceState
    payload_complete: bool
    uncertainty_present: bool
    high_risk_language: bool
    project_selection_required: bool
    generation_identity_ready: bool
    validator_identity_collision: bool
    registry_ready: bool
    budget_available: bool
    fingerprint_projection_ready: bool
    claim_fingerprint: str | None
    visible_targets: tuple[TrustedTargetRef, ...]
    hidden_or_legacy_collision: bool
    lookup_available: bool


@dataclass(frozen=True, slots=True)
class EligibilityPolicyResult:
    decision: AutoReviewEligibilityDecision
    reason_codes: tuple[AutoReviewPolicyReasonCode, ...]
    duplicate_target: TrustedTargetRef | None = None

    @property
    def validation_allowed(self) -> bool:
        return self.decision in {'eligible', 'reuse_trusted'}


@dataclass(frozen=True, slots=True)
class AutoReviewValidationIdentity:
    provider: str
    model: str
    reasoning_effort: str
    prompt_version: str
    output_contract_version: str
    policy_version: str


SUPPORTED_VALIDATION_IDENTITY = AutoReviewValidationIdentity(
    provider=AUTO_REVIEW_VALIDATOR_PROVIDER,
    model=AUTO_REVIEW_VALIDATOR_MODEL,
    reasoning_effort=AUTO_REVIEW_REASONING_EFFORT,
    prompt_version=AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
    output_contract_version=AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
    policy_version=AUTO_REVIEW_POLICY_VERSION,
)


@dataclass(frozen=True, slots=True)
class ValidationPolicyInput:
    eligibility: EligibilityPolicyResult
    expected_candidate_slot_id: str
    expected_claim_fields: tuple[ValidationClaimField, ValidationClaimField]
    expected_evidence_slot_ids: tuple[str, ...]
    validator_available: bool
    validator_malformed: bool
    result: CandidateValidationResult | None
    validation_identity: AutoReviewValidationIdentity
    post_validation_state_matches: bool


@dataclass(frozen=True, slots=True)
class AutoReviewPolicyResult:
    decision: AutoReviewPolicyDecision
    reason_codes: tuple[AutoReviewPolicyReasonCode, ...]
    duplicate_target: TrustedTargetRef | None = None
    minimum_entailment_score: Decimal | None = None


class AutoReviewPolicyEngine:
    """Versioned deterministic authority; owns no DB, model, network, or clock."""

    def evaluate_eligibility(
        self, value: EligibilityPolicyInput
    ) -> EligibilityPolicyResult:
        if value.item_type not in {'timeline_event', 'history_event'}:
            return _eligibility('human_review', 'item_type_not_allowed')
        if value.permission_level not in {'public', 'internal'}:
            return _eligibility('human_review', 'permission_not_allowed')
        if value.evidence_state == 'changed':
            return _eligibility('needs_more_evidence', 'evidence_version_changed')
        if value.evidence_state == 'missing':
            return _eligibility('human_review', 'evidence_missing')
        if value.evidence_state != 'exact':
            return _eligibility('human_review', 'evidence_binding_mismatch')
        if not value.payload_complete:
            return _eligibility('human_review', 'payload_not_supported')
        if value.project_selection_required:
            return _eligibility('human_review', 'project_selection_required')
        if value.uncertainty_present:
            return _eligibility('human_review', 'uncertainty_present')
        if value.high_risk_language:
            return _eligibility('human_review', 'high_risk_language')
        if value.validator_identity_collision:
            return _eligibility('human_review', 'validator_identity_collision')
        if not value.generation_identity_ready:
            return _eligibility('human_review', 'generation_identity_missing')
        if not value.registry_ready:
            return _eligibility('human_review', 'registry_unavailable')
        if not value.budget_available:
            return _eligibility('human_review', 'budget_exceeded')
        if not value.fingerprint_projection_ready or not value.lookup_available:
            return _eligibility('human_review', 'trusted_lookup_unavailable')
        if value.claim_fingerprint is None or len(value.claim_fingerprint) != 64:
            return _eligibility('human_review', 'payload_not_supported')
        if value.hidden_or_legacy_collision:
            return _eligibility('human_review', 'trusted_hidden_collision')
        exact = tuple(
            target
            for target in value.visible_targets
            if target.knowledge_type == value.item_type
            and target.claim_fingerprint == value.claim_fingerprint
        )
        if value.visible_targets:
            if len(value.visible_targets) != 1 or len(exact) != 1:
                return _eligibility('human_review', 'trusted_visible_conflict')
            return EligibilityPolicyResult(
                decision='reuse_trusted',
                reason_codes=('trusted_exact_match',),
                duplicate_target=exact[0],
            )
        return _eligibility('eligible', 'eligible')

    def evaluate_validation(
        self, value: ValidationPolicyInput
    ) -> AutoReviewPolicyResult:
        eligibility = value.eligibility
        if not eligibility.validation_allowed:
            return AutoReviewPolicyResult(
                decision=eligibility.decision,
                reason_codes=eligibility.reason_codes,
            )
        if value.validation_identity != SUPPORTED_VALIDATION_IDENTITY:
            return _policy('human_review', 'validator_unavailable')
        if not value.post_validation_state_matches:
            return _policy('needs_more_evidence', 'post_validation_drift')
        if not value.validator_available:
            return _policy('human_review', 'validator_unavailable')
        if value.validator_malformed or value.result is None:
            return _policy('human_review', 'validator_malformed')
        result = value.result
        if (
            result.candidate_slot_id != value.expected_candidate_slot_id
            or len(result.claim_results) != 2
            or tuple(claim.field_key for claim in result.claim_results)
            != value.expected_claim_fields
        ):
            return _policy('human_review', 'validator_malformed')
        allowed_slots = frozenset(value.expected_evidence_slot_ids)
        if any(
            not claim.evidence_slot_ids
            or len(claim.evidence_slot_ids) != len(set(claim.evidence_slot_ids))
            or not set(claim.evidence_slot_ids).issubset(allowed_slots)
            for claim in result.claim_results
        ):
            return _policy('human_review', 'validator_malformed')
        if result.uncertainty_codes:
            return _policy('human_review', 'validator_uncertain')
        if result.conflict_codes:
            return _policy('human_review', 'validator_conflict')
        if any(claim.verdict != 'supported' for claim in result.claim_results):
            return _policy('human_review', 'validator_not_supported')
        if any(claim.claim_scope != 'direct_fact' for claim in result.claim_results):
            return _policy('human_review', 'validator_not_direct_fact')
        minimum = min(
            claim.entailment_score for claim in result.claim_results
        )
        if minimum < AUTO_REVIEW_MIN_ENTAILMENT:
            return _policy('human_review', 'validator_score_below_threshold')
        if eligibility.decision == 'reuse_trusted':
            return AutoReviewPolicyResult(
                decision='reuse_trusted',
                reason_codes=('trusted_exact_match',),
                duplicate_target=eligibility.duplicate_target,
                minimum_entailment_score=minimum,
            )
        return AutoReviewPolicyResult(
            decision='auto_approve',
            reason_codes=('eligible',),
            minimum_entailment_score=minimum,
        )

    def evaluate_batch(
        self,
        values: tuple[ValidationPolicyInput, ...],
    ) -> tuple[AutoReviewPolicyResult, ...]:
        if not values:
            return ()
        expected_slots = tuple(
            f'C{ordinal:02d}' for ordinal in range(1, len(values) + 1)
        )
        malformed_batch = len(values) > 4 or tuple(
            value.expected_candidate_slot_id for value in values
        ) != expected_slots
        results = tuple(self.evaluate_validation(value) for value in values)
        if malformed_batch or any(
            result.reason_codes == ('validator_malformed',) for result in results
        ):
            malformed = _policy('human_review', 'validator_malformed')
            return tuple(malformed for _ in values)
        return results


def _eligibility(
    decision: AutoReviewEligibilityDecision,
    reason: AutoReviewPolicyReasonCode,
) -> EligibilityPolicyResult:
    return EligibilityPolicyResult(decision=decision, reason_codes=(reason,))


def _policy(
    decision: AutoReviewPolicyDecision,
    reason: AutoReviewPolicyReasonCode,
) -> AutoReviewPolicyResult:
    return AutoReviewPolicyResult(decision=decision, reason_codes=(reason,))
