from dataclasses import replace
from decimal import Decimal
from inspect import Parameter, signature

import pytest

from backend.app.agent_runtime.auto_review_policy import (
    SUPPORTED_VALIDATION_IDENTITY,
    AutoReviewPolicyEngine,
    AutoReviewValidationIdentity,
    EligibilityPolicyInput,
    TrustedTargetRef,
    ValidationPolicyInput,
)
from backend.app.schemas.auto_review import CandidateValidationResult


def _eligible_input(**overrides: object) -> EligibilityPolicyInput:
    value = EligibilityPolicyInput(
        item_type='timeline_event',
        permission_level='internal',
        evidence_state='exact',
        payload_complete=True,
        uncertainty_present=False,
        high_risk_language=False,
        project_selection_required=False,
        generation_identity_ready=True,
        validator_identity_collision=False,
        registry_ready=True,
        budget_available=True,
        fingerprint_projection_ready=True,
        claim_fingerprint='a' * 64,
        visible_targets=(),
        hidden_or_legacy_collision=False,
        lookup_available=True,
    )
    return replace(value, **overrides)


@pytest.mark.parametrize(
    ('overrides', 'decision', 'reason'),
    [
        ({'item_type': 'decision_record'}, 'human_review', 'item_type_not_allowed'),
        ({'item_type': 'todo'}, 'human_review', 'item_type_not_allowed'),
        ({'permission_level': 'restricted'}, 'human_review', 'permission_not_allowed'),
        ({'permission_level': 'unknown'}, 'human_review', 'permission_not_allowed'),
        ({'evidence_state': 'changed'}, 'needs_more_evidence', 'evidence_version_changed'),
        ({'evidence_state': 'missing'}, 'human_review', 'evidence_missing'),
        ({'evidence_state': 'mismatch'}, 'human_review', 'evidence_binding_mismatch'),
        ({'payload_complete': False}, 'human_review', 'payload_not_supported'),
        ({'uncertainty_present': True}, 'human_review', 'uncertainty_present'),
        ({'high_risk_language': True}, 'human_review', 'high_risk_language'),
        ({'project_selection_required': True}, 'human_review', 'project_selection_required'),
        ({'generation_identity_ready': False}, 'human_review', 'generation_identity_missing'),
        ({'validator_identity_collision': True}, 'human_review', 'validator_identity_collision'),
        ({'registry_ready': False}, 'human_review', 'registry_unavailable'),
        ({'budget_available': False}, 'human_review', 'budget_exceeded'),
        ({'fingerprint_projection_ready': False}, 'human_review', 'trusted_lookup_unavailable'),
        ({'lookup_available': False}, 'human_review', 'trusted_lookup_unavailable'),
    ],
)
def test_eligibility_policy_matrix_is_fail_closed(
    overrides: dict[str, object], decision: str, reason: str
) -> None:
    result = AutoReviewPolicyEngine().evaluate_eligibility(
        _eligible_input(**overrides)
    )

    assert result.decision == decision
    assert result.reason_codes == (reason,)
    assert result.validation_allowed is False


def test_exact_visible_duplicate_is_reused_only_without_any_hidden_collision() -> None:
    target = TrustedTargetRef(
        knowledge_type='timeline_event',
        knowledge_id=7,
        claim_fingerprint='a' * 64,
    )
    engine = AutoReviewPolicyEngine()

    exact = engine.evaluate_eligibility(
        _eligible_input(visible_targets=(target,))
    )
    hidden = engine.evaluate_eligibility(
        _eligible_input(
            visible_targets=(target,), hidden_or_legacy_collision=True
        )
    )

    assert exact.decision == 'reuse_trusted'
    assert exact.reason_codes == ('trusted_exact_match',)
    assert exact.duplicate_target == target
    assert exact.validation_allowed is True
    assert hidden.decision == 'human_review'
    assert hidden.reason_codes == ('trusted_hidden_collision',)
    assert hidden.duplicate_target is None


def test_visible_mismatch_or_multiplicity_is_human_review() -> None:
    mismatch = TrustedTargetRef('timeline_event', 1, 'b' * 64)
    exact = TrustedTargetRef('timeline_event', 2, 'a' * 64)
    engine = AutoReviewPolicyEngine()

    for targets in ((mismatch,), (exact, exact), (exact, mismatch)):
        result = engine.evaluate_eligibility(
            _eligible_input(visible_targets=targets)
        )
        assert result.decision == 'human_review'
        assert result.reason_codes == ('trusted_visible_conflict',)
        assert result.duplicate_target is None


def _validator_result(
    *,
    score: Decimal = Decimal('0.9800'),
    verdict: str = 'supported',
    scope: str = 'direct_fact',
    uncertainty: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> CandidateValidationResult:
    return CandidateValidationResult.model_validate(
        {
            'candidate_slot_id': 'C01',
            'claim_results': [
                {
                    'field_key': 'title',
                    'verdict': verdict,
                    'claim_scope': scope,
                    'entailment_score': score,
                    'evidence_slot_ids': ['E01'],
                },
                {
                    'field_key': 'result_summary',
                    'verdict': verdict,
                    'claim_scope': scope,
                    'entailment_score': score,
                    'evidence_slot_ids': ['E02'],
                },
            ],
            'uncertainty_codes': uncertainty or [],
            'conflict_codes': conflicts or [],
        }
    )


def _validation_input(
    *,
    eligibility=None,
    result: CandidateValidationResult | None = None,
) -> ValidationPolicyInput:
    return ValidationPolicyInput(
        eligibility=eligibility
        or AutoReviewPolicyEngine().evaluate_eligibility(_eligible_input()),
        expected_candidate_slot_id='C01',
        expected_claim_fields=('title', 'result_summary'),
        expected_evidence_slot_ids=('E01', 'E02'),
        validator_available=True,
        validator_malformed=False,
        result=result or _validator_result(),
        validation_identity=SUPPORTED_VALIDATION_IDENTITY,
        post_validation_state_matches=True,
    )


def test_post_provider_authority_inputs_are_explicit_and_have_no_defaults() -> None:
    parameters = signature(ValidationPolicyInput).parameters

    assert parameters['validation_identity'].default is Parameter.empty
    assert parameters['post_validation_state_matches'].default is Parameter.empty


@pytest.mark.parametrize(
    ('result', 'reason'),
    [
        (_validator_result(score=Decimal('0.9799')), 'validator_score_below_threshold'),
        (_validator_result(verdict='partially_supported'), 'validator_not_supported'),
        (_validator_result(scope='inference'), 'validator_not_direct_fact'),
        (_validator_result(uncertainty=['unknown']), 'validator_uncertain'),
        (_validator_result(conflicts=['unknown']), 'validator_conflict'),
    ],
)
def test_validator_cannot_override_deterministic_policy(
    result: CandidateValidationResult, reason: str
) -> None:
    decision = AutoReviewPolicyEngine().evaluate_validation(
        _validation_input(result=result)
    )

    assert decision.decision == 'human_review'
    assert decision.reason_codes == (reason,)


def test_exact_decimal_boundary_auto_approves_and_is_deterministic() -> None:
    engine = AutoReviewPolicyEngine()
    policy_input = _validation_input(
        result=_validator_result(score=Decimal('0.9800'))
    )

    first = engine.evaluate_validation(policy_input)
    second = engine.evaluate_validation(policy_input)

    assert first == second
    assert first.decision == 'auto_approve'
    assert first.reason_codes == ('eligible',)
    assert first.minimum_entailment_score == Decimal('0.9800')


def test_reuse_target_still_requires_the_same_successful_validation() -> None:
    target = TrustedTargetRef('timeline_event', 7, 'a' * 64)
    eligibility = AutoReviewPolicyEngine().evaluate_eligibility(
        _eligible_input(visible_targets=(target,))
    )

    approved = AutoReviewPolicyEngine().evaluate_validation(
        _validation_input(eligibility=eligibility)
    )
    rejected = AutoReviewPolicyEngine().evaluate_validation(
        _validation_input(
            eligibility=eligibility,
            result=_validator_result(score=Decimal('0.9799')),
        )
    )

    assert approved.decision == 'reuse_trusted'
    assert approved.duplicate_target == target
    assert rejected.decision == 'human_review'
    assert rejected.duplicate_target is None


def test_missing_malformed_or_wrong_slot_validator_output_is_human_only() -> None:
    engine = AutoReviewPolicyEngine()
    unavailable = replace(
        _validation_input(), validator_available=False, result=None
    )
    malformed = replace(_validation_input(), validator_malformed=True)
    wrong_slot_result = _validator_result().model_copy(
        update={'candidate_slot_id': 'C02'}
    )

    assert engine.evaluate_validation(unavailable).reason_codes == (
        'validator_unavailable',
    )
    assert engine.evaluate_validation(malformed).reason_codes == (
        'validator_malformed',
    )
    assert engine.evaluate_validation(
        _validation_input(result=wrong_slot_result)
    ).reason_codes == ('validator_malformed',)


def test_duplicate_or_unknown_evidence_slots_are_malformed() -> None:
    engine = AutoReviewPolicyEngine()
    duplicate = _validator_result()
    duplicate.claim_results[0].evidence_slot_ids = ['E01', 'E01']
    unknown = _validator_result()
    unknown.claim_results[0].evidence_slot_ids = ['E03']

    assert engine.evaluate_validation(
        _validation_input(result=duplicate)
    ).reason_codes == ('validator_malformed',)
    assert engine.evaluate_validation(
        _validation_input(result=unknown)
    ).reason_codes == ('validator_malformed',)


def test_malformed_member_rejects_the_whole_bounded_batch() -> None:
    first = _validation_input()
    malformed = replace(
        _validation_input(),
        expected_candidate_slot_id='C02',
        result=_validator_result(),
    )

    results = AutoReviewPolicyEngine().evaluate_batch((first, malformed))

    assert len(results) == 2
    assert all(result.decision == 'human_review' for result in results)
    assert all(result.reason_codes == ('validator_malformed',) for result in results)


def test_validator_version_identity_and_post_validation_state_are_exact() -> None:
    unsupported = AutoReviewValidationIdentity(
        provider='openai',
        model='gpt-5.6-terra',
        reasoning_effort='medium',
        prompt_version='auto-review-validation:v3',
        output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
    )
    engine = AutoReviewPolicyEngine()

    identity_result = engine.evaluate_validation(
        replace(_validation_input(), validation_identity=unsupported)
    )
    drift_result = engine.evaluate_validation(
        replace(_validation_input(), post_validation_state_matches=False)
    )

    assert identity_result.decision == 'human_review'
    assert identity_result.reason_codes == ('validator_unavailable',)
    assert drift_result.decision == 'needs_more_evidence'
    assert drift_result.reason_codes == ('post_validation_drift',)
