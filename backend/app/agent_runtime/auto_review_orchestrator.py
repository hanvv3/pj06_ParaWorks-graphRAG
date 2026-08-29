from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from backend.app.agent_runtime.auto_review_policy import (
    AutoReviewPolicyEngine,
    CandidateValidationRequest,
    ValidationPolicyInput,
)
from backend.app.agent_runtime.auto_review_validation_store import (
    CompletedValidationProjection,
    ValidationAttemptAdmission,
    ValidationBatchClaimRequest,
    ValidationBatchCompletion,
    ValidationBatchFailure,
    ValidationCandidateClaim,
    ValidationCandidateCompletion,
    ValidationClaimResult,
    ValidationLockedContext,
    _token_cost,
    derive_validation_batch_fingerprint,
)
from backend.app.agent_runtime.auto_review_validator import (
    AutoReviewValidationError,
    AutoReviewValidatorFactory,
    PreparedValidationInvocation,
    ValidationFrameSizer,
    ValidationUsage,
)
from backend.app.core.config import Settings
from backend.app.schemas.auto_review import AutoReviewPolicyReasonCode


class ValidationStore(Protocol):
    def claim_or_replay(
        self, request: ValidationBatchClaimRequest
    ) -> ValidationClaimResult: ...

    def mark_attempt_started(self, admission: ValidationAttemptAdmission): ...

    def complete(
        self,
        context: ValidationLockedContext,
        completion: ValidationBatchCompletion,
    ) -> tuple[CompletedValidationProjection, ...]: ...

    def fail(
        self,
        context: ValidationLockedContext,
        failure: ValidationBatchFailure,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedValidationBatch:
    claim: ValidationBatchClaimRequest
    admission: ValidationAttemptAdmission
    invocation: PreparedValidationInvocation
    policy_inputs: tuple[ValidationPolicyInput, ...]


@dataclass(frozen=True, slots=True)
class ValidationPreparationCandidate:
    claim: ValidationCandidateClaim
    request: CandidateValidationRequest
    policy_input: ValidationPolicyInput


@dataclass(frozen=True, slots=True)
class PreparedValidationPartition:
    batches: tuple[PreparedValidationBatch, ...]
    human_only_review_item_ids: tuple[int, ...]


def prepare_validation_partition(
    candidates: Sequence[ValidationPreparationCandidate],
    *,
    workflow_thread_id: str,
    settings: Settings,
    validator: object,
    expected_owner_permission_hmac: str,
    expected_source_state_hmac: str,
    expected_effective_mode: Literal['shadow', 'enforce'],
) -> PreparedValidationPartition:
    """Deterministic first-fit partition with one final prepare per batch."""
    ordered = tuple(
        sorted(candidates, key=lambda item: item.claim.validation_key)
    )
    sizer = ValidationFrameSizer(settings=settings)
    bins: list[list[ValidationPreparationCandidate]] = []
    human_only: list[int] = []
    for index, candidate in enumerate(ordered):
        claim = candidate.claim
        if index >= 5:
            human_only.append(claim.review_item_id)
            continue
        placed = False
        for bucket in bins:
            tentative = (*bucket, candidate)
            if len(tentative) > 4:
                continue
            try:
                size = sizer.measure(tuple(item.request for item in tentative))
            except AutoReviewValidationError:
                continue
            if size.evidence_slot_count <= 12:
                bucket.append(candidate)
                placed = True
                break
        if placed:
            continue
        try:
            size = sizer.measure((candidate.request,))
        except AutoReviewValidationError:
            human_only.append(claim.review_item_id)
            continue
        if size.evidence_slot_count > 12 or len(bins) >= 2:
            human_only.append(claim.review_item_id)
            continue
        bins.append([candidate])

    prepared_batches: list[PreparedValidationBatch] = []
    prepare_many = getattr(validator, 'prepare_many', None)
    if not callable(prepare_many):
        raise AutoReviewValidationError('auto-review validation is unavailable')
    for bucket in bins:
        requests = tuple(item.request for item in bucket)
        invocation = prepare_many(requests)
        claims = tuple(item.claim for item in bucket)
        identity = claims[0].identity
        reserved_cost = _token_cost(
            invocation.framed_input_tokens,
            invocation.max_output_tokens,
            input_price=identity.input_cost_per_1m_tokens,
            output_price=identity.output_cost_per_1m_tokens,
        )
        fingerprint = derive_validation_batch_fingerprint(
            workflow_thread_id=workflow_thread_id,
            candidates=claims,
            prepared_content_hmac=invocation.prepared_content_hmac,
            serialized_character_count=invocation.character_count,
            framed_input_tokens=invocation.framed_input_tokens,
            reserved_input_tokens=invocation.framed_input_tokens,
            reserved_output_tokens=invocation.max_output_tokens,
            reserved_cost_usd=reserved_cost,
            settings=settings,
        )
        claim_request = ValidationBatchClaimRequest(
            workflow_thread_id=workflow_thread_id,
            batch_fingerprint=fingerprint,
            candidates=claims,
            max_provider_attempts=1,
            prepared_content_hmac=invocation.prepared_content_hmac,
            serialized_character_count=invocation.character_count,
            framed_input_tokens=invocation.framed_input_tokens,
            reserved_input_tokens=invocation.framed_input_tokens,
            reserved_output_tokens=invocation.max_output_tokens,
            reserved_cost_usd=reserved_cost,
        )
        admission = ValidationAttemptAdmission(
            validation_call_id=0,
            workflow_thread_id=workflow_thread_id,
            batch_fingerprint=fingerprint,
            lease_token='',
            prepared_content_hmac=invocation.prepared_content_hmac,
            serialized_character_count=invocation.character_count,
            framed_input_tokens=invocation.framed_input_tokens,
            max_output_tokens=invocation.max_output_tokens,
            recomputed_reserved_cost_usd=reserved_cost,
            expected_owner_permission_hmac=expected_owner_permission_hmac,
            expected_source_state_hmac=expected_source_state_hmac,
            fingerprint_key_version=identity.fingerprint_key_version,
            fingerprint_key_material_verifier=(
                identity.fingerprint_key_material_verifier
            ),
            token_estimator_version=identity.token_estimator_version,
            cost_policy_version=identity.cost_policy_version,
            expected_effective_mode=expected_effective_mode,
            provider_timeout_seconds=identity.provider_timeout_seconds,
            provider_send_start_window_seconds=(
                identity.provider_send_start_window_seconds
            ),
            provider_attempt_lease_seconds=(
                identity.provider_attempt_lease_seconds
            ),
            provider_commit_grace_seconds=identity.provider_commit_grace_seconds,
        )
        prepared_batches.append(
            PreparedValidationBatch(
                claim=claim_request,
                admission=admission,
                invocation=invocation,
                policy_inputs=tuple(
                    replace(
                        item.policy_input,
                        expected_candidate_slot_id=(
                            invocation.candidate_slot_ids[index]
                        ),
                        expected_evidence_slot_ids=(
                            invocation.candidate_evidence_slot_ids[index]
                        ),
                    )
                    for index, item in enumerate(bucket)
                ),
            )
        )
    return PreparedValidationPartition(
        batches=tuple(prepared_batches),
        human_only_review_item_ids=tuple(item for item in human_only if item > 0),
    )


@dataclass(frozen=True, slots=True)
class ValidationOrchestrationResult:
    disposition: Literal['completed', 'replayed', 'busy', 'failed']
    validation_call_id: int
    completed: tuple[CompletedValidationProjection, ...]
    failure_reason_code: AutoReviewPolicyReasonCode | None


class AutoReviewValidationOrchestrator:
    """Coordinates one prepared batch without holding DB state across Terra."""

    def __init__(
        self,
        *,
        store: ValidationStore,
        validator_factory: AutoReviewValidatorFactory,
        policy_engine: AutoReviewPolicyEngine | None = None,
    ) -> None:
        self._store = store
        self._validator_factory = validator_factory
        self._policy = policy_engine or AutoReviewPolicyEngine()

    def execute(
        self, batch: PreparedValidationBatch
    ) -> ValidationOrchestrationResult:
        claim = self._store.claim_or_replay(batch.claim)
        if claim.disposition == 'busy':
            return ValidationOrchestrationResult(
                disposition='busy',
                validation_call_id=claim.validation_call_id,
                completed=(),
                failure_reason_code=None,
            )
        if claim.disposition == 'replayed':
            return ValidationOrchestrationResult(
                disposition=(
                    'replayed' if claim.terminal_status == 'completed' else 'failed'
                ),
                validation_call_id=claim.validation_call_id,
                completed=claim.completed,
                failure_reason_code=claim.failure_reason_code,
            )
        if (
            claim.lease_token is None
            or batch.admission.workflow_thread_id != batch.claim.workflow_thread_id
            or batch.admission.batch_fingerprint != batch.claim.batch_fingerprint
        ):
            raise AutoReviewValidationError(
                'validation admission does not match the canonical claim'
            )

        context = ValidationLockedContext(
            validation_call_id=claim.validation_call_id,
            workflow_thread_id=batch.claim.workflow_thread_id,
            batch_fingerprint=batch.claim.batch_fingerprint,
            lease_token=claim.lease_token,
        )
        admission = replace(
            batch.admission,
            validation_call_id=claim.validation_call_id,
            lease_token=claim.lease_token,
        )
        grant = self._store.mark_attempt_started(admission)
        usage: list[ValidationUsage] = []
        validator = self._validator_factory.create(usage.append)
        try:
            results = tuple(validator.invoke_prepared(batch.invocation, grant))
        except AutoReviewValidationError:
            self._store.fail(
                context,
                ValidationBatchFailure(
                    validation_call_id=context.validation_call_id,
                    workflow_thread_id=context.workflow_thread_id,
                    batch_fingerprint=context.batch_fingerprint,
                    lease_token=context.lease_token,
                    reason_code='validator_unavailable',
                    usage_known=False,
                    input_tokens=None,
                    output_tokens=None,
                ),
            )
            return ValidationOrchestrationResult(
                disposition='failed',
                validation_call_id=context.validation_call_id,
                completed=(),
                failure_reason_code='validator_unavailable',
            )
        if len(usage) != 1 or len(results) != len(batch.policy_inputs):
            self._store.fail(
                context,
                ValidationBatchFailure(
                    validation_call_id=context.validation_call_id,
                    workflow_thread_id=context.workflow_thread_id,
                    batch_fingerprint=context.batch_fingerprint,
                    lease_token=context.lease_token,
                    reason_code='validator_malformed',
                    usage_known=(len(usage) == 1),
                    input_tokens=usage[0].input_tokens if len(usage) == 1 else None,
                    output_tokens=usage[0].output_tokens if len(usage) == 1 else None,
                ),
            )
            return ValidationOrchestrationResult(
                disposition='failed',
                validation_call_id=context.validation_call_id,
                completed=(),
                failure_reason_code='validator_malformed',
            )

        decisions = self._policy.evaluate_batch(
            tuple(
                replace(policy_input, result=result)
                for policy_input, result in zip(
                    batch.policy_inputs, results, strict=True
                )
            )
        )
        ordered_claims = tuple(
            sorted(batch.claim.candidates, key=lambda item: item.validation_key)
        )
        completion = ValidationBatchCompletion(
            validation_call_id=context.validation_call_id,
            workflow_thread_id=context.workflow_thread_id,
            batch_fingerprint=context.batch_fingerprint,
            lease_token=context.lease_token,
            candidates=tuple(
                ValidationCandidateCompletion(
                    review_item_id=candidate.review_item_id,
                    validation_key=candidate.validation_key,
                    result=result,
                    policy_decision=decision.decision,
                    policy_reason_codes=decision.reason_codes,
                )
                for candidate, result, decision in zip(
                    ordered_claims, results, decisions, strict=True
                )
            ),
            input_tokens=usage[0].input_tokens,
            output_tokens=usage[0].output_tokens,
        )
        completed = self._store.complete(context, completion)
        return ValidationOrchestrationResult(
            disposition='completed',
            validation_call_id=context.validation_call_id,
            completed=completed,
            failure_reason_code=None,
        )
