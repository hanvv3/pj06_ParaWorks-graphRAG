from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from backend.app.agent_runtime.auto_review_orchestrator import (
    AutoReviewValidationOrchestrator,
    PreparedValidationBatch,
    ValidationPreparationCandidate,
    prepare_validation_partition,
)
from backend.app.agent_runtime.auto_review_policy import (
    SUPPORTED_VALIDATION_IDENTITY,
    CandidateValidationRequest,
    EligibilityPolicyResult,
    ValidationClaimInput,
    ValidationEvidenceSlot,
    ValidationPolicyInput,
)
from backend.app.agent_runtime.auto_review_validation_store import (
    AutoReviewValidationIdentity,
    ValidationAttemptAdmission,
    ValidationBatchClaimRequest,
    ValidationCandidateClaim,
    ValidationClaimResult,
)
from backend.app.agent_runtime.auto_review_validator import (
    AutoReviewValidationError,
    AutoReviewValidator,
    PreparedValidationInvocation,
    ValidationUsage,
)
from backend.app.core.config import Settings
from backend.app.schemas.auto_review import CandidateValidationResult


def _identity() -> AutoReviewValidationIdentity:
    return AutoReviewValidationIdentity(
        workflow_execution_identity_hmac='1' * 64,
        security_scope_hmac='2' * 64,
        candidate_key='3' * 64,
        evidence_version_hash='4' * 64,
        normalized_claim_fingerprint='5' * 64,
        candidate_generation_fingerprint='6' * 64,
        validator_provider='openai',
        validator_model='gpt-5.6-terra',
        reasoning_effort='medium',
        validator_prompt_version='auto-review-validation:v1',
        validator_output_contract_version='candidate-validation-batch:v1',
        policy_version='auto-review-policy:v1',
        fingerprint_key_version='task9-key-v1',
        fingerprint_key_material_verifier='7' * 64,
        token_estimator_version='openai-o200k-chat:v1',
        tokenizer_encoding='o200k_base',
        max_input_tokens=6000,
        max_output_tokens=3072,
        max_candidates_per_batch=4,
        max_batches_per_workflow=2,
        max_candidates_per_workflow=5,
        max_provider_attempts=1,
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
        input_cost_per_1m_tokens=Decimal('2.000000'),
        output_cost_per_1m_tokens=Decimal('12.000000'),
        cost_policy_version='auto-review-cost:v1',
        provider_safety_state_version=1,
        rollout_control_epoch=2,
        authorized_percentage_at_launch=10,
        rollout_authorization_generation=3,
        confirmed_extraction_cost_ceiling_usd=Decimal('0.083580'),
        confirmed_validation_cost_ceiling_usd=Decimal('0.097728'),
        confirmed_total_cost_ceiling_usd=Decimal('0.181308'),
    )


def _result() -> CandidateValidationResult:
    return CandidateValidationResult.model_validate(
        {
            'candidate_slot_id': 'C01',
            'claim_results': [
                {
                    'field_key': 'title',
                    'verdict': 'supported',
                    'claim_scope': 'direct_fact',
                    'entailment_score': '0.9900',
                    'evidence_slot_ids': ['E01'],
                },
                {
                    'field_key': 'result_summary',
                    'verdict': 'supported',
                    'claim_scope': 'direct_fact',
                    'entailment_score': '0.9900',
                    'evidence_slot_ids': ['E01'],
                },
            ],
            'uncertainty_codes': [],
            'conflict_codes': [],
        }
    )


def _batch() -> PreparedValidationBatch:
    identity = _identity()
    candidate = ValidationCandidateClaim(
        review_item_id=101,
        validation_key='8' * 64,
        identity=identity,
    )
    claim = ValidationBatchClaimRequest(
        workflow_thread_id='workflow-task9',
        batch_fingerprint='9' * 64,
        candidates=(candidate,),
        max_provider_attempts=1,
        prepared_content_hmac='a' * 64,
        serialized_character_count=900,
        framed_input_tokens=1200,
        reserved_input_tokens=1200,
        reserved_output_tokens=3072,
        reserved_cost_usd=Decimal('0.039264'),
    )
    admission = ValidationAttemptAdmission(
        validation_call_id=0,
        workflow_thread_id=claim.workflow_thread_id,
        batch_fingerprint=claim.batch_fingerprint,
        lease_token='',
        prepared_content_hmac=claim.prepared_content_hmac,
        serialized_character_count=claim.serialized_character_count,
        framed_input_tokens=claim.framed_input_tokens,
        max_output_tokens=3072,
        recomputed_reserved_cost_usd=claim.reserved_cost_usd,
        expected_owner_permission_hmac='b' * 64,
        expected_source_state_hmac='c' * 64,
        fingerprint_key_version=identity.fingerprint_key_version,
        fingerprint_key_material_verifier=identity.fingerprint_key_material_verifier,
        token_estimator_version=identity.token_estimator_version,
        cost_policy_version=identity.cost_policy_version,
        expected_effective_mode='shadow',
        provider_timeout_seconds=60,
        provider_send_start_window_seconds=5,
        provider_attempt_lease_seconds=120,
        provider_commit_grace_seconds=30,
    )
    invocation = PreparedValidationInvocation(
        messages=(('system', 'bounded'), ('human', 'bounded')),
        canonical_text='bounded',
        canonical_bytes=b'bounded',
        response_schema_framing='{}',
        prepared_content_hmac=claim.prepared_content_hmac,
        character_count=claim.serialized_character_count,
        encoded_input_tokens=688,
        framed_input_tokens=claim.framed_input_tokens,
        max_output_tokens=3072,
        candidate_slot_ids=('C01',),
        evidence_slot_ids=('E01',),
        expected_claim_fields=(('title', 'result_summary'),),
        candidate_evidence_slot_ids=(('E01',),),
    )
    policy = ValidationPolicyInput(
        eligibility=EligibilityPolicyResult(
            decision='eligible', reason_codes=('eligible',)
        ),
        expected_candidate_slot_id='C01',
        expected_claim_fields=('title', 'result_summary'),
        expected_evidence_slot_ids=('E01',),
        validator_available=True,
        validator_malformed=False,
        result=None,
        validation_identity=SUPPORTED_VALIDATION_IDENTITY,
        post_validation_state_matches=True,
    )
    return PreparedValidationBatch(
        claim=claim,
        admission=admission,
        invocation=invocation,
        policy_inputs=(policy,),
    )


class _BoundaryStore:
    def __init__(self) -> None:
        self.transaction_open = False
        self.failed = []
        self.completed = []

    def claim_or_replay(self, request):
        self.transaction_open = True
        try:
            return ValidationClaimResult(
                disposition='claimed',
                validation_call_id=42,
                lease_token='d' * 64,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=120),
                terminal_status=None,
                failure_reason_code=None,
                completed=(),
            )
        finally:
            self.transaction_open = False

    def mark_attempt_started(self, admission):
        self.transaction_open = True
        try:
            assert admission.validation_call_id == 42
            assert admission.lease_token == 'd' * 64
            return object()
        finally:
            self.transaction_open = False

    def complete(self, context, completion):
        self.transaction_open = True
        try:
            self.completed.append((context, completion))
            return ()
        finally:
            self.transaction_open = False

    def fail(self, context, failure):
        self.transaction_open = True
        try:
            self.failed.append((context, failure))
        finally:
            self.transaction_open = False


class _Factory:
    def __init__(self, store: _BoundaryStore, *, fail: bool = False) -> None:
        self.store = store
        self.fail = fail
        self.calls = 0

    def create(self, usage_sink):
        parent = self

        class Validator:
            @staticmethod
            def invoke_prepared(invocation, grant):
                del invocation, grant
                assert parent.store.transaction_open is False
                parent.calls += 1
                if parent.fail:
                    raise AutoReviewValidationError('bounded failure')
                usage_sink(ValidationUsage(101, 51, Decimal('0.000814')))
                return [_result()]

        return Validator()


def test_provider_call_observes_no_open_session_or_row_lock() -> None:
    store = _BoundaryStore()
    factory = _Factory(store)
    orchestrator = AutoReviewValidationOrchestrator(
        store=store,
        validator_factory=factory,
    )

    result = orchestrator.execute(_batch())

    assert result.disposition == 'completed'
    assert factory.calls == 1
    assert len(store.completed) == 1
    assert not store.failed
    completion = store.completed[0][1]
    assert completion.input_tokens == 101
    assert completion.output_tokens == 51
    assert completion.candidates[0].policy_decision == 'auto_approve'


def test_malformed_or_timeout_result_leaves_candidate_pending() -> None:
    store = _BoundaryStore()
    factory = _Factory(store, fail=True)
    orchestrator = AutoReviewValidationOrchestrator(
        store=store,
        validator_factory=factory,
    )

    result = orchestrator.execute(_batch())

    assert result.disposition == 'failed'
    assert result.failure_reason_code == 'validator_unavailable'
    assert len(store.failed) == 1
    assert store.failed[0][1].usage_known is False
    assert not store.completed


def test_replayed_result_never_constructs_or_invokes_validator() -> None:
    store = _BoundaryStore()
    store.claim_or_replay = lambda request: ValidationClaimResult(
        disposition='replayed',
        validation_call_id=42,
        lease_token=None,
        lease_expires_at=None,
        terminal_status='failed',
        failure_reason_code='validator_unavailable',
        completed=(),
    )
    factory = _Factory(store)
    orchestrator = AutoReviewValidationOrchestrator(
        store=store,
        validator_factory=factory,
    )

    result = orchestrator.execute(_batch())

    assert result.disposition == 'failed'
    assert factory.calls == 0


def test_candidate_permutation_has_same_partition_and_one_final_prepare() -> None:
    settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='task9-secret-at-least-32-bytes-long',
        agent_runtime_fingerprint_key_version='task9-key-v1',
    )
    base = _batch()
    first_claim = base.claim.candidates[0]
    second_claim = replace(
        first_claim,
        review_item_id=102,
        validation_key='7' * 64,
        identity=replace(
            first_claim.identity,
            candidate_key='6' * 64,
            normalized_claim_fingerprint='5' * 64,
        ),
    )
    first_request = CandidateValidationRequest(
        candidate_slot_id='C01',
        item_type='timeline_event',
        claims=(
            ValidationClaimInput(field_key='title', text='제목'),
            ValidationClaimInput(field_key='result_summary', text='완료 결과'),
        ),
        evidence_slots=(ValidationEvidenceSlot(slot_id='E01', text='근거 1'),),
    )
    second_request = CandidateValidationRequest(
        candidate_slot_id='C01',
        item_type='history_event',
        claims=(
            ValidationClaimInput(field_key='title', text='이력 제목'),
            ValidationClaimInput(field_key='reason', text='결정 이유'),
        ),
        evidence_slots=(ValidationEvidenceSlot(slot_id='E01', text='근거 2'),),
    )
    first = ValidationPreparationCandidate(
        claim=first_claim,
        request=first_request,
        policy_input=base.policy_inputs[0],
    )
    second = ValidationPreparationCandidate(
        claim=second_claim,
        request=second_request,
        policy_input=replace(
            base.policy_inputs[0],
            expected_claim_fields=('title', 'reason'),
        ),
    )

    class CountingValidator:
        def __init__(self) -> None:
            self.calls = 0
            self.inner = AutoReviewValidator(
                settings=settings,
                dispatcher=object(),
                usage_sink=lambda usage: None,
            )

        def prepare_many(self, requests):
            self.calls += 1
            return self.inner.prepare_many(requests)

    def prepare(values):
        validator = CountingValidator()
        partition = prepare_validation_partition(
            values,
            workflow_thread_id='workflow-task9',
            settings=settings,
            validator=validator,
            expected_owner_permission_hmac='b' * 64,
            expected_source_state_hmac='c' * 64,
            expected_effective_mode='shadow',
        )
        return partition, validator.calls

    forward, forward_calls = prepare((first, second))
    reverse, reverse_calls = prepare((second, first))

    assert forward_calls == reverse_calls == 1
    assert len(forward.batches) == 1
    assert not forward.human_only_review_item_ids
    assert forward.batches[0].claim.batch_fingerprint == (
        reverse.batches[0].claim.batch_fingerprint
    )
    assert tuple(
        candidate.validation_key
        for candidate in forward.batches[0].claim.candidates
    ) == ('7' * 64, '8' * 64)
    invocation = forward.batches[0].invocation
    assert invocation.candidate_slot_ids == ('C01', 'C02')
    assert invocation.candidate_evidence_slot_ids == (('E01',), ('E02',))
    assert forward.batches[0].claim.serialized_character_count == (
        invocation.character_count
    )
    assert forward.batches[0].claim.framed_input_tokens == (
        invocation.framed_input_tokens
    )
