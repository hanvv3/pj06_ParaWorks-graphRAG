import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from backend.app.agent_runtime.auto_review_validation_store import (
    AutoReviewValidationIdentity,
    ValidationAttemptAdmission,
    ValidationBatchClaimRequest,
    ValidationBatchCompletion,
    ValidationBatchFailure,
    ValidationCallLedger,
    ValidationCallSnapshot,
    ValidationCandidateClaim,
    ValidationCandidateCompletion,
    ValidationLockedContext,
    ValidationStoreError,
    derive_validation_batch_fingerprint,
    derive_validation_key,
)
from backend.app.core.config import Settings
from backend.app.schemas.auto_review import CandidateValidationResult


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='validation-store-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='validation-store-v1',
    )


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
        fingerprint_key_version='validation-store-v1',
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


def _different_value(name: str, value: object) -> object:
    if isinstance(value, Decimal):
        return value + Decimal('0.000001')
    if isinstance(value, int):
        return value + 1
    assert isinstance(value, str)
    if len(value) == 64 and set(value) <= set('123456789abcdef'):
        return 'f' * 64
    return f'{value}:changed'


def test_validation_key_changes_for_every_frozen_identity_component() -> None:
    settings = _settings()
    identity = _identity()
    baseline = derive_validation_key(identity, settings=settings)

    changed = {
        field.name: derive_validation_key(
            replace(
                identity,
                **{
                    field.name: _different_value(
                        field.name, getattr(identity, field.name)
                    )
                },
            ),
            settings=settings,
        )
        for field in fields(identity)
    }

    assert len(baseline) == 64
    assert all(value != baseline for value in changed.values())
    assert len(set(changed.values())) == len(changed)


def test_validation_key_changes_with_provider_safety_or_rollout_control() -> None:
    settings = _settings()
    identity = _identity()
    values = {
        derive_validation_key(identity, settings=settings),
        derive_validation_key(
            replace(identity, provider_safety_state_version=2),
            settings=settings,
        ),
        derive_validation_key(
            replace(identity, rollout_control_epoch=3),
            settings=settings,
        ),
        derive_validation_key(
            replace(identity, rollout_authorization_generation=4),
            settings=settings,
        ),
    }

    assert len(values) == 4


def test_same_content_in_two_workflows_and_scopes_never_collides() -> None:
    settings = _settings()
    identity = _identity()
    keys = {
        derive_validation_key(identity, settings=settings),
        derive_validation_key(
            replace(identity, workflow_execution_identity_hmac='8' * 64),
            settings=settings,
        ),
        derive_validation_key(
            replace(identity, security_scope_hmac='9' * 64),
            settings=settings,
        ),
    }

    assert len(keys) == 3


def test_candidate_permutation_produces_identical_batch_fingerprint() -> None:
    settings = _settings()
    identity = _identity()
    candidates = tuple(
        ValidationCandidateClaim(
            review_item_id=index,
            validation_key=derive_validation_key(
                replace(identity, candidate_key=str(index) * 64),
                settings=settings,
            ),
            identity=replace(identity, candidate_key=str(index) * 64),
        )
        for index in (1, 2, 3)
    )
    kwargs = {
        'workflow_thread_id': 'workflow-v21-1',
        'prepared_content_hmac': 'a' * 64,
        'serialized_character_count': 900,
        'framed_input_tokens': 1200,
        'reserved_input_tokens': 1200,
        'reserved_output_tokens': 3072,
        'reserved_cost_usd': Decimal('0.039264'),
        'settings': settings,
    }

    forward = derive_validation_batch_fingerprint(
        candidates=candidates,
        **kwargs,
    )
    reverse = derive_validation_batch_fingerprint(
        candidates=tuple(reversed(candidates)),
        **kwargs,
    )

    assert forward == reverse
    assert forward != derive_validation_batch_fingerprint(
        candidates=candidates,
        **{**kwargs, 'prepared_content_hmac': 'b' * 64},
    )


def _claim_request(*, candidate_count: int = 1) -> ValidationBatchClaimRequest:
    settings = _settings()
    identity = _identity()
    candidates = tuple(
        ValidationCandidateClaim(
            review_item_id=100 + index,
            validation_key=derive_validation_key(
                replace(identity, candidate_key=str(index) * 64),
                settings=settings,
            ),
            identity=replace(identity, candidate_key=str(index) * 64),
        )
        for index in range(1, candidate_count + 1)
    )
    values = {
        'workflow_thread_id': 'workflow-v21-1',
        'candidates': candidates,
        'max_provider_attempts': 1,
        'prepared_content_hmac': 'a' * 64,
        'serialized_character_count': 900,
        'framed_input_tokens': 1200,
        'reserved_input_tokens': 1200,
        'reserved_output_tokens': 3072,
        'reserved_cost_usd': Decimal('0.039264'),
    }
    return ValidationBatchClaimRequest(
        batch_fingerprint=derive_validation_batch_fingerprint(
            settings=settings,
            **{key: value for key, value in values.items() if key != 'max_provider_attempts'},
        ),
        **values,
    )


def _replace_claim(
    request: ValidationBatchClaimRequest,
    **changes: object,
) -> ValidationBatchClaimRequest:
    values = {
        field.name: getattr(request, field.name)
        for field in fields(request)
        if field.name != 'batch_fingerprint'
    }
    values.update(changes)
    return ValidationBatchClaimRequest(
        batch_fingerprint=derive_validation_batch_fingerprint(
            settings=_settings(),
            **{
                key: value
                for key, value in values.items()
                if key != 'max_provider_attempts'
            },
        ),
        **values,
    )


def test_only_expired_claim_can_be_reclaimed_with_cas() -> None:
    now = [datetime(2026, 8, 30, 9, 0, tzinfo=UTC)]
    tokens = iter(('a' * 64, 'b' * 64))
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now[0],
        lease_token_factory=lambda: next(tokens),
    )
    request = _claim_request()

    claimed = store.claim_or_replay(request)
    busy = store.claim_or_replay(request)
    now[0] += timedelta(seconds=119)
    still_busy = store.claim_or_replay(request)
    now[0] += timedelta(seconds=1)
    reclaimed = store.claim_or_replay(request)

    assert claimed.disposition == 'claimed'
    assert claimed.lease_token == 'a' * 64
    assert busy.disposition == still_busy.disposition == 'busy'
    assert busy.lease_token is None
    assert reclaimed.disposition == 'claimed'
    assert reclaimed.lease_token == 'b' * 64


def test_store_owns_lease_token_and_database_timestamps() -> None:
    database_now = datetime(2026, 8, 30, 10, 30, tzinfo=UTC)
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: database_now,
        lease_token_factory=lambda: 'c' * 64,
    )

    claimed = store.claim_or_replay(_claim_request())

    assert claimed.lease_token == 'c' * 64
    assert claimed.lease_expires_at == database_now + timedelta(seconds=120)


def test_concurrent_claims_produce_one_canonical_owner() -> None:
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: datetime(2026, 8, 30, 10, 45, tzinfo=UTC),
    )
    request = _claim_request()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _: store.claim_or_replay(request), range(8)))

    assert sum(result.disposition == 'claimed' for result in results) == 1
    assert sum(result.disposition == 'busy' for result in results) == 7
    assert len({result.validation_call_id for result in results}) == 1


@pytest.mark.parametrize(
    'change',
    (
        {'serialized_character_count': 12001},
        {'framed_input_tokens': 6001, 'reserved_input_tokens': 6001},
        {'reserved_output_tokens': 3073},
        {'reserved_cost_usd': Decimal('0.039263')},
    ),
)
def test_claim_rejects_cap_or_reservation_drift(change) -> None:
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: datetime(2026, 8, 30, 10, 50, tzinfo=UTC),
    )

    with pytest.raises(ValidationStoreError):
        store.claim_or_replay(_replace_claim(_claim_request(), **change))


def test_claim_rejects_validation_key_or_price_registry_drift() -> None:
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: datetime(2026, 8, 30, 10, 55, tzinfo=UTC),
    )
    request = _claim_request()
    bad_key_candidate = replace(request.candidates[0], validation_key='f' * 64)
    bad_price_identity = replace(
        request.candidates[0].identity,
        input_cost_per_1m_tokens=Decimal('2.100000'),
    )
    bad_price_candidate = replace(
        request.candidates[0],
        identity=bad_price_identity,
        validation_key=derive_validation_key(
            bad_price_identity,
            settings=_settings(),
        ),
    )

    with pytest.raises(ValidationStoreError):
        store.claim_or_replay(
            _replace_claim(request, candidates=(bad_key_candidate,))
        )
    with pytest.raises(ValidationStoreError):
        store.claim_or_replay(
            _replace_claim(request, candidates=(bad_price_candidate,))
        )


def _admission(
    request: ValidationBatchClaimRequest,
    *,
    validation_call_id: int,
    lease_token: str,
) -> ValidationAttemptAdmission:
    identity = request.candidates[0].identity
    return ValidationAttemptAdmission(
        validation_call_id=validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=lease_token,
        prepared_content_hmac=request.prepared_content_hmac,
        serialized_character_count=request.serialized_character_count,
        framed_input_tokens=request.framed_input_tokens,
        max_output_tokens=identity.max_output_tokens,
        recomputed_reserved_cost_usd=request.reserved_cost_usd,
        expected_owner_permission_hmac='d' * 64,
        expected_source_state_hmac='e' * 64,
        fingerprint_key_version=identity.fingerprint_key_version,
        fingerprint_key_material_verifier=(
            identity.fingerprint_key_material_verifier
        ),
        token_estimator_version=identity.token_estimator_version,
        cost_policy_version=identity.cost_policy_version,
        expected_effective_mode='shadow',
        provider_timeout_seconds=identity.provider_timeout_seconds,
        provider_send_start_window_seconds=(
            identity.provider_send_start_window_seconds
        ),
        provider_attempt_lease_seconds=(
            identity.provider_attempt_lease_seconds
        ),
        provider_commit_grace_seconds=identity.provider_commit_grace_seconds,
    )


def test_attempt_grant_exists_only_after_marker_commit() -> None:
    now = datetime(2026, 8, 30, 11, 0, tzinfo=UTC)
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now,
        lease_token_factory=lambda: 'f' * 64,
    )
    request = _claim_request()
    claim = store.claim_or_replay(request)
    admission = _admission(
        request,
        validation_call_id=claim.validation_call_id,
        lease_token=claim.lease_token or '',
    )
    commits: list[str] = []

    grant = store.mark_attempt_started(
        admission,
        commit=lambda: commits.append('committed'),
    )

    assert commits == ['committed']
    assert grant.provider_timeout_seconds == 60
    assert grant.authoritative_lease_expires_at == now + timedelta(seconds=120)
    assert grant.permit.consumed is False


def test_failed_attempt_marker_commit_issues_no_grant_and_can_retry_marker() -> None:
    now = datetime(2026, 8, 30, 11, 30, tzinfo=UTC)
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now,
        lease_token_factory=lambda: '1' * 64,
    )
    request = _claim_request()
    claim = store.claim_or_replay(request)
    admission = _admission(
        request,
        validation_call_id=claim.validation_call_id,
        lease_token=claim.lease_token or '',
    )

    with pytest.raises(RuntimeError, match='commit failed'):
        store.mark_attempt_started(
            admission,
            commit=lambda: (_ for _ in ()).throw(RuntimeError('commit failed')),
        )

    grant = store.mark_attempt_started(admission, commit=lambda: None)
    assert grant.permit.consumed is False


def test_validation_send_permit_is_one_use_and_nonserializable() -> None:
    now = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now,
        lease_token_factory=lambda: '2' * 64,
    )
    request = _claim_request()
    claim = store.claim_or_replay(request)
    admission = _admission(
        request,
        validation_call_id=claim.validation_call_id,
        lease_token=claim.lease_token or '',
    )
    grant = store.mark_attempt_started(admission, commit=lambda: None)

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072
        canonical_bytes = b'must-not-be-read-by-hook'

    calls: list[tuple[int, object]] = []

    result = store.dispatch_prepared_validation(
        invocation=Invocation(),
        provider=lambda invocation, *, timeout, http_hook: calls.append(
            (timeout, http_hook)
        )
        or 'ok',
        grant=grant,
    )

    assert result == 'ok'
    assert grant.permit.consumed is True
    assert len(calls) == 1
    with pytest.raises(ValidationStoreError, match='grant'):
        store.dispatch_prepared_validation(
            invocation=Invocation(),
            provider=lambda *args, **kwargs: None,
            grant=grant,
        )
    with pytest.raises(TypeError):
        copy.copy(grant.permit)
    with pytest.raises(TypeError):
        pickle.dumps(grant.permit)


def _validation_result(candidate_slot_id: str) -> CandidateValidationResult:
    return CandidateValidationResult.model_validate(
        {
            'candidate_slot_id': candidate_slot_id,
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


def _started_call(*, candidate_count: int = 1):
    now = datetime(2026, 8, 30, 13, 0, tzinfo=UTC)
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now,
        lease_token_factory=lambda: '3' * 64,
    )
    request = _claim_request(candidate_count=candidate_count)
    claim = store.claim_or_replay(request)
    admission = _admission(
        request,
        validation_call_id=claim.validation_call_id,
        lease_token=claim.lease_token or '',
    )
    grant = store.mark_attempt_started(admission, commit=lambda: None)

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072
        canonical_bytes = b'ephemeral'

    store.dispatch_prepared_validation(
        invocation=Invocation(),
        provider=lambda *args, **kwargs: 'provider-result',
        grant=grant,
    )
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token or '',
    )
    return store, request, context


def test_completed_validation_replays_without_second_provider_call() -> None:
    store, request, context = _started_call(candidate_count=2)
    completions = tuple(
        ValidationCandidateCompletion(
            review_item_id=candidate.review_item_id,
            validation_key=candidate.validation_key,
            result=_validation_result(f'C{index:02d}'),
            policy_decision='auto_approve',
            policy_reason_codes=('eligible',),
        )
        for index, candidate in enumerate(request.candidates, start=1)
    )

    completed = store.complete(
        context,
        ValidationBatchCompletion(
            validation_call_id=context.validation_call_id,
            workflow_thread_id=context.workflow_thread_id,
            batch_fingerprint=context.batch_fingerprint,
            lease_token=context.lease_token,
            candidates=completions,
            input_tokens=121,
            output_tokens=41,
        ),
    )
    replay = store.claim_or_replay(request)

    assert replay.disposition == 'replayed'
    assert replay.terminal_status == 'completed'
    assert replay.completed == completed
    assert sum(item.input_tokens for item in completed) == 121
    assert sum(item.output_tokens for item in completed) == 41
    assert sum(item.estimated_cost_usd for item in completed) == Decimal(
        '0.000734'
    )
    assert all(item.estimated_cost_usd >= 0 for item in completed)


def test_failed_validation_replay_is_sanitized_and_keeps_batch_pending() -> None:
    store, request, context = _started_call(candidate_count=2)

    store.fail(
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
    replay = store.claim_or_replay(request)

    assert replay.disposition == 'replayed'
    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'validator_unavailable'
    assert replay.completed == ()
    snapshot = store.snapshot(context)
    assert isinstance(snapshot, ValidationCallSnapshot)
    assert snapshot.charged_input_tokens == request.reserved_input_tokens
    assert snapshot.charged_output_tokens == request.reserved_output_tokens
    assert snapshot.charged_cost_usd == request.reserved_cost_usd


def test_known_usage_over_reservation_opens_breaker_once_and_never_completes() -> None:
    store, request, context = _started_call(candidate_count=1)
    candidate = request.candidates[0]
    completion = ValidationBatchCompletion(
        validation_call_id=context.validation_call_id,
        workflow_thread_id=context.workflow_thread_id,
        batch_fingerprint=context.batch_fingerprint,
        lease_token=context.lease_token,
        candidates=(
            ValidationCandidateCompletion(
                review_item_id=candidate.review_item_id,
                validation_key=candidate.validation_key,
                result=_validation_result('C01'),
                policy_decision='auto_approve',
                policy_reason_codes=('eligible',),
            ),
        ),
        input_tokens=request.reserved_input_tokens + 1,
        output_tokens=request.reserved_output_tokens,
    )

    completed = store.complete(context, completion)
    first_snapshot = store.snapshot(context)
    replay = store.claim_or_replay(request)
    second_snapshot = store.snapshot(context)

    assert completed == ()
    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'budget_exceeded'
    assert first_snapshot.budget_overrun is True
    assert first_snapshot.charged_cost_usd > request.reserved_cost_usd
    assert store.validation_breaker_open is True
    assert store.provider_safety_event_count == 1
    assert second_snapshot == first_snapshot
    assert store.provider_safety_event_count == 1


def test_expired_started_attempt_never_retries_and_recovery_charges_reserve() -> None:
    now = [datetime(2026, 8, 30, 14, 0, tzinfo=UTC)]
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now[0],
        lease_token_factory=lambda: '4' * 64,
    )
    request = _claim_request()
    claim = store.claim_or_replay(request)
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token or '',
    )
    store.mark_attempt_started(
        _admission(
            request,
            validation_call_id=claim.validation_call_id,
            lease_token=claim.lease_token or '',
        ),
        commit=lambda: None,
    )
    now[0] += timedelta(seconds=121)

    assert store.claim_or_replay(request).disposition == 'busy'
    store.recover_expired(context)
    replay = store.claim_or_replay(request)
    snapshot = store.snapshot(context)

    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'validator_unavailable'
    assert snapshot.provider_attempt_count == 1
    assert snapshot.charged_cost_usd == request.reserved_cost_usd


def test_cancel_before_attempt_is_zero_charge_and_releases_reserve() -> None:
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: datetime(2026, 8, 30, 14, 30, tzinfo=UTC),
        lease_token_factory=lambda: '5' * 64,
    )
    request = _claim_request()
    claim = store.claim_or_replay(request)
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token or '',
    )

    store.cancel(context)
    replay = store.claim_or_replay(request)
    snapshot = store.snapshot(context)

    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'post_validation_drift'
    assert snapshot.provider_attempt_count == 0
    assert snapshot.charged_cost_usd == Decimal('0')


def test_cancel_after_marker_discards_result_and_charges_known_usage_once() -> None:
    now = datetime(2026, 8, 30, 15, 0, tzinfo=UTC)
    store = ValidationCallLedger(
        settings=_settings(),
        db_clock=lambda: now,
        lease_token_factory=lambda: '6' * 64,
    )
    request = _claim_request()
    claim = store.claim_or_replay(request)
    context = ValidationLockedContext(
        validation_call_id=claim.validation_call_id,
        workflow_thread_id=request.workflow_thread_id,
        batch_fingerprint=request.batch_fingerprint,
        lease_token=claim.lease_token or '',
    )
    grant = store.mark_attempt_started(
        _admission(
            request,
            validation_call_id=claim.validation_call_id,
            lease_token=claim.lease_token or '',
        ),
        commit=lambda: None,
    )
    store.cancel(context)

    class Invocation:
        prepared_content_hmac = request.prepared_content_hmac
        character_count = request.serialized_character_count
        framed_input_tokens = request.framed_input_tokens
        max_output_tokens = 3072
        canonical_bytes = b'ephemeral'

    store.dispatch_prepared_validation(
        invocation=Invocation(),
        provider=lambda *args, **kwargs: 'late-result',
        grant=grant,
    )
    candidate = request.candidates[0]
    completed = store.complete(
        context,
        ValidationBatchCompletion(
            validation_call_id=context.validation_call_id,
            workflow_thread_id=context.workflow_thread_id,
            batch_fingerprint=context.batch_fingerprint,
            lease_token=context.lease_token,
            candidates=(
                ValidationCandidateCompletion(
                    review_item_id=candidate.review_item_id,
                    validation_key=candidate.validation_key,
                    result=_validation_result('C01'),
                    policy_decision='auto_approve',
                    policy_reason_codes=('eligible',),
                ),
            ),
            input_tokens=120,
            output_tokens=40,
        ),
    )
    replay = store.claim_or_replay(request)
    snapshot = store.snapshot(context)

    assert completed == ()
    assert replay.terminal_status == 'failed'
    assert replay.failure_reason_code == 'post_validation_drift'
    assert snapshot.charged_cost_usd == Decimal('0.000720')
