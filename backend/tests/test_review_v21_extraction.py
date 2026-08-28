import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.agent_runtime.contracts import (
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
)
from backend.app.agent_runtime.provider_send_fence import ProviderSendFenceError
from backend.app.agent_runtime.review_v21_extraction import (
    ExtractionCallLedger,
    ExtractionCallStateError,
    ExtractionProviderSafetySnapshot,
    ProviderUsage,
    build_prepared_extraction_plan_set,
    invoke_prepared_extraction,
    parse_extraction_result,
)
from backend.app.core.config import Settings
from backend.app.schemas.auto_review import HistoryExtractionResult


def _settings() -> Settings:
    return Settings(
        agent_runtime_fingerprint_secret='extraction-test-secret-at-least-32-bytes',
        agent_runtime_fingerprint_key_version='extract-v1',
    )


def _message(text='직접 확인된 한국어 근거', index=1) -> EvidenceMessage:
    return EvidenceMessage(
        source_id='gmail:raw-id',
        source_url='https://secret.test/message',
        text=text,
        author='person@example.test',
        timestamp='2026-08-28T00:00:00Z',
        permission_level='internal',
        source_snippet_override=text,
        metadata={
            'stable_message_identity': f'message-{index}',
            'workflow_evidence_ref_id': index,
        },
    )


def _packet(text='직접 확인된 한국어 근거') -> EvidencePacket:
    return EvidencePacket(
        'company_memory',
        'window',
        [_message(text, index) for index in range(1, 4)],
        PermissionContext('owner', 'employee'),
    )


def _safety(
    *, version=1, breaker=False, purpose='extraction'
) -> ExtractionProviderSafetySnapshot:
    return ExtractionProviderSafetySnapshot(
        purpose=purpose,
        provider='openai',
        model='gpt-5.4-mini-2026-03-17',
        reasoning_effort='none',
        state_version=version,
        cost_policy_version='auto-review-extraction-cost:v1',
        token_estimator_version='openai-o200k-extraction:v1',
        tokenizer_encoding='o200k_base',
        reply_priming_tokens=16,
        framing_safety_tokens=512,
        input_usd_per_1m=Decimal('0.750000'),
        output_usd_per_1m=Decimal('4.500000'),
        breaker_open=breaker,
    )


def _plans(agent_names=('history_agent',), packet=None, safety=None):
    return build_prepared_extraction_plan_set(
        packet=packet or _packet(),
        selected_agent_names=agent_names,
        settings=_settings(),
        fingerprint_key_material_verifier='a' * 64,
        safety_snapshots=safety or (_safety(),),
    )


def _candidate():
    return {
        'result_kind': 'candidate',
        'candidate': {
            'item_type': 'history_event',
            'title': '제목',
            'summary': '요약',
            'reason': '직접 근거',
            'confidence_score': '0.9800',
            'uncertainty_reason': None,
            'field_evidence_bindings': [
                {'field_key': 'title', 'evidence_slot_id': 'S01'},
                {'field_key': 'summary', 'evidence_slot_id': 'S02'},
                {'field_key': 'reason', 'evidence_slot_id': 'S03'},
            ],
        },
        'no_candidate_reason': None,
    }


def test_v21_extraction_plan_binds_every_agent_route_prompt_output_contract_and_price():
    plans = _plans(
        (
            'mail_document_agent',
            'timeline_agent',
            'history_agent',
            'decision_record_agent',
            'todo_agent',
        )
    )
    assert len(plans.plans) == 5
    assert {
        (p.provider, p.model, p.reasoning_effort, p.route_version) for p in plans.plans
    } == {
        ('openai', 'gpt-5.4-mini-2026-03-17', 'none', 'auto-review-extraction-route:v1')
    }
    assert {p.reserved_cost_usd for p in plans.plans} == {Decimal('0.016716')}


def test_extraction_safety_set_rejects_duplicate_or_unselected_snapshots():
    safety = _safety()
    with pytest.raises(ExtractionCallStateError, match='duplicate'):
        _plans(safety=(safety, safety))
    with pytest.raises(ExtractionCallStateError, match='unselected'):
        _plans(
            safety=(
                safety,
                replace(safety, model='unselected-model'),
            )
        )


def test_v21_extraction_uses_exact_o200k_rendered_count_for_korean_not_len_div_four():
    invocation = _plans().plans[0].invocation
    assert invocation.framed_input_tokens != len(invocation.canonical_text) // 4
    assert invocation.framed_input_tokens == invocation.encoded_input_tokens + 16 + 512


@pytest.mark.parametrize(
    'text', ('가' * 24001, 'x' * 50000), ids=('korean-character-cap', 'ascii-token-cap')
)
def test_v21_extraction_enforces_24000_char_10000_input_2048_output_and_one_candidate_hard_caps(
    text,
):
    with pytest.raises(ExtractionCallStateError, match='input'):
        _plans(packet=_packet(text))
    with pytest.raises(ValidationError):
        HistoryExtractionResult.model_validate(
            {**_candidate(), 'candidate': [_candidate()['candidate']]}
        )


def test_extraction_canonical_output_budget_accepts_exactly_2048_and_rejects_2049_tokens():
    accepted = parse_extraction_result(HistoryExtractionResult, _candidate())
    assert accepted.canonical_token_count() <= 2048
    with pytest.raises(ExtractionCallStateError, match='output'):
        parse_extraction_result(
            HistoryExtractionResult, _candidate(), canonical_token_count_override=2049
        )


def test_extraction_confidence_signed_zero_canonicalizes_to_positive_fixed_zero():
    payload = _candidate()
    payload['candidate']['confidence_score'] = '-0'  # type: ignore[index]
    result = parse_extraction_result(HistoryExtractionResult, payload)
    assert '"confidence_score":"0.0000"' in result.canonical_json()


def test_v21_extraction_individually_valid_over_2048_canonical_envelope_fails_without_review_item_or_no_candidate_marker():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    grant = ledger.mark_attempt_started(context)
    grant.permit.consume_at_dispatch()
    with pytest.raises(ExtractionCallStateError):
        ledger.complete(
            context,
            result=_candidate(),
            usage=ProviderUsage(10, 2049),
            canonical_output_tokens=2049,
        )
    assert ledger.snapshot(context).result_kind is None


def test_v21_extraction_truncated_partial_json_is_failed_never_repaired_or_cached():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    with pytest.raises(ExtractionCallStateError):
        ledger.complete(
            context,
            result='{"result_kind":',
            usage=ProviderUsage(10, 10),
            native_truncated=True,
        )
    assert ledger.snapshot(context).status == 'failed'


def test_v21_extraction_route_has_zero_sdk_retries_no_provider_fallback_and_cache_trace_off():
    dto = _plans().plans[0].invocation.provider_options
    assert dto == {
        'transport': 'responses',
        'cache': False,
        'max_retries': 0,
        'callbacks': (),
        'tracing': False,
        'verbose': False,
    }
    with pytest.raises(TypeError):
        dto['max_retries'] = 1


def test_v21_extraction_prepares_once_and_invokes_same_object_without_open_db_session():
    plan = _plans().plans[0]
    invocation = plan.invocation
    seen = []
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', plan)
    grant = ledger.mark_attempt_started(context)
    invoke_prepared_extraction(
        invocation,
        lambda same, *, timeout: (
            seen.append(same)
            or {
                'result_kind': 'no_candidate',
                'candidate': None,
                'no_candidate_reason': 'no_relevant_evidence',
            }
        ),
        store=ledger,
        grant=grant,
    )
    assert seen == [invocation]


def test_v21_extraction_rejects_missing_grant_before_provider_invocation():
    called = False

    def provider(invocation, *, timeout):
        nonlocal called
        called = True
        return invocation, timeout

    with pytest.raises(ExtractionCallStateError, match='grant'):
        invoke_prepared_extraction(
            _plans().plans[0].invocation,
            provider,
            store=ExtractionCallLedger(),
            grant=None,
        )
    assert called is False


def test_v21_extraction_rejects_forged_wrapper_around_committed_grant():
    plan = _plans().plans[0]
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', plan)
    committed = ledger.mark_attempt_started(context)

    class _ForgedGrant:
        attempt_id = committed.attempt_id
        permit = committed.permit
        provider_timeout_seconds = committed.provider_timeout_seconds
        authoritative_lease_expires_at = committed.authoritative_lease_expires_at
        transport = committed.transport

    with pytest.raises(ExtractionCallStateError, match='committed'):
        invoke_prepared_extraction(
            plan.invocation,
            lambda *_args, **_kwargs: pytest.fail('provider called'),
            store=ledger,
            grant=_ForgedGrant(),
        )


def test_v21_extraction_provider_dto_contains_only_local_aliases_and_allowlisted_plaintext():
    text = _plans().plans[0].invocation.canonical_text
    assert (
        'S01' in text
        and 'gmail:raw-id' not in text
        and 'https://secret.test' not in text
        and '@example.test' not in text
    )


def test_v21_extraction_credential_match_is_zero_call_zero_cache_zero_trace():
    with pytest.raises(ExtractionCallStateError, match='sensitive'):
        _plans(
            packet=_packet('OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456')
        )


@pytest.mark.parametrize('logger_name', ('paraworks', 'openai'))
def test_v21_extraction_global_debug_or_verbose_is_zero_call_and_writes_no_prompt_bytes(
    monkeypatch, caplog, logger_name
):
    monkeypatch.setenv('OPENAI_LOG', 'debug')
    with pytest.raises(ExtractionCallStateError):
        invoke_prepared_extraction(
            _plans().plans[0].invocation,
            lambda *_: pytest.fail('called'),
            store=ExtractionCallLedger(),
            grant=None,
        )
    assert '직접 확인된' not in caplog.text


def test_v21_extraction_openai_sdk_debug_logging_is_zero_call_and_caplog_receives_no_prompt_bytes(
    caplog,
):
    logging.getLogger('openai').setLevel(logging.DEBUG)
    try:
        with pytest.raises(ExtractionCallStateError):
            invoke_prepared_extraction(
                _plans().plans[0].invocation,
                lambda *_: pytest.fail('called'),
                store=ExtractionCallLedger(),
                grant=None,
            )
        assert '직접 확인된' not in caplog.text
    finally:
        logging.getLogger('openai').setLevel(logging.WARNING)


def test_v21_extraction_plan_permutation_is_stable_and_any_route_or_prompt_drift_changes_hash():
    a = _plans(('history_agent', 'timeline_agent'))
    b = _plans(('timeline_agent', 'history_agent'))
    assert a.plan_set_hmac == b.plan_set_hmac
    assert replace(a.plans[0], prompt_version='changed').identity_hmac(
        _settings()
    ) != a.plans[0].identity_hmac(_settings())


def test_extraction_call_reserves_marks_attempt_and_charges_once_outside_transaction():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    assert ledger.snapshot(context).reserved_cost_usd == Decimal('0.016716')
    ledger.mark_attempt_started(context)
    ledger.complete(
        context,
        result={
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        },
        usage=ProviderUsage(10, 10),
    )
    first = ledger.snapshot(context).charged_cost_usd
    with pytest.raises(ExtractionCallStateError):
        ledger.complete(context, result=_candidate(), usage=ProviderUsage(10, 10))
    assert ledger.snapshot(context).charged_cost_usd == first


def test_extraction_plan_and_call_bind_all_shared_provider_timing_values():
    plan = _plans().plans[0]
    ledger = ExtractionCallLedger()
    state = ledger.snapshot(ledger.claim_or_replay('workflow', plan))
    assert state.timing_tuple == (60, 5, 120, 30) == plan.timing_tuple


def test_database_clock_owns_claim_attempt_and_lease_timestamps_despite_host_clock_skew():
    db_now = datetime(2026, 8, 28, tzinfo=UTC)
    ledger = ExtractionCallLedger(db_clock=lambda: db_now)
    context = ledger.claim_or_replay(
        'workflow', _plans().plans[0], expected_now=datetime(1999, 1, 1, tzinfo=UTC)
    )
    ledger.mark_attempt_started(context)
    assert ledger.snapshot(context).attempt_started_at == db_now
    assert ledger.snapshot(context).lease_expires_at == db_now + timedelta(seconds=120)


def test_cancel_between_extraction_marker_and_send_latches_without_premature_terminalization():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    grant = ledger.mark_attempt_started(context)
    ledger.cancel(context)
    assert ledger.snapshot(context).status == 'claimed'
    grant.permit.consume_at_dispatch()


def test_terminal_extraction_call_can_never_send_after_lease_recovery():
    now = [datetime(2026, 8, 28, tzinfo=UTC)]
    ledger = ExtractionCallLedger(db_clock=lambda: now[0])
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    grant = ledger.mark_attempt_started(context)
    now[0] += timedelta(seconds=120)
    ledger.recover_expired(context)
    with pytest.raises(ProviderSendFenceError):
        grant.permit.consume_at_dispatch()


def test_extraction_crash_after_send_never_retries_and_charges_reserved_ceiling():
    now = [datetime(2026, 8, 28, tzinfo=UTC)]
    ledger = ExtractionCallLedger(db_clock=lambda: now[0])
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    now[0] += timedelta(seconds=120)
    ledger.recover_expired(context)
    assert ledger.snapshot(context).charged_cost_usd == Decimal('0.016716')
    with pytest.raises(ExtractionCallStateError):
        ledger.mark_attempt_started(context)


def test_extraction_known_overrun_records_actual_opens_extraction_breaker_and_skips_validation():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    with pytest.raises(ExtractionCallStateError, match='overrun'):
        ledger.complete(context, result=_candidate(), usage=ProviderUsage(10001, 1))
    assert ledger.breaker_open and ledger.snapshot(context).budget_overrun


def test_extraction_permission_or_source_drift_before_attempt_is_zero_call():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.set_drift(context)
    with pytest.raises(ExtractionCallStateError):
        ledger.mark_attempt_started(context)
    assert ledger.snapshot(context).charged_cost_usd == 0


def test_extraction_permission_or_source_drift_during_call_discards_output_and_charges_once():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    ledger.set_drift(context)
    with pytest.raises(ExtractionCallStateError):
        ledger.complete(context, result=_candidate(), usage=ProviderUsage(10, 10))
    assert (
        ledger.snapshot(context).status == 'failed'
        and ledger.snapshot(context).charged_cost_usd > 0
    )


def test_claimed_or_failed_agent_run_is_never_reused_as_complete_cache_hit():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    assert not ledger.snapshot(context).cacheable
    ledger.fail(context, reason_code='provider_failure')
    assert not ledger.snapshot(context).cacheable


def test_one_workflow_agent_cannot_claim_a_second_call_with_a_different_plan_hmac():
    ledger = ExtractionCallLedger()
    ledger.claim_or_replay('workflow', _plans().plans[0])
    with pytest.raises(ExtractionCallStateError):
        ledger.claim_or_replay(
            'workflow', replace(_plans().plans[0], plan_hmac='f' * 64)
        )


def test_cancel_before_extraction_attempt_marker_is_zero_call_zero_charge_terminal():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.cancel(context)
    assert (
        ledger.snapshot(context).status == 'failed'
        and ledger.snapshot(context).charged_cost_usd == 0
    )


def test_cancel_during_extraction_call_charges_once_and_late_completion_creates_no_candidate():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    ledger.cancel(context)
    with pytest.raises(ExtractionCallStateError):
        ledger.complete(context, result=_candidate(), usage=ProviderUsage(10, 10))
    assert ledger.snapshot(context).result_kind is None


def test_global_disabled_before_extraction_claim_or_attempt_marker_is_zero_call_zero_charge():
    ledger = ExtractionCallLedger()
    ledger.disabled = True
    with pytest.raises(ExtractionCallStateError):
        ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.disabled = False
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.disabled = True
    with pytest.raises(ExtractionCallStateError):
        ledger.mark_attempt_started(context)
    assert ledger.snapshot(context).charged_cost_usd == 0


def test_disable_resume_cancel_and_attempt_marker_interleavings_never_start_a_late_call():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.cancel(context)
    ledger.disabled = True
    with pytest.raises(ExtractionCallStateError):
        ledger.mark_attempt_started(context)


def test_validation_breaker_open_before_extraction_claim_is_cross_purpose_zero_call():
    ledger = ExtractionCallLedger()
    ledger.validation_breaker_open = True
    with pytest.raises(ExtractionCallStateError):
        ledger.claim_or_replay('workflow', _plans().plans[0])


def test_extraction_claim_locks_all_required_purpose_rows_in_global_sorted_order():
    ledger = ExtractionCallLedger()
    ledger.claim_or_replay('workflow', _plans().plans[0])
    assert ledger.last_lock_order[:3] == (
        'runtime_key_state',
        'provider_safety:extraction',
        'provider_safety:validation',
    )


def test_extraction_completion_and_exact_candidate_set_or_zero_marker_commit_atomically():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    ledger.complete(
        context,
        result=_candidate(),
        usage=ProviderUsage(10, 10),
        candidate_pair=('candidate-key', 'evidence-hmac'),
    )
    state = ledger.snapshot(context)
    assert (
        state.status == 'completed'
        and state.result_kind == 'candidate'
        and state.result_candidate_count == 1
    )


def test_zero_candidate_completion_persists_explicit_empty_set_hmac_not_only_zero_children():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    ledger.complete(
        context,
        result={
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        },
        usage=ProviderUsage(10, 10),
    )
    state = ledger.snapshot(context)
    assert (
        state.result_kind == 'no_candidate'
        and state.result_candidate_count == 0
        and len(state.result_candidate_set_hmac) == 64
    )


def test_completed_candidate_replay_rejects_missing_extra_or_corrupt_child_set():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    ledger.complete(
        context,
        result=_candidate(),
        usage=ProviderUsage(10, 10),
        candidate_pair=('candidate-key', 'evidence-hmac'),
    )
    with pytest.raises(ExtractionCallStateError):
        ledger.verify_replay(context, candidate_pairs=[])


def test_completed_call_without_terminal_result_marker_is_invalid_not_cacheable():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.force_corrupt_completed(context)
    with pytest.raises(ExtractionCallStateError):
        ledger.verify_replay(context, candidate_pairs=[])


def test_crash_before_atomic_extraction_completion_replays_without_second_provider_call():
    ledger = ExtractionCallLedger()
    context = ledger.claim_or_replay('workflow', _plans().plans[0])
    ledger.mark_attempt_started(context)
    same = ledger.claim_or_replay('workflow', _plans().plans[0])
    assert same == context
    with pytest.raises(ExtractionCallStateError):
        ledger.mark_attempt_started(same)


def test_concurrent_extraction_claims_and_resume_converge_on_one_agent_call():
    ledger = ExtractionCallLedger()
    plan = _plans().plans[0]
    assert ledger.claim_or_replay('workflow', plan) == ledger.claim_or_replay(
        'workflow', plan
    )
    assert ledger.call_count == 1


def test_signed_extraction_plus_validation_obligation_never_admits_above_total_ceiling():
    ledger = ExtractionCallLedger(workflow_budget_limit=Decimal('0.20'))
    with pytest.raises(ExtractionCallStateError, match='budget'):
        ledger.claim_or_replay(
            'workflow', _plans().plans[0], validation_obligation=Decimal('0.19')
        )
