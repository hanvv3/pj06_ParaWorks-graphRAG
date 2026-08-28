import tomllib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError


def _validation_result(score: Decimal = Decimal('0.9800')) -> dict[str, object]:
    return {
        'candidate_slot_id': 'C01',
        'claim_results': [
            {
                'field_key': 'title',
                'verdict': 'supported',
                'claim_scope': 'direct_fact',
                'entailment_score': score,
                'evidence_slot_ids': ['E01'],
            },
            {
                'field_key': 'result_summary',
                'verdict': 'supported',
                'claim_scope': 'direct_fact',
                'entailment_score': score,
                'evidence_slot_ids': ['E02'],
            },
        ],
        'uncertainty_codes': [],
        'conflict_codes': [],
    }


def test_auto_review_defaults_are_disabled_and_fixed_to_terra_medium() -> None:
    from backend.app.core.config import Settings
    from backend.app.schemas.auto_review import (
        AUTO_REVIEW_REASONING_EFFORT,
        AUTO_REVIEW_VALIDATOR_MODEL,
        AUTO_REVIEW_VALIDATOR_PROVIDER,
    )

    settings = Settings(_env_file=None)

    assert settings.auto_review_mode == 'disabled'
    assert settings.auto_review_enforce_percentage == 0
    assert (AUTO_REVIEW_VALIDATOR_PROVIDER, AUTO_REVIEW_VALIDATOR_MODEL) == (
        'openai',
        'gpt-5.6-terra',
    )
    assert AUTO_REVIEW_REASONING_EFFORT == 'medium'


def test_v21_validation_output_forbids_extra_fields_and_out_of_range_slots() -> None:
    from backend.app.schemas.auto_review import CandidateValidationBatchResult

    good = {'results': [_validation_result()]}
    assert CandidateValidationBatchResult.model_validate(good).results[0].candidate_slot_id == 'C01'
    with pytest.raises(ValidationError):
        CandidateValidationBatchResult.model_validate(good | {'rationale': 'not allowed'})
    with pytest.raises(ValidationError):
        CandidateValidationBatchResult.model_validate(
            {'results': [_validation_result() | {'candidate_slot_id': 'C05'}]}
        )


def test_v21_dry_run_exposes_exact_validator_output_contract_and_cost_policy_identity() -> None:
    from backend.app.schemas.review_workflow import ReviewWorkflowDryRunResponseV21

    response = ReviewWorkflowDryRunResponseV21(
        workflow_name='company-memory-review',
        graph_version='company-memory-review-v2.1-auto-review',
        source_count=1,
        agent_names=['timeline_agent'],
        selection_policy_version='company-memory-review-selection:v1',
        estimated_input_tokens=10_000,
        estimated_output_tokens=2_048,
        estimated_cost_usd=0.016716,
        budget_limit_usd=0.20,
        budget_status='within_budget',
        cache_hit=False,
        requires_explicit_run=True,
        auto_review_mode='shadow',
        auto_review_policy_version='auto-review-policy:v1',
        auto_review_validator_provider='openai',
        auto_review_validator_model='gpt-5.6-terra',
        auto_review_reasoning_effort='medium',
        auto_review_validator_prompt_version='auto-review-validation:v1',
        auto_review_validator_output_contract_version='candidate-validation-batch:v1',
        auto_review_cost_policy_version='auto-review-cost:v1',
        auto_review_enforce_percentage=0,
        auto_review_estimated_input_tokens=6_000,
        auto_review_estimated_output_tokens=3_072,
        auto_review_estimated_cost_usd=0.048864,
        total_estimated_input_tokens=16_000,
        total_estimated_output_tokens=5_120,
        total_estimated_cost_usd=0.065580,
        launch_confirmation_token='x' * 32,
    )

    assert set(response.model_dump()) == {
        'workflow_name', 'graph_version', 'source_count', 'agent_names',
        'selection_policy_version', 'estimated_input_tokens',
        'estimated_output_tokens', 'estimated_cost_usd', 'budget_limit_usd',
        'budget_status', 'cache_hit', 'requires_explicit_run', 'auto_review_mode',
        'auto_review_policy_version', 'auto_review_validator_provider',
        'auto_review_validator_model', 'auto_review_reasoning_effort',
        'auto_review_validator_prompt_version',
        'auto_review_validator_output_contract_version',
        'auto_review_cost_policy_version', 'auto_review_enforce_percentage',
        'auto_review_estimated_input_tokens', 'auto_review_estimated_output_tokens',
        'auto_review_estimated_cost_usd', 'total_estimated_input_tokens',
        'total_estimated_output_tokens', 'total_estimated_cost_usd',
        'launch_confirmation_token',
    }


def test_exact_09800_pass_boundary_is_preserved_as_decimal() -> None:
    from backend.app.schemas.auto_review import CandidateValidationBatchResult

    exact = CandidateValidationBatchResult.model_validate({'results': [_validation_result()]})
    below = CandidateValidationBatchResult.model_validate(
        {'results': [_validation_result(Decimal('0.9799'))]}
    )

    assert exact.results[0].claim_results[0].entailment_score == Decimal('0.9800')
    assert below.results[0].claim_results[0].entailment_score < Decimal('0.9800')


def test_v21_run_requires_launch_token_but_v20_forbids_it() -> None:
    from backend.app.schemas.review_workflow import (
        ReviewWorkflowRunRequest,
        ReviewWorkflowRunRequestV21,
    )

    payload = {
        'source_refs': [{'source_type': 'gmail', 'source_id': 'g-1', 'version_or_signature': 'v1'}],
        'agent_names': ['timeline_agent'],
    }
    with pytest.raises(ValidationError):
        ReviewWorkflowRunRequestV21.model_validate(payload)
    with pytest.raises(ValidationError):
        ReviewWorkflowRunRequest.model_validate(payload | {'launch_confirmation_token': 'x' * 32})


def test_public_audit_projection_has_no_internal_identity_or_reason() -> None:
    from backend.app.schemas.auto_review import AutoReviewAuditPublicSummary

    assert set(
        AutoReviewAuditPublicSummary(
            status='completed', outcome='confirmed', action_required=False
        ).model_dump()
    ) == {'status', 'outcome', 'action_required'}
    with pytest.raises(ValidationError):
        AutoReviewAuditPublicSummary.model_validate(
            {'status': 'completed', 'outcome': 'confirmed', 'action_required': False, 'reason': 'hidden'}
        )


def test_auto_review_policy_reason_literals_include_sensitive_only_for_internal_policy() -> None:
    from backend.app.schemas.auto_review import AutoReviewPolicyReasonCode

    assert 'sensitive_input_detected' in AutoReviewPolicyReasonCode.__args__


@pytest.mark.parametrize(
    'values',
    [
        {'auto_review_mode': 'shadow', 'auto_review_enforce_percentage': 10},
        {'auto_review_mode': 'enforce', 'auto_review_enforce_percentage': 0},
        {'auto_review_mode': 'enforce', 'auto_review_enforce_percentage': 50},
        {'auto_review_max_total_cost_usd': Decimal('0.21')},
        {'auto_review_provider_attempt_lease_seconds': 95},
    ],
)
def test_invalid_mode_percentage_price_and_cost_caps_fail_settings_validation(values: dict[str, object]) -> None:
    from backend.app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_tiktoken_is_a_direct_bounded_dependency_and_o200k_encoding_loads() -> None:
    import tiktoken

    pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
    assert 'tiktoken>=0.12.0,<0.13.0' in pyproject['project']['dependencies']
    assert tiktoken.get_encoding('o200k_base').encode('한국어 ASCII')


def test_langsmith_is_direct_bounded_dependency_for_no_trace_context() -> None:
    pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
    assert 'langsmith>=0.8.0,<0.9.0' in pyproject['project']['dependencies']


def test_shared_provider_lease_exceeds_send_window_plus_timeout_plus_commit_grace() -> None:
    from backend.app.core.config import Settings

    settings = Settings(_env_file=None)
    assert settings.auto_review_provider_attempt_lease_seconds > (
        settings.auto_review_provider_send_start_window_seconds
        + settings.auto_review_provider_timeout_seconds
        + settings.auto_review_provider_commit_grace_seconds
    )


def test_non_disabled_mode_rejects_local_default_fingerprint_secret() -> None:
    from backend.app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            auto_review_mode='shadow',
            auto_review_enforce_percentage=0,
        )


@pytest.mark.parametrize(
    ('secret', 'key_version'), [('x', 'v1'), ('x' * 32, '')]
)
def test_non_disabled_readiness_rejects_short_secret_or_empty_key_version(
    secret: str, key_version: str
) -> None:
    settings = _live_settings(
        agent_runtime_fingerprint_secret=secret,
        agent_runtime_fingerprint_key_version=key_version,
    )

    with pytest.raises(ValueError, match='durable C.5 key'):
        settings.require_auto_review_live_readiness()


def test_non_disabled_readiness_rejects_mismatched_confirmation_prices() -> None:
    from backend.app.core.config import Settings

    settings = Settings(
        _env_file=None,
        auto_review_mode='shadow',
        auto_review_enforce_percentage=0,
        agent_runtime_fingerprint_secret='x' * 32,
        openai_api_key='test-key',
        auto_review_extraction_input_cost_per_1m_tokens=Decimal('0.74'),
        auto_review_extraction_output_cost_per_1m_tokens=Decimal('4.500000'),
        auto_review_validator_input_cost_per_1m_tokens=Decimal('2.000000'),
        auto_review_validator_output_cost_per_1m_tokens=Decimal('12.000000'),
    )

    with pytest.raises(ValueError, match='confirmation prices'):
        settings.require_auto_review_live_readiness()


def test_v21_extraction_prices_are_decimal_distinct_from_legacy_float_estimates() -> None:
    from backend.app.core.config import Settings

    settings = Settings(
        _env_file=None,
        auto_review_extraction_input_cost_per_1m_tokens=Decimal('0.750000'),
        auto_review_extraction_output_cost_per_1m_tokens=Decimal('4.500000'),
    )
    assert isinstance(settings.auto_review_extraction_input_cost_per_1m_tokens, Decimal)
    assert isinstance(settings.auto_review_extraction_output_cost_per_1m_tokens, Decimal)
    assert isinstance(settings.agent_llm_input_cost_per_1m_tokens, float)


def test_v21_budget_status_uses_exact_total_extraction_plus_validation_cost() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        AUTO_REVIEW_MAX_EXTRACTION_COST_USD,
        AUTO_REVIEW_MAX_PROFILE_COST_USD,
        AUTO_REVIEW_MAX_VALIDATION_COST_USD,
    )

    assert AUTO_REVIEW_MAX_EXTRACTION_COST_USD + AUTO_REVIEW_MAX_VALIDATION_COST_USD == AUTO_REVIEW_MAX_PROFILE_COST_USD


def test_v21_total_budget_is_positive_and_never_exceeds_twenty_cents() -> None:
    from backend.app.core.config import Settings

    budget = Settings(_env_file=None).auto_review_max_total_cost_usd
    assert Decimal('0') < budget <= Decimal('0.20')


def test_extraction_strict_envelope_rejects_second_candidate_extra_fields_and_wrong_item_type() -> None:
    from backend.app.schemas.auto_review import TimelineExtractionResult

    candidate = {
        'item_type': 'timeline_event', 'title': '제목', 'summary': '요약', 'result_summary': '결과',
        'confidence_score': '0.9800', 'uncertainty_reason': '없음',
        'field_evidence_bindings': [
            {'field_key': 'title', 'evidence_slot_id': 'S01'},
            {'field_key': 'result_summary', 'evidence_slot_id': 'S02'},
        ],
    }
    with pytest.raises(ValidationError):
        TimelineExtractionResult.model_validate({'result_kind': 'candidate', 'candidate': candidate | {'item_type': 'history_event'}})
    with pytest.raises(ValidationError):
        TimelineExtractionResult.model_validate({'result_kind': 'candidate', 'candidate': candidate, 'no_candidate_reason': 'extra'})
    with pytest.raises(ValidationError):
        TimelineExtractionResult.model_validate({'result_kind': 'candidate', 'candidate': candidate, 'second_candidate': candidate})


def test_v21_rejects_retry_fallback_or_missing_extraction_estimator_route() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import get_extraction_route

    assert get_extraction_route(
        'timeline_agent',
        extraction_cost_policy_version='auto-review-extraction-cost:v1',
        provider='openai', model='gpt-5.4-mini-2026-03-17', reasoning_effort='none',
        route_version='auto-review-extraction-route:v1',
        prompt_version='timeline-extraction:c5-v1',
        output_contract_version='timeline-candidate:c5-v1',
    ).max_provider_attempts == 1


def test_v20_models_keep_exact_json_field_sets() -> None:
    from backend.app.schemas.review_workflow import (
        ReviewWorkflowDryRunResponse,
        ReviewWorkflowStatusResponse,
    )

    v20_dry_run = ReviewWorkflowDryRunResponse(
        workflow_name='company-memory-review', graph_version='company-memory-review-v2.0',
        source_count=1, agent_names=['timeline_agent'],
        selection_policy_version='company-memory-review-selection:v1',
        estimated_input_tokens=1, estimated_output_tokens=1, estimated_cost_usd=0.0,
        budget_limit_usd=None, budget_status='within_budget', cache_hit=False,
        requires_explicit_run=True,
    )
    v20_status = ReviewWorkflowStatusResponse(
        thread_id='t', status='completed', review_item_count=0,
        review_status_counts={'pending_review': 0}, durable=False,
        graph_version='company-memory-review-v2.0', review_resolution_ready=True,
        checkpoint_resumable=False, resume_allowed=False, retry_allowed=False,
        created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
        error_code=None, resume_error_code=None,
    )
    assert set(v20_dry_run.model_dump()) == {
        'workflow_name', 'graph_version', 'source_count', 'agent_names',
        'selection_policy_version', 'estimated_input_tokens',
        'estimated_output_tokens', 'estimated_cost_usd', 'budget_limit_usd',
        'budget_status', 'cache_hit', 'requires_explicit_run',
    }
    assert set(v20_status.model_dump()) == {
        'thread_id', 'status', 'review_item_count', 'review_status_counts',
        'durable', 'graph_version', 'review_resolution_ready',
        'checkpoint_resumable', 'resume_allowed', 'retry_allowed', 'created_at',
        'updated_at', 'error_code', 'resume_error_code',
    }


def _timeline_candidate(*, duplicate_slot: bool = False) -> dict[str, object]:
    return {
        'item_type': 'timeline_event',
        'title': '한국어 제목',
        'summary': 'ASCII summary',
        'result_summary': '한국어 결과',
        'confidence_score': '0.9800',
        'field_evidence_bindings': [
            {'field_key': 'title', 'evidence_slot_id': 'S01'},
            {'field_key': 'summary', 'evidence_slot_id': 'S01' if duplicate_slot else 'S02'},
            {'field_key': 'result_summary', 'evidence_slot_id': 'S03'},
        ],
    }


def test_extraction_no_candidate_reason_and_evidence_slots_are_exact_and_unique() -> None:
    from backend.app.schemas.auto_review import TimelineExtractionResult

    assert TimelineExtractionResult.model_validate(
        {
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        }
    ).no_candidate_reason == 'no_relevant_evidence'
    with pytest.raises(ValidationError):
        TimelineExtractionResult.model_validate(
            {
                'result_kind': 'no_candidate',
                'candidate': None,
                'no_candidate_reason': 'unbounded reason',
            }
        )
    with pytest.raises(ValidationError):
        TimelineExtractionResult.model_validate(
            {'result_kind': 'candidate', 'candidate': _timeline_candidate(duplicate_slot=True)}
        )


def test_mail_document_extraction_is_explicitly_discriminated_and_uncertainty_is_optional() -> None:
    from backend.app.schemas.auto_review import MailDocumentExtractionResult

    payload = _timeline_candidate() | {'item_type': 'history_event', 'reason': '직접 근거'}
    payload.pop('result_summary')
    payload['field_evidence_bindings'][-1] = {
        'field_key': 'reason', 'evidence_slot_id': 'S03'
    }
    result = MailDocumentExtractionResult.model_validate(
        {'result_kind': 'candidate', 'candidate': payload}
    )

    assert type(result.candidate).__name__ == 'HistoryCandidate'
    assert result.candidate.uncertainty_reason is None


def _live_settings(**overrides: object):
    from backend.app.core.config import Settings

    defaults: dict[str, object] = {
        'auto_review_mode': 'shadow',
        'auto_review_enforce_percentage': 0,
        'agent_runtime_fingerprint_secret': 'x' * 32,
        'agent_runtime_fingerprint_key_version': 'v1',
        'openai_api_key': 'test-key',
        'auto_review_extraction_input_cost_per_1m_tokens': Decimal('0.750000'),
        'auto_review_extraction_output_cost_per_1m_tokens': Decimal('4.500000'),
        'auto_review_validator_input_cost_per_1m_tokens': Decimal('2.000000'),
        'auto_review_validator_output_cost_per_1m_tokens': Decimal('12.000000'),
    }
    return Settings(_env_file=None, **(defaults | overrides))


def test_non_disabled_readiness_rejects_langchain_global_debug_or_verbose_model() -> None:
    from langchain_core.globals import set_debug

    settings = _live_settings()
    try:
        set_debug(True)
        with pytest.raises(ValueError, match='debug'):
            settings.require_auto_review_live_readiness()
    finally:
        set_debug(False)
    with pytest.raises(ValueError, match='verbose'):
        settings.require_auto_review_live_readiness(
            model=type('VerboseModel', (), {'verbose': True})()
        )


def test_non_disabled_readiness_rejects_openai_log_debug_effective_debug_logger_or_unapproved_http_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging

    settings = _live_settings()
    monkeypatch.setenv('OPENAI_LOG', 'DeBuG')
    with pytest.raises(ValueError, match='debug'):
        settings.require_auto_review_live_readiness()
    monkeypatch.delenv('OPENAI_LOG')
    logger = logging.getLogger('openai')
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(ValueError, match='logger'):
            settings.require_auto_review_live_readiness()
    finally:
        logger.setLevel(old_level)
    with pytest.raises(ValueError, match='HTTP hook'):
        settings.require_auto_review_live_readiness(http_hook=object())


@pytest.mark.parametrize(
    ('secret', 'key_version'),
    [
        ('x' * 31, 'v1'),
        ('x' * 32, ''),
        ('local-development-agent-runtime-fingerprint-secret', 'v1'),
    ],
)
def test_disabled_postgres_c5_bootstrap_rejects_placeholder_or_short_secret(
    secret: str, key_version: str
) -> None:
    from backend.app.core.config import Settings

    settings = Settings(
        _env_file=None,
        auto_review_mode='disabled',
        agent_runtime_fingerprint_secret=secret,
        agent_runtime_fingerprint_key_version=key_version,
    )
    with pytest.raises(ValueError, match='durable C.5 key'):
        settings.require_c5_durable_key_ready()


def test_disabled_sqlite_smoke_may_use_process_local_placeholder_without_durable_ready_state() -> None:
    from backend.app.core.config import Settings

    settings = Settings(_env_file=None, auto_review_mode='disabled')
    assert settings.allows_c5_process_local_sqlite_smoke()
    with pytest.raises(ValueError, match='durable C.5 key'):
        settings.require_c5_durable_key_ready()


def test_file_backed_sqlite_placeholder_refuses_every_c5_bound_durable_write() -> None:
    from backend.app.core.config import Settings

    with pytest.raises(ValueError, match='durable C.5 key'):
        Settings(_env_file=None).require_c5_durable_key_ready()


def test_each_extraction_schema_largest_accepted_korean_ascii_fixture_fits_2048_total_output_tokens() -> None:
    from backend.app.schemas.auto_review import (
        DecisionRecordExtractionResult,
        HistoryExtractionResult,
        MailDocumentExtractionResult,
        TimelineExtractionResult,
        TodoExtractionResult,
    )

    candidate = _timeline_candidate()
    history = _timeline_candidate() | {'item_type': 'history_event', 'reason': '근거'}
    history.pop('result_summary')
    history['field_evidence_bindings'][-1] = {'field_key': 'reason', 'evidence_slot_id': 'S03'}
    decision = _timeline_candidate() | {'item_type': 'decision_record', 'decision_summary': '결정'}
    decision.pop('result_summary')
    decision['field_evidence_bindings'][-1] = {'field_key': 'decision_summary', 'evidence_slot_id': 'S03'}
    todo = _timeline_candidate() | {'item_type': 'todo', 'priority': 'high', 'priority_reason': '긴급'}
    todo.pop('result_summary')
    todo['field_evidence_bindings'][-1] = {'field_key': 'priority', 'evidence_slot_id': 'S03'}
    todo['field_evidence_bindings'].append({'field_key': 'priority_reason', 'evidence_slot_id': 'S04'})
    for model, value in (
        (TimelineExtractionResult, candidate),
        (HistoryExtractionResult, history),
        (DecisionRecordExtractionResult, decision),
        (TodoExtractionResult, todo),
        (MailDocumentExtractionResult, candidate),
    ):
        parsed = model.model_validate({'result_kind': 'candidate', 'candidate': value})
        assert parsed.canonical_token_count() <= 2_048
    boundary = _timeline_candidate()
    boundary['title'] = '가' * 160
    boundary['summary'] = '가' * 800
    boundary['result_summary'] = '가' * 800
    boundary['uncertainty_reason'] = '가' * 195
    assert TimelineExtractionResult.model_validate(
        {'result_kind': 'candidate', 'candidate': boundary}
    ).canonical_token_count() == 2_048


def test_each_extraction_schema_rejects_individually_valid_fields_when_canonical_envelope_exceeds_2048_tokens() -> None:
    import json

    import tiktoken

    from backend.app.schemas.auto_review import TimelineExtractionResult

    candidate = _timeline_candidate()
    candidate['title'] = '가' * 160
    candidate['summary'] = '가' * 800
    candidate['result_summary'] = '가' * 800
    candidate['uncertainty_reason'] = '가' * 196
    canonical = json.dumps(
        {
            'result_kind': 'candidate',
            'candidate': candidate,
            'no_candidate_reason': None,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    assert len(tiktoken.get_encoding('o200k_base').encode(canonical)) == 2_049
    with pytest.raises(ValidationError, match='2048'):
        TimelineExtractionResult.model_validate(
            {'result_kind': 'candidate', 'candidate': candidate}
        )
