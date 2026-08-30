from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from langsmith import tracing_context

from backend.app.agent_runtime.auto_review_cost_policy import (
    _SERVER_OWNED_FENCED_SEND_HOOK,
    AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
    VALIDATION_COST_POLICY,
)
from backend.app.agent_runtime.auto_review_policy import (
    SUPPORTED_VALIDATION_IDENTITY,
    AutoReviewPolicyEngine,
    CandidateValidationRequest,
    EligibilityPolicyResult,
    ValidationClaimInput,
    ValidationEvidenceSlot,
    ValidationPolicyInput,
)
from backend.app.agent_runtime.auto_review_validator import (
    _parse_provider_result,
    _prepare_many,
    _structured_schema_from_framing,
)
from backend.app.agent_runtime.model_router import (
    ReviewModelUnavailableError,
    build_auto_review_validator_model_route,
)
from backend.app.core.config import Settings
from backend.app.schemas.auto_review import (
    AUTO_REVIEW_POLICY_VERSION,
    AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
    AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
)

SCHEMA_VERSION = 'auto-review-evaluation-report:v1'
FIXTURE_SCHEMA_VERSION = 'auto-review-golden:v1'
_PROHIBITED = (
    'permission_or_version_violation',
    'duplicate_promotion',
    'cross_item_revoke',
    'malformed_output_approval',
    'validation_replay_mismatch',
)


class EvaluationConfigurationError(ValueError):
    pass


class LiveEvaluationError(RuntimeError):
    pass


def _money(input_tokens: int, output_tokens: int) -> Decimal:
    policy = VALIDATION_COST_POLICY
    return (
        (
            Decimal(input_tokens) * policy.input_cost_per_1m_tokens
            + Decimal(output_tokens) * policy.output_cost_per_1m_tokens
        )
        / Decimal(1_000_000)
    ).quantize(Decimal('0.000001'), rounding=ROUND_CEILING)


def _load_fixture(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise EvaluationConfigurationError('fixture_invalid') from None
    if not isinstance(value, dict) or value.get('schema_version') != FIXTURE_SCHEMA_VERSION:
        raise EvaluationConfigurationError('fixture_invalid')
    cases = value.get('cases')
    usage = value.get('usage')
    if not isinstance(cases, list) or not cases or not isinstance(usage, dict):
        raise EvaluationConfigurationError('fixture_invalid')
    required = {'category', 'expected_approval', 'predicted_approval', 'live_input'}
    if any(not isinstance(case, dict) or not required <= set(case) for case in cases):
        raise EvaluationConfigurationError('fixture_invalid')
    if any(
        not isinstance(case['expected_approval'], bool)
        or not isinstance(case['predicted_approval'], bool)
        for case in cases
    ):
        raise EvaluationConfigurationError('fixture_invalid')
    for case in cases:
        live_input = case['live_input']
        if (
            not isinstance(live_input, dict)
            or live_input.get('item_type') not in {'timeline_event', 'history_event'}
            or not isinstance(live_input.get('title'), str)
            or not isinstance(live_input.get('fact'), str)
            or not isinstance(live_input.get('evidence'), list)
            or not live_input['evidence']
            or any(not isinstance(value, str) or not value.strip() for value in live_input['evidence'])
        ):
            raise EvaluationConfigurationError('fixture_invalid')
    return value


def _evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    expected_identities = {
        'prompt_version': AUTO_REVIEW_VALIDATOR_PROMPT_VERSION,
        'output_contract_version': AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION,
        'policy_version': AUTO_REVIEW_POLICY_VERSION,
        'cost_policy_version': VALIDATION_COST_POLICY.cost_policy_version,
    }
    identities = fixture.get('identities')
    identity_match = identities == expected_identities
    cases = fixture['cases']
    predicted_positive = sum(bool(case['predicted_approval']) for case in cases)
    true_positive = sum(
        bool(case['predicted_approval']) and bool(case['expected_approval'])
        for case in cases
    )
    expected_positive = sum(bool(case['expected_approval']) for case in cases)
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    recall = true_positive / expected_positive if expected_positive else 1.0
    metrics: dict[str, int | float] = {
        'auto_approval_precision': round(precision, 6),
        'hard_negative_false_approval_count': sum(
            bool(case.get('hard_negative')) and bool(case['predicted_approval'])
            for case in cases
        ),
        'permission_or_version_violation_count': 0,
        'duplicate_promotion_count': 0,
        'cross_item_revoke_count': 0,
        'malformed_output_approval_count': 0,
        'validation_replay_mismatch_count': 0,
        'queue_reduction_rate': round(predicted_positive / len(cases), 6),
        'recall': round(recall, 6),
    }
    for name in _PROHIBITED:
        metrics[f'{name}_count'] = sum(
            bool(case.get(name)) for case in cases
        )
    usage = fixture['usage']
    try:
        input_tokens = int(usage['input_tokens'])
        output_tokens = int(usage['output_tokens'])
    except (KeyError, TypeError, ValueError):
        raise EvaluationConfigurationError('fixture_invalid') from None
    if input_tokens < 0 or output_tokens < 0:
        raise EvaluationConfigurationError('fixture_invalid')
    gate_passed = (
        identity_match
        and precision >= 0.99
        and all(metrics[f'{name}_count'] == 0 for name in _PROHIBITED)
        and metrics['hard_negative_false_approval_count'] == 0
    )
    return {
        'schema_version': SCHEMA_VERSION,
        'safety_key': {
            'purpose': 'validation',
            'provider': VALIDATION_COST_POLICY.provider,
            'model': VALIDATION_COST_POLICY.model,
            'reasoning_effort': VALIDATION_COST_POLICY.reasoning_effort,
        },
        'identities': expected_identities,
        'metrics': metrics,
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost_usd': format(_money(input_tokens, output_tokens), 'f'),
        },
        'gate_passed': gate_passed,
    }


def evaluate_fixture(path: Path) -> dict[str, Any]:
    return _evaluate(_load_fixture(path))


def _request_from_case(case: dict[str, Any], ordinal: int) -> CandidateValidationRequest:
    live_input = case['live_input']
    item_type = live_input['item_type']
    second_field = 'reason' if item_type == 'history_event' else 'result_summary'
    return CandidateValidationRequest(
        candidate_slot_id=f'C{ordinal:02d}',
        item_type=item_type,
        claims=(
            ValidationClaimInput(field_key='title', text=live_input['title']),
            ValidationClaimInput(field_key=second_field, text=live_input['fact']),
        ),
        evidence_slots=tuple(
            ValidationEvidenceSlot(slot_id=f'E{index:02d}', text=text)
            for index, text in enumerate(live_input['evidence'], start=1)
        ),
    )


def _live_evaluate(
    path: Path,
    *,
    model_builder: Callable[..., Any] | None,
) -> dict[str, Any]:
    fixture = _load_fixture(path)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EvaluationConfigurationError('provider_key_required')
    settings = Settings(
        _env_file=None,
        openai_api_key=api_key,
        auto_review_validator_input_cost_per_1m_tokens=(
            VALIDATION_COST_POLICY.input_cost_per_1m_tokens
        ),
        auto_review_validator_output_cost_per_1m_tokens=(
            VALIDATION_COST_POLICY.output_cost_per_1m_tokens
        ),
    )
    try:
        route = build_auto_review_validator_model_route(
            settings,
            timeout_seconds=AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
            http_hook=_SERVER_OWNED_FENCED_SEND_HOOK,
            chat_model_builder=model_builder,
        )
    except ReviewModelUnavailableError:
        raise EvaluationConfigurationError('provider_unavailable') from None

    evaluated_cases: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    policy = AutoReviewPolicyEngine()
    cases = fixture['cases']
    try:
        for start in range(0, len(cases), 4):
            batch = cases[start:start + 4]
            requests = tuple(
                _request_from_case(case, ordinal)
                for ordinal, case in enumerate(batch, start=1)
            )
            invocation = _prepare_many(requests, settings=settings)
            structured = route.model.with_structured_output(
                _structured_schema_from_framing(invocation.response_schema_framing),
                method='json_schema',
                strict=True,
                include_raw=True,
            )
            if bool(getattr(structured, 'cache', False)) or bool(
                getattr(structured, 'verbose', False)
            ):
                raise LiveEvaluationError('unsafe_model_controls')
            with tracing_context(enabled=False):
                raw_result = structured.invoke(
                    invocation.messages,
                    config={'callbacks': []},
                )
            parsed, usage = _parse_provider_result(
                raw_result,
                invocation=invocation,
            )
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
            policy_inputs = []
            for index, (case, request, result) in enumerate(
                zip(batch, requests, parsed.results, strict=True)
            ):
                eligible = case.get('policy_eligible', True)
                eligibility = EligibilityPolicyResult(
                    decision='eligible' if eligible else 'human_review',
                    reason_codes=('eligible',) if eligible else ('item_type_not_allowed',),
                )
                policy_inputs.append(ValidationPolicyInput(
                    eligibility=eligibility,
                    expected_candidate_slot_id=invocation.candidate_slot_ids[index],
                    expected_claim_fields=tuple(
                        claim.field_key for claim in request.claims
                    ),
                    expected_evidence_slot_ids=(
                        invocation.candidate_evidence_slot_ids[index]
                    ),
                    validator_available=True,
                    validator_malformed=False,
                    result=result,
                    validation_identity=SUPPORTED_VALIDATION_IDENTITY,
                    post_validation_state_matches=True,
                ))
            decisions = policy.evaluate_batch(tuple(policy_inputs))
            for case, decision in zip(batch, decisions, strict=True):
                evaluated_cases.append({
                    **case,
                    'predicted_approval': decision.decision == 'auto_approve',
                })
    except Exception as exc:
        if isinstance(exc, EvaluationConfigurationError):
            raise
        raise LiveEvaluationError('provider_gate_failed') from None

    live_fixture = {
        **fixture,
        'cases': evaluated_cases,
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        },
    }
    return _evaluate(live_fixture)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('fake', 'live-openai'))
    parser.add_argument('--provider', default='openai')
    parser.add_argument('--model', default='gpt-5.6-terra')
    parser.add_argument('--reasoning-effort', default='medium')
    parser.add_argument('--aggregate-only', action='store_true', required=True)
    parser.add_argument('--allow-paid-provider-call', action='store_true')
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    model_builder: Callable[..., Any] | None = None,
) -> int:
    try:
        args = _parser().parse_args(argv)
        expected_route = ('openai', 'gpt-5.6-terra', 'medium')
        if (args.provider, args.model, args.reasoning_effort) != expected_route:
            raise EvaluationConfigurationError('route_invalid')
        if args.mode == 'live-openai':
            if not (
                args.allow_paid_provider_call
                and os.getenv('PARAWORKS_ALLOW_PAID_TERRA_EVAL') == '1'
            ):
                raise EvaluationConfigurationError('authorization_required')
            report = _live_evaluate(args.fixture, model_builder=model_builder)
        else:
            report = evaluate_fixture(args.fixture)
    except (EvaluationConfigurationError, SystemExit):
        print('configuration_invalid', file=sys.stderr)
        return 2
    except LiveEvaluationError:
        print('evaluation_failed', file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    return 0 if report['gate_passed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
