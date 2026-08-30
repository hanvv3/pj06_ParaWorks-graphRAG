from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langsmith import tracing_context

from backend.app.agent_runtime.auto_review_cost_policy import (
    AUTO_REVIEW_MAX_EXTRACTION_COST_USD,
    AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
    EXTRACTION_ROUTE_POLICIES,
)
from backend.app.agent_runtime.auto_review_validator import (
    _validate_response_usage_metadata,
)
from backend.app.agent_runtime.contracts import (
    EvidenceMessage,
    EvidencePacket,
    PermissionContext,
)
from backend.app.agent_runtime.review_v21_extraction import (
    EXTRACTION_REGISTRY_VERSION,
    ExtractionCallStateError,
    _render_invocation,
    parse_extraction_result,
)
from backend.app.core.config import Settings

SCHEMA_VERSION = 'auto-review-extraction-compat-report:v1'
FIXTURE_SCHEMA_VERSION = 'auto-review-extraction-compat:v1'


class CompatibilityConfigurationError(ValueError):
    pass


class LiveCompatibilityError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CompatibilityConfigurationError('fixture_invalid') from None
    if not isinstance(value, dict) or value.get('schema_version') != FIXTURE_SCHEMA_VERSION:
        raise CompatibilityConfigurationError('fixture_invalid')
    routes = value.get('routes')
    if not isinstance(routes, list):
        raise CompatibilityConfigurationError('fixture_invalid')
    for route in routes:
        live_cases = route.get('live_cases') if isinstance(route, dict) else None
        if (
            not isinstance(live_cases, list)
            or len(live_cases) != 2
            or [case.get('expected_outcome') for case in live_cases]
            != ['candidate', 'no_candidate']
            or any(
                not isinstance(case.get('evidence'), list)
                or not case['evidence']
                or any(
                    not isinstance(text, str) or not text.strip()
                    for text in case['evidence']
                )
                for case in live_cases
                if isinstance(case, dict)
            )
        ):
            raise CompatibilityConfigurationError('fixture_invalid')
    return value


def _maximum_cost(policy: Any) -> Decimal:
    return (
        (
            Decimal(policy.max_input_tokens) * policy.input_cost_per_1m_tokens
            + Decimal(policy.max_output_tokens) * policy.output_cost_per_1m_tokens
        )
        / Decimal(1_000_000)
    ).quantize(Decimal('0.000001'), rounding=ROUND_CEILING)


def evaluate_fixture(path: Path) -> dict[str, Any]:
    return _evaluate(_load(path))


def _evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    routes = fixture.get('routes')
    checks = fixture.get('checks')
    usage = fixture.get('usage')
    if not isinstance(routes, list) or not isinstance(checks, dict) or not isinstance(usage, dict):
        raise CompatibilityConfigurationError('fixture_invalid')
    expected = [
        {
            'agent_name': route.agent_name,
            'prompt_version': route.prompt_version,
            'output_contract_version': route.output_contract_version,
        }
        for route in EXTRACTION_ROUTE_POLICIES
    ]
    route_passes = [
        actual.get('agent_name') == frozen['agent_name']
        and actual.get('prompt_version') == frozen['prompt_version']
        and actual.get('output_contract_version') == frozen['output_contract_version']
        and actual.get('outcomes') == ['candidate', 'no_candidate']
        for actual, frozen in zip(routes, expected, strict=False)
        if isinstance(actual, dict)
    ]
    route_set_passed = len(routes) == len(expected) == len(route_passes) and all(route_passes)
    required_checks = {
        'singular_cardinality', 'payload_integrity', 'field_evidence_integrity',
        'accepted_2048_boundary', 'rejected_2049_boundary',
        'rendered_input_within_cap', 'one_attempt', 'no_fallback', 'usage_parsed',
    }
    checks_passed = set(checks) == required_checks and all(checks.values())
    per_route = {_maximum_cost(route) for route in EXTRACTION_ROUTE_POLICIES}
    try:
        input_tokens = int(usage['input_tokens'])
        output_tokens = int(usage['output_tokens'])
    except (KeyError, TypeError, ValueError):
        raise CompatibilityConfigurationError('fixture_invalid') from None
    if input_tokens < 0 or output_tokens < 0:
        raise CompatibilityConfigurationError('fixture_invalid')
    total_cost = (
        Decimal(input_tokens) * Decimal('0.750000')
        + Decimal(output_tokens) * Decimal('4.500000')
    ) / Decimal(1_000_000)
    gate_passed = (
        route_set_passed
        and checks_passed
        and fixture.get('cost_policy_version') == 'auto-review-extraction-cost:v1'
        and per_route == {Decimal('0.016716')}
        and Decimal('0.083580') == AUTO_REVIEW_MAX_EXTRACTION_COST_USD
    )
    return {
        'schema_version': SCHEMA_VERSION,
        'safety_key': {
            'purpose': 'extraction', 'provider': 'openai',
            'model': 'gpt-5.4-mini-2026-03-17', 'reasoning_effort': 'none',
        },
        'registry_version': EXTRACTION_REGISTRY_VERSION,
        'cost_policy_version': 'auto-review-extraction-cost:v1',
        'route_identities': expected,
        'aggregate': {
            'route_count': len(expected),
            'passed_route_count': sum(route_passes) if route_set_passed else 0,
            'check_count': len(required_checks),
            'passed_check_count': sum(bool(checks.get(key)) for key in required_checks),
            'per_route_full_cap_reserve_usd': '0.016716',
            'five_route_full_cap_reserve_usd': '0.083580',
        },
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost_usd': format(total_cost.quantize(Decimal('0.000001')), 'f'),
        },
        'gate_passed': gate_passed,
    }


def _packet(agent_name: str, texts: list[str]) -> EvidencePacket:
    return EvidencePacket(
        source_type='internal_document',
        source_window=f'offline-evaluation:{agent_name}',
        messages=[
            EvidenceMessage(
                source_id=f'sanitized-{agent_name}-{index}',
                source_url='https://example.invalid/sanitized-evaluation',
                text=text,
                author='sanitized-evaluator',
                timestamp=f'2026-08-{index:02d}T00:00:00Z',
                permission_level='internal',
                metadata={'stable_message_identity': f'M{index:02d}'},
            )
            for index, text in enumerate(texts, start=1)
        ],
        permission_context=PermissionContext(
            user_id='offline-evaluator',
            role='evaluator',
            allowed_permission_levels=('internal',),
        ),
    )


def _provider_usage(value: Any, *, max_input: int, max_output: int) -> tuple[int, int]:
    if not isinstance(value, Mapping) or value.get('parsing_error') is not None:
        raise LiveCompatibilityError('provider_result_invalid')
    raw = value.get('raw')
    if not isinstance(raw, AIMessage) or not isinstance(raw.usage_metadata, Mapping):
        raise LiveCompatibilityError('provider_usage_invalid')
    input_tokens = raw.usage_metadata.get('input_tokens')
    output_tokens = raw.usage_metadata.get('output_tokens')
    total_tokens = raw.usage_metadata.get('total_tokens')
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or input_tokens < 0
        or output_tokens < 0
        or input_tokens > max_input
        or output_tokens > max_output
        or not isinstance(total_tokens, int)
        or isinstance(total_tokens, bool)
        or total_tokens != input_tokens + output_tokens
    ):
        raise LiveCompatibilityError('provider_usage_invalid')
    _validate_response_usage_metadata(
        raw.response_metadata,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return input_tokens, output_tokens


def _build_model(
    *,
    policy: Any,
    api_key: str,
    model_builder: Callable[..., Any] | None,
) -> Any:
    builder = model_builder
    if builder is None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise CompatibilityConfigurationError('provider_unavailable') from None
        builder = ChatOpenAI
    try:
        model = builder(
            model=policy.model,
            api_key=api_key,
            reasoning_effort=policy.reasoning_effort,
            use_responses_api=True,
            timeout=AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
            max_retries=0,
            max_completion_tokens=policy.max_output_tokens,
            verbose=False,
            cache=False,
        )
    except Exception:
        raise CompatibilityConfigurationError('provider_unavailable') from None
    if bool(getattr(model, 'cache', False)) or bool(getattr(model, 'verbose', False)):
        raise CompatibilityConfigurationError('provider_unavailable')
    return model


def _live_evaluate(
    path: Path,
    *,
    model_builder: Callable[..., Any] | None,
) -> dict[str, Any]:
    fixture = _load(path)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise CompatibilityConfigurationError('provider_key_required')
    settings = Settings(
        _env_file=None,
        agent_runtime_fingerprint_secret='e' * 32,
        agent_runtime_fingerprint_key_version='offline-eval-v1',
    )
    fixture_routes = {
        route['agent_name']: route for route in fixture['routes']
    }
    live_routes: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    rendered_within_cap = True
    outcomes_match = True
    payload_integrity = True
    field_evidence_integrity = True
    usage_parsed = True
    try:
        for policy in EXTRACTION_ROUTE_POLICIES:
            fixture_route = fixture_routes.get(policy.agent_name)
            if fixture_route is None:
                raise CompatibilityConfigurationError('fixture_invalid')
            model = _build_model(
                policy=policy,
                api_key=api_key,
                model_builder=model_builder,
            )
            observed_outcomes: list[str] = []
            for live_case in fixture_route['live_cases']:
                invocation = _render_invocation(
                    _packet(policy.agent_name, live_case['evidence']),
                    policy,
                    settings=settings,
                )
                rendered_within_cap = rendered_within_cap and (
                    invocation.framed_input_tokens <= policy.max_input_tokens
                    and invocation.max_output_tokens == policy.max_output_tokens
                )
                structured = model.with_structured_output(
                    invocation.output_schema,
                    method='json_schema',
                    strict=True,
                    include_raw=True,
                )
                if bool(getattr(structured, 'cache', False)) or bool(
                    getattr(structured, 'verbose', False)
                ):
                    raise LiveCompatibilityError('unsafe_model_controls')
                with tracing_context(enabled=False):
                    raw_result = structured.invoke(
                        (('human', invocation.canonical_text),),
                        config={'callbacks': []},
                    )
                parsed_value = raw_result.get('parsed') if isinstance(raw_result, Mapping) else None
                parsed = parse_extraction_result(
                    invocation.output_schema,
                    parsed_value,
                )
                observed_outcomes.append(parsed.result_kind)
                outcomes_match = outcomes_match and (
                    parsed.result_kind == live_case['expected_outcome']
                )
                payload_integrity = payload_integrity and (
                    (parsed.result_kind == 'candidate') == (parsed.candidate is not None)
                )
                if parsed.candidate is not None:
                    bindings = parsed.candidate.field_evidence_bindings
                    field_evidence_integrity = field_evidence_integrity and bool(bindings)
                    field_evidence_integrity = field_evidence_integrity and all(
                        binding.evidence_slot_id in invocation.evidence_slot_ids
                        for binding in bindings
                    )
                used_input, used_output = _provider_usage(
                    raw_result,
                    max_input=policy.max_input_tokens,
                    max_output=policy.max_output_tokens,
                )
                input_tokens += used_input
                output_tokens += used_output
            live_routes.append({
                'agent_name': policy.agent_name,
                'prompt_version': policy.prompt_version,
                'output_contract_version': policy.output_contract_version,
                'outcomes': observed_outcomes,
                'live_cases': fixture_route['live_cases'],
            })
        boundary_schema = EXTRACTION_ROUTE_POLICIES[0]
        boundary_result = {
            'result_kind': 'no_candidate',
            'candidate': None,
            'no_candidate_reason': 'no_relevant_evidence',
        }
        schema = _render_invocation(
            _packet(boundary_schema.agent_name, ['관련 후보가 없다.']),
            boundary_schema,
            settings=settings,
        ).output_schema
        parse_extraction_result(
            schema,
            boundary_result,
            canonical_token_count_override=2048,
        )
        accepted_boundary = True
        try:
            parse_extraction_result(
                schema,
                boundary_result,
                canonical_token_count_override=2049,
            )
        except ExtractionCallStateError:
            rejected_boundary = True
        else:
            rejected_boundary = False
    except CompatibilityConfigurationError:
        raise
    except Exception:
        raise LiveCompatibilityError('provider_gate_failed') from None

    live_fixture = {
        **fixture,
        'routes': live_routes,
        'checks': {
            'singular_cardinality': outcomes_match,
            'payload_integrity': payload_integrity,
            'field_evidence_integrity': field_evidence_integrity,
            'accepted_2048_boundary': accepted_boundary,
            'rejected_2049_boundary': rejected_boundary,
            'rendered_input_within_cap': rendered_within_cap,
            'one_attempt': True,
            'no_fallback': True,
            'usage_parsed': usage_parsed,
        },
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        },
    }
    return _evaluate(live_fixture)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('fake', 'live-openai'))
    parser.add_argument('--provider', default='openai')
    parser.add_argument('--model', default='gpt-5.4-mini-2026-03-17')
    parser.add_argument('--reasoning-effort', default='none')
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
        if (args.provider, args.model, args.reasoning_effort) != (
            'openai', 'gpt-5.4-mini-2026-03-17', 'none'
        ):
            raise CompatibilityConfigurationError('route_invalid')
        if args.mode == 'live-openai':
            if not (
                args.allow_paid_provider_call
                and os.getenv('PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL') == '1'
            ):
                raise CompatibilityConfigurationError('authorization_required')
            report = _live_evaluate(args.fixture, model_builder=model_builder)
        else:
            report = evaluate_fixture(args.fixture)
    except (CompatibilityConfigurationError, SystemExit):
        print('configuration_invalid', file=sys.stderr)
        return 2
    except LiveCompatibilityError:
        print('evaluation_failed', file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    return 0 if report['gate_passed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
