from decimal import Decimal

import pytest


def test_validation_policy_is_the_single_exact_terra_medium_registry_entry() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        AUTO_REVIEW_MAX_BATCH_COST_USD,
        AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
        AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
        get_validation_policy,
    )

    policy = get_validation_policy(
        cost_policy_version='auto-review-cost:v1',
        provider='openai',
        model='gpt-5.6-terra',
        reasoning_effort='medium',
    )

    assert policy is not None
    assert policy.input_cost_per_1m_tokens == Decimal('2.000000')
    assert policy.output_cost_per_1m_tokens == Decimal('12.000000')
    assert Decimal('2.000000') == AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M
    assert Decimal('12.000000') == AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M
    assert Decimal('0.048864') == AUTO_REVIEW_MAX_BATCH_COST_USD


@pytest.mark.parametrize(
    ('provider', 'model', 'reasoning_effort'),
    [
        ('azure_openai', 'gpt-5.6-terra', 'medium'),
        ('openai', 'gpt-5.6-terra', 'none'),
        ('openai', 'gpt-5.6', 'medium'),
    ],
)
def test_validation_policy_has_no_alias_or_nearest_fallback(
    provider: str, model: str, reasoning_effort: str
) -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import get_validation_policy

    assert (
        get_validation_policy(
            cost_policy_version='auto-review-cost:v1',
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        is None
    )


def test_extraction_registry_contains_exactly_five_snapshot_none_reasoning_routes() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        EXTRACTION_ROUTE_POLICIES,
    )

    assert tuple(policy.agent_name for policy in EXTRACTION_ROUTE_POLICIES) == (
        'mail_document_agent',
        'timeline_agent',
        'history_agent',
        'decision_record_agent',
        'todo_agent',
    )
    assert {
        (policy.provider, policy.model, policy.reasoning_effort)
        for policy in EXTRACTION_ROUTE_POLICIES
    } == {('openai', 'gpt-5.4-mini-2026-03-17', 'none')}


def test_each_extraction_route_has_exact_agent_prompt_output_contract_and_one_candidate_cap() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        EXTRACTION_ROUTE_POLICIES,
    )

    assert [
        (policy.agent_name, policy.prompt_version, policy.output_contract_version)
        for policy in EXTRACTION_ROUTE_POLICIES
    ] == [
        ('mail_document_agent', 'mail-document-extraction:c5-v2', 'mail-document-candidate:c5-v2'),
        ('timeline_agent', 'timeline-extraction:c5-v2', 'timeline-candidate:c5-v2'),
        ('history_agent', 'history-extraction:c5-v2', 'history-candidate:c5-v2'),
        ('decision_record_agent', 'decision-record-extraction:c5-v2', 'decision-record-candidate:c5-v2'),
        ('todo_agent', 'todo-extraction:c5-v2', 'todo-candidate:c5-v2'),
    ]
    assert all(policy.max_candidates == 1 for policy in EXTRACTION_ROUTE_POLICIES)
    assert all(policy.max_provider_attempts == 1 for policy in EXTRACTION_ROUTE_POLICIES)


def test_extraction_registry_rejects_alias_azure_gemini_fallback_or_unknown_agent() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import get_extraction_route

    exact = {
        'extraction_cost_policy_version': 'auto-review-extraction-cost:v1',
        'provider': 'openai',
        'model': 'gpt-5.4-mini-2026-03-17',
        'reasoning_effort': 'none',
        'route_version': 'auto-review-extraction-route:v1',
        'prompt_version': 'timeline-extraction:c5-v2',
        'output_contract_version': 'timeline-candidate:c5-v2',
    }
    assert get_extraction_route('timeline_agent', **exact) is not None
    for mutation in (
        {'model': 'gpt-5.4-mini'},
        {'provider': 'azure_openai'},
        {'provider': 'gemini'},
        {'reasoning_effort': 'medium'},
        {'route_version': 'fallback'},
    ):
        assert get_extraction_route('timeline_agent', **(exact | mutation)) is None
    assert get_extraction_route('unregistered_agent', **exact) is None


def test_v1_terra_price_registry_is_exact_six_place_two_and_twelve() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import VALIDATION_COST_POLICY

    assert VALIDATION_COST_POLICY.input_cost_per_1m_tokens == Decimal('2.000000')
    assert VALIDATION_COST_POLICY.output_cost_per_1m_tokens == Decimal('12.000000')


def test_v1_extraction_price_registry_is_exact_six_place_point75_and_four_point5() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        EXTRACTION_ROUTE_POLICIES,
    )

    assert {
        (route.input_cost_per_1m_tokens, route.output_cost_per_1m_tokens)
        for route in EXTRACTION_ROUTE_POLICIES
    } == {(Decimal('0.750000'), Decimal('4.500000'))}


def test_five_extraction_calls_reserve_0083580_and_two_validation_batches_reserve_0097728() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        AUTO_REVIEW_MAX_EXTRACTION_COST_USD,
        AUTO_REVIEW_MAX_VALIDATION_COST_USD,
    )

    assert Decimal('0.083580') == AUTO_REVIEW_MAX_EXTRACTION_COST_USD
    assert Decimal('0.097728') == AUTO_REVIEW_MAX_VALIDATION_COST_USD


def test_max_profile_reserve_is_0181308_below_twenty_cent_budget() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        AUTO_REVIEW_MAX_PROFILE_COST_USD,
        AUTO_REVIEW_MAX_WORKFLOW_COST_USD,
    )

    assert Decimal('0.181308') == AUTO_REVIEW_MAX_PROFILE_COST_USD
    assert AUTO_REVIEW_MAX_PROFILE_COST_USD < AUTO_REVIEW_MAX_WORKFLOW_COST_USD
    assert Decimal('0.20') == AUTO_REVIEW_MAX_WORKFLOW_COST_USD


def test_shadow_enforce_rejects_price_config_below_above_or_different_from_registry() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        confirmation_prices_match_registry,
    )

    assert not confirmation_prices_match_registry(
        extraction_input=Decimal('0.749999'),
        extraction_output=Decimal('4.500000'),
        validation_input=Decimal('2.000000'),
        validation_output=Decimal('12.000000'),
    )
    assert not confirmation_prices_match_registry(
        extraction_input=Decimal('0.750001'),
        extraction_output=Decimal('4.500000'),
        validation_input=Decimal('2.000000'),
        validation_output=Decimal('12.000000'),
    )


def test_validation_preview_admission_authorization_and_launch_resolve_one_exact_policy_entry() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import get_validation_policy

    assert get_validation_policy(
        cost_policy_version='auto-review-cost:v1',
        provider='openai',
        model='gpt-5.6-terra',
        reasoning_effort='medium',
    ) is not None


def test_each_extraction_route_preview_admission_authorization_and_launch_resolve_its_exact_registry_entry() -> None:
    from backend.app.agent_runtime.auto_review_cost_policy import (
        EXTRACTION_ROUTE_POLICIES,
        get_extraction_route,
    )

    for route in EXTRACTION_ROUTE_POLICIES:
        assert get_extraction_route(
            route.agent_name,
            extraction_cost_policy_version=route.extraction_cost_policy_version,
            provider=route.provider,
            model=route.model,
            reasoning_effort=route.reasoning_effort,
            route_version=route.route_version,
            prompt_version=route.prompt_version,
            output_contract_version=route.output_contract_version,
        ) == route
