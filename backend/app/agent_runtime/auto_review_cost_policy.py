from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from backend.app.schemas.review_workflow import DEFAULT_REVIEW_AGENT_NAMES

AUTO_REVIEW_COST_POLICY_VERSION = 'auto-review-cost:v1'
AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION = 'auto-review-extraction-cost:v1'
AUTO_REVIEW_VALIDATOR_PROVIDER = 'openai'
AUTO_REVIEW_VALIDATOR_MODEL = 'gpt-5.6-terra'
AUTO_REVIEW_REASONING_EFFORT = 'medium'
AUTO_REVIEW_EXTRACTION_PROVIDER = 'openai'
AUTO_REVIEW_EXTRACTION_MODEL = 'gpt-5.4-mini-2026-03-17'
AUTO_REVIEW_EXTRACTION_REASONING_EFFORT = 'none'
AUTO_REVIEW_EXTRACTION_ROUTE_VERSION = 'auto-review-extraction-route:v1'
AUTO_REVIEW_EXTRACTION_AGENT_NAMES = DEFAULT_REVIEW_AGENT_NAMES
AUTO_REVIEW_MAX_SELECTED_EXTRACTION_AGENTS = 5
AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS = 60
AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS = 5
AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS = 120
AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS = 30
AUTO_REVIEW_TOKENIZER_ENCODING = 'o200k_base'
AUTO_REVIEW_REPLY_PRIMING_TOKENS = 16
AUTO_REVIEW_FRAMING_SAFETY_TOKENS = 512
AUTO_REVIEW_MAX_INPUT_TOKENS = 6_000
AUTO_REVIEW_MAX_OUTPUT_TOKENS = 3_072
AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH = 4
AUTO_REVIEW_MAX_BATCHES_PER_WORKFLOW = 2
AUTO_REVIEW_MAX_CANDIDATES_PER_WORKFLOW = 5
AUTO_REVIEW_MAX_PROVIDER_ATTEMPTS = 1
AUTO_REVIEW_MAX_EXTRACTION_INPUT_CHARS_PER_AGENT = 24_000
AUTO_REVIEW_MAX_EXTRACTION_INPUT_TOKENS_PER_AGENT = 10_000
AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT = 2_048
AUTO_REVIEW_MAX_EXTRACTION_CANDIDATES_PER_AGENT = 1
AUTO_REVIEW_EXTRACTION_MAX_PROVIDER_ATTEMPTS_PER_AGENT = 1
AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M = Decimal('2.000000')
AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M = Decimal('12.000000')
AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M = Decimal('0.750000')
AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M = Decimal('4.500000')
AUTO_REVIEW_MAX_BATCH_COST_USD = Decimal('0.048864')
AUTO_REVIEW_MAX_VALIDATION_COST_USD = Decimal('0.097728')
AUTO_REVIEW_MAX_EXTRACTION_COST_USD = Decimal('0.083580')
AUTO_REVIEW_MAX_PROFILE_COST_USD = Decimal('0.181308')
AUTO_REVIEW_MAX_WORKFLOW_COST_USD = Decimal('0.20')


class ServerOwnedFencedSendHook:
    """Opaque Task 1 identity reserved for Task 3's fenced send transport."""

    __slots__ = ()


_SERVER_OWNED_FENCED_SEND_HOOK = ServerOwnedFencedSendHook()


def is_server_owned_fenced_send_hook(hook: object) -> bool:
    return hook is _SERVER_OWNED_FENCED_SEND_HOOK


@dataclass(frozen=True)
class ValidationCostPolicy:
    cost_policy_version: str
    provider: str
    model: str
    reasoning_effort: str
    token_estimator_version: str
    tokenizer_encoding: str
    reply_priming_tokens: int
    framing_safety_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    max_candidates_per_batch: int
    max_batches_per_workflow: int
    max_provider_attempts: int
    input_cost_per_1m_tokens: Decimal
    output_cost_per_1m_tokens: Decimal


@dataclass(frozen=True)
class ExtractionRoutePolicy:
    extraction_cost_policy_version: str
    agent_name: str
    provider: str
    model: str
    reasoning_effort: str
    route_version: str
    prompt_version: str
    output_contract_version: str
    token_estimator_version: str
    tokenizer_encoding: str
    reply_priming_tokens: int
    framing_safety_tokens: int
    max_input_chars: int
    max_input_tokens: int
    max_output_tokens: int
    max_candidates: int
    max_provider_attempts: int
    input_cost_per_1m_tokens: Decimal
    output_cost_per_1m_tokens: Decimal


VALIDATION_COST_POLICY = ValidationCostPolicy(
    cost_policy_version=AUTO_REVIEW_COST_POLICY_VERSION,
    provider=AUTO_REVIEW_VALIDATOR_PROVIDER,
    model=AUTO_REVIEW_VALIDATOR_MODEL,
    reasoning_effort=AUTO_REVIEW_REASONING_EFFORT,
    token_estimator_version='openai-o200k-chat:v1',
    tokenizer_encoding=AUTO_REVIEW_TOKENIZER_ENCODING,
    reply_priming_tokens=AUTO_REVIEW_REPLY_PRIMING_TOKENS,
    framing_safety_tokens=AUTO_REVIEW_FRAMING_SAFETY_TOKENS,
    max_input_tokens=AUTO_REVIEW_MAX_INPUT_TOKENS,
    max_output_tokens=AUTO_REVIEW_MAX_OUTPUT_TOKENS,
    max_candidates_per_batch=AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH,
    max_batches_per_workflow=AUTO_REVIEW_MAX_BATCHES_PER_WORKFLOW,
    max_provider_attempts=AUTO_REVIEW_MAX_PROVIDER_ATTEMPTS,
    input_cost_per_1m_tokens=AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
    output_cost_per_1m_tokens=AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
)


def _route(agent_name: str, prompt_version: str, output_contract_version: str) -> ExtractionRoutePolicy:
    return ExtractionRoutePolicy(
        extraction_cost_policy_version=AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION,
        agent_name=agent_name,
        provider=AUTO_REVIEW_EXTRACTION_PROVIDER,
        model=AUTO_REVIEW_EXTRACTION_MODEL,
        reasoning_effort=AUTO_REVIEW_EXTRACTION_REASONING_EFFORT,
        route_version=AUTO_REVIEW_EXTRACTION_ROUTE_VERSION,
        prompt_version=prompt_version,
        output_contract_version=output_contract_version,
        token_estimator_version='openai-o200k-extraction:v1',
        tokenizer_encoding=AUTO_REVIEW_TOKENIZER_ENCODING,
        reply_priming_tokens=AUTO_REVIEW_REPLY_PRIMING_TOKENS,
        framing_safety_tokens=AUTO_REVIEW_FRAMING_SAFETY_TOKENS,
        max_input_chars=AUTO_REVIEW_MAX_EXTRACTION_INPUT_CHARS_PER_AGENT,
        max_input_tokens=AUTO_REVIEW_MAX_EXTRACTION_INPUT_TOKENS_PER_AGENT,
        max_output_tokens=AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT,
        max_candidates=AUTO_REVIEW_MAX_EXTRACTION_CANDIDATES_PER_AGENT,
        max_provider_attempts=AUTO_REVIEW_EXTRACTION_MAX_PROVIDER_ATTEMPTS_PER_AGENT,
        input_cost_per_1m_tokens=AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M,
        output_cost_per_1m_tokens=AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M,
    )


EXTRACTION_ROUTE_POLICIES = (
    _route('mail_document_agent', 'mail-document-extraction:c5-v1', 'mail-document-candidate:c5-v1'),
    _route('timeline_agent', 'timeline-extraction:c5-v1', 'timeline-candidate:c5-v1'),
    _route('history_agent', 'history-extraction:c5-v1', 'history-candidate:c5-v1'),
    _route('decision_record_agent', 'decision-record-extraction:c5-v1', 'decision-record-candidate:c5-v1'),
    _route('todo_agent', 'todo-extraction:c5-v1', 'todo-candidate:c5-v1'),
)


def _maximum_cost(input_tokens: int, output_tokens: int, input_price: Decimal, output_price: Decimal) -> Decimal:
    return ((Decimal(input_tokens) * input_price + Decimal(output_tokens) * output_price) / Decimal(1_000_000)).quantize(Decimal('0.000001'), rounding=ROUND_CEILING)


assert tuple(route.agent_name for route in EXTRACTION_ROUTE_POLICIES) == AUTO_REVIEW_EXTRACTION_AGENT_NAMES
assert len(EXTRACTION_ROUTE_POLICIES) == AUTO_REVIEW_MAX_SELECTED_EXTRACTION_AGENTS
assert len({route.prompt_version for route in EXTRACTION_ROUTE_POLICIES}) == len(EXTRACTION_ROUTE_POLICIES)
assert len({route.output_contract_version for route in EXTRACTION_ROUTE_POLICIES}) == len(EXTRACTION_ROUTE_POLICIES)
assert {
    (route.provider, route.model, route.reasoning_effort, route.route_version,
     route.max_input_chars, route.max_input_tokens, route.max_output_tokens,
     route.max_candidates, route.max_provider_attempts, route.input_cost_per_1m_tokens,
     route.output_cost_per_1m_tokens)
    for route in EXTRACTION_ROUTE_POLICIES
} == {(
    AUTO_REVIEW_EXTRACTION_PROVIDER, AUTO_REVIEW_EXTRACTION_MODEL,
    AUTO_REVIEW_EXTRACTION_REASONING_EFFORT, AUTO_REVIEW_EXTRACTION_ROUTE_VERSION,
    AUTO_REVIEW_MAX_EXTRACTION_INPUT_CHARS_PER_AGENT,
    AUTO_REVIEW_MAX_EXTRACTION_INPUT_TOKENS_PER_AGENT,
    AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT,
    AUTO_REVIEW_MAX_EXTRACTION_CANDIDATES_PER_AGENT,
    AUTO_REVIEW_EXTRACTION_MAX_PROVIDER_ATTEMPTS_PER_AGENT,
    AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M,
    AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M,
)}
assert _maximum_cost(10_000, 2_048, AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M, AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M) == Decimal('0.016716')
assert _maximum_cost(6_000, 3_072, AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M, AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M) == AUTO_REVIEW_MAX_BATCH_COST_USD
assert Decimal(AUTO_REVIEW_MAX_SELECTED_EXTRACTION_AGENTS) * Decimal('0.016716') == AUTO_REVIEW_MAX_EXTRACTION_COST_USD
assert Decimal(AUTO_REVIEW_MAX_BATCHES_PER_WORKFLOW) * AUTO_REVIEW_MAX_BATCH_COST_USD == AUTO_REVIEW_MAX_VALIDATION_COST_USD
assert AUTO_REVIEW_MAX_EXTRACTION_COST_USD + AUTO_REVIEW_MAX_VALIDATION_COST_USD == AUTO_REVIEW_MAX_PROFILE_COST_USD


def get_validation_policy(*, cost_policy_version: str, provider: str, model: str, reasoning_effort: str) -> ValidationCostPolicy | None:
    policy = VALIDATION_COST_POLICY
    if (cost_policy_version, provider, model, reasoning_effort) != (policy.cost_policy_version, policy.provider, policy.model, policy.reasoning_effort):
        return None
    return policy


def get_extraction_route(agent_name: str, *, extraction_cost_policy_version: str, provider: str, model: str, reasoning_effort: str, route_version: str, prompt_version: str, output_contract_version: str) -> ExtractionRoutePolicy | None:
    for policy in EXTRACTION_ROUTE_POLICIES:
        if (agent_name, extraction_cost_policy_version, provider, model, reasoning_effort, route_version, prompt_version, output_contract_version) == (policy.agent_name, policy.extraction_cost_policy_version, policy.provider, policy.model, policy.reasoning_effort, policy.route_version, policy.prompt_version, policy.output_contract_version):
            return policy
    return None


def confirmation_prices_match_registry(*, extraction_input: Decimal | None, extraction_output: Decimal | None, validation_input: Decimal | None, validation_output: Decimal | None) -> bool:
    return (extraction_input, extraction_output, validation_input, validation_output) == (
        AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M,
        AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M,
        AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M,
        AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M,
    )
