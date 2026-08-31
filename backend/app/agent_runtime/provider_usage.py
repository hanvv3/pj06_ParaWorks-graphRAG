from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

from backend.app.rag.retrieval import StrictProviderUsage

_TOKEN_FIELD_NAMES = frozenset({
    'input_tokens',
    'prompt_tokens',
    'output_tokens',
    'completion_tokens',
    'total_tokens',
})
_USAGE_WRAPPER_NAMES = frozenset({'token_usage', 'usage'})
_ALL_AUTHORITY_NAMES = _TOKEN_FIELD_NAMES | _USAGE_WRAPPER_NAMES

RagRunProductOutcome = Literal[
    'supported',
    'search_projected',
    'no_match',
    'hidden_only',
    'safety_filter_empty',
    'insufficient_evidence',
    'evidence_unavailable',
]
AssistantSafePersistedErrorOutcome = Literal[
    'budget_exceeded',
    'retriever_not_configured',
    'retriever_unavailable',
    'runtime_version_unavailable',
    'model_unavailable',
    'model_provider_failed',
    'provider_response_identity_invalid',
    'provider_embedding_payload_invalid',
    'provider_usage_overrun',
    'provider_safety_unavailable',
    'structured_output_invalid',
    'citation_validation_failed',
    'unexpected_internal_error',
]
RagResultOutcome: TypeAlias = (
    RagRunProductOutcome | AssistantSafePersistedErrorOutcome
)
AssistantProductOutcome = Literal[
    'supported',
    'no_match',
    'hidden_only',
    'safety_filter_empty',
    'insufficient_evidence',
    'evidence_unavailable',
]
AssistantPersistedOutcome: TypeAlias = (
    AssistantProductOutcome | AssistantSafePersistedErrorOutcome
)
AssistantApplicationOutcome: TypeAlias = AssistantPersistedOutcome | Literal[
    'persistence_failed',
    'commit_unknown',
    'permission_denied',
    'owner_not_found',
    'invalid_input',
    'input_safety_blocked',
    'input_scanner_unavailable',
]
RagCannedMessageIdentity = Literal[
    'rag-canned-no-evidence:v1',
    'rag-canned-evidence-unavailable:v1',
    'rag-canned-budget-failure:v1',
    'rag-canned-generation-failure:v1',
]


class StrictChatUsageParser:
    def parse_message(self, message: object) -> StrictProviderUsage:
        try:
            primary_value = getattr(message, 'usage_metadata', None)
            response_value = getattr(message, 'response_metadata', {})
        except Exception:
            raise ValueError('chat provider usage is unavailable') from None
        primary = _mapping_snapshot(
            primary_value,
            context='chat usage',
        )
        if set(primary).intersection(
            {'prompt_tokens', 'completion_tokens'} | _USAGE_WRAPPER_NAMES
        ):
            raise ValueError(
                'chat primary usage contains extra authoritative aliases'
            )
        _validate_non_authoritative_details(
            primary,
            allowed_authority_names={
                'input_tokens',
                'output_tokens',
                'total_tokens',
            },
            context='chat usage',
        )
        input_tokens = _required_exact_token(primary, 'input_tokens')
        output_tokens = _required_exact_token(primary, 'output_tokens')
        total_tokens = _required_exact_token(primary, 'total_tokens')
        usage = _strict_usage(input_tokens, output_tokens, total_tokens)

        response = _mapping_snapshot(response_value, context='chat usage')
        if set(response).intersection(_TOKEN_FIELD_NAMES):
            raise ValueError('chat response usage authority is misplaced')
        _validate_non_authoritative_details(
            response,
            allowed_authority_names=_USAGE_WRAPPER_NAMES,
            context='chat usage',
        )
        for key in ('token_usage', 'usage'):
            if key not in response:
                continue
            alias = _mapping_snapshot(response[key], context='chat usage')
            if set(alias).intersection(_USAGE_WRAPPER_NAMES):
                raise ValueError('chat response usage wrapper is nested')
            _validate_non_authoritative_details(
                alias,
                allowed_authority_names=_TOKEN_FIELD_NAMES,
                context='chat usage',
            )
            alias_input = _required_synonym_token(
                alias,
                primary_key='input_tokens',
                secondary_key='prompt_tokens',
            )
            alias_output = _required_synonym_token(
                alias,
                primary_key='output_tokens',
                secondary_key='completion_tokens',
            )
            alias_total = _required_exact_token(alias, 'total_tokens')
            alias_usage = _strict_usage(
                alias_input,
                alias_output,
                alias_total,
            )
            if alias_usage != usage:
                raise ValueError('chat provider usage authorities conflict')
        return usage


class StrictEmbeddingUsageParser:
    def parse_usage(self, usage: object) -> StrictProviderUsage:
        metadata = _mapping_snapshot(usage, context='embedding usage')
        if set(metadata).intersection(_USAGE_WRAPPER_NAMES):
            raise ValueError('embedding provider usage wrapper is misplaced')
        _validate_non_authoritative_details(
            metadata,
            allowed_authority_names=_TOKEN_FIELD_NAMES,
            context='embedding usage',
        )
        prompt_tokens = _required_exact_token(metadata, 'prompt_tokens')
        if 'input_tokens' in metadata:
            input_tokens = _required_exact_token(metadata, 'input_tokens')
            if input_tokens != prompt_tokens:
                raise ValueError('embedding provider usage authorities conflict')
        for key in ('output_tokens', 'completion_tokens'):
            if key in metadata and _required_exact_token(metadata, key) != 0:
                raise ValueError('embedding provider usage output must be zero')
        total_tokens = _required_exact_token(metadata, 'total_tokens')
        return _strict_usage(prompt_tokens, 0, total_tokens)


def _mapping_snapshot(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{context} must be a mapping')
    try:
        items = tuple(value.items())
    except Exception:
        raise ValueError(f'{context} must be a stable mapping') from None
    result: dict[str, object] = {}
    for key, item in items:
        if type(key) is not str or key in result:
            raise ValueError(f'{context} contains invalid keys')
        result[key] = item
    return result


def _validate_non_authoritative_details(
    values: dict[str, object],
    *,
    allowed_authority_names: frozenset[str] | set[str],
    context: str,
) -> None:
    for key, value in values.items():
        if key in allowed_authority_names:
            continue
        _reject_nested_authority(value, context=context, depth=0)


def _reject_nested_authority(
    value: object,
    *,
    context: str,
    depth: int,
) -> None:
    if depth > 32:
        raise ValueError(f'{context} detail nesting is invalid')
    if isinstance(value, Mapping):
        nested = _mapping_snapshot(value, context=context)
        if set(nested).intersection(_ALL_AUTHORITY_NAMES):
            raise ValueError(f'{context} contains nested usage authority')
        for item in nested.values():
            _reject_nested_authority(
                item,
                context=context,
                depth=depth + 1,
            )
        return
    if type(value) in {list, tuple}:
        for item in value:
            _reject_nested_authority(
                item,
                context=context,
                depth=depth + 1,
            )


def _required_exact_token(values: dict[str, object], key: str) -> int:
    if key not in values:
        raise ValueError('provider usage is missing an authoritative token field')
    value = values[key]
    if type(value) is not int or value < 0:
        raise ValueError('provider usage token fields must be exact nonnegative integers')
    return value


def _required_synonym_token(
    values: dict[str, object],
    *,
    primary_key: str,
    secondary_key: str,
) -> int:
    present = tuple(
        key for key in (primary_key, secondary_key) if key in values
    )
    if not present:
        raise ValueError('provider usage is missing an authoritative token field')
    selected = _required_exact_token(values, present[0])
    if len(present) == 2 and _required_exact_token(values, present[1]) != selected:
        raise ValueError('provider usage token aliases conflict')
    return selected


def _strict_usage(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> StrictProviderUsage:
    if total_tokens != input_tokens + output_tokens:
        raise ValueError('provider usage total does not match its components')
    try:
        return StrictProviderUsage(input_tokens, output_tokens, total_tokens)
    except ValueError:
        raise ValueError('provider usage totals are outside supported bounds') from None
