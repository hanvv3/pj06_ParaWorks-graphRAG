from __future__ import annotations

from collections.abc import Mapping, Sequence
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
_MAX_DETAIL_DEPTH = 32
_MAX_DETAIL_NODES = 4_096

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
            return self._parse_message(message)
        except Exception:
            raise ValueError('chat provider usage is invalid') from None

    def _parse_message(self, message: object) -> StrictProviderUsage:
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
        try:
            return self._parse_usage(usage)
        except Exception:
            raise ValueError('embedding provider usage is invalid') from None

    def _parse_usage(self, usage: object) -> StrictProviderUsage:
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

    def bounded_items() -> tuple[tuple[object, object], ...]:
        try:
            items: list[tuple[object, object]] = []
            for item in value.items():
                if type(item) is not tuple or len(item) != 2:
                    raise ValueError
                if len(items) >= _MAX_DETAIL_NODES:
                    raise ValueError
                items.append(item)
            return tuple(items)
        except Exception:
            raise ValueError(f'{context} must be a stable mapping') from None

    def validated_items() -> tuple[tuple[str, object], ...]:
        items = bounded_items()
        seen: set[str] = set()
        validated: list[tuple[str, object]] = []
        for key, item in items:
            if type(key) is not str or key in seen:
                raise ValueError(f'{context} contains invalid keys')
            seen.add(key)
            validated.append((key, item))
        return tuple(validated)

    items = validated_items()
    if type(value) is not dict:
        observed_again = validated_items()
        try:
            unstable = len(items) != len(observed_again) or any(
                first_key != second_key or first_value is not second_value
                for (first_key, first_value), (second_key, second_value)
                in zip(items, observed_again, strict=True)
            )
        except Exception:
            raise ValueError(f'{context} must be a stable mapping') from None
        if unstable:
            raise ValueError(f'{context} must be a stable mapping')
    return dict(items)


def _validate_non_authoritative_details(
    values: dict[str, object],
    *,
    allowed_authority_names: frozenset[str] | set[str],
    context: str,
) -> None:
    active_containers: set[int] = set()
    remaining_nodes = [_MAX_DETAIL_NODES]
    for key, value in values.items():
        if key in allowed_authority_names:
            continue
        _reject_nested_authority(
            value,
            context=context,
            depth=0,
            active_containers=active_containers,
            remaining_nodes=remaining_nodes,
        )


def _reject_nested_authority(
    value: object,
    *,
    context: str,
    depth: int,
    active_containers: set[int],
    remaining_nodes: list[int],
) -> None:
    remaining_nodes[0] -= 1
    if depth > _MAX_DETAIL_DEPTH or remaining_nodes[0] < 0:
        raise ValueError(f'{context} detail nesting is invalid')
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f'{context} detail nesting is invalid')
        active_containers.add(identity)
        try:
            nested = _mapping_snapshot(value, context=context)
            if set(nested).intersection(_ALL_AUTHORITY_NAMES):
                raise ValueError(f'{context} contains nested usage authority')
            for item in nested.values():
                _reject_nested_authority(
                    item,
                    context=context,
                    depth=depth + 1,
                    active_containers=active_containers,
                    remaining_nodes=remaining_nodes,
                )
        finally:
            active_containers.remove(identity)
        return
    if type(value) in {list, tuple}:
        identity = id(value)
        if identity in active_containers or len(value) > remaining_nodes[0]:
            raise ValueError(f'{context} detail nesting is invalid')
        active_containers.add(identity)
        try:
            for item in value:
                _reject_nested_authority(
                    item,
                    context=context,
                    depth=depth + 1,
                    active_containers=active_containers,
                    remaining_nodes=remaining_nodes,
                )
        finally:
            active_containers.remove(identity)
        return
    if isinstance(value, Sequence) and type(value) not in {str, bytes}:
        raise ValueError(f'{context} detail container is invalid')


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
