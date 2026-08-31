from __future__ import annotations

from collections.abc import Mapping

import pytest

from backend.app.agent_runtime.provider_usage import (
    StrictChatUsageParser,
    StrictEmbeddingUsageParser,
)
from backend.app.rag.retrieval import StrictProviderUsage


class _Message:
    def __init__(self, *, usage_metadata: object, response_metadata: object) -> None:
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _Mapping(Mapping[str, object]):
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _message(
    primary: object | None = None,
    *,
    response: object | None = None,
) -> _Message:
    return _Message(
        usage_metadata=primary
        if primary is not None
        else {'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15},
        response_metadata={} if response is None else response,
    )


def test_chat_parser_uses_required_primary_and_all_equal_optional_aliases() -> None:
    usage = StrictChatUsageParser().parse_message(
        _message(
            response=_Mapping({
                'token_usage': _Mapping({
                    'input_tokens': 12,
                    'prompt_tokens': 12,
                    'output_tokens': 3,
                    'completion_tokens': 3,
                    'total_tokens': 15,
                    'cached_tokens': 2,
                }),
                'usage': {
                    'prompt_tokens': 12,
                    'completion_tokens': 3,
                    'total_tokens': 15,
                },
                'request_id': 'ignored-non-authoritative-detail',
            })
        )
    )

    assert usage == StrictProviderUsage(12, 3, 15)


@pytest.mark.parametrize(
    'message',
    (
        object(),
        _message(primary={}),
        _message(primary={'input_tokens': 1, 'output_tokens': 2}),
        _message(primary={'input_tokens': 1, 'output_tokens': 2, 'total_tokens': 4}),
        _message(primary={'input_tokens': -1, 'output_tokens': 2, 'total_tokens': 1}),
        _message(primary={'input_tokens': True, 'output_tokens': 2, 'total_tokens': 3}),
        _message(primary={'input_tokens': _IntSubclass(1), 'output_tokens': 2, 'total_tokens': 3}),
        _message(primary={'input_tokens': 1.0, 'output_tokens': 2, 'total_tokens': 3}),
        _message(primary={'input_tokens': 12, 'prompt_tokens': 11, 'output_tokens': 3, 'total_tokens': 15}),
        _message(response={'token_usage': object()}),
        _message(response={'token_usage': {'output_tokens': 3, 'total_tokens': 15}}),
        _message(response={'token_usage': {'input_tokens': 12, 'output_tokens': 3}}),
        _message(response={'token_usage': {'input_tokens': 12, 'prompt_tokens': 11, 'output_tokens': 3, 'total_tokens': 15}}),
        _message(response={'token_usage': {'input_tokens': 12, 'output_tokens': 3, 'completion_tokens': 2, 'total_tokens': 15}}),
        _message(response={'token_usage': {'input_tokens': 11, 'output_tokens': 3, 'total_tokens': 14}}),
        _message(response={'token_usage': {'input_tokens': 12, 'output_tokens': -3, 'total_tokens': 9}}),
        _message(response={'token_usage': {'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 16}}),
        _message(response={'token_usage': {'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15}, 'usage': {'input_tokens': 12, 'output_tokens': 4, 'total_tokens': 16}}),
    ),
)
def test_chat_parser_rejects_missing_malformed_negative_or_conflicting_authority(
    message: object,
) -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictChatUsageParser().parse_message(message)


def test_response_less_transport_exception_is_not_a_usage_object() -> None:
    transport_error = RuntimeError('response was never returned')

    with pytest.raises(ValueError, match='usage'):
        StrictChatUsageParser().parse_message(transport_error)


def test_embedding_parser_accepts_equal_input_alias_and_zero_output_aliases() -> None:
    usage = StrictEmbeddingUsageParser().parse_usage(_Mapping({
        'prompt_tokens': 19,
        'input_tokens': 19,
        'total_tokens': 19,
        'output_tokens': 0,
        'completion_tokens': 0,
        'provider_detail': {'ignored': True},
    }))

    assert usage == StrictProviderUsage(19, 0, 19)


@pytest.mark.parametrize(
    'usage',
    (
        None,
        object(),
        {},
        {'prompt_tokens': 1},
        {'total_tokens': 1},
        {'prompt_tokens': 1, 'total_tokens': 2},
        {'prompt_tokens': -1, 'total_tokens': -1},
        {'prompt_tokens': True, 'total_tokens': 1},
        {'prompt_tokens': _IntSubclass(1), 'total_tokens': 1},
        {'prompt_tokens': '1', 'total_tokens': 1},
        {'prompt_tokens': 1.0, 'total_tokens': 1},
        {'prompt_tokens': 1, 'input_tokens': 2, 'total_tokens': 1},
        {'prompt_tokens': 1, 'total_tokens': 1, 'output_tokens': 1},
        {'prompt_tokens': 1, 'total_tokens': 1, 'completion_tokens': -1},
        {'prompt_tokens': 1, 'total_tokens': 1, 'output_tokens': 0, 'completion_tokens': 1},
    ),
)
def test_embedding_parser_rejects_missing_malformed_conflicting_or_nonzero_output(
    usage: object,
) -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictEmbeddingUsageParser().parse_usage(usage)


@pytest.mark.parametrize(
    'usage',
    (
        {
            'prompt_tokens': 4,
            'total_tokens': 4,
            'usage': {'prompt_tokens': 999, 'total_tokens': 999},
        },
        {
            'prompt_tokens': 4,
            'total_tokens': 4,
            'token_usage': {'prompt_tokens': 999, 'total_tokens': 999},
        },
        {
            'prompt_tokens': 4,
            'total_tokens': 4,
            'detail': {'usage': {'prompt_tokens': 999}},
        },
    ),
)
def test_embedding_parser_rejects_authority_wrappers_at_wrong_nesting(
    usage: object,
) -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictEmbeddingUsageParser().parse_usage(usage)


@pytest.mark.parametrize(
    'response',
    (
        {
            'token_usage': {
                'input_tokens': 12,
                'output_tokens': 3,
                'total_tokens': 15,
                'usage': {
                    'input_tokens': 999,
                    'output_tokens': 0,
                    'total_tokens': 999,
                },
            }
        },
        {
            'usage': {
                'input_tokens': 12,
                'output_tokens': 3,
                'total_tokens': 15,
                'token_usage': {
                    'input_tokens': 999,
                    'output_tokens': 0,
                    'total_tokens': 999,
                },
            }
        },
        {
            'input_tokens': 12,
            'output_tokens': 3,
            'total_tokens': 15,
        },
        {
            'provider_detail': {
                _IntSubclass(1): 'invalid key subclass',
            },
        },
        {
            _StrSubclass('usage'): {
                'input_tokens': 12,
                'output_tokens': 3,
                'total_tokens': 15,
            },
        },
    ),
)
def test_chat_parser_rejects_authority_at_wrong_response_position(
    response: object,
) -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictChatUsageParser().parse_message(_message(response=response))


def test_non_authoritative_nested_provider_detail_remains_allowed() -> None:
    message = _message(response={
        'provider_detail': {
            'cached': {'count': 2},
            'request_family': 'direct-standard',
        },
    })
    embedding = {
        'prompt_tokens': 4,
        'total_tokens': 4,
        'provider_detail': {'cached': {'count': 2}},
    }

    assert StrictChatUsageParser().parse_message(message).total_tokens == 15
    assert StrictEmbeddingUsageParser().parse_usage(embedding).total_tokens == 4
