from __future__ import annotations

from collections.abc import Mapping, Sequence

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


class _ListSubclass(list):
    pass


class _TupleSubclass(tuple):
    pass


class _Sequence(Sequence[object]):
    def __init__(self, values: tuple[object, ...]) -> None:
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class _UnstableMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self._calls = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        self._calls += 1
        if self._calls == 1:
            return (('safe', 1),)
        return (('usage', {'input_tokens': 999}),)


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


@pytest.mark.parametrize(
    'detail',
    (
        _ListSubclass([{'usage': {'input_tokens': 999}}]),
        _TupleSubclass(({'token_usage': {'input_tokens': 999}},)),
        _Sequence(({'usage': {'input_tokens': 999}},)),
    ),
)
def test_usage_parser_rejects_sequence_subclasses_and_custom_sequences(
    detail: object,
) -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictChatUsageParser().parse_message(_message(response={
            'provider_detail': detail,
        }))


def test_usage_parser_rejects_unstable_custom_mapping() -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictEmbeddingUsageParser().parse_usage({
            'prompt_tokens': 4,
            'total_tokens': 4,
            'provider_detail': _UnstableMapping(),
        })


def test_usage_parser_rejects_nested_authority_in_stable_custom_mapping() -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictChatUsageParser().parse_message(_message(response={
            'provider_detail': _Mapping({
                'detail': _Mapping({'usage': {'input_tokens': 999}}),
            }),
        }))


@pytest.mark.parametrize('container_type', (list, tuple))
def test_usage_parser_rejects_cyclic_or_excessively_deep_details(
    container_type,
) -> None:
    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    cyclic: object = cyclic_list if container_type is list else (cyclic_list,)
    deep: object = {'safe': True}
    for _ in range(40):
        deep = [deep]

    for detail in (cyclic, deep):
        with pytest.raises(ValueError, match='usage'):
            StrictChatUsageParser().parse_message(_message(response={
                'provider_detail': detail,
            }))


def test_usage_parser_rejects_excessive_detail_size() -> None:
    with pytest.raises(ValueError, match='usage'):
        StrictEmbeddingUsageParser().parse_usage({
            'prompt_tokens': 4,
            'total_tokens': 4,
            'provider_detail': [0] * 5_000,
        })


def test_usage_parser_preserves_bounded_exact_sequence_details() -> None:
    message = _message(response={
        'provider_detail': [
            {'cached': (1, 2)},
            {'request_family': 'direct-standard'},
        ],
    })

    assert StrictChatUsageParser().parse_message(message).total_tokens == 15
