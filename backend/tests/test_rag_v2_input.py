from __future__ import annotations

import unicodedata
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.agents.rag_orchestrator_agent.v2_input import (
    AssistantContextMessage,
    RagInputSafetyError,
    prepare_assistant_request_text,
    prepare_direct_request_text,
)

KEY = b'test-only-rag-v2-fingerprint-key'


def _message(
    message_id: int,
    *,
    seconds: int,
    role: str = 'user',
    content: str,
) -> AssistantContextMessage:
    return AssistantContextMessage(
        message_id=message_id,
        created_at=datetime(2026, 8, 31, tzinfo=UTC) + timedelta(seconds=seconds),
        role=role,  # type: ignore[arg-type]
        content=content,
    )


def test_direct_request_preserves_whitespace_and_long_caller_text_exactly() -> None:
    caller_text = '   ' + ('질문 ' * 1_001)

    prepared = prepare_direct_request_text(caller_text, key=KEY)

    assert prepared.caller_text == caller_text
    assert prepared.retrieval_query_text == caller_text
    assert prepared.answer_question_text == caller_text
    assert prepared.query_context_version == 'direct-query:v1'
    assert prepared.normalized_current_user_text == ' '.join(caller_text.split())


def test_direct_request_uses_exact_bytes_but_normalized_current_hmac() -> None:
    composed = '담당자 김하나'
    decomposed = unicodedata.normalize('NFD', composed)

    left = prepare_direct_request_text(composed, key=KEY)
    right = prepare_direct_request_text(decomposed, key=KEY)

    assert left.current_text_hmac == right.current_text_hmac
    assert left.retrieval_query_hmac != right.retrieval_query_hmac
    assert left.answer_question_hmac != right.answer_question_hmac


def test_assistant_context_is_chronological_user_only_deduplicated_and_retrieval_only() -> None:
    messages = (
        _message(4, seconds=4, role='assistant', content='prior assistant must not appear'),
        _message(2, seconds=2, content='first user'),
        _message(1, seconds=1, content='first user'),
        _message(3, seconds=3, content='second   user'),
        _message(5, seconds=5, content='same current'),
    )

    prepared = prepare_assistant_request_text(' same current ', messages, key=KEY)

    assert prepared.retrieval_query_text == (
        '최근 대화:\nuser: first user\nuser: second user\n현재 질문: same current'
    )
    assert prepared.answer_question_text == 'same current'
    assert 'prior assistant' not in prepared.retrieval_query_text
    assert '최근 대화:' not in prepared.answer_question_text


def test_assistant_context_truncates_at_500_scalars_and_drops_oldest_whole_lines() -> None:
    long_prior = 'x' * 501
    current = 'q' * 7_700
    messages = tuple(
        _message(index, seconds=index, content=long_prior + str(index))
        for index in range(1, 7)
    )

    prepared = prepare_assistant_request_text(current, messages, key=KEY)

    assert 'user: ' + ('x' * 499) + '…' not in prepared.retrieval_query_text
    assert len(prepared.retrieval_query_text) <= 8_000
    assert prepared.retrieval_query_text.endswith(f'현재 질문: {current}')


def test_assistant_context_excludes_sensitive_history_before_selecting_six_unique_rows() -> None:
    messages = tuple(
        _message(index, seconds=index, content=f'safe {index}')
        for index in range(1, 8)
    ) + (
        _message(8, seconds=8, content='OPENAI_API_KEY=abcdEFGH1234ijklMNOP'),
        _message(9, seconds=9, content='safe 7'),
    )

    prepared = prepare_assistant_request_text('current', messages, key=KEY)

    assert 'OPENAI_API_KEY' not in prepared.retrieval_query_text
    assert 'user: safe 1' not in prepared.retrieval_query_text
    assert 'user: safe 2' in prepared.retrieval_query_text
    assert prepared.retrieval_query_text.count('user: safe 7') == 1


def test_assistant_context_removes_sensitive_prior_message_and_rejects_current_without_plaintext() -> None:
    secret = 'OPENAI_API_KEY=abcdEFGH1234ijklMNOP'
    prepared = prepare_assistant_request_text(
        'safe current',
        (_message(1, seconds=1, content=secret), _message(2, seconds=2, content='safe prior')),
        key=KEY,
    )

    assert secret not in prepared.retrieval_query_text
    assert 'safe prior' in prepared.retrieval_query_text
    with pytest.raises(RagInputSafetyError) as exc_info:
        prepare_assistant_request_text(secret, (), key=KEY)
    assert secret not in str(exc_info.value)
