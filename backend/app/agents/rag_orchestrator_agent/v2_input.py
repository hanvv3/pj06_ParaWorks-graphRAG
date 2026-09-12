from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from backend.app.agent_runtime.auto_review_input_safety import (
    scan_auto_review_plaintext,
)
from backend.app.agent_runtime.fingerprints import keyed_fingerprint
from backend.app.agent_runtime.rag_v2_identity import (
    StrictUnicodeScalarValidator,
    exact_utf8_bytes,
)


@dataclass(frozen=True, slots=True)
class PreparedRagRequestText:
    caller_text: str
    normalized_current_user_text: str
    retrieval_query_text: str
    answer_question_text: str
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1']
    current_text_hmac: str
    retrieval_query_hmac: str
    answer_question_hmac: str


@dataclass(frozen=True, slots=True)
class AssistantContextMessage:
    message_id: int
    created_at: datetime
    role: Literal['user', 'assistant']
    content: str


class RagInputSafetyError(ValueError):
    code = 'input_safety_blocked'

    def __init__(self) -> None:
        super().__init__('rag input safety policy blocked the request')


class RagInputScannerUnavailableError(RuntimeError):
    code = 'input_scanner_unavailable'


def prepare_direct_request_text(text: str, *, key: bytes) -> PreparedRagRequestText:
    caller_text = StrictUnicodeScalarValidator.validate(text)
    normalized = _normalized_copy(caller_text)
    _scan_current(caller_text, normalized)
    return _prepared(
        caller_text=caller_text,
        normalized_current_user_text=normalized,
        retrieval_query_text=caller_text,
        answer_question_text=caller_text,
        query_context_version='direct-query:v1',
        key=key,
    )


def prepare_assistant_request_text(
    current_text: str,
    prior_messages: Sequence[AssistantContextMessage],
    *,
    key: bytes,
) -> PreparedRagRequestText:
    caller_text = StrictUnicodeScalarValidator.validate(current_text)
    normalized = _normalized_copy(caller_text)
    _scan_current(caller_text, normalized)
    answer_question = caller_text.strip()
    if not answer_question:
        raise ValueError('assistant message content is invalid')

    prior_user_messages = _validated_prior_user_messages(
        prior_messages, answer_question
    )
    context_lines = _context_lines(prior_user_messages)
    retrieval_query = _bounded_contextual_query(context_lines, answer_question)
    _scan_value(retrieval_query)
    return _prepared(
        caller_text=caller_text,
        normalized_current_user_text=normalized,
        retrieval_query_text=retrieval_query,
        answer_question_text=answer_question,
        query_context_version='assistant-context:v1',
        key=key,
    )


def _prepared(
    *,
    caller_text: str,
    normalized_current_user_text: str,
    retrieval_query_text: str,
    answer_question_text: str,
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1'],
    key: bytes,
) -> PreparedRagRequestText:
    return PreparedRagRequestText(
        caller_text=caller_text,
        normalized_current_user_text=normalized_current_user_text,
        retrieval_query_text=retrieval_query_text,
        answer_question_text=answer_question_text,
        query_context_version=query_context_version,
        current_text_hmac=keyed_fingerprint(
            {'normalized_text': normalized_current_user_text},
            secret=key,
            schema_version='rag-current-text-normalized:v1',
            policy_version=query_context_version,
        ),
        retrieval_query_hmac=keyed_fingerprint(
            exact_utf8_bytes(retrieval_query_text),
            secret=key,
            schema_version='rag-retrieval-query-bytes:v1',
            policy_version=query_context_version,
        ),
        answer_question_hmac=keyed_fingerprint(
            {'answer_question_bytes': exact_utf8_bytes(answer_question_text)},
            secret=key,
            schema_version='rag-answer-question-bytes:v1',
            policy_version=query_context_version,
        ),
    )


def _normalized_copy(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', value).split())


def _scan_current(caller_text: str, normalized: str) -> None:
    _scan_value(caller_text)
    _scan_value(normalized)


def _scan_value(value: str) -> None:
    try:
        allowed = scan_auto_review_plaintext(value).allowed
        if type(allowed) is not bool:
            raise TypeError('invalid scanner decision')
    except Exception:
        raise RagInputScannerUnavailableError() from None
    if not allowed:
        raise RagInputSafetyError()


def _validated_prior_user_messages(
    prior_messages: Sequence[AssistantContextMessage], answer_question: str
) -> list[AssistantContextMessage]:
    validated: list[AssistantContextMessage] = []
    for message in prior_messages:
        if type(message.message_id) is not int or message.message_id <= 0:
            raise ValueError('assistant context message is invalid')
        if not isinstance(message.created_at, datetime):
            raise ValueError('assistant context message is invalid')
        if message.role not in {'user', 'assistant'}:
            raise ValueError('assistant context message is invalid')
        StrictUnicodeScalarValidator.validate(message.content)
        validated.append(message)
    try:
        ordered = sorted(
            validated, key=lambda message: (message.created_at, message.message_id)
        )
    except TypeError as exc:
        raise ValueError('assistant context ordering is invalid') from exc
    users = [message for message in ordered if message.role == 'user']
    if users and users[-1].content.strip() == answer_question:
        users.pop()
    return [
        message
        for message in users
        if scan_auto_review_plaintext(message.content).allowed
    ]


def _context_lines(messages: Sequence[AssistantContextMessage]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for message in messages:
        compacted = ' '.join(message.content.strip().split())
        if not compacted:
            continue
        if len(compacted) > 500:
            compacted = compacted[:499].rstrip() + '…'
        line = f'user: {compacted}'
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines[-6:]


def _bounded_contextual_query(
    context_lines: Sequence[str], answer_question: str
) -> str:
    lines = list(context_lines)
    while True:
        query_lines = (
            (['최근 대화:'] if lines else [])
            + lines
            + [f'현재 질문: {answer_question}']
        )
        query = '\n'.join(query_lines)
        if len(query) <= 8_000:
            return query
        if not lines:
            raise ValueError('assistant contextual query is too long')
        lines.pop(0)
