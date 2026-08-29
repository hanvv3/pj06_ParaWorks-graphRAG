from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol

from backend.app.connectors.base import SourceEvent

SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA = 'server-source-content:v1'
SERVER_PARSER_POLICY_VERSION = 'server-source-parser-policy:v1'
SERVER_PARSER_VERSION = 'source-event-paragraph-parser:v1'
SERVER_CHUNK_POLICY_VERSION = 'paragraph-chunks:1200:v1'
SERVER_CHUNK_MAX_CHARS = 1_200

_SEMANTIC_METADATA_KEYS: dict[str, tuple[str, ...]] = {
    'gmail': (),
    'gmail_attachment': ('filename', 'mime_type'),
    'drive': ('mime_type',),
    'calendar': (
        'attendee_domains',
        'end',
        'event_status',
        'location',
        'organizer_email',
        'start',
    ),
}
_CALENDAR_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_GOOGLE_MILLIS = re.compile(r'^\d+$')
SERVER_ALLOWED_MIME_TYPES = frozenset(
    {
        'application/haansofthwp',
        'application/pdf',
        'application/vnd.google-apps.document',
        'application/vnd.google-apps.presentation',
        'application/vnd.google-apps.spreadsheet',
        'application/vnd.hancom.hwpx',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/x-hwp',
        'text/csv',
        'text/markdown',
        'text/plain',
    }
)


class SourceContentUnverifiableError(ValueError):
    """The server cannot derive the frozen canonical source envelope."""


class CanonicalSourceState(Protocol):
    source_type: str
    server_content_signature_schema: str | None
    server_content_signature: str | None
    permission_level: str
    raw_metadata: dict


class CurrentParserRunState(Protocol):
    server_content_signature_schema: str | None
    server_content_signature: str | None
    parser_policy_version: str | None
    parser_name: str
    parser_status: str
    parser_status_reason: str | None
    parser_version: str | None
    chunk_policy_version: str | None
    mime_type: str
    content_signature: str


@dataclass(frozen=True)
class CanonicalSourceContentSignature:
    schema: str
    canonical_json: str
    signature: str


@dataclass(frozen=True)
class ServerParserPolicy:
    parser_policy_version: str
    parser_name: str
    parser_version: str
    chunk_policy_version: str
    mime_type: str
    chunk_max_chars: int


@dataclass(frozen=True)
class SourceStateChangeClassification:
    content_changed: bool
    permission_changed: bool
    parser_policy_changed: bool
    primary_code: str


def canonical_source_content_signature(
    event: SourceEvent,
) -> CanonicalSourceContentSignature:
    semantic_keys = _SEMANTIC_METADATA_KEYS.get(event.source_type)
    if semantic_keys is None:
        raise SourceContentUnverifiableError('unsupported source type')
    title = _required_text(event.title, field_name='title')
    body = _required_text(event.body, field_name='body')
    author = _optional_text(event.author, field_name='author')
    participants = _participant_list(event.participants, field_name='participants')
    semantic_timestamp = _semantic_timestamp(event)
    payload: dict[str, object] = {
        'schema': SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
        'source_type': event.source_type,
        'title': title,
        'body': body,
        'author': author,
        'participants': participants,
        'semantic_timestamp': semantic_timestamp,
    }
    for key in semantic_keys:
        value = event.raw_metadata.get(key)
        if key == 'attendee_domains':
            payload[key] = _participant_list(value, field_name=key)
        elif event.source_type == 'calendar' and key in {'start', 'end'}:
            payload[key] = _calendar_time(value, field_name=key, required=key == 'start')
        else:
            payload[key] = _optional_text(value, field_name=key)
    if event.source_type == 'calendar' and payload['start'] != semantic_timestamp:
        raise SourceContentUnverifiableError('calendar start provenance mismatch')
    try:
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SourceContentUnverifiableError('canonical JSON is unverifiable') from exc
    signature = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    return CanonicalSourceContentSignature(
        schema=SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
        canonical_json=canonical_json,
        signature=signature,
    )


def server_parser_policy_for_source(
    source_type: str,
    *,
    mime_type: object = None,
) -> ServerParserPolicy:
    if source_type not in _SEMANTIC_METADATA_KEYS:
        raise SourceContentUnverifiableError('unsupported source type')
    if source_type == 'gmail':
        normalized_mime_type = 'message/rfc822'
    elif source_type == 'calendar':
        normalized_mime_type = 'text/calendar'
    else:
        if mime_type is None:
            normalized_mime_type = 'application/octet-stream'
        elif not isinstance(mime_type, str):
            raise SourceContentUnverifiableError('mime_type must be a string or null')
        else:
            normalized = mime_type.strip().lower()
            normalized_mime_type = (
                normalized
                if normalized in SERVER_ALLOWED_MIME_TYPES
                else 'application/octet-stream'
            )
    return ServerParserPolicy(
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_name=f'server_{source_type}_source_event',
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        mime_type=normalized_mime_type,
        chunk_max_chars=SERVER_CHUNK_MAX_CHARS,
    )


def server_parser_policy_for_event(event: SourceEvent) -> ServerParserPolicy:
    return server_parser_policy_for_source(
        source_type=event.source_type,
        mime_type=event.raw_metadata.get('mime_type'),
    )


def server_parser_run_matches_authority(
    *,
    source: CanonicalSourceState,
    parser_run: CurrentParserRunState,
) -> bool:
    server_content_signature = source.server_content_signature
    if (
        source.server_content_signature_schema
        != SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA
        or server_content_signature is None
    ):
        return False
    try:
        expected = server_parser_policy_for_source(
            source_type=source.source_type,
            mime_type=(source.raw_metadata or {}).get('mime_type'),
        )
    except SourceContentUnverifiableError:
        return False
    return bool(
        parser_run.server_content_signature_schema
        == SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA
        and parser_run.server_content_signature == server_content_signature
        and parser_run.content_signature == server_content_signature
        and parser_run.parser_policy_version == expected.parser_policy_version
        and parser_run.parser_name == expected.parser_name
        and parser_run.parser_status == 'parsed'
        and parser_run.parser_status_reason is None
        and parser_run.parser_version == expected.parser_version
        and parser_run.chunk_policy_version == expected.chunk_policy_version
        and parser_run.mime_type == expected.mime_type
    )


def classify_source_state_change(
    *,
    source: CanonicalSourceState | None,
    event: SourceEvent,
    current_parser_run: CurrentParserRunState | None,
    computed_signature: CanonicalSourceContentSignature | None = None,
    parser_policy: ServerParserPolicy | None = None,
) -> SourceStateChangeClassification:
    computed = computed_signature or canonical_source_content_signature(event)
    policy = parser_policy or server_parser_policy_for_event(event)
    normalized_permission = normalize_source_permission(event.permission_level)
    content_changed = bool(
        source is None
        or source.server_content_signature_schema != computed.schema
        or source.server_content_signature != computed.signature
    )
    permission_changed = bool(
        source is not None
        and normalize_source_permission(source.permission_level)
        != normalized_permission
    )
    parser_policy_changed = bool(
        source is not None
        and (
            current_parser_run is None
            or current_parser_run.server_content_signature_schema
            != source.server_content_signature_schema
            or current_parser_run.server_content_signature
            != source.server_content_signature
            or current_parser_run.parser_policy_version
            != policy.parser_policy_version
            or current_parser_run.parser_name != policy.parser_name
            or current_parser_run.parser_version != policy.parser_version
            or current_parser_run.chunk_policy_version
            != policy.chunk_policy_version
            or current_parser_run.mime_type != policy.mime_type
        )
    )
    primary_code = source_state_primary_code(
        content_changed=content_changed,
        permission_changed=permission_changed,
        parser_policy_changed=parser_policy_changed,
    )
    return SourceStateChangeClassification(
        content_changed=content_changed,
        permission_changed=permission_changed,
        parser_policy_changed=parser_policy_changed,
        primary_code=primary_code,
    )


def source_state_primary_code(
    *,
    content_changed: bool,
    permission_changed: bool,
    parser_policy_changed: bool,
) -> str:
    if content_changed:
        return 'content_changed'
    if permission_changed:
        return 'permission_changed'
    if parser_policy_changed:
        return 'parser_policy_changed'
    return 'unchanged'


def normalize_source_permission(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceContentUnverifiableError('permission level is unverifiable')
    return value.strip().lower()


def connector_content_signature(event: SourceEvent) -> str | None:
    value = event.raw_metadata.get('content_signature')
    return value if isinstance(value, str) and value else None


def _semantic_timestamp(event: SourceEvent) -> str | dict[str, str]:
    raw = event.semantic_timestamp_raw
    if event.source_type == 'calendar':
        return _calendar_time(raw, field_name='semantic_timestamp_raw', required=True)
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise SourceContentUnverifiableError('semantic timestamp is missing or malformed')
    try:
        if event.source_type in {'gmail', 'gmail_attachment'}:
            if not raw.isascii() or _GOOGLE_MILLIS.fullmatch(raw) is None:
                raise ValueError('Gmail semantic timestamp must be internalDate millis')
            value = datetime.fromtimestamp(int(raw) / 1_000, tz=UTC)
        else:
            value = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError('timestamp is naive')
    except (OverflowError, OSError, TypeError, ValueError) as exc:
        raise SourceContentUnverifiableError('semantic timestamp is missing or malformed') from exc
    return _utc_microseconds(value)


def _calendar_time(
    value: object,
    *,
    field_name: str,
    required: bool,
) -> dict[str, str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceContentUnverifiableError(f'{field_name} is missing or malformed')
    if _CALENDAR_DATE.fullmatch(value):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise SourceContentUnverifiableError(
                f'{field_name} is missing or malformed'
            ) from exc
        if parsed.isoformat() != value:
            raise SourceContentUnverifiableError(f'{field_name} is missing or malformed')
        return {'kind': 'date', 'value': value}
    try:
        parsed_datetime = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed_datetime.tzinfo is None or parsed_datetime.utcoffset() is None:
            raise ValueError('timestamp is naive')
    except (TypeError, ValueError) as exc:
        raise SourceContentUnverifiableError(f'{field_name} is missing or malformed') from exc
    return {'kind': 'instant', 'value': _utc_microseconds(parsed_datetime)}


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SourceContentUnverifiableError(f'{field_name} must be a string')
    return _normalize_text(value)


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SourceContentUnverifiableError(f'{field_name} must be a string or null')
    return _normalize_text(value)


def _participant_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SourceContentUnverifiableError(f'{field_name} must be a string list')
    normalized = {_normalize_text(item) for item in value}
    return sorted(normalized, key=lambda item: item.encode('utf-8'))


def _normalize_text(value: str) -> str:
    return unicodedata.normalize('NFC', value.replace('\r\n', '\n').replace('\r', '\n'))


def _utc_microseconds(value: datetime) -> str:
    try:
        return value.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    except (OverflowError, OSError, ValueError) as exc:
        raise SourceContentUnverifiableError('timestamp is outside the supported range') from exc
