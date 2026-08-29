from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from backend.app.connectors.base import SourceEvent
from backend.app.ingestion.source_content_signature import (
    SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
    SourceContentUnverifiableError,
    canonical_source_content_signature,
    classify_source_state_change,
    server_parser_policy_for_event,
)
from backend.app.ingestion.source_versions import current_content_signature


def _event(
    source_type: str,
    *,
    title: str = 'Title',
    body: str = 'Body',
    author: str | None = 'owner@example.com',
    participants: list[str] | None = None,
    permission_level: str = 'internal',
    semantic_timestamp_raw: str | None = '2026-05-01T09:00:00+00:00',
    raw_metadata: dict | None = None,
) -> SourceEvent:
    return SourceEvent(
        source_type=source_type,
        source_id=f'{source_type}:source-1',
        source_url='https://example.test/source-1',
        title=title,
        body=body,
        author=author,
        participants=participants or [],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level=permission_level,
        raw_metadata=raw_metadata or {},
        semantic_timestamp_raw=semantic_timestamp_raw,
    )


def test_server_signature_normalization_boundary_vectors_are_frozen_for_every_source_type() -> None:
    gmail = _event(
        'gmail',
        title=' Cafe\u0301\r\n ',
        body='\r\n Body  \rTail ',
        author=' Autho\u0301r\rX',
        participants=[
            'b@example.com',
            'A@example.com',
            'b@example.com',
            'a@example.com',
        ],
        semantic_timestamp_raw='0',
        raw_metadata={
            'sync_cursor': 'ignored',
            'content_signature': 'connector-only',
        },
    )

    computed = canonical_source_content_signature(gmail)

    assert computed.schema == SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA
    assert computed.canonical_json == (
        '{"author":" Auth\u00f3r\\nX","body":"\\n Body  \\nTail ",'
        '"participants":["A@example.com","a@example.com","b@example.com"],'
        '"schema":"server-source-content:v1",'
        '"semantic_timestamp":"1970-01-01T00:00:00.000000Z",'
        '"source_type":"gmail","title":" Caf\u00e9\\n "}'
    )
    assert computed.signature == '44dd97883f6938f9867373a138e46d0f2492d73204607735679c239015407797'

    attachment = canonical_source_content_signature(
        _event(
            'gmail_attachment',
            raw_metadata={'filename': ' budget\r\n.pdf ', 'mime_type': 'application/pdf'},
            semantic_timestamp_raw='0',
        )
    )
    drive = canonical_source_content_signature(
        _event(
            'drive',
            raw_metadata={'mime_type': 'text/plain'},
            semantic_timestamp_raw='2026-05-01T18:00:00+09:00',
        )
    )
    calendar = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw='2026-05-02',
            raw_metadata={
                'attendee_domains': ['z.example', 'a.example', 'z.example'],
                'end': '2026-05-03',
                'event_status': 'confirmed',
                'location': '',
                'organizer_email': None,
                'start': '2026-05-02',
            },
        )
    )

    assert '"filename":" budget\\n.pdf "' in attachment.canonical_json
    assert '"mime_type":"application/pdf"' in attachment.canonical_json
    assert '"mime_type":"text/plain"' in drive.canonical_json
    assert '"semantic_timestamp":"2026-05-01T09:00:00.000000Z"' in drive.canonical_json
    assert '"attendee_domains":["a.example","z.example"]' in calendar.canonical_json
    assert '"start":{"kind":"date","value":"2026-05-02"}' in calendar.canonical_json


def test_gmail_attachment_signature_uses_filename_not_nonexistent_attachment_name() -> None:
    first = canonical_source_content_signature(
        _event(
            'gmail_attachment',
            raw_metadata={
                'filename': 'budget.pdf',
                'attachment_name': 'ignored-a.pdf',
                'mime_type': 'application/pdf',
            },
            semantic_timestamp_raw='0',
        )
    )
    operational_noise = canonical_source_content_signature(
        _event(
            'gmail_attachment',
            raw_metadata={
                'filename': 'budget.pdf',
                'attachment_name': 'ignored-b.pdf',
                'mime_type': 'application/pdf',
            },
            semantic_timestamp_raw='0',
        )
    )
    renamed = canonical_source_content_signature(
        _event(
            'gmail_attachment',
            raw_metadata={
                'filename': 'approved-budget.pdf',
                'attachment_name': 'ignored-b.pdf',
                'mime_type': 'application/pdf',
            },
            semantic_timestamp_raw='0',
        )
    )

    assert first.signature == operational_noise.signature
    assert renamed.signature != first.signature


def test_calendar_all_day_date_and_offset_equivalent_instant_canonicalization_is_stable() -> None:
    base_metadata = {
        'attendee_domains': [],
        'end': '2026-05-02T11:00:00+09:00',
        'event_status': 'confirmed',
        'location': None,
        'organizer_email': 'owner@example.com',
        'start': '2026-05-02T10:00:00+09:00',
    }
    first = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw='2026-05-02T10:00:00+09:00',
            raw_metadata=base_metadata,
        )
    )
    equivalent = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw='2026-05-02T01:00:00Z',
            raw_metadata={
                **base_metadata,
                'start': '2026-05-02T01:00:00Z',
                'end': '2026-05-02T02:00:00Z',
            },
        )
    )
    all_day = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw='2026-05-02',
            raw_metadata={
                **base_metadata,
                'start': '2026-05-02',
                'end': '2026-05-03',
            },
        )
    )

    assert equivalent.signature == first.signature
    assert all_day.signature != first.signature
    assert '"kind":"instant"' in first.canonical_json
    assert '"kind":"date"' in all_day.canonical_json


def test_calendar_updated_revision_and_event_context_key_are_operational_not_semantic() -> None:
    semantic = {
        'attendee_domains': ['example.com'],
        'end': '2026-05-02T11:00:00Z',
        'event_status': 'confirmed',
        'location': 'Room A',
        'organizer_email': 'lead@example.com',
        'start': '2026-05-02T10:00:00Z',
    }
    first = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw=semantic['start'],
            raw_metadata={
                **semantic,
                'updated': '2026-05-01T10:00:00Z',
                'connector_revision': 'r1',
                'event_context_key': 'event:r1',
                'attendee_response_statuses': {'accepted': 1},
            },
        )
    )
    noisy = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw=semantic['start'],
            raw_metadata={
                **semantic,
                'updated': '2026-05-01T11:00:00Z',
                'connector_revision': 'r2',
                'event_context_key': 'event:r2',
                'attendee_response_statuses': {'declined': 7},
            },
        )
    )
    moved = canonical_source_content_signature(
        _event(
            'calendar',
            semantic_timestamp_raw=semantic['start'],
            raw_metadata={**semantic, 'location': 'Room B'},
        )
    )

    assert noisy.signature == first.signature
    assert moved.signature != first.signature


def test_semantic_metadata_change_changes_signature_but_cursor_account_or_scope_noise_does_not() -> None:
    first = canonical_source_content_signature(
        _event(
            'drive',
            raw_metadata={
                'mime_type': 'text/plain',
                'sync_cursor': 'cursor-a',
                'account_id': 'account-a',
                'required_scopes': ['drive.readonly'],
            },
        )
    )
    operational_noise = canonical_source_content_signature(
        _event(
            'drive',
            raw_metadata={
                'mime_type': 'text/plain',
                'sync_cursor': 'cursor-b',
                'account_id': 'account-b',
                'required_scopes': ['drive.full'],
            },
        )
    )
    semantic_change = canonical_source_content_signature(
        _event(
            'drive',
            raw_metadata={
                'mime_type': 'application/pdf',
                'sync_cursor': 'cursor-b',
                'account_id': 'account-b',
                'required_scopes': ['drive.full'],
            },
        )
    )

    assert operational_noise.signature == first.signature
    assert semantic_change.signature != first.signature


@pytest.mark.parametrize(
    'event',
    [
        _event('gmail', semantic_timestamp_raw=None),
        _event('gmail', semantic_timestamp_raw='not-a-timestamp'),
        _event('gmail', semantic_timestamp_raw='2026-05-01T09:00:00Z'),
        _event('gmail_attachment', semantic_timestamp_raw='2026-05-01T09:00:00Z'),
        _event('drive', semantic_timestamp_raw='2026-05-01T09:00:00'),
        _event(
            'calendar',
            semantic_timestamp_raw='2026-05-02',
            raw_metadata={
                'attendee_domains': [],
                'end': 'bad',
                'event_status': None,
                'location': None,
                'organizer_email': None,
                'start': '2026-05-02',
            },
        ),
        _event('gmail', title=None),  # type: ignore[arg-type]
        _event('gmail', body=None),  # type: ignore[arg-type]
    ],
)
def test_missing_malformed_or_ambiguous_external_signature_never_causes_unchanged_skip(
    event: SourceEvent,
) -> None:
    with pytest.raises(SourceContentUnverifiableError):
        canonical_source_content_signature(event)


def test_server_parser_registry_ignores_connector_parser_chunk_and_snippet_authority() -> None:
    first = _event(
        'drive',
        raw_metadata={
            'mime_type': 'text/plain',
            'parser_name': 'connector-a',
            'parser_status': 'unsupported',
            'chunk_max_chars': 999_999,
            'source_snippet': 'connector snippet a',
        },
    )
    second = replace(
        first,
        raw_metadata={
            **first.raw_metadata,
            'parser_name': 'connector-b',
            'parser_status': 'parsed',
            'chunk_max_chars': 12,
            'source_snippet': 'connector snippet b',
        },
    )

    assert server_parser_policy_for_event(first) == server_parser_policy_for_event(second)


def test_source_state_classification_has_three_authority_flags_and_one_primary_code() -> None:
    event = _event('drive', raw_metadata={'mime_type': 'text/plain'})
    signature = canonical_source_content_signature(event)
    policy = server_parser_policy_for_event(event)
    source = SimpleNamespace(
        server_content_signature_schema=signature.schema,
        server_content_signature=signature.signature,
        permission_level='internal',
    )
    parser_run = SimpleNamespace(
        server_content_signature_schema=signature.schema,
        server_content_signature=signature.signature,
        parser_policy_version=policy.parser_policy_version,
        parser_name=policy.parser_name,
        parser_version=policy.parser_version,
        chunk_policy_version=policy.chunk_policy_version,
        mime_type=policy.mime_type,
    )

    unchanged = classify_source_state_change(
        source=source,
        event=event,
        current_parser_run=parser_run,
    )
    permission_changed = classify_source_state_change(
        source=source,
        event=replace(event, permission_level='restricted'),
        current_parser_run=parser_run,
    )
    parser_changed = classify_source_state_change(
        source=source,
        event=event,
        current_parser_run=SimpleNamespace(**{**vars(parser_run), 'chunk_policy_version': 'old'}),
    )
    parser_signature_changed = classify_source_state_change(
        source=source,
        event=event,
        current_parser_run=SimpleNamespace(
            **{**vars(parser_run), 'server_content_signature': '0' * 64}
        ),
    )
    content_changed = classify_source_state_change(
        source=source,
        event=replace(event, body='Changed body'),
        current_parser_run=parser_run,
    )

    assert unchanged.__dict__ == {
        'content_changed': False,
        'permission_changed': False,
        'parser_policy_changed': False,
        'primary_code': 'unchanged',
    }
    assert permission_changed.__dict__ == {
        'content_changed': False,
        'permission_changed': True,
        'parser_policy_changed': False,
        'primary_code': 'permission_changed',
    }
    assert parser_changed.primary_code == 'parser_policy_changed'
    assert parser_signature_changed.primary_code == 'parser_policy_changed'
    assert content_changed.primary_code == 'content_changed'
    assert content_changed.parser_policy_changed is False


def test_current_content_signature_rejects_noncanonical_64_character_legacy_value() -> None:
    source = SimpleNamespace(
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='G' * 64,
    )

    assert current_content_signature(source) is None
