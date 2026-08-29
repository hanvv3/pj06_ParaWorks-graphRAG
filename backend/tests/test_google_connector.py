from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy.orm import Session

from backend.app.connectors.google import (
    GOOGLE_CONNECTOR_SCOPES,
    GoogleApiError,
    GoogleConnector,
    GoogleConnectorConfig,
    GoogleWebApiClient,
)
from backend.app.core.config import Settings
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import Source


class FakeGoogleClient:
    def __init__(
        self,
        drive_files: list[dict] | None = None,
        drive_exports: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.gmail_after_internal_date: str | None = None
        self.drive_modified_after: str | None = None
        self.calendar_updated_min: str | None = None
        self.calendar_event_calls: list[dict[str, str | None]] = []
        self.drive_export_requests: list[tuple[str, str]] = []
        self._drive_files = drive_files
        self._drive_exports = drive_exports or {}

    def gmail_messages(self, *, after_internal_date: str | None = None) -> list[dict]:
        self.gmail_after_internal_date = after_internal_date
        return [
            {
                'id': 'msg-1',
                'threadId': 'thread-1',
                'labelIds': ['INBOX', 'IMPORTANT'],
                'snippet': '계약 검토 일정은 금요일까지 확정합니다.',
                'internalDate': '1777600800000',
                'payload': {
                    'mimeType': 'multipart/alternative',
                    'headers': [
                        {'name': 'Subject', 'value': '계약 검토 일정'},
                        {'name': 'From', 'value': 'min@example.com'},
                        {'name': 'To', 'value': 'para@example.com'},
                        {'name': 'Cc', 'value': 'partner@client.co.kr'},
                        {'name': 'Date', 'value': 'Fri, 1 May 2026 10:00:00 +0900'},
                    ],
                    'parts': [
                        {
                            'mimeType': 'text/html',
                            'body': {'data': 'PGI-SFRNTDw_Yj4'},
                        },
                        {
                            'mimeType': 'text/plain',
                            'body': {
                                'data': '6rOE7JW9IOqygO2GoCDsnbzsoJXsnYAg6riI7JqU7J286rmM7KeAIO2Zleygle2VqeuLiOuLpC4',
                            },
                        },
                    ]
                },
            }
        ]

    def drive_files(self, *, modified_after: str | None = None) -> list[dict]:
        self.drive_modified_after = modified_after
        if self._drive_files is not None:
            return self._drive_files
        return [
            {
                'id': 'file-1',
                'name': '사업계획서',
                'mimeType': 'application/vnd.google-apps.document',
                'description': '2026년 상반기 매출 목표와 채용 계획',
                'webViewLink': 'https://drive.google.com/file/d/file-1/view',
                'modifiedTime': '2026-05-01T09:00:00Z',
                'createdTime': '2026-04-30T09:00:00Z',
                'version': '42',
                'headRevisionId': 'rev-42',
                'owners': [{'emailAddress': 'owner@example.com'}],
                'lastModifyingUser': {'emailAddress': 'editor@example.com'},
            }
        ]

    def drive_file_text_export(self, *, file_id: str, export_mime_type: str) -> str:
        self.drive_export_requests.append((file_id, export_mime_type))
        if (file_id, export_mime_type) in self._drive_exports:
            return self._drive_exports[(file_id, export_mime_type)]
        return '휴가 신청은 HR 시스템에서 진행합니다.\n승인은 팀장이 검토합니다.'

    def calendar_list(self) -> list[dict]:
        return [
            {
                'id': 'primary',
                'summary': 'Primary Calendar',
                'primary': True,
                'accessRole': 'owner',
            }
        ]

    def calendar_events(
        self,
        *,
        calendar_id: str,
        time_min: str | None = None,
        time_max: str | None = None,
        updated_min: str | None = None,
    ) -> list[dict]:
        self.calendar_updated_min = updated_min
        self.calendar_event_calls.append(
            {
                'calendar_id': calendar_id,
                'time_min': time_min,
                'time_max': time_max,
                'updated_min': updated_min,
            }
        )
        return [
            {
                'id': 'event-1',
                'summary': 'PM 회의',
                'description': '런칭 일정 점검',
                'location': '회의실 A',
                'htmlLink': 'https://calendar.google.com/event?eid=event-1',
                'status': 'confirmed',
                'updated': '2026-05-01T10:00:00Z',
                'start': {'dateTime': '2026-05-02T09:00:00+09:00'},
                'end': {'dateTime': '2026-05-02T10:00:00+09:00'},
                'organizer': {'email': 'lead@example.com'},
                'creator': {'email': 'pm@example.com'},
                'attendees': [
                    {'email': 'pm@example.com', 'responseStatus': 'accepted'},
                    {'email': 'dev@example.com', 'responseStatus': 'needsAction'},
                    {'email': 'client@customer.co.kr', 'responseStatus': 'declined'},
                ],
                'recurringEventId': 'series-1',
            }
        ]


def test_google_connector_maps_gmail_messages_to_source_events() -> None:
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='gmail',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=FakeGoogleClient(),
    )

    events = connector.fetch_events()

    assert len(events) == 1
    event = events[0]
    assert event.source_type == 'gmail'
    assert event.source_id == 'gmail:msg-1'
    assert event.source_url == 'https://mail.google.com/mail/u/?authuser=para%40example.com#all/thread-1'
    assert event.title == '계약 검토 일정'
    assert event.body == '계약 검토 일정\n\nFrom: min@example.com\nDate: Fri, 1 May 2026 10:00:00 +0900\n\n계약 검토 일정은 금요일까지 확정합니다.'
    assert event.author == 'min@example.com'
    assert event.participants == ['min@example.com', 'para@example.com', 'partner@client.co.kr']
    assert event.timestamp == datetime.fromtimestamp(1777600800, tz=UTC)
    assert event.permission_level == 'internal'
    assert event.raw_metadata['required_scopes'] == list(GOOGLE_CONNECTOR_SCOPES['gmail'])
    assert event.raw_metadata['sync_partition'] == 'gmail'
    assert event.raw_metadata['sync_cursor'] == '1777600800000'
    assert event.raw_metadata['content_signature'] == 'gmail:msg-1:1777600800000'
    assert event.raw_metadata['thread_id'] == 'thread-1'
    assert event.raw_metadata['thread_context_key'] == 'thread-1:msg-1'
    assert event.raw_metadata['label_ids'] == ['INBOX', 'IMPORTANT']
    assert event.raw_metadata['from_domain'] == 'example.com'
    assert event.raw_metadata['participant_domains'] == ['client.co.kr', 'example.com']
    assert event.raw_metadata['external_domains'] == ['client.co.kr']
    assert event.raw_metadata['has_external_participants'] is True
    assert event.raw_metadata['body_source'] == 'payload'
    assert event.raw_metadata['body_truncated'] is False
    assert event.semantic_timestamp_raw == '1777600800000'


def test_google_sync_injects_authenticated_account_scope_and_security_context(
    db_session: Session,
) -> None:
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='gmail',
            oauth_token='fake-google-oauth-token',
            account_id='trusted-google-account',
            account_name='para@example.com',
        ),
        client=FakeGoogleClient(),
    )

    sync_connector_events(
        db=db_session,
        connector=connector,
        settings=Settings(agent_runtime_security_scope_id='trusted-scope'),
    )

    source = db_session.query(Source).one()
    assert source.raw_metadata['account_id'] == 'trusted-google-account'
    assert source.raw_metadata['required_scopes'] == list(
        GOOGLE_CONNECTOR_SCOPES['gmail']
    )
    assert source.raw_metadata['security_scope_id'] == 'trusted-scope'


def test_google_connector_decodes_encoded_gmail_sender_names() -> None:
    class EncodedHeaderGoogleClient(FakeGoogleClient):
        def gmail_messages(self, *, after_internal_date: str | None = None) -> list[dict]:
            self.gmail_after_internal_date = after_internal_date
            return [
                {
                    'id': 'msg-encoded-1',
                    'threadId': 'thread-encoded-1',
                    'labelIds': ['INBOX'],
                    'snippet': 'Encoded sender sample.',
                    'internalDate': '1777600800000',
                    'payload': {
                        'mimeType': 'text/plain',
                        'headers': [
                            {'name': 'Subject', 'value': 'Encoded sender'},
                            {'name': 'From', 'value': '=?utf-8?b?6rmA7ZiE7IiY?= <kim@example.com>'},
                            {'name': 'To', 'value': 'para@example.com'},
                            {'name': 'Date', 'value': 'Fri, 1 May 2026 10:00:00 +0900'},
                        ],
                        'body': {'data': 'RW5jb2RlZCBzZW5kZXIgc2FtcGxlLg'},
                    },
                }
            ]

    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='gmail',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=EncodedHeaderGoogleClient(),
    )

    event = connector.fetch_events()[0]

    assert event.author == '김현수 <kim@example.com>'
    assert event.participants == ['kim@example.com', 'para@example.com']
    assert 'From: 김현수 <kim@example.com>' in event.body
    assert event.raw_metadata['from_domain'] == 'example.com'


def test_google_connector_maps_gmail_attachments_to_source_events() -> None:
    class AttachmentGoogleClient(FakeGoogleClient):
        def gmail_messages(self, *, after_internal_date: str | None = None) -> list[dict]:
            self.gmail_after_internal_date = after_internal_date
            return [
                {
                    'id': 'msg-attach-1',
                    'threadId': 'thread-attach-1',
                    'labelIds': ['INBOX'],
                    'snippet': 'See attached proposal.',
                    'internalDate': '1777600800000',
                    'payload': {
                        'mimeType': 'multipart/mixed',
                        'headers': [
                            {'name': 'Subject', 'value': 'Proposal review'},
                            {'name': 'From', 'value': 'min@example.com'},
                            {'name': 'To', 'value': 'para@example.com'},
                        ],
                        'parts': [
                            {
                                'mimeType': 'text/plain',
                                'body': {'data': 'U2VlIGF0dGFjaGVkIHByb3Bvc2FsLg'},
                            },
                            {
                                'filename': 'proposal.pdf',
                                'mimeType': 'application/pdf',
                                'body': {'attachmentId': 'att-1', 'size': 2048},
                            },
                        ],
                    },
                }
            ]

    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='gmail',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=AttachmentGoogleClient(),
    )

    events = connector.fetch_events()

    assert [event.source_id for event in events] == [
        'gmail:msg-attach-1',
        'gmail_attachment:msg-attach-1:att-1',
    ]
    attachment = events[1]
    assert attachment.source_type == 'gmail_attachment'
    assert attachment.source_url == 'https://mail.google.com/mail/u/?authuser=para%40example.com#all/thread-attach-1'
    assert attachment.title == 'Attachment: proposal.pdf'
    assert attachment.body == (
        'Gmail attachment: proposal.pdf\n'
        'Parent subject: Proposal review\n'
        'Mime type: application/pdf\n'
        'Attachment size: 2048'
    )
    assert attachment.author == 'min@example.com'
    assert attachment.participants == ['min@example.com', 'para@example.com']
    assert attachment.permission_level == 'internal'
    assert attachment.raw_metadata['parent_source_id'] == 'gmail:msg-attach-1'
    assert attachment.raw_metadata['message_id'] == 'msg-attach-1'
    assert attachment.raw_metadata['thread_id'] == 'thread-attach-1'
    assert attachment.raw_metadata['attachment_id'] == 'att-1'
    assert attachment.raw_metadata['filename'] == 'proposal.pdf'
    assert attachment.raw_metadata['mime_type'] == 'application/pdf'
    assert attachment.raw_metadata['attachment_size'] == 2048
    assert attachment.raw_metadata['parser_name'] == 'gmail_attachment_metadata'
    assert attachment.raw_metadata['parser_status'] == 'parsed'
    assert attachment.raw_metadata['parser_status_reason'] == ''
    assert attachment.raw_metadata['document_version'] == '1777600800000'
    assert attachment.raw_metadata['content_signature'] == 'gmail_attachment:msg-attach-1:att-1:2048'
    assert attachment.semantic_timestamp_raw == '1777600800000'


def test_google_connector_maps_drive_files_to_source_events() -> None:
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=FakeGoogleClient(),
    )

    events = connector.fetch_events()

    assert len(events) == 1
    event = events[0]
    assert event.source_type == 'drive'
    assert event.source_id == 'drive:file-1'
    assert event.source_url == 'https://drive.google.com/file/d/file-1/view'
    assert event.title == '사업계획서'
    assert event.body == '휴가 신청은 HR 시스템에서 진행합니다.\n승인은 팀장이 검토합니다.'
    assert event.author == 'owner@example.com'
    assert event.timestamp == datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    assert event.raw_metadata['mime_type'] == 'application/vnd.google-apps.document'
    assert event.raw_metadata['sync_partition'] == 'drive'
    assert event.raw_metadata['sync_cursor'] == '2026-05-01T09:00:00Z'
    assert event.raw_metadata['description'] == '2026년 상반기 매출 목표와 채용 계획'
    assert event.raw_metadata['created_time'] == '2026-04-30T09:00:00Z'
    assert event.raw_metadata['last_modifying_user_email'] == 'editor@example.com'
    assert event.raw_metadata['parser_name'] == 'google_drive_text_export'
    assert event.raw_metadata['parser_status'] == 'parsed'
    assert event.raw_metadata['parser_status_reason'] is None
    assert event.raw_metadata['document_version'] == '42'
    assert event.raw_metadata['revision_id'] == 'rev-42'
    assert event.raw_metadata['content_signature'] == 'drive:file-1:42:rev-42'
    assert event.semantic_timestamp_raw == '2026-05-01T09:00:00Z'


def test_google_connector_exports_google_docs_text_into_drive_source_events() -> None:
    client = FakeGoogleClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    event = connector.fetch_events()[0]

    assert client.drive_export_requests == [('file-1', 'text/plain')]
    assert event.body == '휴가 신청은 HR 시스템에서 진행합니다.\n승인은 팀장이 검토합니다.'
    assert event.permission_level == 'restricted'
    assert event.raw_metadata['parser_name'] == 'google_drive_text_export'
    assert event.raw_metadata['parser_status'] == 'parsed'
    assert event.raw_metadata['parser_status_reason'] is None
    assert event.raw_metadata['source_snippet'] == '휴가 신청은 HR 시스템에서 진행합니다. 승인은 팀장이 검토합니다.'


def test_google_connector_exports_google_sheets_csv_into_drive_source_events() -> None:
    client = FakeGoogleClient(
        drive_files=[
            {
                'id': 'sheet-1',
                'name': '비용 정산표',
                'mimeType': 'application/vnd.google-apps.spreadsheet',
                'webViewLink': 'https://drive.google.com/file/d/sheet-1/view',
                'modifiedTime': '2026-05-01T09:00:00Z',
                'version': '7',
                'headRevisionId': 'sheet-rev-7',
                'owners': [{'emailAddress': 'owner@example.com'}],
            }
        ],
        drive_exports={
            ('sheet-1', 'text/csv'): '항목,금액\n출장비,120000\n식대,45000',
        },
    )
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    event = connector.fetch_events()[0]

    assert client.drive_export_requests == [('sheet-1', 'text/csv')]
    assert event.body == '항목,금액\n출장비,120000\n식대,45000'
    assert event.raw_metadata['parser_name'] == 'google_drive_sheets_csv_export'
    assert event.raw_metadata['parser_status'] == 'parsed'
    assert event.raw_metadata['parser_status_reason'] is None
    assert event.raw_metadata['document_version'] == '7'
    assert event.raw_metadata['revision_id'] == 'sheet-rev-7'
    assert event.raw_metadata['content_signature'] == 'drive:sheet-1:7:sheet-rev-7'
    assert event.raw_metadata['source_snippet'] == '항목,금액 출장비,120000 식대,45000'


def test_google_connector_exports_google_slides_text_into_drive_source_events() -> None:
    client = FakeGoogleClient(
        drive_files=[
            {
                'id': 'slides-1',
                'name': 'Customer proposal deck',
                'mimeType': 'application/vnd.google-apps.presentation',
                'webViewLink': 'https://drive.google.com/file/d/slides-1/view',
                'modifiedTime': '2026-05-01T09:00:00Z',
                'version': '8',
                'headRevisionId': 'slides-rev-8',
                'owners': [{'emailAddress': 'owner@example.com'}],
            }
        ],
        drive_exports={
            ('slides-1', 'text/plain'): 'Slide 1\nCustomer rollout plan\nSlide 2\nApproval evidence required',
        },
    )
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    event = connector.fetch_events()[0]

    assert client.drive_export_requests == [('slides-1', 'text/plain')]
    assert event.body == 'Slide 1\nCustomer rollout plan\nSlide 2\nApproval evidence required'
    assert event.raw_metadata['parser_name'] == 'google_drive_slides_text_export'
    assert event.raw_metadata['parser_status'] == 'parsed'
    assert event.raw_metadata['parser_status_reason'] is None
    assert event.raw_metadata['document_version'] == '8'
    assert event.raw_metadata['revision_id'] == 'slides-rev-8'
    assert event.raw_metadata['content_signature'] == 'drive:slides-1:8:slides-rev-8'
    assert event.raw_metadata['source_snippet'] == 'Slide 1 Customer rollout plan Slide 2 Approval evidence required'


@pytest.mark.parametrize(
    ('mime_type', 'expected_status', 'expected_reason'),
    [
        ('application/pdf', 'parsed', ''),
        (
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'parsed',
            '',
        ),
        ('application/x-hwp', 'unsupported', 'hwp_parser_not_decided'),
        ('application/haansofthwp', 'unsupported', 'hwp_parser_not_decided'),
        ('application/vnd.hancom.hwpx', 'unsupported', 'hwp_parser_not_decided'),
    ],
)
def test_google_connector_marks_drive_parser_status_by_mime_type(
    mime_type: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    client = FakeGoogleClient(
        drive_files=[
            {
                'id': 'file-typed',
                'name': '타입별 문서',
                'mimeType': mime_type,
                'webViewLink': 'https://drive.google.com/file/d/file-typed/view',
                'modifiedTime': '2026-05-01T09:00:00Z',
                'version': '42',
                'headRevisionId': 'rev-42',
                'owners': [{'emailAddress': 'owner@example.com'}],
            }
        ]
    )
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    event = connector.fetch_events()[0]

    assert client.drive_export_requests == []
    assert event.raw_metadata['parser_name'] == 'google_drive_metadata'
    assert event.raw_metadata['parser_status'] == expected_status
    assert event.raw_metadata['parser_status_reason'] == expected_reason
    assert event.raw_metadata['mime_type'] == mime_type
    assert event.raw_metadata['document_version'] == '42'
    assert event.raw_metadata['revision_id'] == 'rev-42'
    assert event.raw_metadata['content_signature'] == 'drive:file-typed:42:rev-42'
    assert 'Google Drive file changed: 타입별 문서' in event.body


def test_google_connector_fetches_gmail_events_since_latest_cursor() -> None:
    client = FakeGoogleClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='gmail',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    events = connector.fetch_events_since({'gmail': '1777600800000'})

    assert len(events) == 1
    assert client.gmail_after_internal_date == '1777600800000'


def test_google_connector_fetches_drive_events_since_latest_cursor() -> None:
    client = FakeGoogleClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    events = connector.fetch_events_since({'drive': '2026-05-01T09:00:00Z'})

    assert len(events) == 1
    assert client.drive_modified_after == '2026-05-01T09:00:00Z'


def test_google_connector_fetches_calendar_events_since_latest_cursor() -> None:
    client = FakeGoogleClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='calendar',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    events = connector.fetch_events_since({'calendar:primary': '2026-05-01T10:00:00Z'})

    assert len(events) == 1
    assert client.calendar_updated_min == '2026-05-01T10:00:00Z'
    assert client.calendar_event_calls == [
        {
            'calendar_id': 'primary',
            'time_min': None,
            'time_max': None,
            'updated_min': '2026-05-01T10:00:00Z',
        }
    ]


def test_google_connector_fetches_all_calendar_events_with_initial_window() -> None:
    class MultiCalendarClient(FakeGoogleClient):
        def calendar_list(self) -> list[dict]:
            return [
                {'id': 'primary', 'summary': 'Primary Calendar', 'primary': True, 'accessRole': 'owner'},
                {'id': 'team@example.com', 'summary': 'Team Calendar', 'accessRole': 'reader'},
            ]

    client = MultiCalendarClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='calendar',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    events = connector.fetch_events()

    assert [event.source_id for event in events] == [
        'calendar:primary:event-1',
        'calendar:team@example.com:event-1',
    ]
    assert [call['calendar_id'] for call in client.calendar_event_calls] == ['primary', 'team@example.com']
    assert all(call['updated_min'] is None for call in client.calendar_event_calls)
    assert all(call['time_min'] for call in client.calendar_event_calls)
    assert all(call['time_max'] for call in client.calendar_event_calls)


def test_google_connector_uses_calendar_partition_cursors_per_calendar() -> None:
    class MultiCalendarClient(FakeGoogleClient):
        def calendar_list(self) -> list[dict]:
            return [
                {'id': 'primary', 'summary': 'Primary Calendar', 'primary': True, 'accessRole': 'owner'},
                {'id': 'team@example.com', 'summary': 'Team Calendar', 'accessRole': 'reader'},
            ]

    client = MultiCalendarClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='calendar',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    connector.fetch_events_since(
        {
            'calendar:primary': '2026-05-01T10:00:00Z',
            'calendar:team@example.com': '2026-05-02T10:00:00Z',
        }
    )

    assert client.calendar_event_calls == [
        {
            'calendar_id': 'primary',
            'time_min': None,
            'time_max': None,
            'updated_min': '2026-05-01T10:00:00Z',
        },
        {
            'calendar_id': 'team@example.com',
            'time_min': None,
            'time_max': None,
            'updated_min': '2026-05-02T10:00:00Z',
        },
    ]


def test_google_connector_refetches_calendar_window_when_updated_min_is_too_old() -> None:
    class ExpiredCursorClient(FakeGoogleClient):
        def calendar_events(
            self,
            *,
            calendar_id: str,
            time_min: str | None = None,
            time_max: str | None = None,
            updated_min: str | None = None,
        ) -> list[dict]:
            self.calendar_event_calls.append(
                {
                    'calendar_id': calendar_id,
                    'time_min': time_min,
                    'time_max': time_max,
                    'updated_min': updated_min,
                }
            )
            if updated_min:
                raise GoogleApiError('The requested minimum modification time lies too far in the past.')
            return [
                {
                    'id': 'event-refetched',
                    'summary': 'Refetched event',
                    'updated': '2026-05-15T09:00:00Z',
                    'start': {'dateTime': '2026-05-15T15:00:00+09:00'},
                    'end': {'dateTime': '2026-05-15T16:00:00+09:00'},
                }
            ]

    client = ExpiredCursorClient()
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='calendar',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=client,
    )

    events = connector.fetch_events_since({'calendar:primary': '2026-04-10T17:52:54.663Z'})

    assert [event.source_id for event in events] == ['calendar:primary:event-refetched']
    assert client.calendar_event_calls[0] == {
        'calendar_id': 'primary',
        'time_min': None,
        'time_max': None,
        'updated_min': '2026-04-10T17:52:54.663Z',
    }
    assert client.calendar_event_calls[1]['calendar_id'] == 'primary'
    assert client.calendar_event_calls[1]['updated_min'] is None
    assert client.calendar_event_calls[1]['time_min'] is not None
    assert client.calendar_event_calls[1]['time_max'] is not None
    assert client.calendar_event_calls == [
        {
            'calendar_id': 'primary',
            'time_min': None,
            'time_max': None,
            'updated_min': '2026-04-10T17:52:54.663Z',
        },
        {
            'calendar_id': 'primary',
            'time_min': client.calendar_event_calls[1]['time_min'],
            'time_max': client.calendar_event_calls[1]['time_max'],
            'updated_min': None,
        },
    ]


def test_google_connector_maps_calendar_events_to_source_events() -> None:
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='calendar',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=FakeGoogleClient(),
    )

    events = connector.fetch_events()

    assert len(events) == 1
    event = events[0]
    assert event.source_type == 'calendar'
    assert event.source_id == 'calendar:primary:event-1'
    assert event.source_url == 'https://calendar.google.com/event?eid=event-1'
    assert event.title == 'PM 회의'
    assert event.body == (
        'PM 회의\n\nDescription: 런칭 일정 점검\nLocation: 회의실 A\n'
        'Start: 2026-05-02T09:00:00+09:00\nEnd: 2026-05-02T10:00:00+09:00'
    )
    assert event.author == 'pm@example.com'
    assert event.participants == ['pm@example.com', 'dev@example.com', 'client@customer.co.kr']
    assert event.timestamp == datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    assert event.raw_metadata['sync_partition'] == 'calendar:primary'
    assert event.raw_metadata['sync_cursor'] == '2026-05-01T10:00:00Z'
    assert event.raw_metadata['calendar_id'] == 'primary'
    assert event.raw_metadata['calendar_summary'] == 'Primary Calendar'
    assert event.raw_metadata['calendar_primary'] is True
    assert event.raw_metadata['calendar_access_role'] == 'owner'
    assert event.raw_metadata['content_signature'] == 'calendar:primary:event-1:2026-05-01T10:00:00Z'
    assert event.raw_metadata['event_start'] == '2026-05-02T09:00:00+09:00'
    assert event.raw_metadata['event_end'] == '2026-05-02T10:00:00+09:00'
    assert event.raw_metadata['location'] == '회의실 A'
    assert event.raw_metadata['attendee_count'] == 3
    assert event.raw_metadata['event_context_key'] == 'event-1:2026-05-01T10:00:00Z'
    assert event.raw_metadata['event_status'] == 'confirmed'
    assert event.raw_metadata['organizer_email'] == 'lead@example.com'
    assert event.raw_metadata['creator_email'] == 'pm@example.com'
    assert event.raw_metadata['recurring_event_id'] == 'series-1'
    assert event.raw_metadata['attendee_response_statuses'] == {
        'accepted': 1,
        'declined': 1,
        'needsAction': 1,
    }
    assert event.raw_metadata['attendee_domains'] == ['customer.co.kr', 'example.com']
    assert event.raw_metadata['external_domains'] == ['customer.co.kr']
    assert event.raw_metadata['has_external_attendees'] is True
    assert event.raw_metadata['duration_minutes'] == 60
    assert event.semantic_timestamp_raw == '2026-05-02T09:00:00+09:00'


@pytest.mark.parametrize('internal_date', [None, '', 'not-millis', 'NaN'])
def test_gmail_missing_or_malformed_internal_date_is_rejected_without_now_fallback(
    internal_date: object,
) -> None:
    class InvalidGmailTimestampClient(FakeGoogleClient):
        def gmail_messages(self, *, after_internal_date: str | None = None) -> list[dict]:
            message = super().gmail_messages(after_internal_date=after_internal_date)[0]
            message['internalDate'] = internal_date
            return [message]

    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='gmail',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=InvalidGmailTimestampClient(),
    )

    with pytest.raises(GoogleApiError, match='Gmail internalDate is missing or malformed'):
        connector.fetch_events()


@pytest.mark.parametrize(
    'modified_time',
    [None, '', 'not-a-timestamp', '2026-05-01T09:00:00'],
)
def test_drive_missing_or_malformed_modified_time_is_rejected_without_now_fallback(
    modified_time: object,
) -> None:
    file = FakeGoogleClient().drive_files()[0]
    file['modifiedTime'] = modified_time
    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='drive',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=FakeGoogleClient(drive_files=[file]),
    )

    with pytest.raises(GoogleApiError, match='Drive modifiedTime is missing or malformed'):
        connector.fetch_events()


def test_calendar_missing_updated_uses_deterministic_start_for_display_but_signature_uses_raw_start() -> None:
    class AllDayWithoutUpdatedClient(FakeGoogleClient):
        def calendar_events(
            self,
            *,
            calendar_id: str,
            time_min: str | None = None,
            time_max: str | None = None,
            updated_min: str | None = None,
        ) -> list[dict]:
            return [
                {
                    'id': 'all-day-1',
                    'summary': 'Company holiday',
                    'start': {'date': '2026-05-03'},
                    'end': {'date': '2026-05-04'},
                }
            ]

    connector = GoogleConnector(
        config=GoogleConnectorConfig(
            connector_type='calendar',
            oauth_token='google-oauth-token',
            account_id='google-user-1',
            account_name='para@example.com',
        ),
        client=AllDayWithoutUpdatedClient(),
    )

    event = connector.fetch_events()[0]

    assert event.timestamp == datetime(2026, 5, 3, 0, 0, tzinfo=UTC)
    assert event.semantic_timestamp_raw == '2026-05-03'
    assert event.raw_metadata['start'] == '2026-05-03'
    assert event.raw_metadata['sync_cursor'] == '2026-05-03'


def test_calendar_semantic_time_rejects_ambiguous_start_and_preserves_missing_end_as_null() -> None:
    class CalendarSemanticTimeClient(FakeGoogleClient):
        def __init__(self, event: dict) -> None:
            super().__init__()
            self.event = event

        def calendar_events(
            self,
            *,
            calendar_id: str,
            time_min: str | None = None,
            time_max: str | None = None,
            updated_min: str | None = None,
        ) -> list[dict]:
            return [self.event]

    config = GoogleConnectorConfig(
        connector_type='calendar',
        oauth_token='google-oauth-token',
        account_id='google-user-1',
        account_name='para@example.com',
    )
    ambiguous = {
        'id': 'ambiguous-time',
        'start': {
            'date': '2026-05-03',
            'dateTime': '2026-05-03T09:00:00+09:00',
        },
    }

    with pytest.raises(GoogleApiError, match='Calendar start is missing or malformed'):
        GoogleConnector(
            config=config,
            client=CalendarSemanticTimeClient(ambiguous),
        ).fetch_events()

    with pytest.raises(GoogleApiError, match='Calendar location is malformed'):
        GoogleConnector(
            config=config,
            client=CalendarSemanticTimeClient(
                {
                    'id': 'malformed-location',
                    'location': {'room': 'A'},
                    'start': {'date': '2026-05-03'},
                }
            ),
        ).fetch_events()

    event = GoogleConnector(
        config=config,
        client=CalendarSemanticTimeClient(
            {
                'id': 'missing-end',
                'updated': 'not-an-aware-timestamp',
                'start': {'date': '2026-05-03'},
            }
        ),
    ).fetch_events()[0]

    assert event.raw_metadata['end'] is None
    assert event.raw_metadata['event_end'] is None
    assert event.raw_metadata['sync_cursor'] == '2026-05-03'
    assert event.timestamp == datetime(2026, 5, 3, 0, 0, tzinfo=UTC)


def test_supported_google_adapter_populates_exact_semantic_timestamp_raw() -> None:
    gmail = GoogleConnector(
        config=GoogleConnectorConfig('gmail', 'token', 'account', 'para@example.com'),
        client=FakeGoogleClient(),
    ).fetch_events()[0]
    drive = GoogleConnector(
        config=GoogleConnectorConfig('drive', 'token', 'account', 'para@example.com'),
        client=FakeGoogleClient(),
    ).fetch_events()[0]
    calendar = GoogleConnector(
        config=GoogleConnectorConfig('calendar', 'token', 'account', 'para@example.com'),
        client=FakeGoogleClient(),
    ).fetch_events()[0]

    assert gmail.semantic_timestamp_raw == '1777600800000'
    assert drive.semantic_timestamp_raw == '2026-05-01T09:00:00Z'
    assert calendar.semantic_timestamp_raw == '2026-05-02T09:00:00+09:00'


def test_google_web_api_client_attaches_bearer_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers['authorization'] == 'Bearer google-oauth-token'
        if request.url.path == '/gmail/v1/users/me/messages/msg-1':
            return httpx.Response(200, json={'id': 'msg-1'})
        return httpx.Response(200, json={'messages': [{'id': 'msg-1'}]})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.gmail_messages() == [{'id': 'msg-1'}]
    assert requests[0].url.path == '/gmail/v1/users/me/messages'
    assert requests[1].url.path == '/gmail/v1/users/me/messages/msg-1'


def test_google_web_api_client_paginates_and_hydrates_gmail_messages() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/gmail/v1/users/me/messages' and request.url.params.get('pageToken') is None:
            return httpx.Response(200, json={'messages': [{'id': 'msg-1'}], 'nextPageToken': 'page-2'})
        if request.url.path == '/gmail/v1/users/me/messages' and request.url.params.get('pageToken') == 'page-2':
            return httpx.Response(200, json={'messages': [{'id': 'msg-2'}]})
        if request.url.path == '/gmail/v1/users/me/messages/msg-1':
            return httpx.Response(
                200,
                json={
                    'id': 'msg-1',
                    'snippet': 'first detail',
                    'payload': {'headers': [{'name': 'Subject', 'value': 'First'}]},
                },
            )
        if request.url.path == '/gmail/v1/users/me/messages/msg-2':
            return httpx.Response(
                200,
                json={
                    'id': 'msg-2',
                    'snippet': 'second detail',
                    'payload': {'headers': [{'name': 'Subject', 'value': 'Second'}]},
                },
            )
        raise AssertionError(f'unexpected request: {request.url}')

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_limit=1,
    )

    messages = client.gmail_messages()

    assert [message['id'] for message in messages] == ['msg-1', 'msg-2']
    assert messages[0]['payload']['headers'][0]['value'] == 'First'
    assert messages[1]['snippet'] == 'second detail'
    request_paths = [request.url.path for request in requests]
    assert request_paths.count('/gmail/v1/users/me/messages') == 2
    assert '/gmail/v1/users/me/messages/msg-1' in request_paths
    assert '/gmail/v1/users/me/messages/msg-2' in request_paths
    assert any(request.url.params.get('pageToken') == 'page-2' for request in requests)
    detail_request = next(request for request in requests if request.url.path == '/gmail/v1/users/me/messages/msg-1')
    assert detail_request.url.params['format'] == 'full'


def test_google_web_api_client_uses_business_focused_gmail_query_by_default() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={'messages': []})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.gmail_messages() == []

    list_request = requests[0]
    query = list_request.url.params['q']
    assert 'newer_than:90d' in query
    assert '-in:spam' in query
    assert '-in:trash' in query
    assert '-category:promotions' in query
    assert '-category:social' in query


def test_google_web_api_client_paginates_drive_files() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get('pageToken') is None:
            return httpx.Response(
                200,
                json={
                    'files': [{'id': 'file-1', 'name': 'First'}],
                    'nextPageToken': 'drive-page-2',
                },
            )
        return httpx.Response(200, json={'files': [{'id': 'file-2', 'name': 'Second'}]})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_limit=1,
    )

    files = client.drive_files()

    assert [file['id'] for file in files] == ['file-1', 'file-2']
    assert requests[0].url.path == '/drive/v3/files'
    assert 'description' in requests[0].url.params['fields']
    assert 'lastModifyingUser' in requests[0].url.params['fields']
    assert 'version' in requests[0].url.params['fields']
    assert 'headRevisionId' in requests[0].url.params['fields']
    assert requests[1].url.params['pageToken'] == 'drive-page-2'


def test_google_web_api_client_exports_drive_file_as_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text='휴가 신청은 HR 시스템에서 진행합니다.')

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    text = client.drive_file_text_export(file_id='file-1', export_mime_type='text/plain')

    assert text == '휴가 신청은 HR 시스템에서 진행합니다.'
    assert requests[0].url.path == '/drive/v3/files/file-1/export'
    assert requests[0].url.params['mimeType'] == 'text/plain'


def test_google_web_api_client_paginates_calendar_list() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get('pageToken') is None:
            return httpx.Response(
                200,
                json={
                    'items': [{'id': 'primary', 'summary': 'Primary'}],
                    'nextPageToken': 'calendar-page-2',
                },
            )
        return httpx.Response(200, json={'items': [{'id': 'team@example.com', 'summary': 'Team'}]})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_limit=1,
    )

    calendars = client.calendar_list()

    assert [calendar['id'] for calendar in calendars] == ['primary', 'team@example.com']
    assert requests[0].url.path == '/calendar/v3/users/me/calendarList'
    assert requests[1].url.params['pageToken'] == 'calendar-page-2'


def test_google_web_api_client_paginates_calendar_events_for_calendar_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get('pageToken') is None:
            return httpx.Response(
                200,
                json={
                    'items': [{'id': 'event-1', 'summary': 'First'}],
                    'nextPageToken': 'calendar-page-2',
                },
            )
        return httpx.Response(200, json={'items': [{'id': 'event-2', 'summary': 'Second'}]})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_limit=1,
    )

    events = client.calendar_events(calendar_id='team@example.com')

    assert [event['id'] for event in events] == ['event-1', 'event-2']
    assert requests[0].url.path == '/calendar/v3/calendars/team@example.com/events'
    assert requests[1].url.params['pageToken'] == 'calendar-page-2'


def test_google_web_api_client_sends_delta_params_for_gmail_drive_and_calendar() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/gmail/v1/users/me/messages':
            return httpx.Response(200, json={'messages': []})
        if request.url.path == '/drive/v3/files':
            return httpx.Response(200, json={'files': []})
        if request.url.path == '/calendar/v3/calendars/team@example.com/events':
            return httpx.Response(200, json={'items': []})
        raise AssertionError(f'unexpected request: {request.url}')

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.gmail_messages(after_internal_date='1777600800000') == []
    assert client.drive_files(modified_after='2026-05-01T09:00:00Z') == []
    assert client.calendar_events(
        calendar_id='team@example.com',
        time_min='2026-04-01T00:00:00Z',
        time_max='2026-11-01T00:00:00Z',
        updated_min='2026-05-01T10:00:00Z',
    ) == []

    gmail_request = next(request for request in requests if request.url.path == '/gmail/v1/users/me/messages')
    drive_request = next(request for request in requests if request.url.path == '/drive/v3/files')
    calendar_request = next(request for request in requests if request.url.path == '/calendar/v3/calendars/team@example.com/events')
    gmail_query = gmail_request.url.params['q']
    assert 'after:1777600800' in gmail_query
    assert '-in:spam' in gmail_query
    assert '-in:trash' in gmail_query
    assert '-category:promotions' in gmail_query
    assert '-category:social' in gmail_query
    assert drive_request.url.params['q'] == "modifiedTime > '2026-05-01T09:00:00Z'"
    assert calendar_request.url.params['updatedMin'] == '2026-05-01T10:00:00Z'
    assert calendar_request.url.params['timeMin'] == '2026-04-01T00:00:00Z'
    assert calendar_request.url.params['timeMax'] == '2026-11-01T00:00:00Z'


def test_google_web_api_client_retries_rate_limited_requests_with_retry_after() -> None:
    requests: list[httpx.Request] = []
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={'Retry-After': '2'}, json={'error': {'message': 'rate limit'}})
        return httpx.Response(200, json={'files': []})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep_calls.append,
    )

    assert client.drive_files() == []
    assert len(requests) == 2
    assert sleep_calls == [2.0]


def test_google_web_api_client_stops_retrying_rate_limited_requests_after_limit() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, json={'error': {'message': 'rate limit'}})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
        sleep=lambda seconds: None,
    )

    with pytest.raises(GoogleApiError, match='rate_limited'):
        client.drive_files()
    assert len(requests) == 2


def test_google_web_api_client_raises_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={'error': {'message': 'missing scope'}})

    client = GoogleWebApiClient(
        oauth_token='google-oauth-token',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(GoogleApiError, match='missing scope'):
        client.gmail_messages()
