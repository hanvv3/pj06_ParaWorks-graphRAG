import base64
import binascii
import html
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.header import decode_header, make_header
from email.utils import parseaddr
from typing import Protocol
from urllib.parse import quote

import httpx

from backend.app.connectors.base import ConnectorManifest, SourceEvent
from backend.app.documents.adapters import (
    DocxDocumentParser,
    PdfDocumentParser,
    TextDocumentParser,
)
from backend.app.documents.parsers import parser_adapter_decision_for_mime_type

GoogleQueryParams = dict[str, str | list[str]]
GOOGLE_TEXT_BODY_LIMIT = 4_000
GOOGLE_GMAIL_INITIAL_QUERY = 'newer_than:90d -in:spam -in:trash -category:social -category:promotions -category:forums'
GOOGLE_GMAIL_DELTA_EXCLUDES = '-in:spam -in:trash -category:social -category:promotions -category:forums'
GOOGLE_DRIVE_DOC_MIME_TYPE = 'application/vnd.google-apps.document'
GOOGLE_DRIVE_TEXT_EXPORT_MIME_TYPE = 'text/plain'
GOOGLE_DRIVE_SHEETS_MIME_TYPE = 'application/vnd.google-apps.spreadsheet'
GOOGLE_DRIVE_SHEETS_EXPORT_MIME_TYPE = 'text/csv'
GOOGLE_DRIVE_SLIDES_MIME_TYPE = 'application/vnd.google-apps.presentation'
GOOGLE_DRIVE_SLIDES_EXPORT_MIME_TYPE = 'text/plain'
GOOGLE_CALENDAR_INITIAL_PAST_DAYS = 30
GOOGLE_CALENDAR_INITIAL_FUTURE_DAYS = 180

GOOGLE_CONNECTOR_SCOPES: dict[str, tuple[str, ...]] = {
    'gmail': (
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
    ),
    'drive': ('https://www.googleapis.com/auth/drive.readonly',),
    'calendar': ('https://www.googleapis.com/auth/calendar.readonly',),
}
GOOGLE_CONNECTOR_TYPES = frozenset(GOOGLE_CONNECTOR_SCOPES)


class GoogleApiClient(Protocol):
    def gmail_messages(self, *, after_internal_date: str | None = None) -> list[dict]:
        raise NotImplementedError

    def gmail_attachment_download(self, *, message_id: str, attachment_id: str) -> bytes:
        raise NotImplementedError

    def drive_files(self, *, modified_after: str | None = None) -> list[dict]:
        raise NotImplementedError

    def drive_file_text_export(self, *, file_id: str, export_mime_type: str) -> str:
        raise NotImplementedError

    def drive_file_content_download(self, *, file_id: str) -> bytes:
        raise NotImplementedError

    def calendar_list(self) -> list[dict]:
        raise NotImplementedError

    def calendar_events(
        self,
        *,
        calendar_id: str,
        time_min: str | None = None,
        time_max: str | None = None,
        updated_min: str | None = None,
    ) -> list[dict]:
        raise NotImplementedError


class GoogleApiError(RuntimeError):
    pass


class GoogleWebApiClient:
    def __init__(
        self,
        *,
        oauth_token: str,
        http_client: httpx.Client | None = None,
        gmail_base_url: str = 'https://gmail.googleapis.com',
        drive_base_url: str = 'https://www.googleapis.com',
        calendar_base_url: str = 'https://www.googleapis.com',
        page_limit: int = 100,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        gmail_initial_query: str = GOOGLE_GMAIL_INITIAL_QUERY,
        gmail_delta_excludes: str = GOOGLE_GMAIL_DELTA_EXCLUDES,
    ) -> None:
        self.oauth_token = oauth_token
        self.http_client = http_client or httpx.Client(timeout=30.0)
        self.gmail_base_url = gmail_base_url.rstrip('/')
        self.drive_base_url = drive_base_url.rstrip('/')
        self.calendar_base_url = calendar_base_url.rstrip('/')
        self.page_limit = page_limit
        self.max_retries = max_retries
        self.sleep = sleep
        self.gmail_initial_query = gmail_initial_query
        self.gmail_delta_excludes = gmail_delta_excludes

    def gmail_messages(self, *, after_internal_date: str | None = None) -> list[dict]:
        messages: list[dict] = []
        params: GoogleQueryParams = {'maxResults': str(self.page_limit)}
        params['q'] = _gmail_query(
            after_internal_date=after_internal_date,
            initial_query=self.gmail_initial_query,
            delta_excludes=self.gmail_delta_excludes,
        )
        for message_ref in self._get_paged_items(
            f'{self.gmail_base_url}/gmail/v1/users/me/messages',
            item_key='messages',
            params=params,
        ):
            message_id = str(message_ref['id'])
            messages.append(
                self._get_json(
                    f'{self.gmail_base_url}/gmail/v1/users/me/messages/{message_id}',
                    params={'format': 'full'},
                )
            )
        return messages

    def gmail_attachment_download(self, *, message_id: str, attachment_id: str) -> bytes:
        payload = self._get_json(
            f'{self.gmail_base_url}/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}',
            params={},
        )
        data = payload.get('data') or ''
        import base64
        return base64.urlsafe_b64decode(data.encode('ascii') + b'=' * (-len(data) % 4))

    def drive_files(self, *, modified_after: str | None = None) -> list[dict]:
        params: GoogleQueryParams = {
            'pageSize': str(self.page_limit),
            'fields': (
                'nextPageToken,'
                'files(id,name,description,mimeType,webViewLink,createdTime,modifiedTime,owners,'
                'lastModifyingUser(emailAddress,displayName),version,headRevisionId)'
            ),
        }
        if modified_after:
            params['q'] = f"modifiedTime > '{modified_after}'"
        return self._get_paged_items(
            f'{self.drive_base_url}/drive/v3/files',
            item_key='files',
            params=params,
        )

    def drive_file_text_export(self, *, file_id: str, export_mime_type: str) -> str:
        return self._get_text(
            f'{self.drive_base_url}/drive/v3/files/{file_id}/export',
            params={'mimeType': export_mime_type},
        )

    def drive_file_content_download(self, *, file_id: str) -> bytes:
        return self._get_bytes(
            f'{self.drive_base_url}/drive/v3/files/{file_id}',
            params={'alt': 'media'},
        )

    def calendar_list(self) -> list[dict]:
        params: GoogleQueryParams = {
            'maxResults': str(self.page_limit),
            'fields': 'nextPageToken,items(id,summary,primary,accessRole,selected,hidden)',
        }
        return self._get_paged_items(
            f'{self.calendar_base_url}/calendar/v3/users/me/calendarList',
            item_key='items',
            params=params,
        )

    def calendar_events(
        self,
        *,
        calendar_id: str,
        time_min: str | None = None,
        time_max: str | None = None,
        updated_min: str | None = None,
    ) -> list[dict]:
        params: GoogleQueryParams = {
            'maxResults': str(self.page_limit),
            'singleEvents': 'true',
            'orderBy': 'updated',
        }
        if time_min:
            params['timeMin'] = time_min
        if time_max:
            params['timeMax'] = time_max
        if updated_min:
            params['updatedMin'] = updated_min
        escaped_calendar_id = quote(calendar_id, safe='')
        return self._get_paged_items(
            f'{self.calendar_base_url}/calendar/v3/calendars/{escaped_calendar_id}/events',
            item_key='items',
            params=params,
        )

    def _get_json(self, url: str, *, params: GoogleQueryParams) -> dict:
        for attempt in range(self.max_retries + 1):
            response = self.http_client.get(
                url,
                headers={'Authorization': f'Bearer {self.oauth_token}'},
                params=params,
            )
            if response.status_code < 400:
                return response.json()
            if not _should_retry(response.status_code):
                raise GoogleApiError(_google_error_message(response))
            if attempt >= self.max_retries:
                if response.status_code == 429:
                    raise GoogleApiError('Google API request failed: rate_limited')
                raise GoogleApiError(f'Google API request failed: http_{response.status_code}')
            self.sleep(_retry_after_seconds(response))
        raise GoogleApiError('Google API request failed')

    def _get_text(self, url: str, *, params: GoogleQueryParams) -> str:
        for attempt in range(self.max_retries + 1):
            response = self.http_client.get(
                url,
                headers={'Authorization': f'Bearer {self.oauth_token}'},
                params=params,
            )
            if response.status_code < 400:
                return response.text
            if not _should_retry(response.status_code):
                raise GoogleApiError(_google_error_message(response))
            if attempt >= self.max_retries:
                if response.status_code == 429:
                    raise GoogleApiError('Google API request failed: rate_limited')
                raise GoogleApiError(f'Google API request failed: http_{response.status_code}')
            self.sleep(_retry_after_seconds(response))
        raise GoogleApiError('Google API request failed')

    def _get_bytes(self, url: str, *, params: GoogleQueryParams) -> bytes:
        for attempt in range(self.max_retries + 1):
            response = self.http_client.get(
                url,
                headers={'Authorization': f'Bearer {self.oauth_token}'},
                params=params,
            )
            if response.status_code < 400:
                return response.content
            if not _should_retry(response.status_code):
                raise GoogleApiError(_google_error_message(response))
            if attempt >= self.max_retries:
                if response.status_code == 429:
                    raise GoogleApiError('Google API request failed: rate_limited')
                raise GoogleApiError(f'Google API request failed: http_{response.status_code}')
            self.sleep(_retry_after_seconds(response))
        raise GoogleApiError('Google API request failed')

    def _get_paged_items(self, url: str, *, item_key: str, params: GoogleQueryParams) -> list[dict]:
        items: list[dict] = []
        page_token: str | None = None
        while True:
            page_params = dict(params)
            if page_token:
                page_params['pageToken'] = page_token
            payload = self._get_json(url, params=page_params)
            items.extend(payload.get(item_key, []))
            page_token = payload.get('nextPageToken')
            if not page_token:
                return items


@dataclass(frozen=True)
class GoogleConnectorConfig:
    connector_type: str
    oauth_token: str
    account_id: str
    account_name: str


@dataclass(frozen=True)
class GoogleConnector:
    config: GoogleConnectorConfig
    client: GoogleApiClient

    @property
    def source_type(self) -> str:
        return self.config.connector_type

    @property
    def manifest(self) -> ConnectorManifest:
        return ConnectorManifest(
            connector_type=self.config.connector_type,
            display_name=_display_name(self.config.connector_type),
            mode='live',
            auth_type='oauth',
            required_scopes=GOOGLE_CONNECTOR_SCOPES[self.config.connector_type],
            sync_strategy='incremental',
            cost_policy='Fetch Google source deltas before agent or embedding work.',
        )

    def fetch_events(self) -> list[SourceEvent]:
        if self.config.connector_type == 'gmail':
            events: list[SourceEvent] = []
            for message in self.client.gmail_messages():
                events.extend(self._gmail_message_to_source_events(message))
            return events
        if self.config.connector_type == 'drive':
            return [self._drive_file_to_source_event(file) for file in self.client.drive_files()]
        if self.config.connector_type == 'calendar':
            return self._calendar_source_events(latest_cursors_by_partition=None)
        raise GoogleApiError(f'Unsupported Google connector: {self.config.connector_type}')

    def fetch_events_since(self, latest_cursors_by_partition: dict[str, str]) -> list[SourceEvent]:
        if self.config.connector_type == 'gmail':
            events: list[SourceEvent] = []
            for message in self.client.gmail_messages(
                after_internal_date=latest_cursors_by_partition.get('gmail')
            ):
                events.extend(self._gmail_message_to_source_events(message))
            return events
        if self.config.connector_type == 'drive':
            return [
                self._drive_file_to_source_event(file)
                for file in self.client.drive_files(modified_after=latest_cursors_by_partition.get('drive'))
            ]
        if self.config.connector_type == 'calendar':
            return self._calendar_source_events(latest_cursors_by_partition=latest_cursors_by_partition)
        return self.fetch_events()

    def _calendar_source_events(self, *, latest_cursors_by_partition: dict[str, str] | None) -> list[SourceEvent]:
        events: list[SourceEvent] = []
        calendars = self.client.calendar_list()
        time_min, time_max = _calendar_initial_window()
        for calendar in calendars:
            calendar_id = str(calendar.get('id') or '').strip()
            if not calendar_id:
                continue
            partition = f'calendar:{calendar_id}'
            updated_min = latest_cursors_by_partition.get(partition) if latest_cursors_by_partition else None
            if updated_min:
                try:
                    fetched = self.client.calendar_events(calendar_id=calendar_id, updated_min=updated_min)
                except GoogleApiError as exc:
                    if not _is_calendar_updated_min_too_old(exc):
                        raise
                    fetched = self.client.calendar_events(
                        calendar_id=calendar_id,
                        time_min=time_min,
                        time_max=time_max,
                    )
            else:
                fetched = self.client.calendar_events(
                    calendar_id=calendar_id,
                    time_min=time_min,
                    time_max=time_max,
                )
            events.extend(
                self._calendar_event_to_source_event(event, calendar=calendar)
                for event in fetched
            )
        return events

    def _gmail_message_to_source_events(self, message: dict) -> list[SourceEvent]:
        message_event = self._gmail_message_to_source_event(message)
        return [
            message_event,
            *self._gmail_attachment_source_events(message=message, parent_event=message_event),
        ]

    def _gmail_message_to_source_event(self, message: dict) -> SourceEvent:
        message_id = str(message['id'])
        internal_date = _required_google_millis(
            message.get('internalDate'),
            field_name='Gmail internalDate',
        )
        subject = _header_value(message, 'Subject') or f'Gmail message {message_id}'
        author = _header_value(message, 'From') or self.config.account_name
        to_header = _header_value(message, 'To') or ''
        cc_header = _header_value(message, 'Cc') or ''
        date_header = _header_value(message, 'Date')
        snippet = str(message.get('snippet') or '')
        extracted_body = _gmail_text_body(message)
        source_body = extracted_body or snippet
        body_text, body_truncated = _bounded_text(source_body)
        body_source = 'payload' if extracted_body else 'snippet'
        participants = _gmail_participants(author=author, to_header=to_header, cc_header=cc_header)
        domain_metadata = _gmail_domain_metadata(
            author=author,
            participants=participants,
            account_name=self.config.account_name,
        )
        thread_id = str(message.get('threadId') or '')
        header_lines = [f'From: {author}'] if author else []
        if date_header:
            header_lines.append(f'Date: {date_header}')
        return SourceEvent(
            source_type='gmail',
            source_id=f'gmail:{message_id}',
            source_url=_gmail_web_url(
                message_id=message_id,
                thread_id=thread_id,
                account_name=self.config.account_name,
            ),
            title=subject,
            body='\n\n'.join(part for part in [subject, '\n'.join(header_lines), body_text] if part).strip(),
            author=author,
            participants=participants,
            timestamp=_timestamp_from_google_millis(internal_date),
            permission_level='internal',
            raw_metadata={
                'message_id': message_id,
                'thread_id': thread_id or None,
                'thread_context_key': f'{thread_id}:{message_id}' if thread_id else message_id,
                'label_ids': message.get('labelIds') or [],
                'date_header': date_header,
                'account_id': self.config.account_id,
                'sync_partition': 'gmail',
                'sync_cursor': internal_date,
                'content_signature': f'gmail:{message_id}:{internal_date}',
                'body_source': body_source,
                'body_truncated': body_truncated,
                **domain_metadata,
                'required_scopes': list(GOOGLE_CONNECTOR_SCOPES['gmail']),
            },
            semantic_timestamp_raw=internal_date,
        )

    def _gmail_attachment_source_events(self, *, message: dict, parent_event: SourceEvent) -> list[SourceEvent]:
        events: list[SourceEvent] = []
        message_id = str(message['id'])
        thread_id = str(message.get('threadId') or '')
        internal_date = str(message.get('internalDate') or '')
        subject = _header_value(message, 'Subject') or f'Gmail message {message_id}'
        for attachment in _gmail_attachment_parts(message.get('payload') or {}):
            attachment_id = attachment['attachment_id']
            filename = attachment['filename']
            mime_type = attachment['mime_type']
            attachment_size = attachment['attachment_size']
            decision = parser_adapter_decision_for_mime_type(mime_type)
            source_id = f'gmail_attachment:{message_id}:{attachment_id}'
            body_lines = [
                f'Gmail attachment: {filename}',
                f'Parent subject: {subject}',
                f'Mime type: {mime_type}',
                f'Attachment size: {attachment_size}',
            ]
            if decision.parser_status == 'parsed':
                downloader = getattr(self.client, 'gmail_attachment_download', None)
                if downloader:
                    try:
                        payload = downloader(message_id=message_id, attachment_id=str(attachment_id))
                        if mime_type == 'application/pdf':
                            parsed = PdfDocumentParser().parse(payload, metadata={})
                            if parsed.parser_run.parser_status == 'parsed' and parsed.chunks:
                                body_lines.append('\n\n'.join(chunk.text for chunk in parsed.chunks))
                        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                            parsed = DocxDocumentParser().parse(payload, metadata={})
                            if parsed.parser_run.parser_status == 'parsed' and parsed.chunks:
                                body_lines.append('\n\n'.join(chunk.text for chunk in parsed.chunks))
                        elif mime_type in ('text/plain', 'text/markdown'):
                            parsed = TextDocumentParser().parse(payload, metadata={})
                            if parsed.parser_run.parser_status == 'parsed' and parsed.chunks:
                                body_lines.append('\n\n'.join(chunk.text for chunk in parsed.chunks))
                    except Exception:
                        decision = parser_adapter_decision_for_mime_type('application/octet-stream')  # Fallback
            events.append(
                SourceEvent(
                    source_type='gmail_attachment',
                    source_id=source_id,
                    source_url=parent_event.source_url,
                    title=f'Attachment: {filename}',
                    body='\n'.join(body_lines),
                    author=parent_event.author,
                    participants=parent_event.participants,
                    timestamp=parent_event.timestamp,
                    permission_level=parent_event.permission_level,
                    raw_metadata={
                        'parent_source_id': parent_event.source_id,
                        'message_id': message_id,
                        'thread_id': thread_id or None,
                        'thread_context_key': f'{thread_id}:{message_id}:{attachment_id}'
                        if thread_id
                        else f'{message_id}:{attachment_id}',
                        'attachment_id': attachment_id,
                        'filename': filename,
                        'mime_type': mime_type,
                        'attachment_size': attachment_size,
                        'account_id': self.config.account_id,
                        'sync_partition': 'gmail',
                        'sync_cursor': internal_date,
                        'parser_name': 'gmail_attachment_metadata',
                        'parser_status': decision.parser_status,
                        'parser_status_reason': decision.parser_status_reason,
                        'document_version': internal_date or attachment_id,
                        'revision_id': attachment_id,
                        'content_signature': f'{source_id}:{attachment_size}',
                        'source_snippet': f'Gmail attachment {filename} ({mime_type})',
                        'required_scopes': list(GOOGLE_CONNECTOR_SCOPES['gmail']),
                    },
                    semantic_timestamp_raw=parent_event.semantic_timestamp_raw,
                )
            )
        return events

    def _drive_file_to_source_event(self, file: dict) -> SourceEvent:
        file_id = str(file['id'])
        title = str(file.get('name') or f'Drive file {file_id}')
        author = _first_owner_email(file) or self.config.account_name
        modified_time = _required_aware_iso(
            file.get('modifiedTime'),
            field_name='Drive modifiedTime',
        )
        description = str(file.get('description') or '')
        last_modifying_user_email = str((file.get('lastModifyingUser') or {}).get('emailAddress') or '')
        export_result = _drive_exported_text(client=self.client, file=file)
        parser_metadata = _drive_parser_metadata(
            file_id=file_id,
            file=file,
            modified_time=modified_time,
            exported_text=export_result[0] if export_result else None,
            export_parser_name=export_result[1] if export_result else None,
        )
        metadata_body_lines = [
            f'Google Drive file changed: {title}',
            f'Mime type: {file.get("mimeType")}' if file.get('mimeType') else '',
            f'Description: {description}' if description else '',
            f'Owner: {author}' if author else '',
            f'Last modifier: {last_modifying_user_email}' if last_modifying_user_email else '',
            f'Modified: {modified_time}' if modified_time else '',
        ]
        body = export_result[0] if export_result else '\n'.join(line for line in metadata_body_lines if line)
        return SourceEvent(
            source_type='drive',
            source_id=f'drive:{file_id}',
            source_url=str(file.get('webViewLink') or f'https://drive.google.com/file/d/{file_id}/view'),
            title=title,
            body=body,
            author=author,
            participants=[author] if author else [],
            timestamp=_timestamp_from_iso(modified_time),
            permission_level='restricted',
            raw_metadata={
                'file_id': file_id,
                'mime_type': file.get('mimeType'),
                'description': description,
                'created_time': file.get('createdTime'),
                'modified_time': modified_time,
                'last_modifying_user_email': last_modifying_user_email,
                'account_id': self.config.account_id,
                'sync_partition': 'drive',
                'sync_cursor': modified_time,
                **parser_metadata,
                'required_scopes': list(GOOGLE_CONNECTOR_SCOPES['drive']),
            },
            semantic_timestamp_raw=modified_time,
        )

    def _calendar_event_to_source_event(self, event: dict, *, calendar: dict) -> SourceEvent:
        event_id = str(event['id'])
        calendar_id = str(calendar.get('id') or 'primary')
        source_id = f'calendar:{calendar_id}:{event_id}'
        title = str(event.get('summary') or f'Calendar event {event_id}')
        author = str((event.get('creator') or {}).get('email') or self.config.account_name)
        updated = _optional_aware_iso(event.get('updated'))
        description = str(event.get('description') or '')
        location = _optional_string_value(
            event.get('location'),
            field_name='Calendar location',
        )
        start = _calendar_time_value(
            event.get('start'),
            field_name='Calendar start',
            required=True,
        )
        end = _calendar_time_value(
            event.get('end'),
            field_name='Calendar end',
            required=False,
        )
        display_timestamp = _calendar_display_timestamp(
            updated=updated,
            start=start,
        )
        participants = [
            str(attendee['email'])
            for attendee in event.get('attendees', [])
            if attendee.get('email')
        ]
        if author and author not in participants:
            participants.insert(0, author)
        calendar_metadata = _calendar_quality_metadata(
            event_id=event_id,
            event=event,
            participants=participants,
            account_name=self.config.account_name,
            updated=updated,
            start=start,
            end=end,
        )
        body_lines = [
            title,
            '',
            f'Description: {description}' if description else '',
            f'Location: {location}' if location else '',
            f'Start: {start}' if start else '',
            f'End: {end}' if end else '',
        ]
        return SourceEvent(
            source_type='calendar',
            source_id=source_id,
            source_url=str(event.get('htmlLink') or 'https://calendar.google.com'),
            title=title,
            body='\n'.join(line for line in body_lines if line or line == '').strip(),
            author=author,
            participants=participants,
            timestamp=display_timestamp,
            permission_level='internal',
            raw_metadata={
                'event_id': event_id,
                'calendar_id': calendar_id,
                'calendar_summary': str(calendar.get('summary') or calendar_id),
                'calendar_primary': bool(calendar.get('primary')),
                'calendar_access_role': str(calendar.get('accessRole') or ''),
                'location': location,
                'start': start,
                'end': end,
                'event_start': start,
                'event_end': end,
                'attendee_count': len(participants),
                'content_signature': f'{source_id}:{updated or start or end}',
                **calendar_metadata,
                'account_id': self.config.account_id,
                'sync_partition': f'calendar:{calendar_id}',
                'sync_cursor': updated or start,
                'required_scopes': list(GOOGLE_CONNECTOR_SCOPES['calendar']),
            },
            semantic_timestamp_raw=start,
        )


def _google_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f'Google API request failed with {response.status_code}'
    error = payload.get('error') or {}
    if isinstance(error, dict):
        return str(error.get('message') or f'Google API request failed with {response.status_code}')
    return str(error)


def _gmail_query(
    *,
    after_internal_date: str | None,
    initial_query: str,
    delta_excludes: str,
) -> str:
    if after_internal_date:
        return f'{_gmail_after_query(after_internal_date)} {delta_excludes}'.strip()
    return initial_query


def _gmail_after_query(after_internal_date: str) -> str:
    return f'after:{int(int(after_internal_date) / 1000)}'


def _should_retry(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _retry_after_seconds(response: httpx.Response) -> float:
    retry_after = response.headers.get('Retry-After')
    if not retry_after:
        return 1.0
    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        return 1.0


def _display_name(connector_type: str) -> str:
    if connector_type == 'gmail':
        return 'Gmail'
    if connector_type == 'drive':
        return 'Google Drive'
    if connector_type == 'calendar':
        return 'Google Calendar'
    return connector_type


def _calendar_initial_window() -> tuple[str, str]:
    now = datetime.now(UTC)
    return (
        _rfc3339_utc(now - timedelta(days=GOOGLE_CALENDAR_INITIAL_PAST_DAYS)),
        _rfc3339_utc(now + timedelta(days=GOOGLE_CALENDAR_INITIAL_FUTURE_DAYS)),
    )


def _is_calendar_updated_min_too_old(exc: GoogleApiError) -> bool:
    return 'requested minimum modification time lies too far in the past' in str(exc).lower()


def _rfc3339_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _header_value(message: dict, name: str) -> str | None:
    headers = (message.get('payload') or {}).get('headers') or []
    for header in headers:
        if str(header.get('name', '')).lower() == name.lower():
            return _decode_header_value(str(header.get('value') or ''))
    return None


def _decode_header_value(value: str) -> str:
    value = value.strip()
    if not value:
        return ''
    try:
        decoded = str(make_header(decode_header(value)))
    except (LookupError, UnicodeDecodeError, ValueError):
        return value
    display_name, address = parseaddr(decoded)
    if address and display_name:
        return f'{display_name} <{address}>'
    if address and decoded.strip() == address:
        return address
    return decoded


def _gmail_web_url(*, message_id: str, thread_id: str, account_name: str) -> str:
    target = thread_id or message_id
    account_addresses = _email_addresses(account_name)
    if account_addresses:
        return f'https://mail.google.com/mail/u/?authuser={quote(account_addresses[0], safe="")}#all/{target}'
    return f'https://mail.google.com/mail/u/0/#all/{target}'


def _gmail_participants(*, author: str | None, to_header: str, cc_header: str) -> list[str]:
    addresses: list[str] = []
    for value in [author or '', to_header, cc_header]:
        for address in _email_addresses(value):
            if address not in addresses:
                addresses.append(address)
    return addresses


def _gmail_domain_metadata(*, author: str | None, participants: list[str], account_name: str) -> dict[str, object]:
    account_domain = _email_domain(account_name)
    participant_domains = sorted({
        domain
        for participant in participants
        if (domain := _email_domain(participant))
    })
    external_domains = [domain for domain in participant_domains if account_domain and domain != account_domain]
    return {
        'from_domain': _email_domain(author or ''),
        'participant_domains': participant_domains,
        'external_domains': external_domains,
        'has_external_participants': bool(external_domains),
    }


def _email_addresses(value: str) -> list[str]:
    return [match.group(0).lower() for match in re.finditer(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', value)]


def _email_domain(value: str) -> str:
    addresses = _email_addresses(value)
    if not addresses:
        return ''
    return addresses[0].split('@', 1)[1]


def _drive_parser_metadata(
    *,
    file_id: str,
    file: dict,
    modified_time: str,
    exported_text: str | None,
    export_parser_name: str | None,
) -> dict[str, str | None]:
    version = str(file.get('version') or '')
    revision_id = str(file.get('headRevisionId') or '')
    document_version = version or revision_id or modified_time
    signature_parts = [part for part in [f'drive:{file_id}', document_version, revision_id] if part]
    if exported_text:
        snippet, _ = _bounded_text(exported_text)
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        return {
            'parser_name': export_parser_name or 'google_drive_text_export',
            'parser_status': 'parsed',
            'parser_status_reason': None,
            'document_version': document_version,
            'revision_id': revision_id,
            'content_signature': ':'.join(signature_parts),
            'source_snippet': snippet,
        }
    parser_status, parser_status_reason = _drive_parser_status_for_mime_type(str(file.get('mimeType') or ''))
    return {
        'parser_name': 'google_drive_metadata',
        'parser_status': parser_status,
        'parser_status_reason': parser_status_reason,
        'document_version': document_version,
        'revision_id': revision_id,
        'content_signature': ':'.join(signature_parts),
    }


def _drive_parser_status_for_mime_type(mime_type: str) -> tuple[str, str]:
    if mime_type == GOOGLE_DRIVE_DOC_MIME_TYPE:
        return ('metadata_only', 'google_docs_export_not_available')
    if mime_type == GOOGLE_DRIVE_SHEETS_MIME_TYPE:
        return ('metadata_only', 'sheets_export_not_enabled')
    if mime_type == GOOGLE_DRIVE_SLIDES_MIME_TYPE:
        return ('metadata_only', 'slides_export_not_enabled')
    decision = parser_adapter_decision_for_mime_type(mime_type)
    return (decision.parser_status, decision.parser_status_reason)


def _drive_exported_text(*, client: GoogleApiClient, file: dict) -> tuple[str, str] | None:
    mime_type = file.get('mimeType')
    if mime_type == GOOGLE_DRIVE_DOC_MIME_TYPE:
        export_mime_type = GOOGLE_DRIVE_TEXT_EXPORT_MIME_TYPE
        parser_name = 'google_drive_text_export'
    elif mime_type == GOOGLE_DRIVE_SHEETS_MIME_TYPE:
        export_mime_type = GOOGLE_DRIVE_SHEETS_EXPORT_MIME_TYPE
        parser_name = 'google_drive_sheets_csv_export'
    elif mime_type == GOOGLE_DRIVE_SLIDES_MIME_TYPE:
        export_mime_type = GOOGLE_DRIVE_SLIDES_EXPORT_MIME_TYPE
        parser_name = 'google_drive_slides_text_export'
    elif mime_type == 'application/pdf':
        downloader = getattr(client, 'drive_file_content_download', None)
        if not downloader:
            return None
        payload = downloader(file_id=str(file['id']))
        parsed = PdfDocumentParser().parse(payload, metadata={})
        if parsed.parser_run.parser_status == 'parsed' and parsed.chunks:
            return ('\n\n'.join(chunk.text for chunk in parsed.chunks), 'pypdf')
        return None
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        downloader = getattr(client, 'drive_file_content_download', None)
        if not downloader:
            return None
        payload = downloader(file_id=str(file['id']))
        parsed = DocxDocumentParser().parse(payload, metadata={})
        if parsed.parser_run.parser_status == 'parsed' and parsed.chunks:
            return ('\n\n'.join(chunk.text for chunk in parsed.chunks), 'python-docx')
        return None
    elif mime_type in ('text/plain', 'text/markdown'):
        downloader = getattr(client, 'drive_file_content_download', None)
        if not downloader:
            return None
        payload = downloader(file_id=str(file['id']))
        parsed = TextDocumentParser().parse(payload, metadata={})
        if parsed.parser_run.parser_status == 'parsed' and parsed.chunks:
            return ('\n\n'.join(chunk.text for chunk in parsed.chunks), 'built-in-text')
        return None
    else:
        return None
    exporter = getattr(client, 'drive_file_text_export', None)
    if exporter is None:
        return None
    text = exporter(
        file_id=str(file['id']),
        export_mime_type=export_mime_type,
    ).strip()
    if not text:
        return None
    return (text, parser_name)


def _calendar_quality_metadata(
    *,
    event_id: str,
    event: dict,
    participants: list[str],
    account_name: str,
    updated: str,
    start: str,
    end: str | None,
) -> dict[str, object]:
    account_domain = _email_domain(account_name)
    attendee_domains = sorted({
        domain
        for participant in participants
        if (domain := _email_domain(participant))
    })
    external_domains = [domain for domain in attendee_domains if account_domain and domain != account_domain]
    organizer = event.get('organizer')
    if organizer is not None and not isinstance(organizer, dict):
        raise GoogleApiError('Calendar organizer is malformed')
    return {
        'event_context_key': f'{event_id}:{updated}' if updated else event_id,
        'event_status': _optional_string_value(
            event.get('status'),
            field_name='Calendar status',
        ),
        'organizer_email': _optional_string_value(
            (organizer or {}).get('email'),
            field_name='Calendar organizer email',
        ),
        'creator_email': str((event.get('creator') or {}).get('email') or ''),
        'recurring_event_id': str(event.get('recurringEventId') or ''),
        'attendee_response_statuses': _calendar_response_status_counts(event),
        'attendee_domains': attendee_domains,
        'external_domains': external_domains,
        'has_external_attendees': bool(external_domains),
        'duration_minutes': _calendar_duration_minutes(start, end),
    }


def _calendar_response_status_counts(event: dict) -> dict[str, int]:
    counts = Counter(
        str(attendee.get('responseStatus') or 'unknown')
        for attendee in event.get('attendees', [])
        if attendee.get('email')
    )
    return {status: counts[status] for status in sorted(counts)}


def _calendar_duration_minutes(start: str, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        started_at = _calendar_display_timestamp(updated='', start=start)
        ended_at = _calendar_display_timestamp(updated='', start=end)
    except GoogleApiError:
        return None
    return max(int((ended_at - started_at).total_seconds() / 60), 0)


def _gmail_text_body(message: dict) -> str:
    payload = message.get('payload') or {}
    candidates = _gmail_payload_text_candidates(payload)
    if not candidates:
        return ''
    plain_text = [text for mime_type, text in candidates if mime_type == 'text/plain' and text]
    if plain_text:
        return _normalize_text('\n\n'.join(plain_text))
    html_text = [text for mime_type, text in candidates if mime_type == 'text/html' and text]
    if html_text:
        return _normalize_text(_strip_html('\n\n'.join(html_text)))
    return _normalize_text(candidates[0][1])


def _gmail_payload_text_candidates(payload: dict) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    mime_type = str(payload.get('mimeType') or '')
    data = (payload.get('body') or {}).get('data')
    if isinstance(data, str) and mime_type.startswith('text/'):
        decoded = _decode_base64url(data)
        if decoded:
            candidates.append((mime_type, decoded))
    for part in payload.get('parts') or []:
        candidates.extend(_gmail_payload_text_candidates(part))
    return candidates


def _gmail_attachment_parts(payload: dict) -> list[dict[str, object]]:
    attachments: list[dict[str, object]] = []
    body = payload.get('body') or {}
    attachment_id = body.get('attachmentId')
    filename = str(payload.get('filename') or '').strip()
    if attachment_id and filename:
        attachments.append(
            {
                'attachment_id': str(attachment_id),
                'filename': filename,
                'mime_type': str(payload.get('mimeType') or 'application/octet-stream'),
                'attachment_size': int(body.get('size') or 0),
            }
        )
    for part in payload.get('parts') or []:
        attachments.extend(_gmail_attachment_parts(part))
    return attachments


def _decode_base64url(value: str) -> str:
    padded = value + '=' * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8', errors='replace')
    except (binascii.Error, ValueError):
        return ''


def _strip_html(value: str) -> str:
    without_tags = re.sub(r'<[^>]+>', ' ', value)
    return html.unescape(without_tags)


def _normalize_text(value: str) -> str:
    # Collapse multiple spaces but preserve paragraph breaks (newlines)
    lines = value.splitlines()
    normalized_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    # Remove empty lines that are more than 2 consecutive
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(normalized_lines)).strip()


def _bounded_text(value: str, limit: int = GOOGLE_TEXT_BODY_LIMIT) -> tuple[str, bool]:
    normalized = _normalize_text(value)
    if len(normalized) <= limit:
        return normalized, False
    return normalized[:limit].rstrip(), True


def _first_owner_email(file: dict) -> str | None:
    owners = file.get('owners') or []
    if not owners:
        return None
    return owners[0].get('emailAddress')


def _calendar_time_value(
    value: object,
    *,
    field_name: str,
    required: bool,
) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, dict):
        raise GoogleApiError(f'{field_name} is missing or malformed')
    present = [key for key in ('date', 'dateTime') if key in value]
    if len(present) != 1:
        raise GoogleApiError(f'{field_name} is missing or malformed')
    raw = value[present[0]]
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise GoogleApiError(f'{field_name} is missing or malformed')
    if present[0] == 'date':
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', raw):
            raise GoogleApiError(f'{field_name} is missing or malformed')
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise GoogleApiError(f'{field_name} is missing or malformed') from exc
        if parsed.date().isoformat() != raw:
            raise GoogleApiError(f'{field_name} is missing or malformed')
        return raw
    try:
        return _required_aware_iso(raw, field_name=field_name)
    except GoogleApiError as exc:
        raise GoogleApiError(f'{field_name} is missing or malformed') from exc


def _optional_aware_iso(value: object) -> str:
    if value is None or value == '':
        return ''
    try:
        return _required_aware_iso(value, field_name='Calendar updated')
    except GoogleApiError:
        return ''


def _optional_string_value(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GoogleApiError(f'{field_name} is malformed')
    return value


def _timestamp_from_google_millis(value: object) -> datetime:
    raw = _required_google_millis(value, field_name='Google millisecond timestamp')
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise GoogleApiError('Google millisecond timestamp is missing or malformed') from exc


def _timestamp_from_iso(value: object) -> datetime:
    raw = _required_aware_iso(value, field_name='Google ISO timestamp')
    return datetime.fromisoformat(raw.replace('Z', '+00:00')).astimezone(UTC)


def _required_google_millis(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise GoogleApiError(f'{field_name} is missing or malformed')
    try:
        datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise GoogleApiError(f'{field_name} is missing or malformed') from exc
    return value


def _required_aware_iso(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GoogleApiError(f'{field_name} is missing or malformed')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('timestamp is naive')
    except (TypeError, ValueError) as exc:
        raise GoogleApiError(f'{field_name} is missing or malformed') from exc
    return value


def _calendar_display_timestamp(*, updated: str, start: str) -> datetime:
    if updated:
        try:
            return _timestamp_from_iso(updated)
        except GoogleApiError:
            pass
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', start):
        try:
            return datetime.fromisoformat(start).replace(tzinfo=UTC)
        except ValueError as exc:
            raise GoogleApiError('Calendar start is missing or malformed') from exc
    try:
        return _timestamp_from_iso(start)
    except GoogleApiError as exc:
        raise GoogleApiError('Calendar start is missing or malformed') from exc
