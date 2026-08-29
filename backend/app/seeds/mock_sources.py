from datetime import UTC, datetime

from backend.app.connectors.base import SourceEvent

SEED_EVENTS = [
    SourceEvent(
        source_type='slack',
        source_id='slack-project-alpha-redis-thread',
        source_url='https://slack.mock/project-alpha/redis-thread',
        title='Project Alpha Redis job status thread',
        body='Internal discussion: Redis keeps low latency job status available for workers, and Celery Redis remains the queue backend for Project Alpha.',
        author='maya@example.com',
        participants=['maya@example.com', 'noah@example.com'],
        timestamp=datetime(2026, 4, 30, 9, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={'scenario': 'project-alpha-redis-decision'},
    ),
    SourceEvent(
        source_type='gmail',
        source_id='gmail:project-alpha-redis-summary',
        source_url='https://gmail.mock/project-alpha/redis-summary',
        title='Project Alpha Redis summary',
        body='Redis should be used for transient job state while PostgreSQL remains the system of record for durable source evidence and review data.',
        author='noah@example.com',
        participants=['maya@example.com', 'noah@example.com', 'lee@example.com'],
        timestamp=datetime(2026, 4, 30, 10, 15, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'scenario': 'project-alpha-redis-decision',
            'message_id': 'project-alpha-redis-summary',
            'thread_id': 'thread-project-alpha-redis',
            'thread_context_key': (
                'thread-project-alpha-redis:project-alpha-redis-summary'
            ),
            'sync_partition': 'gmail',
            'sync_cursor': '1777544100000',
            'content_signature': (
                'gmail:project-alpha-redis-summary:1777544100000'
            ),
        },
        semantic_timestamp_raw='1777544100000',
    ),
    SourceEvent(
        source_type='gmail_attachment',
        source_id='gmail_attachment:project-alpha-redis-summary:att-budget-pdf',
        source_url='https://gmail.mock/project-alpha/redis-summary',
        title='Attachment: project-alpha-budget.pdf',
        body='Gmail attachment: project-alpha-budget.pdf\nParent subject: Project Alpha Redis summary\nMime type: application/pdf\nAttachment size: 2048',
        author='noah@example.com',
        participants=['maya@example.com', 'noah@example.com', 'lee@example.com'],
        timestamp=datetime(2026, 4, 30, 10, 15, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'scenario': 'project-alpha-redis-decision',
            'parent_source_id': 'gmail:project-alpha-redis-summary',
            'message_id': 'project-alpha-redis-summary',
            'thread_id': 'thread-project-alpha-redis',
            'thread_context_key': (
                'thread-project-alpha-redis:project-alpha-redis-summary:'
                'att-budget-pdf'
            ),
            'attachment_id': 'att-budget-pdf',
            'filename': 'project-alpha-budget.pdf',
            'mime_type': 'application/pdf',
            'attachment_size': 2048,
            'sync_partition': 'gmail',
            'sync_cursor': '1777544100000',
            'parser_name': 'gmail_attachment_metadata',
            'parser_status': 'metadata_only',
            'parser_status_reason': 'pdf_parser_not_enabled',
            'document_version': '1777544100000',
            'revision_id': 'att-budget-pdf',
            'content_signature': (
                'gmail_attachment:project-alpha-redis-summary:'
                'att-budget-pdf:2048'
            ),
            'source_snippet': 'Gmail attachment project-alpha-budget.pdf (application/pdf)',
        },
        semantic_timestamp_raw='1777544100000',
    ),
    SourceEvent(
        source_type='drive',
        source_id='drive:project-alpha-architecture-note',
        source_url='https://drive.mock/project-alpha/architecture-note',
        title='Project Alpha architecture note',
        body='Architecture note: reject database polling for worker progress and publish job status through Redis-backed updates instead.',
        author='lee@example.com',
        participants=['maya@example.com', 'lee@example.com'],
        timestamp=datetime(2026, 4, 30, 11, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'scenario': 'project-alpha-redis-decision',
            'file_id': 'project-alpha-architecture-note',
            'mime_type': 'application/vnd.google-apps.document',
            'modified_time': '2026-04-30T11:00:00Z',
            'document_version': '1',
            'revision_id': 'project-alpha-architecture-note-rev-1',
            'content_signature': (
                'drive:project-alpha-architecture-note:1:'
                'project-alpha-architecture-note-rev-1'
            ),
            'sync_partition': 'drive',
            'sync_cursor': '2026-04-30T11:00:00Z',
        },
        semantic_timestamp_raw='2026-04-30T11:00:00Z',
    ),
    SourceEvent(
        source_type='calendar',
        source_id='calendar:primary:project-beta-scope-meeting',
        source_url='https://calendar.mock/project-beta/scope-meeting',
        title='Project Beta scope meeting',
        body='Meeting notes: advanced document diff UI moves out of MVP scope; Review Queue and Source Evidence are required for launch readiness.',
        author='sara@example.com',
        participants=['sara@example.com', 'maya@example.com', 'noah@example.com'],
        timestamp=datetime(2026, 4, 30, 12, 30, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'scenario': 'project-beta-scope-cut',
            'event_id': 'project-beta-scope-meeting',
            'calendar_id': 'primary',
            'calendar_summary': 'Primary Calendar',
            'calendar_primary': True,
            'calendar_access_role': 'owner',
            'location': None,
            'start': '2026-04-30T12:30:00Z',
            'end': '2026-04-30T13:30:00Z',
            'event_start': '2026-04-30T12:30:00Z',
            'event_end': '2026-04-30T13:30:00Z',
            'event_status': 'confirmed',
            'organizer_email': 'sara@example.com',
            'creator_email': 'sara@example.com',
            'attendee_count': 3,
            'attendee_domains': ['example.com'],
            'external_domains': [],
            'has_external_attendees': False,
            'duration_minutes': 60,
            'content_signature': (
                'calendar:primary:project-beta-scope-meeting:'
                '2026-04-30T12:30:00Z'
            ),
            'sync_partition': 'calendar:primary',
            'sync_cursor': '2026-04-30T12:30:00Z',
        },
        semantic_timestamp_raw='2026-04-30T12:30:00Z',
    ),
    SourceEvent(
        source_type='slack',
        source_id='slack-project-beta-followup',
        source_url='https://slack.mock/project-beta/followup',
        title='Project Beta follow-up',
        body='Follow-up todo: confirm the MVP scope, complete evidence inspection before launch, and verify every Review Queue item links back to source evidence.',
        author='sara@example.com',
        participants=['sara@example.com', 'lee@example.com'],
        timestamp=datetime(2026, 4, 30, 13, 45, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={'scenario': 'project-beta-scope-cut'},
    ),
    SourceEvent(
        source_type='drive',
        source_id='drive:permission-leakage-case',
        source_url='https://drive.mock/permission-leakage-case',
        title='Restricted pricing note',
        body='Restricted note with confidential pricing. This must not appear for viewer users in the demo harness.',
        author='finance@example.com',
        participants=['finance@example.com', 'admin@example.com'],
        timestamp=datetime(2026, 4, 30, 14, 20, tzinfo=UTC),
        permission_level='restricted',
        raw_metadata={
            'scenario': 'permission-leakage-case',
            'file_id': 'permission-leakage-case',
            'mime_type': 'application/vnd.google-apps.document',
            'modified_time': '2026-04-30T14:20:00Z',
            'document_version': '1',
            'revision_id': 'permission-leakage-case-rev-1',
            'content_signature': (
                'drive:permission-leakage-case:1:'
                'permission-leakage-case-rev-1'
            ),
            'sync_partition': 'drive',
            'sync_cursor': '2026-04-30T14:20:00Z',
        },
        semantic_timestamp_raw='2026-04-30T14:20:00Z',
    ),
]
