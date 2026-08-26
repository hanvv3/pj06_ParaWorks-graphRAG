from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from backend.app.connectors.base import ConnectorManifest, SourceEvent
from backend.app.connectors.registry import (
    get_connector_manifest,
    list_connector_manifests,
)
from backend.app.ingestion import service as ingestion_service
from backend.app.ingestion import sync as ingestion_sync
from backend.app.ingestion.source_versions import SourceVersionRef
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import DocumentChunk, DocumentParserRun, Source, SyncJob


def source_event(source_id: str = 'contract-event-1') -> SourceEvent:
    return SourceEvent(
        source_type='slack',
        source_id=source_id,
        source_url=f'https://slack.mock/{source_id}',
        title='Contract event',
        body='Redis incident timeline should enter company memory.',
        author='u123',
        participants=['u123'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={
            'channel_id': 'C123',
            'external_updated_at': '2026-05-01T09:00:00+00:00',
            'ts': '1777600800.000100',
        },
    )


def drive_source_event(
    *,
    version: str = '42',
    revision_id: str = 'rev-42',
    body: str = '휴가 신청은 HR 시스템에서 진행합니다.',
    parser_status: str = 'parsed',
    parser_status_reason: str | None = None,
) -> SourceEvent:
    return SourceEvent(
        source_type='drive',
        source_id='drive:file-1',
        source_url='https://drive.google.com/file/d/file-1/view',
        title='휴가 정책',
        body=body,
        author='owner@example.com',
        participants=['owner@example.com'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level='restricted',
        raw_metadata={
            'sync_partition': 'drive',
            'sync_cursor': '2026-05-01T09:00:00Z',
            'document_version': version,
            'revision_id': revision_id,
            'content_signature': f'drive:file-1:{version}:{revision_id}',
            'parser_name': 'google_drive_text_export',
            'parser_status': parser_status,
            'parser_status_reason': parser_status_reason,
            'source_snippet': body,
        },
    )


@dataclass(frozen=True)
class ContractConnector:
    source_type: str = 'slack'
    manifest: ConnectorManifest = ConnectorManifest(
        connector_type='slack',
        display_name='Slack',
        mode='mock',
        auth_type='oauth',
        required_scopes=('channels:history', 'groups:history', 'im:history', 'mpim:history'),
        sync_strategy='incremental',
        cost_policy='Fetch source deltas first; embed only changed chunks after review approval.',
    )

    def fetch_events(self) -> list[SourceEvent]:
        return [source_event()]


@dataclass
class IncrementalContractConnector:
    observed_cursor: dict[str, str] | None = None
    source_type: str = 'slack'
    manifest: ConnectorManifest = ConnectorManifest(
        connector_type='slack',
        display_name='Slack',
        mode='live',
        auth_type='oauth',
        required_scopes=('channels:history', 'groups:history', 'im:history', 'mpim:history'),
        sync_strategy='incremental',
        cost_policy='Fetch source deltas first; embed only changed chunks after review approval.',
    )

    def fetch_events(self) -> list[SourceEvent]:
        raise AssertionError('incremental connector should receive a sync cursor')

    def fetch_events_since(self, latest_timestamps_by_partition: dict[str, str]) -> list[SourceEvent]:
        self.observed_cursor = latest_timestamps_by_partition
        return [source_event('contract-event-2')]


@dataclass(frozen=True)
class FailingConnector:
    source_type: str = 'gmail'
    manifest: ConnectorManifest = ConnectorManifest(
        connector_type='gmail',
        display_name='Gmail',
        mode='live',
        auth_type='oauth',
        required_scopes=('gmail.readonly',),
        sync_strategy='incremental',
        cost_policy='Fetch message deltas before review extraction.',
    )

    def fetch_events(self) -> list[SourceEvent]:
        raise RuntimeError('oauth token expired')


@dataclass
class DriveContentSignatureConnector:
    events: list[SourceEvent]
    source_type: str = 'drive'
    manifest: ConnectorManifest = ConnectorManifest(
        connector_type='drive',
        display_name='Google Drive',
        mode='live',
        auth_type='oauth',
        required_scopes=('drive.readonly',),
        sync_strategy='incremental',
        cost_policy='Fetch Drive deltas before parser and embedding work.',
    )

    def fetch_events(self) -> list[SourceEvent]:
        return self.events


def test_connector_manifests_define_parallel_ingestion_contracts() -> None:
    manifests = {manifest.connector_type: manifest for manifest in list_connector_manifests()}

    assert manifests['slack'].auth_type == 'oauth'
    assert manifests['slack'].sync_strategy == 'incremental'
    assert 'channels:history' in manifests['slack'].required_scopes
    assert 'embed only changed' in manifests['slack'].cost_policy
    assert manifests['gmail'].auth_type == 'oauth'
    assert 'gmail.readonly' in manifests['gmail'].required_scopes
    assert get_connector_manifest('drive').display_name == 'Google Drive'


def test_live_gmail_manifest_reports_send_scope_for_approval_actions() -> None:
    manifests = {manifest.connector_type: manifest for manifest in list_connector_manifests(demo_mode=False)}

    assert 'https://www.googleapis.com/auth/gmail.readonly' in manifests['gmail'].required_scopes
    assert 'https://www.googleapis.com/auth/gmail.send' in manifests['gmail'].required_scopes


def test_sync_connector_events_records_job_and_changed_source_ids(db_session: Session) -> None:
    result = sync_connector_events(db=db_session, connector=ContractConnector())

    job = db_session.query(SyncJob).one()
    chunk = db_session.query(DocumentChunk).one()
    parser_run = db_session.query(DocumentParserRun).one()
    assert result.job_id == job.job_id
    assert result.connector_type == 'slack'
    assert result.status == 'complete'
    assert result.fetched_events == 1
    assert result.created_review_items == 0
    assert result.changed_source_ids == ['contract-event-1']
    assert result.skipped_events == 0
    assert job.status == 'complete'
    assert job.message == 'fetched=1 created_review_items=0 skipped_events=0'
    assert job.progress_pct == 100
    assert chunk.metadata_['source_id'] == 'contract-event-1'
    assert chunk.metadata_['source_type'] == 'slack'
    assert chunk.metadata_['permission_level'] == 'internal'
    assert chunk.metadata_['participants'] == ['u123']
    assert chunk.metadata_['channel_id'] == 'C123'
    assert chunk.metadata_['external_updated_at'] == '2026-05-01T09:00:00+00:00'
    assert chunk.metadata_['ts'] == '1777600800.000100'
    assert parser_run.source_id == chunk.source_id
    assert parser_run.document_version_id == chunk.version_id
    assert parser_run.parser_name == 'slack_source_event'
    assert parser_run.parser_status == 'parsed'
    assert parser_run.parser_status_reason is None
    assert parser_run.mime_type == 'slack'
    assert parser_run.document_version_label == 'v1'
    assert parser_run.content_signature == 'contract-event-1:v1'
    assert parser_run.chunk_count == 1
    assert parser_run.metadata_['source_id'] == 'contract-event-1'


def test_sync_returns_canonical_refs_after_ingestion_commit(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordering: list[str] = []
    real_commit = db_session.commit
    real_expire_all = db_session.expire_all
    real_scalars = db_session.scalars
    real_source_version_refs = ingestion_service.source_version_refs
    expired = False

    def tracked_commit() -> None:
        real_commit()
        ordering.append('commit')

    def tracked_expire_all() -> None:
        nonlocal expired
        real_expire_all()
        expired = True
        ordering.append('expire_all')

    def tracked_scalars(statement, *args, **kwargs):
        result = real_scalars(statement, *args, **kwargs)
        if expired and 'FROM sources' in str(statement):
            ordering.append('source_reload')
        return result

    def tracked_source_version_refs(sources):
        rows = list(sources)
        assert ordering[-2:] == ['expire_all', 'source_reload']
        assert all(inspect(source).persistent for source in rows)
        ordering.append('canonical_refs')
        return real_source_version_refs(rows)

    monkeypatch.setattr(db_session, 'commit', tracked_commit)
    monkeypatch.setattr(db_session, 'expire_all', tracked_expire_all)
    monkeypatch.setattr(db_session, 'scalars', tracked_scalars)
    monkeypatch.setattr(
        ingestion_service,
        'source_version_refs',
        tracked_source_version_refs,
    )

    result = sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([drive_source_event()]),
    )

    source = db_session.query(Source).one()
    assert source.id is not None
    assert result.changed_source_ids == ['drive:file-1']
    assert result.changed_source_refs == [
        SourceVersionRef(
            source_type='drive',
            source_id='drive:file-1',
            version_or_signature='drive:file-1:42:rev-42',
        )
    ]
    assert ordering.index('commit') < ordering.index('expire_all')
    assert ordering.index('source_reload') < ordering.index('canonical_refs')


def test_v2_sync_completion_rolls_back_job_identity_and_waterline_together(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingestion_sync,
        'with_review_batch_marker',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('marker failed')),
        raising=False,
    )

    with pytest.raises(RuntimeError, match='marker failed'):
        sync_connector_events(
            db=db_session,
            connector=DriveContentSignatureConnector([drive_source_event()]),
            review_batch_mode='v2_explicit',
        )

    db_session.expire_all()
    job = db_session.query(SyncJob).one()
    source = db_session.query(Source).one()
    assert job.status == 'failed'
    assert job.progress_pct == 100
    assert source.raw_metadata.get('last_changed_sync_job_id') is None
    assert source.raw_metadata.get('review_batch_mode') is None
    assert source.raw_metadata.get('review_batch_signature') is None


def test_sync_connector_events_reports_skipped_duplicates(db_session: Session) -> None:
    sync_connector_events(db=db_session, connector=ContractConnector())

    result = sync_connector_events(db=db_session, connector=ContractConnector())

    assert result.status == 'complete'
    assert result.fetched_events == 1
    assert result.created_review_items == 0
    assert result.changed_source_ids == []
    assert result.skipped_events == 1
    assert db_session.query(DocumentParserRun).count() == 1


def test_sync_connector_events_persists_parser_run_provenance(db_session: Session) -> None:
    result = sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector(
            [
                SourceEvent(
                    source_type='drive',
                    source_id='drive:parser-test',
                    source_url='https://drive.mock/parser-test',
                    title='Parser test document',
                    body='Google Drive file changed: Parser test document',
                    author='owner@example.com',
                    participants=['owner@example.com'],
                    timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
                    permission_level='restricted',
                    raw_metadata={
                        'mime_type': 'application/pdf',
                        'parser_name': 'google_drive_metadata',
                        'parser_status': 'metadata_only',
                        'parser_status_reason': 'pdf_parser_not_enabled',
                        'document_version': '42',
                        'revision_id': 'rev-42',
                        'content_signature': 'drive:parser-test:42:rev-42',
                        'source_snippet': 'Parser test document',
                    },
                )
            ]
        ),
    )

    parser_run = db_session.query(DocumentParserRun).one()
    assert result.fetched_events == 1
    assert result.changed_source_ids == ['drive:parser-test']
    assert parser_run.parser_name == 'google_drive_metadata'
    assert parser_run.parser_status == 'metadata_only'
    assert parser_run.parser_status_reason == 'pdf_parser_not_enabled'
    assert parser_run.mime_type == 'application/pdf'
    assert parser_run.document_version_label == '42'
    assert parser_run.revision_id == 'rev-42'
    assert parser_run.content_signature == 'drive:parser-test:42:rev-42'
    assert parser_run.chunk_count == 1
    assert parser_run.metadata_ == {
        'source_id': 'drive:parser-test',
        'source_url': 'https://drive.mock/parser-test',
        'permission_level': 'restricted',
        'source_snippet': 'Parser test document',
    }


def test_sync_connector_events_skips_same_content_signature(db_session: Session) -> None:
    sync_connector_events(db=db_session, connector=DriveContentSignatureConnector([drive_source_event()]))

    result = sync_connector_events(db=db_session, connector=DriveContentSignatureConnector([drive_source_event()]))

    assert result.status == 'complete'
    assert result.fetched_events == 1
    assert result.created_review_items == 0
    assert result.changed_source_ids == []
    assert result.skipped_events == 1
    assert db_session.query(DocumentChunk).count() == 1


def test_sync_connector_events_ingests_changed_content_signature(db_session: Session) -> None:
    sync_connector_events(db=db_session, connector=DriveContentSignatureConnector([drive_source_event()]))

    result = sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector(
            [
                drive_source_event(
                    version='43',
                    revision_id='rev-43',
                    body='휴가 신청 승인자가 인사팀으로 변경되었습니다.',
                )
            ]
        ),
    )

    assert result.status == 'complete'
    assert result.fetched_events == 1
    assert result.created_review_items == 0
    assert result.changed_source_ids == ['drive:file-1']
    assert result.skipped_events == 0
    chunks = db_session.query(DocumentChunk).order_by(DocumentChunk.id).all()
    assert len(chunks) == 2
    assert chunks[0].metadata_['content_signature'] == 'drive:file-1:42:rev-42'
    assert chunks[1].metadata_['content_signature'] == 'drive:file-1:43:rev-43'


def test_sync_connector_events_reports_parser_status_counts(db_session: Session) -> None:
    result = sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector(
            [
                drive_source_event(),
                drive_source_event(
                    version='43',
                    revision_id='rev-43',
                    body='PDF metadata only',
                    parser_status='metadata_only',
                    parser_status_reason='pdf_parser_not_enabled',
                ),
            ]
        ),
    )

    assert result.parser_status_counts == {
        'metadata_only': 1,
        'parsed': 1,
    }


def test_sync_connector_events_passes_latest_slack_timestamp_cursor(db_session: Session) -> None:
    sync_connector_events(db=db_session, connector=ContractConnector())

    connector = IncrementalContractConnector()
    result = sync_connector_events(db=db_session, connector=connector)

    assert connector.observed_cursor == {'C123': '1777600800.000100'}
    assert result.fetched_events == 1
    assert result.created_review_items == 0
    assert result.changed_source_ids == ['contract-event-2']


def test_sync_connector_events_passes_latest_generic_sync_cursor(db_session: Session) -> None:
    db_session.add(
        Source(
            source_type='gmail',
            source_id='gmail:older',
            source_url='https://mail.google.com/mail/u/0/#all/older',
            title='Older message',
            author='min@example.com',
            permission_level='internal',
            raw_metadata={'sync_partition': 'gmail', 'sync_cursor': '1777600800000'},
        )
    )
    db_session.add(
        Source(
            source_type='gmail',
            source_id='gmail:newer',
            source_url='https://mail.google.com/mail/u/0/#all/newer',
            title='Newer message',
            author='min@example.com',
            permission_level='internal',
            raw_metadata={'sync_partition': 'gmail', 'sync_cursor': '1777600900000'},
        )
    )
    db_session.commit()
    connector = IncrementalContractConnector(source_type='gmail')

    result = sync_connector_events(db=db_session, connector=connector)

    assert connector.observed_cursor == {'gmail': '1777600900000'}
    assert result.fetched_events == 1
    assert result.created_review_items == 0
    assert result.changed_source_ids == ['contract-event-2']


def test_sync_connector_events_marks_job_failed_on_connector_error(db_session: Session) -> None:
    with pytest.raises(RuntimeError, match='oauth token expired'):
        sync_connector_events(db=db_session, connector=FailingConnector())

    job = db_session.query(SyncJob).one()
    assert job.connector_type == 'gmail'
    assert job.status == 'failed'
    assert job.message == 'failed: oauth token expired'
    assert job.progress_pct == 100
