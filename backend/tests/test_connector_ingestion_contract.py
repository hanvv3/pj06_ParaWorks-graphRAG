from dataclasses import dataclass, replace
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
from backend.app.ingestion import source_content_signature as source_signature
from backend.app.ingestion import sync as ingestion_sync
from backend.app.ingestion.source_versions import SourceVersionRef
from backend.app.ingestion.sync import sync_connector_events
from backend.app.models import (
    AgentRun,
    AutoReviewExtractionCall,
    AutoReviewValidation,
    AutoReviewValidationCall,
    Document,
    DocumentChunk,
    DocumentParserRun,
    ReviewItem,
    Source,
    SyncJob,
)
from backend.app.rag.embeddings import EmbeddingBatchResult
from backend.app.rag.indexing import (
    build_rag_index_documents,
    index_changed_vector_documents,
)


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
    source_id: str = 'drive:file-1',
    version: str = '42',
    revision_id: str = 'rev-42',
    body: str = '휴가 신청은 HR 시스템에서 진행합니다.',
    permission_level: str = 'restricted',
    parser_status: str = 'parsed',
    parser_status_reason: str | None = None,
) -> SourceEvent:
    return SourceEvent(
        source_type='drive',
        source_id=source_id,
        source_url=f'https://drive.google.com/file/d/{source_id.removeprefix("drive:")}/view',
        title='휴가 정책',
        body=body,
        author='owner@example.com',
        participants=['owner@example.com'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level=permission_level,
        raw_metadata={
            'sync_partition': 'drive',
            'sync_cursor': '2026-05-01T09:00:00Z',
            'document_version': version,
            'revision_id': revision_id,
            'content_signature': f'{source_id}:{version}:{revision_id}',
            'parser_name': 'google_drive_text_export',
            'parser_status': parser_status,
            'parser_status_reason': parser_status_reason,
            'source_snippet': body,
        },
        semantic_timestamp_raw='2026-05-01T09:00:00Z',
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
            version_or_signature=source.server_content_signature,
        )
    ]
    assert ordering.index('commit') < ordering.index('expire_all')
    assert ordering.index('source_reload') < ordering.index('canonical_refs')


def test_source_event_semantic_timestamp_is_additive_and_optional() -> None:
    event = source_event()

    assert event.semantic_timestamp_raw is None

    exact = SourceEvent(
        source_type='gmail',
        source_id='gmail:exact-time',
        source_url='https://mail.google.com/mail/u/0/#all/exact-time',
        title='Exact timestamp',
        body='body',
        author='owner@example.com',
        participants=['owner@example.com'],
        timestamp=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        permission_level='internal',
        raw_metadata={},
        semantic_timestamp_raw='1777600800000',
    )
    assert exact.semantic_timestamp_raw == '1777600800000'


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
                    semantic_timestamp_raw='2026-05-01T09:00:00Z',
                )
            ]
        ),
    )

    parser_run = db_session.query(DocumentParserRun).one()
    assert result.fetched_events == 1
    assert result.changed_source_ids == ['drive:parser-test']
    assert parser_run.parser_name == 'server_drive_source_event'
    assert parser_run.parser_status == 'parsed'
    assert parser_run.parser_status_reason is None
    assert parser_run.mime_type == 'application/pdf'
    assert parser_run.document_version_label == '42'
    assert parser_run.revision_id == 'rev-42'
    assert len(parser_run.content_signature) == 64
    assert parser_run.server_content_signature == parser_run.content_signature
    assert parser_run.server_content_signature_schema == 'server-source-content:v1'
    assert parser_run.parser_policy_version == 'server-source-parser-policy:v1'
    assert parser_run.parser_version == 'source-event-paragraph-parser:v1'
    assert parser_run.chunk_policy_version == 'paragraph-chunks:1200:v1'
    assert parser_run.chunk_count == 1
    assert parser_run.metadata_ == {
        'source_id': 'drive:parser-test',
        'source_url': 'https://drive.mock/parser-test',
        'permission_level': 'restricted',
        'source_snippet': 'Google Drive file changed: Parser test document',
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
    assert len(chunks[0].metadata_['content_signature']) == 64
    assert len(chunks[1].metadata_['content_signature']) == 64
    assert chunks[0].metadata_['content_signature'] != chunks[1].metadata_['content_signature']


def test_deferred_slack_repeated_event_keeps_legacy_dedupe_but_never_gains_c5_authority(
    db_session: Session,
) -> None:
    first = sync_connector_events(db=db_session, connector=ContractConnector())
    second = sync_connector_events(db=db_session, connector=ContractConnector())

    source = db_session.query(Source).one()
    document = db_session.query(Document).one()
    parser_run = db_session.query(DocumentParserRun).one()
    chunk = db_session.query(DocumentChunk).one()
    assert first.skipped_events == 0
    assert second.skipped_events == 1
    assert source.server_content_signature_schema is None
    assert source.server_content_signature is None
    assert document.current_document_version_id is None
    assert parser_run.parser_policy_version is None
    assert chunk.parser_run_id is None


def test_same_content_permission_only_event_is_not_skipped_and_does_not_reparse(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_event = drive_source_event()
    public_event = SourceEvent(
        **{
            **public_event.__dict__,
            'permission_level': 'public',
        }
    )
    internal_event = SourceEvent(
        **{
            **public_event.__dict__,
            'permission_level': 'internal',
        }
    )
    reconciled: list[tuple] = []

    class RecordingReconciliationService:
        def __init__(self, db, *, settings, vector_writer=None) -> None:
            self.db = db

        def reconcile(self, changed_states):
            states = tuple(changed_states)
            assert all(self.db.get(Source, state.source_id) is not None for state in states)
            reconciled.append(states)
            return object()

    monkeypatch.setattr(
        ingestion_sync,
        'AutoReviewSourceReconciliationService',
        RecordingReconciliationService,
        raising=False,
    )
    connector = DriveContentSignatureConnector([public_event])
    sync_connector_events(db=db_session, connector=connector)
    reconciled.clear()

    result = sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([internal_event]),
    )

    assert result.skipped_events == 0
    assert result.changed_source_ids == ['drive:file-1']
    assert db_session.query(DocumentParserRun).count() == 1
    assert len(reconciled) == 1
    assert reconciled[0][0].content_changed is False
    assert reconciled[0][0].permission_changed is True


def test_permission_change_is_visible_to_preflight_before_any_provider_call(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_event = SourceEvent(
        **{
            **drive_source_event().__dict__,
            'permission_level': 'public',
        }
    )
    restricted_event = SourceEvent(
        **{
            **public_event.__dict__,
            'permission_level': 'restricted',
        }
    )
    observed_permissions: list[str] = []

    class ProviderTripwireWriter:
        def upsert_with_embedding(self, document, embedding) -> None:
            raise AssertionError('permission preflight must happen before provider work')

        def delete_many(self, document_ids) -> int:
            return len(tuple(document_ids))

        def narrow_permissions(self, document_ids, permission_level: str) -> int:
            return len(tuple(document_ids))

    class RecordingReconciliationService:
        def __init__(self, db, *, settings, vector_writer=None) -> None:
            self.db = db

        def reconcile(self, changed_states):
            source_pk = tuple(changed_states)[0].source_id
            observed_permissions.append(self.db.get(Source, source_pk).permission_level)
            return object()

    monkeypatch.setattr(
        ingestion_sync,
        'AutoReviewSourceReconciliationService',
        RecordingReconciliationService,
    )
    writer = ProviderTripwireWriter()
    sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([public_event]),
        vector_writer=writer,
    )
    observed_permissions.clear()

    sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([restricted_event]),
        vector_writer=writer,
    )

    assert observed_permissions == ['restricted']


def test_policy_rechunk_reindex_embeds_only_changed_content_after_incremental_skip(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = drive_source_event(
        source_id='drive:file-1',
        body='First source original body.',
        permission_level='internal',
    )
    second = drive_source_event(
        source_id='drive:file-2',
        body='Second source remains unchanged.',
        permission_level='internal',
    )
    ingestion_service.ingest_events_with_result(db_session, [first, second])
    for event in (first, second):
        db_session.add(
            ReviewItem(
                item_type='history_event',
                payload={
                    'title': 'Approved source chunk',
                    'summary': 'Human-approved source for incremental indexing.',
                    'source_ids': [event.source_id],
                },
                source_links=[event.source_url],
                source_snippets=[event.body],
                confidence_score=0.9,
                permission_level='internal',
                status='approved',
            )
        )
    db_session.commit()

    class RecordingBatchEmbeddingModel:
        dimensions = 2

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def embed(self, text: str) -> list[float]:
            raise AssertionError('incremental indexing must batch provider calls')

        def embed_many(self, texts: list[str]) -> EmbeddingBatchResult:
            self.batches.append(list(texts))
            return EmbeddingBatchResult(
                embeddings=[[1.0, 2.0] for _ in texts],
                request_count=1 if texts else 0,
            )

    class RecordingIndexWriter:
        def __init__(self) -> None:
            self.document_ids: list[str] = []

        def upsert_with_embedding(self, document, embedding) -> None:
            self.document_ids.append(document.document_id)

    embedding = RecordingBatchEmbeddingModel()
    writer = RecordingIndexWriter()
    index_changed_vector_documents(
        db=db_session,
        documents=build_rag_index_documents(db_session),
        writer=writer,
        embedding_model=embedding,
        embedding_model_name='fake-embedding',
    )
    assert embedding.batches == [
        ['First source original body.', 'Second source remains unchanged.']
    ]
    embedding.batches.clear()
    writer.document_ids.clear()
    queued_jobs: list[str] = []
    reindex_results = []
    handoff_order: list[str] = []
    real_reconciliation_service = (
        ingestion_sync.AutoReviewSourceReconciliationService
    )

    class RecordingReconciliationService:
        def __init__(self, db, *, settings, vector_writer=None) -> None:
            self.delegate = real_reconciliation_service(
                db,
                settings=settings,
                vector_writer=vector_writer,
            )

        def reconcile(self, changed_states):
            handoff_order.append('reconciled')
            return self.delegate.reconcile(changed_states)

    monkeypatch.setattr(
        ingestion_sync,
        'AutoReviewSourceReconciliationService',
        RecordingReconciliationService,
    )

    def execute_fake_incremental_reindex(job_id: str) -> None:
        assert handoff_order[-1] == 'reconciled'
        handoff_order.append('enqueued')
        queued_jobs.append(job_id)
        queued_job = (
            db_session.query(SyncJob)
            .filter(SyncJob.job_id == job_id)
            .one()
        )
        assert queued_job.status == 'queued'
        reindex_results.append(
            index_changed_vector_documents(
                db=db_session,
                documents=build_rag_index_documents(db_session),
                writer=writer,
                embedding_model=embedding,
                embedding_model_name='fake-embedding',
            )
        )

    changed = replace(first, body='First source changed body.')
    sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([changed]),
        incremental_reindex_enqueuer=execute_fake_incremental_reindex,
    )

    assert len(queued_jobs) == 1
    assert embedding.batches == [['First source changed body.']]
    assert reindex_results[-1].indexed_count == 1
    assert reindex_results[-1].skipped_count == 1

    real_policy = source_signature.server_parser_policy_for_source
    monkeypatch.setattr(
        source_signature,
        'server_parser_policy_for_source',
        lambda source_type, *, mime_type=None: replace(
            real_policy(source_type, mime_type=mime_type),
            chunk_policy_version='paragraph-chunks:1200:v2',
        ),
    )
    sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([changed]),
        incremental_reindex_enqueuer=execute_fake_incremental_reindex,
    )

    assert len(queued_jobs) == 2
    assert embedding.batches == [
        ['First source changed body.'],
        ['First source changed body.'],
    ]
    assert db_session.query(ReviewItem).count() == 2
    assert db_session.query(AgentRun).count() == 0
    assert db_session.query(AutoReviewExtractionCall).count() == 0
    assert db_session.query(AutoReviewValidationCall).count() == 0
    assert db_session.query(AutoReviewValidation).count() == 0
    assert embedding.batches[-1] == ['First source changed body.']

    restricted = replace(changed, permission_level='restricted')
    sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([restricted]),
        incremental_reindex_enqueuer=execute_fake_incremental_reindex,
    )
    sync_connector_events(
        db=db_session,
        connector=DriveContentSignatureConnector([restricted]),
        incremental_reindex_enqueuer=execute_fake_incremental_reindex,
    )

    assert len(queued_jobs) == 2
    assert len(embedding.batches) == 2
    assert handoff_order == [
        'reconciled',
        'enqueued',
        'reconciled',
        'enqueued',
        'reconciled',
    ]


def test_unchanged_event_performs_zero_reconciliation(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciliation_calls: list[tuple] = []

    class RecordingReconciliationService:
        def __init__(self, db, *, settings, vector_writer=None) -> None:
            pass

        def reconcile(self, changed_states):
            reconciliation_calls.append(tuple(changed_states))
            return object()

    monkeypatch.setattr(
        ingestion_sync,
        'AutoReviewSourceReconciliationService',
        RecordingReconciliationService,
        raising=False,
    )
    connector = DriveContentSignatureConnector([drive_source_event()])
    sync_connector_events(db=db_session, connector=connector)
    reconciliation_calls.clear()

    result = sync_connector_events(db=db_session, connector=connector)

    assert result.skipped_events == 1
    assert reconciliation_calls == []


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
