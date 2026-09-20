from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from uuid import uuid4

from sqlalchemy import Numeric, cast, func, select
from sqlalchemy.orm import Session

from backend.app.connectors.base import Connector, SourceEvent
from backend.app.connectors.google import (
    GOOGLE_CONNECTOR_SCOPES,
    GoogleConnector,
)
from backend.app.connectors.slack import (
    SLACK_KNOWN_MESSAGE_WINDOW,
    SLACK_MAX_CHANNELS,
    SlackApiError,
    SlackConnector,
)
from backend.app.connectors.slack_synthetic import LocalSyntheticSlackConnector
from backend.app.core.config import Settings, get_settings
from backend.app.ingestion.service import ingest_events_with_result
from backend.app.ingestion.source_versions import (
    ReviewBatchMode,
    SourceVersionRef,
    with_review_batch_marker,
)
from backend.app.models import Source, SyncJob
from backend.app.rag.indexing import VectorIndexWriter
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore
from backend.app.review.auto_review_source_reconciliation import (
    AutoReviewSourceReconciliationService,
    CommittedSourceStateChange,
)
from backend.app.tasks.rag_indexing import enqueue_rag_reindex_job


@dataclass(frozen=True)
class ConnectorSyncResult:
    job_id: str
    connector_type: str
    status: str
    fetched_events: int
    created_review_items: int
    skipped_events: int
    changed_source_ids: list[str] = field(default_factory=list)
    changed_source_refs: list[SourceVersionRef] = field(default_factory=list)
    parser_status_counts: dict[str, int] = field(default_factory=dict)


def sync_connector_events(
    db: Session,
    connector: Connector,
    job_id: str | None = None,
    review_batch_mode: ReviewBatchMode | None = None,
    settings: Settings | None = None,
    vector_writer: VectorIndexWriter | None = None,
    incremental_reindex_enqueuer: Callable[[str], None] | None = None,
) -> ConnectorSyncResult:
    resolved_settings = settings or get_settings()
    job = (
        db.scalar(select(SyncJob).where(SyncJob.job_id == job_id))
        if job_id is not None
        else None
    )
    if job is None:
        job = SyncJob(
            job_id=job_id or f'{connector.source_type}-{uuid4().hex}',
            connector_type=connector.source_type,
            status='running',
            message='sync running',
            progress_pct=10,
        )
        db.add(job)
    else:
        job.connector_type = connector.source_type
        job.status = 'running'
        job.message = 'sync running'
        job.progress_pct = 10
        job.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(job)

    try:
        if isinstance(connector, SlackConnector):
            cursors, known_messages = _slack_sync_context(db, connector)
            events = connector.fetch_events_since(cursors, known_messages_by_channel=known_messages)
        elif hasattr(connector, 'fetch_events_since'):
            events = connector.fetch_events_since(_latest_cursors_by_partition(db, connector.source_type))
        else:
            events = connector.fetch_events()
        parser_status_counts = _parser_status_counts(events)
        resolved_writer = vector_writer or _source_mutation_vector_writer(
            db,
            settings=resolved_settings,
        )
        ingestion_result = ingest_events_with_result(
            db,
            events,
            vector_writer=resolved_writer,
            settings=resolved_settings,
            synthetic_slack_adapter=(
                connector if type(connector) is LocalSyntheticSlackConnector else None
            ),
            authenticated_source_metadata_by_id=(
                _authenticated_source_metadata_by_id(
                    connector,
                    events=events,
                    settings=resolved_settings,
                )
            ),
        )
        skipped_events = ingestion_result.skipped_events
        job.status = 'complete'
        job.message = (
            f'fetched={len(events)} '
            f'created_review_items={ingestion_result.created_review_items} '
            f'skipped_events={skipped_events}'
        )
        job.progress_pct = 100
        job.updated_at = datetime.now(UTC)
        _mark_changed_sources_for_job(
            db,
            changed_source_ids=ingestion_result.changed_source_ids,
            job_id=job.job_id,
            review_batch_mode=review_batch_mode,
        )
        db.commit()
        if ingestion_result.changed_source_states:
            AutoReviewSourceReconciliationService(
                db,
                settings=resolved_settings,
                vector_writer=resolved_writer,
            ).reconcile(ingestion_result.changed_source_states)
        if _requires_incremental_reindex(ingestion_result.changed_source_states):
            resolved_enqueuer = _incremental_reindex_enqueuer(
                db,
                settings=resolved_settings,
                explicit=incremental_reindex_enqueuer,
            )
            if resolved_enqueuer is not None:
                _enqueue_incremental_reindex(
                    db,
                    enqueuer=resolved_enqueuer,
                )
    except Exception as exc:
        db.rollback()
        failed_job = db.scalar(select(SyncJob).where(SyncJob.job_id == job.job_id))
        if failed_job is not None:
            failed_job.status = 'failed'
            failed_job.message = f'failed: {exc}'
            failed_job.progress_pct = 100
            failed_job.updated_at = datetime.now(UTC)
            db.commit()
        raise

    return ConnectorSyncResult(
        job_id=job.job_id,
        connector_type=connector.source_type,
        status=job.status,
        fetched_events=len(events),
        created_review_items=ingestion_result.created_review_items,
        skipped_events=skipped_events,
        changed_source_ids=ingestion_result.changed_source_ids,
        changed_source_refs=ingestion_result.changed_source_refs,
        parser_status_counts=parser_status_counts,
    )


def _mark_changed_sources_for_job(
    db: Session,
    *,
    changed_source_ids: list[str],
    job_id: str,
    review_batch_mode: ReviewBatchMode | None,
) -> None:
    if not changed_source_ids:
        return
    sources = db.scalars(
        select(Source).where(Source.source_id.in_(changed_source_ids))
    ).all()
    for source in sources:
        if review_batch_mode is None:
            source.raw_metadata = {
                **(source.raw_metadata or {}),
                'last_changed_sync_job_id': job_id,
            }
        else:
            source.raw_metadata = with_review_batch_marker(
                source,
                mode=review_batch_mode,
                sync_job_id=job_id,
            )


def _slack_sync_context(db: Session, connector: SlackConnector) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Bounded observations only; legacy unsigned sources gain no authority.

    Revisit the newest 50 message observations per selected channel. Older/unknown
    threads outside this window require a later explicit upstream discovery path.
    """
    channel = Source.raw_metadata['channel_id'].as_string()
    ts = Source.raw_metadata['ts'].as_string()
    scope = [Source.source_type == 'slack',
             Source.raw_metadata['workspace_url'].as_string() == connector.config.workspace_url.rstrip('/')]
    if connector.config.channel_ids:
        if len(set(connector.config.channel_ids)) > SLACK_MAX_CHANNELS:
            raise SlackApiError('Slack sync failed: channel_limit_exceeded')
        scope.append(channel.in_(connector.config.channel_ids))
    rows = db.execute(select(channel, func.max(cast(ts, Numeric(24, 6))))
                      .where(*scope).group_by(channel).limit(SLACK_MAX_CHANNELS + 1)).all()
    if len(rows) > SLACK_MAX_CHANNELS:
        raise SlackApiError('Slack sync failed: channel_limit_exceeded')
    cursors, known = {}, {}
    for channel_id, cursor in rows:
        if not channel_id or cursor is None:
            continue
        cursors[channel_id] = str(cursor)
        observations = db.scalars(select(Source).where(*scope, channel == channel_id)
                                  .order_by(cast(ts, Numeric(24, 6)).desc(), Source.id.desc())
                                  .limit(SLACK_KNOWN_MESSAGE_WINDOW)).all()
        known[channel_id] = [{**source.raw_metadata, 'permission_level': source.permission_level}
                             for source in observations]
    return cursors, known


def _latest_cursors_by_partition(db: Session, source_type: str) -> dict[str, str]:
    latest: dict[str, tuple[tuple[int, object], str]] = {}
    sources = db.scalars(select(Source).where(Source.source_type == source_type)).all()
    for source in sources:
        raw_metadata = source.raw_metadata or {}
        partition = raw_metadata.get('sync_partition') or raw_metadata.get('channel_id')
        cursor = raw_metadata.get('sync_cursor') or raw_metadata.get('ts')
        
        if not isinstance(partition, str) or not isinstance(cursor, str) or not cursor:
            continue
            
        cursor_value = _cursor_sort_key(cursor)
        previous = latest.get(partition)
        if previous is None or cursor_value > previous[0]:
            latest[partition] = (cursor_value, cursor)
            
    return {partition: cursor for partition, (_, cursor) in latest.items()}


def _source_mutation_vector_writer(
    db: Session,
    *,
    settings: Settings,
) -> VectorIndexWriter | None:
    if db.get_bind().dialect.name != 'postgresql':
        return None
    return PgVectorStore(
        session=db,
        config=PgVectorConfig(
            embedding_dimensions=settings.openai_embedding_dimensions
        ),
        settings=settings,
    )


def _authenticated_source_metadata_by_id(
    connector: Connector,
    *,
    events: list[SourceEvent],
    settings: Settings,
) -> dict[str, dict[str, object]]:
    if not isinstance(connector, GoogleConnector):
        return {}
    result: dict[str, dict[str, object]] = {}
    for event in events:
        connector_type = (
            'gmail'
            if event.source_type == 'gmail_attachment'
            else event.source_type
        )
        scopes = GOOGLE_CONNECTOR_SCOPES.get(connector_type)
        if scopes is None:
            continue
        result[event.source_id] = {
            'account_id': connector.config.account_id,
            'required_scopes': list(scopes),
            'security_scope_id': settings.agent_runtime_security_scope_id,
        }
    return result


def _requires_incremental_reindex(
    changed_states: list[CommittedSourceStateChange],
) -> bool:
    return any(
        state.content_changed or state.parser_policy_changed
        for state in changed_states
    )


def _incremental_reindex_enqueuer(
    db: Session,
    *,
    settings: Settings,
    explicit: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    if explicit is not None:
        return explicit
    if db.get_bind().dialect.name != 'postgresql' or not settings.openai_api_key:
        return None
    return lambda job_id: enqueue_rag_reindex_job(job_id=job_id, dry_run=False)


def _enqueue_incremental_reindex(
    db: Session,
    *,
    enqueuer: Callable[[str], None],
) -> None:
    job = SyncJob(
        job_id=f'rag-index-{uuid4().hex}',
        connector_type='rag-index',
        status='queued',
        message='incremental source reindex queued',
        progress_pct=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    enqueuer(job.job_id)


def _parser_status_counts(events: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        parser_status = event.raw_metadata.get('parser_status')
        if not parser_status:
            continue
        key = str(parser_status)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _cursor_sort_key(cursor: str) -> tuple[int, object]:
    try:
        return (0, Decimal(cursor))
    except InvalidOperation:
        pass
    try:
        normalized = cursor.replace('Z', '+00:00')
        return (1, datetime.fromisoformat(normalized).astimezone(UTC))
    except ValueError:
        return (2, cursor)
