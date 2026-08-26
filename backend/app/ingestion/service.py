from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.connectors.base import SourceEvent
from backend.app.documents.service import (
    parsed_document_from_source_event,
    persist_parsed_document,
)
from backend.app.ingestion.source_versions import SourceVersionRef, source_version_refs
from backend.app.models import (
    DocumentChunk,
    Source,
)


@dataclass(frozen=True)
class IngestionResult:
    created_review_items: int
    changed_source_ids: list[str]
    changed_source_refs: list[SourceVersionRef]


def ingest_events(db: Session, events: list[SourceEvent]) -> int:
    return ingest_events_with_result(db, events).created_review_items


def ingest_events_with_result(db: Session, events: list[SourceEvent]) -> IngestionResult:
    created_chunks: list[DocumentChunk] = []
    changed_source_ids: list[str] = []

    for event in events:
        existing_source = db.scalar(select(Source).where(Source.source_id == event.source_id))
        if existing_source is not None and _same_content_signature(existing_source, event):
            continue
        changed_source_ids.append(event.source_id)

        if existing_source is None:
            source = Source(
                source_type=event.source_type,
                source_id=event.source_id,
                source_url=event.source_url,
                title=event.title,
                author=event.author,
                permission_level=event.permission_level,
                raw_metadata={**event.raw_metadata, 'participants': list(event.participants)},
            )
            db.add(source)
            db.flush()
        else:
            source = existing_source
            source.source_url = event.source_url
            source.title = event.title
            source.author = event.author
            source.permission_level = event.permission_level
            source.raw_metadata = {**event.raw_metadata, 'participants': list(event.participants)}

        parsed_document = parsed_document_from_source_event(event)
        source.raw_metadata = {
            **(source.raw_metadata or {}),
            'content_signature': parsed_document.content_signature,
        }
        created_chunks.extend(
            persist_parsed_document(
                db,
                source=source,
                title=event.title,
                parsed=parsed_document,
                metadata={
                    **event.raw_metadata,
                    'source_id': event.source_id,
                    'source_url': event.source_url,
                    'source_type': event.source_type,
                    'permission_level': event.permission_level,
                    'participants': list(event.participants),
                    'scenario': event.raw_metadata.get('scenario'),
                },
            )
        )

    db.commit()
    db.expire_all()
    changed_sources = (
        db.scalars(select(Source).where(Source.source_id.in_(changed_source_ids))).all()
        if changed_source_ids
        else []
    )
    # 룰 기반 추출기를 제거하였으므로 생성된 ReviewItem 개수는 0으로 반환합니다. 
    # 실제 리뷰 아이템은 AI Agent를 통해 별도로 생성됩니다.
    return IngestionResult(
        created_review_items=0,
        changed_source_ids=changed_source_ids,
        changed_source_refs=source_version_refs(changed_sources),
    )


def _same_content_signature(source: Source, event: SourceEvent) -> bool:
    existing_signature = (source.raw_metadata or {}).get('content_signature')
    incoming_signature = event.raw_metadata.get('content_signature')
    if existing_signature and incoming_signature:
        return existing_signature == incoming_signature
    return True
