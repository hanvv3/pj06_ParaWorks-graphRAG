from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    KeyGenerationLockedContext,
    acquire_projection,
    lock_runtime_state,
)
from backend.app.connectors.base import SourceEvent
from backend.app.connectors.slack_synthetic import LocalSyntheticSlackConnector
from backend.app.core.config import Settings, get_settings
from backend.app.documents.service import (
    parsed_document_from_source_event,
    persist_parsed_document,
)
from backend.app.ingestion.source_content_signature import (
    CanonicalSourceContentSignature,
    ServerParserPolicy,
    SourceStateChangeClassification,
    canonical_source_content_signature,
    classify_source_state_change,
    connector_content_signature,
    normalize_source_permission,
    server_parser_policy_for_event,
    server_parser_run_matches_authority,
    source_state_primary_code,
)
from backend.app.ingestion.source_versions import SourceVersionRef, source_version_refs
from backend.app.models import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
    VectorIndexState,
)
from backend.app.rag.indexing import VectorIndexWriter, compute_vector_document_hash
from backend.app.rag.serving_generation import (
    RagServingGenerationLockedContext,
    advance_corpus_generation,
    increment_vector_index_generation,
    lock_rag_serving_generation,
)
from backend.app.rag.serving_locks import VectorServingLockManager
from backend.app.rag.vector_store import VectorDocument
from backend.app.review.auto_review_source_reconciliation import (
    CommittedSourceStateChange,
)

_C5_SOURCE_TYPES = frozenset({'gmail', 'gmail_attachment', 'drive', 'calendar'})
_OPERATIONAL_METADATA_KEYS = (
    'sync_cursor',
    'sync_partition',
    'connector_revision',
    'connector_updated_at',
)
_SERVER_RESOLVED_METADATA_KEYS = frozenset(
    {
        'account_id',
        'oauth_scope',
        'oauth_scopes',
        'required_scopes',
        'security_scope_id',
        'server_security_scope',
    }
)
_CONNECTOR_AUTHORITY_KEYS = frozenset(
    {
        'content_signature',
        'current_document_version_id',
        'server_content_signature',
        'server_content_signature_schema',
    }
)
_CONNECTOR_PARSER_AUTHORITY_KEYS = frozenset(
    {
        'chunk_max_chars',
        'chunk_policy',
        'chunk_policy_version',
        'content_hash',
        'mime_type',
        'page_number',
        'parser_name',
        'parser_policy_version',
        'parser_status',
        'parser_status_reason',
        'parser_version',
        'section_path',
        'source_snippet',
    }
)
_PERMISSION_RANK = {'public': 0, 'internal': 1, 'restricted': 2}


@dataclass(frozen=True)
class IngestionResult:
    created_review_items: int
    changed_source_ids: list[str]
    changed_source_refs: list[SourceVersionRef]
    changed_source_states: list[CommittedSourceStateChange]
    skipped_events: int = 0


@dataclass
class _VectorMutations:
    delete_ids: set[str]
    narrowings: dict[str, set[str]]

    @classmethod
    def empty(cls) -> _VectorMutations:
        return cls(delete_ids=set(), narrowings={})


def ingest_events(db: Session, events: list[SourceEvent]) -> int:
    return ingest_events_with_result(db, events).created_review_items


def ingest_events_with_result(
    db: Session,
    events: list[SourceEvent],
    *,
    vector_writer: VectorIndexWriter | None = None,
    settings: Settings | None = None,
    synthetic_slack_adapter: LocalSyntheticSlackConnector | None = None,
    authenticated_source_metadata_by_id: Mapping[
        str, Mapping[str, object]
    ] | None = None,
) -> IngestionResult:
    resolved_settings = settings or get_settings()
    production_vector_mutation = bool(
        vector_writer is not None
        and vector_writer.__class__.__name__ == 'PgVectorStore'
        and db.get_bind().dialect.name == 'postgresql'
    )
    synthetic_slack = type(synthetic_slack_adapter) is LocalSyntheticSlackConnector
    canonical_mutation = any(event.source_type in _C5_SOURCE_TYPES
                             or (synthetic_slack and event.source_type == 'slack')
                             for event in events)
    barrier = (
        KeyedMutationGuard.generation_barrier(db)
        if production_vector_mutation or canonical_mutation
        else nullcontext()
    )
    try:
        with barrier:
            key_context = (
                lock_runtime_state(db)
                if production_vector_mutation or canonical_mutation
                else None
            )
            if production_vector_mutation and key_context is None:
                raise RuntimeError('C.5 source mutation key runtime unavailable')
            generation_context = None
            if canonical_mutation:
                generation_context = lock_rag_serving_generation(
                    db,
                    settings=resolved_settings,
                    key_context=key_context,
                )
                acquire_projection(db, key_context)
            return _ingest_events_transaction(
                db,
                events,
                vector_writer=vector_writer,
                settings=resolved_settings,
                key_context=key_context,
                rag_generation_context=generation_context,
                synthetic_slack=synthetic_slack,
                authenticated_source_metadata_by_id=(
                    authenticated_source_metadata_by_id or {}
                ),
            )
    except Exception:
        db.rollback()
        raise


def _ingest_events_transaction(
    db: Session,
    events: list[SourceEvent],
    *,
    vector_writer: VectorIndexWriter | None,
    settings: Settings,
    key_context: KeyGenerationLockedContext | None,
    rag_generation_context: RagServingGenerationLockedContext | None,
    synthetic_slack: bool,
    authenticated_source_metadata_by_id: Mapping[
        str, Mapping[str, object]
    ],
) -> IngestionResult:
    existing_sources = _locked_sources_by_external_id(
        db, [event.source_id for event in events]
    )
    _lock_existing_document_state(db, existing_sources.values())
    changed_source_ids: list[str] = []
    changed_state_rows: list[tuple[Source, SourceStateChangeClassification]] = []
    mutations = _VectorMutations.empty()
    skipped_events = 0

    for event in events:
        existing_source = existing_sources.get(event.source_id)
        _validate_source_event_identity(event, existing_source=existing_source)
        if (event.source_type == 'slack' and synthetic_slack
                and existing_source is not None and not existing_source.server_content_signature):
            raise ValueError('unsigned Slack source collision')
        if event.source_type == 'slack' and not synthetic_slack:
            if existing_source is not None and existing_source.server_content_signature:
                raise ValueError('normal connector cannot update signed synthetic Slack source')
            if _legacy_slack_event_is_unchanged(existing_source, event):
                if _narrow_legacy_slack_permissions(
                    db, source=existing_source, event=event, mutations=mutations
                ):
                    changed_source_ids.append(event.source_id)
                else:
                    skipped_events += 1
                continue
            source = _persist_legacy_slack_event(
                db,
                event=event,
                existing_source=existing_source,
            )
            existing_sources[event.source_id] = source
            changed_source_ids.append(event.source_id)
            continue
        if event.source_type not in _C5_SOURCE_TYPES and not (synthetic_slack and event.source_type == 'slack'):
            raise ValueError('unsupported ingestion source type')

        current_parser_run = _current_parser_run(db, existing_source)
        computed_signature = canonical_source_content_signature(event)
        parser_policy = server_parser_policy_for_event(event)
        classification = classify_source_state_change(
            source=existing_source,
            event=event,
            current_parser_run=current_parser_run,
            computed_signature=computed_signature,
            parser_policy=parser_policy,
        )
        if not (
            classification.content_changed
            or classification.permission_changed
            or classification.parser_policy_changed
        ):
            _advance_unchanged_operational_state(
                source=existing_source,
                event=event,
            )
            skipped_events += 1
            continue

        source = _upsert_c5_source(
            db,
            event=event,
            existing_source=existing_source,
            signature=computed_signature,
            parser_policy=parser_policy,
            authenticated_metadata=authenticated_source_metadata_by_id.get(
                event.source_id
            ),
        )
        existing_sources[event.source_id] = source
        changed_source_ids.append(event.source_id)

        if (
            classification.permission_changed
            and not classification.content_changed
            and not classification.parser_policy_changed
        ):
            _apply_permission_only_change(
                db,
                source=source,
                incoming_permission=normalize_source_permission(event.permission_level),
                mutations=mutations,
            )
        else:
            old_chunk_ids = _source_chunk_document_ids(db, source.id)
            parsed_document = parsed_document_from_source_event(
                event,
                server_signature=computed_signature,
                parser_policy=parser_policy,
                canonical_permission_level=source.permission_level,
            )
            persist_parsed_document(
                db,
                source=source,
                title=event.title,
                parsed=parsed_document,
                metadata={
                    **_canonical_source_metadata(
                        event,
                        existing_metadata=source.raw_metadata,
                        authenticated_metadata=authenticated_source_metadata_by_id.get(
                            event.source_id
                        ),
                        parser_policy=parser_policy,
                    ),
                    'source_id': event.source_id,
                    'source_url': event.source_url,
                    'source_type': event.source_type,
                    'permission_level': source.permission_level,
                    'participants': list(event.participants),
                    'scenario': event.raw_metadata.get('scenario'),
                },
                server_signature=computed_signature,
                parser_policy=parser_policy,
                rag_generation_context=rag_generation_context,
            )
            mutations.delete_ids.update(old_chunk_ids)
        db.flush()
        changed_state_rows.append((source, classification))

    d_vector_mutated = _apply_vector_mutations(
        db,
        mutations=mutations,
        vector_writer=vector_writer,
        settings=settings,
        key_context=key_context,
    )
    if d_vector_mutated:
        if rag_generation_context is None:
            raise TypeError('D vector mutation requires a RAG generation context')
        increment_vector_index_generation(
            db,
            context=rag_generation_context,
        )
    if changed_state_rows:
        if rag_generation_context is None:
            raise TypeError('Canonical ingestion requires a RAG generation context')
        advance_corpus_generation(
            db,
            settings=settings,
            context=rag_generation_context,
        )
    db.commit()
    db.expire_all()
    changed_sources = (
        db.scalars(
            select(Source).where(Source.source_id.in_(changed_source_ids))
        ).all()
        if changed_source_ids
        else []
    )
    changed_states = [
        CommittedSourceStateChange(
            source_id=source.id,
            content_changed=classification.content_changed,
            permission_changed=classification.permission_changed,
            parser_policy_changed=classification.parser_policy_changed,
            primary_code=source_state_primary_code(
                content_changed=classification.content_changed,
                permission_changed=classification.permission_changed,
                parser_policy_changed=classification.parser_policy_changed,
            ),
        )
        for source, classification in changed_state_rows
    ]
    return IngestionResult(
        created_review_items=0,
        changed_source_ids=changed_source_ids,
        changed_source_refs=source_version_refs(changed_sources),
        changed_source_states=changed_states,
        skipped_events=skipped_events,
    )


def _validate_source_event_identity(
    event: SourceEvent,
    *,
    existing_source: Source | None,
) -> None:
    if (
        existing_source is not None
        and existing_source.source_type != event.source_type
    ):
        raise ValueError('source type conflicts with existing source')
    if event.source_type in _C5_SOURCE_TYPES:
        prefix = f'{event.source_type}:'
        if (
            event.source_id != event.source_id.strip()
            or not event.source_id.startswith(prefix)
            or event.source_id == prefix
        ):
            raise ValueError('source id does not match source type')
        return
    if event.source_type == 'slack' and any(
        event.source_id.startswith(f'{source_type}:')
        for source_type in _C5_SOURCE_TYPES
    ):
        raise ValueError('source id does not match source type')


def _locked_sources_by_external_id(
    db: Session,
    source_ids: list[str],
) -> dict[str, Source]:
    normalized = sorted(set(source_ids))
    if not normalized:
        return {}
    statement = (
        select(Source)
        .where(Source.source_id.in_(normalized))
        .order_by(Source.id)
    )
    if db.get_bind().dialect.name == 'postgresql':
        statement = statement.with_for_update()
    return {source.source_id: source for source in db.scalars(statement).all()}


def _lock_existing_document_state(
    db: Session,
    sources: Iterable[Source],
) -> None:
    if db.get_bind().dialect.name != 'postgresql':
        return
    source_ids = sorted(
        source.id
        for source in sources
        if source.source_type in _C5_SOURCE_TYPES or source.server_content_signature is not None
    )
    if not source_ids:
        return
    documents = tuple(
        db.scalars(
            select(Document)
            .where(Document.source_id.in_(source_ids))
            .order_by(Document.id)
            .with_for_update()
        ).all()
    )
    document_ids = [document.id for document in documents]
    if not document_ids:
        return
    tuple(
        db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id.in_(document_ids))
            .order_by(DocumentVersion.id)
            .with_for_update()
        ).all()
    )
    tuple(
        db.scalars(
            select(DocumentParserRun)
            .where(DocumentParserRun.document_id.in_(document_ids))
            .order_by(DocumentParserRun.id)
            .with_for_update()
        ).all()
    )
    tuple(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.source_id.in_(source_ids))
            .order_by(DocumentChunk.id)
            .with_for_update()
        ).all()
    )


def _upsert_c5_source(
    db: Session,
    *,
    event: SourceEvent,
    existing_source: Source | None,
    signature: CanonicalSourceContentSignature,
    parser_policy: ServerParserPolicy,
    authenticated_metadata: Mapping[str, object] | None,
) -> Source:
    previous_metadata = (
        existing_source.raw_metadata if existing_source is not None else None
    )
    metadata = _canonical_source_metadata(
        event,
        existing_metadata=previous_metadata,
        authenticated_metadata=authenticated_metadata,
        parser_policy=parser_policy,
    )
    permission = normalize_source_permission(event.permission_level)
    if existing_source is None:
        source = Source(
            source_type=event.source_type,
            source_id=event.source_id,
            source_url=event.source_url,
            title=event.title,
            author=event.author,
            permission_level=permission,
            raw_metadata=metadata,
            server_content_signature_schema=signature.schema,
            server_content_signature=signature.signature,
            connector_content_signature=connector_content_signature(event),
        )
        db.add(source)
        db.flush()
        return source
    existing_source.source_type = event.source_type
    existing_source.source_url = event.source_url
    existing_source.title = event.title
    existing_source.author = event.author
    existing_source.permission_level = permission
    existing_source.raw_metadata = metadata
    existing_source.server_content_signature_schema = signature.schema
    existing_source.server_content_signature = signature.signature
    existing_source.connector_content_signature = connector_content_signature(event)
    return existing_source


def _canonical_source_metadata(
    event: SourceEvent,
    *,
    existing_metadata: Mapping[str, object] | None,
    authenticated_metadata: Mapping[str, object] | None,
    parser_policy: ServerParserPolicy,
) -> dict:
    incoming = {
        key: value
        for key, value in event.raw_metadata.items()
        if key not in _CONNECTOR_AUTHORITY_KEYS
        and key not in _CONNECTOR_PARSER_AUTHORITY_KEYS
        and key not in _SERVER_RESOLVED_METADATA_KEYS
    }
    if existing_metadata is not None:
        for key in _SERVER_RESOLVED_METADATA_KEYS:
            if key in existing_metadata:
                incoming[key] = existing_metadata[key]
    if authenticated_metadata is not None:
        unexpected = set(authenticated_metadata) - _SERVER_RESOLVED_METADATA_KEYS
        if unexpected:
            raise ValueError('authenticated source metadata contains unsupported keys')
        for key, value in authenticated_metadata.items():
            incoming.setdefault(key, value)
    incoming['participants'] = list(event.participants)
    incoming['semantic_timestamp_raw'] = event.semantic_timestamp_raw
    incoming['mime_type'] = parser_policy.mime_type
    return incoming


def _advance_unchanged_operational_state(
    *,
    source: Source | None,
    event: SourceEvent,
) -> None:
    if source is None:
        raise RuntimeError('unchanged source state requires an existing source')
    metadata = dict(source.raw_metadata or {})
    for key in _OPERATIONAL_METADATA_KEYS:
        if key in event.raw_metadata:
            metadata[key] = event.raw_metadata[key]
    source.raw_metadata = metadata
    source.source_url = event.source_url
    source.connector_content_signature = connector_content_signature(event)


def _current_parser_run(
    db: Session,
    source: Source | None,
) -> DocumentParserRun | None:
    if source is None:
        return None
    document = db.scalar(select(Document).where(Document.source_id == source.id))
    if document is None or document.current_document_version_id is None:
        return None
    runs = tuple(
        db.scalars(
            select(DocumentParserRun)
            .where(
                DocumentParserRun.document_id == document.id,
                DocumentParserRun.source_id == source.id,
                DocumentParserRun.document_version_id
                == document.current_document_version_id,
            )
            .order_by(DocumentParserRun.id)
        ).all()
    )
    if len(runs) != 1:
        return None
    parser_run = runs[0]
    chunks = tuple(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.version_id
                == document.current_document_version_id
            )
            .order_by(DocumentChunk.chunk_index, DocumentChunk.id)
        ).all()
    )
    if parser_run.chunk_count != len(chunks) or not chunks:
        return None
    if [chunk.chunk_index for chunk in chunks] != list(range(len(chunks))):
        return None
    if any(
        chunk.source_id != source.id
        or chunk.parser_run_id != parser_run.id
        for chunk in chunks
    ):
        return None
    return parser_run


def _source_chunk_document_ids(db: Session, source_id: int) -> set[str]:
    return {
        f'chunk:{chunk_id}'
        for chunk_id in db.scalars(
            select(DocumentChunk.id).where(DocumentChunk.source_id == source_id)
        ).all()
    }


def _apply_permission_only_change(
    db: Session,
    *,
    source: Source,
    incoming_permission: str,
    mutations: _VectorMutations,
) -> None:
    chunks = tuple(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.source_id == source.id)
            .order_by(DocumentChunk.id)
        ).all()
    )
    document_ids = {f'chunk:{chunk.id}' for chunk in chunks}
    if incoming_permission not in _PERMISSION_RANK:
        for chunk in chunks:
            chunk.permission_level = incoming_permission
            chunk.metadata_ = {
                **(chunk.metadata_ or {}),
                'permission_level': incoming_permission,
            }
        mutations.delete_ids.update(document_ids)
        return
    for chunk in chunks:
        current = normalize_source_permission(chunk.permission_level)
        effective = max(
            (current, incoming_permission),
            key=lambda value: _PERMISSION_RANK.get(value, len(_PERMISSION_RANK)),
        )
        chunk.permission_level = effective
        chunk.metadata_ = {
            **(chunk.metadata_ or {}),
            'permission_level': effective,
        }
        if effective != current:
            mutations.narrowings.setdefault(effective, set()).add(
                f'chunk:{chunk.id}'
            )
    db.flush()
    for permission, document_ids_for_permission in mutations.narrowings.items():
        for document_id in document_ids_for_permission:
            _refresh_chunk_index_state_hash(
                db,
                source=source,
                document_id=document_id,
                permission_level=permission,
            )


def _refresh_chunk_index_state_hash(
    db: Session,
    *,
    source: Source,
    document_id: str,
    permission_level: str,
) -> None:
    states = tuple(
        db.scalars(
            select(VectorIndexState).where(
                VectorIndexState.document_id == document_id
            )
        ).all()
    )
    if not states:
        return
    chunk = db.get(DocumentChunk, int(document_id.split(':', maxsplit=1)[1]))
    if chunk is None:
        return
    version = db.get(DocumentVersion, chunk.version_id)
    parser_run = (
        db.get(DocumentParserRun, chunk.parser_run_id)
        if chunk.parser_run_id is not None
        else None
    )
    document = db.get(Document, version.document_id) if version is not None else None
    if (
        version is None
        or parser_run is None
        or document is None
        or document.source_id != source.id
        or document.current_document_version_id != version.id
        or parser_run.document_id != document.id
        or parser_run.document_version_id != version.id
        or parser_run.source_id != source.id
        or parser_run.document_version_label != version.version
        or source.server_content_signature is None
        or not server_parser_run_matches_authority(
            source=source,
            parser_run=parser_run,
        )
    ):
        raise RuntimeError('current server parser authority is unavailable')
    parser_metadata: dict[str, object] = {
        'parser_name': parser_run.parser_name,
        'parser_status': parser_run.parser_status,
        'parser_status_reason': parser_run.parser_status_reason,
        'mime_type': parser_run.mime_type,
        'document_version': version.version,
        'revision_id': parser_run.revision_id,
        'content_signature': parser_run.content_signature,
        'content_hash': sha256(chunk.text.encode('utf-8')).hexdigest(),
    }
    for key in ('section_path', 'page_number'):
        if key in chunk.metadata_:
            parser_metadata[key] = chunk.metadata_[key]
    metadata = {
        'chunk_id': chunk.id,
        'source_pk': source.id,
        'source_id': source.source_id,
        'source_type': source.source_type,
        'author': source.author,
        'author_name': source.raw_metadata.get('author_name') or source.author,
        'channel_name': source.raw_metadata.get('channel_name'),
        'timestamp': str(
            source.raw_metadata.get('ts') or source.created_at.isoformat()
        ),
        'created_at_date': source.raw_metadata.get('created_at_date'),
        'category': chunk.metadata_.get('category'),
        'topic_tag': chunk.metadata_.get('topic_tag'),
        'importance': chunk.metadata_.get('importance'),
        'scenario': source.raw_metadata.get('scenario'),
        **parser_metadata,
    }
    content_hash = compute_vector_document_hash(
        VectorDocument(
            document_id=document_id,
            text=chunk.text,
            source_url=source.source_url,
            source_snippet=chunk.source_snippet,
            permission_level=permission_level,
            metadata=metadata,
        )
    )
    for state in states:
        if state.serving_kind is not None:
            continue
        state.content_hash = content_hash


def _apply_vector_mutations(
    db: Session,
    *,
    mutations: _VectorMutations,
    vector_writer: VectorIndexWriter | None,
    settings: Settings,
    key_context: KeyGenerationLockedContext | None,
) -> bool:
    all_document_ids = sorted(
        mutations.delete_ids.union(
            document_id
            for ids in mutations.narrowings.values()
            for document_id in ids
        )
    )
    if not all_document_ids:
        return False
    d_tracked_document_ids = set(
        db.scalars(
            select(VectorIndexState.document_id).where(
                VectorIndexState.document_id.in_(all_document_ids),
                VectorIndexState.serving_kind.is_not(None),
                VectorIndexState.index_policy_version
                == 'rag-v2-serving-index:v1',
                VectorIndexState.status == 'indexed',
            )
        ).all()
    )
    d_vector_mutated = False
    locked_context = None
    production_pgvector = bool(
        vector_writer is not None
        and vector_writer.__class__.__name__ == 'PgVectorStore'
        and db.get_bind().dialect.name == 'postgresql'
    )
    if production_pgvector:
        if key_context is None:
            raise RuntimeError('C.5 vector mutation key context unavailable')
        manager = VectorServingLockManager(db=db, settings=settings)
        bound = manager.bind_transaction(key_context)
        locked_context = manager.acquire_documents(bound, all_document_ids)
        tuple(
            db.scalars(
                select(VectorIndexState)
                .where(VectorIndexState.document_id.in_(all_document_ids))
                .order_by(VectorIndexState.id)
                .with_for_update()
            ).all()
        )
    if mutations.delete_ids:
        db.execute(
            delete(VectorIndexState).where(
                VectorIndexState.document_id.in_(sorted(mutations.delete_ids))
            )
        )
        if vector_writer is not None:
            if production_pgvector:
                deleted_count = vector_writer.delete_many(
                    sorted(mutations.delete_ids),
                    locked_context=locked_context,  # type: ignore[call-arg]
                )
            else:
                deleted_count = vector_writer.delete_many(
                    sorted(mutations.delete_ids)
                )
            d_vector_mutated = bool(
                d_vector_mutated
                or (
                    deleted_count > 0
                    and d_tracked_document_ids.intersection(
                        mutations.delete_ids
                    )
                )
            )
    if vector_writer is None:
        return False
    for permission_level in sorted(
        mutations.narrowings,
        key=lambda value: _PERMISSION_RANK.get(value, len(_PERMISSION_RANK)),
    ):
        document_ids = sorted(
            mutations.narrowings[permission_level] - mutations.delete_ids
        )
        if not document_ids:
            continue
        if production_pgvector:
            narrowed_count = vector_writer.narrow_permissions(
                document_ids,
                permission_level,
                locked_context=locked_context,  # type: ignore[call-arg]
            )
        else:
            narrowed_count = vector_writer.narrow_permissions(
                document_ids, permission_level
            )
        d_vector_mutated = bool(
            d_vector_mutated
            or (
                narrowed_count > 0
                and d_tracked_document_ids.intersection(document_ids)
            )
        )
    return d_vector_mutated


def _narrow_legacy_slack_permissions(
    db: Session,
    *,
    source: Source,
    event: SourceEvent,
    mutations: _VectorMutations,
) -> bool:
    # A connector body hash is not a permission fingerprint. Preserve unsigned
    # authority while narrowing even previously inconsistent stored chunks.
    effective = max(
        (normalize_source_permission(source.permission_level),
         normalize_source_permission(event.permission_level)),
        key=lambda value: _PERMISSION_RANK.get(value, len(_PERMISSION_RANK)),
    )
    chunk_permissions = db.scalars(
        select(DocumentChunk.permission_level).where(DocumentChunk.source_id == source.id)
    )
    chunk_narrows = any(
        _PERMISSION_RANK.get(normalize_source_permission(value), len(_PERMISSION_RANK))
        < _PERMISSION_RANK.get(effective, len(_PERMISSION_RANK))
        for value in chunk_permissions
    )
    if source.permission_level == effective and not chunk_narrows:
        return False
    source.permission_level = effective
    _apply_permission_only_change(
        db, source=source, incoming_permission=effective, mutations=mutations
    )
    return True


def _legacy_slack_event_is_unchanged(
    source: Source | None,
    event: SourceEvent,
) -> bool:
    if source is None:
        return False
    existing_signature = source.connector_content_signature
    if existing_signature is None:
        existing_signature = (source.raw_metadata or {}).get('content_signature')
    incoming_signature = connector_content_signature(event)
    if isinstance(existing_signature, str) and incoming_signature is not None:
        return existing_signature == incoming_signature
    return True


def _persist_legacy_slack_event(
    db: Session,
    *,
    event: SourceEvent,
    existing_source: Source | None,
) -> Source:
    metadata = {**event.raw_metadata, 'participants': list(event.participants)}
    if existing_source is None:
        source = Source(
            source_type=event.source_type,
            source_id=event.source_id,
            source_url=event.source_url,
            title=event.title,
            author=event.author,
            permission_level=event.permission_level,
            raw_metadata=metadata,
            connector_content_signature=connector_content_signature(event),
        )
        db.add(source)
        db.flush()
    else:
        source = existing_source
        source.source_url = event.source_url
        source.title = event.title
        source.author = event.author
        source.permission_level = event.permission_level
        source.raw_metadata = metadata
        source.connector_content_signature = connector_content_signature(event)
    parsed_document = parsed_document_from_source_event(event)
    source.raw_metadata = {
        **(source.raw_metadata or {}),
        'content_signature': parsed_document.content_signature,
    }
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
    return source
