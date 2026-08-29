import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Protocol

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.keyed_mutation_guard import (
    KeyedMutationGuard,
    lock_runtime_state,
)
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    server_parser_run_matches_authority,
)
from backend.app.ingestion.source_versions import current_content_signature
from backend.app.knowledge.trusted_serving_eligibility import (
    TrustedServingEligibilityService,
)
from backend.app.models import (
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    HistoryEvent,
    ReviewItem,
    Source,
    TimelineEvent,
    Todo,
    VectorIndexState,
    VectorServingTombstone,
)
from backend.app.rag.embeddings import EmbeddingBatchResult, EmbeddingModel
from backend.app.rag.serving_locks import (
    ServingMutationLockCoordinator,
    build_serving_lock_plan,
)
from backend.app.rag.vector_store import VectorDocument


class VectorIndexWriter(Protocol):
    def upsert_with_embedding(self, document: VectorDocument, embedding: list[float]) -> None:
        raise NotImplementedError

    def delete_many(self, document_ids: Sequence[str]) -> int:
        raise NotImplementedError

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        raise NotImplementedError


class EmbeddingBudgetExceededError(ValueError):
    def __init__(self, decision: dict[str, float | int | str | None]) -> None:
        self.decision = decision
        super().__init__('embedding budget exceeded')


@dataclass(frozen=True)
class VectorIndexResult:
    indexed_count: int
    document_ids: list[str]
    embedding_dimensions: int
    skipped_count: int = 0
    skipped_document_ids: list[str] | None = None
    saved_embedding_calls: int = 0
    saved_serving_writes: int = 0
    embedding_request_count: int = 0
    embedding_prompt_tokens: int = 0
    embedding_total_tokens: int = 0
    embedding_budget: dict[str, float | int | str | None] | None = None


class PreviewVectorIndexWriter:
    def __init__(self) -> None:
        self.upserts: list[tuple[VectorDocument, list[float]]] = []
        self.deletes: list[tuple[str, ...]] = []
        self.permission_narrowings: list[tuple[tuple[str, ...], str]] = []

    def upsert_with_embedding(self, document: VectorDocument, embedding: list[float]) -> None:
        self.upserts.append((document, embedding))

    def delete_many(self, document_ids: Sequence[str]) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.deletes.append(normalized)
        return len(normalized)

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        normalized = tuple(sorted(set(document_ids)))
        self.permission_narrowings.append((normalized, permission_level))
        return len(normalized)


def index_vector_documents(
    *,
    documents: list[VectorDocument],
    writer: VectorIndexWriter,
    embedding_model: EmbeddingModel,
) -> VectorIndexResult:
    document_ids: list[str] = []
    embedding_dimensions = 0
    for document in documents:
        embedding = embedding_model.embed(document.text)
        embedding_dimensions = len(embedding)
        writer.upsert_with_embedding(document, embedding)
        document_ids.append(document.document_id)

    return VectorIndexResult(
        indexed_count=len(document_ids),
        document_ids=document_ids,
        embedding_dimensions=embedding_dimensions or _model_dimensions(embedding_model),
        skipped_document_ids=[],
    )


def index_changed_vector_documents(
    *,
    db: Session,
    documents: list[VectorDocument],
    writer: VectorIndexWriter,
    embedding_model: EmbeddingModel,
    embedding_model_name: str,
    persist_state: bool = True,
    embedding_cost_per_1m_tokens: float = 0.0,
    max_embedding_cost_usd: float | None = None,
    enforce_embedding_budget: bool = True,
    settings: Settings | None = None,
) -> VectorIndexResult:
    changed_documents: list[tuple[VectorDocument, str, VectorIndexState | None]] = []
    skipped_document_ids: list[str] = []
    embedding_dimensions = _model_dimensions(embedding_model)
    production_pgvector = (
        writer.__class__.__name__ == 'PgVectorStore'
        and db.get_bind().dialect.name == 'postgresql'
    )
    if production_pgvector and settings is None:
        raise ValueError('PostgreSQL indexing requires serving-lock settings')
    canonical_live_documents = (
        {
            document.document_id: document
            for document in build_rag_index_documents(db)
        }
        if production_pgvector
        else {}
    )

    tombstoned_document_ids = set(
        db.scalars(
            select(VectorServingTombstone.document_id).where(
                VectorServingTombstone.document_id.in_(
                    [document.document_id for document in documents]
                )
            )
        ).all()
    )

    for document in documents:
        if document.document_id in tombstoned_document_ids:
            skipped_document_ids.append(document.document_id)
            continue
        if production_pgvector:
            canonical = canonical_live_documents.get(document.document_id)
            if (
                canonical is None
                or compute_vector_document_hash(canonical)
                != compute_vector_document_hash(document)
            ):
                skipped_document_ids.append(document.document_id)
                continue
        content_hash = compute_vector_document_hash(document)
        state = _get_index_state(
            db=db,
            document_id=document.document_id,
            embedding_model_name=embedding_model_name,
        )
        if state and state.status == 'indexed' and state.content_hash == content_hash:
            skipped_document_ids.append(document.document_id)
            continue

        changed_documents.append((document, content_hash, state))

    changed_texts = [document.text for document, _, _ in changed_documents]
    budget_decision = estimate_embedding_budget(
        texts=changed_texts,
        embedding_model_name=embedding_model_name,
        cost_per_1m_tokens=embedding_cost_per_1m_tokens,
        max_cost_usd=max_embedding_cost_usd,
    )
    if enforce_embedding_budget and budget_decision['action'] == 'block':
        raise EmbeddingBudgetExceededError(budget_decision)

    if production_pgvector:
        # Provider work must not hold an application transaction or advisory lock.
        db.rollback()
    batch = (
        _embed_many(embedding_model, changed_texts)
        if changed_texts
        else EmbeddingBatchResult(embeddings=[], request_count=0)
    )
    if production_pgvector:
        return _persist_locked_pgvector_batch(
            db=db,
            writer=writer,
            settings=settings,
            changed_documents=changed_documents,
            embeddings=batch.embeddings,
            skipped_document_ids=skipped_document_ids,
            embedding_model_name=embedding_model_name,
            embedding_dimensions=embedding_dimensions,
            persist_state=persist_state,
            batch=batch,
            budget_decision=budget_decision,
        )

    indexed_document_ids: list[str] = []
    for (document, content_hash, state), embedding in zip(changed_documents, batch.embeddings, strict=True):
        embedding_dimensions = len(embedding)
        writer.upsert_with_embedding(document, embedding)
        indexed_document_ids.append(document.document_id)
        if persist_state:
            _upsert_index_state(
                db=db,
                state=state,
                document=document,
                embedding_model_name=embedding_model_name,
                embedding_dimensions=embedding_dimensions,
                content_hash=content_hash,
            )

    if persist_state:
        db.commit()

    return VectorIndexResult(
        indexed_count=len(indexed_document_ids),
        document_ids=indexed_document_ids,
        embedding_dimensions=embedding_dimensions,
        skipped_count=len(skipped_document_ids),
        skipped_document_ids=skipped_document_ids,
        saved_embedding_calls=len(skipped_document_ids),
        embedding_request_count=batch.request_count,
        embedding_prompt_tokens=batch.prompt_tokens,
        embedding_total_tokens=batch.total_tokens,
        embedding_budget=budget_decision,
    )


def _persist_locked_pgvector_batch(
    *,
    db: Session,
    writer: VectorIndexWriter,
    settings: Settings,
    changed_documents: list[tuple[VectorDocument, str, VectorIndexState | None]],
    embeddings: list[list[float]],
    skipped_document_ids: list[str],
    embedding_model_name: str,
    embedding_dimensions: int,
    persist_state: bool,
    batch: EmbeddingBatchResult,
    budget_decision: dict[str, float | int | str | None],
) -> VectorIndexResult:
    indexed: list[str] = []
    pre_provider_skip_count = len(skipped_document_ids)
    post_provider_skip_count = 0
    stale_skips = list(skipped_document_ids)
    document_ids = [document.document_id for document, _, _ in changed_documents]
    if not document_ids:
        return VectorIndexResult(
            indexed_count=0,
            document_ids=[],
            embedding_dimensions=embedding_dimensions,
            skipped_count=len(stale_skips),
            skipped_document_ids=stale_skips,
            saved_embedding_calls=pre_provider_skip_count,
            embedding_request_count=batch.request_count,
            embedding_prompt_tokens=batch.prompt_tokens,
            embedding_total_tokens=batch.total_tokens,
            embedding_budget=budget_decision,
        )
    plan = build_serving_lock_plan(db, document_ids)
    with KeyedMutationGuard.generation_barrier(db):
        key_context = lock_runtime_state(db)
        if key_context is None:
            raise ValueError('PostgreSQL indexing key runtime unavailable')
        try:
            locked = ServingMutationLockCoordinator(
                db=db, settings=settings
            ).acquire(key_context=key_context, plan=plan)
        except RuntimeError as exc:
            if str(exc) != 'Serving mutation dependency plan changed':
                raise
            db.rollback()
            stale_skips.extend(document_ids)
            post_provider_skip_count += len(document_ids)
            return VectorIndexResult(
                indexed_count=0,
                document_ids=[],
                embedding_dimensions=embedding_dimensions,
                skipped_count=len(stale_skips),
                skipped_document_ids=stale_skips,
                saved_embedding_calls=pre_provider_skip_count,
                saved_serving_writes=post_provider_skip_count,
                embedding_request_count=batch.request_count,
                embedding_prompt_tokens=batch.prompt_tokens,
                embedding_total_tokens=batch.total_tokens,
                embedding_budget=budget_decision,
            )
        eligibility = TrustedServingEligibilityService(db)
        canonical_documents = {
            current.document_id: current
            for current in build_rag_index_documents(db)
            if current.document_id in document_ids
        }
        for (document, content_hash, _), embedding in zip(
            changed_documents, embeddings, strict=True
        ):
            live = eligibility.for_document(document.document_id)
            canonical = canonical_documents.get(document.document_id)
            if (
                not live.eligible
                or live.effective_permission is None
                or canonical is None
                or db.scalar(
                    select(VectorServingTombstone.id).where(
                        VectorServingTombstone.document_id
                        == document.document_id
                    )
                )
                is not None
            ):
                stale_skips.append(document.document_id)
                post_provider_skip_count += 1
                continue
            canonical_hash = compute_vector_document_hash(canonical)
            exact_snapshot = canonical == document and canonical_hash == content_hash
            permission_only_narrowing = _permission_only_narrowing(
                embedded=document,
                canonical=canonical,
            )
            if not exact_snapshot and not permission_only_narrowing:
                stale_skips.append(document.document_id)
                post_provider_skip_count += 1
                continue
            writer.upsert_with_embedding(
                canonical, embedding, locked_context=locked  # type: ignore[call-arg]
            )
            indexed.append(document.document_id)
            embedding_dimensions = len(embedding)
            if persist_state:
                state = _get_index_state(
                    db=db,
                    document_id=document.document_id,
                    embedding_model_name=embedding_model_name,
                )
                _upsert_index_state(
                    db=db,
                    state=state,
                    document=canonical,
                    embedding_model_name=embedding_model_name,
                    embedding_dimensions=embedding_dimensions,
                    content_hash=canonical_hash,
                )
        db.commit()
    return VectorIndexResult(
        indexed_count=len(indexed),
        document_ids=indexed,
        embedding_dimensions=embedding_dimensions,
        skipped_count=len(stale_skips),
        skipped_document_ids=stale_skips,
        saved_embedding_calls=pre_provider_skip_count,
        saved_serving_writes=post_provider_skip_count,
        embedding_request_count=batch.request_count,
        embedding_prompt_tokens=batch.prompt_tokens,
        embedding_total_tokens=batch.total_tokens,
        embedding_budget=budget_decision,
    )


def _permission_only_narrowing(
    *, embedded: VectorDocument, canonical: VectorDocument
) -> bool:
    permission_rank = {'public': 0, 'internal': 1, 'restricted': 2}
    if (
        embedded.permission_level not in permission_rank
        or canonical.permission_level not in permission_rank
        or permission_rank[canonical.permission_level]
        <= permission_rank[embedded.permission_level]
    ):
        return False
    return replace(
        canonical, permission_level=embedded.permission_level
    ) == embedded


def compute_vector_document_hash(document: VectorDocument) -> str:
    payload = {
        'document_id': document.document_id,
        'text': document.text,
        'source_url': document.source_url,
        'source_snippet': document.source_snippet,
        'permission_level': document.permission_level,
        'metadata': document.metadata,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    return sha256(encoded).hexdigest()


def estimate_embedding_budget(
    *,
    texts: list[str],
    embedding_model_name: str,
    cost_per_1m_tokens: float,
    max_cost_usd: float | None,
) -> dict[str, float | int | str | None]:
    estimated_input_tokens = sum(_estimate_embedding_tokens(text) for text in texts)
    estimated_cost = (
        Decimal(estimated_input_tokens) * Decimal(str(cost_per_1m_tokens)) / Decimal(1_000_000)
    )
    estimated_cost_usd = float(estimated_cost)

    if not texts:
        return {
            'embedding_model': embedding_model_name,
            'changed_document_count': 0,
            'estimated_input_tokens': 0,
            'estimated_cost_usd': 0.0,
            'budget_limit_usd': max_cost_usd,
            'budget_status': 'no_input',
            'action': 'skip',
            'reason': 'no_changed_documents',
        }

    if max_cost_usd is not None and estimated_cost_usd > max_cost_usd:
        return {
            'embedding_model': embedding_model_name,
            'changed_document_count': len(texts),
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_cost_usd': estimated_cost_usd,
            'budget_limit_usd': max_cost_usd,
            'budget_status': 'over_budget',
            'action': 'block',
            'reason': 'estimated_embedding_cost_exceeds_budget',
        }

    return {
        'embedding_model': embedding_model_name,
        'changed_document_count': len(texts),
        'estimated_input_tokens': estimated_input_tokens,
        'estimated_cost_usd': estimated_cost_usd,
        'budget_limit_usd': max_cost_usd,
        'budget_status': 'within_budget' if max_cost_usd is not None else 'not_limited',
        'action': 'run',
        'reason': 'within_embedding_budget',
    }


def build_rag_index_documents(db: Session) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    documents.extend(_chunk_documents(db))
    eligibility = TrustedServingEligibilityService(db)
    documents.extend(_decision_documents(db, eligibility))
    documents.extend(_history_documents(db, eligibility))
    documents.extend(_timeline_documents(db, eligibility))
    documents.extend(_todo_documents(db, eligibility))
    return documents


def _chunk_documents(db: Session) -> list[VectorDocument]:
    # Phase 2: 승인 기반 RAG (Approval-only RAG)
    # 사람이 '승인(approved)'한 ReviewItem에 포함된 source_id 목록만 수집
    approved_payloads = db.execute(
        select(ReviewItem.payload).where(
            ReviewItem.status == 'approved',
            or_(
                ReviewItem.resolution_source.is_(None),
                ReviewItem.resolution_source == 'human',
            ),
        )
    ).scalars().all()
    
    approved_sid_set: set[str] = set()
    for p in approved_payloads:
        if not isinstance(p, dict):
            continue
        source_ids = p.get('source_ids')
        if not isinstance(source_ids, list):
            continue
        approved_sid_set.update(
            source_id.strip()
            for source_id in source_ids
            if isinstance(source_id, str) and source_id.strip()
        )
            
    if not approved_sid_set:
        return []

    rows = db.execute(
        select(
            DocumentChunk,
            Source,
            DocumentVersion,
            Document,
            DocumentParserRun,
        )
        .join(Source, DocumentChunk.source_id == Source.id)
        .join(DocumentVersion, DocumentVersion.id == DocumentChunk.version_id)
        .join(
            Document,
            and_(
                Document.id == DocumentVersion.document_id,
                Document.source_id == Source.id,
                Document.current_document_version_id == DocumentVersion.id,
            ),
        )
        .join(
            DocumentParserRun,
            and_(
                DocumentParserRun.id == DocumentChunk.parser_run_id,
                DocumentParserRun.document_id == Document.id,
                DocumentParserRun.document_version_id == DocumentVersion.id,
                DocumentParserRun.source_id == Source.id,
            ),
        )
        .where(Source.source_id.in_(list(approved_sid_set)))
        .order_by(DocumentChunk.id)
    ).all()
    
    documents: list[VectorDocument] = []
    eligibility = TrustedServingEligibilityService(db)
    for chunk, source, version, document, parser_run in rows:
        serving = eligibility.for_document(f'chunk:{chunk.id}')
        if not serving.eligible or serving.effective_permission is None:
            continue
        if not _is_exact_current_server_chunk(
            db,
            source=source,
            document=document,
            version=version,
            parser_run=parser_run,
            chunk=chunk,
        ):
            continue
        timestamp = source.raw_metadata.get('ts') or source.created_at.isoformat()
        
        # 메타데이터 보강 (정적 태그 + 동적 태그)
        metadata = {
            'chunk_id': chunk.id,
            'source_pk': source.id,
            'source_id': source.source_id,
            'source_type': source.source_type,
            'author': source.author,
            'author_name': source.raw_metadata.get('author_name') or source.author,
            'channel_name': source.raw_metadata.get('channel_name'),
            'timestamp': str(timestamp),
            'created_at_date': source.raw_metadata.get('created_at_date'),
            'category': chunk.metadata_.get('category'),
            'topic_tag': chunk.metadata_.get('topic_tag'),
            'importance': chunk.metadata_.get('importance'),
            'scenario': source.raw_metadata.get('scenario'),
            **_document_parser_metadata(
                chunk=chunk,
                version=version,
                parser_run=parser_run,
            ),
        }
        
        documents.append(
            VectorDocument(
                document_id=f'chunk:{chunk.id}',
                text=chunk.text,
                source_url=source.source_url,
                source_snippet=chunk.source_snippet,
                permission_level=serving.effective_permission,
                metadata=metadata,
            )
        )
    return documents


def _is_exact_current_server_chunk(
    db: Session,
    *,
    source: Source,
    document: Document,
    version: DocumentVersion,
    parser_run: DocumentParserRun,
    chunk: DocumentChunk,
) -> bool:
    signature = current_content_signature(source)
    if (
        signature is None
        or document.current_document_version_id != version.id
        or parser_run.document_version_label != version.version
        or not server_parser_run_matches_authority(
            source_type=source.source_type,
            server_content_signature=signature,
            parser_run=parser_run,
        )
    ):
        return False
    current_runs = tuple(
        db.scalars(
            select(DocumentParserRun)
            .where(
                DocumentParserRun.source_id == source.id,
                DocumentParserRun.document_id == document.id,
                DocumentParserRun.document_version_id == version.id,
            )
            .order_by(DocumentParserRun.id)
        ).all()
    )
    if len(current_runs) != 1 or current_runs[0].id != parser_run.id:
        return False
    current_chunks = tuple(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.version_id == version.id)
            .order_by(DocumentChunk.chunk_index, DocumentChunk.id)
        ).all()
    )
    return bool(
        current_chunks
        and parser_run.chunk_count == len(current_chunks)
        and [current.chunk_index for current in current_chunks]
        == list(range(len(current_chunks)))
        and all(
            current.source_id == source.id
            and current.parser_run_id == parser_run.id
            for current in current_chunks
        )
        and chunk in current_chunks
    )


def _document_parser_metadata(
    *,
    chunk: DocumentChunk,
    version: DocumentVersion,
    parser_run: DocumentParserRun,
) -> dict[str, object]:
    metadata: dict[str, object] = {
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
            metadata[key] = chunk.metadata_[key]
    return metadata


def _decision_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    # 결정사항 테이블 조회 (이미 승인된 것만 저장됨)
    decisions = db.scalars(
        select(DecisionRecord)
        .where(DecisionRecord.review_status == 'approved')
        .order_by(DecisionRecord.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'decision_record:{decision.id}',
            source_type='decision_record',
            title=decision.title,
            text=f'결정사항: {decision.title}\n내용: {decision.decision_summary}',
            source_links=decision.source_links,
            source_snippets=decision.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=decision.created_at.isoformat(),
            project_key=decision.project_key,
        )
        for decision in decisions
        if (result := eligibility.for_knowledge('decision_record', decision.id)).eligible
    ]


def _history_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    # 기록/공유 테이블 조회
    events = db.scalars(
        select(HistoryEvent)
        .where(HistoryEvent.review_status == 'approved')
        .order_by(HistoryEvent.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'history_event:{event.id}',
            source_type='history_event',
            title=event.title,
            text=f'기록/공유: {event.title}\n내용: {event.reason}',
            source_links=event.source_links,
            source_snippets=event.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=event.created_at.isoformat(),
            project_key=event.project_key,
        )
        for event in events
        if (result := eligibility.for_knowledge('history_event', event.id)).eligible
    ]


def _timeline_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    events = db.scalars(
        select(TimelineEvent)
        .where(TimelineEvent.review_status == 'approved')
        .order_by(TimelineEvent.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'timeline_event:{event.id}',
            source_type='timeline_event',
            title=event.title,
            text=f'Timeline: {event.title}\nSummary: {event.result_summary}',
            source_links=event.source_links,
            source_snippets=event.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=event.created_at.isoformat(),
            project_key=event.project_key,
        )
        for event in events
        if (result := eligibility.for_knowledge('timeline_event', event.id)).eligible
    ]


def _todo_documents(
    db: Session, eligibility: TrustedServingEligibilityService
) -> list[VectorDocument]:
    # 할 일 테이블 조회
    todos = db.scalars(
        select(Todo)
        .where(Todo.review_status == 'approved')
        .order_by(Todo.id)
    ).all()
    return [
        _knowledge_document(
            document_id=f'todo:{todo.id}',
            source_type='todo',
            title=todo.title,
            text=f'할 일: {todo.title}\n우선순위: {todo.priority}\n상세: {todo.priority_reason}',
            source_links=todo.source_links,
            source_snippets=todo.source_snippets,
            permission_level=result.effective_permission or 'restricted',
            timestamp=todo.created_at.isoformat(),
            project_key=todo.project_key,
        )
        for todo in todos
        if (result := eligibility.for_knowledge('todo', todo.id)).eligible
    ]


def _knowledge_document(
    *,
    document_id: str,
    source_type: str,
    title: str,
    text: str,
    source_links: list[str],
    source_snippets: list[str],
    permission_level: str,
    timestamp: str,
    project_key: str | None = None,
) -> VectorDocument:
    # 지식 항목 인덱싱 시에도 메타데이터 최대한 보강
    return VectorDocument(
        document_id=document_id,
        text=text,
        source_url=source_links[0] if source_links else f'knowledge://{document_id}',
        source_snippet=source_snippets[0] if source_snippets else text[:240],
        permission_level=permission_level,
        metadata={
            'knowledge_id': document_id,
            'source_type': source_type,
            'title': title,
            'author': 'ParaWorks AI (Verified)',
            'timestamp': timestamp,
            'project_key': project_key,
        },
    )


def _model_dimensions(embedding_model: EmbeddingModel) -> int:
    return int(getattr(embedding_model, 'dimensions', 0))


def _estimate_embedding_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative preflight estimate: UTF-8 bytes / 4 tracks English reasonably
    # and errs high for Korean before any paid embedding call is made.
    return max(1, (len(text.encode('utf-8')) + 3) // 4)


def _embed_many(embedding_model: EmbeddingModel, texts: list[str]) -> EmbeddingBatchResult:
    if hasattr(embedding_model, 'embed_many'):
        return embedding_model.embed_many(texts)
    return EmbeddingBatchResult(
        embeddings=[embedding_model.embed(text) for text in texts],
        request_count=len(texts),
    )


def _get_index_state(
    *,
    db: Session,
    document_id: str,
    embedding_model_name: str,
) -> VectorIndexState | None:
    return db.scalar(
        select(VectorIndexState).where(
            VectorIndexState.document_id == document_id,
            VectorIndexState.embedding_model == embedding_model_name,
        )
    )


def _upsert_index_state(
    *,
    db: Session,
    state: VectorIndexState | None,
    document: VectorDocument,
    embedding_model_name: str,
    embedding_dimensions: int,
    content_hash: str,
) -> None:
    now = datetime.now(UTC)
    if state is None:
        db.add(
            VectorIndexState(
                document_id=document.document_id,
                embedding_model=embedding_model_name,
                embedding_dimensions=embedding_dimensions,
                content_hash=content_hash,
                status='indexed',
                last_error=None,
                indexed_at=now,
            )
        )
        return

    state.embedding_dimensions = embedding_dimensions
    state.content_hash = content_hash
    state.status = 'indexed'
    state.last_error = None
    state.indexed_at = now
