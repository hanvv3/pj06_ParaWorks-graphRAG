from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.core.config import Settings
from backend.app.models import (
    DecisionRecord,
    DocumentChunk,
    HistoryEvent,
    RagLexicalServingProjection,
    RagServingCorpusGeneration,
    Source,
    TimelineEvent,
    Todo,
)
from backend.app.rag.indexing import build_rag_v2_index_documents
from backend.app.rag.serving_generation import RAG_LEXICAL_COMPAT_VERSION
from backend.app.rag.vector_store import VectorDocument


def refresh_rag_lexical_projections(
    db: Session,
    *,
    settings: Settings,
    corpus_generation: int,
) -> int:
    """Replace the actorless lexical projection for one locked generation."""
    corpus = db.get(RagServingCorpusGeneration, 1)
    if corpus is None or corpus.corpus_generation != corpus_generation:
        raise TypeError('Lexical refresh requires the locked current corpus generation')
    documents = build_rag_v2_index_documents(db, settings=settings)
    existing = {
        row.serving_document_id: row
        for row in db.scalars(select(RagLexicalServingProjection)).all()
    }
    expected_ids = {document.document_id for document in documents}
    for document_id, row in tuple(existing.items()):
        if document_id not in expected_ids:
            db.delete(row)
            existing.pop(document_id)
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    now = datetime.now(UTC)
    projected_count = 0
    for document in documents:
        title = _document_title(db, document)
        fields = _projection_hmac_fields(document)
        if title is None or fields is None:
            stale = existing.get(document.document_id)
            if stale is not None:
                db.delete(stale)
            continue
        row = existing.get(document.document_id)
        if row is None:
            row = RagLexicalServingProjection(
                corpus_generation_id=1,
                serving_document_id=document.document_id,
                created_at=now,
            )
            db.add(row)
        row.corpus_generation = corpus_generation
        row.serving_kind = fields['serving_kind']
        row.support_mode = fields['support_mode']
        row.effective_permission = document.permission_level
        row.serving_identity_hmac = fields['serving_identity_hmac']
        row.serving_version_fingerprint = fields[
            'serving_version_fingerprint'
        ]
        row.model_content_hmac = fields['model_content_hmac']
        row.canonical_citation_projection_hmac = fields[
            'canonical_citation_projection_hmac'
        ]
        row.title_lower = title.lower()
        row.searchable_lower = f'{title}\n{document.text}'.lower()
        row.lexical_contract_version = RAG_LEXICAL_COMPAT_VERSION
        row.fingerprint_key_version = (
            settings.agent_runtime_fingerprint_key_version
        )
        row.fingerprint_key_material_verifier = verifier
        row.updated_at = now
        projected_count += 1
    db.flush()
    return projected_count


def current_rag_lexical_projections(
    db: Session,
    *,
    settings: Settings,
) -> tuple[RagLexicalServingProjection, ...]:
    """Return only byte-exact, current-generation canonical projections."""
    corpus = db.get(RagServingCorpusGeneration, 1)
    if corpus is None:
        return ()
    expected = {
        document.document_id: document
        for document in build_rag_v2_index_documents(db, settings=settings)
    }
    verifier = fingerprint_key_material_verifier(
        settings.agent_runtime_fingerprint_secret
    )
    rows = tuple(
        db.scalars(
            select(RagLexicalServingProjection)
            .where(
                RagLexicalServingProjection.corpus_generation
                == corpus.corpus_generation,
                RagLexicalServingProjection.lexical_contract_version
                == RAG_LEXICAL_COMPAT_VERSION,
                RagLexicalServingProjection.fingerprint_key_version
                == settings.agent_runtime_fingerprint_key_version,
                RagLexicalServingProjection.fingerprint_key_material_verifier
                == verifier,
            )
            .order_by(RagLexicalServingProjection.serving_document_id)
        ).all()
    )
    current: list[RagLexicalServingProjection] = []
    for row in rows:
        document = expected.get(row.serving_document_id)
        title = _document_title(db, document) if document is not None else None
        fields = (
            _projection_hmac_fields(document) if document is not None else None
        )
        if document is None or title is None or fields is None:
            continue
        if (
            row.serving_kind != fields['serving_kind']
            or row.support_mode != fields['support_mode']
            or row.effective_permission != document.permission_level
            or row.serving_identity_hmac != fields['serving_identity_hmac']
            or row.serving_version_fingerprint
            != fields['serving_version_fingerprint']
            or row.model_content_hmac != fields['model_content_hmac']
            or row.canonical_citation_projection_hmac
            != fields['canonical_citation_projection_hmac']
            or row.title_lower != title.lower()
            or row.searchable_lower != f'{title}\n{document.text}'.lower()
        ):
            continue
        current.append(row)
    return tuple(current)


def canonical_rag_corpus_snapshot(
    db: Session,
    *,
    settings: Settings,
) -> tuple[tuple[str, ...], ...]:
    """Capture exact canonical bytes in memory for same-transaction mutation checks."""
    rows: list[tuple[str, ...]] = []
    for document in build_rag_v2_index_documents(db, settings=settings):
        title = _document_title(db, document)
        if title is None:
            continue
        rows.append(
            (
                document.document_id,
                title,
                document.text,
                document.source_url,
                document.source_snippet,
                document.permission_level,
                json.dumps(
                    document.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(',', ':'),
                    default=str,
                ),
            )
        )
    return tuple(sorted(rows))


def tokenize_rag_lexical_query(question: str) -> tuple[str, ...]:
    terms: list[str] = []
    for raw_term in question.replace(',', ' ').replace('.', ' ').split():
        stripped = raw_term.strip()
        if len(stripped) >= 3:
            terms.append(stripped.lower())
    return tuple(terms)


def score_rag_lexical_candidate(
    *,
    question: str,
    title: str,
    text: str,
) -> tuple[float, tuple[str, ...]]:
    query_terms = tokenize_rag_lexical_query(question)
    if not query_terms:
        return 0.0, ()
    title_lower = title.lower()
    searchable = f'{title}\n{text}'.lower()
    matched_terms = tuple(term for term in query_terms if term in searchable)
    if not matched_terms:
        return 0.0, ()
    exact_phrase_bonus = 1.0 if question.strip().lower() in searchable else 0.0
    coverage = len(matched_terms) / len(query_terms)
    title_hits = sum(1 for term in matched_terms if term in title_lower)
    title_bonus = min(title_hits * 0.15, 0.45)
    return round(coverage + exact_phrase_bonus + title_bonus, 6), matched_terms


def _projection_hmac_fields(
    document: VectorDocument,
) -> dict[str, str] | None:
    keys = (
        'serving_kind',
        'support_mode',
        'serving_identity_hmac',
        'serving_version_fingerprint',
        'model_content_hmac',
        'canonical_citation_projection_hmac',
    )
    values = {key: document.metadata.get(key) for key in keys}
    if any(not isinstance(value, str) or not value for value in values.values()):
        return None
    return {key: value for key, value in values.items() if isinstance(value, str)}


def _document_title(db: Session, document: VectorDocument) -> str | None:
    kind, separator, raw_id = document.document_id.partition(':')
    if not separator or not raw_id.isdigit() or int(raw_id) <= 0:
        return None
    row_id = int(raw_id)
    if kind == 'chunk':
        chunk = db.get(DocumentChunk, row_id)
        source = db.get(Source, chunk.source_id) if chunk is not None else None
        title = source.title if source is not None else None
    else:
        model = {
            'decision_record': DecisionRecord,
            'history_event': HistoryEvent,
            'timeline_event': TimelineEvent,
            'todo': Todo,
        }.get(kind)
        target = db.get(model, row_id) if model is not None else None
        title = target.title if target is not None else None
    if not isinstance(title, str) or '\x00' in title:
        return None
    return title
