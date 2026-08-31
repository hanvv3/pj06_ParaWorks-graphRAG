from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.models import RagServingCorpusGeneration
from backend.app.rag.embeddings import (
    DeterministicHashEmbeddingModel,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
)
from backend.app.rag.indexing import (
    EmbeddingBudgetExceededError,
    PreviewVectorIndexWriter,
    VectorIndexWriter,
    build_rag_index_documents,
    build_rag_v2_index_documents,
    index_changed_vector_documents,
)
from backend.app.rag.pgvector_store import PgVectorConfig, PgVectorStore


class ReindexConfigurationError(ValueError):
    pass


def run_reindex(*, db: Session, settings: Settings, dry_run: bool) -> dict:
    writer, embedding_model, embedding_model_name, storage_backend, persist_state = reindex_components(
        db=db,
        settings=settings,
        dry_run=dry_run,
    )
    documents = build_rag_index_documents(db)
    try:
        result = index_changed_vector_documents(
            db=db,
            documents=documents,
            writer=writer,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
            persist_state=persist_state,
            embedding_cost_per_1m_tokens=settings.openai_embedding_input_cost_per_1m_tokens,
            max_embedding_cost_usd=settings.rag_embedding_max_estimated_cost_usd,
            enforce_embedding_budget=not dry_run,
            settings=settings,
        )
    except EmbeddingBudgetExceededError as exc:
        decision = exc.decision
        raise ReindexConfigurationError(
            'embedding budget exceeded: '
            f"estimated=${float(decision['estimated_cost_usd']):.6f} "
            f"> budget=${float(decision['budget_limit_usd'] or 0):.6f}"
        ) from exc
    embedding_budget = dict(result.embedding_budget or {})
    if dry_run and embedding_budget:
        embedding_budget['embedding_model'] = settings.openai_embedding_model

    return {
        'dry_run': dry_run,
        'indexed_count': result.indexed_count,
        'skipped_count': result.skipped_count,
        'saved_embedding_calls': result.saved_embedding_calls,
        'embedding_request_count': result.embedding_request_count,
        'embedding_prompt_tokens': result.embedding_prompt_tokens,
        'embedding_total_tokens': result.embedding_total_tokens,
        'embedding_dimensions': result.embedding_dimensions,
        'document_ids': result.document_ids,
        'skipped_document_ids': result.skipped_document_ids or [],
        'incremental': True,
        'storage_backend': storage_backend,
        'embedding_budget': embedding_budget,
        'parser_status_counts': _parser_status_counts(documents),
    }


def run_rag_v2_reindex(
    *,
    db: Session,
    settings: Settings,
    dry_run: bool,
    operator_authorized: bool,
) -> dict:
    if not dry_run and not operator_authorized:
        raise ReindexConfigurationError(
            'RAG V2 live reindex requires separate operator authorization.'
        )
    writer, embedding_model, embedding_model_name, storage_backend, persist_state = (
        reindex_components(
            db=db,
            settings=settings,
            dry_run=dry_run,
        )
    )
    documents = build_rag_v2_index_documents(db, settings=settings)
    try:
        result = index_changed_vector_documents(
            db=db,
            documents=documents,
            writer=writer,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
            persist_state=persist_state,
            embedding_cost_per_1m_tokens=(
                settings.openai_embedding_input_cost_per_1m_tokens
            ),
            max_embedding_cost_usd=(
                settings.rag_embedding_max_estimated_cost_usd
            ),
            enforce_embedding_budget=not dry_run,
            settings=settings,
            rag_v2=True,
            operator_authorized=operator_authorized,
        )
    except EmbeddingBudgetExceededError as exc:
        decision = exc.decision
        raise ReindexConfigurationError(
            'embedding budget exceeded: '
            f"estimated=${float(decision['estimated_cost_usd']):.6f} "
            f"> budget=${float(decision['budget_limit_usd'] or 0):.6f}"
        ) from exc
    generation = db.get(RagServingCorpusGeneration, 1)
    embedding_budget = dict(result.embedding_budget or {})
    if dry_run and embedding_budget:
        embedding_budget['embedding_model'] = settings.openai_embedding_model
    return {
        'dry_run': dry_run,
        'serving_lane': 'rag_v2',
        'indexed_count': result.indexed_count,
        'skipped_count': result.skipped_count,
        'tombstoned_count': result.tombstoned_count,
        'saved_embedding_calls': result.saved_embedding_calls,
        'embedding_request_count': result.embedding_request_count,
        'embedding_prompt_tokens': result.embedding_prompt_tokens,
        'embedding_total_tokens': result.embedding_total_tokens,
        'embedding_dimensions': result.embedding_dimensions,
        'document_ids': result.document_ids,
        'skipped_document_ids': result.skipped_document_ids or [],
        'incremental': True,
        'storage_backend': storage_backend,
        'embedding_budget': embedding_budget,
        'parser_status_counts': _parser_status_counts(documents),
        'corpus_generation': (
            generation.corpus_generation if generation is not None else 0
        ),
        'vector_index_generation': (
            generation.vector_index_generation if generation is not None else 0
        ),
    }


def _parser_status_counts(documents: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        parser_status = document.metadata.get('parser_status')
        if not parser_status:
            continue
        key = str(parser_status)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def reindex_components(
    *,
    db: Session,
    settings: Settings,
    dry_run: bool,
) -> tuple[VectorIndexWriter, object, str, str, bool]:
    if dry_run:
        return (
            PreviewVectorIndexWriter(),
            DeterministicHashEmbeddingModel(dimensions=16),
            'deterministic-hash:v1',
            'preview',
            False,
        )

    if db.bind is None or db.bind.dialect.name != 'postgresql':
        raise ReindexConfigurationError('pgvector writes require a PostgreSQL database.')
    if not settings.openai_api_key:
        raise ReindexConfigurationError('OPENAI_API_KEY is required for pgvector writes.')

    writer = PgVectorStore(
        session=db,
        config=PgVectorConfig(embedding_dimensions=settings.openai_embedding_dimensions),
        settings=settings,
    )
    writer.ensure_schema()
    db.commit()
    return (
        writer,
        OpenAIEmbeddingModel(
            config=OpenAIEmbeddingConfig(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
                timeout_seconds=settings.openai_embedding_timeout_seconds,
            )
        ),
        settings.openai_embedding_model,
        'pgvector',
        True,
    )
