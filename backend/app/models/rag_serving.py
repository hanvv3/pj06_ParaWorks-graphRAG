from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_positive_id(column: str, prefix: str) -> str:
    suffix_start = len(prefix) + 1
    return (
        f"substr({column}, 1, {len(prefix)}) = '{prefix}' AND "
        f'CAST(substr({column}, {suffix_start}) AS BIGINT) > 0 AND '
        f'CAST(CAST(substr({column}, {suffix_start}) AS BIGINT) AS TEXT) = '
        f'substr({column}, {suffix_start})'
    )


_CANONICAL_SERVING_IDENTITY = (
    "CASE WHEN serving_kind = 'raw_chunk' THEN "
    "support_mode = 'source_observation' AND "
    f'({_canonical_positive_id("serving_document_id", "chunk:")}) '
    "WHEN serving_kind = 'trusted_knowledge' THEN support_mode = 'trusted_fact' AND ("
    f'({_canonical_positive_id("serving_document_id", "decision_record:")}) OR '
    f'({_canonical_positive_id("serving_document_id", "history_event:")}) OR '
    f'({_canonical_positive_id("serving_document_id", "timeline_event:")}) OR '
    f'({_canonical_positive_id("serving_document_id", "todo:")})) '
    'ELSE false END'
)


class RagServingCorpusGeneration(Base):
    __tablename__ = 'rag_serving_corpus_generations'
    __table_args__ = (
        CheckConstraint('id = 1', name='ck_rag_serving_corpus_singleton'),
        CheckConstraint(
            'corpus_generation >= 0 AND vector_index_generation >= 0',
            name='ck_rag_serving_corpus_generations_nonnegative',
        ),
        CheckConstraint(
            'length(embedding_model) BETWEEN 1 AND 120 AND '
            'embedding_dimensions > 0 AND '
            "index_policy_version = 'rag-v2-serving-index:v1' AND "
            "pgvector_cosine_policy_version = 'pgvector-cosine-indexable:v1'",
            name='ck_rag_serving_corpus_policy_identity',
        ),
        CheckConstraint(
            'length(fingerprint_key_version) BETWEEN 1 AND 64 AND '
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_rag_serving_corpus_key_identity',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    corpus_generation: Mapped[int] = mapped_column(Integer, default=0)
    vector_index_generation: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(120))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    index_policy_version: Mapped[str] = mapped_column(String(64))
    pgvector_cosine_policy_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )


class RagLexicalServingProjection(Base):
    __tablename__ = 'rag_lexical_serving_projections'
    __table_args__ = (
        UniqueConstraint(
            'serving_document_id',
            name='uq_rag_lexical_serving_projection_document',
        ),
        UniqueConstraint(
            'serving_identity_hmac',
            name='uq_rag_lexical_serving_projection_identity_hmac',
        ),
        CheckConstraint(
            _CANONICAL_SERVING_IDENTITY,
            name='ck_rag_lexical_serving_identity',
        ),
        CheckConstraint(
            'length(serving_identity_hmac) = 64 AND '
            'length(serving_version_fingerprint) = 64 AND '
            'length(model_content_hmac) = 64 AND '
            'length(canonical_citation_projection_hmac) = 64 AND '
            'length(fingerprint_key_material_verifier) = 64',
            name='ck_rag_lexical_serving_hmacs',
        ),
        CheckConstraint(
            "lexical_contract_version = 'rag-keyword-lexical-compat:v1' AND "
            'length(fingerprint_key_version) BETWEEN 1 AND 64',
            name='ck_rag_lexical_serving_contract',
        ),
        CheckConstraint(
            "effective_permission IN ('public', 'internal', 'restricted') AND "
            "support_mode IN ('trusted_fact', 'source_observation')",
            name='ck_rag_lexical_serving_permission_support',
        ),
        CheckConstraint(
            'corpus_generation_id = 1 AND corpus_generation >= 0',
            name='ck_rag_lexical_serving_generation',
        ),
        Index(
            'ix_rag_lexical_serving_projection_permission',
            'corpus_generation',
            'effective_permission',
        ),
        Index(
            'ix_rag_lexical_serving_projection_support_mode',
            'corpus_generation',
            'support_mode',
        ),
        Index(
            'ix_rag_lexical_serving_projection_searchable_lower',
            'searchable_lower',
            postgresql_using='hash',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    corpus_generation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            'rag_serving_corpus_generations.id',
            name='fk_rag_lexical_serving_projection_corpus',
            ondelete='RESTRICT',
        ),
    )
    corpus_generation: Mapped[int] = mapped_column(Integer)
    serving_document_id: Mapped[str] = mapped_column(String(200))
    serving_kind: Mapped[str] = mapped_column(String(32))
    support_mode: Mapped[str] = mapped_column(String(32))
    effective_permission: Mapped[str] = mapped_column(String(32))
    serving_identity_hmac: Mapped[str] = mapped_column(String(64))
    serving_version_fingerprint: Mapped[str] = mapped_column(String(64))
    model_content_hmac: Mapped[str] = mapped_column(String(64))
    canonical_citation_projection_hmac: Mapped[str] = mapped_column(String(64))
    title_lower: Mapped[str] = mapped_column(Text)
    searchable_lower: Mapped[str] = mapped_column(Text)
    lexical_contract_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_version: Mapped[str] = mapped_column(String(64))
    fingerprint_key_material_verifier: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
