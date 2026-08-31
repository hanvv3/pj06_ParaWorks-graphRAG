from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base
from backend.app.models.rag_serving import _CANONICAL_SERVING_IDENTITY

_VECTOR_CANONICAL_SERVING_IDENTITY = _CANONICAL_SERVING_IDENTITY.replace(
    'serving_document_id', 'document_id'
)


class VectorIndexState(Base):
    __tablename__ = 'vector_index_states'
    __table_args__ = (
        UniqueConstraint(
            'document_id',
            'embedding_model',
            name='uq_vector_index_state_document_model',
        ),
        CheckConstraint(
            '(corpus_generation_id IS NULL AND serving_kind IS NULL AND '
            'support_mode IS NULL AND effective_permission IS NULL AND '
            'serving_identity_hmac IS NULL AND serving_version_fingerprint IS NULL AND '
            'model_content_hmac IS NULL AND canonical_citation_projection_hmac IS NULL AND '
            'index_policy_version IS NULL AND pgvector_cosine_policy_version IS NULL AND '
            'cosine_indexable IS NULL AND vector_index_generation IS NULL AND '
            'vector_index_state_hmac IS NULL AND fingerprint_key_version IS NULL AND '
            'fingerprint_key_material_verifier IS NULL) OR '
            '(corpus_generation_id = 1 AND serving_kind IS NOT NULL AND '
            'support_mode IS NOT NULL AND effective_permission IS NOT NULL AND '
            'serving_identity_hmac IS NOT NULL AND '
            'serving_version_fingerprint IS NOT NULL AND model_content_hmac IS NOT NULL AND '
            'canonical_citation_projection_hmac IS NOT NULL AND '
            'index_policy_version IS NOT NULL AND '
            'pgvector_cosine_policy_version IS NOT NULL AND cosine_indexable IS NOT NULL AND '
            'vector_index_generation IS NOT NULL AND vector_index_state_hmac IS NOT NULL AND '
            'fingerprint_key_version IS NOT NULL AND '
            'fingerprint_key_material_verifier IS NOT NULL)',
            name='ck_vector_index_state_d_provenance',
        ),
        CheckConstraint(
            'serving_kind IS NULL OR ('
            "index_policy_version = 'rag-v2-serving-index:v1' AND "
            "pgvector_cosine_policy_version = 'pgvector-cosine-indexable:v1' AND "
            'length(embedding_model) BETWEEN 1 AND 120 AND '
            'embedding_dimensions > 0 AND length(content_hash) = 64 AND '
            'vector_index_generation >= 0 AND '
            "effective_permission IN ('public', 'internal', 'restricted') AND "
            f'({_VECTOR_CANONICAL_SERVING_IDENTITY}) AND '
            "((status = 'indexed' AND cosine_indexable = true) OR status = 'tombstoned'))",
            name='ck_vector_index_state_d_policy',
        ),
        CheckConstraint(
            'serving_kind IS NULL OR ('
            'length(serving_identity_hmac) = 64 AND '
            'length(serving_version_fingerprint) = 64 AND '
            'length(model_content_hmac) = 64 AND '
            'length(canonical_citation_projection_hmac) = 64 AND '
            'length(vector_index_state_hmac) = 64 AND '
            'length(fingerprint_key_version) BETWEEN 1 AND 64 AND '
            'length(fingerprint_key_material_verifier) = 64)',
            name='ck_vector_index_state_d_hmacs',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(String(200), index=True)
    embedding_model: Mapped[str] = mapped_column(String(120), index=True)
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default='indexed', index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    corpus_generation_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(
            'rag_serving_corpus_generations.id',
            name='fk_vector_index_state_rag_serving_corpus',
            ondelete='RESTRICT',
        ),
        nullable=True,
    )
    serving_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    support_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    effective_permission: Mapped[str | None] = mapped_column(String(32), nullable=True)
    serving_identity_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True)
    serving_version_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    model_content_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True)
    canonical_citation_projection_hmac: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    index_policy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pgvector_cosine_policy_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    cosine_indexable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    vector_index_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vector_index_state_hmac: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    fingerprint_key_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    fingerprint_key_material_verifier: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
