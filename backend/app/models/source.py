from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

_LOWER_HEX_64_REMAINDER = 'server_content_signature'
for _character in '0123456789abcdef':
    _LOWER_HEX_64_REMAINDER = (
        f"replace({_LOWER_HEX_64_REMAINDER}, '{_character}', '')"
    )


class Source(Base):
    __tablename__ = 'sources'
    __table_args__ = (
        CheckConstraint(
            '(server_content_signature IS NULL AND '
            'server_content_signature_schema IS NULL) OR '
            "(server_content_signature_schema = 'server-source-content:v1' AND "
            'length(server_content_signature) = 64 AND '
            f'{_LOWER_HEX_64_REMAINDER} = \'\')',
            name='ck_sources_server_content_signature_authority',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source_url: Mapped[str] = mapped_column(String(500))
    title: Mapped[str] = mapped_column(String(300))
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    raw_metadata: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    server_content_signature_schema: Mapped[str | None] = mapped_column(String(64))
    server_content_signature: Mapped[str | None] = mapped_column(String(64))
    connector_content_signature: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    documents: Mapped[list['Document']] = relationship(back_populates='source', cascade='all, delete-orphan')


class Document(Base):
    __tablename__ = 'documents'
    __table_args__ = (
        ForeignKeyConstraint(
            ['current_document_version_id', 'id'],
            ['document_versions.id', 'document_versions.document_id'],
            name='fk_documents_current_version_same_document',
            use_alter=True,
            deferrable=True,
            initially='DEFERRED',
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement='ignore_fk'
    )
    source_id: Mapped[int] = mapped_column(ForeignKey('sources.id'), index=True)
    title: Mapped[str] = mapped_column(String(300))
    current_version: Mapped[str] = mapped_column(String(64), default='v1')
    current_document_version_id: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[Source] = relationship(back_populates='documents')
    versions: Mapped[list['DocumentVersion']] = relationship(
        back_populates='document',
        cascade='all, delete-orphan',
        foreign_keys='DocumentVersion.document_id',
    )


class DocumentVersion(Base):
    __tablename__ = 'document_versions'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'document_id',
            name='uq_document_versions_id_document',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey('documents.id'), index=True)
    version: Mapped[str] = mapped_column(String(64), default='v1')
    body: Mapped[str] = mapped_column(Text)
    document: Mapped[Document] = relationship(
        back_populates='versions', foreign_keys=[document_id]
    )
    chunks: Mapped[list['DocumentChunk']] = relationship(back_populates='version', cascade='all, delete-orphan')
    parser_runs: Mapped[list['DocumentParserRun']] = relationship(back_populates='document_version', cascade='all, delete-orphan')


class DocumentParserRun(Base):
    __tablename__ = 'document_parser_runs'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'document_version_id',
            'source_id',
            name='uq_document_parser_runs_identity',
        ),
        CheckConstraint(
            '(server_content_signature IS NULL AND '
            'server_content_signature_schema IS NULL AND '
            'parser_policy_version IS NULL AND parser_version IS NULL AND '
            'chunk_policy_version IS NULL) OR '
            "(server_content_signature_schema = 'server-source-content:v1' AND "
            'length(server_content_signature) = 64 AND '
            f'{_LOWER_HEX_64_REMAINDER} = \'\' AND '
            'parser_policy_version IS NOT NULL AND parser_version IS NOT NULL AND '
            'chunk_policy_version IS NOT NULL)',
            name='ck_document_parser_runs_c5_identity',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey('documents.id'), index=True)
    document_version_id: Mapped[int] = mapped_column(ForeignKey('document_versions.id'), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey('sources.id'), index=True)
    parser_name: Mapped[str] = mapped_column(String(128), index=True)
    parser_status: Mapped[str] = mapped_column(String(32), index=True)
    parser_status_reason: Mapped[str | None] = mapped_column(String(300), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(160), default='')
    document_version_label: Mapped[str] = mapped_column(String(64), default='v1')
    revision_id: Mapped[str] = mapped_column(String(128), default='')
    content_signature: Mapped[str] = mapped_column(String(300), default='')
    server_content_signature_schema: Mapped[str | None] = mapped_column(String(64))
    server_content_signature: Mapped[str | None] = mapped_column(String(64))
    parser_policy_version: Mapped[str | None] = mapped_column(String(64))
    parser_version: Mapped[str | None] = mapped_column(String(64))
    chunk_policy_version: Mapped[str | None] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    metadata_: Mapped[dict] = mapped_column('metadata', MutableDict.as_mutable(JSON), default=dict)
    document_version: Mapped[DocumentVersion] = relationship(back_populates='parser_runs')


class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    __table_args__ = (
        UniqueConstraint(
            'id',
            'version_id',
            'source_id',
            'parser_run_id',
            name='uq_document_chunks_exact_identity',
        ),
        UniqueConstraint(
            'parser_run_id',
            'chunk_index',
            name='uq_document_chunks_parser_run_index',
        ),
        ForeignKeyConstraint(
            ['parser_run_id', 'version_id', 'source_id'],
            [
                'document_parser_runs.id',
                'document_parser_runs.document_version_id',
                'document_parser_runs.source_id',
            ],
            name='fk_document_chunks_parser_run_identity',
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey('document_versions.id'), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey('sources.id'), index=True)
    parser_run_id: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    source_snippet: Mapped[str] = mapped_column(Text)
    permission_level: Mapped[str] = mapped_column(String(32), index=True)
    metadata_: Mapped[dict] = mapped_column('metadata', MutableDict.as_mutable(JSON), default=dict)
    version: Mapped[DocumentVersion] = relationship(back_populates='chunks')
