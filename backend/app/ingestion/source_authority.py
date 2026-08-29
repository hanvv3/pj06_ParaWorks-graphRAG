from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.ingestion.source_content_signature import (
    SERVER_ALLOWED_MIME_TYPES,
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
    SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA,
    server_parser_run_matches_authority,
)
from backend.app.ingestion.source_versions import current_content_signature
from backend.app.models import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)


@dataclass(frozen=True, slots=True)
class ExactSourceAuthority:
    source: Source
    document: Document
    version: DocumentVersion
    parser_run: DocumentParserRun
    chunks: tuple[DocumentChunk, ...]


def resolve_exact_source_authority(
    db: Session,
    *,
    source: Source,
) -> ExactSourceAuthority | None:
    """Resolve the one exact server-owned bundle allowed to authorize serving."""
    signature = current_content_signature(source)
    if signature is None:
        return None
    documents = tuple(
        db.scalars(
            select(Document)
            .where(Document.source_id == source.id)
            .order_by(Document.id)
            .limit(2)
        ).all()
    )
    if len(documents) != 1:
        return None
    document = documents[0]
    if document.current_document_version_id is None:
        return None
    version = db.scalar(
        select(DocumentVersion).where(
            DocumentVersion.id == document.current_document_version_id,
            DocumentVersion.document_id == document.id,
        )
    )
    if version is None:
        return None
    parser_runs = tuple(
        db.scalars(
            select(DocumentParserRun)
            .where(
                DocumentParserRun.source_id == source.id,
                DocumentParserRun.document_id == document.id,
                DocumentParserRun.document_version_id == version.id,
            )
            .order_by(DocumentParserRun.id)
            .limit(2)
        ).all()
    )
    if len(parser_runs) != 1:
        return None
    parser_run = parser_runs[0]
    if (
        parser_run.document_version_label != version.version
        or not server_parser_run_matches_authority(
            source=source,
            parser_run=parser_run,
        )
    ):
        return None
    chunks = tuple(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.version_id == version.id)
            .order_by(DocumentChunk.chunk_index, DocumentChunk.id)
        ).all()
    )
    if (
        not chunks
        or parser_run.chunk_count != len(chunks)
        or [chunk.chunk_index for chunk in chunks] != list(range(len(chunks)))
        or any(
            chunk.source_id != source.id or chunk.parser_run_id != parser_run.id
            for chunk in chunks
        )
    ):
        return None
    return ExactSourceAuthority(
        source=source,
        document=document,
        version=version,
        parser_run=parser_run,
        chunks=chunks,
    )


def exact_authority_contains_chunk(
    authority: ExactSourceAuthority,
    chunk: DocumentChunk,
) -> bool:
    return any(current.id == chunk.id for current in authority.chunks)


def postgres_exact_source_authority_sql(
    *,
    source_alias: str,
    prefix: str,
    evidence_ref_sql: str | None = None,
    required_chunk_id_sql: str | None = None,
    dialect: Literal['postgresql', 'sqlite'] = 'postgresql',
) -> str:
    """Build a bounded SQL equivalent of ``resolve_exact_source_authority``."""
    if dialect == 'postgresql':
        mime_json_type = f"json_typeof({source_alias}.raw_metadata->'mime_type')"
        string_json_type = 'string'
    elif dialect == 'sqlite':
        mime_json_type = f"json_type({source_alias}.raw_metadata, '$.mime_type')"
        string_json_type = 'text'
    else:
        raise ValueError('exact source authority SQL dialect is unsupported')
    allowed_mimes = ', '.join(
        f"'{mime_type}'" for mime_type in sorted(SERVER_ALLOWED_MIME_TYPES)
    )
    expected_mime = f"""
        CASE {source_alias}.source_type
            WHEN 'gmail' THEN 'message/rfc822'
            WHEN 'calendar' THEN 'text/calendar'
            ELSE CASE
                WHEN lower(trim(COALESCE({source_alias}.raw_metadata->>'mime_type', '')))
                     IN ({allowed_mimes})
                THEN lower(trim({source_alias}.raw_metadata->>'mime_type'))
                ELSE 'application/octet-stream'
            END
        END
    """
    evidence_clause = ''
    if evidence_ref_sql is not None:
        evidence_clause = f"""
            AND (
                {evidence_ref_sql} = {source_alias}.server_content_signature
                OR (
                    {prefix}_parser_runs.revision_id IS NOT NULL
                    AND {prefix}_parser_runs.revision_id <> ''
                    AND {evidence_ref_sql} = {prefix}_parser_runs.revision_id
                )
            )
        """
    chunk_clause = ''
    if required_chunk_id_sql is not None:
        chunk_clause = f"""
            AND EXISTS (
                SELECT 1
                FROM document_chunks {prefix}_required_chunks
                WHERE {prefix}_required_chunks.id = {required_chunk_id_sql}
                  AND {prefix}_required_chunks.version_id = {prefix}_versions.id
                  AND {prefix}_required_chunks.source_id = {source_alias}.id
                  AND {prefix}_required_chunks.parser_run_id = {prefix}_parser_runs.id
            )
        """
    return f"""
        {source_alias}.server_content_signature_schema = '{SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA}'
        AND {source_alias}.server_content_signature IS NOT NULL
        AND {source_alias}.source_type IN ('gmail', 'gmail_attachment', 'drive', 'calendar')
        AND (
            {source_alias}.source_type NOT IN ('gmail_attachment', 'drive')
            OR {mime_json_type} IS NULL
            OR {mime_json_type} IN ('null', '{string_json_type}')
        )
        AND (
            SELECT count(*)
            FROM documents {prefix}_all_documents
            WHERE {prefix}_all_documents.source_id = {source_alias}.id
        ) = 1
        AND EXISTS (
            SELECT 1
            FROM documents {prefix}_documents
            JOIN document_versions {prefix}_versions
              ON {prefix}_versions.id = {prefix}_documents.current_document_version_id
             AND {prefix}_versions.document_id = {prefix}_documents.id
            JOIN document_parser_runs {prefix}_parser_runs
              ON {prefix}_parser_runs.document_id = {prefix}_documents.id
             AND {prefix}_parser_runs.document_version_id = {prefix}_versions.id
             AND {prefix}_parser_runs.source_id = {source_alias}.id
            WHERE {prefix}_documents.source_id = {source_alias}.id
              AND (
                  SELECT count(*)
                  FROM document_parser_runs {prefix}_all_runs
                  WHERE {prefix}_all_runs.document_id = {prefix}_documents.id
                    AND {prefix}_all_runs.document_version_id = {prefix}_versions.id
                    AND {prefix}_all_runs.source_id = {source_alias}.id
              ) = 1
              AND {prefix}_parser_runs.server_content_signature_schema = '{SERVER_SOURCE_CONTENT_SIGNATURE_SCHEMA}'
              AND {prefix}_parser_runs.server_content_signature = {source_alias}.server_content_signature
              AND {prefix}_parser_runs.content_signature = {source_alias}.server_content_signature
              AND {prefix}_parser_runs.parser_policy_version = '{SERVER_PARSER_POLICY_VERSION}'
              AND {prefix}_parser_runs.parser_name = 'server_' || {source_alias}.source_type || '_source_event'
              AND {prefix}_parser_runs.parser_status = 'parsed'
              AND {prefix}_parser_runs.parser_status_reason IS NULL
              AND {prefix}_parser_runs.parser_version = '{SERVER_PARSER_VERSION}'
              AND {prefix}_parser_runs.chunk_policy_version = '{SERVER_CHUNK_POLICY_VERSION}'
              AND {prefix}_parser_runs.mime_type = ({expected_mime})
              AND {prefix}_parser_runs.document_version_label = {prefix}_versions.version
              AND {prefix}_parser_runs.chunk_count > 0
              AND (
                  SELECT count(*)
                  FROM document_chunks {prefix}_all_chunks
                  WHERE {prefix}_all_chunks.version_id = {prefix}_versions.id
              ) = {prefix}_parser_runs.chunk_count
              AND NOT EXISTS (
                  SELECT 1
                  FROM document_chunks {prefix}_bad_chunks
                  WHERE {prefix}_bad_chunks.version_id = {prefix}_versions.id
                    AND (
                        {prefix}_bad_chunks.source_id <> {source_alias}.id
                        OR {prefix}_bad_chunks.parser_run_id <> {prefix}_parser_runs.id
                    )
              )
              AND (
                  SELECT count(DISTINCT {prefix}_indices.chunk_index)
                  FROM document_chunks {prefix}_indices
                  WHERE {prefix}_indices.version_id = {prefix}_versions.id
                    AND {prefix}_indices.chunk_index >= 0
                    AND {prefix}_indices.chunk_index < {prefix}_parser_runs.chunk_count
              ) = {prefix}_parser_runs.chunk_count
              {evidence_clause}
              {chunk_clause}
        )
    """
