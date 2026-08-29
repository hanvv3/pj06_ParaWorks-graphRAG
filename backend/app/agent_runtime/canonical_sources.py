from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_content_signature import (
    server_parser_run_matches_authority,
)
from backend.app.ingestion.source_versions import (
    SourceVersionRef,
    current_content_signature,
)
from backend.app.models.source import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)

CANONICAL_SOURCE_FINGERPRINT_SCHEMA = 'canonical-source-version:v1'
CANONICAL_SOURCE_FINGERPRINT_POLICY = 'review-evidence-resolution:v1'


class ReviewWorkflowPreflightError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ResolvedSourceVersion:
    source_type: str
    canonical_table: str
    canonical_row_id: int
    document_version_id: int | None
    external_revision: str | None
    content_signature: str
    permission_level: str
    content_fingerprint: str


def build_keyed_fingerprint(
    value: object,
    *,
    settings: Settings,
    schema_version: str,
    policy_version: str,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        value,
        secret=secret,
        schema_version=schema_version,
        policy_version=policy_version,
    )


def resolve_source_versions(
    db: Session,
    *,
    refs: tuple[SourceVersionRef, ...],
    actor: DemoUser,
    settings: Settings,
) -> tuple[ResolvedSourceVersion, ...]:
    return tuple(
        _resolve_source_version(db, ref=ref, actor=actor, settings=settings)
        for ref in refs
    )


def _resolve_source_version(
    db: Session,
    *,
    ref: SourceVersionRef,
    actor: DemoUser,
    settings: Settings,
) -> ResolvedSourceVersion:
    source = db.scalars(
        select(Source).where(Source.source_id == ref.source_id)
    ).first()
    if source is None or source.permission_level not in actor.permission_levels:
        raise ReviewWorkflowPreflightError(
            'not_found',
            'source reference was not found',
        )
    if source.source_type != ref.source_type:
        raise ReviewWorkflowPreflightError(
            'invalid_input',
            'canonical source type does not match the request',
        )

    content_signature = current_content_signature(source)
    if content_signature is None:
        _raise_evidence_changed()
    if content_signature != ref.version_or_signature:
        _raise_evidence_changed()

    documents = tuple(
        db.scalars(
            select(Document)
            .where(Document.source_id == source.id)
            .order_by(Document.id)
        ).all()
    )
    if len(documents) != 1:
        _raise_evidence_changed()
    document = documents[0]
    if document.current_document_version_id is None:
        _raise_evidence_changed()
    document_version = db.scalar(
        select(DocumentVersion).where(
            DocumentVersion.id == document.current_document_version_id,
            DocumentVersion.document_id == document.id,
        )
    )
    if document_version is None:
        _raise_evidence_changed()
    parser_runs = tuple(
        db.scalars(
            select(DocumentParserRun)
            .where(
                DocumentParserRun.source_id == source.id,
                DocumentParserRun.document_id == document.id,
                DocumentParserRun.document_version_id == document_version.id,
            )
            .order_by(DocumentParserRun.id)
        ).all()
    )
    if len(parser_runs) != 1:
        _raise_evidence_changed()
    parser_run = parser_runs[0]
    if (
        parser_run.document_version_label != document_version.version
        or not server_parser_run_matches_authority(
            source_type=source.source_type,
            server_content_signature=content_signature,
            parser_run=parser_run,
        )
    ):
        _raise_evidence_changed()
    chunks = tuple(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.version_id == document_version.id)
            .order_by(DocumentChunk.chunk_index, DocumentChunk.id)
        ).all()
    )
    if (
        not chunks
        or parser_run.chunk_count != len(chunks)
        or [chunk.chunk_index for chunk in chunks] != list(range(len(chunks)))
        or any(
            chunk.source_id != source.id
            or chunk.parser_run_id != parser_run.id
            for chunk in chunks
        )
    ):
        _raise_evidence_changed()

    external_revision = _optional_string(parser_run.revision_id)
    fingerprint_value = {
        'canonical_row_id': source.id,
        'canonical_table': 'sources',
        'chunk_ids': [chunk.id for chunk in chunks],
        'content_signature': content_signature,
        'document_version': document_version.version,
        'document_version_id': document_version.id,
        'external_revision': external_revision,
        'parser_name': parser_run.parser_name,
        'parser_policy_version': parser_run.parser_policy_version,
        'parser_signature': parser_run.content_signature,
        'parser_status': parser_run.parser_status,
        'parser_version': parser_run.parser_version,
        'parser_version_label': parser_run.document_version_label,
        'chunk_policy_version': parser_run.chunk_policy_version,
        'mime_type': parser_run.mime_type,
        'permission_level': source.permission_level,
        'source_type': ref.source_type,
    }
    return ResolvedSourceVersion(
        source_type=ref.source_type,
        canonical_table='sources',
        canonical_row_id=source.id,
        document_version_id=document_version.id,
        external_revision=external_revision,
        content_signature=content_signature,
        permission_level=source.permission_level,
        content_fingerprint=build_keyed_fingerprint(
            fingerprint_value,
            settings=settings,
            schema_version=CANONICAL_SOURCE_FINGERPRINT_SCHEMA,
            policy_version=CANONICAL_SOURCE_FINGERPRINT_POLICY,
        ),
    )


def _raise_evidence_changed() -> None:
    raise ReviewWorkflowPreflightError(
        'evidence_changed',
        'source evidence changed; synchronize again',
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
