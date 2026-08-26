from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_versions import SourceVersionRef
from backend.app.models.source import (
    Document,
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

    metadata = source.raw_metadata or {}
    content_signature = metadata.get('content_signature')
    if not isinstance(content_signature, str) or not content_signature:
        raise ReviewWorkflowPreflightError(
            'evidence_changed',
            'source evidence changed; synchronize again',
        )
    if content_signature != ref.version_or_signature:
        raise ReviewWorkflowPreflightError(
            'evidence_changed',
            'source evidence changed; synchronize again',
        )

    document_version_id: int | None = None
    document_version_label: str | None = None
    parser_name: str | None = None
    parser_status: str | None = None
    parser_signature: str | None = None
    external_revision = _optional_string(
        metadata.get('revision_id') or metadata.get('external_revision')
    )
    document = db.scalars(
        select(Document)
        .where(Document.source_id == source.id)
        .order_by(Document.id.desc())
    ).first()
    if document is not None:
        document_version = db.scalars(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document.id,
                DocumentVersion.version == document.current_version,
            )
        ).first()
        if document_version is None:
            raise ReviewWorkflowPreflightError(
                'evidence_changed',
                'source evidence changed; synchronize again',
            )
        document_version_id = document_version.id
        document_version_label = document_version.version
        parser_run = db.scalars(
            select(DocumentParserRun)
            .where(
                DocumentParserRun.source_id == source.id,
                DocumentParserRun.document_id == document.id,
                DocumentParserRun.document_version_id == document_version.id,
            )
            .order_by(
                DocumentParserRun.finished_at.desc(),
                DocumentParserRun.id.desc(),
            )
        ).first()
        if parser_run is not None:
            parser_name = parser_run.parser_name
            parser_status = parser_run.parser_status
            parser_signature = parser_run.content_signature or None
            external_revision = parser_run.revision_id or external_revision

    fingerprint_value = {
        'canonical_row_id': source.id,
        'canonical_table': 'sources',
        'content_signature': content_signature,
        'document_version': document_version_label,
        'document_version_id': document_version_id,
        'external_revision': external_revision,
        'parser_name': parser_name,
        'parser_signature': parser_signature,
        'parser_status': parser_status,
        'permission_level': source.permission_level,
        'source_type': ref.source_type,
    }
    return ResolvedSourceVersion(
        source_type=ref.source_type,
        canonical_table='sources',
        canonical_row_id=source.id,
        document_version_id=document_version_id,
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


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
