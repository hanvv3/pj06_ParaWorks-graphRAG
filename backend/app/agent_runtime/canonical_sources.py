from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.ingestion.source_authority import resolve_exact_source_authority
from backend.app.ingestion.source_versions import (
    SourceVersionRef,
    current_content_signature,
)
from backend.app.models.source import Source

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

    authority = resolve_exact_source_authority(db, source=source)
    if authority is None:
        _raise_evidence_changed()
    document_version = authority.version
    parser_run = authority.parser_run
    chunks = authority.chunks

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
