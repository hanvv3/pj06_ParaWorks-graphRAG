import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

CanonicalSourceType = Literal['gmail', 'gmail_attachment', 'drive', 'calendar']
ReviewBatchMode = Literal['v2_explicit', 'legacy_inline']


class SourceRefLike(Protocol):
    source_type: CanonicalSourceType
    source_id: str
    version_or_signature: str


class CanonicalSourceLike(Protocol):
    source_type: str
    source_id: str
    raw_metadata: dict
    server_content_signature_schema: str | None
    server_content_signature: str | None


@dataclass(frozen=True)
class SourceVersionRef:
    source_type: CanonicalSourceType
    source_id: str
    version_or_signature: str


def current_content_signature(source: CanonicalSourceLike) -> str | None:
    if source.server_content_signature_schema != 'server-source-content:v1':
        return None
    value = source.server_content_signature
    return value if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) else None


def source_version_ref(source: CanonicalSourceLike) -> SourceVersionRef | None:
    if source.source_type not in {'gmail', 'gmail_attachment', 'drive', 'calendar'}:
        return None
    signature = current_content_signature(source)
    if signature is None or not source.source_id.startswith(f'{source.source_type}:'):
        return None
    return SourceVersionRef(
        source_type=cast(CanonicalSourceType, source.source_type),
        source_id=source.source_id,
        version_or_signature=signature,
    )


def source_version_refs(
    sources: Iterable[CanonicalSourceLike],
) -> list[SourceVersionRef]:
    refs = [ref for source in sources if (ref := source_version_ref(source)) is not None]
    return list(normalize_source_version_refs(refs))


def review_batch_marker_is_v2(
    source: CanonicalSourceLike,
    *,
    signature: str,
) -> bool:
    metadata = source.raw_metadata or {}
    return (
        metadata.get('review_batch_mode') == 'v2_explicit'
        and metadata.get('review_batch_signature') == signature
    )


def with_review_batch_marker(
    source: CanonicalSourceLike,
    *,
    mode: ReviewBatchMode,
    sync_job_id: str | None = None,
) -> dict:
    signature = current_content_signature(source)
    if signature is None:
        raise ValueError('source has no current content signature')
    metadata = {
        **(source.raw_metadata or {}),
        'review_batch_mode': mode,
        'review_batch_signature': signature,
    }
    if sync_job_id is not None:
        metadata['last_changed_sync_job_id'] = sync_job_id
    return metadata


def normalize_source_version_refs(
    refs: Iterable[SourceRefLike],
) -> tuple[SourceVersionRef, ...]:
    normalized: set[SourceVersionRef] = set()
    for ref in refs:
        source_type = ref.source_type
        source_id = ref.source_id
        version_or_signature = ref.version_or_signature
        if (
            not source_id
            or source_id != source_id.strip()
            or not source_id.startswith(f'{source_type}:')
            or not version_or_signature
            or version_or_signature != version_or_signature.strip()
        ):
            raise ValueError('invalid canonical source reference')
        normalized.add(
            SourceVersionRef(
                source_type=source_type,
                source_id=source_id,
                version_or_signature=version_or_signature,
            )
        )
    return tuple(
        sorted(
            normalized,
            key=lambda ref: (
                ref.source_type,
                ref.source_id,
                ref.version_or_signature,
            ),
        )
    )
