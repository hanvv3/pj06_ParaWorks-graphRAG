from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

CanonicalSourceType = Literal['gmail', 'gmail_attachment', 'drive', 'calendar']


class SourceRefLike(Protocol):
    source_type: CanonicalSourceType
    source_id: str
    version_or_signature: str


@dataclass(frozen=True)
class SourceVersionRef:
    source_type: CanonicalSourceType
    source_id: str
    version_or_signature: str


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
