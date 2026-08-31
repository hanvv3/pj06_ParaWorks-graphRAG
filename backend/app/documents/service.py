from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.connectors.base import SourceEvent
from backend.app.documents.parsers import ParsedDocument, ParsedDocumentChunk, ParserRun
from backend.app.ingestion.source_content_signature import (
    CanonicalSourceContentSignature,
    ServerParserPolicy,
)
from backend.app.models import (
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    Source,
)
from backend.app.rag.serving_generation import (
    RagServingGenerationLockedContext,
    assert_rag_serving_generation_context,
)

DEFAULT_CHUNK_MAX_CHARS = 1_200
SOURCE_SNIPPET_MAX_CHARS = 240
_SERVER_PARSED_METADATA_AUTHORITY_KEYS = frozenset(
    {
        'chunk_max_chars',
        'chunk_policy',
        'chunk_policy_version',
        'content_hash',
        'content_signature',
        'document_version',
        'mime_type',
        'page_number',
        'parser_name',
        'parser_policy_version',
        'parser_status',
        'parser_status_reason',
        'parser_version',
        'revision_id',
        'section_path',
        'source_snippet',
    }
)


def parsed_document_from_source_event(
    event: SourceEvent,
    *,
    server_signature: CanonicalSourceContentSignature | None = None,
    parser_policy: ServerParserPolicy | None = None,
    canonical_permission_level: str | None = None,
) -> ParsedDocument:
    metadata = event.raw_metadata
    document_version = str(metadata.get('document_version') or 'v1')
    revision_id = str(metadata.get('revision_id') or '')
    if (server_signature is None) != (parser_policy is None):
        raise ValueError('server signature and parser policy must be supplied together')
    if server_signature is not None and canonical_permission_level is None:
        raise ValueError('server parsing requires canonical permission')
    if server_signature is None:
        content_signature = str(
            metadata.get('content_signature') or f'{event.source_id}:{document_version}'
        )
        source_snippet = str(metadata.get('source_snippet') or _snippet(event.body))
        chunk_max_chars = _chunk_max_chars(metadata)
        parser_run = ParserRun(
            parser_name=str(
                metadata.get('parser_name') or f'{event.source_type}_source_event'
            ),
            parser_status=str(metadata.get('parser_status') or 'parsed'),
            parser_status_reason=_optional_string(
                metadata.get('parser_status_reason')
            ),
        )
        mime_type = str(metadata.get('mime_type') or event.source_type)
    else:
        content_signature = server_signature.signature
        source_snippet = _snippet(event.body)
        chunk_max_chars = parser_policy.chunk_max_chars
        parser_run = ParserRun(
            parser_name=parser_policy.parser_name,
            parser_status='parsed',
            parser_status_reason=None,
        )
        mime_type = parser_policy.mime_type
    return ParsedDocument(
        source_id=event.source_id,
        source_url=event.source_url,
        source_snippet=source_snippet,
        permission_level=(
            canonical_permission_level
            if canonical_permission_level is not None
            else event.permission_level
        ),
        mime_type=mime_type,
        document_version=document_version,
        revision_id=revision_id,
        content_signature=content_signature,
        parser_run=parser_run,
        chunks=_parsed_chunks_from_text(event.body, max_chars=chunk_max_chars),
    )


def persist_parsed_document(
    db: Session,
    *,
    source: Source,
    title: str,
    parsed: ParsedDocument,
    metadata: dict,
    server_signature: CanonicalSourceContentSignature | None = None,
    parser_policy: ServerParserPolicy | None = None,
    rag_generation_context: RagServingGenerationLockedContext | None = None,
) -> list[DocumentChunk]:
    if (server_signature is None) != (parser_policy is None):
        raise ValueError('server signature and parser policy must be supplied together')
    if server_signature is not None:
        assert_rag_serving_generation_context(db, rag_generation_context)
    document = db.scalar(select(Document).where(Document.source_id == source.id))
    if document is None:
        document = Document(
            source_id=source.id,
            title=title,
            current_version=parsed.document_version,
        )
        db.add(document)
        db.flush()
    else:
        document.title = title
        document.current_version = parsed.document_version

    version = DocumentVersion(
        document_id=document.id,
        version=parsed.document_version,
        body='\n'.join(chunk.text for chunk in parsed.chunks),
    )
    db.add(version)
    db.flush()

    if server_signature is not None:
        document.current_document_version_id = version.id

    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name=parsed.parser_run.parser_name,
        parser_status=parsed.parser_run.parser_status,
        parser_status_reason=parsed.parser_run.parser_status_reason,
        mime_type=parsed.mime_type,
        document_version_label=parsed.document_version,
        revision_id=parsed.revision_id,
        content_signature=parsed.content_signature,
        server_content_signature_schema=(
            server_signature.schema if server_signature is not None else None
        ),
        server_content_signature=(
            server_signature.signature if server_signature is not None else None
        ),
        parser_policy_version=(
            parser_policy.parser_policy_version if parser_policy is not None else None
        ),
        parser_version=(parser_policy.parser_version if parser_policy is not None else None),
        chunk_policy_version=(
            parser_policy.chunk_policy_version if parser_policy is not None else None
        ),
        chunk_count=len(parsed.chunks),
        metadata_={
            'source_id': parsed.source_id,
            'source_url': parsed.source_url,
            'permission_level': parsed.permission_level,
            'source_snippet': parsed.source_snippet,
        },
    )
    db.add(parser_run)
    db.flush()

    chunk_base_metadata = metadata
    if server_signature is not None:
        chunk_base_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in _SERVER_PARSED_METADATA_AUTHORITY_KEYS
        }
        chunk_base_metadata.update(
            {
                'parser_name': parser_run.parser_name,
                'parser_status': parser_run.parser_status,
                'parser_status_reason': parser_run.parser_status_reason,
                'mime_type': parser_run.mime_type,
                'document_version': parser_run.document_version_label,
                'revision_id': parser_run.revision_id,
                'content_signature': parser_run.content_signature,
            }
        )

    chunks: list[DocumentChunk] = []
    for parsed_chunk in parsed.chunks:
        chunk = DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            parser_run_id=parser_run.id if server_signature is not None else None,
            chunk_index=parsed_chunk.chunk_index,
            text=parsed_chunk.text,
            source_snippet=parsed_chunk.source_snippet,
            permission_level=parsed_chunk.permission_level,
            metadata_={
                **chunk_base_metadata,
                **parsed_chunk.metadata,
                'content_hash': parsed_chunk.content_hash,
            },
        )
        db.add(chunk)
        chunks.append(chunk)
    return chunks


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _chunk_max_chars(metadata: dict) -> int:
    value = metadata.get('chunk_max_chars')
    if value is None:
        return DEFAULT_CHUNK_MAX_CHARS
    try:
        return max(int(value), SOURCE_SNIPPET_MAX_CHARS)
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_MAX_CHARS


def _parsed_chunks_from_text(text: str, *, max_chars: int) -> list[ParsedDocumentChunk]:
    paragraphs = _paragraphs(text)
    if not paragraphs:
        return [ParsedDocumentChunk(chunk_index=0, text='', source_snippet='')]

    chunks: list[tuple[str, str | None]] = []
    current: list[str] = []
    current_section: str | None = None

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph) and current:
            chunks.append(('\n\n'.join(current), current_section))
            current = []
            current_section = None

        if current_section is None:
            current_section = _section_path(paragraph)

        candidate = '\n\n'.join([*current, paragraph]) if current else paragraph
        if len(candidate) <= max_chars:
            current.append(paragraph)
            continue

        if current:
            chunks.append(('\n\n'.join(current), current_section))
            current = []
            current_section = _section_path(paragraph)

        if len(paragraph) <= max_chars:
            current.append(paragraph)
            continue

        for part in _split_long_paragraph(paragraph, max_chars=max_chars):
            chunks.append((part, current_section))
        current_section = None

    if current:
        chunks.append(('\n\n'.join(current), current_section))

    return [
        ParsedDocumentChunk(
            chunk_index=index,
            text=chunk_text,
            source_snippet=_snippet(chunk_text),
            section_path=section_path,
        )
        for index, (chunk_text, section_path) in enumerate(chunks)
    ]


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    return [paragraph.strip() for paragraph in normalized.split('\n\n') if paragraph.strip()]


def _looks_like_heading(paragraph: str) -> bool:
    return '\n' not in paragraph and len(paragraph) <= 80 and not paragraph.endswith(('.', '?', '!', ':', ';'))


def _section_path(paragraph: str) -> str | None:
    first_line = paragraph.splitlines()[0].strip()
    if not first_line:
        return None
    return first_line[:120]


def _split_long_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    return [paragraph[index : index + max_chars].strip() for index in range(0, len(paragraph), max_chars)]


def _snippet(text: str) -> str:
    return ' '.join(text.split())[:SOURCE_SNIPPET_MAX_CHARS]
