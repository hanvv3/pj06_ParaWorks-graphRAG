from typing import Any

from backend.app.agent_runtime.contracts import EvidencePacket

_APPROVED_METADATA_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('source_type', ('source_type',)),
    ('chunk_id', ('chunk_id',)),
    ('parser_run_id', ('parser_run_id',)),
    ('parser_name', ('parser_name',)),
    ('parser_status', ('parser_status',)),
    ('parser_status_reason', ('parser_status_reason',)),
    ('uncertainty_reason', ('uncertainty_reason',)),
    ('mime_type', ('mime_type',)),
    ('document_version', ('document_version', 'document_version_label')),
    ('revision_id', ('revision_id',)),
    ('content_signature', ('content_signature',)),
    ('content_hash', ('content_hash',)),
    ('section_path', ('section_path',)),
    ('page_number', ('page_number',)),
    ('fallback_body', ('fallback_body',)),
    ('calendar_id', ('calendar_id',)),
    ('calendar_name', ('calendar_summary', 'calendar_name')),
    ('calendar_start', ('event_start', 'start')),
    ('calendar_end', ('event_end', 'end')),
    ('calendar_location', ('location',)),
    ('calendar_organizer', ('organizer_email',)),
    ('calendar_attendee_domains', ('attendee_domains',)),
    ('event_context_key', ('event_context_key',)),
    ('event_status', ('event_status',)),
)


def render_bounded_evidence_rows(
    packet: EvidencePacket,
    *,
    max_input_chars: int,
) -> list[dict[str, str]]:
    """Render only approved evidence values inside one shared character window."""

    rows: list[dict[str, str]] = []
    remaining_chars = max(0, max_input_chars)
    for message in packet.messages:
        if remaining_chars <= 0:
            break
        metadata_values: list[tuple[str, Any]] = []
        for output_name, source_names in _APPROVED_METADATA_FIELDS:
            raw_value = _first_metadata_value(message.metadata, source_names)
            if output_name == 'source_type' and raw_value is None:
                raw_value = packet.source_type
            metadata_values.append((output_name, raw_value))
        values: list[tuple[str, Any]] = [
            ('source_id', message.source_id),
            ('source_url', message.source_url),
            ('permission_level', message.permission_level),
            ('timestamp', message.timestamp),
            ('author', message.author),
            ('source_snippet', message.source_snippet),
            *metadata_values,
            ('text', message.text),
        ]
        row: dict[str, str] = {}
        for field_name, raw_value in values:
            canonical_value = _canonical_value(raw_value)
            if not canonical_value:
                continue
            bounded_value = canonical_value[:remaining_chars]
            if bounded_value:
                row[field_name] = bounded_value
                remaining_chars -= len(bounded_value)
            if remaining_chars <= 0:
                break
        if row:
            rows.append(row)
    return rows


def _first_metadata_value(
    metadata: dict,
    source_names: tuple[str, ...],
) -> Any:
    for source_name in source_names:
        value = metadata.get(source_name)
        if value is not None:
            return value
    return None


def _canonical_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        items = [
            _canonical_value(item)
            for item in value
            if isinstance(item, (str, bool, int, float))
        ]
        return ','.join(item for item in items if item)
    return ''
