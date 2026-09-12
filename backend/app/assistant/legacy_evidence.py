"""Current V1 snapshots and the only new-write signer; no D eligibility."""

from backend.app.models import DocumentChunk, Source


def current_legacy_evidence(db, serving_document_id):
    from backend.app.assistant.service import _serving_content_hash
    from backend.app.knowledge.serving_text import canonical_knowledge_text
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
        knowledge_model_for_type,
    )

    kind, raw_id = serving_document_id.split(':')
    identifier = int(raw_id)
    if identifier <= 0 or str(identifier) != raw_id:
        raise ValueError('legacy serving identity is invalid')
    current = TrustedServingEligibilityService(db).for_document(serving_document_id)
    if not current.eligible:
        raise ValueError('legacy evidence is unavailable')
    if kind == 'chunk':
        chunk = db.get(DocumentChunk, identifier)
        source = db.get(Source, chunk.source_id) if chunk is not None else None
        if chunk is None or source is None:
            raise ValueError('legacy source is unavailable')
        public_id, source_type, text = source.source_id, source.source_type, chunk.text
        links, snippets = [source.source_url], [chunk.source_snippet]
        serving_kind = 'raw_chunk'
    else:
        target = db.get(knowledge_model_for_type(kind), identifier)
        if target is None:
            raise ValueError('legacy knowledge is unavailable')
        public_id, source_type = serving_document_id, kind
        text = canonical_knowledge_text(kind, target)
        links, snippets = list(target.source_links), list(target.source_snippets)
        serving_kind = 'trusted_knowledge'
    if not links or len(links) != len(snippets):
        raise ValueError('legacy evidence arrays are unavailable')
    version = _serving_content_hash(
        source_id=public_id,
        text=text,
        source_url=links[0],
        source_snippet=snippets[0],
        permission_level=current.effective_permission,
    )
    return (
        public_id,
        source_type,
        text,
        links,
        snippets,
        current.effective_permission,
        version,
        serving_kind,
    )


def sign_legacy_message(*, db, message, settings, actor=None):
    from backend.app.assistant.legacy_integrity_v3 import sign_message

    return sign_message(db=db, message=message, settings=settings, actor=actor)
