"""Keyed V1 content snapshots. Integrity never confers D serving eligibility."""

import struct

from sqlalchemy import select

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import exact_utf8_bytes
from backend.app.models import (
    AssistantMessageEvidenceDependency,
    AssistantMessageKnowledgeEvidenceRef,
    AutoReviewRuntimeKeyState,
    DocumentChunk,
    Source,
)
from backend.app.rag.evidence_projection import (
    build_v1_selected_evidence_projection_hmac,
)
from backend.app.rag.serving_contracts import (
    build_canonical_citation_projection_hmac,
    build_model_content_hmac,
)


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
    version = _serving_content_hash(source_id=public_id, text=text,
        source_url=links[0], source_snippet=snippets[0],
        permission_level=current.effective_permission)
    return public_id, source_type, text, links, snippets, current.effective_permission, version, serving_kind


def sign_legacy_message(*, db, message, settings):
    """Sign current resolver snapshots after legacy liveness and before commit."""
    secret, version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    runtime = db.scalar(select(AutoReviewRuntimeKeyState).where(
        AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'))
    if (runtime is None or not runtime.ready or runtime.fingerprint_key_version != version
        or runtime.fingerprint_key_material_verifier != verifier):
        raise ValueError('legacy evidence key runtime unavailable')
    def fp(value, schema, policy='assistant-evidence:v1'):
        return keyed_fingerprint(value, secret=secret, schema_version=schema, policy_version=policy)
    old = list(db.scalars(select(AssistantMessageEvidenceDependency).where(
        AssistantMessageEvidenceDependency.assistant_message_id == message.id
    ).order_by(AssistantMessageEvidenceDependency.candidate_ordinal)))
    if not old or len(old) != len(message.citations):
        raise ValueError('legacy selected evidence is incomplete')
    children, citation_hmacs = [], []
    for ordinal, (snapshot, citation) in enumerate(zip(old, message.citations, strict=True)):
        public_id, kind, text, links, snippets, permission, legacy_hash, serving_kind = current_legacy_evidence(db, snapshot.serving_document_id)
        if snapshot.serving_content_hash != legacy_hash or snapshot.permission_level != permission:
            raise ValueError('legacy serving snapshot changed')
        expected = {'source_id': public_id, 'source_type': kind, 'source_url': links[0],
                    'source_snippet': snippets[0], 'permission_level': permission}
        if any(citation.get(key) != value for key, value in expected.items()):
            raise ValueError('legacy citation snapshot changed')
        citation_hmac = fp({
            'source_id_bytes': exact_utf8_bytes(public_id),
            'source_url_bytes': exact_utf8_bytes(links[0]),
            'source_type_bytes': exact_utf8_bytes(kind), 'permission_level': permission,
            'source_snippet_bytes': exact_utf8_bytes(snippets[0]),
            'relevance_score_binary64_be_hex': struct.pack('>d', citation['relevance_score']).hex(),
            'matched_terms_bytes': [exact_utf8_bytes(term) for term in citation['matched_terms']],
        }, 'rag-v1-evidence-projection:citation:v1', 'rag-v1-evidence-projection:v1')
        citation_hmacs.append(citation_hmac)
        model_hmac = build_model_content_hmac(serving_kind=serving_kind, model_content=text, settings=settings)
        canonical = build_canonical_citation_projection_hmac(public_source_id=public_id,
            public_source_type=kind, source_url=links[0], source_snippet=snippets[0],
            effective_permission=permission, settings=settings)
        identity = fp({
            'canonical_citation_projection_hmac': canonical, 'effective_permission': permission,
            'legacy_public_source_id_bytes': exact_utf8_bytes(public_id),
            'legacy_source_links_bytes': [exact_utf8_bytes(value) for value in links],
            'legacy_source_snippets_bytes': [exact_utf8_bytes(value) for value in snippets],
            'model_content_hmac': model_hmac, 'serving_version_fingerprint': legacy_hash,
        }, 'assistant-legacy-dependency-snapshot:v1')
        child_hmac = fp({
            'approval_link_id': None, 'approval_provenance_hmac': None,
            'candidate_ordinal': ordinal, 'canonical_citation_projection_hmac': canonical,
            'dependency_kind': 'legacy_unbound', 'dependency_role': 'selected_citation',
            'dependency_serving_scope': 'legacy_v1_only', 'effective_permission': permission,
            'evidence_link_ids': [], 'evidence_link_set_hmac': None,
            'legacy_dependency_identity_hmac': identity, 'model_content_hmac': model_hmac,
            'raw_document_chunk_id': None, 'selected_v1_citation_projection_hmac': citation_hmac,
            'serving_identity_hmac': None, 'serving_version_fingerprint': None,
            'support_mode': None, 'trusted_knowledge_id': None,
        }, 'assistant-dependency-child-hmac:v2')
        children.append(AssistantMessageEvidenceDependency(
            assistant_message_id=message.id, candidate_ordinal=ordinal,
            serving_document_id=snapshot.serving_document_id, dependency_kind='legacy_unbound',
            serving_content_hash=legacy_hash, permission_level=permission,
            fingerprint_key_version=version, fingerprint_key_material_verifier=verifier,
            dependency_serving_scope='legacy_v1_only', dependency_role='selected_citation',
            dependency_child_hmac=child_hmac, legacy_dependency_identity_hmac=identity,
            model_content_hmac=model_hmac, canonical_citation_projection_hmac=canonical,
            selected_v1_citation_projection_hmac=citation_hmac,
        ))
    selected = build_v1_selected_evidence_projection_hmac(citation_hmacs=tuple(citation_hmacs),
        source_ids=tuple(message.source_ids), source_links=tuple(message.source_links),
        source_snippets=tuple(message.source_snippets), settings=settings)
    metadata = dict(message.metadata_)
    agent, prompt = metadata.get('agent_name'), metadata.get('prompt_version')
    backend = metadata.setdefault('effective_backend', 'deterministic_lexical')
    origin = fp({
        'agent_name_bytes': exact_utf8_bytes(agent),
        'content_bytes_hmac': fp({'content_bytes': exact_utf8_bytes(message.content)}, 'assistant-legacy-content-bytes:v1'),
        'effective_backend': backend, 'legacy_result_contract_version': 'rag-answer:v1',
        'output_permission': message.permission_level,
        'prompt_version_bytes': exact_utf8_bytes(prompt) if prompt is not None else None,
        'selected_evidence_projection_hmac': selected,
    }, 'assistant-legacy-evidence-origin:v1')
    content = fp({
        'agent_name_bytes': exact_utf8_bytes(agent), 'assistant_message_id': message.id,
        'content_bytes': exact_utf8_bytes(message.content), 'content_origin': 'legacy_evidence',
        'content_origin_hmac': origin, 'content_write_mode': 'legacy_trimmed',
        'conversation_id': message.conversation_id, 'linked_agent_run_id': None,
        'message_role': 'assistant', 'prompt_version_bytes': exact_utf8_bytes(prompt) if prompt is not None else None,
        'rag_result_hmac': None,
    }, 'assistant-message-content-hmac:v1')
    whole = fp({
        'assistant_message_content_hmac': content, 'content_origin': 'legacy_evidence',
        'content_origin_hmac': origin, 'content_write_mode': 'legacy_trimmed',
        'dependencies': [{'candidate_ordinal': child.candidate_ordinal, 'dependency_child_hmac': child.dependency_child_hmac} for child in children],
        'dependency_count': len(children), 'dependency_serving_scope': 'legacy_v1_only',
        'evidence_contract_version': 'assistant-evidence:v1', 'linked_agent_run_id': None,
        'model_influence_set_hmac': None, 'parent_selected_evidence_projection_hmac': selected,
        'rag_result_hmac': None,
    }, 'assistant-dependency-set-hmac:v2')
    for snapshot in old:
        db.query(AssistantMessageKnowledgeEvidenceRef).filter_by(dependency_id=snapshot.id).delete()
        db.delete(snapshot)
    db.flush()
    for child in children:
        child.dependency_set_hmac = whole
    message.content_write_mode = 'legacy_trimmed'
    message.content_origin = 'legacy_evidence'
    message.content_hmac_schema_version = 'assistant-message-content-hmac:v1'
    message.assistant_message_content_hmac = content
    message.content_hmac_key_version = version
    message.content_hmac_key_material_verifier = verifier
    message.content_origin_hmac = origin
    message.dependency_set_hmac_schema_version = 'assistant-dependency-set-hmac:v2'
    message.dependency_set_hmac = whole
    message.parent_selected_evidence_projection_hmac = selected
    message.metadata_ = metadata
    db.add_all(children)
    db.flush()
