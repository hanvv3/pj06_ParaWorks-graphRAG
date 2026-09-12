"""Exact current V1 provenance and keyed bytes; never D serving authority."""

import math
import struct
from hmac import compare_digest

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
    ReviewItem,
    Source,
    TrustedKnowledgeApprovalLink,
    TrustedKnowledgeEvidenceLink,
)
from backend.app.rag.evidence_projection import (
    build_v1_selected_evidence_projection_hmac,
)
from backend.app.rag.serving_contracts import (
    build_canonical_citation_projection_hmac,
    build_model_content_hmac,
)

SCHEMA = 'assistant-dependency-set-hmac:v3'
SNAPSHOT_FIELDS = (
    'document_chunk_id',
    'document_version_id',
    'source_id',
    'parser_run_id',
    'current_document_version_id',
    'server_content_signature_schema',
    'server_content_signature',
    'parser_policy_version',
    'parser_version',
    'chunk_policy_version',
    'knowledge_type',
    'knowledge_id',
    'approval_link_id',
    'legacy_human_base',
    'legacy_source_review_item_id',
)
APPROVAL_FIELDS = (
    'id',
    'knowledge_type',
    'knowledge_id',
    'review_item_id',
    'security_scope_id',
    'promotion_effect_kind',
    'resolution_source',
    'claim_fingerprint',
    'permission_level',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
    'active',
)
REVIEW_FIELDS = (
    'id',
    'status',
    'permission_level',
    'resolution_source',
    'candidate_contract_version',
    'source_links',
    'source_snippets',
)
LINK_FIELDS = (
    'id',
    'approval_link_id',
    'canonical_source_kind',
    'canonical_source_id',
    'canonical_version_or_signature',
    'evidence_hash',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
)


def exact_object(value):
    if type(value) is str:
        return exact_utf8_bytes(value)
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, list):
        return [exact_object(item) for item in value]
    if isinstance(value, dict):
        return {key: exact_object(item) for key, item in value.items()}
    raise ValueError('legacy authority contains unsupported values')


def action_payload(metadata):
    action, derived, draft = (
        metadata.get('action_type'),
        metadata.get('evidence_derived', False),
        metadata.get('email_draft'),
    )
    if (action is not None and type(action) is not str) or type(derived) is not bool:
        raise ValueError('invalid protected legacy action')
    if draft is not None and (
        not isinstance(draft, dict)
        or set(draft) != {'to', 'subject', 'body'}
        or type(draft['to']) is not list
        or not draft['to']
        or any(type(item) is not str or not item.strip() for item in draft['to'])
        or any(
            type(draft[key]) is not str or not draft[key].strip()
            for key in ('subject', 'body')
        )
    ):
        raise ValueError('invalid protected email payload')
    if derived and action == 'email_draft' and draft is None:
        raise ValueError('evidence-derived email requires protected draft')
    return exact_object(
        {'action_type': action, 'evidence_derived': derived, 'email_draft': draft}
    )


def _fields(row, names):
    return {name: getattr(row, name) for name in names}


def _provenance(db, message, child):
    from backend.app.assistant.service import _dependency_is_live
    from backend.app.knowledge.trusted_serving_eligibility import (
        TrustedServingEligibilityService,
        knowledge_model_for_type,
        knowledge_type_storage_aliases,
    )

    snapshot = _fields(child, SNAPSHOT_FIELDS)
    approval = review = None
    links = []
    refs = list(
        db.scalars(
            select(AssistantMessageKnowledgeEvidenceRef).where(
                AssistantMessageKnowledgeEvidenceRef.dependency_id == child.id
            )
        )
    )
    if child.dependency_kind == 'legacy_unbound':
        kind, identifier = child.serving_document_id.split(':')
        target = db.get(knowledge_model_for_type(kind), int(identifier))
        if (
            kind == 'chunk'
            or target is None
            or target.source_review_item_id is not None
            or any(
                value is not None
                for key, value in snapshot.items()
                if key != 'legacy_human_base'
            )
            or child.legacy_human_base
            or refs
            or db.scalar(
                select(TrustedKnowledgeApprovalLink.id).where(
                    TrustedKnowledgeApprovalLink.knowledge_id == target.id,
                    TrustedKnowledgeApprovalLink.knowledge_type.in_(
                        knowledge_type_storage_aliases(kind)
                    ),
                )
            )
            is not None
        ):
            raise ValueError('unbound legacy authority is not pre-provenance')
    else:
        if not _dependency_is_live(
            db, TrustedServingEligibilityService(db), message, child
        ):
            raise ValueError('legacy exact authority is unavailable')
        if child.dependency_kind == 'raw_chunk':
            chunk, source = (
                db.get(DocumentChunk, child.document_chunk_id),
                db.get(Source, child.source_id),
            )
            if (
                child.serving_document_id != f'chunk:{chunk.id}'
                or chunk.source_id != source.id
                or source.server_content_signature_schema
                != child.server_content_signature_schema
                or any(snapshot[name] is None for name in SNAPSHOT_FIELDS[:10])
                or any(
                    snapshot[name] is not None
                    for name in (
                        'knowledge_type',
                        'knowledge_id',
                        'approval_link_id',
                        'legacy_source_review_item_id',
                    )
                )
                or child.legacy_human_base
                or refs
            ):
                raise ValueError('legacy raw authority changed')
        elif child.dependency_kind == 'trusted_knowledge':
            kind, identifier = child.serving_document_id.split(':')
            if int(identifier) != child.knowledge_id or knowledge_model_for_type(
                kind
            ) is not knowledge_model_for_type(child.knowledge_type):
                raise ValueError('legacy trusted target changed')
            if any(snapshot[name] is not None for name in SNAPSHOT_FIELDS[:10]):
                raise ValueError('trusted legacy has raw authority')
            if child.approval_link_id is not None:
                row = db.get(TrustedKnowledgeApprovalLink, child.approval_link_id)
                approval = _fields(row, APPROVAL_FIELDS)
                item = db.get(ReviewItem, row.review_item_id)
                if (
                    row.revoked_at is not None
                    or item is None
                    or item.revoked_at is not None
                ):
                    raise ValueError('selected legacy approval revoked')
                review = _fields(item, REVIEW_FIELDS)
                current = list(
                    db.scalars(
                        select(TrustedKnowledgeEvidenceLink)
                        .where(TrustedKnowledgeEvidenceLink.approval_link_id == row.id)
                        .order_by(TrustedKnowledgeEvidenceLink.id)
                    )
                )
                if (
                    child.legacy_human_base
                    or child.legacy_source_review_item_id is not None
                    or len(refs) != len(current)
                    or not current
                    or {ref.trusted_knowledge_evidence_link_id for ref in refs}
                    != {link.id for link in current}
                    or any(
                        ref.assistant_message_id != message.id
                        or ref.approval_link_id != row.id
                        for ref in refs
                    )
                ):
                    raise ValueError('selected legacy approval refs changed')
                links = [_fields(link, LINK_FIELDS) for link in current]
            else:
                if not child.legacy_human_base or refs:
                    raise ValueError('legacy human authority missing')
                if child.legacy_source_review_item_id is not None:
                    item = db.get(ReviewItem, child.legacy_source_review_item_id)
                    if item is None or item.revoked_at is not None:
                        raise ValueError('legacy human review unavailable')
                    review = _fields(item, REVIEW_FIELDS)
        else:
            raise ValueError('unknown legacy dependency kind')
    return {
        'snapshot': snapshot,
        'approval': approval,
        'review': review,
        'evidence_links': links,
    }


def compute_material(*, db, message, settings, actor=None):
    """Read-only derivation shared by writer and immutable-reader verification."""
    from backend.app.assistant.legacy_evidence import current_legacy_evidence

    secret, version = fingerprint_secret_bytes(settings)
    verifier = fingerprint_key_material_verifier(secret.decode('utf-8'))
    runtime = db.scalar(
        select(AutoReviewRuntimeKeyState).where(
            AutoReviewRuntimeKeyState.component == 'auto_review_trust_promotion'
        )
    )
    if (
        runtime is None
        or not runtime.ready
        or runtime.fingerprint_key_version != version
        or runtime.fingerprint_key_material_verifier != verifier
    ):
        raise ValueError('legacy evidence key runtime unavailable')

    def fp(value, schema, policy='assistant-evidence:v1'):
        return keyed_fingerprint(
            value, secret=secret, schema_version=schema, policy_version=policy
        )

    children = list(
        db.scalars(
            select(AssistantMessageEvidenceDependency)
            .where(
                AssistantMessageEvidenceDependency.assistant_message_id == message.id
            )
            .order_by(AssistantMessageEvidenceDependency.candidate_ordinal)
        )
    )
    if (
        message.role != 'assistant'
        or message.content != message.content.strip()
        or not message.content
        or not children
        or len(children) != message.serving_dependency_count
    ):
        raise ValueError('legacy parent/dependency shape invalid')
    citations = message.citations
    if not isinstance(citations, list):
        raise ValueError('legacy citations invalid')
    for target, field in (
        (message.source_ids, 'source_id'),
        (message.source_links, 'source_url'),
        (message.source_snippets, 'source_snippet'),
    ):
        if target != list(dict.fromkeys(citation[field] for citation in citations)):
            raise ValueError('legacy public projection changed')
    metadata = dict(message.metadata_ or {})
    if (
        message.content_write_mode is not None
        and not {'agent_name', 'effective_backend'} <= metadata.keys()
    ):
        raise ValueError('stored legacy origin metadata missing')
    metadata.setdefault('agent_name', 'rag_orchestrator_agent')
    metadata.setdefault('effective_backend', 'deterministic_lexical')
    protected = action_payload(metadata)
    if not citations and metadata.get('evidence_derived') is not True:
        raise ValueError('uncited legacy influences require evidence-derived action')
    if metadata['effective_backend'] not in {'deterministic_lexical', 'pgvector'}:
        raise ValueError('invalid legacy backend')
    materials, selected = [], [None] * len(citations)
    for ordinal, child in enumerate(children):
        if child.candidate_ordinal != ordinal:
            raise ValueError('legacy dependency ordinal changed')
        (
            public_id,
            kind,
            text,
            links,
            snippets,
            permission,
            content_hash,
            serving_kind,
        ) = current_legacy_evidence(db, child.serving_document_id)
        if (
            child.serving_content_hash != content_hash
            or child.permission_level != permission
            or (actor is not None and permission not in actor.permission_levels)
        ):
            raise ValueError('legacy current content or permission changed')
        if message.permission_level is not None:
            from backend.app.rag.serving_contracts import strictest_permission

            if (
                strictest_permission((message.permission_level, permission))
                != message.permission_level
            ):
                raise ValueError('legacy output permission broadens evidence')
        provenance = _provenance(db, message, child)
        canonical_fields = {
            'source_id': public_id,
            'source_type': kind,
            'source_url': links[0],
            'source_snippet': snippets[0],
            'permission_level': permission,
        }
        matching = [
            index
            for index, citation in enumerate(citations)
            if all(
                citation.get(key) == value for key, value in canonical_fields.items()
            )
        ]
        if len(matching) > 1 or (matching and selected[matching[0]] is not None):
            raise ValueError('ambiguous legacy citation owner')
        role, citation_hmac = 'legacy_evidence_influence', None
        if matching:
            index = matching[0]
            citation = citations[index]
            if set(citation) != {*canonical_fields, 'relevance_score', 'matched_terms'}:
                raise ValueError('legacy citation shape invalid')
            score, terms = citation['relevance_score'], citation['matched_terms']
            if (
                type(score) not in {float, int}
                or not math.isfinite(score)
                or type(terms) is not list
            ):
                raise ValueError('legacy citation score/terms invalid')
            citation_hmac = fp(
                {
                    'source_id_bytes': exact_utf8_bytes(public_id),
                    'source_url_bytes': exact_utf8_bytes(links[0]),
                    'source_type_bytes': exact_utf8_bytes(kind),
                    'permission_level': permission,
                    'source_snippet_bytes': exact_utf8_bytes(snippets[0]),
                    'relevance_score_binary64_be_hex': struct.pack('>d', score).hex(),
                    'matched_terms_bytes': [exact_utf8_bytes(term) for term in terms],
                },
                'rag-v1-evidence-projection:citation:v1',
                'rag-v1-evidence-projection:v1',
            )
            selected[index], role = citation_hmac, 'selected_citation'
        model = build_model_content_hmac(
            serving_kind=serving_kind, model_content=text, settings=settings
        )
        canonical = build_canonical_citation_projection_hmac(
            public_source_id=public_id,
            public_source_type=kind,
            source_url=links[0],
            source_snippet=snippets[0],
            effective_permission=permission,
            settings=settings,
        )
        identity = fp(
            {
                'serving_document_id_bytes': exact_utf8_bytes(
                    child.serving_document_id
                ),
                'dependency_kind': child.dependency_kind,
                'dependency_role': role,
                'effective_permission': permission,
                'serving_version_fingerprint': content_hash,
                'model_content_hmac': model,
                'canonical_citation_projection_hmac': canonical,
                'legacy_public_source_id_bytes': exact_utf8_bytes(public_id),
                'legacy_source_links_bytes': [
                    exact_utf8_bytes(value) for value in links
                ],
                'legacy_source_snippets_bytes': [
                    exact_utf8_bytes(value) for value in snippets
                ],
                'provenance': exact_object(provenance),
            },
            'assistant-legacy-dependency-snapshot:v2',
        )
        approval_hmac = (
            fp(
                exact_object(
                    {'approval': provenance['approval'], 'review': provenance['review']}
                ),
                'assistant-legacy-approval-provenance:v1',
            )
            if child.approval_link_id is not None
            else None
        )
        refs_hmac = (
            fp(
                exact_object({'evidence_links': provenance['evidence_links']}),
                'assistant-legacy-evidence-links:v1',
            )
            if child.approval_link_id is not None
            else None
        )
        values = {
            'dependency_serving_scope': 'legacy_v1_only',
            'dependency_role': role,
            'legacy_dependency_identity_hmac': identity,
            'model_content_hmac': model,
            'canonical_citation_projection_hmac': canonical,
            'selected_v1_citation_projection_hmac': citation_hmac,
            'approval_provenance_hmac': approval_hmac,
            'evidence_link_set_hmac': refs_hmac,
            'serving_identity_hmac': None,
            'serving_version_fingerprint': None,
            'support_mode': None,
            'fingerprint_key_version': version,
            'fingerprint_key_material_verifier': verifier,
        }
        child_payload = {
            key: value
            for key, value in values.items()
            if not key.startswith('fingerprint_')
        }
        child_payload.update(
            approval_link_id=child.approval_link_id,
            candidate_ordinal=ordinal,
            dependency_kind=child.dependency_kind,
            effective_permission=permission,
            evidence_link_ids=[link['id'] for link in provenance['evidence_links']],
            raw_document_chunk_id=child.document_chunk_id,
            trusted_knowledge_id=child.knowledge_id,
        )
        values['dependency_child_hmac'] = fp(
            child_payload, 'assistant-dependency-child-hmac:v3'
        )
        materials.append((child, values))
    if any(item is None for item in selected):
        raise ValueError('legacy citation has no exact dependency')
    projection = build_v1_selected_evidence_projection_hmac(
        citation_hmacs=tuple(selected),
        source_ids=tuple(message.source_ids),
        source_links=tuple(message.source_links),
        source_snippets=tuple(message.source_snippets),
        settings=settings,
    )
    agent, prompt = metadata['agent_name'], metadata.get('prompt_version')
    origin = fp(
        {
            'agent_name_bytes': exact_utf8_bytes(agent),
            'content_bytes_hmac': fp(
                {'content_bytes': exact_utf8_bytes(message.content)},
                'assistant-legacy-content-bytes:v1',
            ),
            'effective_backend': metadata['effective_backend'],
            'legacy_result_contract_version': 'rag-answer:v1',
            'output_permission': message.permission_level,
            'prompt_version_bytes': exact_utf8_bytes(prompt)
            if prompt is not None
            else None,
            'selected_evidence_projection_hmac': projection,
            'legacy_action_payload_hmac': fp(
                protected, 'assistant-legacy-action-payload:v1'
            ),
            'hidden_match_count': message.hidden_match_count,
            'permission_notice_bytes': exact_utf8_bytes(message.permission_notice)
            if message.permission_notice is not None
            else None,
        },
        'assistant-legacy-evidence-origin:v2',
    )
    content = fp(
        {
            'agent_name_bytes': exact_utf8_bytes(agent),
            'assistant_message_id': message.id,
            'content_bytes': exact_utf8_bytes(message.content),
            'content_origin': 'legacy_evidence',
            'content_origin_hmac': origin,
            'content_write_mode': 'legacy_trimmed',
            'conversation_id': message.conversation_id,
            'linked_agent_run_id': None,
            'message_role': 'assistant',
            'prompt_version_bytes': exact_utf8_bytes(prompt)
            if prompt is not None
            else None,
            'rag_result_hmac': None,
        },
        'assistant-message-content-hmac:v1',
    )
    whole = fp(
        {
            'assistant_message_content_hmac': content,
            'content_origin': 'legacy_evidence',
            'content_origin_hmac': origin,
            'content_write_mode': 'legacy_trimmed',
            'dependencies': [
                {
                    'candidate_ordinal': child.candidate_ordinal,
                    'dependency_child_hmac': values['dependency_child_hmac'],
                }
                for child, values in materials
            ],
            'dependency_count': len(children),
            'dependency_serving_scope': 'legacy_v1_only',
            'evidence_contract_version': 'assistant-evidence:v1',
            'linked_agent_run_id': None,
            'model_influence_set_hmac': None,
            'parent_selected_evidence_projection_hmac': projection,
            'rag_result_hmac': None,
        },
        SCHEMA,
    )
    parent = {
        'content_write_mode': 'legacy_trimmed',
        'content_origin': 'legacy_evidence',
        'content_hmac_schema_version': 'assistant-message-content-hmac:v1',
        'assistant_message_content_hmac': content,
        'content_hmac_key_version': version,
        'content_hmac_key_material_verifier': verifier,
        'content_origin_hmac': origin,
        'dependency_set_hmac_schema_version': SCHEMA,
        'dependency_set_hmac': whole,
        'parent_selected_evidence_projection_hmac': projection,
        'model_influence_set_hmac': None,
        'rag_result_hmac': None,
        'linked_agent_run_id': None,
        'evidence_contract_version': 'assistant-evidence:v1',
    }
    for _, values in materials:
        values['dependency_set_hmac'] = whole
    return parent, materials, metadata


def sign_message(*, db, message, settings, actor=None):
    with db.no_autoflush:
        if message.content_write_mode is not None:
            raise ValueError('legacy signing requires a new unsigned staging row')
        parent, children, metadata = compute_material(
            db=db, message=message, settings=settings, actor=actor
        )
        for child, values in children:
            for key, value in values.items():
                setattr(child, key, value)
        db.flush([child for child, _ in children])
        message.metadata_ = metadata
        for key, value in parent.items():
            setattr(message, key, value)
        db.flush([message])


def is_live(*, db, message, actor, settings):
    parent, children, _ = compute_material(
        db=db, message=message, settings=settings, actor=actor
    )
    for row, expected in [(message, parent), *children]:
        for key, value in expected.items():
            stored = getattr(row, key)
            if type(value) is str and key.endswith('hmac'):
                if type(stored) is not str or not compare_digest(stored, value):
                    return False
            elif stored != value:
                return False
    return True
