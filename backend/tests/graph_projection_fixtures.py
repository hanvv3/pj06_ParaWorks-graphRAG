"""Fixed E fixture: one approved relation with two exact source children."""

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.models import RagServingCorpusGeneration, TrustedKnowledgeEvidenceLink
from backend.app.rag.lexical_projection import refresh_rag_lexical_projections
from backend.app.rag.serving_generation import (
    RAG_COSINE_POLICY_VERSION,
    RAG_INDEX_POLICY_VERSION,
)
from backend.tests.test_rag_trusted_evidence import (
    _scope,
    _seed_item,
    _seed_link,
    _seed_source,
    _seed_target,
    _settings,
)

SETTINGS = _settings()
SCOPE = _scope(permissions=('public', 'internal'))
CASES = {
    'relation': {
        'query': '프로젝트 결정과 후속 근거',
        'expected_sources': ('gmail:trusted-1', 'gmail:trusted-2'),
        'baseline_ids': ('history_event:1', 'chunk:1'),
    },
    'single': {
        'query': '단일 문서 확인',
        'expected_sources': ('gmail:trusted-3',),
        'baseline_ids': ('chunk:3',),
    },
    'absent': {'query': 'corpus 밖 질문', 'expected_sources': (), 'baseline_ids': ()},
    'restricted': {
        'query': '프로젝트 결정과 후속 근거',
        'expected_sources': ('gmail:trusted-1',),
        'baseline_ids': ('chunk:1',),
    },
    'revoke': {
        'query': '프로젝트 결정과 후속 근거',
        'expected_sources': ('gmail:trusted-1',),
        'baseline_ids': ('chunk:1',),
    },
}


def vector(axis):
    return [1.0 if i == axis else 0.0 for i in range(1536)]


def seed_corpus(db, case='relation'):
    sources, chunks = zip(*(_seed_source(db, ordinal=i) for i in (1, 2, 3)), strict=True)
    item = _seed_item(db, source=sources[0], chunk=chunks[0], resolution_source='human')
    item.source_links = [sources[0].source_url, sources[1].source_url]
    item.source_snippets = [chunks[0].source_snippet, chunks[1].source_snippet]
    target = _seed_target(db, source_review_item_id=item.id)
    approval = _seed_link(
        db, target=target, item=item, source=sources[0], resolution_source='human'
    )
    db.add(
        TrustedKnowledgeEvidenceLink(
            approval_link_id=approval.id,
            canonical_source_kind='gmail',
            canonical_source_id=str(sources[1].id),
            canonical_version_or_signature=sources[1].server_content_signature,
            evidence_hash='d' * 64,
            fingerprint_key_version=approval.fingerprint_key_version,
            fingerprint_key_material_verifier=approval.fingerprint_key_material_verifier,
        )
    )
    if case == 'restricted':
        sources[1].permission_level = chunks[1].permission_level = 'restricted'
    if case == 'revoke':
        approval.active = False
        item.status = 'revoked'
    db.add(
        RagServingCorpusGeneration(
            id=1,
            corpus_generation=1,
            vector_index_generation=1,
            embedding_model=SETTINGS.openai_embedding_model,
            embedding_dimensions=1536,
            index_policy_version=RAG_INDEX_POLICY_VERSION,
            pgvector_cosine_policy_version=RAG_COSINE_POLICY_VERSION,
            fingerprint_key_version=SETTINGS.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                SETTINGS.agent_runtime_fingerprint_secret
            ),
        )
    )
    db.flush()
    refresh_rag_lexical_projections(db, settings=SETTINGS, corpus_generation=1)
    db.commit()
    return sources, chunks, target, approval
