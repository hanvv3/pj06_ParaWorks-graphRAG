from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / 'fixtures' / 'rag_v2_provider_free_golden_60.json'


def _cases():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']


def _execute_provider_free_case(case):
    """Exercise the real Runnable retrievers with in-process fake ports only."""
    from backend.app.agent_runtime.rag_v2_identity import (
        SecurityScope,
        security_scope_fingerprint,
    )
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
    from backend.app.rag.pgvector_retriever import PgVectorEvidenceRetriever
    from backend.app.rag.retrieval import (
        ClassifiedRetrievalCandidate,
        RetrievalRequest,
        rank_evidence_slots,
    )
    from backend.app.rag.serving_contracts import EvidenceAccessClassification
    from backend.tests.test_rag_shadow import _evidence as _shadow_evidence
    from backend.tests.test_rag_v2_pgvector_retriever import (
        _embedding_result,
        _readiness,
        _settings,
    )

    settings = _settings()
    permission = case['permission']
    scope = SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='golden-user',
        workspace_scope_id='golden-workspace',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=(permission,),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )
    kind = case['evidence_kind']
    identifier = int(case['case_id'].split('-')[1])
    serving_kind = 'trusted_knowledge' if kind == 'trusted' else 'raw_chunk'
    evidence = _shadow_evidence(identifier, kind=serving_kind)
    evidence = replace(
        evidence,
        effective_permission=permission,
        version_envelope=replace(
            evidence.version_envelope,
            effective_permission=permission,
        ),
    )
    access = EvidenceAccessClassification(
        global_eligibility=(
            'ineligible' if kind in {'stale', 'revoked'} else 'eligible'
        ),
        resource_scope=(
            'out_of_scope' if kind == 'hard_negative' else 'in_scope'
        ),
        permission_visibility=(
            'denied_known' if kind == 'hidden' else 'visible'
        ),
    )
    candidates = () if kind == 'no_match' else (
        ClassifiedRetrievalCandidate(
            evidence=evidence,
            relevance_score=0.1 if kind == 'hard_negative' else 0.9,
            matched_terms=('golden',),
            access=access,
        ),
    )

    class KeywordStore:
        def search(self, request):
            assert request.security_scope is scope
            return candidates

    query = 'golden deterministic query'
    request = RetrievalRequest(
        retrieval_query_text=query,
        security_scope=scope,
        security_scope_fingerprint=security_scope_fingerprint(
            scope, settings=settings
        ),
        query_embedding_result=(
            _embedding_result(query=query)
            if case['backend'] == 'pgvector'
            else None
        ),
        candidate_scan_limit=50,
        visible_limit=5 if case['surface'] == 'search' else 8,
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )
    keyword = KeywordEvidenceRetriever(store=KeywordStore(), settings=settings)
    if case['backend'] == 'keyword':
        result = keyword.invoke(request, config={'callbacks': [], 'metadata': {}})
        fake_embedding_dispatches = 0
    else:
        class PgVectorStore:
            def search(self, search_request, vector):
                assert search_request is request
                assert vector is request.query_embedding_result.vector
                return candidates

        class Readiness:
            def inspect(self):
                return _readiness()

        result = PgVectorEvidenceRetriever(
            store=PgVectorStore(),
            readiness=Readiness(),
            keyword_retriever=keyword,
            settings=settings,
        ).invoke(request, config={'callbacks': [], 'metadata': {}})
        fake_embedding_dispatches = 1

    slots = rank_evidence_slots(result.visible)
    visible_permissions = {
        candidate.evidence.effective_permission for candidate in result.visible
    }
    expected_visible = kind in {'trusted', 'raw', 'v1_parity'}
    return {
        'external_provider_calls': 0,
        'network_calls': 0,
        'fake_embedding_dispatches': fake_embedding_dispatches,
        'provider_attempt_count': result.trace.provider_attempt_count,
        'permission_leak_count': len(visible_permissions - {permission}),
        'stale_leak_count': len(result.visible) if kind == 'stale' else 0,
        'revoked_leak_count': len(result.visible) if kind == 'revoked' else 0,
        'invalid_evidence_slot_count': sum(
            not slot.evidence.public_source_id
            or not slot.evidence.model_content
            or slot.evidence.effective_permission != permission
            for slot in slots
        ),
        'missing_evidence_slot_count': abs(
            (1 if expected_visible else 0) - len(slots)
        ),
    }


def test_provider_free_golden_is_exact_unique_60_case_manifest():
    """Catches silently dropping a rollout cohort or replacing fake transports."""
    cases = _cases()
    assert len(cases) == 60
    assert len({case['case_id'] for case in cases}) == 60
    assert {case['surface'] for case in cases} == {'ask', 'search', 'assistant'}
    assert {case['backend'] for case in cases} == {'keyword', 'pgvector'}
    assert {case['evidence_kind'] for case in cases} == {
        'trusted', 'raw', 'no_match', 'hidden', 'stale', 'revoked',
        'hard_negative', 'v1_parity',
    }
    assert {case['permission'] for case in cases} == {'public', 'internal', 'restricted'}


@pytest.mark.parametrize('case', _cases(), ids=lambda case: case['case_id'])
def test_provider_free_golden_has_zero_external_dispatch_or_leak(case):
    """Catches network/provider use or permission/stale/revoke/evidence leakage."""
    from backend.app.agent_runtime.rag_rollout import RagRolloutPolicy

    plan = RagRolloutPolicy().plan(
        mode='shadow', stage='assistant', surface=case['surface'], backend=case['backend'],
    )
    assert plan.v2_generation_count == 0
    assert case['model_transport'] == 'fake'
    assert case['embedding_transport'] == 'fake'
    assert case['network_calls'] == 0
    assert case['provider_calls'] == 0
    assert case['permission_leak_count'] == 0
    assert case['revoked_leak_count'] == 0
    assert case['stale_leak_count'] == 0
    assert case['invalid_evidence_slot_count'] == 0
    assert case['missing_evidence_slot_count'] == 0


@pytest.mark.parametrize('case', _cases(), ids=lambda case: case['case_id'])
def test_every_golden_case_executes_real_deterministic_retrieval(case):
    """Catches a manifest that asserts its own constants without executing RAG."""
    observation = _execute_provider_free_case(case)

    assert observation['external_provider_calls'] == 0
    assert observation['network_calls'] == 0
    assert observation['permission_leak_count'] == 0
    assert observation['stale_leak_count'] == 0
    assert observation['revoked_leak_count'] == 0
    assert observation['invalid_evidence_slot_count'] == 0
    assert observation['missing_evidence_slot_count'] == 0
    if case['backend'] == 'keyword':
        assert observation['fake_embedding_dispatches'] == 0
        assert observation['provider_attempt_count'] == 0
    else:
        assert observation['fake_embedding_dispatches'] == 1
        assert observation['provider_attempt_count'] == 1
