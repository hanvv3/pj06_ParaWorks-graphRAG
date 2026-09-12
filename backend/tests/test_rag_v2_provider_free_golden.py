from __future__ import annotations

import json
import socket
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from backend.app.agent_runtime.rag_provider_transport import (
    _DirectOpenAIProviderClient,
)
from backend.app.rag.embeddings import OpenAIEmbeddingModel

FIXTURE = Path(__file__).parent / 'fixtures' / 'rag_v2_provider_free_golden_60.json'
_ORIGINAL_SOCKET_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_PROVIDER_SEND = _DirectOpenAIProviderClient.send
_ORIGINAL_EMBED_MANY = OpenAIEmbeddingModel.embed_many
_ORIGINAL_HTTPX_SEND = httpx.Client.send
_ORIGINAL_ASYNC_HTTPX_SEND = httpx.AsyncClient.send
_EXTERNAL_GUARD_LOCK = threading.RLock()


def _cases():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']


@contextmanager
def _observe_external_calls():
    """Block real I/O and restore every process-global hook after one case."""
    counters = {'external_provider_calls': 0, 'network_calls': 0}
    with _EXTERNAL_GUARD_LOCK, ExitStack() as stack:
        previous_socket_create = socket.create_connection
        previous_provider_send = _DirectOpenAIProviderClient.send
        previous_embed_many = OpenAIEmbeddingModel.embed_many
        previous_httpx_send = httpx.Client.send
        previous_async_httpx_send = httpx.AsyncClient.send

        def socket_create_connection(*args, **kwargs):
            counters['network_calls'] += 1
            if previous_socket_create is not _ORIGINAL_SOCKET_CREATE_CONNECTION:
                return previous_socket_create(*args, **kwargs)
            return object()

        def socket_connect(*_args, **_kwargs):
            counters['network_calls'] += 1
            raise RuntimeError('golden network call blocked')

        def provider_send(*args, **kwargs):
            counters['external_provider_calls'] += 1
            if previous_provider_send is not _ORIGINAL_PROVIDER_SEND:
                return previous_provider_send(*args, **kwargs)
            return object()

        def embed_many(*args, **kwargs):
            counters['external_provider_calls'] += 1
            if previous_embed_many is not _ORIGINAL_EMBED_MANY:
                return previous_embed_many(*args, **kwargs)
            return object()

        def httpx_send(*args, **kwargs):
            counters['network_calls'] += 1
            if previous_httpx_send is not _ORIGINAL_HTTPX_SEND:
                return previous_httpx_send(*args, **kwargs)
            return object()

        async def async_httpx_send(*args, **kwargs):
            counters['network_calls'] += 1
            if previous_async_httpx_send is not _ORIGINAL_ASYNC_HTTPX_SEND:
                return await previous_async_httpx_send(*args, **kwargs)
            return object()

        from unittest.mock import patch

        stack.enter_context(patch.object(socket, 'create_connection', socket_create_connection))
        stack.enter_context(patch.object(socket.socket, 'connect', socket_connect))
        stack.enter_context(patch.object(socket.socket, 'connect_ex', socket_connect))
        stack.enter_context(patch.object(_DirectOpenAIProviderClient, 'send', provider_send))
        stack.enter_context(patch.object(OpenAIEmbeddingModel, 'embed_many', embed_many))
        stack.enter_context(patch.object(httpx.Client, 'send', httpx_send))
        stack.enter_context(patch.object(httpx.AsyncClient, 'send', async_httpx_send))
        yield counters


def _execute_provider_free_case(case):
    with _observe_external_calls() as counters:
        observation = _execute_provider_free_case_guarded(case)
    return {**observation, **counters}


def _execute_provider_free_case_guarded(case):
    """Exercise the real Runnable retrievers with in-process fake ports only."""
    from backend.app.agent_runtime.rag_v2_identity import (
        SecurityScope,
        security_scope_fingerprint,
    )
    from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
        StrictQueryEmbeddingAdapter,
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
        _readiness,
        _settings,
        _StrictCostPolicy,
        _StrictPermit,
        _StrictUsageParser,
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
        query_embedding_result=None,
        candidate_scan_limit=50,
        visible_limit=5 if case['surface'] == 'search' else 8,
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )
    embedding_transport_calls = []

    class InProcessFakeEmbeddingTransport:
        def dispatch(self, prepared):
            embedding_transport_calls.append(prepared.attempt_fence_hmac)
            return {
                'object': 'list',
                'model': 'text-embedding-3-small',
                'data': [
                    {
                        'object': 'embedding',
                        'index': 0,
                        'embedding': [1.0, *([0.0] * 1535)],
                    }
                ],
                'usage': {'prompt_tokens': 3, 'total_tokens': 3},
            }

    if case['backend'] == 'pgvector':
        adapter = StrictQueryEmbeddingAdapter(
            usage_parser=_StrictUsageParser(),
            cost_policy=_StrictCostPolicy(),
            transport=InProcessFakeEmbeddingTransport(),
            settings=settings,
        )
        prepared = adapter.prepare(request, _readiness())
        request = replace(
            request,
            query_embedding_result=adapter.dispatch_once(
                prepared,
                _StrictPermit(),
            ),
        )
    keyword = KeywordEvidenceRetriever(store=KeywordStore(), settings=settings)
    if case['backend'] == 'keyword':
        result = keyword.invoke(request, config={'callbacks': [], 'metadata': {}})
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

    comparison = None
    legacy = None
    fake_v1_observation_count = 0
    if kind == 'v1_parity':
        from backend.app.rag.shadow import (
            LegacyRetrievalCandidateObservation,
            LegacyRetrievalObservation,
            ShadowComparator,
        )

        legacy = LegacyRetrievalObservation.build(
            surface=case['surface'],
            configured_backend=result.configured_backend,
            effective_backend=result.effective_backend,
            query_context_version=(
                'assistant-context:v1'
                if case['surface'] == 'assistant'
                else 'direct-query:v1'
            ),
            retrieval_query_hmac='1' * 64,
            security_scope_fingerprint=request.security_scope_fingerprint,
            candidate_window=(
                LegacyRetrievalCandidateObservation(
                    ordinal=0,
                    legacy_serving_document_id=evidence.serving_document_id,
                    visibility='visible',
                    legacy_public_source_id=evidence.public_source_id,
                    support_class='raw_candidate',
                    effective_permission=permission,
                    relevance_score=0.9,
                    matched_terms=('golden',),
                    public_projection_hmac=(
                        evidence.canonical_citation_projection_hmac
                    ),
                    candidate_identity_hmac=evidence.serving_identity_hmac,
                ),
            ),
            hidden_match_count=0,
            hidden_count_capped=False,
            query_embedding_attempt_fence_hmac=(
                request.query_embedding_result.prepared.attempt_fence_hmac
                if request.query_embedding_result is not None
                else None
            ),
            public_agent_run_correlation_hmac='2' * 64,
            latency_ms=0,
            settings=settings,
        )
        fake_v1_observation_count = 1
        comparison = ShadowComparator(settings).compare(
            legacy=legacy,
            v2=result,
            scope=scope,
        )

    slots = rank_evidence_slots(result.visible)
    visible_permissions = {
        candidate.evidence.effective_permission for candidate in result.visible
    }
    expected_visible = kind in {'trusted', 'raw', 'v1_parity'}
    return {
        'fake_embedding_dispatches': len(embedding_transport_calls),
        'fake_embedding_transport_calls': len(embedding_transport_calls),
        'fake_v1_observation_count': fake_v1_observation_count,
        'shadow_comparison_count': int(comparison is not None),
        'comparison_surface': comparison.surface if comparison is not None else None,
        'comparison_backend': (
            comparison.configured_backend if comparison is not None else None
        ),
        'v1_permission': (
            legacy.candidate_window[0].effective_permission
            if legacy is not None
            else None
        ),
        'v1_query_context_version': (
            legacy.query_context_version if legacy is not None else None
        ),
        'shadow_outcome': comparison.outcome if comparison is not None else None,
        'common_cohort_count': (
            comparison.common_cohort_count if comparison is not None else 0
        ),
        'common_cohort_exact_match_count': (
            comparison.common_cohort_exact_match_count
            if comparison is not None
            else 0
        ),
        'intended_delta_count': (
            comparison.v2_source_observation_delta_count
            + comparison.trust_tier_reorder_delta_count
            + comparison.raw_public_identity_repair_delta_count
            + comparison.bounded_hidden_delta_count
            + comparison.assistant_context_security_delta_count
            if comparison is not None
            else 0
        ),
        'unclassified_shadow_mismatch_count': (
            comparison.unclassified_shadow_mismatch_count
            if comparison is not None
            else 0
        ),
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


@pytest.mark.parametrize(
    'case',
    [case for case in _cases() if case['evidence_kind'] == 'v1_parity'],
    ids=lambda case: case['case_id'],
)
def test_v1_parity_cases_execute_comparator_over_fake_v1_and_real_v2(case):
    """Catches relabeling an ordinary V2 result as an executed V1 parity case."""
    observation = _execute_provider_free_case(case)

    assert observation['fake_v1_observation_count'] == 1
    assert observation['shadow_comparison_count'] == 1
    assert observation['comparison_surface'] == case['surface']
    assert observation['comparison_backend'] == case['backend']
    assert observation['v1_permission'] == case['permission']
    assert observation['v1_query_context_version'] == (
        'assistant-context:v1'
        if case['surface'] == 'assistant'
        else 'direct-query:v1'
    )
    assert observation['shadow_outcome'] == 'shadow_match'
    assert observation['common_cohort_count'] == 1
    assert observation['common_cohort_exact_match_count'] == 1
    assert observation['intended_delta_count'] == 0
    assert observation['unclassified_shadow_mismatch_count'] == 0


def test_v1_parity_execution_is_distinct_from_ordinary_raw_case():
    """Catches the v1_parity manifest label having no executable effect."""
    parity_case = next(
        case for case in _cases() if case['evidence_kind'] == 'v1_parity'
    )
    raw_case = {**parity_case, 'evidence_kind': 'raw'}

    parity = _execute_provider_free_case(parity_case)
    raw = _execute_provider_free_case(raw_case)

    assert parity['fake_v1_observation_count'] == 1
    assert parity['shadow_comparison_count'] == 1
    assert raw['fake_v1_observation_count'] == 0
    assert raw['shadow_comparison_count'] == 0


@pytest.mark.parametrize('case', _cases(), ids=lambda case: case['case_id'])
def test_embedding_count_comes_from_in_process_fake_transport_calls(case):
    """Catches injecting a prepared embedding and assigning a synthetic count."""
    observation = _execute_provider_free_case(case)

    assert observation['fake_embedding_transport_calls'] == (
        1 if case['backend'] == 'pgvector' else 0
    )


def test_injected_socket_call_is_observed_by_zero_network_gate(monkeypatch):
    """Catches a Runnable opening a socket while the golden reports literal zero."""
    import socket

    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever

    actual_calls = []
    original_invoke = KeywordEvidenceRetriever.invoke

    def fake_create_connection(*args, **kwargs):
        actual_calls.append((args, kwargs))
        return object()

    def regressed_invoke(self, *args, **kwargs):
        socket.create_connection(('example.test', 443))
        return original_invoke(self, *args, **kwargs)

    monkeypatch.setattr(socket, 'create_connection', fake_create_connection)
    monkeypatch.setattr(KeywordEvidenceRetriever, 'invoke', regressed_invoke)
    case = next(case for case in _cases() if case['backend'] == 'keyword')

    observation = _execute_provider_free_case(case)

    assert len(actual_calls) == 1
    assert observation['network_calls'] == 1


def test_injected_external_provider_send_is_observed_by_zero_call_gate(monkeypatch):
    """Catches an external provider send hidden behind a literal zero counter."""
    from backend.app.agent_runtime.rag_provider_transport import (
        _DirectOpenAIProviderClient,
    )
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever

    actual_calls = []
    original_invoke = KeywordEvidenceRetriever.invoke

    def fake_provider_send(*args, **kwargs):
        actual_calls.append((args, kwargs))
        return object()

    def regressed_invoke(self, *args, **kwargs):
        _DirectOpenAIProviderClient.send(
            object(), b'{}', timeout_seconds=30, max_retries=0
        )
        return original_invoke(self, *args, **kwargs)

    monkeypatch.setattr(_DirectOpenAIProviderClient, 'send', fake_provider_send)
    monkeypatch.setattr(KeywordEvidenceRetriever, 'invoke', regressed_invoke)
    case = next(case for case in _cases() if case['backend'] == 'keyword')

    observation = _execute_provider_free_case(case)

    assert len(actual_calls) == 1
    assert observation['external_provider_calls'] == 1
