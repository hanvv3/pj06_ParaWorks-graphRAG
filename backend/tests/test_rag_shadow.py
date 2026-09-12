from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.rag.retrieval import (
    RetrievalCandidate,
    RetrievalResult,
    SanitizedRetrievalTrace,
)
from backend.app.rag.serving_contracts import (
    RawChunkProvenance,
    RawServingVersionEnvelope,
    ServingEvidence,
)

SETTINGS = Settings(
    _env_file=None,
    agent_runtime_fingerprint_secret='task21-shadow-test-secret-32-bytes',
)


def _scope() -> SecurityScope:
    return SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='user-1',
        workspace_scope_id='default',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public', 'internal'),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )


def _evidence(identifier: int, *, kind: str = 'raw_chunk') -> ServingEvidence:
    document_id = f'chunk:{identifier}' if kind == 'raw_chunk' else f'decision_record:{identifier}'
    version = RawServingVersionEnvelope(
        serving_document_id=document_id,
        source_row_id=identifier,
        public_source_id=f'source-{identifier}',
        document_id=identifier,
        document_version_id=identifier,
        current_document_version_id=identifier,
        document_chunk_id=identifier,
        parser_run_id=identifier,
        external_revision=None,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='a' * 64,
        parser_policy_version='parser-policy:v1',
        parser_version='parser:v1',
        chunk_policy_version='chunk:v1',
        model_content_hmac=f'{identifier + 100:064x}',
        canonical_citation_projection_hmac=f'{identifier + 200:064x}',
        effective_permission='public',
    )
    provenance = RawChunkProvenance(branch='raw_chunk', raw_version=version)
    return ServingEvidence(
        serving_document_id=document_id,
        serving_kind=kind,
        public_source_id=f'source-{identifier}',
        public_source_type='gmail',
        support_mode='source_observation' if kind == 'raw_chunk' else 'trusted_fact',
        model_content=f'content {identifier}',
        title=f'title {identifier}',
        effective_permission='public',
        serving_identity_hmac=f'{identifier:064x}',
        serving_version_fingerprint=f'{identifier + 300:064x}',
        model_content_hmac=version.model_content_hmac,
        canonical_citation_projection_hmac=version.canonical_citation_projection_hmac,
        version_envelope=version,
        provenance=provenance,
    )


def _result(*evidence: ServingEvidence, hidden: int = 0, capped: bool = False) -> RetrievalResult:
    return RetrievalResult(
        configured_backend='keyword',
        effective_backend='deterministic_lexical',
        visible=tuple(
            RetrievalCandidate(item, 0.9 - index * 0.1, ('alpha',))
            for index, item in enumerate(evidence)
        ),
        hidden_match_count=hidden,
        hidden_count_capped=capped,
        top_candidate_window_hmac='9' * 64,
        query_embedding_receipt=None,
        trace=SanitizedRetrievalTrace(len(evidence) + hidden, len(evidence), hidden, 0, 1, None),
    )


def _legacy(*evidence: ServingEvidence, hidden: int = 0):
    from backend.app.rag.shadow import (
        LegacyRetrievalCandidateObservation,
        LegacyRetrievalObservation,
    )

    scope = _scope()
    candidates = tuple(
        LegacyRetrievalCandidateObservation(
            ordinal=index,
            legacy_serving_document_id=item.serving_document_id,
            visibility='visible',
            legacy_public_source_id=item.public_source_id,
            support_class='raw_candidate' if item.serving_kind == 'raw_chunk' else 'trusted_candidate',
            effective_permission=item.effective_permission,
            relevance_score=0.9 - index * 0.1,
            matched_terms=('alpha',),
            public_projection_hmac=item.canonical_citation_projection_hmac,
            candidate_identity_hmac=item.serving_identity_hmac,
        )
        for index, item in enumerate(evidence)
    )
    return LegacyRetrievalObservation.build(
        surface='ask',
        configured_backend='keyword',
        effective_backend='deterministic_lexical',
        query_context_version='direct-query:v1',
        retrieval_query_hmac='1' * 64,
        security_scope_fingerprint=security_scope_fingerprint(scope, settings=SETTINGS),
        candidate_window=candidates,
        hidden_match_count=hidden,
        hidden_count_capped=False,
        query_embedding_attempt_fence_hmac=None,
        public_agent_run_correlation_hmac='2' * 64,
        latency_ms=1,
        settings=SETTINGS,
    )


def _rebuild_legacy(legacy, **changes):
    from backend.app.rag.shadow import LegacyRetrievalObservation

    values = {
        'surface': legacy.surface,
        'configured_backend': legacy.configured_backend,
        'effective_backend': legacy.effective_backend,
        'query_context_version': legacy.query_context_version,
        'retrieval_query_hmac': legacy.retrieval_query_hmac,
        'security_scope_fingerprint': legacy.security_scope_fingerprint,
        'candidate_window': legacy.candidate_window,
        'hidden_match_count': legacy.hidden_match_count,
        'hidden_count_capped': legacy.hidden_count_capped,
        'query_embedding_attempt_fence_hmac': legacy.query_embedding_attempt_fence_hmac,
        'public_agent_run_correlation_hmac': legacy.public_agent_run_correlation_hmac,
        'latency_ms': legacy.latency_ms,
        'settings': SETTINGS,
    }
    values.update(changes)
    return LegacyRetrievalObservation.build(**values)


def test_comparator_exact_common_cohort_matches_without_raw_identity_persistence():
    """Catches a comparator that persists raw IDs or misses exact parity."""
    from backend.app.rag.shadow import ShadowComparator

    evidence = (_evidence(1, kind='trusted_knowledge'), _evidence(2))
    comparison = ShadowComparator(SETTINGS).compare(
        legacy=_legacy(*evidence), v2=_result(*evidence), scope=_scope(),
    )

    assert comparison.outcome == 'shadow_match'
    assert comparison.common_cohort_count == 2
    assert comparison.common_cohort_exact_match_count == 2
    assert comparison.unclassified_shadow_mismatch_count == 0
    serialized = repr(comparison)
    assert 'chunk:' not in serialized
    assert 'source-' not in serialized


@pytest.mark.parametrize(
    ('mutation', 'expected_field'),
    (
        ('v2_raw_extra', 'v2_source_observation_delta_count'),
        ('trust_reorder', 'trust_tier_reorder_delta_count'),
        ('raw_projection_repair', 'raw_public_identity_repair_delta_count'),
        ('bounded_hidden', 'bounded_hidden_delta_count'),
    ),
)
def test_comparator_classifies_only_allowed_non_assistant_deltas(mutation, expected_field):
    """Catches treating an intended D delta as an unclassified parity failure."""
    from backend.app.rag.shadow import ShadowComparator

    trusted, raw = _evidence(1, kind='trusted_knowledge'), _evidence(2)
    legacy = _legacy(trusted, raw)
    v2 = _result(trusted, raw)
    if mutation == 'v2_raw_extra':
        v2 = _result(trusted, raw, _evidence(3))
    elif mutation == 'trust_reorder':
        rows = tuple(
            replace(row, ordinal=index)
            for index, row in enumerate(reversed(legacy.candidate_window))
        )
        legacy = _rebuild_legacy(legacy, candidate_window=rows)
    elif mutation == 'raw_projection_repair':
        rows = list(legacy.candidate_window)
        rows[1] = replace(rows[1], public_projection_hmac='f' * 64)
        legacy = _rebuild_legacy(legacy, candidate_window=tuple(rows))
    elif mutation == 'bounded_hidden':
        legacy = _legacy(trusted, raw, hidden=27)
        v2 = _result(trusted, raw, hidden=20, capped=True)

    comparison = ShadowComparator(SETTINGS).compare(legacy=legacy, v2=v2, scope=_scope())
    assert getattr(comparison, expected_field) == 1
    assert comparison.unclassified_shadow_mismatch_count == 0


@pytest.mark.parametrize('mutation', ('permission', 'relevance', 'within_tier_rank', 'trusted_projection'))
def test_comparator_marks_every_non_allowlisted_difference_red(mutation):
    """Catches permission/rank/projection/relevance drift hidden as an allowed delta."""
    from backend.app.rag.shadow import ShadowComparator

    one, two = _evidence(1), _evidence(2)
    legacy = _legacy(one, two)
    v2 = _result(one, two)
    if mutation == 'permission':
        v2 = _result(replace(one, effective_permission='internal'), two)
    elif mutation == 'relevance':
        rows = list(legacy.candidate_window)
        rows[0] = replace(rows[0], relevance_score=0.42)
        legacy = _rebuild_legacy(legacy, candidate_window=tuple(rows))
    elif mutation == 'within_tier_rank':
        legacy = _legacy(two, one)
    else:
        trusted = _evidence(1, kind='trusted_knowledge')
        legacy = _legacy(trusted)
        v2 = _result(replace(trusted, canonical_citation_projection_hmac='f' * 64))

    comparison = ShadowComparator(SETTINGS).compare(legacy=legacy, v2=v2, scope=_scope())
    assert comparison.outcome == 'shadow_mismatch'
    assert comparison.unclassified_shadow_mismatch_count + comparison.trusted_comparison_projection_unavailable_count >= 1


def test_assistant_context_delta_is_sanitized_without_a_v2_result():
    """Catches fabricating shared-vector/V2 parity for a prior-assistant query."""
    from backend.app.rag.shadow import ShadowComparator

    legacy = _rebuild_legacy(
        _legacy(_evidence(1)),
        surface='assistant',
        query_context_version='assistant-context:v1',
    )
    comparison = ShadowComparator(SETTINGS).assistant_context_security_delta(
        legacy=legacy, scope=_scope(),
    )
    assert comparison.outcome == 'shadow_match'
    assert comparison.assistant_context_security_delta_count == 1
    assert comparison.v2_candidate_count == 0
    assert comparison.unclassified_shadow_mismatch_count == 0


def test_shadow_audit_persists_only_aggregates_and_domain_hmacs(db_session):
    """Catches raw query/evidence/public/internal identifiers entering shadow audit."""
    from backend.app.models import AuditLog
    from backend.app.rag.shadow import ShadowAuditWriter, ShadowComparator

    evidence = _evidence(987654)
    comparison = ShadowComparator(SETTINGS).compare(
        legacy=_legacy(evidence), v2=_result(evidence), scope=_scope(),
    )
    row = ShadowAuditWriter(SETTINGS).append(db_session, comparison)
    db_session.flush()

    assert db_session.get(AuditLog, row.id) is row
    assert row.action == 'rag_shadow_compared'
    assert row.target_id == comparison.comparison_hmac
    assert set(row.metadata_) == set(comparison.__dataclass_fields__)
    serialized = repr(row.metadata_)
    for forbidden in ('chunk:987654', 'source-987654', 'content 987654', 'title 987654'):
        assert forbidden not in serialized
    for key, value in row.metadata_.items():
        if key.endswith('_hmac') and value is not None:
            assert isinstance(value, str) and len(value) == 64


def test_tampered_legacy_observation_is_rejected_before_comparison():
    """Catches comparing caller-mutated raw observations under a stale HMAC."""
    from backend.app.rag.shadow import ShadowComparator

    legacy = _legacy(_evidence(1))
    row = replace(legacy.candidate_window[0], relevance_score=0.5)
    with pytest.raises(ValueError, match='observation changed'):
        ShadowComparator(SETTINGS).compare(
            legacy=replace(legacy, candidate_window=(row,)),
            v2=_result(_evidence(1)),
            scope=_scope(),
        )


def test_default_keyword_shadow_runs_real_retriever_and_writes_one_audit(
    db_session,
):
    """Catches a production shadow runner that is only a fake/no-op seam."""
    from types import SimpleNamespace

    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from backend.app.agents.rag_orchestrator_agent.v2_input import (
        prepare_direct_request_text,
    )
    from backend.app.core.demo_auth import DemoUser
    from backend.app.models import AuditLog
    from backend.app.rag.shadow import run_keyword_shadow
    from backend.tests.test_rag_v2_keyword_retriever import (
        _seed_sqlite_raw_projection,
        _settings,
    )

    _seed_sqlite_raw_projection(db_session)
    settings = _settings().model_copy(update={
        'langgraph_rag_v2_mode': 'shadow',
        'langgraph_rag_v2_stage': 'ask',
        'rag_retrieval_backend': 'keyword',
    })
    actor = DemoUser(
        id='shadow-user', email='shadow@example.test', role='admin',
        permission_levels={'public', 'internal', 'restricted'},
        name='Shadow User', title='Tester', department='Platform',
    )
    prepared = prepare_direct_request_text(
        'Exact raw observation',
        key=settings.agent_runtime_fingerprint_secret.encode(),
    )
    result = run_keyword_shadow(
        session_factory=sessionmaker(bind=db_session.get_bind(), expire_on_commit=False),
        settings=settings,
        actor=actor,
        surface='ask',
        prepared_text=prepared,
        legacy_delivery=SimpleNamespace(projection=SimpleNamespace(agent_run_id=41)),
    )

    assert result is not None
    assert result.outcome == 'shadow_match'
    assert result.v2_candidate_count == 1
    db_session.expire_all()
    audits = tuple(db_session.scalars(select(AuditLog).where(
        AuditLog.action == 'rag_shadow_compared'
    )))
    assert len(audits) == 1
    assert audits[0].metadata_['comparison_hmac'] == result.comparison_hmac
