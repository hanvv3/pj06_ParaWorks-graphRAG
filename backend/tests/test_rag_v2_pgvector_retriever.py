from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest
from langchain_core.runnables import Runnable

from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    security_scope_fingerprint,
)
from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
    StrictQueryEmbeddingAdapter,
)
from backend.app.core.config import Settings
from backend.app.rag.embeddings import validate_query_embedding_vector
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.pgvector_retriever import (
    PgVectorEvidenceRetriever,
    PgVectorSearchRuntimeError,
)
from backend.app.rag.pgvector_store import PgVectorServingCandidateRow
from backend.app.rag.retrieval import (
    ClassifiedRetrievalCandidate,
    PreparedPaidCallBudget,
    QueryEmbeddingCallResult,
    RetrievalRequest,
    RetrievalResult,
    SanitizedRetrievalTrace,
    StrictProviderUsage,
)
from backend.app.rag.search_store import SqlAlchemyPgVectorSearchStore
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    RawChunkProvenance,
    RawServingVersionEnvelope,
    ServingEvidence,
)
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.tests.test_rag_v2_keyword_retriever import (
    _request as _canonical_request,
)
from backend.tests.test_rag_v2_keyword_retriever import (
    _scope as _canonical_scope,
)
from backend.tests.test_rag_v2_keyword_retriever import (
    _seed_sqlite_raw_projection,
)
from backend.tests.test_rag_v2_keyword_retriever import (
    _settings as _canonical_settings,
)


def _settings() -> Settings:
    return Settings(
        database_url='sqlite://',
        agent_runtime_fingerprint_secret='task-7-retrieval-secret-with-32-bytes',
        agent_runtime_fingerprint_key_version='task7-v1',
        openai_embedding_model='text-embedding-3-small',
        openai_embedding_dimensions=1536,
    )


def _scope() -> SecurityScope:
    return SecurityScope(
        contract_version='rag-security-scope:v1',
        principal_subject='user-1',
        workspace_scope_id='workspace-1',
        resource_scope_mode='all_current_scope',
        project_constraints=(),
        source_constraints=(),
        allowed_permission_levels=('public',),
        auth_policy_version='demo-auth:v1',
        permission_policy_version='rag-permission-policy:v1',
    )


def _readiness(*, corpus: int = 7, vector: int = 11, hmac: str = 'a' * 64, ready: bool = True) -> RagServingIndexReadiness:
    return RagServingIndexReadiness(
        ready=ready,
        corpus_generation=corpus,
        vector_index_generation=vector,
        expected_document_count=1,
        live_vector_count=1,
        tombstone_count=0,
        mismatch_count_capped_at_20=0 if ready else 1,
        embedding_model='text-embedding-3-small',
        embedding_dimensions=1536,
        index_policy_version='rag-v2-serving-index:v1',
        readiness_snapshot_hmac=hmac,
    )


class _StrictUsageParser:
    def parse_usage(self, usage: object) -> StrictProviderUsage:
        assert usage == {'prompt_tokens': 3, 'total_tokens': 3}
        return StrictProviderUsage(input_tokens=3, output_tokens=0, total_tokens=3)


class _StrictCostPolicy:
    def prepare_query_embedding(self, value) -> PreparedPaidCallBudget:
        assert value.retrieval_query_utf8
        return PreparedPaidCallBudget(
            component='query_embedding',
            estimated_input_tokens=3,
            maximum_output_tokens=0,
            reserved_cost_usd=Decimal('0.000777'),
            cost_policy_snapshot_hmac='b' * 64,
            estimator_input_hmac='c' * 64,
        )

    def charge_actual(self, component, usage) -> Decimal:
        assert component == 'query_embedding'
        assert usage.input_tokens == 3
        return Decimal('0.000123')


class _StrictTransport:
    def dispatch(self, prepared) -> object:
        del prepared
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


class _StrictPermit:
    def __init__(self) -> None:
        self.used = False

    def consume_at_dispatch(self) -> None:
        assert self.used is False
        self.used = True


def _embedding_result(*, query: str = 'exact query') -> QueryEmbeddingCallResult:
    settings = _settings()
    scope = _scope()
    request = RetrievalRequest(
        retrieval_query_text=query,
        security_scope=scope,
        security_scope_fingerprint=security_scope_fingerprint(
            scope,
            settings=settings,
        ),
        query_embedding_result=None,
        candidate_scan_limit=50,
        visible_limit=5,
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )
    adapter = StrictQueryEmbeddingAdapter(
        usage_parser=_StrictUsageParser(),
        cost_policy=_StrictCostPolicy(),
        transport=_StrictTransport(),
        settings=settings,
    )
    prepared = adapter.prepare(request, _readiness())
    return adapter.dispatch_once(prepared, _StrictPermit())


def _request(
    *,
    query: str = 'exact query',
    fingerprint: str | None = None,
    visible_limit: int = 5,
) -> RetrievalRequest:
    settings = _settings()
    scope = _scope()
    return RetrievalRequest(
        retrieval_query_text=query,
        security_scope=scope,
        security_scope_fingerprint=(
            security_scope_fingerprint(scope, settings=settings)
            if fingerprint is None
            else fingerprint
        ),
        query_embedding_result=_embedding_result(query=query),
        candidate_scan_limit=50,
        visible_limit=visible_limit,  # type: ignore[arg-type]
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )


def _evidence(identifier: int, permission: str = 'public') -> ServingEvidence:
    version = RawServingVersionEnvelope(
        serving_document_id=f'chunk:{identifier}',
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
        chunk_policy_version='chunk-policy:v1',
        model_content_hmac='b' * 64,
        canonical_citation_projection_hmac='c' * 64,
        effective_permission=permission,  # type: ignore[arg-type]
    )
    provenance = RawChunkProvenance(branch='raw_chunk', raw_version=version)
    return ServingEvidence(
        serving_document_id=f'chunk:{identifier}',
        serving_kind='raw_chunk',
        public_source_id=f'source-{identifier}',
        public_source_type='gmail',
        support_mode='source_observation',
        model_content=f'evidence {identifier}',
        title=f'title {identifier}',
        effective_permission=permission,  # type: ignore[arg-type]
        serving_identity_hmac=f'{identifier:064x}',
        serving_version_fingerprint=f'{identifier + 1000:064x}',
        model_content_hmac='b' * 64,
        canonical_citation_projection_hmac='c' * 64,
        version_envelope=version,
        provenance=provenance,
    )


def _candidate(identifier: int, score: float, visibility: str = 'visible') -> ClassifiedRetrievalCandidate:
    return ClassifiedRetrievalCandidate(
        evidence=_evidence(identifier, 'public' if visibility == 'visible' else 'internal'),
        relevance_score=score,
        matched_terms=(),
        access=EvidenceAccessClassification(
            global_eligibility='eligible',
            resource_scope='in_scope',
            permission_visibility=visibility,  # type: ignore[arg-type]
        ),
    )


class _Store:
    def __init__(self, result=()) -> None:
        self.result = result
        self.calls: list[tuple[RetrievalRequest, object]] = []

    def search(self, request: RetrievalRequest, vector) -> tuple[ClassifiedRetrievalCandidate, ...]:
        self.calls.append((request, vector))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _Readiness:
    def __init__(
        self,
        values: list[RagServingIndexReadiness | Exception],
    ) -> None:
        self.values = values
        self.calls = 0

    def inspect(self) -> RagServingIndexReadiness:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


class _Keyword(Runnable[RetrievalRequest, RetrievalResult]):
    def __init__(self) -> None:
        self.requests: list[RetrievalRequest] = []

    def invoke(self, input: RetrievalRequest, config=None) -> RetrievalResult:
        del config
        self.requests.append(input)
        return RetrievalResult(
            configured_backend='keyword',
            effective_backend='deterministic_lexical',
            visible=(),
            hidden_match_count=2,
            hidden_count_capped=False,
            top_candidate_window_hmac='9' * 64,
            query_embedding_receipt=None,
            trace=SanitizedRetrievalTrace(
                candidate_window_count=2,
                visible_count=0,
                hidden_match_count=2,
                provider_attempt_count=0,
                latency_ms=1,
                fallback_category=None,
            ),
        )


def _retriever(store: _Store, readiness: _Readiness, keyword: _Keyword | None = None) -> PgVectorEvidenceRetriever:
    return PgVectorEvidenceRetriever(
        store=store,
        readiness=readiness,
        keyword_retriever=keyword or _Keyword(),
        settings=_settings(),
    )


def test_pgvector_retriever_is_runnable_and_applies_score_window_then_permission() -> None:
    candidates = tuple(
        _candidate(index + 1, 1.0 - index / 100.0, 'denied_known')
        for index in range(50)
    ) + tuple(_candidate(index + 51, 0.5, 'visible') for index in range(10))
    store = _Store(candidates)
    readiness = _Readiness([_readiness(), _readiness()])
    retriever = _retriever(store, readiness)
    assert isinstance(retriever, Runnable)
    request = _request()

    result = retriever.invoke(request)

    assert result.visible == ()
    assert result.hidden_match_count == 20
    assert result.hidden_count_capped is True
    assert result.trace.candidate_window_count == 50
    assert result.trace.provider_attempt_count == 1
    assert store.calls[0][1] is request.query_embedding_result.vector


@pytest.mark.parametrize(('visible_limit', 'expected'), ((5, 5), (8, 8)))
def test_pgvector_retriever_uses_cosine_threshold_and_exact_visible_bound(
    visible_limit: int,
    expected: int,
) -> None:
    candidates = tuple(
        _candidate(index + 1, 0.9 - index / 100.0) for index in range(10)
    ) + (_candidate(99, 0.249999),)
    store = _Store(candidates)
    result = _retriever(store, _Readiness([_readiness(), _readiness()])).invoke(
        _request(visible_limit=visible_limit)
    )
    assert len(result.visible) == expected
    assert all(item.relevance_score >= 0.25 for item in result.visible)
    assert all(item.evidence.serving_document_id != 'chunk:99' for item in result.visible)
    assert result.effective_backend == 'pgvector'


def test_scope_fingerprint_and_query_carrier_are_checked_before_store_or_fallback() -> None:
    store = _Store()
    readiness = _Readiness([_readiness()])
    keyword = _Keyword()
    retriever = _retriever(store, readiness, keyword)
    with pytest.raises(ValueError, match='fingerprint'):
        retriever.invoke(_request(fingerprint='0' * 64))
    mismatch = _request(query='exact query')
    mismatch = replace(mismatch, retrieval_query_text='different query')
    with pytest.raises(ValueError, match='query bytes'):
        retriever.invoke(mismatch)
    assert readiness.calls == 0
    assert store.calls == []
    assert keyword.requests == []


def test_generation_drift_after_sql_discards_vector_result_and_falls_back_once() -> None:
    store = _Store((_candidate(1, 0.9),))
    readiness = _Readiness([_readiness(), _readiness(corpus=8)])
    keyword = _Keyword()
    request = _request()
    result = _retriever(store, readiness, keyword).invoke(request)

    assert result.visible == ()
    assert result.effective_backend == 'deterministic_lexical'
    assert result.configured_backend == 'pgvector'
    assert result.query_embedding_receipt is request.query_embedding_result.receipt
    assert result.trace.fallback_category == 'serving_corpus_changed_during_pgvector_query'
    assert result.trace.provider_attempt_count == 1
    assert len(store.calls) == 1
    assert len(keyword.requests) == 1
    assert keyword.requests[0].security_scope is request.security_scope
    assert keyword.requests[0].security_scope_fingerprint == request.security_scope_fingerprint
    assert keyword.requests[0].query_embedding_result is None


def test_storage_runtime_failure_after_valid_carrier_is_the_only_runtime_fallback() -> None:
    keyword = _Keyword()
    request = _request()
    result = _retriever(
        _Store(PgVectorSearchRuntimeError('raw storage detail')),
        _Readiness([_readiness()]),
        keyword,
    ).invoke(request)
    assert result.trace.fallback_category == 'pgvector_storage_runtime_failure'
    assert result.query_embedding_receipt is request.query_embedding_result.receipt
    assert len(keyword.requests) == 1


def test_store_failure_still_reads_post_readiness_and_drift_category_wins() -> None:
    keyword = _Keyword()
    request = _request()
    readiness = _Readiness([_readiness(), _readiness(corpus=8)])

    result = _retriever(
        _Store(PgVectorSearchRuntimeError('storage failed during overlap')),
        readiness,
        keyword,
    ).invoke(request)

    assert readiness.calls == 2
    assert result.trace.fallback_category == (
        'serving_corpus_changed_during_pgvector_query'
    )
    assert result.trace.provider_attempt_count == 1
    assert result.query_embedding_receipt is request.query_embedding_result.receipt
    assert len(keyword.requests) == 1
    assert keyword.requests[0].security_scope is request.security_scope
    assert keyword.requests[0].query_embedding_result is None


def test_post_readiness_inspection_failure_fails_closed_without_partial_result() -> None:
    keyword = _Keyword()
    readiness = _Readiness(
        [_readiness(), RuntimeError('raw readiness inspection detail')]
    )

    with pytest.raises(
        RuntimeError,
        match='pgvector post-readiness observation failed',
    ):
        _retriever(
            _Store((_candidate(1, 0.9),)),
            readiness,
            keyword,
        ).invoke(_request())

    assert readiness.calls == 2
    assert keyword.requests == []


def test_pre_readiness_inspection_failure_fails_closed_before_store_or_fallback() -> None:
    keyword = _Keyword()
    store = _Store((_candidate(1, 0.9),))
    readiness = _Readiness([RuntimeError('raw readiness inspection detail')])

    with pytest.raises(
        RuntimeError,
        match='pgvector pre-readiness observation failed',
    ):
        _retriever(store, readiness, keyword).invoke(_request())

    assert readiness.calls == 1
    assert store.calls == []
    assert keyword.requests == []


def _mutate_model_hmac(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        prepared=replace(result.prepared, model_config_snapshot_hmac='0' * 64),
    )


def _mutate_provider_hmac(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        receipt=replace(result.receipt, provider_policy_snapshot_hmac='0' * 64),
    )


def _mutate_query_hmac(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        prepared=replace(result.prepared, retrieval_query_hmac='0' * 64),
    )


def _mutate_attempt_fence(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        prepared=replace(result.prepared, attempt_fence_hmac='0' * 64),
    )


def _mutate_readiness_hmac(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        prepared=replace(result.prepared, readiness_snapshot_hmac='d' * 64),
    )


def _mutate_readiness_generation(
    result: QueryEmbeddingCallResult,
) -> QueryEmbeddingCallResult:
    return replace(
        result,
        prepared=replace(result.prepared, corpus_generation=8),
    )


def _mutate_budget_binding(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        prepared=replace(
            result.prepared,
            budget=replace(
                result.prepared.budget,
                estimator_input_hmac='d' * 64,
            ),
        ),
    )


def _mutate_receipt_model_identity(
    result: QueryEmbeddingCallResult,
) -> QueryEmbeddingCallResult:
    return replace(
        result,
        receipt=replace(result.receipt, model_config_snapshot_hmac='d' * 64),
    )


def _mutate_wrong_hash(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        vector=replace(
            result.vector,
            canonical_big_endian_float32_sha256='0' * 64,
        ),
    )


def _mutate_wrong_dimensions(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        vector=replace(result.vector, coordinates=(1.0,)),
    )


def _mutate_zero_vector(result: QueryEmbeddingCallResult) -> QueryEmbeddingCallResult:
    return replace(
        result,
        vector=replace(result.vector, coordinates=(0.0,) * 1536),
    )


@pytest.mark.parametrize(
    'mutation',
    (
        _mutate_model_hmac,
        _mutate_provider_hmac,
        _mutate_query_hmac,
        _mutate_attempt_fence,
        _mutate_readiness_hmac,
        _mutate_readiness_generation,
        _mutate_budget_binding,
        _mutate_receipt_model_identity,
        _mutate_wrong_hash,
        _mutate_wrong_dimensions,
        _mutate_zero_vector,
    ),
)
def test_forged_embedding_carrier_is_terminal_before_readiness_store_or_fallback(
    mutation,
) -> None:
    request = _request()
    request = replace(
        request,
        query_embedding_result=mutation(request.query_embedding_result),
    )
    store = _Store()
    readiness = _Readiness([_readiness()])
    keyword = _Keyword()

    with pytest.raises(ValueError, match='query embedding carrier is invalid'):
        _retriever(store, readiness, keyword).invoke(request)

    assert readiness.calls == 0
    assert store.calls == []
    assert keyword.requests == []


def test_strict_adapter_result_is_accepted_end_to_end_by_pgvector_retriever() -> None:
    request = _request()
    store = _Store((_candidate(1, 0.9),))
    result = _retriever(
        store,
        _Readiness([_readiness(), _readiness()]),
    ).invoke(request)

    assert result.effective_backend == 'pgvector'
    assert result.query_embedding_receipt is request.query_embedding_result.receipt
    assert store.calls[0][1] is request.query_embedding_result.vector


def test_unexpected_store_error_does_not_fallback_but_post_embedding_drift_does() -> None:
    keyword = _Keyword()
    with pytest.raises(RuntimeError, match='programming defect'):
        _retriever(
            _Store(RuntimeError('programming defect')),
            _Readiness([_readiness()]),
            keyword,
        ).invoke(_request())
    assert keyword.requests == []

    result = _retriever(
        _Store(),
        _Readiness([_readiness(ready=False)]),
        keyword,
    ).invoke(_request())
    assert result.trace.fallback_category == (
        'serving_corpus_changed_during_pgvector_query'
    )
    assert len(keyword.requests) == 1


def test_store_side_value_error_is_not_remapped_to_allowed_keyword_fallback() -> None:
    keyword = _Keyword()
    readiness = _Readiness([_readiness()])

    with pytest.raises(ValueError, match='invalid admitted carrier'):
        _retriever(
            _Store(ValueError('invalid admitted carrier')),
            readiness,
            keyword,
        ).invoke(_request())

    assert readiness.calls == 1
    assert keyword.requests == []


def test_sqlalchemy_vector_store_projects_from_fresh_canonical_rows(db_session) -> None:
    source, chunk = _seed_sqlite_raw_projection(db_session)
    settings = _canonical_settings()
    observation_request = _canonical_request(
        scope=_canonical_scope(
            allowed_permission_levels=('public', 'internal'),
        )
    )
    observation = CanonicalSourceObservationResolver(
        db=db_session,
        settings=settings,
    ).resolve_for_index(chunk.id)
    assert observation is not None
    evidence = observation.evidence
    row = PgVectorServingCandidateRow(
        serving_document_id=evidence.serving_document_id,
        serving_kind=evidence.serving_kind,
        support_mode=evidence.support_mode,
        effective_permission=evidence.effective_permission,
        serving_identity_hmac=evidence.serving_identity_hmac,
        serving_version_fingerprint=evidence.serving_version_fingerprint,
        model_content_hmac=evidence.model_content_hmac,
        canonical_citation_projection_hmac=(
            evidence.canonical_citation_projection_hmac
        ),
        title_lower=evidence.title.lower(),
        searchable_lower=f'{evidence.title}\n{evidence.model_content}'.lower(),
        score=0.75,
    )

    class _RankOnlyStore:
        def search_rag_v2(self, *, request, query_embedding):
            del request, query_embedding
            return (row,)

    store = SqlAlchemyPgVectorSearchStore(
        db=db_session,
        store=_RankOnlyStore(),  # type: ignore[arg-type]
        settings=settings,
    )
    carrier = validate_query_embedding_vector(
        [1.0, *([0.0] * 1535)],
    )

    candidates = store.search(observation_request, carrier)

    assert len(candidates) == 1
    assert candidates[0].evidence.public_source_id == source.source_id
    assert candidates[0].evidence.model_content == chunk.text
    assert candidates[0].access.permission_visibility == 'visible'
