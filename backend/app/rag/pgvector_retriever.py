from __future__ import annotations

from dataclasses import replace
from time import perf_counter_ns
from typing import Protocol

from langchain_core.runnables import Runnable, RunnableConfig

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_identity import (
    exact_utf8_bytes,
    verify_serialized_security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.rag.embeddings import ValidatedQueryEmbeddingVector
from backend.app.rag.index_readiness import RagServingIndexReadiness
from backend.app.rag.retrieval import (
    ClassifiedRetrievalCandidate,
    RagServingReadinessSnapshot,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    SanitizedRetrievalTrace,
    validate_query_embedding_call_result,
    validate_rag_serving_index_readiness,
)


class PgVectorSearchRuntimeError(RuntimeError):
    """Sanitized PostgreSQL/pgvector read failure eligible for lexical fallback."""


class PgVectorFallbackRequiredError(RuntimeError):
    def __init__(self, category: str):
        if category not in {'pgvector_storage_runtime_failure', 'serving_corpus_changed_during_pgvector_query'}:
            raise ValueError('invalid pgvector fallback category')
        self.category = category
        super().__init__(category)


class PgVectorSearchStorePort(Protocol):
    def search(
        self,
        request: RetrievalRequest,
        vector: ValidatedQueryEmbeddingVector,
    ) -> tuple[ClassifiedRetrievalCandidate, ...]: ...


class ReadinessSnapshotPort(Protocol):
    def inspect(self) -> RagServingIndexReadiness: ...


class PgVectorEvidenceRetriever(Runnable[RetrievalRequest, RetrievalResult]):
    def __init__(
        self,
        *,
        store: PgVectorSearchStorePort,
        readiness: ReadinessSnapshotPort,
        keyword_retriever: Runnable[RetrievalRequest, RetrievalResult],
        settings: Settings,
        defer_fallback: bool = False,
    ) -> None:
        self._store = store
        self._readiness = readiness
        self._keyword_retriever = keyword_retriever
        self._settings = settings
        self._defer_fallback = defer_fallback

    def with_graph_fallback(self) -> PgVectorEvidenceRetriever:
        """Keep this request's ports, but let the graph execute lexical fallback."""
        return PgVectorEvidenceRetriever(store=self._store, readiness=self._readiness,
            keyword_retriever=self._keyword_retriever, settings=self._settings, defer_fallback=True)

    def invoke(
        self,
        input: RetrievalRequest,
        config: RunnableConfig | None = None,
    ) -> RetrievalResult:
        started_ns = perf_counter_ns()
        verify_serialized_security_scope_fingerprint(
            input.security_scope,
            serialized_fingerprint=input.security_scope_fingerprint,
            settings=self._settings,
        )
        embedding = _require_exact_embedding_carrier(input, settings=self._settings)
        try:
            before_sql = validate_rag_serving_index_readiness(
                self._readiness.inspect()
            )
        except Exception:
            raise RuntimeError('pgvector pre-readiness observation failed') from None
        if not _matches_prepared_readiness(before_sql, input):
            return self._keyword_fallback(
                input,
                config=config,
                category='serving_corpus_changed_during_pgvector_query',
                started_ns=started_ns,
            )
        store_failure: Exception | None = None
        try:
            candidates = self._store.search(input, embedding.vector)
        except Exception as exc:
            store_failure = exc
            candidates = ()
        try:
            after_sql = validate_rag_serving_index_readiness(
                self._readiness.inspect()
            )
        except Exception:
            raise RuntimeError('pgvector post-readiness observation failed') from None
        if not _matches_prepared_readiness(after_sql, input):
            return self._keyword_fallback(
                input,
                config=config,
                category='serving_corpus_changed_during_pgvector_query',
                started_ns=started_ns,
            )
        if isinstance(store_failure, PgVectorSearchRuntimeError):
            return self._keyword_fallback(
                input,
                config=config,
                category='pgvector_storage_runtime_failure',
                started_ns=started_ns,
            )
        if store_failure is not None:
            raise store_failure
        window = _candidate_window(candidates, limit=input.candidate_scan_limit)
        visible_internal = tuple(
            candidate
            for candidate in window
            if candidate.access.permission_visibility == 'visible'
        )[: input.visible_limit]
        visible = tuple(
            RetrievalCandidate(
                evidence=candidate.evidence,
                relevance_score=candidate.relevance_score,
                matched_terms=candidate.matched_terms,
            )
            for candidate in visible_internal
        )
        denied_count = sum(
            candidate.access.permission_visibility == 'denied_known'
            for candidate in window
        )
        hidden = min(denied_count, 20)
        latency_ms = _latency_ms(started_ns)
        return RetrievalResult(
            configured_backend='pgvector',
            effective_backend='pgvector',
            visible=visible,
            hidden_match_count=hidden,
            hidden_count_capped=denied_count > 20,
            top_candidate_window_hmac=_window_hmac(window, settings=self._settings),
            query_embedding_receipt=embedding.receipt,
            trace=SanitizedRetrievalTrace(
                candidate_window_count=len(window),
                visible_count=len(visible),
                hidden_match_count=hidden,
                provider_attempt_count=1,
                latency_ms=latency_ms,
                fallback_category=None,
            ),
        )

    def _keyword_fallback(
        self,
        request: RetrievalRequest,
        *,
        config: RunnableConfig | None,
        category: str,
        started_ns: int,
    ) -> RetrievalResult:
        if self._defer_fallback:
            raise PgVectorFallbackRequiredError(category)
        embedding = request.query_embedding_result
        assert embedding is not None
        lexical_request = replace(request, query_embedding_result=None)
        lexical = self._keyword_retriever.invoke(lexical_request, config=config)
        return RetrievalResult(
            configured_backend='pgvector',
            effective_backend='deterministic_lexical',
            visible=lexical.visible,
            hidden_match_count=lexical.hidden_match_count,
            hidden_count_capped=lexical.hidden_count_capped,
            top_candidate_window_hmac=lexical.top_candidate_window_hmac,
            query_embedding_receipt=embedding.receipt,
            trace=SanitizedRetrievalTrace(
                candidate_window_count=lexical.trace.candidate_window_count,
                visible_count=len(lexical.visible),
                hidden_match_count=lexical.hidden_match_count,
                provider_attempt_count=1,
                latency_ms=_latency_ms(started_ns),
                fallback_category=category,
            ),
        )


def _require_exact_embedding_carrier(
    input: RetrievalRequest,
    *,
    settings: Settings,
):
    return validate_query_embedding_call_result(input, settings=settings)


def _matches_prepared_readiness(
    readiness: RagServingReadinessSnapshot,
    request: RetrievalRequest,
) -> bool:
    result = request.query_embedding_result
    assert result is not None
    prepared = result.prepared
    return (
        (
            readiness[0],
            readiness[1],
            readiness[2],
            readiness[10],
        )
        == (
            True,
            prepared.corpus_generation,
            prepared.vector_index_generation,
            prepared.readiness_snapshot_hmac,
        )
    )


def _candidate_window(
    candidates: tuple[ClassifiedRetrievalCandidate, ...],
    *,
    limit: int,
) -> tuple[ClassifiedRetrievalCandidate, ...]:
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.relevance_score >= 0.25
        and candidate.access.global_eligibility == 'eligible'
        and candidate.access.resource_scope == 'in_scope'
        and candidate.access.permission_visibility in {'visible', 'denied_known'}
    )
    return tuple(
        sorted(
            eligible,
            key=lambda candidate: (
                0 if candidate.evidence.serving_kind == 'trusted_knowledge' else 1,
                -candidate.relevance_score,
                candidate.evidence.serving_document_id,
            ),
        )[:limit]
    )


def _window_hmac(
    window: tuple[ClassifiedRetrievalCandidate, ...],
    *,
    settings: Settings,
) -> str:
    secret, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        [
            {
                'serving_identity_hmac': candidate.evidence.serving_identity_hmac,
                'serving_version_fingerprint': (
                    candidate.evidence.serving_version_fingerprint
                ),
                'score': candidate.relevance_score,
                'matched_term_bytes': [
                    exact_utf8_bytes(term) for term in candidate.matched_terms
                ],
                'permission_visibility': candidate.access.permission_visibility,
            }
            for candidate in window
        ],
        secret=secret,
        schema_version='rag-pgvector-candidate-window:v1',
        policy_version='rag-retrieval-policy:v2.0',
    )


def _latency_ms(started_ns: int) -> int:
    return max(0, (perf_counter_ns() - started_ns) // 1_000_000)
