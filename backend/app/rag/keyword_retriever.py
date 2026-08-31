from __future__ import annotations

from time import perf_counter_ns

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
from backend.app.rag.retrieval import (
    ClassifiedRetrievalCandidate,
    KeywordSearchStorePort,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    SanitizedRetrievalTrace,
)


class KeywordEvidenceRetriever(Runnable[RetrievalRequest, RetrievalResult]):
    """Exact V1-compatible lexical search behind the D Runnable contract."""

    def __init__(
        self,
        *,
        store: KeywordSearchStorePort,
        settings: Settings,
    ) -> None:
        self._store = store
        self._settings = settings

    def invoke(
        self,
        input: RetrievalRequest,
        config: RunnableConfig | None = None,
    ) -> RetrievalResult:
        del config
        started_ns = perf_counter_ns()
        verify_serialized_security_scope_fingerprint(
            input.security_scope,
            serialized_fingerprint=input.security_scope_fingerprint,
            settings=self._settings,
        )
        if input.query_embedding_result is not None:
            raise ValueError('keyword retrieval does not accept a query embedding')

        candidates = self._store.search(input)
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
        public_hidden_count = min(denied_count, 20)
        latency_ms = max(0, (perf_counter_ns() - started_ns) // 1_000_000)
        trace = SanitizedRetrievalTrace(
            candidate_window_count=len(window),
            visible_count=len(visible),
            hidden_match_count=public_hidden_count,
            provider_attempt_count=0,
            latency_ms=latency_ms,
            fallback_category=None,
        )
        return RetrievalResult(
            configured_backend='keyword',
            effective_backend='deterministic_lexical',
            visible=visible,
            hidden_match_count=public_hidden_count,
            hidden_count_capped=denied_count > 20,
            top_candidate_window_hmac=_window_hmac(
                window,
                settings=self._settings,
            ),
            query_embedding_receipt=None,
            trace=trace,
        )


def _candidate_window(
    candidates: tuple[ClassifiedRetrievalCandidate, ...],
    *,
    limit: int,
) -> tuple[ClassifiedRetrievalCandidate, ...]:
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.access.global_eligibility == 'eligible'
        and candidate.access.resource_scope == 'in_scope'
        and candidate.access.permission_visibility in {'visible', 'denied_known'}
    )
    return tuple(
        sorted(
            eligible,
            key=lambda candidate: (
                0
                if candidate.evidence.serving_kind == 'trusted_knowledge'
                else 1,
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
        schema_version='rag-keyword-candidate-window:v1',
        policy_version='rag-retrieval-policy:v2.0',
    )
