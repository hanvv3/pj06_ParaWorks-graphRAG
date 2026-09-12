from __future__ import annotations

import hmac
import math
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from time import perf_counter_ns
from typing import Literal

from sqlalchemy.orm import Session

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)
from backend.app.agent_runtime.rag_v2_contracts import (
    RagEffectiveBackend,
    RagRetrievalBackend,
    RagSurface,
)
from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    exact_utf8_bytes,
    security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.models import AuditLog
from backend.app.rag.retrieval import RetrievalResult

ShadowDeltaKind = Literal[
    'v2_source_observation_delta',
    'trust_tier_reorder_delta',
    'raw_public_identity_repair_delta',
    'bounded_hidden_delta',
    'assistant_context_security_delta',
]

_HEX = frozenset('0123456789abcdef')


def _require_hmac(value: object, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError('shadow HMAC is invalid')
    return value


def _require_nonnegative(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError('shadow aggregate is invalid')
    return value


def _fingerprint(value: object, *, settings: Settings, schema: str) -> str:
    return keyed_fingerprint(
        value,
        secret=fingerprint_secret_bytes(settings)[0],
        schema_version=schema,
        policy_version='rag-shadow:v2',
    )


def _term_set_hmac(terms: tuple[str, ...], *, settings: Settings) -> str:
    if type(terms) is not tuple or any(type(term) is not str for term in terms):
        raise ValueError('shadow matched terms are invalid')
    return _fingerprint(
        [exact_utf8_bytes(term) for term in terms],
        settings=settings,
        schema='rag-shadow-matched-terms:v1',
    )


@dataclass(frozen=True, slots=True)
class LegacyRetrievalCandidateObservation:
    ordinal: int
    legacy_serving_document_id: str
    visibility: Literal['visible', 'denied_known']
    legacy_public_source_id: str | None
    support_class: Literal['trusted_candidate', 'raw_candidate']
    effective_permission: Literal['public', 'internal', 'restricted']
    relevance_score: float
    matched_terms: tuple[str, ...]
    public_projection_hmac: str | None
    candidate_identity_hmac: str

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.legacy_serving_document_id) is not str
            or not self.legacy_serving_document_id
            or self.visibility not in {'visible', 'denied_known'}
            or (
                self.legacy_public_source_id is not None
                and type(self.legacy_public_source_id) is not str
            )
            or self.support_class not in {'trusted_candidate', 'raw_candidate'}
            or self.effective_permission not in {'public', 'internal', 'restricted'}
            or type(self.relevance_score) is not float
            or not math.isfinite(self.relevance_score)
            or self.relevance_score <= 0
            or type(self.matched_terms) is not tuple
            or any(type(term) is not str for term in self.matched_terms)
        ):
            raise ValueError('legacy shadow candidate is invalid')
        _require_hmac(self.public_projection_hmac, nullable=True)
        _require_hmac(self.candidate_identity_hmac)


@dataclass(frozen=True, slots=True)
class LegacyRetrievalObservation:
    surface: RagSurface
    configured_backend: RagRetrievalBackend
    effective_backend: RagEffectiveBackend
    query_context_version: Literal['direct-query:v1', 'assistant-context:v1']
    retrieval_query_hmac: str
    security_scope_fingerprint: str
    candidate_window: tuple[LegacyRetrievalCandidateObservation, ...]
    candidate_window_hmac: str
    visible_result_set_projection_hmac: str
    hidden_match_count: int
    hidden_count_capped: bool
    query_embedding_attempt_fence_hmac: str | None
    public_agent_run_correlation_hmac: str | None
    latency_ms: int
    observation_hmac: str

    @classmethod
    def build(
        cls,
        *,
        surface: RagSurface,
        configured_backend: RagRetrievalBackend,
        effective_backend: RagEffectiveBackend,
        query_context_version: Literal['direct-query:v1', 'assistant-context:v1'],
        retrieval_query_hmac: str,
        security_scope_fingerprint: str,
        candidate_window: tuple[LegacyRetrievalCandidateObservation, ...],
        hidden_match_count: int,
        hidden_count_capped: bool,
        query_embedding_attempt_fence_hmac: str | None,
        public_agent_run_correlation_hmac: str | None,
        latency_ms: int,
        settings: Settings,
    ) -> LegacyRetrievalObservation:
        if type(candidate_window) is not tuple:
            raise ValueError('legacy candidate window is invalid')
        _validate_ordinals(candidate_window)
        candidate_window_hmac = _legacy_window_hmac(candidate_window, settings=settings)
        visible_hmac = _fingerprint(
            [
                {
                    'candidate_identity_hmac': row.candidate_identity_hmac,
                    'public_projection_hmac': row.public_projection_hmac,
                }
                for row in candidate_window
                if row.visibility == 'visible'
            ],
            settings=settings,
            schema='rag-shadow-legacy-visible-projection:v1',
        )
        observation_payload = {
            'surface': surface,
            'configured_backend': configured_backend,
            'effective_backend': effective_backend,
            'query_context_version': query_context_version,
            'retrieval_query_hmac': retrieval_query_hmac,
            'security_scope_fingerprint': security_scope_fingerprint,
            'candidate_window_hmac': candidate_window_hmac,
            'visible_result_set_projection_hmac': visible_hmac,
            'hidden_match_count': hidden_match_count,
            'hidden_count_capped': hidden_count_capped,
            'query_embedding_attempt_fence_hmac': query_embedding_attempt_fence_hmac,
            'public_agent_run_correlation_hmac': public_agent_run_correlation_hmac,
            'latency_ms': latency_ms,
        }
        observation_hmac = _fingerprint(
            observation_payload,
            settings=settings,
            schema='rag-shadow-legacy-observation:v1',
        )
        return cls(
            surface=surface,
            configured_backend=configured_backend,
            effective_backend=effective_backend,
            query_context_version=query_context_version,
            retrieval_query_hmac=retrieval_query_hmac,
            security_scope_fingerprint=security_scope_fingerprint,
            candidate_window=candidate_window,
            candidate_window_hmac=candidate_window_hmac,
            visible_result_set_projection_hmac=visible_hmac,
            hidden_match_count=hidden_match_count,
            hidden_count_capped=hidden_count_capped,
            query_embedding_attempt_fence_hmac=query_embedding_attempt_fence_hmac,
            public_agent_run_correlation_hmac=public_agent_run_correlation_hmac,
            latency_ms=latency_ms,
            observation_hmac=observation_hmac,
        )

    def verify(self, *, settings: Settings) -> None:
        rebuilt = type(self).build(
            surface=self.surface,
            configured_backend=self.configured_backend,
            effective_backend=self.effective_backend,
            query_context_version=self.query_context_version,
            retrieval_query_hmac=self.retrieval_query_hmac,
            security_scope_fingerprint=self.security_scope_fingerprint,
            candidate_window=self.candidate_window,
            hidden_match_count=self.hidden_match_count,
            hidden_count_capped=self.hidden_count_capped,
            query_embedding_attempt_fence_hmac=self.query_embedding_attempt_fence_hmac,
            public_agent_run_correlation_hmac=self.public_agent_run_correlation_hmac,
            latency_ms=self.latency_ms,
            settings=settings,
        )
        for field_name in (
            'candidate_window_hmac',
            'visible_result_set_projection_hmac',
            'observation_hmac',
        ):
            if not hmac.compare_digest(getattr(self, field_name), getattr(rebuilt, field_name)):
                raise ValueError('legacy shadow observation changed')

    def __post_init__(self) -> None:
        if (
            self.surface not in {'ask', 'search', 'assistant'}
            or self.configured_backend not in {'keyword', 'pgvector'}
            or self.effective_backend not in {'deterministic_lexical', 'pgvector'}
            or self.query_context_version
            != ('assistant-context:v1' if self.surface == 'assistant' else 'direct-query:v1')
            or type(self.candidate_window) is not tuple
            or type(self.hidden_count_capped) is not bool
        ):
            raise ValueError('legacy shadow observation is invalid')
        _validate_ordinals(self.candidate_window)
        _require_hmac(self.retrieval_query_hmac)
        _require_hmac(self.security_scope_fingerprint)
        _require_hmac(self.candidate_window_hmac)
        _require_hmac(self.visible_result_set_projection_hmac)
        _require_nonnegative(self.hidden_match_count)
        _require_hmac(self.query_embedding_attempt_fence_hmac, nullable=True)
        _require_hmac(self.public_agent_run_correlation_hmac, nullable=True)
        _require_nonnegative(self.latency_ms)
        _require_hmac(self.observation_hmac)


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    surface: RagSurface
    configured_backend: RagRetrievalBackend
    effective_backend: RagEffectiveBackend
    outcome: Literal['shadow_match', 'shadow_mismatch']
    security_scope_fingerprint: str
    legacy_observation_hmac: str
    legacy_public_run_correlation_hmac: str | None
    v2_top_candidate_window_hmac: str
    legacy_candidate_count: int
    v2_candidate_count: int
    common_cohort_count: int
    common_cohort_exact_match_count: int
    legacy_hidden_match_count: int
    v2_hidden_match_count: int
    v2_source_observation_delta_count: int
    trust_tier_reorder_delta_count: int
    raw_public_identity_repair_delta_count: int
    bounded_hidden_delta_count: int
    assistant_context_security_delta_count: int
    trusted_comparison_projection_unavailable_count: int
    unclassified_shadow_mismatch_count: int
    common_cohort_hmac: str
    intended_delta_set_hmac: str
    comparison_hmac: str
    latency_ms: int

    def audit_metadata(self) -> dict[str, object]:
        """Return only aggregates and domain-separated HMACs."""
        return asdict(self)


class ShadowComparator:
    def __init__(self, settings: Settings):
        self._settings = settings

    def compare(
        self,
        *,
        legacy: LegacyRetrievalObservation,
        v2: RetrievalResult,
        scope: SecurityScope,
    ) -> ShadowComparison:
        started = perf_counter_ns()
        legacy.verify(settings=self._settings)
        expected_scope = security_scope_fingerprint(scope, settings=self._settings)
        if not hmac.compare_digest(legacy.security_scope_fingerprint, expected_scope):
            raise ValueError('shadow security scope changed')
        _validate_result(v2)
        expected_context = (
            'assistant-context:v1'
            if legacy.surface == 'assistant'
            else 'direct-query:v1'
        )
        if (
            legacy.configured_backend != v2.configured_backend
            or legacy.effective_backend != v2.effective_backend
            or legacy.query_context_version != expected_context
        ):
            raise ValueError('shadow retrieval authority changed')

        legacy_visible = tuple(row for row in legacy.candidate_window if row.visibility == 'visible')
        legacy_by_id = _unique_legacy(legacy_visible)
        v2_by_id = _unique_v2(v2)
        common_ids = tuple(identity for identity in legacy_by_id if identity in v2_by_id)
        common_exact = 0
        raw_repairs = 0
        trusted_unavailable = 0
        unclassified = 0
        for identity in common_ids:
            before = legacy_by_id[identity]
            after = v2_by_id[identity]
            expected_class = 'trusted_candidate' if after.evidence.serving_kind == 'trusted_knowledge' else 'raw_candidate'
            structural = (
                before.support_class == expected_class
                and before.effective_permission == after.evidence.effective_permission
                and before.relevance_score == after.relevance_score
                and before.matched_terms == after.matched_terms
            )
            projection_equal = before.public_projection_hmac == after.evidence.canonical_citation_projection_hmac
            if not structural:
                unclassified += 1
            elif projection_equal:
                common_exact += 1
            elif before.support_class == 'raw_candidate':
                raw_repairs += 1
                common_exact += 1
            else:
                trusted_unavailable += 1

        v2_only = tuple(v2_by_id[key] for key in v2_by_id if key not in legacy_by_id)
        source_delta = sum(item.evidence.support_mode == 'source_observation' for item in v2_only)
        unclassified += len(v2_only) - source_delta
        unclassified += sum(identity not in v2_by_id for identity in legacy_by_id)

        legacy_tiers = _tier_orders_legacy(legacy_visible, common_ids)
        v2_tiers = _tier_orders_v2(v2, common_ids)
        trust_reorder = 0
        if legacy_tiers != v2_tiers:
            unclassified += 1
        elif tuple(row.candidate_identity_hmac for row in legacy_visible if row.candidate_identity_hmac in common_ids) != tuple(
            row.evidence.serving_identity_hmac for row in v2.visible if row.evidence.serving_identity_hmac in common_ids
        ):
            trust_reorder = 1

        bounded_hidden = 0
        if legacy.hidden_match_count != v2.hidden_match_count:
            if v2.hidden_count_capped and v2.hidden_match_count == 20 and legacy.hidden_match_count > 20:
                bounded_hidden = 1
            else:
                unclassified += 1
        elif legacy.hidden_count_capped != v2.hidden_count_capped:
            unclassified += 1

        return self._comparison(
            legacy=legacy,
            v2_top_hmac=v2.top_candidate_window_hmac,
            v2_candidate_count=len(v2.visible),
            common_ids=common_ids,
            common_exact=common_exact,
            v2_hidden=v2.hidden_match_count,
            source_delta=source_delta,
            trust_reorder=trust_reorder,
            raw_repairs=raw_repairs,
            bounded_hidden=bounded_hidden,
            assistant_delta=0,
            trusted_unavailable=trusted_unavailable,
            unclassified=unclassified,
            latency_ms=max(0, (perf_counter_ns() - started) // 1_000_000),
        )

    def assistant_context_security_delta(
        self,
        *,
        legacy: LegacyRetrievalObservation,
        scope: SecurityScope,
    ) -> ShadowComparison:
        legacy.verify(settings=self._settings)
        expected_scope = security_scope_fingerprint(scope, settings=self._settings)
        if (
            legacy.surface != 'assistant'
            or legacy.query_context_version != 'assistant-context:v1'
            or not hmac.compare_digest(legacy.security_scope_fingerprint, expected_scope)
        ):
            raise ValueError('assistant shadow context is invalid')
        empty_hmac = _fingerprint([], settings=self._settings, schema='rag-shadow-v2-window:v1')
        return self._comparison(
            legacy=legacy,
            v2_top_hmac=empty_hmac,
            v2_candidate_count=0,
            common_ids=(),
            common_exact=0,
            v2_hidden=0,
            source_delta=0,
            trust_reorder=0,
            raw_repairs=0,
            bounded_hidden=0,
            assistant_delta=1,
            trusted_unavailable=0,
            unclassified=0,
            latency_ms=0,
        )

    def _comparison(
        self,
        *,
        legacy: LegacyRetrievalObservation,
        v2_top_hmac: str,
        v2_candidate_count: int,
        common_ids: tuple[str, ...],
        common_exact: int,
        v2_hidden: int,
        source_delta: int,
        trust_reorder: int,
        raw_repairs: int,
        bounded_hidden: int,
        assistant_delta: int,
        trusted_unavailable: int,
        unclassified: int,
        latency_ms: int,
    ) -> ShadowComparison:
        common_hmac = _fingerprint(
            list(common_ids), settings=self._settings, schema='rag-shadow-common-cohort:v1'
        )
        delta_counts = {
            'v2_source_observation_delta': source_delta,
            'trust_tier_reorder_delta': trust_reorder,
            'raw_public_identity_repair_delta': raw_repairs,
            'bounded_hidden_delta': bounded_hidden,
            'assistant_context_security_delta': assistant_delta,
        }
        delta_hmac = _fingerprint(
            delta_counts, settings=self._settings, schema='rag-shadow-intended-deltas:v1'
        )
        outcome = 'shadow_match' if unclassified == 0 and trusted_unavailable == 0 else 'shadow_mismatch'
        payload = {
            'surface': legacy.surface,
            'configured_backend': legacy.configured_backend,
            'effective_backend': legacy.effective_backend,
            'outcome': outcome,
            'security_scope_fingerprint': legacy.security_scope_fingerprint,
            'legacy_observation_hmac': legacy.observation_hmac,
            'legacy_public_run_correlation_hmac': legacy.public_agent_run_correlation_hmac,
            'v2_top_candidate_window_hmac': v2_top_hmac,
            'legacy_candidate_count': len(legacy.candidate_window),
            'v2_candidate_count': v2_candidate_count,
            'common_cohort_count': len(common_ids),
            'common_cohort_exact_match_count': common_exact,
            'legacy_hidden_match_count': legacy.hidden_match_count,
            'v2_hidden_match_count': v2_hidden,
            **{f'{key}_count': value for key, value in delta_counts.items()},
            'trusted_comparison_projection_unavailable_count': trusted_unavailable,
            'unclassified_shadow_mismatch_count': unclassified,
            'common_cohort_hmac': common_hmac,
            'intended_delta_set_hmac': delta_hmac,
            'latency_ms': latency_ms,
        }
        comparison_hmac = _fingerprint(
            payload, settings=self._settings, schema='rag-shadow-comparison:v1'
        )
        return ShadowComparison(**payload, comparison_hmac=comparison_hmac)


@dataclass(frozen=True, slots=True)
class ShadowAuditWriter:
    settings: Settings

    def append(self, db: Session, comparison: ShadowComparison) -> AuditLog:
        if type(comparison) is not ShadowComparison:
            raise ValueError('shadow comparison is required')
        row = AuditLog(
            actor_id='system-rag-shadow',
            actor_email='system@paraworks.local',
            actor_role='system',
            action='rag_shadow_compared',
            target_type='rag_shadow_comparison',
            target_id=comparison.comparison_hmac,
            status='success' if comparison.outcome == 'shadow_match' else 'mismatch',
            metadata_=comparison.audit_metadata(),
        )
        db.add(row)
        db.flush()
        return row


@dataclass(frozen=True, slots=True)
class ShadowSafetyBlockerAuditWriter:
    settings: Settings

    def append(self, db: Session, *, surface: RagSurface) -> AuditLog:
        payload = {
            'surface': surface,
            'configured_backend': 'pgvector',
            'outcome': 'provider_safety_unavailable',
            'v2_admission_count': 0,
            'v2_comparison_count': 0,
            'shared_embedding_count': 0,
            'advancement_allowed': False,
        }
        blocker_hmac = _fingerprint(
            payload,
            settings=self.settings,
            schema='rag-shadow-provider-safety-blocker:v1',
        )
        row = AuditLog(
            actor_id='system-rag-shadow',
            actor_email='system@paraworks.local',
            actor_role='system',
            action='rag_shadow_provider_safety_unavailable',
            target_type='rag_shadow_safety_blocker',
            target_id=blocker_hmac,
            status='blocked',
            metadata_={**payload, 'blocker_hmac': blocker_hmac},
        )
        db.add(row)
        db.flush()
        return row


def _validate_ordinals(rows: tuple[LegacyRetrievalCandidateObservation, ...]) -> None:
    if tuple(row.ordinal for row in rows) != tuple(range(len(rows))):
        raise ValueError('legacy candidate ordinals are invalid')
    if len({row.candidate_identity_hmac for row in rows}) != len(rows):
        raise ValueError('legacy candidate identities are not unique')


def _legacy_window_hmac(
    rows: tuple[LegacyRetrievalCandidateObservation, ...], *, settings: Settings
) -> str:
    return _fingerprint(
        [
            {
                'ordinal': row.ordinal,
                'visibility': row.visibility,
                'support_class': row.support_class,
                'effective_permission': row.effective_permission,
                'relevance_score': row.relevance_score,
                'matched_terms_hmac': _term_set_hmac(row.matched_terms, settings=settings),
                'public_projection_hmac': row.public_projection_hmac,
                'candidate_identity_hmac': row.candidate_identity_hmac,
            }
            for row in rows
        ],
        settings=settings,
        schema='rag-shadow-legacy-candidate-window:v1',
    )


def _validate_result(result: object) -> RetrievalResult:
    if type(result) is not RetrievalResult:
        raise ValueError('V2 retrieval result is invalid')
    _require_hmac(result.top_candidate_window_hmac)
    _require_nonnegative(result.hidden_match_count)
    if type(result.hidden_count_capped) is not bool or type(result.visible) is not tuple:
        raise ValueError('V2 retrieval result is invalid')
    return result


def _unique_legacy(rows):
    result = {row.candidate_identity_hmac: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('legacy visible identities are not unique')
    return result


def _unique_v2(result: RetrievalResult):
    values = {row.evidence.serving_identity_hmac: row for row in result.visible}
    if len(values) != len(result.visible):
        raise ValueError('V2 visible identities are not unique')
    return values


def _tier_orders_legacy(rows, common_ids):
    common = frozenset(common_ids)
    return tuple(
        tuple(row.candidate_identity_hmac for row in rows if row.candidate_identity_hmac in common and row.support_class == tier)
        for tier in ('trusted_candidate', 'raw_candidate')
    )


def _tier_orders_v2(result, common_ids):
    common = frozenset(common_ids)
    return tuple(
        tuple(row.evidence.serving_identity_hmac for row in result.visible if row.evidence.serving_identity_hmac in common and row.evidence.serving_kind == tier)
        for tier in ('trusted_knowledge', 'raw_chunk')
    )


def run_keyword_shadow(
    *,
    session_factory,
    settings: Settings,
    actor,
    surface: RagSurface,
    prepared_text,
    legacy_delivery,
) -> ShadowComparison | None:
    """Run a real provider-free lexical comparison after V1 is committed."""
    if settings.rag_retrieval_backend != 'keyword':
        raise ValueError('keyword shadow runner requires keyword configuration')
    if surface not in {'ask', 'search', 'assistant'}:
        raise ValueError('keyword shadow surface is invalid')
    from backend.app.agent_runtime.rag_v2_identity import (
        ServerRagSecurityScopeResolver,
    )
    from backend.app.agents.rag_orchestrator_agent import service
    from backend.app.permissions.service import can_access_permission
    from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
    from backend.app.rag.retrieval import RetrievalRequest
    from backend.app.rag.search_store import SqlAlchemyKeywordSearchStore

    started = perf_counter_ns()
    with session_factory() as db:
        scope = ServerRagSecurityScopeResolver(settings).resolve(db=db, actor=actor)
        scope_hmac = security_scope_fingerprint(scope, settings=settings)
        candidates = service.retrieve_matching_evidence_candidates(
            db=db,
            question=prepared_text.retrieval_query_text,
        )
        candidates = service.filter_live_serving_candidates(
            db=db,
            candidates=candidates,
        )[:50]
        rows = tuple(
            _observe_legacy_candidate(
                db=db,
                settings=settings,
                actor=actor,
                candidate=candidate,
                ordinal=ordinal,
                can_access_permission=can_access_permission,
            )
            for ordinal, candidate in enumerate(candidates)
        )
        projection = getattr(legacy_delivery, 'projection', legacy_delivery)
        public_run_id = getattr(projection, 'agent_run_id', None)
        correlation = (
            _fingerprint(
                {'legacy_public_run_id': public_run_id},
                settings=settings,
                schema='rag-shadow-legacy-public-run-correlation:v1',
            )
            if type(public_run_id) is int and public_run_id > 0
            else None
        )
        legacy = LegacyRetrievalObservation.build(
            surface=surface,
            configured_backend='keyword',
            effective_backend='deterministic_lexical',
            query_context_version=prepared_text.query_context_version,
            retrieval_query_hmac=prepared_text.retrieval_query_hmac,
            security_scope_fingerprint=scope_hmac,
            candidate_window=rows,
            hidden_match_count=sum(row.visibility == 'denied_known' for row in rows),
            hidden_count_capped=False,
            query_embedding_attempt_fence_hmac=None,
            public_agent_run_correlation_hmac=correlation,
            latency_ms=max(0, (perf_counter_ns() - started) // 1_000_000),
            settings=settings,
        )
        request = RetrievalRequest(
            retrieval_query_text=prepared_text.retrieval_query_text,
            security_scope=scope,
            security_scope_fingerprint=scope_hmac,
            query_embedding_result=None,
            candidate_scan_limit=50,
            visible_limit=5 if surface == 'search' else 8,
            relevance_policy_version='rag-retrieval-policy:v2.0',
        )
        v2 = KeywordEvidenceRetriever(
            store=SqlAlchemyKeywordSearchStore(db=db, settings=settings),
            settings=settings,
        ).invoke(request, config={'callbacks': [], 'metadata': {}})
        comparison = ShadowComparator(settings).compare(
            legacy=legacy,
            v2=v2,
            scope=scope,
        )
        ShadowAuditWriter(settings).append(db, comparison)
        db.commit()
        return comparison


class RagShadowProviderOverrideError(RuntimeError):
    """A paid shared-embedding failure that must replace legacy parity delivery."""

    def __init__(self, outcome: str, *, component: str = 'query_embedding') -> None:
        self.outcome = outcome
        self.component = component
        super().__init__(outcome)


def run_pgvector_shadow(
    *,
    request_factory,
    session_factory,
    settings: Settings,
    actor,
    surface: RagSurface,
    prepared_text,
    legacy_invoke,
    legacy_observation_factory=None,
):
    """Own one shared pgvector embedding and never invoke answer generation."""
    if settings.rag_retrieval_backend != 'pgvector':
        raise ValueError('pgvector shadow runner requires pgvector configuration')
    if surface not in {'ask', 'search', 'assistant'}:
        raise ValueError('pgvector shadow surface is invalid')
    from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
        share_query_embedding_result,
    )
    from backend.app.rag.retrieval import RetrievalRequest

    with request_factory(
        session_factory=session_factory,
        settings=settings,
        actor=actor,
        surface=surface,
        assistant_target=None,
    ) as services:
        scope = services.security_scope_resolver.resolve(db=services.db, actor=actor)
        scope_hmac = security_scope_fingerprint(scope, settings=settings)
        request = RetrievalRequest(
            retrieval_query_text=prepared_text.retrieval_query_text,
            security_scope=scope,
            security_scope_fingerprint=scope_hmac,
            query_embedding_result=None,
            candidate_scan_limit=50,
            visible_limit=5 if surface == 'search' else 8,
            relevance_policy_version='rag-retrieval-policy:v2.0',
        )
        query_snapshot = next(
            snapshot
            for snapshot in services.policy_snapshots
            if snapshot.component == 'query_embedding'
        )
        from backend.app.agent_runtime.rag_provider_safety import (
            RagProviderSafetyError,
            RagProviderSafetyInspectionError,
        )

        try:
            preflight_safety = getattr(
                services.cost_ledger,
                'require_component_safety_ready',
                None,
            )
            if preflight_safety is not None:
                preflight_safety(query_snapshot)
        except RagProviderSafetyInspectionError:
            raise RagShadowProviderOverrideError(
                'provider_safety_unavailable'
            ) from None
        except RagProviderSafetyError:
            with suppress(Exception):
                ShadowSafetyBlockerAuditWriter(settings).append(
                    services.db,
                    surface=surface,
                )
                services.db.commit()
            return legacy_invoke(None)
        readiness = services.index_readiness.inspect(db=services.db)
        prepared_embedding = services.query_embedding_adapter.prepare_shadow_legacy(
            request, readiness
        )
        components = tuple(
            (
                snapshot,
                prepared_embedding.budget
                if snapshot.component == 'query_embedding'
                else services.cost_policy.reserve_unused_component(snapshot.component),
            )
            for snapshot in services.policy_snapshots
        )
        run_id = services.allocate_run_id()
        services.cost_ledger.create_admission(
            agent_run_id=run_id,
            surface=surface,
            mode='shadow',
            cutover_stage=settings.langgraph_rag_v2_stage,
            configured_backend='pgvector',
            query_context_version=prepared_text.query_context_version,
            current_text_hmac=prepared_text.current_text_hmac,
            retrieval_query_hmac=prepared_text.retrieval_query_hmac,
            security_scope_fingerprint=scope_hmac,
            admission_cache_identity_hmac=None,
            source_window=f'rag-v2:admission:shadow:{surface}:pgvector',
            components=components,
        )
        try:
            grant = services.cost_ledger.claim_component(
                run_id=run_id,
                component='query_embedding',
                prepared=prepared_embedding.budget,
            )
            transport = (
                services.provider_transport
                if getattr(services, 'provider_transport', None) is not None
                else services.provider_transport_factory()
            )
            dispatch = transport.prepare(grant=grant, prepared=prepared_embedding)
            delivery = transport.dispatch_and_finalize(grant=grant, prepared=dispatch)
        except Exception as exc:
            from backend.app.agent_runtime.rag_cost_ledger import (
                RagPreclaimSafetyRefusalError,
            )

            terminal = getattr(exc, 'terminal', None)
            outcome = (
                'provider_safety_unavailable'
                if isinstance(exc, RagPreclaimSafetyRefusalError)
                else getattr(exc, 'outcome', None)
                or getattr(terminal, 'outcome', None)
                or 'retriever_unavailable'
            )
            raise RagShadowProviderOverrideError(outcome) from None
        if delivery.output is None:
            final = delivery.component_final
            raise RagShadowProviderOverrideError(
                final.terminal_outcome or 'retriever_unavailable',
                component=final.component,
            )
        shared = share_query_embedding_result(
            delivery.output,
            legacy_query_text=prepared_text.retrieval_query_text,
            v2_query_text=prepared_text.retrieval_query_text,
        )
        public_delivery = legacy_invoke(shared)
        try:
            _finish_pgvector_shadow(
                services=services,
                settings=settings,
                actor=actor,
                surface=surface,
                prepared_text=prepared_text,
                shared=shared,
                scope=scope,
                scope_hmac=scope_hmac,
                public_delivery=public_delivery,
                request=request,
                readiness=readiness,
                run_id=run_id,
                legacy_observation_factory=legacy_observation_factory,
            )
        except Exception:
            with suppress(Exception):
                services.cost_ledger.finalize_inter_component_failure(
                    run_id=run_id,
                    outcome='unexpected_internal_error',
                )
        return public_delivery


def _finish_pgvector_shadow(
    *,
    services,
    settings,
    actor,
    surface,
    prepared_text,
    shared,
    scope,
    scope_hmac,
    public_delivery,
    request,
    readiness,
    run_id,
    legacy_observation_factory,
):
    if readiness.ready is not True:
        services.cost_ledger.finalize_shadow_run(
            run_id=run_id,
            outcome='serving_index_not_ready',
            comparison=None,
            settings=settings,
        )
        return
    legacy = (
        legacy_observation_factory(
            services=services,
            actor=actor,
            surface=surface,
            prepared_text=prepared_text,
            shared=shared,
            scope=scope,
            scope_hmac=scope_hmac,
            public_delivery=public_delivery,
        )
        if legacy_observation_factory is not None
        else _observe_pgvector_legacy(
            services=services,
            settings=settings,
            actor=actor,
            surface=surface,
            prepared_text=prepared_text,
            shared=shared,
            scope_hmac=scope_hmac,
            public_delivery=public_delivery,
        )
    )
    from backend.app.rag.pgvector_retriever import PgVectorEvidenceRetriever

    retriever = services.retrievers.resolve('pgvector')
    if isinstance(retriever, PgVectorEvidenceRetriever):
        retriever = retriever.with_graph_fallback()
    v2 = retriever.invoke(
        replace(request, query_embedding_result=shared),
        config={'callbacks': [], 'metadata': {}},
    )
    comparison = ShadowComparator(settings).compare(
        legacy=legacy,
        v2=v2,
        scope=scope,
    )
    services.cost_ledger.finalize_shadow_run(
        run_id=run_id,
        outcome=comparison.outcome,
        comparison=comparison,
        settings=settings,
    )


def run_assistant_context_security_delta(
    *,
    session_factory,
    settings: Settings,
    actor,
    contextual_query: str,
    public_agent_run_id: int | None,
) -> ShadowComparison:
    """Record the deliberate no-share branch without retaining context or ids."""
    from backend.app.agent_runtime.rag_v2_identity import (
        ServerRagSecurityScopeResolver,
    )

    with session_factory() as db:
        scope = ServerRagSecurityScopeResolver(settings).resolve(db=db, actor=actor)
        scope_hmac = security_scope_fingerprint(scope, settings=settings)
        backend = settings.rag_retrieval_backend
        legacy = LegacyRetrievalObservation.build(
            surface='assistant',
            configured_backend=backend,
            effective_backend=(
                'pgvector' if backend == 'pgvector' else 'deterministic_lexical'
            ),
            query_context_version='assistant-context:v1',
            retrieval_query_hmac=_fingerprint(
                {'contextual_query_bytes': exact_utf8_bytes(contextual_query)},
                settings=settings,
                schema='rag-shadow-assistant-context-query:v1',
            ),
            security_scope_fingerprint=scope_hmac,
            candidate_window=(),
            hidden_match_count=0,
            hidden_count_capped=False,
            query_embedding_attempt_fence_hmac=None,
            public_agent_run_correlation_hmac=(
                _fingerprint(
                    {'legacy_public_run_id': public_agent_run_id},
                    settings=settings,
                    schema='rag-shadow-legacy-public-run-correlation:v1',
                )
                if type(public_agent_run_id) is int and public_agent_run_id > 0
                else None
            ),
            latency_ms=0,
            settings=settings,
        )
        comparison = ShadowComparator(settings).assistant_context_security_delta(
            legacy=legacy,
            scope=scope,
        )
        ShadowAuditWriter(settings).append(db, comparison)
        db.commit()
        return comparison


def _observe_pgvector_legacy(
    *,
    services,
    settings: Settings,
    actor,
    surface,
    prepared_text,
    shared,
    scope_hmac,
    public_delivery,
):
    from backend.app.agents.rag_orchestrator_agent import service
    from backend.app.permissions.service import can_access_permission
    from backend.app.rag.search_store import build_pgvector_search_store

    store = build_pgvector_search_store(
        db=services.db,
        settings=settings,
        shared_query_embedding=shared,
    )
    if store is None:
        raise ValueError('legacy pgvector store is unavailable')
    found = store.search(
        query=prepared_text.retrieval_query_text,
        user=actor,
        limit=5 if surface == 'search' else 8,
    )
    candidates = service.filter_live_serving_candidates(
        db=services.db,
        candidates=service.candidates_from_vector_matches(found.matches),
    )
    rows = tuple(
        _observe_legacy_candidate(
            db=services.db,
            settings=settings,
            actor=actor,
            candidate=candidate,
            ordinal=ordinal,
            can_access_permission=can_access_permission,
        )
        for ordinal, candidate in enumerate(candidates)
    )
    projection = getattr(public_delivery, 'projection', None)
    public_run_id = getattr(projection, 'agent_run_id', None)
    correlation = (
        _fingerprint(
            {'legacy_public_run_id': public_run_id},
            settings=settings,
            schema='rag-shadow-legacy-public-run-correlation:v1',
        )
        if type(public_run_id) is int and public_run_id > 0
        else None
    )
    return LegacyRetrievalObservation.build(
        surface=surface,
        configured_backend='pgvector',
        effective_backend='pgvector',
        query_context_version=prepared_text.query_context_version,
        retrieval_query_hmac=prepared_text.retrieval_query_hmac,
        security_scope_fingerprint=scope_hmac,
        candidate_window=rows,
        hidden_match_count=found.hidden_match_count,
        hidden_count_capped=False,
        query_embedding_attempt_fence_hmac=shared.prepared.attempt_fence_hmac,
        public_agent_run_correlation_hmac=correlation,
        latency_ms=shared.receipt.latency_ms,
        settings=settings,
    )


def _observe_legacy_candidate(
    *, db, settings: Settings, actor, candidate, ordinal: int, can_access_permission
) -> LegacyRetrievalCandidateObservation:
    from backend.app.rag.source_observations import CanonicalSourceObservationResolver
    from backend.app.rag.trusted_evidence import TrustedServingEnvelopeResolver

    chunk_id = candidate.metadata.get('chunk_id')
    serving_document_id = (
        f'chunk:{chunk_id}' if type(chunk_id) is int else candidate.source_id
    )
    prefix, separator, raw_identifier = serving_document_id.partition(':')
    canonical = None
    if separator and raw_identifier.isascii() and raw_identifier.isdecimal():
        identifier = int(raw_identifier)
        if identifier > 0 and str(identifier) == raw_identifier:
            if prefix == 'chunk':
                resolved = CanonicalSourceObservationResolver(
                    db=db, settings=settings
                ).resolve_for_index(identifier)
                canonical = None if resolved is None else resolved.evidence
            elif prefix in {'decision_record', 'history_event', 'timeline_event', 'todo'}:
                resolved = TrustedServingEnvelopeResolver(
                    db=db, settings=settings
                ).resolve_for_index(prefix, identifier)
                canonical = None if resolved is None else resolved.evidence
    if canonical is None:
        candidate_hmac = _fingerprint(
            {'legacy_serving_document_id_bytes': exact_utf8_bytes(serving_document_id)},
            settings=settings,
            schema='rag-shadow-unresolved-legacy-candidate:v1',
        )
        public_hmac = _fingerprint(
            {
                'source_id_bytes': exact_utf8_bytes(candidate.source_id),
                'source_url_bytes': exact_utf8_bytes(candidate.source_url),
                'source_snippet_bytes': exact_utf8_bytes(candidate.source_snippet),
            },
            settings=settings,
            schema='rag-shadow-unresolved-legacy-projection:v1',
        )
        support_class = (
            'raw_candidate' if prefix == 'chunk' else 'trusted_candidate'
        )
    else:
        candidate_hmac = canonical.serving_identity_hmac
        public_hmac = (
            canonical.canonical_citation_projection_hmac
            if candidate.source_id == canonical.public_source_id
            else _fingerprint(
                {
                    'legacy_public_source_id_bytes': exact_utf8_bytes(
                        candidate.source_id
                    )
                },
                settings=settings,
                schema='rag-shadow-legacy-public-identity:v1',
            )
        )
        support_class = (
            'trusted_candidate'
            if canonical.serving_kind == 'trusted_knowledge'
            else 'raw_candidate'
        )
    return LegacyRetrievalCandidateObservation(
        ordinal=ordinal,
        legacy_serving_document_id=serving_document_id,
        visibility=(
            'visible'
            if can_access_permission(actor, candidate.permission_level)
            else 'denied_known'
        ),
        legacy_public_source_id=candidate.source_id,
        support_class=support_class,
        effective_permission=candidate.permission_level,
        relevance_score=float(candidate.relevance_score),
        matched_terms=tuple(candidate.matched_terms),
        public_projection_hmac=public_hmac,
        candidate_identity_hmac=candidate_hmac,
    )
