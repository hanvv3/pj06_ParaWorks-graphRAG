from __future__ import annotations

import hmac
import math
from dataclasses import asdict, dataclass
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
        if (
            legacy.configured_backend != v2.configured_backend
            or legacy.effective_backend != v2.effective_backend
            or legacy.query_context_version != 'direct-query:v1'
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
