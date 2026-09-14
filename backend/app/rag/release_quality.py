"""Provider-free, signed quality adjudication for the frozen RAG live gate.

This module never sees a provider, raw prompt, answer, or evidence body.  The
runner may expose those values to authenticated reviewers in process memory,
but this evaluator accepts only the resulting HMAC-bound labels and sanitized
terminal projections.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Literal, cast

from backend.app.agent_runtime.fingerprints import canonical_json_bytes
from backend.app.agent_runtime.rag_safety_identity import (
    rag_identity_hmac,
    require_lower_hmac,
)
from backend.app.rag.evidence_projection import V1EvidenceProjection
from backend.app.rag.release_ledger import RagReleaseLedgerError
from backend.app.rag.release_review import (
    AuthorizedRagLiveGate,
    FrozenCaseClaimManifest,
    FrozenCorpusSnapshot,
    FrozenLiveManifestCase,
    FrozenLiveManifestSnapshot,
    LiveGateLimits,
    LiveGatePreviewError,
    ReviewerRole,
    _corpus_payload,
    _exact_frozen_equal,
    _manifest_payload,
)

ReviewLabel = Literal['entailed', 'not_entailed', 'ambiguous']
LiveCaseKind = Literal['positive', 'hard_negative']
LiveCaseSurface = Literal['ask', 'assistant']
LiveCaseOutcome = Literal[
    'supported',
    'no_match',
    'hidden_only',
    'safety_filter_empty',
    'insufficient_evidence',
    'evidence_unavailable',
    'budget_exceeded',
    'retriever_unavailable',
    'model_unavailable',
    'model_provider_failed',
    'structured_output_invalid',
    'citation_validation_failed',
    'persistence_failed',
    'unexpected_internal_error',
    'provider_safety_unavailable',
    'provider_usage_overrun',
    'provider_response_identity_invalid',
    'provider_embedding_payload_invalid',
    'live_corpus_snapshot_changed',
    'abandoned_unknown',
]
QualityFailureReason = Literal[
    'hard_negative_accuracy',
    'positive_answer_coverage',
    'faithfulness',
    'retrieval_precision',
    'retrieval_recall',
]

_POLICY = 'rag-live-gate:v1'
_RUBRIC = 'rag-live-quality-rubric:v1'
_ROLES: tuple[ReviewerRole, ...] = (
    'reviewer_a',
    'reviewer_b',
    'adjudicator_c',
)
_LABELS = frozenset(('entailed', 'not_entailed', 'ambiguous'))
_QUALITY_OUTCOMES = frozenset(
    (
        'supported',
        'no_match',
        'hidden_only',
        'safety_filter_empty',
        'insufficient_evidence',
        'evidence_unavailable',
    )
)


class RagReleaseQualityError(ValueError):
    """Bounded refusal code; input values are never included in the message."""

    def __init__(self, code: str = 'quality_input_invalid') -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SanitizedLiveBlockResult:
    block_ordinal: int
    block_result_hmac: str
    evidence_projection_hmac: str


@dataclass(frozen=True, slots=True)
class SanitizedLiveCaseResult:
    ordinal: int
    case_id_hmac: str
    case_kind: LiveCaseKind
    surface: LiveCaseSurface
    configured_backend: Literal['keyword', 'pgvector']
    state: Literal['complete', 'failed']
    outcome: LiveCaseOutcome
    runtime_agent_run_id_hmac: str
    case_projection_hmac: str | None
    current_corpus_snapshot_hmac: str
    provider_safety_snapshot_hmac: str
    query_embedding_dispatch_count: Literal[0, 1]
    answer_generation_dispatch_count: Literal[0, 1]
    reserved_cost_usd: Decimal
    charged_cost_usd: Decimal
    legacy_precision_numerator: int
    legacy_precision_denominator: int
    legacy_recall_numerator: int
    legacy_recall_denominator: int
    v2_precision_numerator: int
    v2_precision_denominator: int
    v2_recall_numerator: int
    v2_recall_denominator: int
    expected_no_answer: bool
    hard_negative_correct: bool
    positive_answered: bool
    required_slot_covered: bool
    blocks: tuple[SanitizedLiveBlockResult, ...]


@dataclass(frozen=True, slots=True)
class FrozenLegacyBaselineMetrics:
    baseline_hmac: str
    baseline_policy_snapshot_hmac: str
    evaluator_code_hmac: str
    retriever_source_bundle_hmac: str
    evaluator_source_commit: str
    fixture_manifest_hmac: str
    corpus_snapshot_hmac: str
    precision_numerator: int
    precision_denominator: int
    recall_numerator: int
    recall_denominator: int


@dataclass(frozen=True, slots=True)
class QualityBlockAdjudication:
    block_ordinal: int
    block_result_hmac: str
    evidence_projection_hmac: str
    reviewer_a_label: ReviewLabel
    reviewer_a_signature_hmac: str
    reviewer_b_label: ReviewLabel
    reviewer_b_signature_hmac: str
    adjudicator_label: ReviewLabel | None
    adjudicator_signature_hmac: str | None
    decided_label: ReviewLabel


@dataclass(frozen=True, slots=True)
class QualityCaseAdjudication:
    case_id_hmac: str
    case_projection_hmac: str
    case_kind: LiveCaseKind
    required_slot_covered: bool
    block_labels: tuple[QualityBlockAdjudication, ...]


@dataclass(frozen=True, slots=True)
class RagQualityReport:
    approval_hmac: str
    approval_id_hmac: str
    approved_corpus_snapshot_hmac: str
    current_corpus_snapshot_hmac: str
    baseline_hmac: str
    manifest_hmac: str
    reviewer_roster_hmac: str
    rubric_version: Literal['rag-live-quality-rubric:v1']
    case_adjudications: tuple[QualityCaseAdjudication, ...]
    case_count: Literal[30]
    failure_reasons: tuple[QualityFailureReason, ...]
    gate_outcome: Literal['green', 'quality_gate_failed']
    faithfulness_entailed_blocks: int
    faithfulness_total_blocks: int
    hard_negative_case_count: int
    hard_negative_correct_count: int
    positive_case_count: int
    positive_answered_count: int
    legacy_precision_numerator: int
    legacy_precision_denominator: int
    legacy_recall_numerator: int
    legacy_recall_denominator: int
    v2_precision_numerator: int
    v2_precision_denominator: int
    v2_recall_numerator: int
    v2_recall_denominator: int
    payload_canonical_bytes: bytes
    quality_report_hmac: str


@dataclass(frozen=True, slots=True, repr=False)
class EphemeralReviewBlock:
    case_id_hmac: str
    block_ordinal: int
    block_result_hmac: str
    answer_text: str
    evidence_projection: V1EvidenceProjection
    evidence_projection_hmac: str

    def __repr__(self) -> str:
        return '<EphemeralReviewBlock redacted>'


@dataclass(frozen=True, slots=True)
class SignedReviewLabel:
    case_id_hmac: str
    block_ordinal: int
    reviewer_role: ReviewerRole
    reviewer_subject_hmac: str
    label: ReviewLabel
    signature_hmac: str


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RagReleaseQualityError(code)


def _digest(value: object, code: str) -> str:
    try:
        return require_lower_hmac(value)
    except (TypeError, ValueError):
        raise RagReleaseQualityError(code) from None


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _ratio(
    numerator: object,
    denominator: object,
    *,
    code: str = 'case_result_invalid',
) -> tuple[int, int]:
    _require(
        _nonnegative_int(numerator)
        and _nonnegative_int(denominator)
        and cast(int, numerator) <= cast(int, denominator),
        code,
    )
    return cast(int, numerator), cast(int, denominator)


def _cost(value: object) -> bool:
    if type(value) is not Decimal:
        return False
    try:
        return (
            value.is_finite()
            and value >= 0
            and not value.is_signed()
            and value.as_tuple().exponent == -6
        )
    except DecimalException:
        return False


def build_reviewer_roster_hmac(
    reviewer_subject_hmacs: Mapping[ReviewerRole, str], *, identity_secret: bytes
) -> str:
    """Build the exact frozen role-to-subject roster identity."""

    _require(
        type(identity_secret) is bytes and len(identity_secret) >= 32, 'key_unavailable'
    )
    _require(
        type(reviewer_subject_hmacs) is dict
        and all(type(role) is str for role in reviewer_subject_hmacs)
        and set(reviewer_subject_hmacs) == set(_ROLES),
        'reviewer_roster_invalid',
    )
    subjects = {
        role: _digest(reviewer_subject_hmacs[role], 'reviewer_roster_invalid')
        for role in _ROLES
    }
    _require(len(set(subjects.values())) == 3, 'reviewer_roster_invalid')
    return rag_identity_hmac(
        {
            'adjudicator_c_subject_hmac': subjects['adjudicator_c'],
            'reviewer_a_subject_hmac': subjects['reviewer_a'],
            'reviewer_b_subject_hmac': subjects['reviewer_b'],
            'roster_version': 'rag-live-reviewer-roster:v1',
        },
        secret=identity_secret,
        schema_version='rag-live-reviewer-roster:v1',
        policy_version=_RUBRIC,
    )


def build_review_signature_hmac(
    *,
    approval_hmac: str,
    fixture_manifest_hmac: str,
    case_id_hmac: str,
    block: SanitizedLiveBlockResult,
    reviewer_role: ReviewerRole,
    reviewer_subject_hmac: str,
    label: ReviewLabel,
    identity_secret: bytes,
) -> str:
    """Build the exact role-, case-, block-, and approval-bound review seal."""

    _require(
        type(identity_secret) is bytes
        and len(identity_secret) >= 32
        and type(block) is SanitizedLiveBlockResult
        and type(reviewer_role) is str
        and reviewer_role in _ROLES
        and type(label) is str
        and label in _LABELS,
        'review_labels_invalid',
    )
    return rag_identity_hmac(
        {
            'approval_hmac': _digest(approval_hmac, 'review_labels_invalid'),
            'block_ordinal': block.block_ordinal,
            'block_result_hmac': _digest(
                block.block_result_hmac, 'review_labels_invalid'
            ),
            'case_id_hmac': _digest(case_id_hmac, 'review_labels_invalid'),
            'evidence_projection_hmac': _digest(
                block.evidence_projection_hmac, 'review_labels_invalid'
            ),
            'fixture_manifest_hmac': _digest(
                fixture_manifest_hmac, 'review_labels_invalid'
            ),
            'label': label,
            'reviewer_role': reviewer_role,
            'reviewer_subject_hmac': _digest(
                reviewer_subject_hmac, 'review_labels_invalid'
            ),
            'rubric_version': _RUBRIC,
        },
        secret=identity_secret,
        schema_version='rag-live-review-signature:v1',
        policy_version=_RUBRIC,
    )


class RagReleaseQualityEvaluator:
    """Validate signed human labels and produce one canonical aggregate report."""

    __slots__ = (
        '_identity_secret',
        '_reviewer_subjects',
        '_frozen_reviewer_subjects',
        '_reviewer_roster_hmac',
    )

    def __setattr__(self, name: str, value: object) -> None:
        try:
            object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __init__(
        self,
        *,
        identity_secret: bytes,
        reviewer_subject_hmacs: Mapping[ReviewerRole, str],
    ) -> None:
        _require(
            type(identity_secret) is bytes and len(identity_secret) >= 32,
            'key_unavailable',
        )
        _require(
            type(reviewer_subject_hmacs) is dict
            and all(type(role) is str for role in reviewer_subject_hmacs)
            and set(reviewer_subject_hmacs) == set(_ROLES),
            'reviewer_roster_invalid',
        )
        self._identity_secret = identity_secret
        subjects = {
            role: _digest(reviewer_subject_hmacs.get(role), 'reviewer_roster_invalid')
            for role in _ROLES
        }
        _require(len(set(subjects.values())) == 3, 'reviewer_roster_invalid')
        self._frozen_reviewer_subjects = tuple(
            (role, subjects[role]) for role in _ROLES
        )
        # A mutable compatibility mirror is never authoritative.  Revalidating it
        # against the frozen tuple makes even direct in-process tampering fail
        # closed instead of silently changing the reviewer authority.
        self._reviewer_subjects = dict(self._frozen_reviewer_subjects)
        self._reviewer_roster_hmac = build_reviewer_roster_hmac(
            subjects, identity_secret=identity_secret
        )

    def evaluate(
        self,
        *,
        terminal_cases: Sequence[SanitizedLiveCaseResult],
        signed_labels: Sequence[SignedReviewLabel],
        baseline_metrics: FrozenLegacyBaselineMetrics,
        manifest: FrozenLiveManifestSnapshot,
        corpus: FrozenCorpusSnapshot,
        approval: AuthorizedRagLiveGate,
    ) -> RagQualityReport:
        _require(type(terminal_cases) is tuple, 'case_roster_invalid')
        _require(type(signed_labels) is tuple, 'review_labels_invalid')
        _require(
            type(baseline_metrics) is FrozenLegacyBaselineMetrics,
            'baseline_drift',
        )
        _require(type(manifest) is FrozenLiveManifestSnapshot, 'manifest_invalid')
        _require(type(corpus) is FrozenCorpusSnapshot, 'corpus_drift')
        _require(type(approval) is AuthorizedRagLiveGate, 'approval_drift')
        self._validate_reviewer_roster()
        self._validate_authorities(
            manifest=manifest,
            corpus=corpus,
            approval=approval,
            baseline=baseline_metrics,
        )
        cases = self._validate_cases(
            terminal_cases=terminal_cases,
            manifest=manifest,
            corpus=corpus,
            approval=approval,
        )
        adjudications, entailed, total = self._adjudicate(
            cases=cases,
            labels=cast(tuple[SignedReviewLabel, ...], signed_labels),
            manifest=manifest,
            approval=approval,
        )

        legacy_precision_numerator = sum(
            case.legacy_precision_numerator for case in cases
        )
        legacy_precision_denominator = sum(
            case.legacy_precision_denominator for case in cases
        )
        legacy_recall_numerator = sum(case.legacy_recall_numerator for case in cases)
        legacy_recall_denominator = sum(
            case.legacy_recall_denominator for case in cases
        )
        _require(
            (
                legacy_precision_numerator,
                legacy_precision_denominator,
                legacy_recall_numerator,
                legacy_recall_denominator,
            )
            == (
                baseline_metrics.precision_numerator,
                baseline_metrics.precision_denominator,
                baseline_metrics.recall_numerator,
                baseline_metrics.recall_denominator,
            ),
            'baseline_drift',
        )
        v2_precision_numerator = sum(case.v2_precision_numerator for case in cases)
        v2_precision_denominator = sum(case.v2_precision_denominator for case in cases)
        v2_recall_numerator = sum(case.v2_recall_numerator for case in cases)
        v2_recall_denominator = sum(case.v2_recall_denominator for case in cases)
        _require(
            legacy_precision_denominator > 0
            and legacy_recall_denominator > 0
            and v2_precision_denominator > 0
            and v2_recall_denominator > 0,
            'case_result_invalid',
        )

        hard_negative_cases = [
            case for case in cases if case.case_kind == 'hard_negative'
        ]
        positive_cases = [case for case in cases if case.case_kind == 'positive']
        hard_negative_correct = sum(
            case.hard_negative_correct for case in hard_negative_cases
        )
        positive_answered = sum(case.positive_answered for case in positive_cases)
        required_slots_covered = all(
            case.required_slot_covered for case in positive_cases
        )
        failures: set[QualityFailureReason] = set()
        if hard_negative_correct != len(hard_negative_cases):
            failures.add('hard_negative_accuracy')
        if positive_answered != len(positive_cases) or not required_slots_covered:
            failures.add('positive_answer_coverage')
        if total == 0 or entailed * 100 < total * 95:
            failures.add('faithfulness')
        if (
            v2_precision_numerator * legacy_precision_denominator
            < legacy_precision_numerator * v2_precision_denominator
        ):
            failures.add('retrieval_precision')
        if (
            v2_recall_numerator * legacy_recall_denominator
            < legacy_recall_numerator * v2_recall_denominator
        ):
            failures.add('retrieval_recall')

        failure_reasons = cast(
            tuple[QualityFailureReason, ...], tuple(sorted(failures))
        )
        gate_outcome: Literal['green', 'quality_gate_failed'] = (
            'green' if not failure_reasons else 'quality_gate_failed'
        )
        payload = {
            'approval_hmac': approval.approval_hmac,
            'approval_id_hmac': approval.approval_id_hmac,
            'approved_corpus_snapshot_hmac': approval.corpus.corpus_snapshot_hmac,
            'baseline_hmac': baseline_metrics.baseline_hmac,
            'case_adjudications': [
                {
                    'block_labels': [
                        {
                            'adjudicator_label': block.adjudicator_label,
                            'adjudicator_signature_hmac': block.adjudicator_signature_hmac,
                            'block_ordinal': block.block_ordinal,
                            'block_result_hmac': block.block_result_hmac,
                            'decided_label': block.decided_label,
                            'evidence_projection_hmac': block.evidence_projection_hmac,
                            'reviewer_a_label': block.reviewer_a_label,
                            'reviewer_a_signature_hmac': block.reviewer_a_signature_hmac,
                            'reviewer_b_label': block.reviewer_b_label,
                            'reviewer_b_signature_hmac': block.reviewer_b_signature_hmac,
                        }
                        for block in case.block_labels
                    ],
                    'case_id_hmac': case.case_id_hmac,
                    'case_projection_hmac': case.case_projection_hmac,
                    'case_kind': case.case_kind,
                    'required_slot_covered': case.required_slot_covered,
                }
                for case in adjudications
            ],
            'case_count': 30,
            'current_corpus_snapshot_hmac': corpus.corpus_snapshot_hmac,
            'failure_reasons': list(failure_reasons),
            'faithfulness_entailed_blocks': entailed,
            'faithfulness_total_blocks': total,
            'gate_outcome': gate_outcome,
            'hard_negative_case_count': len(hard_negative_cases),
            'hard_negative_correct_count': hard_negative_correct,
            'legacy_precision_denominator': legacy_precision_denominator,
            'legacy_precision_numerator': legacy_precision_numerator,
            'legacy_recall_denominator': legacy_recall_denominator,
            'legacy_recall_numerator': legacy_recall_numerator,
            'manifest_hmac': manifest.manifest_hmac,
            'positive_answered_count': positive_answered,
            'positive_case_count': len(positive_cases),
            'reviewer_roster_hmac': approval.reviewer_roster_hmac,
            'rubric_version': _RUBRIC,
            'v2_precision_denominator': v2_precision_denominator,
            'v2_precision_numerator': v2_precision_numerator,
            'v2_recall_denominator': v2_recall_denominator,
            'v2_recall_numerator': v2_recall_numerator,
        }
        canonical = canonical_json_bytes(payload)
        report_hmac = rag_identity_hmac(
            payload,
            secret=self._identity_secret,
            schema_version='rag-live-quality-report:v1',
            policy_version=_POLICY,
        )
        return RagQualityReport(
            approval_hmac=approval.approval_hmac,
            approval_id_hmac=approval.approval_id_hmac,
            approved_corpus_snapshot_hmac=approval.corpus.corpus_snapshot_hmac,
            current_corpus_snapshot_hmac=corpus.corpus_snapshot_hmac,
            baseline_hmac=baseline_metrics.baseline_hmac,
            manifest_hmac=manifest.manifest_hmac,
            reviewer_roster_hmac=approval.reviewer_roster_hmac,
            rubric_version=_RUBRIC,
            case_adjudications=adjudications,
            case_count=30,
            failure_reasons=failure_reasons,
            gate_outcome=gate_outcome,
            faithfulness_entailed_blocks=entailed,
            faithfulness_total_blocks=total,
            hard_negative_case_count=len(hard_negative_cases),
            hard_negative_correct_count=hard_negative_correct,
            positive_case_count=len(positive_cases),
            positive_answered_count=positive_answered,
            legacy_precision_numerator=legacy_precision_numerator,
            legacy_precision_denominator=legacy_precision_denominator,
            legacy_recall_numerator=legacy_recall_numerator,
            legacy_recall_denominator=legacy_recall_denominator,
            v2_precision_numerator=v2_precision_numerator,
            v2_precision_denominator=v2_precision_denominator,
            v2_recall_numerator=v2_recall_numerator,
            v2_recall_denominator=v2_recall_denominator,
            payload_canonical_bytes=canonical,
            quality_report_hmac=report_hmac,
        )

    def _frozen_subject_map(self) -> dict[ReviewerRole, str]:
        try:
            frozen = object.__getattribute__(self, '_frozen_reviewer_subjects')
        except AttributeError:
            raise RagReleaseQualityError('reviewer_roster_invalid') from None
        _require(
            type(frozen) is tuple and len(frozen) == len(_ROLES),
            'reviewer_roster_invalid',
        )
        subjects: dict[ReviewerRole, str] = {}
        for expected_role, item in zip(_ROLES, frozen, strict=True):
            _require(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and item[0] == expected_role,
                'reviewer_roster_invalid',
            )
            subjects[expected_role] = _digest(item[1], 'reviewer_roster_invalid')
        _require(len(set(subjects.values())) == 3, 'reviewer_roster_invalid')
        return subjects

    def _validate_reviewer_roster(self) -> None:
        frozen = self._frozen_subject_map()
        try:
            identity_secret = object.__getattribute__(self, '_identity_secret')
            current = object.__getattribute__(self, '_reviewer_subjects')
            cached_hmac = object.__getattribute__(self, '_reviewer_roster_hmac')
        except AttributeError:
            raise RagReleaseQualityError('reviewer_roster_invalid') from None
        _require(
            type(identity_secret) is bytes
            and len(identity_secret) >= 32
            and type(current) is dict
            and all(type(role) is str for role in current)
            and set(current) == set(_ROLES)
            and all(type(current[role]) is str for role in _ROLES),
            'reviewer_roster_invalid',
        )
        cached_hmac = _digest(cached_hmac, 'reviewer_roster_invalid')
        frozen_hmac = build_reviewer_roster_hmac(
            frozen, identity_secret=identity_secret
        )
        current_hmac = build_reviewer_roster_hmac(
            current, identity_secret=identity_secret
        )
        _require(
            hmac.compare_digest(frozen_hmac, cached_hmac)
            and hmac.compare_digest(current_hmac, frozen_hmac),
            'reviewer_roster_invalid',
        )

    def _validate_authorities(self, *, manifest, corpus, approval, baseline) -> None:
        corpus_members = self._validate_corpus(corpus)
        self._validate_manifest(manifest, corpus_members)
        _require(
            type(approval.live_gate_contract_version) is str
            and approval.live_gate_contract_version == _POLICY
            and type(approval.authorization_state) is str
            and approval.authorization_state == 'unused'
            and type(approval.ledger_epoch) is int
            and approval.ledger_epoch > 0
            and type(approval.approval_base_generation) is int
            and approval.approval_base_generation >= 0
            and _exact_frozen_equal(approval.manifest, manifest)
            and _exact_frozen_equal(approval.corpus, corpus)
            and _exact_frozen_equal(approval.limits, manifest.limits),
            'approval_drift',
        )
        for value in (
            approval.approval_hmac,
            approval.approval_id_hmac,
            approval.approval_base_release_marker_file_digest,
            approval.baseline_hmac,
            approval.approved_provider_safety_snapshot_hmac,
            approval.reviewer_roster_hmac,
            approval.validation_database_identity_hmac,
            approval.designated_environment_id_hmac,
            approval.designated_host_id_hmac,
            approval.implementation_plan_reference_hmac,
            manifest.fixture_manifest_hmac,
            manifest.manifest_hmac,
            corpus.corpus_snapshot_hmac,
        ):
            _digest(value, 'approval_drift')
        _require(
            approval.reviewer_roster_hmac == self._reviewer_roster_hmac,
            'reviewer_roster_invalid',
        )
        self._validate_baseline(baseline, manifest, corpus, approval)

    def _validate_corpus(self, corpus: FrozenCorpusSnapshot):
        try:
            payload = _corpus_payload(corpus)
        except (LiveGatePreviewError, AttributeError, TypeError, ValueError):
            raise RagReleaseQualityError('corpus_drift') from None
        snapshot_hmac = _digest(corpus.corpus_snapshot_hmac, 'corpus_drift')
        expected_hmac = rag_identity_hmac(
            payload,
            secret=self._identity_secret,
            schema_version='rag-live-corpus-snapshot:v1',
            policy_version=_POLICY,
        )
        _require(hmac.compare_digest(snapshot_hmac, expected_hmac), 'corpus_drift')
        return {member.serving_identity_hmac: member for member in corpus.members}

    def _validate_manifest(self, manifest, corpus_members) -> None:
        _require(
            type(manifest.live_gate_contract_version) is str
            and manifest.live_gate_contract_version == _POLICY
            and type(manifest.fixture_manifest_version) is str
            and manifest.fixture_manifest_version == 'rag-live-quality-30:v1'
            and type(manifest.fixture_manifest_path) is str
            and manifest.fixture_manifest_path
            == 'backend/tests/fixtures/rag_v2_live_gate_30.json'
            and type(manifest.rubric_version) is str
            and manifest.rubric_version == _RUBRIC
            and type(manifest.fixture_manifest_hmac) is str
            and type(manifest.manifest_hmac) is str
            and manifest.fixture_manifest_hmac == manifest.manifest_hmac
            and all(
                type(getattr(manifest, name)) is int
                for name in (
                    'ask_keyword_count',
                    'ask_pgvector_count',
                    'assistant_keyword_count',
                    'assistant_pgvector_count',
                    'keyword_with_prior_context_count',
                    'keyword_without_prior_context_count',
                    'pgvector_with_prior_context_count',
                    'pgvector_without_prior_context_count',
                )
            )
            and (
                manifest.ask_keyword_count,
                manifest.ask_pgvector_count,
                manifest.assistant_keyword_count,
                manifest.assistant_pgvector_count,
                manifest.keyword_with_prior_context_count,
                manifest.keyword_without_prior_context_count,
                manifest.pgvector_with_prior_context_count,
                manifest.pgvector_without_prior_context_count,
            )
            == (10, 5, 10, 5, 5, 5, 3, 2)
            and type(manifest.limits) is LiveGateLimits
            and manifest.limits.case_claims == 30
            and manifest.limits.answer_generation_dispatches == 30
            and manifest.limits.query_embedding_dispatches == 10
            and manifest.limits.total_dispatches == 40
            and manifest.limits.case_max_cost_usd == Decimal('0.012000')
            and manifest.limits.total_max_cost_usd == Decimal('0.360000')
            and type(manifest.cases) is tuple
            and len(manifest.cases) == 30
            and type(manifest.clean_git_commit) is str
            and re.fullmatch(r'[0-9a-f]{40}', manifest.clean_git_commit) is not None,
            'manifest_invalid',
        )
        _require(
            type(manifest.limits.case_claims) is int
            and type(manifest.limits.answer_generation_dispatches) is int
            and type(manifest.limits.query_embedding_dispatches) is int
            and type(manifest.limits.total_dispatches) is int
            and type(manifest.limits.case_max_cost_usd) is Decimal
            and type(manifest.limits.total_max_cost_usd) is Decimal
            and _cost(manifest.limits.case_max_cost_usd)
            and _cost(manifest.limits.total_max_cost_usd),
            'manifest_invalid',
        )
        _digest(manifest.fixture_manifest_sha256, 'manifest_invalid')
        seen: set[str] = set()
        distribution = {
            ('ask', 'keyword'): 0,
            ('ask', 'pgvector'): 0,
            ('assistant', 'keyword'): 0,
            ('assistant', 'pgvector'): 0,
        }
        context = {
            ('keyword', True): 0,
            ('keyword', False): 0,
            ('pgvector', True): 0,
            ('pgvector', False): 0,
        }
        kinds: set[str] = set()
        for ordinal, case in enumerate(manifest.cases):
            _require(
                type(case) is FrozenLiveManifestCase
                and type(case.ordinal) is int
                and case.ordinal == ordinal
                and type(case.case_kind) is str
                and case.case_kind in {'positive', 'hard_negative'}
                and type(case.surface) is str
                and case.surface in {'ask', 'assistant'}
                and type(case.configured_backend) is str
                and case.configured_backend in {'keyword', 'pgvector'}
                and type(case.question_fixture_id) is str
                and re.fullmatch(r'[a-z][a-z0-9-]{0,63}', case.question_fixture_id)
                is not None
                and type(case.security_scope_fixture_id) is str
                and re.fullmatch(
                    r'[a-z][a-z0-9-]{0,63}', case.security_scope_fixture_id
                )
                is not None
                and (
                    case.prior_context_fixture_id is None
                    or (
                        type(case.prior_context_fixture_id) is str
                        and re.fullmatch(
                            r'[a-z][a-z0-9-]{0,63}',
                            case.prior_context_fixture_id,
                        )
                        is not None
                        and case.surface == 'assistant'
                    )
                )
                and type(case.expected_no_answer) is bool
                and case.expected_no_answer == (case.case_kind == 'hard_negative')
                and type(case.allowed_support_modes) is tuple
                and bool(case.allowed_support_modes)
                and all(type(mode) is str for mode in case.allowed_support_modes)
                and type(case.allowed_slot_ids) is tuple
                and bool(case.allowed_slot_ids)
                and all(
                    type(slot) is str and re.fullmatch(r'E[1-8]', slot)
                    for slot in case.allowed_slot_ids
                )
                and type(case.relevant_serving_identity_hmacs) is tuple
                and type(case.required_serving_identity_hmacs) is tuple
                and all(
                    type(identity) is str
                    for identity in case.relevant_serving_identity_hmacs
                    + case.required_serving_identity_hmacs
                )
                and (
                    case.case_kind == 'hard_negative'
                    or (
                        bool(case.relevant_serving_identity_hmacs)
                        and bool(case.required_serving_identity_hmacs)
                    )
                ),
                'manifest_invalid',
            )
            _require(
                len(set(case.allowed_support_modes)) == len(case.allowed_support_modes)
                and set(case.allowed_support_modes)
                <= {'trusted_fact', 'source_observation'}
                and len(set(case.allowed_slot_ids)) == len(case.allowed_slot_ids)
                and len(set(case.relevant_serving_identity_hmacs))
                == len(case.relevant_serving_identity_hmacs)
                and len(set(case.required_serving_identity_hmacs))
                == len(case.required_serving_identity_hmacs)
                and set(case.required_serving_identity_hmacs)
                <= set(case.relevant_serving_identity_hmacs)
                and set(case.relevant_serving_identity_hmacs) <= corpus_members.keys(),
                'manifest_invalid',
            )
            _digest(case.case_id_hmac, 'manifest_invalid')
            _digest(case.query_bytes_hmac, 'manifest_invalid')
            _require(case.case_id_hmac not in seen, 'manifest_invalid')
            seen.add(case.case_id_hmac)
            for identity in (
                case.relevant_serving_identity_hmacs
                + case.required_serving_identity_hmacs
            ):
                _digest(identity, 'manifest_invalid')
            _require(
                all(
                    corpus_members[identity].support_mode in case.allowed_support_modes
                    and (
                        case.configured_backend != 'pgvector'
                        or corpus_members[identity].vector_index_state_hmac is not None
                    )
                    for identity in case.relevant_serving_identity_hmacs
                ),
                'manifest_invalid',
            )
            _require(
                type(case.query_embedding_required) is bool
                and case.query_embedding_required
                == (case.configured_backend == 'pgvector')
                and case.answer_generation_required is True
                and _cost(case.query_embedding_reserved_cost_usd)
                and _cost(case.answer_generation_reserved_cost_usd)
                and _cost(case.case_total_reserved_cost_usd)
                and case.query_embedding_reserved_cost_usd
                == (
                    Decimal('0.000160')
                    if case.configured_backend == 'pgvector'
                    else Decimal('0.000000')
                )
                and case.answer_generation_reserved_cost_usd >= Decimal('0.009804')
                and case.query_embedding_reserved_cost_usd
                + case.answer_generation_reserved_cost_usd
                == case.case_total_reserved_cost_usd
                <= Decimal('0.012000'),
                'manifest_invalid',
            )
            distribution[(case.surface, case.configured_backend)] += 1
            if case.surface == 'assistant':
                context[
                    (case.configured_backend, case.prior_context_fixture_id is not None)
                ] += 1
            kinds.add(case.case_kind)
        _require(
            distribution
            == {
                ('ask', 'keyword'): 10,
                ('ask', 'pgvector'): 5,
                ('assistant', 'keyword'): 10,
                ('assistant', 'pgvector'): 5,
            }
            and context
            == {
                ('keyword', True): 5,
                ('keyword', False): 5,
                ('pgvector', True): 3,
                ('pgvector', False): 2,
            }
            and kinds == {'positive', 'hard_negative'}
            and sum(case.case_total_reserved_cost_usd for case in manifest.cases)
            == Decimal('0.360000'),
            'manifest_invalid',
        )
        self._validate_executable_manifest(manifest)

    def _validate_executable_manifest(self, manifest) -> None:
        executable = manifest.executable
        try:
            _manifest_payload(executable)
        except (RagReleaseLedgerError, AttributeError, TypeError, ValueError):
            raise RagReleaseQualityError('manifest_invalid') from None
        _require(
            type(executable) is FrozenCaseClaimManifest
            and type(executable.cases) is tuple
            and len(executable.cases) == 30
            and type(executable.source_manifest_hmac) is str
            and executable.source_manifest_hmac == manifest.fixture_manifest_hmac,
            'manifest_invalid',
        )
        for annotation, resolved in zip(manifest.cases, executable.cases, strict=True):
            _require(
                type(resolved.ordinal) is int
                and resolved.ordinal == annotation.ordinal
                and type(resolved.case_id_hmac) is str
                and resolved.case_id_hmac == annotation.case_id_hmac
                and type(resolved.surface) is str
                and resolved.surface == annotation.surface
                and type(resolved.configured_backend) is str
                and resolved.configured_backend == annotation.configured_backend
                and type(resolved.current_text_hmac) is str
                and type(resolved.retrieval_query_hmac) is str
                and resolved.retrieval_query_hmac == annotation.query_bytes_hmac
                and type(resolved.security_scope_fingerprint) is str
                and type(resolved.components) is tuple
                and len(resolved.components) == 2,
                'manifest_invalid',
            )
            for value in (
                resolved.case_id_hmac,
                resolved.current_text_hmac,
                resolved.retrieval_query_hmac,
                resolved.security_scope_fingerprint,
            ):
                _digest(value, 'manifest_invalid')
            query_component, answer_component = resolved.components
            _require(
                _cost(query_component.reserved_cost_usd)
                and _cost(answer_component.reserved_cost_usd)
                and query_component.reserved_cost_usd
                == annotation.query_embedding_reserved_cost_usd
                and answer_component.reserved_cost_usd
                == annotation.answer_generation_reserved_cost_usd
                and query_component.reserved_cost_usd
                + answer_component.reserved_cost_usd
                == annotation.case_total_reserved_cost_usd,
                'manifest_invalid',
            )

    def _validate_baseline(self, baseline, manifest, corpus, approval) -> None:
        for value in (
            baseline.baseline_hmac,
            baseline.baseline_policy_snapshot_hmac,
            baseline.evaluator_code_hmac,
            baseline.retriever_source_bundle_hmac,
            baseline.fixture_manifest_hmac,
            baseline.corpus_snapshot_hmac,
        ):
            _digest(value, 'baseline_drift')
        _require(
            baseline.baseline_hmac == approval.baseline_hmac
            and baseline.fixture_manifest_hmac == manifest.fixture_manifest_hmac
            and baseline.corpus_snapshot_hmac == corpus.corpus_snapshot_hmac
            and baseline.evaluator_source_commit == manifest.clean_git_commit,
            'baseline_drift',
        )
        policy = {
            'annotation_schema_version': 'rag-live-relevance-annotation:v1',
            'evaluator_version': 'rag-live-retrieval-evaluator:v1',
            'keyword_scorer_version': 'rag-keyword-lexical-compat:v1',
            'pgvector_distance_policy_version': 'rag-pgvector-cosine-distance:v1',
            'retrieval_policy_version': 'rag-retrieval-policy:v2.0',
            'rubric_version': _RUBRIC,
        }
        expected_policy = rag_identity_hmac(
            policy,
            secret=self._identity_secret,
            schema_version='rag-live-baseline-policy-snapshot:v1',
            policy_version=_POLICY,
        )
        _require(
            hmac.compare_digest(
                baseline.baseline_policy_snapshot_hmac, expected_policy
            ),
            'baseline_drift',
        )
        definition = {
            'annotation_schema_version': 'rag-live-relevance-annotation:v1',
            'cases': [
                {
                    'case_id_hmac': case.case_id_hmac,
                    'configured_backend': case.configured_backend,
                    'legacy_retriever_version': (
                        f'rag-v1-{case.configured_backend}-retriever:v1'
                    ),
                    'query_bytes_hmac': case.query_bytes_hmac,
                    'relevant_serving_identity_hmacs': list(
                        case.relevant_serving_identity_hmacs
                    ),
                    'v2_retriever_version': (
                        f'rag-v2-{case.configured_backend}-retriever:v1'
                    ),
                }
                for case in manifest.cases
            ],
            'corpus_snapshot_hmac': corpus.corpus_snapshot_hmac,
            'evaluator_code_hmac': baseline.evaluator_code_hmac,
            'evaluator_path': 'backend/app/rag/release_quality.py',
            'evaluator_source_commit': baseline.evaluator_source_commit,
            'evaluator_version': 'rag-live-retrieval-evaluator:v1',
            'fixture_manifest_hmac': manifest.fixture_manifest_hmac,
            'keyword_scorer_version': 'rag-keyword-lexical-compat:v1',
            'pgvector_distance_policy_version': 'rag-pgvector-cosine-distance:v1',
            'policy_snapshot_hmac': baseline.baseline_policy_snapshot_hmac,
            'retriever_source_bundle_hmac': baseline.retriever_source_bundle_hmac,
        }
        expected_baseline = rag_identity_hmac(
            definition,
            secret=self._identity_secret,
            schema_version='rag-live-baseline-definition:v1',
            policy_version=_POLICY,
        )
        _require(
            hmac.compare_digest(baseline.baseline_hmac, expected_baseline),
            'baseline_drift',
        )
        for pair in (
            (baseline.precision_numerator, baseline.precision_denominator),
            (baseline.recall_numerator, baseline.recall_denominator),
        ):
            _ratio(*pair, code='baseline_drift')
        _require(
            baseline.precision_denominator > 0 and baseline.recall_denominator > 0,
            'baseline_drift',
        )

    def _validate_cases(self, *, terminal_cases, manifest, corpus, approval):
        _require(
            type(terminal_cases) is tuple and len(terminal_cases) == 30,
            'case_roster_invalid',
        )
        validated: list[SanitizedLiveCaseResult] = []
        seen: set[str] = set()
        for ordinal, (result, case) in enumerate(
            zip(terminal_cases, manifest.cases, strict=True)
        ):
            _require(
                type(result) is SanitizedLiveCaseResult
                and type(result.ordinal) is int
                and result.ordinal == ordinal
                and type(result.case_id_hmac) is str,
                'case_roster_invalid',
            )
            _digest(result.case_id_hmac, 'case_roster_invalid')
            _require(
                result.case_id_hmac == case.case_id_hmac
                and result.case_id_hmac not in seen,
                'case_roster_invalid',
            )
            seen.add(result.case_id_hmac)
            _require(
                type(result.case_kind) is str
                and type(result.surface) is str
                and type(result.configured_backend) is str
                and type(result.state) is str
                and type(result.outcome) is str
                and type(result.runtime_agent_run_id_hmac) is str
                and type(result.case_projection_hmac) is str
                and type(result.current_corpus_snapshot_hmac) is str
                and type(result.provider_safety_snapshot_hmac) is str
                and type(result.query_embedding_dispatch_count) is int
                and type(result.answer_generation_dispatch_count) is int
                and type(result.expected_no_answer) is bool
                and type(result.hard_negative_correct) is bool
                and type(result.positive_answered) is bool
                and type(result.required_slot_covered) is bool
                and type(result.blocks) is tuple
                and _cost(result.reserved_cost_usd)
                and _cost(result.charged_cost_usd),
                'case_result_invalid',
            )
            for value in (
                result.runtime_agent_run_id_hmac,
                result.case_projection_hmac,
                result.current_corpus_snapshot_hmac,
                result.provider_safety_snapshot_hmac,
            ):
                _digest(value, 'case_result_invalid')
            for pair in (
                (
                    result.legacy_precision_numerator,
                    result.legacy_precision_denominator,
                ),
                (result.legacy_recall_numerator, result.legacy_recall_denominator),
                (result.v2_precision_numerator, result.v2_precision_denominator),
                (result.v2_recall_numerator, result.v2_recall_denominator),
            ):
                _ratio(*pair)
            _require(
                result.case_kind == case.case_kind
                and result.surface == case.surface
                and result.configured_backend == case.configured_backend
                and result.expected_no_answer == case.expected_no_answer
                and result.current_corpus_snapshot_hmac
                == corpus.corpus_snapshot_hmac
                == approval.corpus.corpus_snapshot_hmac
                and result.provider_safety_snapshot_hmac
                == approval.approved_provider_safety_snapshot_hmac,
                'case_identity_drift',
            )
            _require(
                result.state == 'complete'
                and result.outcome in _QUALITY_OUTCOMES
                and result.reserved_cost_usd == case.case_total_reserved_cost_usd
                and result.charged_cost_usd <= result.reserved_cost_usd
                and result.query_embedding_dispatch_count
                == int(case.query_embedding_required)
                and result.answer_generation_dispatch_count == 1,
                'case_result_invalid',
            )
            if case.case_kind == 'positive':
                _require(
                    result.hard_negative_correct is False
                    and result.positive_answered == bool(result.blocks)
                    and (
                        (bool(result.blocks) and result.outcome == 'supported')
                        or (not result.blocks and result.outcome != 'supported')
                    ),
                    'case_result_invalid',
                )
            else:
                _require(
                    result.positive_answered is False
                    and result.required_slot_covered is False
                    and result.hard_negative_correct == (not result.blocks),
                    'case_result_invalid',
                )
                _require(
                    (bool(result.blocks) and result.outcome == 'supported')
                    or (not result.blocks and result.outcome != 'supported'),
                    'case_result_invalid',
                )
            for block_ordinal, block in enumerate(result.blocks):
                _require(
                    type(block) is SanitizedLiveBlockResult
                    and type(block.block_ordinal) is int
                    and block.block_ordinal == block_ordinal,
                    'case_result_invalid',
                )
                _digest(block.block_result_hmac, 'case_result_invalid')
                _digest(block.evidence_projection_hmac, 'case_result_invalid')
            validated.append(result)
        return tuple(validated)

    def _adjudicate(self, *, cases, labels, manifest, approval):
        label_index = 0
        entailed = 0
        total = 0
        case_adjudications = []
        for case in cases:
            block_adjudications = []
            for block in case.blocks:
                reviewer_a = self._take_label(
                    labels,
                    label_index,
                    role='reviewer_a',
                    case=case,
                    block=block,
                    manifest=manifest,
                    approval=approval,
                )
                label_index += 1
                reviewer_b = self._take_label(
                    labels,
                    label_index,
                    role='reviewer_b',
                    case=case,
                    block=block,
                    manifest=manifest,
                    approval=approval,
                )
                label_index += 1
                adjudicator = None
                if reviewer_a.label != reviewer_b.label:
                    adjudicator = self._take_label(
                        labels,
                        label_index,
                        role='adjudicator_c',
                        case=case,
                        block=block,
                        manifest=manifest,
                        approval=approval,
                    )
                    label_index += 1
                    _require(
                        adjudicator.label in {reviewer_a.label, reviewer_b.label},
                        'review_labels_invalid',
                    )
                    decided = adjudicator.label
                else:
                    decided = reviewer_a.label
                total += 1
                entailed += decided == 'entailed'
                block_adjudications.append(
                    QualityBlockAdjudication(
                        block_ordinal=block.block_ordinal,
                        block_result_hmac=block.block_result_hmac,
                        evidence_projection_hmac=block.evidence_projection_hmac,
                        reviewer_a_label=reviewer_a.label,
                        reviewer_a_signature_hmac=reviewer_a.signature_hmac,
                        reviewer_b_label=reviewer_b.label,
                        reviewer_b_signature_hmac=reviewer_b.signature_hmac,
                        adjudicator_label=None
                        if adjudicator is None
                        else adjudicator.label,
                        adjudicator_signature_hmac=None
                        if adjudicator is None
                        else adjudicator.signature_hmac,
                        decided_label=decided,
                    )
                )
            case_adjudications.append(
                QualityCaseAdjudication(
                    case_id_hmac=case.case_id_hmac,
                    case_projection_hmac=cast(str, case.case_projection_hmac),
                    case_kind=case.case_kind,
                    required_slot_covered=case.required_slot_covered,
                    block_labels=tuple(block_adjudications),
                )
            )
        _require(label_index == len(labels), 'review_labels_invalid')
        return tuple(case_adjudications), entailed, total

    def _take_label(
        self, labels, index, *, role, case, block, manifest, approval
    ) -> SignedReviewLabel:
        _require(index < len(labels), 'review_labels_invalid')
        signed = labels[index]
        subjects = self._frozen_subject_map()
        _require(
            type(signed) is SignedReviewLabel
            and type(signed.reviewer_role) is str
            and signed.reviewer_role == role
            and type(signed.reviewer_subject_hmac) is str
            and signed.reviewer_subject_hmac == subjects[role]
            and type(signed.case_id_hmac) is str
            and signed.case_id_hmac == case.case_id_hmac
            and type(signed.block_ordinal) is int
            and signed.block_ordinal == block.block_ordinal
            and type(signed.label) is str
            and signed.label in _LABELS,
            'review_labels_invalid',
        )
        expected = build_review_signature_hmac(
            approval_hmac=approval.approval_hmac,
            fixture_manifest_hmac=manifest.fixture_manifest_hmac,
            case_id_hmac=case.case_id_hmac,
            block=block,
            reviewer_role=role,
            reviewer_subject_hmac=subjects[role],
            label=signed.label,
            identity_secret=self._identity_secret,
        )
        _require(
            hmac.compare_digest(
                _digest(signed.signature_hmac, 'review_labels_invalid'), expected
            ),
            'review_labels_invalid',
        )
        return signed


__all__ = [
    'EphemeralReviewBlock',
    'FrozenLegacyBaselineMetrics',
    'LiveCaseKind',
    'LiveCaseOutcome',
    'LiveCaseSurface',
    'QualityBlockAdjudication',
    'QualityCaseAdjudication',
    'QualityFailureReason',
    'RagQualityReport',
    'RagReleaseQualityError',
    'RagReleaseQualityEvaluator',
    'ReviewLabel',
    'SanitizedLiveBlockResult',
    'SanitizedLiveCaseResult',
    'SignedReviewLabel',
    'build_review_signature_hmac',
    'build_reviewer_roster_hmac',
]
