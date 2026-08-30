from __future__ import annotations

import json
import unicodedata
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, ClassVar, Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21 = 'company-memory-review-v2.1-auto-review'
AUTO_REVIEW_POLICY_VERSION = 'auto-review-policy:v1'
AUTO_REVIEW_VALIDATOR_PROMPT_VERSION = 'auto-review-validation:v2'
AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION = 'candidate-validation-batch:v1'
AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION = 'trusted-claim-normalization:v1'
AUTO_REVIEW_COST_POLICY_VERSION = 'auto-review-cost:v1'
AUTO_REVIEW_TOKEN_ESTIMATOR_VERSION = 'openai-o200k-chat:v1'
AUTO_REVIEW_EXTRACTION_TOKEN_ESTIMATOR_VERSION = 'openai-o200k-extraction:v1'
AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION = 'auto-review-extraction-cost:v1'
AUTO_REVIEW_VALIDATOR_PROVIDER = 'openai'
AUTO_REVIEW_VALIDATOR_MODEL = 'gpt-5.6-terra'
AUTO_REVIEW_REASONING_EFFORT = 'medium'
AUTO_REVIEW_EXTRACTION_PROVIDER = 'openai'
AUTO_REVIEW_EXTRACTION_MODEL = 'gpt-5.4-mini-2026-03-17'
AUTO_REVIEW_EXTRACTION_REASONING_EFFORT = 'none'
AUTO_REVIEW_EXTRACTION_ROUTE_VERSION = 'auto-review-extraction-route:v1'
AUTO_REVIEW_MIN_ENTAILMENT = Decimal('0.9800')
AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH = 4
AUTO_REVIEW_MAX_EVIDENCE_SLOTS = 12
AUTO_REVIEW_MAX_CLAIM_CHARS = 2_000
AUTO_REVIEW_MAX_INPUT_CHARS = 12_000
AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT = 2_048
AUTO_REVIEW_TOKENIZER_ENCODING = 'o200k_base'

AutoReviewConfiguredMode = Literal['disabled', 'shadow', 'enforce']
AutoReviewStoredMode = Literal['shadow', 'enforce']
AutoReviewEffectiveMode = Literal['disabled', 'shadow', 'enforce']
AutoReviewEligibilityDecision = Literal[
    'eligible', 'human_review', 'needs_more_evidence', 'reuse_trusted'
]
AutoReviewPolicyDecision = Literal[
    'auto_approve', 'human_review', 'needs_more_evidence', 'reuse_trusted'
]
AutoReviewPolicyReasonCode = Literal[
    'eligible', 'item_type_not_allowed', 'permission_not_allowed',
    'evidence_binding_missing', 'evidence_binding_mismatch',
    'evidence_version_changed', 'evidence_missing', 'sensitive_input_detected',
    'uncertainty_present', 'high_risk_language', 'project_selection_required',
    'payload_not_supported', 'generation_identity_missing',
    'validator_identity_collision', 'registry_unavailable', 'budget_exceeded',
    'trusted_exact_match', 'trusted_visible_conflict', 'trusted_hidden_collision',
    'trusted_lookup_unavailable', 'validator_unavailable', 'validator_malformed',
    'validator_not_supported', 'validator_not_direct_fact',
    'validator_score_below_threshold', 'validator_uncertain',
    'validator_conflict', 'post_validation_drift', 'enforce_not_selected',
    'rollout_gate_closed', 'breaker_open',
]
CandidateSlotId = Annotated[str, Field(pattern=r'^C0[1-4]$')]
EvidenceSlotId = Annotated[str, Field(pattern=r'^E(?:0[1-9]|1[0-2])$')]
ExtractionEvidenceSlotId = Annotated[str, Field(pattern=r'^S(?:0[1-9]|1[0-2])$')]
NoCandidateReason = Literal[
    'no_relevant_evidence',
    'insufficient_direct_evidence',
    'non_business_evidence',
    'conflicting_evidence',
]


class FieldValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    field_key: Literal['title', 'result_summary', 'reason']
    verdict: Literal['supported', 'partially_supported', 'unsupported', 'contradicted']
    claim_scope: Literal['direct_fact', 'inference', 'decision', 'todo', 'unknown']
    entailment_score: Decimal = Field(
        ge=Decimal('0'), le=Decimal('1'), max_digits=5, decimal_places=4
    )
    evidence_slot_ids: list[EvidenceSlotId] = Field(max_length=12)


class CandidateValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    candidate_slot_id: CandidateSlotId
    claim_results: list[FieldValidationResult] = Field(min_length=2, max_length=2)
    uncertainty_codes: list[
        Literal['ambiguous_subject', 'ambiguous_time', 'conditional_language', 'partial_evidence', 'proposal_language', 'unknown']
    ] = Field(max_length=4)
    conflict_codes: list[
        Literal['source_contradiction', 'trusted_claim_conflict', 'unknown']
    ] = Field(max_length=4)


class CandidateValidationBatchResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    results: list[CandidateValidationResult] = Field(min_length=1, max_length=4)


class FieldEvidenceBinding(BaseModel):
    model_config = ConfigDict(extra='forbid')

    field_key: Literal[
        'title', 'summary', 'result_summary', 'reason', 'decision_summary',
        'priority', 'priority_reason', 'task_summary', 'assignee', 'due_date',
        'evidence_reason', 'source_type', 'project_tag',
    ]
    evidence_slot_id: ExtractionEvidenceSlotId


class _ExtractionCandidateBase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    item_type: str
    title: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=800)
    confidence_score: Decimal = Field(
        ge=Decimal('0'), le=Decimal('1'), max_digits=5, decimal_places=4
    )
    uncertainty_reason: str | None = Field(default=None, min_length=1, max_length=400)
    field_evidence_bindings: list[FieldEvidenceBinding] = Field(
        min_length=1, max_length=10
    )
    _required_evidence_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode='after')
    def _canonicalize_and_validate_evidence(self) -> _ExtractionCandidateBase:
        score = self.confidence_score
        if score == 0:
            score = Decimal('0')
        self.confidence_score = score.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        fields = [binding.field_key for binding in self.field_evidence_bindings]
        if len(fields) != len(set(fields)):
            raise ValueError('each candidate field requires exactly one evidence binding')
        present = set(self._required_evidence_fields)
        for field_name in self.model_fields_set:
            if field_name in {'task_summary', 'assignee', 'due_date', 'evidence_reason', 'source_type', 'project_tag'} and getattr(self, field_name) is not None:
                present.add(field_name)
        if set(fields) != present:
            raise ValueError('candidate evidence bindings do not match present fields')
        return self

    @field_serializer('confidence_score', when_used='json')
    def _serialize_confidence(self, value: Decimal) -> str:
        normalized = Decimal('0') if value == 0 else value
        return f'{normalized.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP):.4f}'


class TimelineCandidate(_ExtractionCandidateBase):
    item_type: Literal['timeline_event']
    result_summary: str = Field(min_length=1, max_length=800)
    _required_evidence_fields: ClassVar[frozenset[str]] = frozenset({'title', 'summary', 'result_summary'})


class HistoryCandidate(_ExtractionCandidateBase):
    item_type: Literal['history_event']
    reason: str = Field(min_length=1, max_length=800)
    _required_evidence_fields: ClassVar[frozenset[str]] = frozenset({'title', 'summary', 'reason'})


class DecisionRecordCandidate(_ExtractionCandidateBase):
    item_type: Literal['decision_record']
    decision_summary: str = Field(min_length=1, max_length=800)
    _required_evidence_fields: ClassVar[frozenset[str]] = frozenset({'title', 'summary', 'decision_summary'})


class TodoCandidate(_ExtractionCandidateBase):
    item_type: Literal['todo']
    priority: Literal['low', 'medium', 'high']
    priority_reason: str = Field(min_length=1, max_length=400)
    task_summary: str | None = Field(default=None, min_length=1, max_length=400)
    assignee: str | None = Field(default=None, min_length=1, max_length=100)
    due_date: str | None = Field(default=None, min_length=1, max_length=64)
    evidence_reason: str | None = Field(default=None, min_length=1, max_length=400)
    source_type: Literal['gmail', 'gmail_attachment', 'drive', 'calendar', 'internal_document'] | None = None
    project_tag: str | None = Field(default=None, min_length=1, max_length=100)
    _required_evidence_fields: ClassVar[frozenset[str]] = frozenset({'title', 'summary', 'priority', 'priority_reason'})


class _ExtractionEnvelopeBase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    result_kind: Literal['candidate', 'no_candidate']
    candidate: _ExtractionCandidateBase | None
    no_candidate_reason: NoCandidateReason | None = None

    @model_validator(mode='after')
    def _validate_singular_result_and_token_budget(self) -> _ExtractionEnvelopeBase:
        if self.result_kind == 'candidate':
            if self.candidate is None or self.no_candidate_reason is not None:
                raise ValueError('candidate result requires one candidate and no no-candidate reason')
        elif self.candidate is not None or self.no_candidate_reason is None:
            raise ValueError('no-candidate result requires a reason and no candidate')
        token_count = len(tiktoken.get_encoding(AUTO_REVIEW_TOKENIZER_ENCODING).encode(self.canonical_json()))
        if token_count > AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT:
            raise ValueError('canonical extraction envelope exceeds 2048 output tokens')
        return self

    def canonical_json(self) -> str:
        serialized = json.dumps(
            self.model_dump(mode='json'), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(',', ':'),
        )
        return unicodedata.normalize('NFC', serialized)

    def canonical_bytes(self) -> bytes:
        return unicodedata.normalize('NFC', self.canonical_json()).encode('utf-8')

    def canonical_token_count(self) -> int:
        return len(
            tiktoken.get_encoding(AUTO_REVIEW_TOKENIZER_ENCODING).encode(
                self.canonical_json()
            )
        )


class TimelineExtractionResult(_ExtractionEnvelopeBase):
    candidate: TimelineCandidate | None = None


class HistoryExtractionResult(_ExtractionEnvelopeBase):
    candidate: HistoryCandidate | None = None


class DecisionRecordExtractionResult(_ExtractionEnvelopeBase):
    candidate: DecisionRecordCandidate | None = None


class TodoExtractionResult(_ExtractionEnvelopeBase):
    candidate: TodoCandidate | None = None


MailDocumentCandidate = Annotated[
    TimelineCandidate | HistoryCandidate | DecisionRecordCandidate | TodoCandidate,
    Field(discriminator='item_type'),
]


class MailDocumentExtractionResult(_ExtractionEnvelopeBase):
    candidate: MailDocumentCandidate | None = None


class AutoReviewAuditPublicSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    status: Literal['pending', 'completed', 'remediation_required']
    outcome: Literal['confirmed', 'incorrect', 'permission_violation', 'source_version_violation', 'policy_violation'] | None
    action_required: bool


class AutoReviewPublicSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    validator_model: Literal['gpt-5.6-terra']
    reasoning_effort: Literal['medium']
    validator_prompt_version: Literal['auto-review-validation:v2']
    validator_output_contract_version: Literal['candidate-validation-batch:v1']
    policy_version: Literal['auto-review-policy:v1']
    supported_substantive_field_count: int = Field(ge=0, le=2)
    minimum_entailment_score: float = Field(ge=0.0, le=1.0)
    policy_reason_codes: list[Literal['direct_fact_supported', 'trusted_exact_reaffirmation']] = Field(min_length=1, max_length=2)
    validated_at: datetime


class RevokeAutoApprovalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reason_code: Literal['business_withdrawal', 'incorrect_content', 'permission_violation', 'wrong_source_version', 'policy_violation']


class RevokeAutoApprovalResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    review_item_id: int
    status: Literal['revoked']
    replayed: bool
    knowledge_remains_trusted: bool
    revoked_document_count: int


class AutoReviewAuditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    outcome: Literal['confirmed', 'incorrect', 'permission_violation', 'source_version_violation', 'policy_violation']
    reason: str = Field(min_length=1, max_length=500)


class AutoReviewAuditResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    audit_status: Literal['completed', 'remediation_required']
    breaker_open: bool
    revoke_status: Literal['not_required', 'revoked', 'remediation_required']
