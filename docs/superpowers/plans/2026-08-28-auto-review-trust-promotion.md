# Auto-Review Trust Promotion C.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned, evidence-first auto-review path that lets low-risk direct Timeline and History facts become trusted knowledge without a human click, while every uncertain, restricted, conflicting, unsupported, or high-risk candidate remains in the existing Review Queue and every automatic approval can be audited and precisely revoked.

**Architecture:** Keep ReviewItem and trusted-knowledge rows in PostgreSQL authoritative. Every candidate still starts as `pending_review`; a deterministic eligibility layer selects only bounded public/internal direct-fact candidates, a separate OpenAI `gpt-5.6-terra` validator returns strict structured evidence judgments through real LangChain, and a versioned code policy—not the model—decides whether the existing locked transition/promotion boundary may run. Add an immutable LangGraph V2.1 beside untouched V2.0, persist validation/provenance/audit state without raw evidence, use tombstones to prevent revoked vectors from reappearing, and expose all new behavior in the existing Integrations, Review, Timeline, History, and Knowledge surfaces without adding navigation depth.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLAlchemy 2, Alembic, PostgreSQL, pgvector, SQLite smoke mode, LangChain 1.3.17, `langchain-openai` 1.6.x, LangGraph 1.2.11, `langgraph-checkpoint-postgres` 3.1.2, pytest, Ruff, Next.js 16, React 19, TypeScript, Tailwind CSS, Playwright

**Spec:** `docs/superpowers/specs/2026-08-28-auto-review-trust-promotion-design.md`

## Global Constraints

- Preserve the current dependency bounds: `langchain>=1.3.17,<1.4.0`, `langgraph>=1.2.11,<1.3.0`, `langchain-openai>=1.6.0,<1.7.0`, `langchain-google-genai>=4.3.5,<4.4.0`, and `langgraph-checkpoint-postgres>=3.1.2,<3.2.0`.
- Use actual LangChain `with_structured_output()` for the production validator and actual LangGraph `StateGraph`, `interrupt()`, and checkpointed version registry. Do not replace either boundary with custom hard-coded model or graph emulation.
- Treat `backend/app/agent_runtime/review_v2_graph.py`, `backend/app/agent_runtime/state.py`, and the exact V2.0 request/response shapes as immutable compatibility surfaces. V2.1 gets separate state and graph modules.
- Keep `AUTO_REVIEW_MODE=disabled` as the default. Disabled new runs select V2.0 and make zero validator calls. Only new `shadow` or `enforce` runs select V2.1; a stored thread never changes graph version, configured mode, model, prompt, policy, percentage, or confirmed cost ceilings on resume.
- Every candidate begins as `pending_review`. Model confidence or a validator verdict alone can never promote knowledge. Only `auto-review-policy:v1` may authorize the internal locked approval directive.
- `decision_record`, `todo`, `restricted`, unknown permission, missing evidence, uncertainty, source drift, conflict, inference, malformed output, unsupported version, generator/validator identity collision, and budget overflow always fail closed to human review or needs-more-evidence as specified.
- Resolve permissions from the exact actor permission set. The auto-policy actor is restricted to `('public', 'internal')`, may never acquire `restricted`, and may never reject, add a human note, invoke a bulk public action, or administer rollout.
- Preserve evidence links, snippets, source identifiers, version/signature, confidence, uncertainty, and the strictest permission. No evidence means no ReviewItem and therefore no validation.
- Never attach canonical ids, ReviewItem ids, external-id/source-URL metadata fields, permissions, credentials, or hidden-collision metadata to the validator; only local `Cxx`/`Exx` aliases and permission-filtered source plaintext are allowed. Tests prove metadata fields and unaliased identifiers are absent rather than making the unprovable claim that ordinary source prose can never contain an id- or URL-like substring. High-confidence credential material in that plaintext is caught by the versioned input-safety scanner and becomes zero-call human-only. Production V2.1 readiness and the final pre-attempt guard require `langchain_core.globals.get_debug() is False`, `OPENAI_LOG` not equal to `debug` case-insensitively, and `logging.getLogger('openai').getEffectiveLevel() > logging.DEBUG`; every isolated model is constructed with explicit `verbose=False`. Never toggle process-global debug/verbose/logger levels around a request because concurrent calls race. Empty callbacks, disabled LangSmith tracing, and cache-off remain required but are not treated as sufficient. The injected OpenAI HTTP client permits only the server-owned body-blind fenced-send transport hook defined below; arbitrary request/response/debug hooks are rejected. Never persist, callback-export, SDK-log, or console-log raw source text, snippets, URLs, free-form model rationale, or provider exceptions in checkpoints, validation rows, logs, stdout/stderr, or AuditLog.
- Use keyed HMACs, canonical JSON, UTF-8 NFC, stable ordering, and the current fingerprint key version plus domain-separated key-material verifier for validation, duplicate, evidence, sampling, and preview fingerprints. Every durable keyed artifact snapshots both non-secret identities. Live operational identities that authorize a new validation/promotion/mutation must match the current runtime generation or become zero-call human-only; never guess or reuse old secret material. Immutable historical attribution fields (past revoke/audit/rollout/key-admin actor HMACs and completed promotion/audit selection proofs) retain their recorded old key identity, are never re-HMACed, and do not by themselves make current runtime readiness false. New control-plane actions always write a new current-key attribution record/fields. Actor attribution domains are frozen as `auto-review-revoke-actor:v1`, `auto-review-audit-actor:v1`, and `auto-review-rollout-actor:v1`; a value from one domain can never replay in another. Do not use semantic similarity for V1 duplicate reaffirmation.
- Every production C.5 transaction follows one optional-prefix total order: `shared key-generation barrier -> AutoReviewRuntimeKeyState FOR SHARE -> all required AutoReviewProviderSafetyState rows sorted by (purpose, provider, model, reasoning_effort) -> projection advisory lock -> sorted rollout rows -> sorted canonical Source rows -> sorted workflow rows -> call/children -> ReviewItem -> promotion decision/audit -> approval/evidence links -> target knowledge -> sorted document locks -> tombstone/index/vector state`. A transaction skips irrelevant classes but never acquires a later class and then an earlier one. Read paths lock every required safety/source row `FOR SHARE`; mutations use `FOR UPDATE`. Unlocked discovery may identify ids, but every membership and identity is rechecked after ordered locks; no code locks a singleton first and discovers the remaining set later. Canonical ingestion is a separate monotonic phase: it may continue after sorted `Source FOR UPDATE` only into that Source's sorted Document/Version/Chunk rows and document advisory/vector state, then commits; it never acquires the earlier provider-safety/projection/rollout/workflow classes after Source. Reconciliation starts a fresh transaction from the global prefix, so no ingestion transaction holds Source while acquiring projection or rollout. A crash between those phases remains fail-closed through live serving eligibility and bounded recovery. Rotation takes the generation barrier exclusively, locks the runtime row `FOR UPDATE`, then takes the projection lock before advancing version/material/generation. The barrier uses one database-global fixed signed-64-bit `AUTO_REVIEW_KEY_GENERATION_LOCK_ID`; it is never derived from the fingerprint secret, key version, material verifier, tenant, or document id, so old and new generations necessarily contend on the same lock. SQL-barrier interleaving tests cover reversed route sets, ingestion versus provider completion, human versus automatic promotion, reconciliation, revoke, and reindex and must complete without deadlock or TOCTOU.
- New C.5 business tables never store a raw human subject id. ReviewItem revoke attribution, post-audit attribution, and rollout authorization attribution use domain-separated subject HMACs plus key version/material verifier; the existing access-controlled AuditLog remains the separately governed identity audit surface.
- Enforce exact bounds before any provider call: each of the five selectable extraction agents returns zero or one strict candidate, so a workflow can produce at most 5 candidates; validation batches at most 4 candidates and 12 batch-local evidence slots, exactly 2 substantive claims per candidate, at most 2,000 characters per claim, at most 12,000 serialized input characters, at most 6,000 framed input tokens, and at most 3,072 total output tokens. A workflow uses at most 2 validation batches/5 candidates, reserves at most USD 0.048864 per maximum batch, USD 0.097728 for validation, USD 0.083580 for all five extraction calls, and USD 0.181308 combined under the immutable USD 0.20 workflow budget. Token counts use the frozen estimator below; a character count is never treated as a token count. Larger corpora are partitioned into additional separately previewed/idempotent workflows instead of weakening evidence or raising an implicit bill.
- A V2.1 total-cost confirmation covers extraction as well as validation. Each selected extraction agent binds the exact registry-owned OpenAI provider, `gpt-5.4-mini-2026-03-17` snapshot, `none` reasoning, agent-specific route/prompt/output contract, exact rendered-input HMAC/token count, 10,000/2,048 token caps, Decimal USD 0.75/M input and USD 4.50/M output prices, and one provider attempt. V2.1 sets SDK retries to zero and has no provider/model fallback; Azure OpenAI, Gemini, aliases, and mutable/fallback routes are unavailable. Every attempt has a durable reserve/attempt/actual ledger and provider-safety breaker. V2.0 routing/retries remain unchanged.
- Call the provider outside database transactions and locks. Use short claim/lease transactions and revalidate lease, status, source version/signature, content fingerprint, permission, duplicate/conflict result, policy, and budget in a fresh transaction before saving or promoting.
- Reuse `ReviewTransitionService` for all human and automatic resolutions. Auto-review may not insert knowledge directly. Exact reaffirmation must use the same locked service with an internal `reuse_existing` directive.
- Keep raw source-chunk indexing human/legacy-human only. Auto-policy approvals may index promoted trusted History/Timeline documents, but do not broaden raw `chunk:{id}` eligibility before Deliverable D.
- Never physically delete ReviewItem, validation, provenance, audit, rollout, or knowledge audit rows. Exact revoke changes status, writes tombstones, removes exact serving/index state, and is replay-safe.
- Any existing post-audit row linked to the promotion decision—`mandatory_50`, `sample_10`, `sample_2`, or later `manual`—cannot be bypassed by direct revoke. A normal direct revoke exists only for reason `business_withdrawal` and only when no audit exists or that row is `completed` with outcome `confirmed`; `pending`, `remediation_required`, or any critical/corrected outcome returns bounded HTTP 409 `audit_required`. Every quality reason always enters the server-owned breaker-first audit/correction coordinator, including after confirmation. Only that coordinator/recovery or source-invalidation may bypass the normal gate.
- Keep the existing one-click path: the current Integrations preview displays extraction, validation, and total cost; the existing `검토 후보 만들기` click submits the signed preview. Do not add a page, wizard, modal, or second confirmation click.
- Tests and smoke checks use fake/deterministic models and fake clients. They must not call live OpenAI, Gemini, Google, Slack, OAuth, embedding, or other provider APIs.
- New `shadow`/`enforce` preview/start requires PostgreSQL advisory-lock and row-lock capability; SQLite returns a bounded zero-call unavailable result. Disabled SQLite smoke keeps the existing V2.0 human Review path through a dialect-aware process-local mutation guard, but it can never mark the C.5 projection/auto-review runtime ready.
- Keep Slack recovery, CDC/streaming, Deliverable D retrieval, Deliverable E Neo4j GraphRAG, and unrelated agent/source work out of C.5.
- Apply TDD to every behavior change: run the named focused test and observe the expected failure, implement the smallest slice, rerun it green, lint touched Python, and commit only that slice.
- If implementation requires changing the approved output schema, permission policy, token/cost caps, Review Queue trust boundary, trusted-knowledge promotion rule, duplicate resolution rule, model/prompt/policy identity, or rollout gate, stop for a new human decision before coding further.

## Human-Gated Contract Frozen by Approval of This Plan

Approval of this plan freezes the following additive contract. The existing V2.0 contract remains byte-for-byte equivalent at the JSON field level.

### Versions, modes, model, and bounds

```python
COMPANY_MEMORY_REVIEW_GRAPH_VERSION_V21 = (
    'company-memory-review-v2.1-auto-review'
)
AUTO_REVIEW_POLICY_VERSION = 'auto-review-policy:v1'
AUTO_REVIEW_VALIDATOR_PROMPT_VERSION = 'auto-review-validation:v1'
AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION = 'candidate-validation-batch:v1'
AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION = 'trusted-claim-normalization:v1'
AUTO_REVIEW_COST_POLICY_VERSION = 'auto-review-cost:v1'

AutoReviewConfiguredMode = Literal['disabled', 'shadow', 'enforce']
AutoReviewStoredMode = Literal['shadow', 'enforce']
AutoReviewEffectiveMode = Literal['disabled', 'shadow', 'enforce']

AUTO_REVIEW_VALIDATOR_PROVIDER = 'openai'
AUTO_REVIEW_VALIDATOR_MODEL = 'gpt-5.6-terra'
AUTO_REVIEW_REASONING_EFFORT = 'medium'
AUTO_REVIEW_MIN_ENTAILMENT = Decimal('0.9800')
AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH = 4
AUTO_REVIEW_MAX_EVIDENCE_SLOTS = 12
AUTO_REVIEW_MAX_CLAIM_CHARS = 2_000
AUTO_REVIEW_MAX_INPUT_CHARS = 12_000
AUTO_REVIEW_TOKEN_ESTIMATOR_VERSION = 'openai-o200k-chat:v1'
AUTO_REVIEW_EXTRACTION_TOKEN_ESTIMATOR_VERSION = 'openai-o200k-extraction:v1'
AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION = 'auto-review-extraction-cost:v1'
AUTO_REVIEW_EXTRACTION_PROVIDER = 'openai'
AUTO_REVIEW_EXTRACTION_MODEL = 'gpt-5.4-mini-2026-03-17'
AUTO_REVIEW_EXTRACTION_REASONING_EFFORT = 'none'
AUTO_REVIEW_EXTRACTION_ROUTE_VERSION = 'auto-review-extraction-route:v1'
AUTO_REVIEW_EXTRACTION_AGENT_NAMES = (
    'mail_document_agent',
    'timeline_agent',
    'history_agent',
    'decision_record_agent',
    'todo_agent',
)
AUTO_REVIEW_MAX_SELECTED_EXTRACTION_AGENTS = 5
AUTO_REVIEW_MAX_EXTRACTION_CANDIDATES_PER_AGENT = 1
AUTO_REVIEW_MAX_EXTRACTION_INPUT_CHARS_PER_AGENT = 24_000
AUTO_REVIEW_MAX_EXTRACTION_INPUT_TOKENS_PER_AGENT = 10_000
AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT = 2_048
AUTO_REVIEW_EXTRACTION_MAX_PROVIDER_ATTEMPTS_PER_AGENT = 1
AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M = Decimal('0.750000')
AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M = Decimal('4.500000')
AUTO_REVIEW_TOKENIZER_ENCODING = 'o200k_base'
AUTO_REVIEW_REPLY_PRIMING_TOKENS = 16
AUTO_REVIEW_FRAMING_SAFETY_TOKENS = 512
AUTO_REVIEW_MAX_INPUT_TOKENS = 6_000
AUTO_REVIEW_MAX_OUTPUT_TOKENS = 3_072
AUTO_REVIEW_MAX_BATCHES_PER_WORKFLOW = 2
AUTO_REVIEW_MAX_CANDIDATES_PER_WORKFLOW = 5
AUTO_REVIEW_MAX_PROVIDER_ATTEMPTS = 1
AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS = 60
AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS = 5
AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS = 120
AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS = 30
AUTO_REVIEW_MAX_BATCH_COST_USD = Decimal('0.048864')
AUTO_REVIEW_MAX_VALIDATION_COST_USD = Decimal('0.097728')
AUTO_REVIEW_MAX_EXTRACTION_COST_USD = Decimal('0.083580')
AUTO_REVIEW_MAX_PROFILE_COST_USD = Decimal('0.181308')
AUTO_REVIEW_MAX_WORKFLOW_COST_USD = Decimal('0.20')
AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M = Decimal('2.000000')
AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M = Decimal('12.000000')
```

`disabled` is a runtime/config value, not a stored V2.1 request mode. New disabled launches remain V2.0. A persisted V2.1 `shadow` or `enforce` request may only be demoted at execution time: global disabled or missing/open/mismatched provider-safety state yields effective disabled/unavailable and zero calls; a rollout/audit breaker, unmet rollout gate, or changed authorization generation demotes enforce to shadow. Each launch snapshots `authorized_percentage_at_launch` and `rollout_authorization_generation`; raising config/latch later never promotes an old stored thread to a stronger mode or percentage, and breaker-close/re-authorization never resurrects an old generation.

`openai-o200k-chat:v1` is a registry-owned conservative estimator, not `len(text) / 4`. It creates a JSON-safe request structure, renders one exact NFC string with sorted compact JSON (`ensure_ascii=False`, `allow_nan=False`), tokenizes that string with `tiktoken.get_encoding('o200k_base').encode(canonical_text)`, and separately uses `canonical_text.encode('utf-8')` as the HMAC/fixture byte sequence. It then adds exactly 16 reply-priming tokens and 512 framing-safety tokens. Thus `estimated_framed_input_tokens = encoded_canonical_request_tokens + 16 + 512`, and that total—not the body-only count—must fit 6,000. The constants and formula are part of the estimator/cost-policy identity and launch token. Golden fixtures store frozen canonical request text/bytes and assert body count directly with `tiktoken` plus the two literal additions, as well as Korean/ASCII/schema boundary vectors; a dependency upgrade that changes a count requires a new estimator/cost-policy and approval. The V2.1 zero-call preview signs the estimator version, tokenizer encoding, both allowance constants, per-batch input/output token caps, maximum batch/candidate count, and the resulting validation/total cost ceilings. Task 9 prepares each final provider invocation exactly once, persists its keyed content HMAC and measured framed-token counts while claiming the call, and immediately before the provider-attempt marker compares that same immutable in-memory object with the stored identity plus current price registry/cost-policy values. It never renders the LangChain request a second time. An unavailable tokenizer/model framing registry is `validator_unavailable` and zero-call human-only; it never falls back to a character heuristic.

The reviewed validation price snapshot is Terra input USD 2.00/M and output USD 12.00/M from the [official GPT-5.6 Terra model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra), checked 2026-08-28. `backend/app/agent_runtime/auto_review_cost_policy.py` owns one versioned immutable validation registry entry keyed by `(cost_policy_version, provider, model, reasoning_effort) = ('auto-review-cost:v1', 'openai', 'gpt-5.6-terra', 'medium')`; preview, admission, provider-safety authorization, and launch verification all read that entry. Deployment price settings are only an explicit confirmation and must equal the registry's exact six-place Decimals before shadow/enforce readiness—arbitrary positive or lower values are rejected. The Responses API `max_output_tokens` covers visible plus reasoning tokens, so the approved 3,072 cap replaces the non-executable 512 draft that could not even express the measured maximum four-candidate strict response. One maximum validation batch reserves `6000*2/1_000_000 + 3072*12/1_000_000 = USD 0.048864`; two batches reserve USD 0.097728. The fixed four-candidate batch shape and five-candidate workflow maximum therefore require at most two calls. A later official price, schema, reasoning, or cap change requires a new registry entry/cost-policy version, reviewed provider-safety authorization, and fresh preview, not an in-place config or arithmetic change.

`openai-o200k-extraction:v1` applies the same exact-count discipline to every paid V2.1 extraction agent. One invocation is capped at 24,000 input characters, 10,000 framed input tokens, 2,048 total output tokens, and exactly zero or one candidate. Every registry entry fixes provider `openai`, model snapshot `gpt-5.4-mini-2026-03-17`, reasoning `none`, route `auto-review-extraction-route:v1`, exact agent/prompt/output-contract identity, and USD 0.75/M input plus USD 4.50/M output prices from the [official GPT-5.4 Mini model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini), checked 2026-08-28. Thus one maximum extraction call reserves `10000*0.75/1_000_000 + 2048*4.50/1_000_000 = USD 0.016716`; all five reserve USD 0.083580. Together with two maximum validation batches the full profile reserves USD 0.181308, leaving USD 0.018692 below the immutable USD 0.20 limit. The selected-agent count is an enforceable candidate bound: preview reserves `selected_agent_count * USD 0.016716` for extraction and `ceil(selected_agent_count / 4) * USD 0.048864` for validation, never a predicted-output discount.

The immutable extraction registry contains exactly these five entries; a missing or extra tuple is unavailable rather than configurable:

| Agent | Prompt version | Output contract / exact schema | Allowed candidate item type |
|---|---|---|---|
| `mail_document_agent` | `mail-document-extraction:c5-v1` | `mail-document-candidate:c5-v1` / `MailDocumentExtractionResult` | `timeline_event|history_event|decision_record|todo` |
| `timeline_agent` | `timeline-extraction:c5-v1` | `timeline-candidate:c5-v1` / `TimelineExtractionResult` | `timeline_event` |
| `history_agent` | `history-extraction:c5-v1` | `history-candidate:c5-v1` / `HistoryExtractionResult` | `history_event` |
| `decision_record_agent` | `decision-record-extraction:c5-v1` | `decision-record-candidate:c5-v1` / `DecisionRecordExtractionResult` | `decision_record` |
| `todo_agent` | `todo-extraction:c5-v1` | `todo-candidate:c5-v1` / `TodoExtractionResult` | `todo` |

All five result classes use `ConfigDict(extra='forbid')` and the same singular envelope: `result_kind: Literal['candidate','no_candidate']`, `candidate: CandidatePayload | None`, and `no_candidate_reason: Literal['no_relevant_evidence','insufficient_direct_evidence','non_business_evidence','conflicting_evidence'] | None`. A model validator requires exactly one non-null `candidate` and a null reason for `candidate`, or a null candidate and one reason for `no_candidate`; no candidate list exists. Common candidate fields are `title` 1–160 characters, `summary` 1–800 characters, `confidence_score` Decimal 0.0000–1.0000 with at most four decimal places, optional `uncertainty_reason` 1–400 characters, and 1–10 `CandidateFieldEvidence` rows. Each evidence row has one item-type-allowlisted substantive `field_name` and 1–12 unique local slots matching `^S(0[1-9]|1[0-2])$`; each required substantive field appears exactly once, all slots must exist in the prepared input, and the union is non-empty. These per-field maxima are independent safety ceilings, not a promise that their Cartesian maximum is a valid response.

The item payloads are also `extra='forbid'`: Timeline requires only `result_summary` 1–800; History requires only `reason` 1–800; Decision requires only `decision_summary` 1–800; Todo requires `priority: low|medium|high` and `priority_reason` 1–400, with optional `task_summary` 1–400, `assignee` 1–100, `due_date` 1–64, `evidence_reason` 1–400, `source_type: gmail|gmail_attachment|drive|calendar|internal_document`, and `project_tag` 1–100. Required evidence field names are `title,summary,result_summary`, `title,summary,reason`, `title,summary,decision_summary`, or `title,summary,priority,priority_reason` plus every present optional Todo field, respectively. `mail_document_agent` uses a discriminated union of those exact four payloads; each other class exposes only its one branch. As part of each frozen output contract, a final model validator first produces a JSON-safe structure equivalent to `model_dump(mode='json')` but canonicalizes `confidence_score` by replacing every exact Decimal zero, including signed `-0`, with positive `Decimal('0')`, then quantizing to four places and formatting it as the fixed string `0.0000`; therefore `-0`, `0`, `0.0000`, `0.98`, and `0.9800` have the expected sign-insensitive/equal-value canonical forms. It then renders one NFC, sorted-key compact JSON string with `ensure_ascii=False` and `allow_nan=False`, counts `o200k_base.encode(canonical_text)`, and rejects any candidate whose complete canonical envelope exceeds 2,048 tokens. The same string encoded once as UTF-8 supplies HMAC/fixture bytes. Frozen Korean/ASCII fixtures cover exact 2,048-token acceptance, 2,049-token rejection, signed-zero normalization, and individually field-valid combinations that exceed the aggregate budget. Unknown/extra fields, a second candidate shape, an invalid item type, a missing/duplicate field binding, an unrecognized slot, or an aggregate over-cap output is sanitized human-only and never truncated into a trusted candidate. A provider response truncated by the native Responses `max_output_tokens=2048` cap is malformed/human-only; partial JSON is never repaired.

V2.1 does not reuse the legacy provider payload verbatim: a dedicated renderer accepts only permission-filtered source text, bounded timestamps, and ephemeral `Sxx`/`Mxx`/`Pxx` aliases; canonical/external source ids, URLs, permission labels, participant email addresses, connector metadata, secrets, and server ownership fields remain outside the provider DTO and are reattached only through server-side evidence bindings. Immediately after permission filtering and before provider DTO construction, the same frozen `credential-scan:v1` policy used by validation runs over the exact plaintext; a match discards the prepared text and is zero extraction calls, zero cache, and zero trace. For each real extraction attempt, source data is loaded in a short read and detached, then exactly one immutable `PreparedExtractionInvocation` is rendered outside every database transaction. Its exact canonical request bytes, HMAC, character/token counts, output cap, route, prices, and provider-safety snapshot are carried unchanged through claim, attempt-marker CAS, and `invoke()`. The adapter is constructed with explicit `reasoning_effort='none'`, `verbose=False`, `cache=False`, and `max_retries=0`, receives `callbacks=[]`, invokes inside `tracing_context(enabled=False)`, and rejects ambient callback/tracer/cache injection. Readiness and E2 refuse while LangChain global debug is enabled. Tests spy on rendering, invocation, stdout/stderr, and callbacks so a second render, an open SQLAlchemy transaction/session, a debug/verbose console handler, or any raw identifier/credential in the outbound request fails before rollout.

### Resolution actor and internal approval directive

```python
ReviewResolutionCapability = Literal[
    'human_review',
    'auto_review',
    'auto_review_rollout_admin',
]

@dataclass(frozen=True)
class ReviewResolutionActor:
    subject_id: str
    actor_type: Literal['human', 'auto_policy']
    allowed_permission_levels: tuple[str, ...]
    capabilities: frozenset[ReviewResolutionCapability]
    policy_version: str | None = None

@dataclass(frozen=True)
class CreateNewPromotion:
    kind: Literal['create_new'] = 'create_new'

@dataclass(frozen=True)
class ReuseExistingPromotion:
    expected_type: Literal['timeline_event', 'history_event']
    expected_id: int
    expected_claim_fingerprint: str
    expected_companion_id: int | None
    expected_companion_claim_fingerprint: str | None
    kind: Literal['reuse_existing'] = 'reuse_existing'

ApprovalDirective = CreateNewPromotion | ReuseExistingPromotion
```

Public review actions remain `approve`, `reject`, and `needs_more_evidence`; no body or dependency may construct an auto-policy actor or `reuse_existing` directive. Existing authenticated users pass current RBAC first and are then adapted to `actor_type='human'`. The internal auto actor is `system:auto-review`, has only `auto_review`, only public/internal permissions, and a matching policy version. For a reusable History, the server-derived directive binds its canonical companion Timeline id and target-specific fingerprint. Under the same locks the resolution service must resolve exactly one active companion whose project/scope/permission/status and normalized written fields match; missing, ambiguous, or mismatched companion state is human-only and may neither create a duplicate nor partially link/revoke a bundle.

### Eligibility, validation, and policy

```python
AutoReviewEligibilityDecision = Literal[
    'eligible',
    'human_review',
    'needs_more_evidence',
    'reuse_trusted',
]
AutoReviewPolicyDecision = Literal[
    'auto_approve',
    'human_review',
    'needs_more_evidence',
    'reuse_trusted',
]
AutoReviewPolicyReasonCode = Literal[
    'eligible',
    'item_type_not_allowed',
    'permission_not_allowed',
    'evidence_binding_missing',
    'evidence_binding_mismatch',
    'evidence_version_changed',
    'evidence_missing',
    'sensitive_input_detected',
    'uncertainty_present',
    'high_risk_language',
    'project_selection_required',
    'payload_not_supported',
    'generation_identity_missing',
    'validator_identity_collision',
    'registry_unavailable',
    'budget_exceeded',
    'trusted_exact_match',
    'trusted_visible_conflict',
    'trusted_hidden_collision',
    'trusted_lookup_unavailable',
    'validator_unavailable',
    'validator_malformed',
    'validator_not_supported',
    'validator_not_direct_fact',
    'validator_score_below_threshold',
    'validator_uncertain',
    'validator_conflict',
    'post_validation_drift',
    'enforce_not_selected',
    'rollout_gate_closed',
    'breaker_open',
]

@dataclass(frozen=True)
class TrustedTargetRef:
    knowledge_type: Literal['timeline_event', 'history_event']
    knowledge_id: int
    claim_fingerprint: str

CandidateSlotId = Annotated[str, Field(pattern=r'^C0[1-4]$')]
EvidenceSlotId = Annotated[str, Field(pattern=r'^E(?:0[1-9]|1[0-2])$')]

@dataclass(frozen=True)
class CandidateValidationRequest:
    candidate_slot_id: CandidateSlotId
    item_type: Literal['timeline_event', 'history_event']
    claims: tuple[ValidationClaimInput, ValidationClaimInput]
    evidence_slots: tuple[ValidationEvidenceSlot, ...]

@dataclass(frozen=True)
class ValidationClaimInput:
    field_key: Literal['title', 'result_summary', 'reason']
    text: str

@dataclass(frozen=True)
class ValidationEvidenceSlot:
    slot_id: EvidenceSlotId
    text: str

class FieldValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    field_key: Literal['title', 'result_summary', 'reason']
    verdict: Literal[
        'supported', 'partially_supported', 'unsupported', 'contradicted'
    ]
    claim_scope: Literal[
        'direct_fact', 'inference', 'decision', 'todo', 'unknown'
    ]
    entailment_score: Decimal = Field(
        ge=Decimal('0'),
        le=Decimal('1'),
        max_digits=5,
        decimal_places=4,
    )
    evidence_slot_ids: list[EvidenceSlotId] = Field(max_length=12)

class CandidateValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    candidate_slot_id: CandidateSlotId
    claim_results: list[FieldValidationResult] = Field(
        min_length=2,
        max_length=2,
    )
    uncertainty_codes: list[Literal[
        'ambiguous_subject',
        'ambiguous_time',
        'conditional_language',
        'partial_evidence',
        'proposal_language',
        'unknown',
    ]] = Field(max_length=4)
    conflict_codes: list[Literal[
        'source_contradiction',
        'trusted_claim_conflict',
        'unknown',
    ]] = Field(max_length=4)

class CandidateValidationBatchResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    results: list[CandidateValidationResult] = Field(min_length=1, max_length=4)

@dataclass(frozen=True)
class PreparedValidationInvocation:
    requests: tuple[CandidateValidationRequest, ...]
    rendered_messages: tuple[BaseMessage, ...]
    serialized_char_count: int
    framed_input_tokens: int
    max_output_tokens: int
    content_hmac: str

class AutoReviewValidator(Protocol):
    def prepare_many(
        self,
        requests: list[CandidateValidationRequest],
    ) -> PreparedValidationInvocation:
        raise NotImplementedError

    def invoke_prepared(
        self,
        invocation: PreparedValidationInvocation,
        grant: ProviderAttemptGrant,
    ) -> list[CandidateValidationResult]:
        raise NotImplementedError
```

Timeline substantive fields are exactly `title` and `result_summary`; History substantive fields are exactly `title` and `reason`, both built from `build_promotion_preview()` normalization. Every field must be `supported`, `direct_fact`, linked to a non-empty known unique evidence slot, and score at least exact `Decimal('0.9800')`. Unknown/duplicate/missing candidate, field, or evidence slots, any extra field, invalid decimal, unrecognized code, and any malformed partial batch make the whole batch `human_review`; attacker-controlled values are discarded rather than persisted.

### V2.1 graph state and routing

```python
REVIEW_V21_STATUS_ORDER = (
    'pending_review',
    'approved',
    'rejected',
    'needs_more_evidence',
    'revoked',
)

class ReviewV21GraphState(TypedDict):
    workflow_thread_id: str
    graph_version: str
    input_hash: str
    evidence_version_hash: str
    review_status_counts: dict[str, int]
    phase: str
    completed_nodes: Annotated[list[str], merge_completed_nodes]
    error_codes: Annotated[list[str], merge_error_codes]
```

The V2.1 topology is `draft -> run_auto_review -> refresh_review_resolution -> route`. Routing priority is exact: `total == 0` terminates through the distinct `finalize_no_candidates` node; otherwise pending greater than zero interrupts; otherwise needs-more-evidence greater than zero terminates `needs_more_evidence`; otherwise `approved + rejected + revoked == total` terminates through `finalize_auto_resolved` without an interrupt. The checkpoint contains only the fields above; it never gains validation ids, ReviewItem ids, source refs/content, model output, costs, actor data, or provider errors.

### Public additive contract

Keep current `ReviewWorkflowDryRunResponse` and `ReviewWorkflowStatusResponse` as exact V2.0 classes. Add separate V2.1 classes and a `graph_version` discriminated union at the route boundary:

```python
class ReviewWorkflowRunRequestV21(ReviewWorkflowRunRequest):
    launch_confirmation_token: str = Field(min_length=32, max_length=2048)

class ReviewWorkflowDryRunResponseV21(BaseModel):
    model_config = ConfigDict(extra='forbid')

    workflow_name: Literal['company-memory-review']
    graph_version: Literal['company-memory-review-v2.1-auto-review']
    source_count: int
    agent_names: list[str]
    selection_policy_version: Literal['company-memory-review-selection:v1']
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    budget_limit_usd: float | None
    budget_status: BudgetStatus
    cache_hit: bool
    requires_explicit_run: Literal[True]
    auto_review_mode: Literal['shadow', 'enforce']
    auto_review_policy_version: Literal['auto-review-policy:v1']
    auto_review_validator_provider: Literal['openai']
    auto_review_validator_model: Literal['gpt-5.6-terra']
    auto_review_reasoning_effort: Literal['medium']
    auto_review_validator_prompt_version: Literal[
        'auto-review-validation:v1'
    ]
    auto_review_validator_output_contract_version: Literal[
        'candidate-validation-batch:v1'
    ]
    auto_review_cost_policy_version: Literal['auto-review-cost:v1']
    auto_review_enforce_percentage: Literal[0, 10, 100]
    auto_review_estimated_input_tokens: int
    auto_review_estimated_output_tokens: int
    auto_review_estimated_cost_usd: float
    total_estimated_input_tokens: int
    total_estimated_output_tokens: int
    total_estimated_cost_usd: float
    launch_confirmation_token: str = Field(min_length=32, max_length=2048)

# V2.1 semantics only: estimated_* is the extraction estimate,
# auto_review_estimated_* is the validation upper bound, and total_* is their sum.
# budget_limit_usd is the configured exact V2.1 total-cost limit serialized for
# transport; budget_status is computed from total_estimated_cost_usd, not only
# the legacy-named extraction fields.

class ReviewStatusCountsV21(BaseModel):
    model_config = ConfigDict(extra='forbid')

    pending_review: int = Field(ge=0)
    approved: int = Field(ge=0)
    rejected: int = Field(ge=0)
    needs_more_evidence: int = Field(ge=0)
    revoked: int = Field(ge=0)

class ReviewWorkflowStatusResponseV21(BaseModel):
    model_config = ConfigDict(extra='forbid')

    thread_id: str
    status: Literal[
        'created',
        'drafting',
        'checkpoint_pending',
        'awaiting_human_review',
        'resuming',
        'completed',
        'needs_more_evidence',
        'checkpoint_failed',
        'failed',
        'cancelled',
    ]
    review_item_count: int
    review_status_counts: ReviewStatusCountsV21
    durable: bool
    graph_version: Literal['company-memory-review-v2.1-auto-review']
    review_resolution_ready: bool
    checkpoint_resumable: bool
    resume_allowed: bool
    retry_allowed: bool
    created_at: datetime
    updated_at: datetime
    error_code: ReviewWorkflowErrorCode | None
    resume_error_code: ReviewWorkflowErrorCode | None
    auto_review_mode: Literal['shadow', 'enforce']
    auto_review_policy_version: Literal['auto-review-policy:v1']
    auto_review_enforce_percentage: Literal[0, 10, 100]
    auto_approved_count: int
    human_review_required_count: int
    auto_review_fallback_count: int

class AutoReviewAuditPublicSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    status: Literal['pending', 'completed', 'remediation_required']
    outcome: Literal[
        'confirmed',
        'incorrect',
        'permission_violation',
        'source_version_violation',
        'policy_violation',
    ] | None
    action_required: bool

class AutoReviewPublicSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    validator_model: Literal['gpt-5.6-terra']
    reasoning_effort: Literal['medium']
    validator_prompt_version: Literal['auto-review-validation:v1']
    validator_output_contract_version: Literal['candidate-validation-batch:v1']
    policy_version: Literal['auto-review-policy:v1']
    supported_substantive_field_count: int = Field(ge=0, le=2)
    minimum_entailment_score: float = Field(ge=0.0, le=1.0)
    policy_reason_codes: list[Literal[
        'direct_fact_supported',
        'trusted_exact_reaffirmation',
    ]] = Field(min_length=1, max_length=2)
    validated_at: datetime

class RevokeAutoApprovalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    reason_code: Literal[
        'business_withdrawal',
        'incorrect_content',
        'permission_violation',
        'wrong_source_version',
        'policy_violation',
    ]

class RevokeAutoApprovalResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    review_item_id: int
    status: Literal['revoked']
    replayed: bool
    knowledge_remains_trusted: bool
    revoked_document_count: int

class AutoReviewAuditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    outcome: Literal[
        'confirmed',
        'incorrect',
        'permission_violation',
        'source_version_violation',
        'policy_violation',
    ]
    reason: str = Field(min_length=1, max_length=500)

class AutoReviewAuditResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    audit_status: Literal['completed', 'remediation_required']
    breaker_open: bool
    revoke_status: Literal[
        'not_required', 'revoked', 'remediation_required'
    ]
```

The additional `total_estimated_input_tokens`, `total_estimated_output_tokens`, validator identity fields, and bounded `auto_review_audit` projection are implementation-plan clarifications required by the already approved cost and badge UX. This approved plan freezes them.

- V2.1 dry-run returns a non-null short-lived `launch_confirmation_token` and all fields in the strict class above.
- V2.1 launch requires the exact token returned by dry-run. V2.0 omits it. Changed or expired preview data returns bounded `cost_preview_changed` with HTTP 409 and requires a fresh zero-call preview.
- V2.1 status adds mode, policy version, enforce percentage, auto-approved count, human-review-required count, and auto-review-fallback count. It exposes counts only, never hidden item identities.
- Review responses add nullable `resolution_source`, nullable `resolution_policy_version`, and a bounded `auto_review_summary` containing validator model, reasoning effort, prompt/output-contract/policy versions, supported-field count, minimum score, reason codes, and validation timestamp. Internal ids, keys, leases, raw output, and raw evidence remain hidden.
- Review responses also add nullable `auto_review_audit: {status, outcome, action_required}`. This bounded projection is required to render `감사 필요` and `조치 필요` truthfully after reload; it never exposes an audit id, cohort counters, hidden provenance, another ReviewItem id, or a raw reason. This approved plan freezes that public-field clarification.
- Add `POST /api/v1/review/{review_item_id}/revoke-auto-approval` with one required strict `reason_code` and no free-text classification/note. `business_withdrawal` is the sole non-quality reason; the other four map deterministically to breaker-first quality audit outcomes. Its response is exactly `review_item_id`, `status='revoked'`, `replayed`, `knowledge_remains_trusted`, and `revoked_document_count`.
- The revoke success schema is returned only after exact revoke commits. If a quality assessment/audit/correction has already committed breaker+quarantine but physical revoke fails or remains pending, the endpoint returns bounded HTTP 409 `remediation_required` with no ids/internal reason; that committed state is not rolled back. The client immediately refetches the item and displays `action_required`. A replay may finish the same idempotent recovery and then return the success schema, or return the same 409 while remediation remains.
- Add `POST /api/v1/review/{review_item_id}/auto-review-audit` for authorized `confirmed`, `incorrect`, `permission_violation`, `source_version_violation`, or `policy_violation` outcomes and a 1–500 character reason. Its response contains only bounded audit status, breaker-open state, and revoke state.
- Add `revoked` to the general Review status union, but not to V2.0 graph state or V2.0 status counts. A revoke endpoint can target only auto-policy-approved V2.1 items, so a paused V2.0 tuple can never acquire this status.

### Persistence and serving invariants

- `AgentWorkflowRequest` stores immutable V2.1 mode, validator provider/model/reasoning, prompt/output-contract/policy/cost-policy identities, enforce percentage, exact token/candidate/timing/price profile, and confirmed extraction/validation/total ceilings.
- `ReviewItemEvidenceRef` is inserted in the candidate draft transaction and is immutable. It links a candidate to exact `AgentWorkflowEvidenceRef` rows without copying source text, URLs, or snippets.
- `AutoReviewValidationCall` is the authoritative one-provider-attempt batch ledger and lease. It atomically reserves the signed worst-case batch/workflow budget, snapshots all shared provider timing values, records total usage/cost once, and owns one or more `AutoReviewValidation` children. Each child has one unique canonical `validation_key`, bounded result/reason codes and a deterministic share of call usage; neither table stores raw evidence, a send permit, or provider exceptions.
- Every candidate created after the C.5 migration—V2.0 or V2.1—gets immutable evidence bindings, and every later human or auto promotion/reaffirmation writes one `TrustedKnowledgeApprovalLink` per primary/companion effect plus one or more child `TrustedKnowledgeEvidenceLink` rows. This normalization resolves the spec's one-effect/many-evidence cardinality without encoding evidence ordinals into effect kinds: the approval link owns claim fingerprint, resolution source, permission, scope, effect kind, and active/revoked state; child evidence links own canonical source/version and evidence HMAC. Existing pre-migration human knowledge with `source_review_item_id`, and a pre-migration pending candidate whose exact binding cannot be backfilled, remain human-only implicit base provenance; C.5 never synthesizes a childless approval link and never revokes that legacy base.
- Enforce `ReviewItem.auto_validation_id` same-item ownership with a named composite foreign key `(review_items.id, review_items.auto_validation_id) -> (auto_review_validations.review_item_id, auto_review_validations.id)` backed by a unique target pair. A nullable plain id FK plus a service-only assertion is not sufficient.
- `TrustedKnowledgeFingerprint` is the indexed serving-side collision projection for active Timeline/History rows. A separate projection-state watermark plus fresh anti-join coverage guard proves completeness; missing/stale rows or key-material mismatch disable auto review; legacy unknown scope is collision-only; hidden checks use server-side `EXISTS` without identity/content/count.
- Exact reaffirmation reuses one visible canonical knowledge id only when normalized HMAC, target type, scope, project, title bucket, and permissions match. Hidden scope-wide collision existence returns only a boolean and forces human review.
- Revoking one reaffirmation revokes only its provenance. Knowledge, companion Timeline, and vectors remain serving while any other active link or legacy-human base provenance remains.
- Last-provenance revoke atomically marks knowledge/companion Timeline revoked, records exact `VectorServingTombstone` rows, deletes exact vector index state and pgvector/in-memory serving documents, and never deletes audit history.
- Reindex and revoke use the same per-document PostgreSQL transaction advisory-lock namespace and the same `Session`/transaction as their guarded writes. Reindex acquires sorted document locks, re-reads approved-and-no-tombstone eligibility, and conditionally upserts; revoke acquires the same sorted locks before tombstone/delete. The SQL `NOT EXISTS` guard remains defense in depth, so an in-flight reindex cannot resurrect revoked content even when its first eligibility read preceded revoke.
- Rollout state is unique by `(security_scope_id, policy_version)`, replay-safe, and persistent. A critical audit opens and commits the breaker before exact revoke begins; a revoke failure leaves remediation required and effective mode shadow.

Workflow summary counts are disjoint live projections frozen as follows: `auto_approved_count` is the number of workflow-owned ReviewItems currently `approved` with `resolution_source='auto_policy'`; `human_review_required_count` is the number currently `pending_review`; and `auto_review_fallback_count` is the number currently `needs_more_evidence`. Validator unavailable/malformed, policy-human-review, and enforce-not-selected cases remain pending and therefore count only in `human_review_required_count`; revoked/rejected/human-approved items count in none of these three. This maps exactly to `자동 승인 N건 | 확인 필요 M건 | 추가 근거 필요 K건`, never includes another workflow, and exposes no item identity.

## File and Interface Map

### Existing compatibility surfaces to reuse

| Responsibility | Existing source |
|---|---|
| V2.0 public schema and exact response contract | `backend/app/schemas/review_workflow.py` |
| V2.0 safe checkpoint state | `backend/app/agent_runtime/state.py` |
| V2.0 graph topology | `backend/app/agent_runtime/review_v2_graph.py` |
| Lifecycle/start/status/resume/cancel orchestration | `backend/app/agent_runtime/review_v2_service.py` |
| Versioned graph registry | `backend/app/agent_runtime/graph_versions.py` |
| Canonical source/version and permission resolution | `backend/app/agent_runtime/canonical_sources.py`, `backend/app/ingestion/source_versions.py` |
| HMAC canonicalization | `backend/app/agent_runtime/fingerprints.py` |
| Candidate drafting and short lease behavior | `backend/app/agent_runtime/review_v2_drafting.py` |
| Review lock and exactly-once promotion | `backend/app/review/transitions.py`, `backend/app/knowledge/promotion.py` |
| Knowledge and Review persistence | `backend/app/models/knowledge.py`, `backend/app/models/review.py` |
| Incremental indexing and pgvector | `backend/app/rag/indexing.py`, `backend/app/rag/reindexing.py`, `backend/app/rag/pgvector_store.py`, `backend/app/rag/vector_store.py` |
| Existing route boundary | `backend/app/api/v1/orchestration_v2.py`, `backend/app/api/v1/review.py` |
| Existing two-step UI surfaces | `frontend/src/app/integrations/page.tsx`, `frontend/src/app/review/page.tsx` |

### New focused modules

| New file | Sole responsibility |
|---|---|
| `backend/app/schemas/auto_review.py` | Strict internal validator, policy, audit, revoke, and V2.1 value objects |
| `backend/app/models/auto_review.py` | C.5 validation-call ledger, candidate evidence, fingerprint projection, approval/evidence provenance, promotion selection, tombstone, post-audit, and rollout ORM rows |
| `backend/app/review/actors.py` | Application-owned human/auto resolution actor and capability checks |
| `backend/app/agent_runtime/auto_review_cost_policy.py` | Sole immutable Terra validation policy plus exact five-entry Mini extraction route/schema/price registry |
| `backend/app/agent_runtime/auto_review_policy.py` | Pure eligibility-result plus validator-result policy decisions and exact decimal rules |
| `backend/app/agent_runtime/auto_review_eligibility.py` | Canonical ref, permission, direct-field, exact-duplicate, and hidden-collision preflight |
| `backend/app/agent_runtime/auto_review_validator.py` | Bounded prompt construction, LangChain structured output, slot integrity, sanitized usage result |
| `backend/app/agent_runtime/auto_review_validation_store.py` | One-call batch lease, atomic budget reservation, child allocation, cache, and revalidation persistence |
| `backend/app/agent_runtime/auto_review_orchestrator.py` | Batching and provider-outside-transaction validation flow |
| `backend/app/review/auto_review_resolution.py` | Internal auto actor, locked create/reuse approval, and canonical replay |
| `backend/app/review/auto_review_rollout.py` | Effective mode, shadow comparisons, canary sampling, counters, audit, and persistent breaker |
| `backend/app/review/auto_review_revoke.py` | Exact replay-safe provenance/knowledge/vector revoke transaction |
| `backend/app/knowledge/claim_fingerprints.py` | Schema-versioned normalized claim HMAC and title collision bucket |
| `backend/app/knowledge/trusted_fingerprint_projection.py` | Indexed exact/legacy-unknown collision projection and readiness/backfill |
| `backend/app/knowledge/trusted_provenance.py` | Active link creation, legacy-base recognition, and active-provenance queries |
| `backend/app/agent_runtime/keyed_mutation_guard.py` | Fixed generation/projection lock constants and dialect-aware shared/exclusive acquisition order |
| `backend/app/rag/serving_locks.py` | Shared per-document PostgreSQL advisory-lock key and acquisition boundary |
| `backend/app/agent_runtime/launch_confirmation.py` | Short-lived signed preview issue/verify without a new idempotency namespace |
| `backend/app/agent_runtime/review_v21_state.py` | Separate five-status bounded checkpoint contract |
| `backend/app/agent_runtime/review_v21_graph.py` | Immutable V2.1 graph nodes and routing only |
| `backend/app/agent_runtime/review_v21_service.py` | V2.1 lifecycle implementation behind the version-neutral facade |
| `backend/app/agent_runtime/review_workflow_facade.py` | V2.0/V2.1 preview, launch, status, resume, and cancel dispatch |
| `backend/app/admin/auto_review_keys.py` | Local/admin key bootstrap, projection rebuild, and disabled-only rotation control plane |
| `backend/app/admin/auto_review_rollout.py` | Admin-only rollout status, authorization, breaker-close, and remediation recovery control plane |
| `backend/app/review/auto_review_audit.py` | Immutable post-audit outcome and breaker-first remediation coordinator |
| `backend/app/review/auto_review_evaluation.py` | Offline deterministic/live-authorized aggregate-only evaluation CLI |
| `backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py` | Additive C.5 schema and migration boundary marker |
| `frontend/src/lib/api/autoReview.ts` | Bounded audit/revoke client wrappers |
| `frontend/src/app/review/AutoReviewTrustPanel.tsx` | Same-screen automatic-trust audit and revoke surface |
| `frontend/src/components/review/AutoReviewBadge.tsx` | Shared human/auto/audit/remediation badge presentation |
| `frontend/src/components/review/AutoReviewActions.tsx` | Inline authorized audit and revoke actions inside the current detail surface |

No new page, queue/broker, outbox, CDC producer, streaming consumer, separate vector database, Slack adapter, D retriever, or E Neo4j projection belongs in this map.

## Spec Coverage and Dependency Map

| Concern | Owning tasks |
|---|---|
| Frozen V2.1 values, strict structured output, exact V2.0 compatibility | 1 |
| Tables, columns, indexes, constraints, retained-data-safe downgrade/reset | 2 |
| Immutable candidate evidence refs and immutable V2.1 request snapshot | 3 |
| Resolution actor, capability enforcement, locked human-path regression | 4 |
| Claim HMAC, exact duplicate/reaffirmation, provenance links, hidden collision guard | 5 |
| Tombstones, exact revoke, delete contract, raw-chunk boundary, reindex race prevention | 6 |
| Deterministic eligibility and pure policy authority | 7 |
| Real LangChain Terra structured validator and privacy/bounds | 8 |
| One-call validation lease, atomic cost ledger, cache, batching, revalidation, failure fallback | 9 |
| Shadow comparison, sampling, canary gates, audit, persistent breaker | 10 |
| Signed preview, immutable selection, lifecycle facade protocol | 11 |
| Immutable V2.1 LangGraph, routing, status projection, V2.0 resume regression | 12 |
| Review revoke/audit endpoints and bounded metadata | 13 |
| Typed frontend union and one-click Integrations cost preview | 14 |
| Review/Timeline/History/Knowledge badges, filters, audit, revoke, no-depth UX | 15 |
| Golden evaluation, PostgreSQL races/restart, full release evidence and docs | 16 |

Tasks 1–3 establish immutable contracts and persistence. Tasks 4–6 establish the only trusted-state mutation boundary and its inverse. Tasks 7–10 add validation and rollout policy without exposing it publicly. Tasks 11–13 integrate V2.1 lifecycle and APIs while retaining V2.0. Tasks 14–15 add the existing-screen UX. Task 16 is the release gate. Every task is an independent review checkpoint and commit.

## Branch and Multi-Agent Integration Strategy

Implementation must follow the repository's shared-contract-first pipeline; do not let parallel workers invent payloads independently:

```text
codex/agent-runtime-contracts
  Tasks 1-2 (including the additive SourceEvent semantic-timestamp contract),
  Task 3A shared runtime/binding/send-fence slice, and Task 4
  merged only after contract/migration/V2.0 compatibility gates
    -> codex/mail-document-agent
       Task 3B Mail/Document and memory-extraction adapter slice, owned by Developer B
       and preserving AgentManifest/AgentRegistry boundaries
    -> codex/rag-orchestrator-agent
       Tasks 5 and 6A revoke/serving/reconciliation contracts, based on the exact
       reviewed 3A + 3B + 4 commits
         -> codex/mail-document-agent
            Task 6B Google connector/ingestion/parser/current-pointer slice, based on
            the exact green Task 6A contract commit and owned by Developer B
         -> codex/rag-orchestrator-agent
            merge the exact green Task 6B commit, run Task 6's integrated gate,
            then execute Tasks 7-13 one green vertical slice/commit at a time
    -> codex/c5-auto-review-ux
       Tasks 14-15, based on the green API contract commit
    -> codex/integration-agent-runtime
       Task 16, merges exact reviewed commits and runs full gates
```

- Use an isolated worktree per implementation branch. Do not implement C.5 on this planning branch.
- Tasks 1–2 and Task 3A are serial shared-contract work. Publish the exact Task 3A contract commit hash before Developer B starts Task 3B. Task 3B changes only the named Mail/Document and memory-extraction adapter files/tests; it may not invent a new payload, import another feature agent directly, or change `AgentManifest`/`AgentRegistry`. Merge the exact green Task 3B commit back before Task 4 and before any Task 5–13 worker starts.
- Task 6 is an explicit ownership handoff, not a broad RAG-branch refactor. Task 6A freezes the revoke, reconciliation, serving, and Review-visibility interfaces on `codex/rag-orchestrator-agent`. Developer B then implements only the named connector/ingestion/parser/current-pointer files and tests as Task 6B on `codex/mail-document-agent`, without importing a feature agent or changing the shared SourceEvent shape frozen in Task 2. The RAG branch merges that exact green commit, runs the complete Task 6 gate, and only then starts Task 7. Any discovered SourceEvent, parser-run, permission, or Review Queue contract change returns to the shared-contract review gate instead of being patched independently on either feature branch.
- The backend and frontend branches may proceed in parallel only after Task 13's public schema/API commit is green. The frontend worker consumes generated/frozen types and may not change backend response shapes.
- Merge each named task commit into the integration branch in order and rerun its focused GREEN command after every merge. Conflict resolution must preserve exact V2.0 snapshots and the frozen V2.1 schema; otherwise stop for human review.
- Human approval is required again for any output schema, permission, cost cap, Review Queue trust boundary, promotion/reaffirmation rule, rollout latch/gate, or duplicate rule change discovered during implementation.
- Do not push branches, open a PR, merge to a user branch, run a paid benchmark, or enable shadow/enforce without the separately stated authorization for that action.

Tasks 2, 10, 12, and 15 contain several coupled sub-slices but are not single free-form assignments. Execute their checkbox steps serially, keep the named RED/GREEN evidence for each step in the implementation log, and stop at the task commit checkpoint before moving to the next dependency. If a worker cannot complete one checkbox in a focused session, split that checkbox in the execution log before editing rather than broadening its ownership.

---

### Task 1: Freeze V2.1 Contracts and Prove V2.0 Is Immutable

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `backend/app/schemas/auto_review.py`
- Modify: `backend/app/schemas/review_workflow.py`
- Modify: `backend/app/core/config.py`
- Create: `backend/app/agent_runtime/auto_review_cost_policy.py`
- Create: `backend/tests/test_auto_review_contracts.py`
- Create: `backend/tests/test_auto_review_cost_policy.py`
- Modify: `backend/tests/test_review_v2_schemas.py`
- Modify: `backend/tests/test_agent_runtime_state.py`
- Modify: `backend/tests/test_review_v2_graph.py`
- Modify: `backend/tests/test_langchain_langgraph_dependency_compat.py`

**Interfaces:**
- Produces the exact constants, strict validator schemas, policy literals, V2.1 dry-run/status models, revoke/audit request/response models, and bounded public audit projection frozen above.
- Keeps `sensitive_input_detected` in the internal policy/audit allowlist only; public item/workflow projections collapse it to the generic Korean human-review copy and never reveal that a source resembled a credential.
- Adds `cost_preview_changed` to the bounded workflow error-code family without adding V2.1 response fields to a V2.0 model dump.
- Adds direct dependencies `tiktoken>=0.12.0,<0.13.0` and `langsmith>=0.8.0,<0.9.0` for the frozen estimators and explicit per-call no-trace boundary (not only current transitive locks), plus settings for configured mode, enforce percentage, launch-token TTL, shared extraction/validation provider timeout/send-window/lease/grace, exact extraction/validation confirmation prices, one exact V2.1 total-cost budget, and fixed caps. Launch tokens reuse the existing agent-runtime fingerprint secret through a dedicated domain/version and bind its key version/material verifier; they do not introduce a second secret. Provider/model snapshot/reasoning/prompt/output-contract/policy/estimator identity comes from the versioned registry constants, not an arbitrary fallback list.
- Adds nullable exact Decimal V2.1 extraction input/output confirmation prices distinct from the legacy float estimate settings. New shadow/enforce readiness requires all selected agents to match one of the five exact registry tuples on OpenAI `gpt-5.4-mini-2026-03-17`, reasoning `none`, 10,000/2,048 token caps, one-candidate envelope, USD 0.75/USD 4.50 prices, and an explicitly authorized extraction provider-safety row. Alias `gpt-5.4-mini`, Azure OpenAI, Gemini, different reasoning, a legacy prompt/output contract, or an unregistered agent is unavailable; it is never repriced with an OpenAI tokenizer or validator price.
- Uses a 600-second launch-token TTL. Validation price settings must exactly equal the immutable `auto-review-cost:v1` Terra USD 2/USD 12 registry entry and extraction price settings must exactly equal the immutable `auto-review-extraction-cost:v1` USD 0.75/USD 4.50 registry entries before `shadow`/`enforce` is ready. Do not accept an arbitrary positive override, alias price, or lower configured value. The fixed maxima are USD 0.083580 extraction, USD 0.097728 validation, USD 0.181308 profile reserve, and USD 0.20 budget.

- [ ] **Step 1: Write strict schema, default-mode, and compatibility tests**

Create `backend/tests/test_auto_review_contracts.py` with tests that assert:

- `test_auto_review_defaults_are_disabled_and_fixed_to_terra_medium`
- `test_v21_validation_output_forbids_extra_fields_and_out_of_range_slots`
- `test_v21_dry_run_exposes_exact_validator_output_contract_and_cost_policy_identity`
- `test_exact_09800_pass_boundary_is_preserved_as_decimal`
- `test_v21_run_requires_launch_token_but_v20_forbids_it`
- `test_public_audit_projection_has_no_internal_identity_or_reason`
- `test_invalid_mode_percentage_price_and_cost_caps_fail_settings_validation`
- `test_tiktoken_is_a_direct_bounded_dependency_and_o200k_encoding_loads`
- `test_langsmith_is_direct_bounded_dependency_for_no_trace_context`
- `test_shared_provider_lease_exceeds_send_window_plus_timeout_plus_commit_grace`
- `test_non_disabled_mode_rejects_local_default_fingerprint_secret`
- `test_non_disabled_readiness_rejects_langchain_global_debug_or_verbose_model`
- `test_non_disabled_readiness_rejects_openai_log_debug_effective_debug_logger_or_unapproved_http_hooks`
- `test_v21_extraction_prices_are_decimal_distinct_from_legacy_float_estimates`
- `test_extraction_registry_contains_exactly_five_snapshot_none_reasoning_routes`
- `test_each_extraction_route_has_exact_agent_prompt_output_contract_and_one_candidate_cap`
- `test_extraction_registry_rejects_alias_azure_gemini_fallback_or_unknown_agent`
- `test_extraction_strict_envelope_rejects_second_candidate_extra_fields_and_wrong_item_type`
- `test_each_extraction_schema_largest_accepted_korean_ascii_fixture_fits_2048_total_output_tokens`
- `test_each_extraction_schema_rejects_individually_valid_fields_when_canonical_envelope_exceeds_2048_tokens`
- `test_v21_budget_status_uses_exact_total_extraction_plus_validation_cost`
- `test_v1_terra_price_registry_is_exact_six_place_two_and_twelve`
- `test_v1_extraction_price_registry_is_exact_six_place_point75_and_four_point5`
- `test_five_extraction_calls_reserve_0083580_and_two_validation_batches_reserve_0097728`
- `test_max_profile_reserve_is_0181308_below_twenty_cent_budget`
- `test_shadow_enforce_rejects_price_config_below_above_or_different_from_registry`
- `test_validation_preview_admission_authorization_and_launch_resolve_one_exact_policy_entry`
- `test_each_extraction_route_preview_admission_authorization_and_launch_resolve_its_exact_registry_entry`
- `test_v21_total_budget_is_positive_and_never_exceeds_twenty_cents`
- `test_v21_rejects_retry_fallback_or_missing_extraction_estimator_route`
- `test_disabled_postgres_c5_bootstrap_rejects_placeholder_or_short_secret`
- `test_disabled_sqlite_smoke_may_use_process_local_placeholder_without_durable_ready_state`
- `test_file_backed_sqlite_placeholder_refuses_every_c5_bound_durable_write`


Extend the V2.0 tests to compare exact key sets, not only values:

```python
assert set(v20_dry_run.model_dump()) == {
    'workflow_name', 'graph_version', 'source_count', 'agent_names',
    'selection_policy_version', 'estimated_input_tokens',
    'estimated_output_tokens', 'estimated_cost_usd', 'budget_limit_usd',
    'budget_status', 'cache_hit', 'requires_explicit_run',
}
assert set(v20_status.model_dump()) == {
    'thread_id', 'status', 'review_item_count', 'review_status_counts',
    'durable', 'graph_version', 'review_resolution_ready',
    'checkpoint_resumable', 'resume_allowed', 'retry_allowed',
    'created_at', 'updated_at', 'error_code', 'resume_error_code',
}
```

Keep the current V2.0 checkpoint eight-key and four-status assertions and add a graph edge snapshot proving the existing V2.0 builder has no auto-review node. Separately assert the exact V2.1 dry-run key set, including `auto_review_validator_output_contract_version` and `auto_review_cost_policy_version`; neither field may be inferred from another version string or omitted as an internal-only detail.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```powershell
uv run --locked pytest backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_cost_policy.py backend/tests/test_review_v2_schemas.py backend/tests/test_agent_runtime_state.py backend/tests/test_review_v2_graph.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

Expected: collection/import failures for the new contracts and assertion failures for missing V2.1 types. The pre-existing V2.0 assertions remain green in the same run.

- [ ] **Step 3: Add the strict contract module and validated settings**

Implement the frozen types in `schemas/auto_review.py` with `ConfigDict(extra='forbid')`, exact `Literal` unions, `Decimal`, bounded strings/lists, and explicit discriminators. Add one shared final model validator/canonicalizer used by all five extraction result classes: fixed-four-place Decimal normalization, JSON-mode safe values, NFC sorted compact JSON, `o200k_base` string token count, UTF-8 fixture/HMAC bytes, exact 2,048 acceptance and 2,049 rejection. Individual field bounds do not bypass this aggregate guard. Keep these existing names unchanged in `schemas/review_workflow.py`:

```python
COMPANY_MEMORY_REVIEW_GRAPH_VERSION = 'company-memory-review-v2.0'
ReviewWorkflowRunRequest
ReviewWorkflowDryRunResponse
ReviewWorkflowStatusResponse
```

Add the exact separately named V2.1 classes from the frozen contract and define `ReviewWorkflowDryRunUnion` and `ReviewWorkflowStatusUnion` as `Annotated` unions discriminated by `graph_version`.

If Pydantic cannot discriminate the existing class because its literal name is reused, introduce a new alias class with the exact same V2.0 fields and assert identical dumps; do not mutate the old class to add optional V2.1 fields.

Add settings with validation equivalent to:

```python
auto_review_mode: Literal['disabled', 'shadow', 'enforce'] = 'disabled'
auto_review_enforce_percentage: Literal[0, 10, 100] = 0
auto_review_launch_confirmation_ttl_seconds: int = Field(
    default=600, ge=60, le=3600
)
auto_review_provider_timeout_seconds: int = Field(
    default=60, ge=10, le=90
)
auto_review_provider_send_start_window_seconds: int = Field(
    default=5, ge=1, le=10
)
auto_review_provider_attempt_lease_seconds: int = Field(
    default=120, ge=60, le=600
)
auto_review_provider_commit_grace_seconds: int = Field(
    default=30, ge=10, le=120
)
auto_review_validator_input_cost_per_1m_tokens: Decimal | None = Field(
    default=None, gt=0
)
auto_review_validator_output_cost_per_1m_tokens: Decimal | None = Field(
    default=None, gt=0
)
auto_review_extraction_input_cost_per_1m_tokens: Decimal | None = Field(
    default=None, gt=0
)
auto_review_extraction_output_cost_per_1m_tokens: Decimal | None = Field(
    default=None, gt=0
)
auto_review_max_total_cost_usd: Decimal = Field(
    default=Decimal('0.20'), gt=0, le=Decimal('0.20')
)
```

Settings validation requires `auto_review_provider_attempt_lease_seconds > auto_review_provider_send_start_window_seconds + auto_review_provider_timeout_seconds + auto_review_provider_commit_grace_seconds`; equality fails closed. Both extraction and validation use these dedicated C.5 settings rather than mutable `agent_llm_timeout_seconds`. At attempt start the database lease expiry becomes `started_at + stored_provider_attempt_lease_seconds`; the one-use in-memory send permit expires after the stored send-start window, the HTTP client uses the stored provider timeout, and recovery starts only after the lease. V2.1 converts no legacy float budget into billing authority: `auto_review_max_total_cost_usd` is exact `Decimal`, is capped by `AUTO_REVIEW_MAX_WORKFLOW_COST_USD`, and applies to the conservative extraction ceiling plus validation upper bound. Changing estimator, tokenizer, token caps, any provider timing value, batch count, prices, or this total budget requires a new purpose-specific cost-policy/preview identity.

Implement `auto_review_cost_policy.py` as the sole immutable authority with frozen `ValidationCostPolicy` and `ExtractionRoutePolicy` dataclasses plus `get_validation_policy(...)` and `get_extraction_route(agent_name, ...)`. The validation key includes cost-policy/provider/model/reasoning; the extraction key includes extraction-cost-policy/agent/provider/model/reasoning/route/prompt/output-contract. Lookups compare the complete tuple and return no nearest/fallback match. Each extraction entry carries the shared 24,000-character, 10,000-input-token, 2,048-total-output-token, one-candidate, one-attempt caps and exact six-place prices. Registry construction asserts its agent set equals `DEFAULT_REVIEW_AGENT_NAMES`, all model snapshots and price/cap identities are identical where required, every prompt/output version is unique to its agent, and the arithmetic constants above recompute exactly with `ROUND_CEILING`.

Add a cross-field settings validator: `disabled` and `shadow` require percentage `0`; `enforce` accepts only `10` or `100`. Arbitrary percentages and every mode/percentage mismatch fail construction. Generic Settings construction may keep the current placeholder only for disabled SQLite/tests, but the PostgreSQL C.5 bootstrap and **every durable keyed write in every mode, including V2.0 while auto review is disabled**, require a non-empty fingerprint key version and a non-placeholder secret of at least 32 UTF-8 bytes. A C.5 PostgreSQL process fails startup before admission if that invariant is absent; it never seeds weak durable rows. Live shadow/enforce readiness additionally requires the OpenAI key and exact equality between configured confirmation values and both immutable registries: Mini USD 0.750000/4.500000 for every selected extraction route and Terra USD 2.000000/12.000000 for validation. Missing, arbitrary, alias-priced, or mismatched values are unavailable. The placeholder can produce only process-local/not-ready SQLite smoke identities and can never sign a launch or reach persisted runtime readiness.

After editing the two direct requirements, run `uv lock`, inspect that the lock diff contains only the intended direct-dependency metadata and resolver consequences, and only then run `uv lock --check`. Do not expect `--check` to create or refresh the lock.

- [ ] **Step 4: Run contract tests and lint GREEN**

Run:

```powershell
uv lock --check
uv run --locked pytest backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_cost_policy.py backend/tests/test_review_v2_schemas.py backend/tests/test_agent_runtime_state.py backend/tests/test_review_v2_graph.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
uv run --locked ruff check backend/app/schemas/auto_review.py backend/app/schemas/review_workflow.py backend/app/core/config.py backend/app/agent_runtime/auto_review_cost_policy.py backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_cost_policy.py backend/tests/test_review_v2_schemas.py backend/tests/test_agent_runtime_state.py backend/tests/test_review_v2_graph.py backend/tests/test_langchain_langgraph_dependency_compat.py
```

Expected: all focused tests and Ruff pass; V2.0 exact-key and topology snapshots are unchanged.

- [ ] **Step 5: Commit the frozen contract slice**

```powershell
git add pyproject.toml uv.lock backend/app/schemas/auto_review.py backend/app/schemas/review_workflow.py backend/app/core/config.py backend/app/agent_runtime/auto_review_cost_policy.py backend/tests/test_auto_review_contracts.py backend/tests/test_auto_review_cost_policy.py backend/tests/test_review_v2_schemas.py backend/tests/test_agent_runtime_state.py backend/tests/test_review_v2_graph.py backend/tests/test_langchain_langgraph_dependency_compat.py
git commit -m "feat: freeze auto review v21 contracts"
```

---

### Task 2: Add C.5 Persistence with Enforced Provenance Ownership

**Files:**
- Create: `backend/app/models/auto_review.py`
- Create: `backend/app/agent_runtime/keyed_mutation_guard.py`
- Create: `backend/app/admin/auto_review_keys.py`
- Modify: `backend/app/models/agent_workflows.py`
- Modify: `backend/app/models/agent_runs.py`
- Modify: `backend/app/models/assistant.py`
- Modify: `backend/app/models/knowledge.py`
- Modify: `backend/app/models/source.py`
- Modify: `backend/app/models/review.py`
- Modify: `backend/app/models/enums.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/connectors/base.py`
- Create: `backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py`
- Create: `backend/tests/test_auto_review_migration.py`
- Modify: `backend/tests/test_agent_runtime_migration.py`
- Modify: `backend/tests/test_db_schema_operations.py`
- Modify: `backend/tests/test_models.py`
- Modify: `backend/tests/test_assistant_models.py`
- Modify: `backend/app/admin/data_reset.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_data_reset.py`
- Create: `backend/tests/test_keyed_mutation_guard.py`
- Create: `backend/tests/test_auto_review_key_bootstrap.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`
- Modify: `backend/tests/test_connector_ingestion_contract.py`

**Interfaces:**
- Adds nullable V2.1 request snapshot columns, exact generator identity columns, relational `ReviewItem.agent_run_id`, nullable server-owned `ReviewItem.candidate_contract_version`, and ReviewItem resolution/revoke columns (including `revoked_by_subject_hmac`, `revoked_by_fingerprint_key_version`, `revoked_by_fingerprint_key_material_verifier`, first-commit `revoke_knowledge_remained_trusted`, and first-commit `revoke_document_count`, never a raw id), `ReviewItemEvidenceRef`, authoritative `AutoReviewExtractionCall`, `AutoReviewValidationCall`, `AutoReviewValidation`, singleton `AutoReviewRuntimeKeyState`, `TrustedKnowledgeFingerprint`, singleton `TrustedKnowledgeFingerprintProjectionState`, `TrustedKnowledgeApprovalLink`, `TrustedKnowledgeEvidenceLink`, immutable `AutoReviewPromotionDecision`, `VectorServingTombstone`, `AutoReviewPostAudit` (auditor-subject HMAC plus its key version/material), immutable `AutoReviewRevocationAssessment`, insert-once/monotonic-remediation `AutoReviewAuditCorrection`, `AutoReviewProviderSafetyState` plus append-only `AutoReviewProviderSafetyEvent`, `AutoReviewRolloutState` plus append-only `AutoReviewRolloutControlEvent`, and immutable `AssistantMessageEvidenceDependency` plus its exact knowledge-evidence children. New C.5-aware V2.0/V2.1 workflow candidates set `candidate_contract_version='c5-v1'`; retained rows remain null. Extraction/validation calls store reserves, one-attempt markers, actual/conservative charge, and actual-overrun flags/costs. A completed extraction call additionally stores an exact singular terminal result contract (`candidate|no_candidate`, count exactly 1 or 0, keyed set HMAC), so legitimate zero output is distinguishable from missing/corrupt children. Provider safety is unique by `(purpose, provider, model, reasoning_effort)` where purpose is `extraction|validation`, and stores the current explicitly authorized estimator/cost-policy/price aggregate plus a persistent purpose-specific cost breaker; every authorization/overrun/clear transition is an immutable event with its own key identity. Rollout state includes a latched `max_authorized_percentage` (`0|10|100`), authorization generation/time, bounded current gate reference, and breaker/counter fields—including `corrected_critical_count`—while every authorization/breaker/generation control mutation is an immutable per-row-keyed event, so crossing a metric threshold or closing a breaker can never erase history or auto-enable enforce.
- Uses revision `7c5a2e9f4b10` with `down_revision='2f6a8b9c0d1e'`.
- Preserves every legacy row and keeps all new ReviewItem/request fields nullable for V2.0 and pre-C.5 data.
- Enforces same-item `auto_validation_id` ownership in the database and models one approval effect to many canonical evidence links without ambiguous effect-kind ordinals.
- Writes one idempotent `AgentRuntimeSchemaVersion(component='auto_review_trust_promotion')` boundary timestamp so the service can distinguish pre-migration unbound human candidates without a client-supplied legacy flag.
- Adds PostgreSQL deferred constraint trigger `trg_review_item_post_c5_evidence_guard` on INSERT and updates of `workflow_thread_id`, `agent_run_id`, or `candidate_contract_version`: while the migration-installed trigger exists, every newly inserted or newly attached workflow-owned ReviewItem must carry `candidate_contract_version='c5-v1'`, a same-workflow `agent_run_id`, and at least one same-workflow `ReviewItemEvidenceRef` by transaction commit. The rule is not conditional on a marker lookup that could be deleted. A child-side deferred UPDATE/DELETE guard revalidates that every C.5 parent retains at least one exact same-workflow ref, and bound ref ownership/source/message-HMAC/key-identity fields are immutable. A companion parent immutability trigger refuses removal/change of ownership fields once a row is C.5-bound, and a separate trigger makes the `auto_review_trust_promotion` schema-boundary marker itself update/delete-immutable. Pre-migration workflow rows are unaffected unless code attempts to reattach/change their identity; non-workflow rows remain exempt while unattached. An allowed empty-schema downgrade drops the guards first and only then removes the marker. This is the database backstop that makes an overlapping old writer, marker-delete attempt, last-child delete, cross-ref rewrite, or insert-null-then-attach bypass fail instead of creating a post-boundary unbound candidate.
- Adds nullable canonical `Document.current_document_version_id` with a same-document composite FK; the legacy `current_version` string remains display-compatible but can never authorize raw-chunk serving. It also separates source signatures into nullable `Source.server_content_signature_schema`, 64-hex `Source.server_content_signature`, and bounded `Source.connector_content_signature`: only the server pair is C.5 authority, while the connector value is non-authoritative evidence.
- Extends the shared additive `SourceEvent` contract with trailing nullable `semantic_timestamp_raw`. V2.0/mock/Slack constructors remain source-compatible through the default, while every supported C.5 Google adapter must later supply the exact upstream Gmail `internalDate`, Drive `modifiedTime`, or Calendar start date/date-time. The operational `timestamp` field can no longer be treated as proof of that upstream value.
- Adds nullable C.5 identity columns to `DocumentParserRun` (`server_content_signature_schema`, `server_content_signature`, `parser_policy_version`, `parser_version`, `chunk_policy_version`) and nullable `DocumentChunk.parser_run_id` for legacy compatibility. A C.5 chunk derives its immutable source signature and parser/chunk policy only through a same-version/same-source parser-run FK; mutable chunk metadata never becomes authority.
- Produces the sole fixed-lock guard API and an idempotent startup/admin key bootstrap before Task 3 writes its first keyed evidence row; later tasks consume this API and never issue their own global advisory-lock SQL.

- [ ] **Step 1: Write model and migration tests before defining rows**

Create tests for fresh upgrade, upgrade from `2f6a8b9c0d1e`, empty-schema downgrade back to that revision, populated-C.5 downgrade refusal, a repeated idempotent upgrade helper path, generated-id cleanup, legacy-row preservation, named constraints, exact indexes, and local data-reset refusal/preservation when C.5 retained rows exist. Assert these relational invariants:

```text
review_item_evidence_refs:
  UNIQUE(review_item_id, workflow_evidence_ref_id)
  UNIQUE(review_item_id, candidate_slot_ordinal)
  FOREIGN KEY(review_item_id, workflow_thread_id)
    -> review_items(id, workflow_thread_id)
  FOREIGN KEY(workflow_evidence_ref_id, workflow_thread_id)
    -> agent_workflow_evidence_refs(id, workflow_thread_id)

agent_workflow_evidence_refs:
  UNIQUE(id, workflow_thread_id)

auto_review_validations:
  UNIQUE(workflow_thread_id, validation_key)
  UNIQUE(review_item_id, id)
  CHECK(status IN ('claimed', 'completed', 'failed'))
  CHECK(estimated_cost_usd >= 0)

auto_review_validation_calls:
  UNIQUE(workflow_thread_id, batch_fingerprint)
  UNIQUE(id, workflow_thread_id)
  CHECK(status IN ('claimed', 'completed', 'failed'))
  CHECK(max_provider_attempts = 1)
  CHECK(provider_attempt_count IN (0, 1))
  CHECK(reserved_input_tokens >= 0 AND reserved_output_tokens >= 0)
  CHECK(reserved_cost_usd >= 0 AND charged_cost_usd >= 0)
  CHECK(charged_input_tokens IS NULL OR charged_input_tokens >= 0)
  CHECK(charged_output_tokens IS NULL OR charged_output_tokens >= 0)
  CHECK(budget_overrun_cost_usd >= 0)
  CHECK(
    (status = 'claimed' AND terminal_at IS NULL
      AND charged_input_tokens IS NULL AND charged_output_tokens IS NULL
      AND charged_cost_usd = 0 AND budget_overrun = false)
    OR
    (status = 'completed' AND provider_attempt_count = 1
      AND attempt_started_at IS NOT NULL AND terminal_at IS NOT NULL
      AND charged_input_tokens IS NOT NULL AND charged_output_tokens IS NOT NULL)
    OR
    (status = 'failed' AND terminal_at IS NOT NULL
      AND ((provider_attempt_count = 0 AND attempt_started_at IS NULL
            AND charged_input_tokens = 0 AND charged_output_tokens = 0
            AND charged_cost_usd = 0 AND budget_overrun = false)
           OR (provider_attempt_count = 1 AND attempt_started_at IS NOT NULL
               AND charged_input_tokens IS NOT NULL
               AND charged_output_tokens IS NOT NULL)))
  )

auto_review_validations:
  FOREIGN KEY(validation_call_id, workflow_thread_id)
    -> auto_review_validation_calls(id, workflow_thread_id)
  FOREIGN KEY(review_item_id, workflow_thread_id)
    -> review_items(id, workflow_thread_id)

review_items:
  UNIQUE(id, workflow_thread_id)
  FOREIGN KEY(agent_run_id, workflow_thread_id)
    -> agent_runs(id, workflow_thread_id)
  FOREIGN KEY(id, auto_validation_id)
    -> auto_review_validations(review_item_id, id)

agent_runs:
  UNIQUE(id, workflow_thread_id)
  UNIQUE(id, workflow_thread_id, agent_name)

auto_review_extraction_calls:
  UNIQUE(agent_run_id)
  UNIQUE(workflow_thread_id, agent_name)
  UNIQUE(workflow_thread_id, agent_name, extraction_plan_hmac)
  FOREIGN KEY(agent_run_id, workflow_thread_id, agent_name)
    -> agent_runs(id, workflow_thread_id, agent_name)
  CHECK(status IN ('claimed', 'completed', 'failed'))
  CHECK(max_provider_attempts = 1)
  CHECK(provider_attempt_count IN (0, 1))
  CHECK(reserved_cost_usd >= 0 AND charged_cost_usd >= 0)
  CHECK(budget_overrun_cost_usd >= 0)
  CHECK(result_candidate_count IS NULL OR result_candidate_count IN (0, 1))
  CHECK(
    (status != 'completed' AND result_kind IS NULL
      AND result_candidate_count IS NULL AND result_candidate_set_hmac IS NULL)
    OR
    (status = 'completed' AND result_kind = 'no_candidate'
      AND result_candidate_count = 0 AND length(result_candidate_set_hmac) = 64)
    OR
    (status = 'completed' AND result_kind = 'candidate'
      AND result_candidate_count = 1 AND length(result_candidate_set_hmac) = 64)
  )

trusted_knowledge_approval_links:
  UNIQUE(id, knowledge_type, knowledge_id)
  UNIQUE(knowledge_type, knowledge_id, review_item_id, promotion_effect_kind)

trusted_knowledge_fingerprints:
  UNIQUE(knowledge_type, knowledge_id)
  INDEX(security_scope_id, knowledge_type, project_scope_hmac,
        normalized_title_bucket_hmac, review_status, permission_level)

auto_review_runtime_key_states:
  UNIQUE(component)
  CHECK(generation >= 1)

auto_review_provider_safety_states:
  UNIQUE(purpose, provider, model, reasoning_effort)
  UNIQUE(id, purpose, provider, model, reasoning_effort)
  CHECK(state_version >= 1)
  CHECK(overrun_count >= 0)
  CHECK((last_event_id IS NULL AND last_event_sequence = 0)
        OR (last_event_id IS NOT NULL AND last_event_sequence > 0))
  FOREIGN KEY(last_event_id, id)
    -> auto_review_provider_safety_events(id, provider_safety_state_id)
auto_review_provider_safety_events:
  UNIQUE(provider_safety_state_id, event_sequence)
  UNIQUE(id, provider_safety_state_id)
  FOREIGN KEY(provider_safety_state_id, purpose, provider, model, reasoning_effort)
    -> auto_review_provider_safety_states(
         id, purpose, provider, model, reasoning_effort
       )
    ON DELETE RESTRICT
  CHECK(event_kind IN ('initial_authorized', 'budget_overrun', 'breaker_cleared'))
  CHECK(length(fingerprint_key_material_verifier) = 64)
  CHECK((event_kind = 'budget_overrun' AND length(call_hmac) = 64
         AND actor_subject_hmac IS NULL)
        OR (event_kind IN ('initial_authorized', 'breaker_cleared')
            AND length(actor_subject_hmac) = 64 AND call_hmac IS NULL))

trusted_knowledge_fingerprint_projection_states:
  UNIQUE(component)
  CHECK(source_active_count >= 0 AND projected_active_count >= 0)

trusted_knowledge_evidence_links:
  UNIQUE(id, approval_link_id)
  UNIQUE(approval_link_id, canonical_source_kind, canonical_source_id,
         canonical_version_or_signature, evidence_hash)

vector_serving_tombstones: UNIQUE(document_id)
auto_review_promotion_decisions:
  UNIQUE(review_item_id)
  UNIQUE(review_item_id, id)
  UNIQUE(security_scope_id, policy_version, promotion_ordinal)
  FOREIGN KEY(security_scope_id, policy_version)
    -> auto_review_rollout_states(security_scope_id, policy_version)
auto_review_post_audits:
  UNIQUE(review_item_id)
  UNIQUE(review_item_id, id)
  UNIQUE(promotion_decision_id)
  FOREIGN KEY(review_item_id, promotion_decision_id)
    -> auto_review_promotion_decisions(review_item_id, id)
  CHECK(audit_reason IS NULL OR length(audit_reason) BETWEEN 1 AND 500)
auto_review_revocation_assessments:
  UNIQUE(id, review_item_id)
  UNIQUE(review_item_id)
  FOREIGN KEY(review_item_id) -> review_items(id)
  CHECK(reason_code IN ('business_withdrawal', 'incorrect_content',
        'permission_violation', 'wrong_source_version', 'policy_violation'))
  CHECK(length(actor_subject_hmac) = 64)
  NOT NULL(actor_fingerprint_key_version,
           actor_fingerprint_key_material_verifier, created_at)
auto_review_audit_corrections:
  UNIQUE(post_audit_id)
  UNIQUE(review_item_id)
  UNIQUE(assessment_id)
  FOREIGN KEY(review_item_id, post_audit_id)
    -> auto_review_post_audits(review_item_id, id)
  FOREIGN KEY(assessment_id, review_item_id)
    -> auto_review_revocation_assessments(id, review_item_id)
  CHECK(effective_outcome IN ('incorrect', 'permission_violation',
        'source_version_violation', 'policy_violation'))
  CHECK(status IN ('remediation_required', 'completed'))
  CHECK((status = 'remediation_required'
         AND system_resolution_code IN ('revoke_pending', 'revoke_failed')
         AND completed_at IS NULL)
        OR (status = 'completed' AND system_resolution_code = 'revoked'
            AND completed_at IS NOT NULL))
  CHECK(length(actor_subject_hmac) = 64)
  NOT NULL(actor_fingerprint_key_version,
           actor_fingerprint_key_material_verifier, created_at)
auto_review_rollout_states:
  UNIQUE(security_scope_id, policy_version)
  UNIQUE(id, security_scope_id, policy_version)
  FOREIGN KEY(last_event_id, id)
    -> auto_review_rollout_control_events(id, rollout_state_id)
  CHECK((last_event_id IS NULL AND last_event_sequence = 0)
        OR (last_event_id IS NOT NULL AND last_event_sequence > 0))
  CHECK(confirmed_mandatory_audit_count >= 0)
  CHECK(pending_mandatory_audit_count >= 0)
  CHECK(invalidated_before_audit_count >= 0)
  CHECK(corrected_critical_count >= 0)
  CHECK(confirmed_mandatory_audit_count <= 50)
  CHECK(pending_mandatory_audit_count <= 50)
  CHECK(confirmed_mandatory_audit_count + pending_mandatory_audit_count <= 50)
auto_review_rollout_control_events:
  UNIQUE(rollout_state_id, event_sequence)
  UNIQUE(id, rollout_state_id)
  FOREIGN KEY(rollout_state_id, security_scope_id, policy_version)
    -> auto_review_rollout_states(id, security_scope_id, policy_version)
    ON DELETE RESTRICT
  CHECK(event_kind IN ('percentage_authorized', 'breaker_opened',
        'breaker_closed', 'generation_invalidated'))
  CHECK(length(actor_subject_hmac) = 64)
  CHECK(length(fingerprint_key_material_verifier) = 64)

document_versions:
  UNIQUE(id, document_id)

documents:
  FOREIGN KEY(current_document_version_id, id)
    -> document_versions(id, document_id)

sources:
  CHECK((server_content_signature IS NULL AND server_content_signature_schema IS NULL)
        OR (server_content_signature_schema = 'server-source-content:v1'
            AND length(server_content_signature) = 64))

document_parser_runs:
  UNIQUE(id, document_version_id, source_id)
  CHECK((server_content_signature IS NULL
         AND server_content_signature_schema IS NULL
         AND parser_policy_version IS NULL
         AND parser_version IS NULL
         AND chunk_policy_version IS NULL)
        OR (server_content_signature_schema = 'server-source-content:v1'
            AND length(server_content_signature) = 64
            AND parser_policy_version IS NOT NULL
            AND parser_version IS NOT NULL
            AND chunk_policy_version IS NOT NULL))

document_chunks:
  UNIQUE(id, version_id, source_id, parser_run_id)
  UNIQUE(parser_run_id, chunk_index)
  FOREIGN KEY(parser_run_id, version_id, source_id)
    -> document_parser_runs(id, document_version_id, source_id)

assistant_messages:
  CHECK(evidence_contract_version IS NULL OR
        evidence_contract_version IN ('none-v1', 'assistant-evidence:v1'))
  CHECK(serving_dependency_count IS NULL OR serving_dependency_count >= 0)

assistant_message_evidence_dependencies:
  UNIQUE(assistant_message_id, candidate_ordinal)
  UNIQUE(assistant_message_id, serving_document_id)
  UNIQUE(id, assistant_message_id, approval_link_id)
  FOREIGN KEY(assistant_message_id) -> assistant_messages(id) ON DELETE RESTRICT
  CHECK(dependency_kind IN ('raw_chunk', 'trusted_knowledge'))
  CHECK(length(serving_content_hash) = 64)
  CHECK(permission_level IN ('public', 'internal', 'restricted'))
  CHECK(
    (dependency_kind = 'raw_chunk'
      AND document_chunk_id IS NOT NULL AND document_version_id IS NOT NULL
      AND source_id IS NOT NULL AND parser_run_id IS NOT NULL
      AND server_content_signature_schema = 'server-source-content:v1'
      AND length(server_content_signature) = 64
      AND knowledge_type IS NULL AND knowledge_id IS NULL
      AND approval_link_id IS NULL AND legacy_human_base = false)
    OR
    (dependency_kind = 'trusted_knowledge'
      AND document_chunk_id IS NULL AND document_version_id IS NULL
      AND source_id IS NULL AND parser_run_id IS NULL
      AND server_content_signature_schema IS NULL
      AND server_content_signature IS NULL
      AND knowledge_type IS NOT NULL AND knowledge_id IS NOT NULL
      AND ((approval_link_id IS NOT NULL AND legacy_human_base = false)
           OR (approval_link_id IS NULL AND legacy_human_base = true)))
  )
  FOREIGN KEY(document_chunk_id, document_version_id, source_id, parser_run_id)
    -> document_chunks(id, version_id, source_id, parser_run_id)
  FOREIGN KEY(approval_link_id, knowledge_type, knowledge_id)
    -> trusted_knowledge_approval_links(id, knowledge_type, knowledge_id)

assistant_message_knowledge_evidence_refs:
  UNIQUE(dependency_id, trusted_knowledge_evidence_link_id)
  FOREIGN KEY(dependency_id, assistant_message_id, approval_link_id)
    -> assistant_message_evidence_dependencies(
         id, assistant_message_id, approval_link_id)
  FOREIGN KEY(trusted_knowledge_evidence_link_id, approval_link_id)
    -> trusted_knowledge_evidence_links(id, approval_link_id)
```

`candidate_slot_ordinal` is the per-ReviewItem evidence-ref ordinal `1..N`; each row for the same ReviewItem receives a different ordinal. The migration does not guess bindings for already-existing candidates from broad packet data. `TrustedKnowledgeFingerprint.security_scope_id` is nullable only for legacy rows whose scope cannot be reconstructed; those rows are collision-only and can never be reused. `AutoReviewRuntimeKeyState` stores only key version plus domain-separated key-material verifier HMAC, generation, ready flag, and timestamp—never the secret. `TrustedKnowledgeFingerprintProjectionState` stores projection schema/key version/material verifier, generation, ready flag, source/projected active counts, separate 64-hex `source_checksum`/`projected_checksum`, `rebuild_required`, and completion time. Add check constraints for `max_authorized_percentage IN (0, 10, 100)` and non-negative generations/counters, including `confirmed_mandatory_audit_count`, `pending_mandatory_audit_count`, `invalidated_before_audit_count`, and `corrected_critical_count` plus the exact at-most-50 slot constraints above; authorization/revoke/correction actors are stored only as purpose-domain keyed subject HMACs, never raw email or subject id. `AutoReviewPostAudit(status='completed', outcome=NULL)` is valid only with immutable `system_resolution_code='source_invalidated_before_audit'`; the code and `audit_reason` are null for pending and system invalidation. A human-completed confirmed/critical audit requires a server-normalized UTF-8 NFC `audit_reason` of 1–500 characters after outer-whitespace trim and stores it immutably with the outcome/auditor key identity. The reason is an access-controlled audit artifact only: it is never returned by Review/public APIs, copied to `AuditLog`, checkpoints, prompts, or normal logs, and raw source/model output is not accepted as a substitute. Source invalidation decrements one pending mandatory slot when applicable and increments only the invalidation counter; a human-confirmed mandatory audit decrements pending and increments confirmed. Every transition is exactly once under rollout CAS. No other null-outcome completion is allowed. A later correction never rewrites the confirmed audit or its reason: its immutable child correction increments `corrected_critical_count`, opens the breaker, and makes the derived effective outcome critical. That counter is never decremented or reset on the same scope/policy row; returning to enforce after such a correction requires a separately reviewed new policy version and fresh rollout row/gates.

`AutoReviewExtractionCall.result_candidate_set_hmac` uses domain `auto-review-extraction-result-set:v1` over canonical compact JSON containing either exactly one `(candidate_key, evidence_version_hash)` pair or an empty list. `no_candidate` signs the same schema with explicit `items=[]`; it is not inferred from the absence of ReviewItems. `candidate` signs `items=[<exactly one pair>]`; a second pair is corruption. Completed-call replay re-derives the set from same-workflow/same-AgentRun children and fails closed on any missing, extra, corrupt, or mismatched child. Claimed/failed calls keep all three result fields null.

Validation call billing/terminal columns follow the database checks above and a named deferred parent/child terminal guard: a claimed call has only claimed children and no terminal charge; a completed call has 1–4 same-workflow completed child validations and authoritative charged usage; a failed call has only failed children and no completed projection. Attempt-zero failure is exactly zero-charge, while attempt-one completion/failure has an attempt marker and actual-or-conservative authoritative charge. Direct child mutation, a completed call with zero children, mixed child status, negative usage/cost, or a claimed parent carrying a terminal child is rejected at commit. Model/migration and store tests exercise the guard instead of trusting service-only SUM invariants.

`AutoReviewRevocationAssessment` stores exactly `id`, unique `review_item_id`, strict reason code, actor subject HMAC with its own key version/material verifier, and created time. It is the one authorized terminal revoke assessment, not an attempted-action ledger: the normal business path inserts it only after the audit/permission gate passes and in the exact revoke transaction, while the quality coordinator inserts it only after ordered authorization checks and immediately before its breaker/quarantine transition. A gate-rejected request writes no assessment, so it cannot consume the unique row. Its reason/actor/key/item fields are update/delete-immutable; a later/concurrent request with the same reason may replay/resume, while a different reason returns bounded `revoke_reason_conflict` and cannot replace or append an assessment. `AutoReviewAuditCorrection` exists only for a previously confirmed audit and stores its same-item audit and assessment ownership, mapped effective critical outcome, actor HMAC with its own key identities, insert time, and the monotonic remediation state above. A database guard allows only `revoke_pending -> revoke_failed` or `revoke_pending|revoke_failed -> completed/revoked`; outcome/actor/ownership never change and delete is refused. Recovery scans the bounded union of `AutoReviewPostAudit(status='remediation_required')` and `AutoReviewAuditCorrection(status='remediation_required')`, then locks each through the normal global order. Replays verify the same immutable assessment before changing only correction remediation state.

`AutoReviewProviderSafetyEvent` and `AutoReviewRolloutControlEvent` are insert-only audit ledgers, not mutable actor slots on the aggregate state row. Each event stores its own domain-separated actor or call HMAC, fingerprint key version/material verifier, bounded reason/gate reference where applicable, exact prior/new state version and control values, sequence, and time. Provider events additionally bind purpose/provider/model/reasoning effort plus estimator/cost-policy/price snapshot; rollout events bind scope/policy plus prior/new latch, breaker, control epoch, and authorization generation. Database guards reject update/delete, sequence gaps/reuse, wrong-parent values, parent deletion, an actor on an overrun, or a call HMAC on an operator event. Aggregate `last_event_id` is nullable only with sequence zero; otherwise both are present and positive. Initial provider authorization inserts its parent with null/zero, appends sequence 1, and updates the backpointer in one transaction. A new rollout sentinel may remain null/zero until its first control mutation; that mutation performs the same event+backpointer CAS. Event-to-parent FKs are immediate `ON DELETE RESTRICT`; the cyclic aggregate backpointer is named, `use_alter=True`, and deferrable/created after both tables so migration order is executable. The mutable state rows store only current aggregate values and last event sequence/id; a transition updates aggregate plus inserts its exact event atomically under CAS. Key rotation never re-HMACs or rewrites old events; later events carry the new key identity, so multi-generation attribution remains verifiable without pretending one shared key tuple describes all history.

`Document.current_version` remains the legacy display label and is never authoritative because multiple rows may share `v1`. The new nullable `current_document_version_id` uses the same-document composite FK above. `server_content_signature_schema='server-source-content:v1'` plus `server_content_signature` is the only signature pair that a new C.5 resolver, parser run, chunk, evidence ref, current-version pointer, or serving predicate may bind. `connector_content_signature` copies a connector-supplied version/signature only as evidence and is never compared to declare content unchanged. Migration may copy legacy `raw_metadata['content_signature']` only into the connector-evidence field; it must never relabel that value as a server signature. A version-specific legacy helper may preserve a retained V2.0 identity, but that identity can never authorize C.5 reuse, automatic trust, current-version selection, or an ingestion skip. C.5 parser-run signature/policy fields are insert-immutable; every C.5 `DocumentChunk` must have a `parser_run_id`, and the same-version/same-source composite FK plus unique `(parser_run_id, chunk_index)` prevents a chunk from claiming a different run through JSON metadata. Legacy runs/chunks may keep all new fields null but remain C.5-ineligible.

`AssistantMessage.evidence_contract_version` and `serving_dependency_count` are nullable only for retained pre-C.5 rows. Every new user or non-evidence assistant/system message writes `none-v1` with count zero; every new RAG answer writes `assistant-evidence:v1` and its complete exact dependency set in the same transaction. A deferred parent/child guard rejects a post-marker null contract, requires the stored count to equal the number of immutable dependency rows, requires at least one row for `assistant-evidence:v1`, and requires zero rows for `none-v1`. Raw-chunk dependencies bind the exact current chunk/version/parser-run/source signature. Trusted-knowledge dependencies bind the exact target plus one deterministic active approval effect and all of that effect's evidence-link children; the only linkless variant is a relationally proven pre-C.5 legacy-human base, whose target identity, content hash, and already-stored permission are frozen. Approval/evidence-link composite target pairs are made unique for these FKs. The message's raw answer/source/citation fields, contract marker/count, dependencies, and dependency children become update/delete-immutable after commit. A retained evidence-bearing assistant row with no complete C.5 binding remains audit storage only and can never be replayed into a public response, summary, context prompt, or email draft.

Migration/backfill sets `current_document_version_id` only when exactly one DocumentVersion is unambiguous, or when exactly one parser run already carries a verified server signature matching the Source server signature. Multiple/no matches remain null and raw-chunk serving fails closed until the bounded deterministic Task 6 repair/re-sync path succeeds—never `MAX(id)`, label equality, connector signature, or timestamp guessing. Every C.5 content ingestion writes the Source server signature, parser/chunk server signature, new exact DocumentVersion, and same-document pointer in one ordered source/document transaction.

Model/migration tests prove duplicate `current_version='v1'` rows with different parser content signatures cannot be selected by label, connector signature, timestamp, or `MAX(id)`, only a unique verified server-signature and current parser/chunk-policy match is backfilled, legacy connector-only signatures and legacy chunks without a same-version/same-source parser-run FK remain C.5-ineligible, and C.5 ingestion atomically advances the same-document pointer. They also freeze the additive SourceEvent field/default, mandatory/correction audit counters, extraction terminal-result checks, strict revocation-reason enum, assessment/correction same-item FKs, per-row actor key identities, immutable fields, allowed monotonic correction transitions, recovery scan state, append-only provider/rollout event ownership/sequence/kind/key-generation history, atomic state+event CAS, and reject any `completed + outcome=NULL` audit without the one allowlisted source-invalidation code. Assistant model tests reject cross-message/cross-effect dependency refs, missing or count-mismatched dependency children, a raw dependency that mixes source/version/parser identities, an explicit knowledge dependency without its complete same-effect evidence children, deletion/rewrite of a committed dependency, and a new evidence-bearing answer mislabeled `none-v1`.

`AutoReviewPostAudit` does not duplicate scope/policy; it derives both only through its immutable `AutoReviewPromotionDecision`, whose composite FK to the rollout state fixes counter/breaker ownership. Add negative PostgreSQL-capable model tests that create validation B for ReviewItem B and prove ReviewItem A cannot reference B's validation, that a call or ReviewItem from workflow B cannot own a validation child in workflow A, that an extraction call cannot name an AgentRun from another workflow or a different agent in the same workflow, that a candidate/evidence-ref pair from different workflows cannot be inserted, that a promotion decision cannot name a different rollout scope/policy, and that Audit A cannot reference PromotionDecision B. Add a same-content/two-workflow/two-scope case proving the workflow-bound validation keys do not collide or cross-replay. A PostgreSQL cutover test first commits migration/marker, then simulates a still-running old binary starting a new old-style insert transaction and proves the deferred trigger rejects its unversioned/no-binding row; separately prove migration waits for a pre-existing writer transaction rather than manufacturing an impossible DDL interleaving. Test that insert-as-non-workflow then attach, direct boundary-marker UPDATE/DELETE, C.5 row-marker removal, workflow reassignment, and AgentRun reassignment cannot bypass the guards. A C.5 writer that inserts the item plus same-workflow refs in one transaction succeeds, while a row committed before migration and non-workflow rows remain valid.

In `test_keyed_mutation_guard.py` freeze the exact constants `1066041229503628369` and `-2972884933094306491`, shared/exclusive SQL, runtime row lock mode, and global acquisition order. In `test_auto_review_key_bootstrap.py` prove first initialization, idempotent replay, table-not-yet-present startup no-op, SQLite disabled-mode compatibility, and refusal when a missing/mismatched runtime row coexists with any retained C.5 keyed artifact.

- [ ] **Step 2: Run migration/model tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_migration.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_schema_operations.py backend/tests/test_models.py backend/tests/test_assistant_models.py backend/tests/test_data_reset.py backend/tests/test_keyed_mutation_guard.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_connector_ingestion_contract.py -q
```

Expected: missing model/revision/table/column failures. Existing runtime-foundation migration tests must continue to validate the old revision while treating the new revision as head.

- [ ] **Step 3: Define exact columns and indexes**

Use bounded storage types rather than unbounded text for identifiers/codes:

```text
HMAC/version keys       String(64)
provider/model identity String(120)
status/reason codes     String(32) or bounded String(64)
claim/code lists        MutableList(JSON)
claim results           MutableList(JSON)
score                    Numeric(5, 4)
USD                      Numeric(12, 6)
percentage/ordinal      Integer
state_version/counters  Integer
timestamps              DateTime(timezone=True)
cache/breaker flags     Boolean
```

Map immutable generation identity explicitly: new nullable `AgentRun.generation_provider`, `generation_reasoning_effort`, `generation_route_version`, and `generation_output_contract_version` columns join the existing exact `model_name` and `prompt_version` snapshots. Never parse provider identity or reasoning from a fallback route string or rely on mutable JSON metadata for eligibility. Legacy null identity is valid for V2.0 human review and always auto-ineligible; V2.1 accepts only the registry-owned `openai/gpt-5.4-mini-2026-03-17/none` tuple before a call.

Store bounded policy reason codes and field results as JSON primitives only. Do not add raw evidence, snippet, URL, rationale, prompt text, provider exception, or credential columns.

Freeze `AutoReviewProviderSafetyState` as the current aggregate: non-null `purpose: extraction|validation`, provider/model/reasoning effort, optimistic `state_version`, authorized cost-policy version, token estimator/encoding, reply-priming/framing-safety constants, exact six-place input/output prices, `breaker_open`, bounded current breaker reason, non-negative `overrun_count`, nullable actual last-overrun cost/time, bounded current regression-gate reference, authorization/clear timestamps, and `last_event_sequence/id`. It stores no mutable shared actor/call HMAC key tuple. Per-call caps/timing/prompt live in the immutable extraction/validation call, not this shared price/framing authority. A row exists only after explicit initial purpose/provider/model/reasoning policy authorization plus its `initial_authorized` event in the same transaction. Breaker-open requires a `budget_overrun` event carrying the call HMAC and that event's own key identities; clear requires a different authorized cost-policy version and a `breaker_cleared` operator event, atomically replacing the exact estimator/price aggregate under CAS without deleting history.

Freeze `AutoReviewExtractionCall` as one row per V2.1 live `AgentRun`: composite same-workflow **and same-agent** ownership, extraction-plan HMAC, provider/model/reasoning/route/prompt/output-contract, immutable extraction-registry and cost-policy versions, `max_candidates_per_agent=1`, estimator/encoding/allowances, prepared-content HMAC and character/framed-input/total-output caps, exact Decimal input/output prices, cost-policy/key identities, shared provider timeout/send-start-window/attempt-lease/commit-grace, signed workflow extraction/total ceilings, status, lease owner/expiry, `max_provider_attempts=1`, attempt count/time, reserved tokens/cost, actual-or-conservative charged tokens/cost, overrun flag/cost, and timestamps. It stores no prompt/source/output or send permit. The V2.1-only AgentRun lifecycle is `claimed -> complete|failed`; only the repository's existing exact success value `complete` is cacheable, while `completed` belongs only to the extraction-call ledger. Existing V2.0 AgentRun status/API behavior remains unchanged. `AgentRun.input_tokens/output_tokens/estimated_cost_usd` become observational mirrors for V2.1; the Numeric call ledger is billing/budget authority.

Freeze `AutoReviewRolloutState.state_version >= 0` for all optimistic mutations and separate `control_epoch >= 0` for launch-affecting latch/authorization/breaker changes only. Metric/audit counters increment `state_version` but not `control_epoch`; authorization, rollout breaker open/close, or generation invalidation increment both. The signed preview binds `control_epoch`, never noisy counters/state version.

Every durable keyed C.5 row stores the non-secret `fingerprint_key_version` and `fingerprint_key_material_verifier`: V2.1 workflow request, ReviewItem evidence binding, extraction/validation call and child, trusted fingerprint, approval/evidence link, promotion selection/audit row, and launch-confirmation identity. The material verifier is a domain-separated HMAC of a fixed marker, not the secret. Add token-estimator version/encoding, reply-priming/framing-safety constants, validation per-batch token caps, maximum validation batches/candidates, shared extraction/validation provider timeout/send-start-window/attempt-lease/commit-grace seconds, exact six-place validator input/output price snapshots, `cost_policy_version`, validator provider-safety state version, aggregate extraction-plan-set HMAC, sorted extraction provider-safety snapshot-set HMAC, exact 24,000-character/10,000-framed-input-token/2,048-total-output-token extraction hard caps plus the zero-or-one-candidate cap and per-agent effective caps, the exact extraction provider/model snapshot/reasoning/route identity, exact six-place V2.1 total-budget limit, rollout control epoch, `max_provider_attempts=1`, `authorized_percentage_at_launch`, and `rollout_authorization_generation` to the immutable V2.1 workflow request snapshot. Existing V2.0 request identity/columns remain compatible; new nullable columns are populated only by the version-specific builder where required. Add nullable `ReviewItem.agent_run_id` plus a named same-workflow composite FK to an `AgentRun(id, workflow_thread_id)` unique target. Give `ReviewItemEvidenceRef` its own non-null `workflow_thread_id` and named composite FKs to same-workflow ReviewItem and `AgentWorkflowEvidenceRef` unique targets; post-migration eligibility never trusts mutable payload ownership or a service-only assertion.

Define the composite FK with a stable name such as `fk_review_items_auto_validation_same_item`, `use_alter=True`, and a deferrable PostgreSQL constraint. The migration must use a batch-safe path for SQLite schema tests and explicit constraint/index names for downgrade. No table should depend on autogenerate-only names.

- [ ] **Step 4: Implement the additive migration**

Follow the repository's inspect-before-create/drop helpers so a fresh database, pre-C.5 database, and test recreation are deterministic. The upgrade order is:

1. add V2.1 request columns, exact AgentRun generator identity snapshots, AssistantMessage evidence-contract/count columns, the three separated Source signature columns, the nullable same-document current-version pointer/target unique key, parser-run signature/policy columns, and the nullable chunk-to-parser-run key;
2. add the AgentRun/ReviewItem/AgentWorkflowEvidenceRef same-workflow unique targets, the parser-run same-version/same-source target, chunk composite FK/unique index, and nullable ReviewItem generator/resolution/revoke columns;
3. create runtime-key/provider-safety parents, then append-only provider-safety events, projection-state parent, trusted fingerprint projection, and candidate evidence refs, then their named same-workflow FKs;
4. create `AutoReviewExtractionCall` against its same-workflow AgentRun, then `AutoReviewValidationCall` and `AutoReviewValidation`, then add validation child FKs and the ReviewItem same-item validation FK;
5. create approval parent links, then evidence child links and their named FKs, then AssistantMessage dependency parents and their exact knowledge-evidence children after every referenced target exists;
6. create rollout state, then append-only rollout-control events, immutable promotion-decision, selected audit, revocation-assessment, audit-correction, and tombstone tables in parent-before-child order;
7. create all remaining named indexes/constraints and PostgreSQL deferred triggers `trg_review_item_post_c5_evidence_guard` and `trg_assistant_message_evidence_guard` (the SQLite migration test uses the equivalent service invariants because SQLite cannot provide the same commit-time child-row checks);
8. insert the C.5 schema-boundary marker only after the complete schema and guard are present.

After migration/model code, implement `AutoReviewKeyBootstrapService.ensure_initialized()` outside Alembic: migration never reads a secret or environment. On PostgreSQL it first requires the strong non-placeholder key invariant regardless of mode, takes the fixed generation barrier exclusively, locks/creates the singleton, and only when every C.5 keyed table/request field is empty writes generation 1 with the configured key version/material verifier and `ready=true`; it also creates matching projection state as `ready=false, rebuild_required=true`. Exact replay is idempotent. Any existing identity mismatch or missing runtime row with retained keyed data is a bounded refusal—never a re-HMAC. Lifespan calls this service before constructing drafting services when the C.5 schema exists; disabled SQLite uses the process-local guard and creates local not-ready state. Full backfill/rotation remains Task 5/admin control-plane work.

The migration marker is also a writer cutover boundary, not only a legacy classifier. The release runbook must execute `AUTO_REVIEW_MODE=disabled -> configure the strong active fingerprint key/version -> stop admission and drain/stop every old V2.0 writer -> migrate -> deploy only the C.5-aware code -> bootstrap -> re-sync supported Gmail/Drive/Calendar sources through the new server-signature/parser-policy boundary -> run pointer status/repair, source reconciliation, projection rebuild, and incremental reindex -> require stale/requires_resync/ambiguous counts all zero -> resume V2.0 admission`; it may not use a rolling old/new writer overlap across the marker or roll back to the old binary after the marker. Slack is not re-created during this cutover: unavailable/legacy Slack rows remain explicitly C.5-ineligible and fail closed. The new binary checks schema capability at startup, while the named PostgreSQL deferred trigger—not an impossible old-binary self-check—rejects any accidental post-marker old-writer commit. This operational drain is required for migration cutover; the generation barrier remains the separate database guarantee for later key rotation.

Operational rollback is configuration-based (`disabled`), not schema downgrade. The downgrade may reverse an otherwise empty/test schema and remove only the C.5 boundary marker. Its retained-state predicate must refuse when **any** new C.5 table has a row (including workflow/assistant evidence refs and runtime/projection state), any thread has graph version V2.1, any workflow request has a non-null C.5 snapshot/ceiling, any ReviewItem has a C.5 generator/resolution/validation/revoke field or `revoked` status, any AgentRun has a new generation-identity field, any AssistantMessage has a non-null evidence-contract/count field, any Source has a non-null server/connector signature column, any Document has a non-null `current_document_version_id`, any parser run has a C.5 signature/policy identity, or any chunk has `parser_run_id`. Add one refusal test per category; validation/provenance-only checks are insufficient. It never destroys retained assistant audit dependencies, signature authority, version pointers, parser lineage, or audit history. Update the local connector-derived reset service to report the same complete predicate and refuse confirmed row deletion when any C.5 retained state exists, directing developers to recreate only an explicitly disposable local database instead of deleting audit rows out of order.

- [ ] **Step 5: Run migration/model tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_migration.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_schema_operations.py backend/tests/test_models.py backend/tests/test_assistant_models.py backend/tests/test_data_reset.py backend/tests/test_keyed_mutation_guard.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_connector_ingestion_contract.py -q
uv run --locked ruff check backend/app/models backend/app/connectors/base.py backend/app/agent_runtime/keyed_mutation_guard.py backend/app/admin/auto_review_keys.py backend/app/admin/data_reset.py backend/app/main.py backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py backend/tests/test_auto_review_migration.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_schema_operations.py backend/tests/test_models.py backend/tests/test_assistant_models.py backend/tests/test_data_reset.py backend/tests/test_keyed_mutation_guard.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_connector_ingestion_contract.py
```

Expected: fresh/upgrade/downgrade/model cases pass with legacy data preserved and no raw-content column.

- [ ] **Step 6: Commit the persistence slice**

```powershell
git add backend/app/models backend/app/connectors/base.py backend/app/agent_runtime/keyed_mutation_guard.py backend/app/admin/auto_review_keys.py backend/app/admin/data_reset.py backend/app/main.py backend/migrations/versions/7c5a2e9f4b10_add_auto_review_trust_promotion.py backend/tests/test_auto_review_migration.py backend/tests/test_agent_runtime_migration.py backend/tests/test_db_schema_operations.py backend/tests/test_models.py backend/tests/test_assistant_models.py backend/tests/test_data_reset.py backend/tests/test_keyed_mutation_guard.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_connector_ingestion_contract.py
git commit -m "feat: persist auto review trust state"
```

---

### Task 3: Persist Immutable V2.1 Request Data and Version-Neutral Candidate Evidence Bindings

**Task 3A shared-contract files (`codex/agent-runtime-contracts`):**
- Modify: `backend/app/agent_runtime/review_v2_preflight.py`
- Modify: `backend/app/agent_runtime/review_v2_drafting.py`
- Modify: `backend/app/agent_runtime/contracts.py`
- Modify: `backend/app/agent_runtime/review_v2_agents.py`
- Modify: `backend/app/agent_runtime/model_router.py`
- Create: `backend/app/agent_runtime/review_v21_extraction.py`
- Create: `backend/app/agent_runtime/auto_review_input_safety.py`
- Create: `backend/app/agent_runtime/provider_send_fence.py`
- Modify: `backend/app/models/agent_workflows.py`
- Modify: `backend/app/models/agent_runs.py`
- Create: `backend/tests/test_provider_send_fence.py`
- Create: `backend/tests/test_review_v21_preflight.py`
- Create: `backend/tests/test_review_v21_drafting.py`
- Create: `backend/tests/test_review_v21_extraction.py`
- Modify: `backend/tests/test_review_v2_preflight.py`
- Modify: `backend/tests/test_review_v2_drafting.py`
- Modify: `backend/tests/test_review_v2_agents.py`

**Task 3B Developer B agent-adapter files (`codex/mail-document-agent`):**
- Modify: `backend/app/agents/mail_document_agent/agent.py`
- Modify: `backend/app/agents/mail_document_agent/llm.py`
- Modify: `backend/app/agents/memory_extraction_agent/agent.py`
- Modify: `backend/app/agents/memory_extraction_agent/langchain_adapter.py`
- Modify: `backend/tests/test_mail_document_agent.py`
- Modify: `backend/tests/test_memory_extraction_langchain_adapter.py`

**Interfaces:**
- Leaves `PreparedReviewRequest` and the current V2.0 identity/HMAC output unchanged.
- Adds a V2.1 prepared request/config carrying stored `shadow|enforce`, exact validator provider/model/reasoning, prompt/output-contract/policy/cost-policy/token-estimator versions, fingerprint key version/material verifier, provider-safety state version, rollout control epoch, reply-priming/framing-safety constants, max provider attempts/batches/candidates, shared provider timeout/send-window/lease/grace seconds, exact Decimal input/output price snapshots, per-batch token caps, requested percentage, authorized percentage at launch, rollout authorization generation, exact V2.1 total-budget limit, and confirmed extraction/validation/total ceilings.
- Builds a sorted immutable V2.1 extraction-plan set for a non-empty subset of the exact five-entry registry. Every live plan binds its agent to provider `openai`, model snapshot `gpt-5.4-mini-2026-03-17`, reasoning `none`, route `auto-review-extraction-route:v1`, the registered prompt/output schema, one-candidate cap, exact rendered-input HMAC/framed tokens, 2,048 total-output cap, exact USD 0.75/USD 4.50 prices, extraction estimator/cost-policy, provider-safety state version, shared provider timeout/send-start-window/attempt-lease/commit-grace, and `max_provider_attempts=1`; the canonical selected-agent set, aggregate plan-set HMAC, and full-cap summed extraction reserve enter the workflow identity and launch token.
- Writes relational `ReviewItem.agent_run_id` and `ReviewItemEvidenceRef` rows in the same transaction as every newly created V2.0 or V2.1 ReviewItem, using the exact same-workflow `AgentRun` and exact `AgentWorkflowEvidenceRef` ids plus one keyed aggregate message-set fingerprint per workflow evidence ref. These bindings are version-neutral provenance infrastructure; only V2.1 graph/version/policy eligibility can invoke auto review, so a V2.0 candidate remains auto-review-ineligible.
- Extends `AgentRunResult` additively with nullable provider/reasoning/route/output-contract identities. V2.0 adapters may report the provider that actually answered after their legacy fallback; V2.1 never falls back and must report the exact registry tuple. Persist exact bounded columns rather than parsing a route or trusting mutable metadata. Missing/wrong identity or the same exact provider/model as the validator is later forced human.
- Owns the shared `FencedProviderSendPermit`/`ProviderAttemptGrant` boundary used by both extraction and validation. A permit is an opaque process-local, one-use capability with only attempt/deadline metadata; it cannot be pickled, copied, converted by Pydantic, JSON-encoded, logged, or reconstructed after restart. `consume_at_dispatch()` uses a monotonic clock and succeeds exactly once before its deadline. `FencedOpenAITransport` consumes it immediately before the one underlying HTTP dispatch, applies only the grant's stored timeout, never reads/logs request or response bodies, and rejects a redirect/retry/second dispatch. The database store—not the caller—creates the grant only after the attempt-marker transaction commits.

- [ ] **Step 1: Write immutable identity and evidence-binding tests**

Cover:

- `test_v20_prepared_identity_is_unchanged`
- `test_v21_identity_changes_with_mode_policy_model_output_contract_percentage_or_ceiling`
- `test_v21_identity_changes_with_key_material_cost_policy_or_attempt_cap`
- `test_v21_identity_changes_with_any_shared_provider_timing_value`
- `test_v21_identity_changes_with_either_validator_price_snapshot`
- `test_v21_identity_changes_with_authorized_percentage_or_rollout_generation`
- `test_v21_identity_changes_with_provider_safety_state_or_rollout_control_epoch`
- `test_v21_identity_changes_with_reply_priming_or_framing_safety_constant`
- `test_v21_thread_replay_requires_every_stored_auto_review_field_to_match`
- `test_v20_and_v21_drafts_link_each_candidate_to_exact_workflow_evidence_refs`
- `test_v20_and_v21_drafts_link_each_candidate_to_same_workflow_agent_run`
- `test_cross_workflow_agent_run_reference_is_rejected_by_database`
- `test_cross_workflow_evidence_ref_reference_is_rejected_by_database`
- `test_agent_run_relational_replay_mismatch_is_invalid_not_payload_repair`
- `test_candidate_and_evidence_refs_rollback_together`
- `test_candidate_replay_neither_duplicates_nor_repairs_ambiguous_refs_silently`
- `test_v20_binding_remains_auto_review_ineligible_by_graph_version`
- `test_two_messages_for_one_source_ref_use_one_sorted_aggregate_fingerprint`
- `test_candidate_evidence_state_hash_is_ref_order_invariant`
- `test_candidate_evidence_state_hash_changes_for_message_version_signature_permission_or_key_identity`
- `test_duplicate_snippet_mapping_is_ambiguous_and_never_guessed`
- `test_ambiguous_candidate_mapping_persists_no_review_item_or_partial_binding`
- `test_generation_identity_is_stable_and_contains_no_raw_evidence`
- `test_generation_fingerprint_changes_for_agent_provider_model_reasoning_prompt_route_or_output_contract`
- `test_v21_null_or_wrong_generation_reasoning_is_ineligible_before_validation`
- `test_mail_fallback_records_the_provider_that_actually_succeeded`
- `test_memory_langchain_result_records_exact_provider_and_model`
- `test_missing_generation_provider_stays_valid_for_v20_but_auto_ineligible`
- `test_v21_extraction_plan_binds_every_agent_route_prompt_output_contract_and_price`
- `test_v21_extraction_uses_exact_o200k_rendered_count_for_korean_not_len_div_four`
- `test_v21_extraction_enforces_24000_char_10000_input_2048_output_and_one_candidate_hard_caps`
- `test_extraction_canonical_output_budget_accepts_exactly_2048_and_rejects_2049_tokens`
- `test_extraction_confidence_signed_zero_canonicalizes_to_positive_fixed_zero`
- `test_v21_extraction_individually_valid_over_2048_canonical_envelope_fails_without_review_item_or_no_candidate_marker`
- `test_v21_extraction_truncated_partial_json_is_failed_never_repaired_or_cached`
- `test_v21_extraction_route_has_zero_sdk_retries_no_provider_fallback_and_cache_trace_off`
- `test_v21_extraction_prepares_once_and_invokes_same_object_without_open_db_session`
- `test_v21_extraction_provider_dto_contains_only_local_aliases_and_allowlisted_plaintext`
- `test_v21_extraction_credential_match_is_zero_call_zero_cache_zero_trace`
- `test_v21_extraction_global_debug_or_verbose_is_zero_call_and_writes_no_prompt_bytes`
- `test_v21_extraction_openai_sdk_debug_logging_is_zero_call_and_caplog_receives_no_prompt_bytes`
- `test_v21_extraction_plan_permutation_is_stable_and_any_route_or_prompt_drift_changes_hash`
- `test_extraction_call_reserves_marks_attempt_and_charges_once_outside_transaction`
- `test_extraction_plan_and_call_bind_all_shared_provider_timing_values`
- `test_extraction_send_permit_is_one_use_nonserializable_and_expires_before_recovery`
- `test_fenced_transport_consumes_only_at_dispatch_and_never_reads_or_logs_body`
- `test_redirect_retry_or_second_dispatch_reuses_no_consumed_permit`
- `test_restart_has_no_reconstructable_send_permit_and_marker_is_not_send_authority`
- `test_attempt_grant_is_not_returned_before_marker_commit`
- `test_database_clock_owns_claim_attempt_and_lease_timestamps_despite_host_clock_skew`
- `test_cancel_between_extraction_marker_and_send_latches_without_premature_terminalization`
- `test_terminal_extraction_call_can_never_send_after_lease_recovery`
- `test_extraction_crash_after_send_never_retries_and_charges_reserved_ceiling`
- `test_extraction_known_overrun_records_actual_opens_extraction_breaker_and_skips_validation`
- `test_extraction_permission_or_source_drift_before_attempt_is_zero_call`
- `test_extraction_permission_or_source_drift_during_call_discards_output_and_charges_once`
- `test_claimed_or_failed_agent_run_is_never_reused_as_complete_cache_hit`
- `test_one_workflow_agent_cannot_claim_a_second_call_with_a_different_plan_hmac`
- `test_cancel_before_extraction_attempt_marker_is_zero_call_zero_charge_terminal`
- `test_cancel_during_extraction_call_charges_once_and_late_completion_creates_no_candidate`
- `test_global_disabled_before_extraction_claim_or_attempt_marker_is_zero_call_zero_charge`
- `test_disable_resume_cancel_and_attempt_marker_interleavings_never_start_a_late_call`
- `test_validation_breaker_open_before_extraction_claim_is_cross_purpose_zero_call`
- `test_extraction_claim_locks_all_required_purpose_rows_in_global_sorted_order`
- `test_extraction_completion_and_exact_candidate_set_or_zero_marker_commit_atomically`
- `test_zero_candidate_completion_persists_explicit_empty_set_hmac_not_only_zero_children`
- `test_completed_candidate_replay_rejects_missing_extra_or_corrupt_child_set`
- `test_completed_call_without_terminal_result_marker_is_invalid_not_cacheable`
- `test_crash_before_atomic_extraction_completion_replays_without_second_provider_call`
- `test_concurrent_extraction_claims_and_resume_converge_on_one_agent_call`
- `test_signed_extraction_plus_validation_obligation_never_admits_above_total_ceiling`


The candidate binding must reject the single candidate when its links/snippets cannot be mapped exactly to its permission-filtered packet. That attempted run is finalized `failed` with the allowlisted aggregate reason `evidence_binding_mismatch`; **no ReviewItem, partial binding row, completed cache entry, or fabricated `no_candidate` result is persisted**. It must never create a permanently unapprovable post-migration pending item or invent a broad packet-level binding.

- [ ] **Step 2: Run focused preflight/draft tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_provider_send_fence.py backend/tests/test_review_v21_preflight.py backend/tests/test_review_v21_drafting.py backend/tests/test_review_v21_extraction.py backend/tests/test_review_v2_preflight.py backend/tests/test_review_v2_drafting.py backend/tests/test_review_v2_agents.py -q
```

Expected: new V2.1 types/bindings are missing; every V2.0 regression remains green.

- [ ] **Step 3: Add a version-specific prepared identity**

Do not add optional V2.1 values to the canonical V2.0 HMAC payload. Add a separate frozen type and builder whose canonical identity includes:

```text
graph_version
configured auto_review_mode
validator provider/model/reasoning effort
validator prompt version
validator output contract version = candidate-validation-batch:v1
policy version
cost policy version
fingerprint key version and key-material verifier
token estimator version/encoding and per-batch input/output token caps
reply priming tokens = 16 and framing safety tokens = 512
maximum validation batches per workflow = 2
maximum validation candidates per batch = 4 and per workflow = 5
max provider attempts = 1
shared provider timeout, send-start window, attempt lease, and commit grace seconds
validator input/output Decimal price per one million tokens
enforce percentage
authorized percentage at launch
rollout authorization generation
provider-safety state version and rollout control epoch
aggregate extraction-plan-set HMAC
sorted extraction provider-safety snapshot-set HMAC
each sorted per-agent extraction provider/model/reasoning/route/prompt/output-contract
each extraction estimator/encoding/16+512 allowances/input-output caps/exact prices
each extraction zero-or-one-candidate contract and strict result-envelope schema
each extraction prepared-content HMAC/framed-token count/provider-safety state/max-attempt
confirmed extraction cost ceiling
confirmed validation cost ceiling
confirmed total cost ceiling
exact V2.1 total-budget limit
```

`review_thread_matches_prepared()` must dispatch by the stored graph version. A V2.1 mismatch is `evidence_changed` or `cost_preview_changed` at the appropriate preflight boundary; it never mutates a previously stored request.

- [ ] **Step 4: Implement the authoritative V2.1 extraction plan and call ledger**

`review_v21_extraction.py` exposes immutable `PreparedExtractionPlan`/`PreparedExtractionPlanSet`/`PreparedExtractionInvocation` plus a store with `claim_or_replay`, `mark_attempt_started -> ProviderAttemptGrant`, `complete(ExtractionLockedContext, ...)`, and `fail(ExtractionLockedContext, ...)`. The store derives lease tokens and every persisted claim/attempt/completion/failure wall timestamp from PostgreSQL `clock_timestamp()` under the locked row; requests may carry an expected timestamp only for equality testing, never authority. SQLite/fake tests inject a DB-clock abstraction. Only `FencedProviderSendPermit` uses the local monotonic clock after marker commit. The plan builder resolves the selected agent only through the immutable extraction registry and accepts exactly `openai/gpt-5.4-mini-2026-03-17/none/auto-review-extraction-route:v1` plus that agent's registered prompt/output schema and `o200k_base` framing policy, then uses the V2.1 privacy-bounded renderer rather than the legacy provider payload. It freezes Responses API transport, `cache=False`, `max_retries=0`, empty callbacks, and disabled tracing, and refuses an alias, Azure OpenAI, Gemini, fallback, unknown tokenizer, missing/mismatched price, mutable route, credential-scan hit, or non-allowlisted provider DTO field. V2.0 alone continues to use the current catalog and fallback policy unchanged.

Zero-call preview renders the bounded selected-source input for each agent, counts `encoded canonical invocation + 16 + 512`, and enforces the fixed 24,000-character/10,000-framed-input-token/2,048-total-output-token upper bounds, the zero-or-one-candidate result contract, and each route's lower caps. The measured count enters the HMAC and later actual-usage verification, but preview/admission reserves the full 10,000 input plus 2,048 output cap—exactly USD 0.016716—for every selected agent; it never discounts from shorter rendered input. Preview HMACs both the sorted plan set and sorted extraction provider-safety snapshot set without persisting plaintext. Start repeats the zero-call preparation before thread creation, compares both aggregate HMACs/current permission/source identities with the token, then stores only keyed plan/safety identities and ceilings on the V2.1 request. For each paid execution, a short read resolves current owner permission/source state, closes the Session, runs `credential-scan:v1`, and renders exactly one immutable `PreparedExtractionInvocation` outside a transaction. It creates the same-workflow/same-agent placeholder `AgentRun(status='claimed')` and its `AutoReviewExtractionCall`; the database uniqueness `(workflow_thread_id, agent_name)` plus service verification that the plan belongs to the stored signed plan set permits exactly one paid call per selected agent, even if a caller supplies a different plan HMAC. Cache/replay queries reuse only `AgentRun.status='complete'`. The exact flow is:

```text
transaction E1:
  shared key-generation/runtime -> sorted required extraction+validation safety
    -> sorted Source -> workflow
  recheck current configured mode is not disabled, cancellation, route/price,
  source/permission/plan+safety HMAC and total signed obligation
  persist prepared invocation HMAC/counts/caps, reserve one call, claim attempt limit; commit
transaction E2:
  same all-purpose safety order -> sorted Source -> workflow -> AgentRun/extraction call
  re-resolve current configured mode/cancellation/owner/source/permission,
  compare the same in-memory invocation
  HMAC/counts plus current registry/safety/timing snapshot, CAS attempt 0 -> 1,
  set attempt_started_at = database clock_timestamp() and
  lease_expires_at = attempt_started_at + stored_provider_attempt_lease_seconds;
  commit, then and only then return ProviderAttemptGrant containing one
  non-serializable monotonic send permit, the stored send-start window,
  stored provider timeout, and authoritative lease expiry
no database transaction:
  body-blind fenced transport consumes the permit at actual request dispatch,
  applies the stored provider timeout, and invokes that exact
  PreparedExtractionInvocation once under disabled tracing;
  no retry/fallback/cache/callback/session
transaction E3:
  same leading order -> sorted Source -> workflow -> AgentRun/extraction call
  re-resolve owner/source/version/permission and prepared identity after the call
  if unchanged, require strict parse plus the complete canonical output-budget guard;
  only then atomically persist actual usage/cost, call completed, AgentRun complete,
  and exactly one valid ReviewItem+evidence-ref set plus exact one-item result-set HMAC, or
  result_kind=no_candidate, count=0, and the explicit empty-set HMAC
  if malformed/truncated/over-budget/changed/cancelled, discard output, persist
  actual-or-reserve charge, mark failed, and write neither ReviewItem nor a fake no_candidate marker
  on overrun atomically open purpose='extraction' provider breaker; commit
```

Unknown usage after the attempt marker charges the full reserve; pre-attempt expiry alone may be reclaimed, and post-attempt expiry never retries. Known input/output/cost or workflow-total overrun uses the same like-for-like rules as Task 9, stores actual cost, opens the extraction breaker across scopes, produces no trusted candidate from that run, and prevents validation. A schema-valid field combination whose canonical envelope is 2,049 or more tokens, or native output that is truncated/partial, is an extraction failure rather than `no_candidate`: E3 charges once, persists no ReviewItem, no result marker, and no cacheable `AgentRun.status='complete'`. Owner/source/version/permission drift discovered after the call is not a provider overrun: it discards all model output and creates no ReviewItem, but still charges actual usage or the conservative reserve exactly once. Global `disabled` or cancellation observed in E1/E2 under the workflow lock terminalizes an attempt-zero claim at zero charge and releases its reservation; neither restart nor resume may bypass that check. Once E2 commits attempt one, cancellation only latches the workflow and output-discard requirement; it does not terminalize the canonical call before the owner finishes or the stored database lease expires. A cancellation racing between marker commit and dispatch is therefore treated as possibly sent: the owner may consume its still-live permit once, but E3 must discard the result and charge actual-or-reserve; cancellation can never manufacture a second call. The one-use `FencedProviderSendPermit` is process-local/non-serializable, is consumed at the HTTP transport's actual dispatch boundary, and rejects after its monotonic send-start deadline; a restarted or late owner therefore cannot send. The stored provider timeout plus send window plus commit grace is strictly shorter than the stored database-clock lease, so expiry recovery cannot race a permitted live send. At/after lease expiry recovery charges the conservative reserve and terminalizes failure; the expired permit can never send afterward. E3 is the only successful completion boundary: after the aggregate output guard, it commits the call, `AgentRun.status='complete'`, exactly one valid ReviewItem and its exact evidence binding plus the domain-separated one-item result-set HMAC, or the explicit signed empty-set marker, together. A second candidate or second ReviewItem child is corruption and fails the run. It never commits a cacheable run first and candidates later; a crash before E3 leaves a non-retryable marked call that recovery finalizes with conservative charge and no phantom cache result. Replay of `complete` re-derives and verifies singular result kind, count in `{0,1}`, and the exact one-child/empty HMAC; absence alone is corruption. `AgentRun` float cost fields are mirrors only. Completion/failure persists no prompt/source/model output or raw exception. The workflow's authoritative paid total is the sum of extraction-call and validation-call Numeric charges exactly once; admission uses other non-final reserves plus final actual charges under the workflow lock.

- [ ] **Step 5: Insert exact evidence refs in the candidate transaction**

Have `_insert_or_get_review_item()` return whether the ReviewItem was newly inserted. For every newly inserted V2.0 or V2.1 item, set server-owned `candidate_contract_version='c5-v1'`, write the exact `agent_run.id` into the relational column, map only the candidate's exact evidence to workflow refs, and insert the rows before the outer transaction commits. Despite its approved legacy name, `candidate_slot_ordinal` is the deterministic one-based **evidence-ref ordinal within one ReviewItem**: each child row receives a distinct value `1..N`; it is not one repeated candidate ordinal. On replay, load and compare the same-workflow AgentRun FK and immutable ref set; a null, cross-workflow, partial, extra, or mismatched binding is `invalid_state_transition`, not a payload lookup or repair write. Pre-migration null AgentRun bindings remain human-only. The V2.0 public schema, HMAC identity, checkpoint, graph, and transition behavior remain byte/shape compatible; only internal provenance rows are additive.

For each bound workflow evidence ref, canonicalize the exact selected message identities/content fingerprints, sort them by stable source-message identity, and HMAC the entire non-empty set with canonical source kind/id/version and the current key version/material verifier. Store exactly one aggregate plus both non-secret key identities; never overwrite one message with another or insert duplicate rows for the same workflow ref. Derive the candidate-level `evidence_version_hash` with domain `candidate-evidence-state:v1` over canonical JSON containing server-keyed `workflow_execution_hmac`, `security_scope_hmac`, immutable `candidate_key`, and the sorted tuples `(workflow_evidence_ref_id, canonical_source_kind, canonical_source_id, canonical_version_or_signature, content_fingerprint, aggregate_message_set_hmac, permission_level, fingerprint_key_version, fingerprint_key_material_verifier)`. Sorting is by same-workflow ref id then canonical source identity; the value is one HMAC, never a concatenation exposed outside the server. Task 9 stores it in the validation identity and A3/B recompute the identical formula from locked/current rows, so ref order is irrelevant while any workflow/scope/candidate/message/version/signature/content/permission/key change invalidates replay. Two indistinguishable snippets that cannot be mapped to exact messages make the candidate auto-ineligible/fail closed rather than guessed. A replay under different key version/material never repairs or re-HMACs an immutable row and is auto-review-ineligible. Source content stays in canonical source storage.

Add `model_provider: str | None = None`, `model_reasoning_effort: str | None = None`, `route_version: str | None = None`, and `output_contract_version: str | None = None` to the end of `AgentRunResult` so existing deterministic constructors remain source-compatible. V2.0 successful LangChain adapters may continue to report their actual provider (`openai`, `azure_openai`, or `gemini`) and route behavior; V2.1 has no fallback and must report exactly `openai`, snapshot `gpt-5.4-mini-2026-03-17`, and reasoning `none`. The owning agent copies exact values into `AgentRunResult`, and `_insert_or_get_agent_run()` persists them in `AgentRun.generation_provider`, `generation_reasoning_effort`, `generation_route_version`, and `generation_output_contract_version`. Existing `agent_name`, `model_name`, and `prompt_version` remain exact generation snapshots.

Compute `candidate_generation_fingerprint` with domain/schema `candidate-generation:v1` over canonical JSON containing exactly `agent_name`, actual `generation_provider`, `model_name`, `generation_reasoning_effort`, `prompt_version`, `generation_route_version`, `generation_output_contract_version`, fingerprint key version, and key-material verifier. It is loaded only through the relational same-workflow `ReviewItem.agent_run_id`; mutable payload metadata is ignored. Any single-field change changes the HMAC/cache identity; a null/unknown field or the exact same provider+model as the validator is human-only. Do not infer provider/reasoning from configured order or parse a route containing multiple fallbacks.

- [ ] **Step 6: Run Task 3A shared preflight/draft/send-fence tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_provider_send_fence.py backend/tests/test_review_v21_preflight.py backend/tests/test_review_v21_drafting.py backend/tests/test_review_v21_extraction.py backend/tests/test_review_v2_preflight.py backend/tests/test_review_v2_drafting.py backend/tests/test_review_v2_agents.py -q
uv run --locked ruff check backend/app/agent_runtime/review_v2_preflight.py backend/app/agent_runtime/review_v2_drafting.py backend/app/agent_runtime/contracts.py backend/app/agent_runtime/review_v2_agents.py backend/app/agent_runtime/model_router.py backend/app/agent_runtime/review_v21_extraction.py backend/app/agent_runtime/auto_review_input_safety.py backend/app/agent_runtime/provider_send_fence.py backend/app/models/agent_workflows.py backend/app/models/agent_runs.py backend/tests/test_provider_send_fence.py backend/tests/test_review_v21_preflight.py backend/tests/test_review_v21_drafting.py backend/tests/test_review_v21_extraction.py backend/tests/test_review_v2_preflight.py backend/tests/test_review_v2_drafting.py backend/tests/test_review_v2_agents.py
```

- [ ] **Step 7: Commit Task 3A shared immutable binding/send-fence slice**

```powershell
git add backend/app/agent_runtime/review_v2_preflight.py backend/app/agent_runtime/review_v2_drafting.py backend/app/agent_runtime/contracts.py backend/app/agent_runtime/review_v2_agents.py backend/app/agent_runtime/model_router.py backend/app/agent_runtime/review_v21_extraction.py backend/app/agent_runtime/auto_review_input_safety.py backend/app/agent_runtime/provider_send_fence.py backend/app/models/agent_workflows.py backend/app/models/agent_runs.py backend/tests/test_provider_send_fence.py backend/tests/test_review_v21_preflight.py backend/tests/test_review_v21_drafting.py backend/tests/test_review_v21_extraction.py backend/tests/test_review_v2_preflight.py backend/tests/test_review_v2_drafting.py backend/tests/test_review_v2_agents.py
git commit -m "feat: bind review candidates to canonical evidence"
```

- [ ] **Step 8: Implement and verify Task 3B agent-owned adapters on `codex/mail-document-agent`**

Developer B bases this branch on the exact Task 3A commit, adapts the Mail/Document and memory-extraction LangChain adapters to the frozen result/provider/send-fence contracts, and leaves registration behind the existing `AgentManifest`/`AgentRegistry`. No direct feature-agent import or shared payload change is allowed.

```powershell
uv run --locked pytest backend/tests/test_mail_document_agent.py backend/tests/test_memory_extraction_langchain_adapter.py backend/tests/test_review_v21_extraction.py -q
uv run --locked ruff check backend/app/agents/mail_document_agent/agent.py backend/app/agents/mail_document_agent/llm.py backend/app/agents/memory_extraction_agent/agent.py backend/app/agents/memory_extraction_agent/langchain_adapter.py backend/tests/test_mail_document_agent.py backend/tests/test_memory_extraction_langchain_adapter.py
git add backend/app/agents/mail_document_agent/agent.py backend/app/agents/mail_document_agent/llm.py backend/app/agents/memory_extraction_agent/agent.py backend/app/agents/memory_extraction_agent/langchain_adapter.py backend/tests/test_mail_document_agent.py backend/tests/test_memory_extraction_langchain_adapter.py
git commit -m "refactor: adapt mail document agents to review v21 contracts"
```

The integration branch merges the exact 3A then 3B commits and reruns the union of both GREEN commands before Task 4. Any required shared-contract edit returns to Task 3A review instead of being smuggled into the agent branch.

---

### Task 4: Introduce Resolution Actors Without Changing Human Review Behavior

**Files:**
- Create: `backend/app/review/actors.py`
- Modify: `backend/app/review/transitions.py`
- Modify: `backend/app/api/v1/review.py`
- Modify: `backend/app/services/audit.py`
- Create: `backend/tests/test_review_resolution_actors.py`
- Modify: `backend/tests/test_review_transitions.py`
- Modify: `backend/tests/test_review_transition_postgres.py`
- Modify: `backend/tests/test_review_rbac.py`
- Modify: `backend/tests/test_audit_logs.py`

**Interfaces:**
- Adapts an already authorized `DemoUser` to `ReviewResolutionActor(actor_type='human')` and keeps every current public action/response/replay behavior unchanged.
- Adds an internal actor-aware transition core, `ApprovalDirective`, and a separate deterministic `mark_evidence_stale` transition; public `ReviewAction` remains the current three-value union.
- Allows `create_new` only for a human with `human_review` or an auto actor with matching `auto_review` policy. Allows `reuse_existing` only for the internal auto-resolution service. Reject and needs-more-evidence remain human-only.
- The internal stale transition is not a general needs-more action: it is exposed only on the injected coordinator-to-transition service (never the public action dispatcher), requires a locked canonical re-resolution proving that a previously bound evidence version changed or disappeared, accepts no note, and can only change `pending_review` to `needs_more_evidence`. It does not add a reusable actor capability.
- Keeps the existing generic `record_audit_log(DemoUser, ...)` contract for all unrelated callers. A narrow review-resolution audit adapter projects a human actor to the existing email/role columns or uses fixed application-owned system values; it writes only allowlisted actor type/id, item id, action/outcome, versions, bounded counts, and cost—never raw reason or evidence and never a fabricated `DemoUser`.

- [ ] **Step 1: Write capability, forgery, and compatibility tests**

Add tests that prove:

- `test_human_adapter_preserves_exact_permission_levels`
- `test_auto_actor_has_only_public_internal_and_auto_review`
- `test_public_request_cannot_supply_actor_type_capability_or_directive`
- `test_auto_actor_cannot_reject_request_evidence_or_bulk_review`
- `test_only_internal_stale_directive_with_locked_canonical_drift_marks_needs_more`
- `test_public_or_current_evidence_cannot_invoke_stale_directive`
- `test_wrong_policy_auto_actor_cannot_approve`
- `test_existing_human_approve_reject_needs_more_and_replay_are_unchanged`
- `test_audit_writer_does_not_log_raw_reason_or_source_content`
- `test_system_resolution_audit_uses_fixed_schema_projection_without_fake_user`


In PostgreSQL, retain the existing row-lock/concurrent approval assertions while changing the service actor type.

- [ ] **Step 2: Run actor/transition tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_review_resolution_actors.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py backend/tests/test_review_rbac.py backend/tests/test_audit_logs.py -q
```

Expected: missing actor/directive failures; legacy behavior tests document the compatibility baseline.

- [ ] **Step 3: Refactor to one actor-aware locked core**

Keep a narrow public wrapper if that minimizes callers:

```python
def transition(
    self,
    db: Session,
    *,
    item_id: int,
    action: ReviewAction,
    actor: ReviewResolutionActor,
    approval_directive: ApprovalDirective | None = None,
) -> ReviewTransitionResult:
    raise NotImplementedError
```

Default `approval_directive` to `CreateNewPromotion()` only for an authorized human approve. The internal resolution path must pass its directive explicitly. Check capability and exact item permission before acquiring effects; recheck under the item lock before mutation. Never construct a fake `DemoUser` for the system actor.

Every new human approve/reject/needs-more transition writes `resolution_source='human'` and null policy/validation fields. Auto approval writes `reviewer_id='system:auto-review'`, `resolution_source='auto_policy'`, matching `resolution_policy_version`, and the same-item completed `auto_validation_id` in the locked transaction. `mark_evidence_stale` writes the bounded system actor and reason code but no model result or human note. Legacy terminal rows retain nullable fields and are not backfilled heuristically.

- [ ] **Step 4: Adapt API and audit call sites**

Convert the authenticated user only after existing API visibility/RBAC checks. Do not accept actor fields, the stale directive, or internal capabilities in Pydantic request bodies, query parameters, headers, or dependency overrides. Keep concealment behavior (foreign/inaccessible item becomes 404) and existing 409 transition mapping.

Do not change `AuditLog` columns or the shared `record_audit_log()` signature. Add a review-only adapter that maps the internal actor to fixed `actor_id='system:auto-review'`, `actor_email='system:auto-review@paraworks.invalid'`, and `actor_role='system'`; human audit rows retain the authenticated user's existing values. Tests assert the system constants and the bounded metadata allowlist.

- [ ] **Step 5: Run actor/transition tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_review_resolution_actors.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py backend/tests/test_review_rbac.py backend/tests/test_audit_logs.py -q
uv run --locked ruff check backend/app/review/actors.py backend/app/review/transitions.py backend/app/api/v1/review.py backend/app/services/audit.py backend/tests/test_review_resolution_actors.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py backend/tests/test_review_rbac.py backend/tests/test_audit_logs.py
```

- [ ] **Step 6: Commit the actor boundary**

```powershell
git add backend/app/review/actors.py backend/app/review/transitions.py backend/app/api/v1/review.py backend/app/services/audit.py backend/tests/test_review_resolution_actors.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py backend/tests/test_review_rbac.py backend/tests/test_audit_logs.py
git commit -m "refactor: add review resolution actor boundary"
```

---

### Task 5: Add Exact Claim Fingerprints, Provenance, and Reaffirmation

**Files:**
- Create: `backend/app/knowledge/claim_fingerprints.py`
- Create: `backend/app/knowledge/trusted_fingerprint_projection.py`
- Create: `backend/app/knowledge/trusted_provenance.py`
- Modify: `backend/app/agent_runtime/keyed_mutation_guard.py`
- Modify: `backend/app/admin/auto_review_keys.py`
- Modify: `backend/app/knowledge/promotion.py`
- Modify: `backend/app/review/transitions.py`
- Create: `backend/app/review/auto_review_resolution.py`
- Create: `backend/tests/test_auto_review_provenance.py`
- Modify: `backend/tests/test_review_knowledge_promotion.py`
- Modify: `backend/tests/test_review_transitions.py`
- Modify: `backend/tests/test_review_transition_postgres.py`

**Interfaces:**
- Computes one schema-versioned HMAC from only the normalized substantive fields returned by `build_promotion_preview()`, target type, project key, security scope, and normalization version.
- Computes a separate normalized-title collision bucket HMAC. It never hashes arbitrary payload metadata into the claim identity and never uses semantic similarity.
- Maintains a keyed `TrustedKnowledgeFingerprint` projection for approved Timeline/History rows with exact/legacy-unknown scope resolution and an indexed title-bucket guard. Auto-review readiness fails closed while the projection is missing/stale.
- Writes one active approval link per primary/companion effect and child evidence links for every exact canonical evidence ref on human and auto approvals of candidates created after the C.5 migration, including new V2.0 human approvals.
- Extends the locked promotion core with `create_new` and `reuse_existing`; exact reaffirmation returns canonical ids with `promotion_effect='reaffirmed'` and creates no duplicate knowledge row.
- Recognizes pre-migration human `source_review_item_id` as implicit active base provenance, but never treats an auto-policy-created row that way.

- [ ] **Step 1: Write fingerprint, cardinality, replay, and collision tests**

Cover:

- `test_timeline_fingerprint_uses_only_title_and_result_summary`
- `test_history_fingerprint_uses_only_title_and_reason`
- `test_human_decision_and_todo_effects_receive_target_specific_fingerprints`
- `test_companion_timeline_uses_its_own_written_target_fingerprint`
- `test_metadata_order_unicode_nfc_and_whitespace_are_canonical`
- `test_different_target_project_scope_or_substantive_value_changes_hmac`
- `test_create_new_writes_primary_companion_and_all_evidence_links_once`
- `test_exact_reaffirmation_reuses_canonical_ids_and_adds_own_links`
- `test_reaffirmation_replay_returns_same_canonical_result`
- `test_visible_nonmatching_title_bucket_forces_conflict`
- `test_hidden_collision_guard_returns_only_exists_and_never_loads_identity`
- `test_legacy_unknown_scope_is_not_reused`
- `test_fingerprint_projection_backfills_known_scope_and_marks_unknown_scope_collision_only`
- `test_missing_or_stale_projection_disables_auto_review_until_rebuilt`
- `test_unprojected_hidden_row_barrier_forces_zero_call_human_only`
- `test_projection_ready_flips_only_after_locked_anti_join_and_count_checksum_match`
- `test_projection_summary_delta_is_exact_for_insert_update_and_remove`
- `test_provenance_and_projection_rows_snapshot_key_version_and_material_verifier`
- `test_rotation_waits_for_inflight_keyed_mutation_before_advancing_generation`
- `test_every_postgres_guard_callsite_uses_the_same_fixed_lock_ids_and_order`
- `test_disabled_sqlite_human_approve_uses_process_guard_and_keeps_projection_not_ready`
- `test_healthy_human_approval_updates_summary_delta_and_preserves_ready`
- `test_unprojectable_human_approval_demotes_to_rebuild_required_without_blocking_human`
- `test_human_approve_succeeds_while_projection_not_ready_and_auto_review_stays_disabled`
- `test_post_migration_v20_human_approval_writes_complete_explicit_provenance`
- `test_pre_migration_unbound_v20_human_approval_uses_legacy_base_without_childless_link`
- `test_reuse_history_requires_exactly_one_locked_matching_companion_bundle`
- `test_missing_ambiguous_or_mismatched_companion_is_human_only`
- `test_reaffirmed_bundle_revoke_is_atomic_for_primary_and_companion`
- `test_key_admin_cli_never_accepts_or_logs_secret_material`
- `test_projection_rebuild_cli_reports_exact_summary_and_replay`
- `test_disabled_only_rotation_waits_for_old_generation_and_rebuilds_new_projection`
- `test_rotation_refuses_attempt_zero_or_inflight_provider_call_until_ledger_recovery`
- `test_rotation_succeeds_only_after_marked_attempt_is_conservatively_charged_terminal`


Use SQL capture or a repository fake to prove the hidden guard executes a server-side `EXISTS` projection, not a query that loads id, content, permission, or count.

- [ ] **Step 2: Run focused promotion tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_provenance.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py -q
```

- [ ] **Step 3: Implement canonical claim and collision HMACs**

Expose narrow functions:

```python
def normalized_claim_fingerprint(
    *, item: ReviewItem, security_scope_id: str, settings: Settings
) -> str:
    raise NotImplementedError

def promoted_effect_fingerprint(
    *,
    knowledge_type: Literal[
        'timeline_event', 'history_event', 'decision_record', 'todo'
    ],
    normalized_persisted_fields: Mapping[str, str | None],
    project_key: str | None,
    security_scope_id: str,
    settings: Settings,
) -> str:
    raise NotImplementedError

def trusted_title_collision_bucket(
    *, item_type: Literal['timeline_event', 'history_event'],
    normalized_title: str,
    settings: Settings,
) -> str:
    raise NotImplementedError

def trusted_project_scope_fingerprint(
    *, project_key: str | None, settings: Settings
) -> str:
    raise NotImplementedError
```

Call `build_promotion_preview(item)` once and reject unknown/missing normalized fields. Use HMAC utilities already in `agent_runtime/fingerprints.py`; never log the normalized content or return it from public APIs.

Define target-specific fingerprints for every promoted effect. Timeline uses normalized `(title, result_summary)`; History uses `(title, reason)`; Decision uses `(title, decision_summary)`; Todo uses the exact persisted `(title, assignee, due_date, priority, priority_reason)`. A companion Timeline gets its own Timeline fingerprint from the exact title/result actually written, not the primary record's fingerprint. Only Timeline/History fingerprints participate in C.5 exact reuse, but human Decision/Todo links remain complete and non-null.

`keyed_mutation_guard.py` is the sole producer of the two global locks: `AUTO_REVIEW_KEY_GENERATION_LOCK_ID = 1066041229503628369` and `TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID = -2972884933094306491`. Its typed API exposes PostgreSQL `acquire_generation_shared(session)`, `acquire_generation_exclusive(session)`, `lock_runtime_state(session, mode='share|update')`, and `acquire_projection(session)`; after the shared barrier plus runtime row are verified it returns an unforgeable `KeyGenerationLockedContext(session_identity, generation, key_version, material_verifier)`. It asserts the order and records no raw ids. All Task 5/6/9/10 callsites import this API instead of issuing advisory SQL. Its SQLite implementation is a process-local re-entrant guard for disabled human/revoke smoke only and always reports projection/auto readiness false.

`trusted_fingerprint_projection.py` builds/repairs projection rows in a locked, deterministic, zero-provider operation. Title bucket and project-scope HMAC are domain-separated from security scope; `security_scope_id` is a separate indexed column so a candidate can safely detect a matching legacy-unknown row. Exact scope is derived only from the originating ReviewItem's stored workflow thread; otherwise mark `scope_resolution='legacy_unknown'` and never reuse it. Candidate collision lookup executes server-side `EXISTS` over `(knowledge_type, project_scope_hmac, normalized_title_bucket_hmac)` for current exact scope **or** legacy-unknown scope, without selecting identity/content/count.

Define `ProjectionSummary(active_count, checksum_hex)` and a pure `ProjectionSummaryDelta` API. A row digest is the big-endian 256-bit HMAC `trusted-fingerprint-summary-row:v1` over canonical projection schema, knowledge type/id, scope resolution/id, project/title/claim HMACs, permission, active status, and key version/material; the aggregate is modular addition of unique row digests modulo `2**256`, rendered as exactly 64 lowercase hex characters. State keeps **separate** source and projected count/checksum pairs. Under the projection lock, insert/update/remove subtracts the old digest and adds the new digest for each side in the same transaction; equality plus the anti-join/mismatch guards proves readiness. This is never XOR and never an unspecified whole-table checksum.

`AutoReviewRuntimeKeyState.ready` means only that the persisted key generation/material is initialized and stable; ordinary projection work never updates that row while holding `FOR SHARE`. Projection rebuild acquires the shared key-generation barrier, locks runtime state `FOR SHARE`, then takes the projection advisory lock before committing **projection-only** `ready=false, rebuild_required=true` and processing rows. Rotation requires disabled mode; its transition transaction acquires the generation barrier **exclusively**, locks runtime state `FOR UPDATE`, and refuses before changing identity if any extraction or validation call remains non-terminal. This includes attempt-zero reservations and attempt-one provider calls, because the shared barrier is intentionally not held across network I/O. Attempt-zero rows must first be cancelled/recovered to zero-charge terminal state; attempt-one rows must first be recovered to actual-or-conservative charged failure under the old generation. Only with zero non-terminal calls does rotation take the projection lock, atomically advance runtime version+material verifier+generation as stable/ready, and reset projection version/material/generation with `ready=false, rebuild_required=true`. That transaction waits for every old-generation database mutation holding the shared barrier; after commit, no old-key call ledger remains able or required to mutate. Rebuild proceeds under new-generation shared barriers. Promotion and revoke use the same shared-generation/runtime-share/projection order and update canonical knowledge, projection, and both summary deltas in one transaction. The final rebuild transaction repeats that shared/read-only-runtime order, runs a server-side active Timeline/History-to-projection anti-join and permission/status/key-material mismatch guard, recomputes both summaries, and sets **only** projection `ready=true, rebuild_required=false` when every check matches. Every auto-review preflight requires runtime ready plus projection ready, matching schema/key version/key-material verifier/generation, equal count/checksum pairs, and a fresh boolean anti-join coverage guard; there is no `FOR SHARE`→`FOR UPDATE` row-lock upgrade path.

`backend.app.admin.auto_review_keys` is both the service boundary and an executable local control plane. It exposes `status()`, `bootstrap()`, `rebuild_trusted_fingerprint_projection() -> ProjectionRebuildResult(source_count, projected_count, source_checksum, projected_checksum, replayed, ready)`, and `rotate_fingerprint_key(expected_version, next_version, reason, principal)`. The exact operator commands are `python -m backend.app.admin.auto_review_keys status|bootstrap|rebuild` and `python -m backend.app.admin.auto_review_keys rotate --expected-version <vN> --next-version <vN+1> --reason <1..500 chars>`. The local CLI accepts no actor or credential override: it runs only under the host's restricted operations identity and attributes the action to fixed `system:local-auto-review-key-admin`, clearly typed as a system operator rather than impersonating a human. A future remote endpoint must obtain a real authenticated admin principal from the server dependency and may never accept a subject in the body. The module obtains every secret only from validated `Settings`; it has no secret CLI option and never echoes a verifier, secret, raw subject, or normalized key material.

Rotation uses a CLI-only `SecretStr` key-ring source from the deployment secret manager containing exactly `(current_version,current_secret,next_version,next_secret)`. It first derives the current material verifier from the old secret and compares it with the locked DB state, then derives the next verifier/projection HMACs only from the next secret. Missing next material, a wrong old material verifier, version mismatch, or equal old/new material is a bounded refusal. The normal application process still receives only its one active version/secret. `rotate` refuses unless global mode is disabled, there is no in-flight migration, and a locked aggregate query finds zero non-terminal extraction/validation calls. `status` prints only the two bounded non-terminal counts; it never prints call ids. The exclusive barrier drains database mutations but is not falsely claimed to cover the provider network gap. Operators resume/cancel or run the bounded Task 12 call-ledger recovery until both counts reach zero, then retry rotation. After the transition and rebuild, admission stays disabled until deployment configuration is cut over to the new single active version/secret and `status` proves runtime/projection identities ready. Exit `0` means the requested invariant is ready, `2` is bounded configuration/authorization refusal, and `3` is retained-state/readiness failure. Lifespan ordering and bounded recovery are finalized in Tasks 10/12; rebuild and rotation remain explicit operator actions.

- [ ] **Step 4: Add approval and evidence provenance services**

`TrustedKnowledgeApprovalLink` owns exactly one primary or companion effect. `TrustedKnowledgeEvidenceLink` is a child keyed by `approval_link_id` and can represent multiple evidence refs. The fingerprint projection and both link levels snapshot the current non-secret fingerprint key version/material verifier used for their keyed values. On replay, compare both key identities and the complete expected child set; do not append a partial repair silently. For History reaffirmation, lock and verify the directive's primary plus companion ids/fingerprints as one bundle and require exactly one canonical active companion whose normalized written fields, project, scope, permission, and status match. Missing, ambiguous, mismatched, or partially linked bundles force human review and no mutation. A new V2.0 or V2.1 candidate with missing bindings is an invalid state and cannot create a childless link.

Active-provenance calculation is:

```text
active explicit approval links
OR legacy human base where source_review_item_id is present and
   originating ReviewItem.resolution_source is NULL or human, and its
   originating ReviewItem lacks `candidate_contract_version='c5-v1'`
```

The transition identifies that narrow legacy cohort from the persisted per-row contract marker plus the database-enforced post-marker writer guard, never from application timestamps, clock comparison, or a client flag. It preserves the existing evidence-bearing ReviewItem and `source_review_item_id`, creates no incomplete approval/evidence link, remains human-only and non-revocable, and is covered by V2.0 approval plus skewed-created-at regressions. An auto-policy originating item never gains implicit legacy-base status merely because the existing knowledge row has `source_review_item_id`.

- [ ] **Step 5: Extend promotion under the existing item lock**

For every human or automatic approve/reaffirmation, use the total order `shared key-generation/runtime -> sorted provider-safety rows on the auto path -> projection -> rollout -> sorted Source -> workflow -> ReviewItem -> promotion decision/audit -> approval/evidence links -> target knowledge -> document locks`. An unlocked locator may discover source/workflow ids, but all ownership and current source state are rechecked after the ordered locks. The **auto** path additionally requires every matching provider-safety row closed, matching runtime/projection key material, both ready states, equal summary pairs, a clean active-row anti-join, and fresh exact duplicate/hidden-collision checks; any failure leaves the item for human review. The **human** path does not lock provider safety and is never blocked by projection incompleteness/rebuild. If projection state started ready/matching and an exact row+summary delta can be applied, it commits canonical human resolution, complete provenance, projection row, and both source/projected deltas while preserving ready. Only when projection state was already unhealthy/key-mismatched or an exact projection delta cannot be produced does it commit the human result while setting **projection-only** `ready=false, rebuild_required=true`; runtime key state remains read-locked and unchanged. This closes check→concurrent promotion→write TOCTOU without stopping shadow comparison every time a human reviews; reject/needs-more transitions skip only irrelevant projection work, not the Source/workflow-before-item order.

For `reuse_existing`, then lock and verify exact target type/id, approved status, current permission, scope, project, and claim fingerprint. The service must create ReviewItem resolution plus approval/evidence links in the same transaction and return:

```python
PromotionResult(
    target_type='history_event',
    created_record_ids=(existing_id,),
    created_timeline_event_ids=(companion_timeline_id,),
    effect='reaffirmed',
)
```

Do not change the legacy response keys; `effect` is additive only in the internal/result and new V2.1-aware response. `create_new` retains the current exactly-once unique source-review behavior and additionally writes explicit provenance.

- [ ] **Step 6: Run promotion/provenance tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_provenance.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py -q
uv run --locked ruff check backend/app/knowledge/claim_fingerprints.py backend/app/knowledge/trusted_fingerprint_projection.py backend/app/knowledge/trusted_provenance.py backend/app/agent_runtime/keyed_mutation_guard.py backend/app/admin/auto_review_keys.py backend/app/knowledge/promotion.py backend/app/review/transitions.py backend/app/review/auto_review_resolution.py backend/tests/test_auto_review_provenance.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py
```

- [ ] **Step 7: Commit the provenance/reaffirmation slice**

```powershell
git add backend/app/knowledge/claim_fingerprints.py backend/app/knowledge/trusted_fingerprint_projection.py backend/app/knowledge/trusted_provenance.py backend/app/agent_runtime/keyed_mutation_guard.py backend/app/admin/auto_review_keys.py backend/app/knowledge/promotion.py backend/app/review/transitions.py backend/app/review/auto_review_resolution.py backend/tests/test_auto_review_provenance.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py
git commit -m "feat: add exact trusted knowledge provenance"
```

---

### Task 6: Make Auto Approval Precisely Revocable and Non-Resurrectable

**Task 6A RAG/orchestrator-owned files (`codex/rag-orchestrator-agent`):**
- Create: `backend/app/review/auto_review_revoke.py`
- Create: `backend/app/review/auto_review_source_reconciliation.py`
- Create: `backend/app/review/auto_review_audit_transitions.py`
- Create: `backend/app/admin/auto_review_source_reconciliation.py`
- Create: `backend/app/knowledge/trusted_serving_eligibility.py`
- Create: `backend/app/review/evidence_visibility.py`
- Create: `backend/app/rag/serving_locks.py`
- Modify: `backend/app/rag/indexing.py`
- Modify: `backend/app/rag/reindexing.py`
- Modify: `backend/app/rag/vector_store.py`
- Modify: `backend/app/rag/pgvector_store.py`
- Modify: `backend/app/models/vector_index.py`
- Modify: `backend/app/api/v1/knowledge.py`
- Modify: `backend/app/api/v1/dashboard.py`
- Modify: `backend/app/api/v1/review.py`
- Modify: `backend/app/api/v1/todos.py`
- Modify: `backend/app/api/v1/notifications.py`
- Modify: `backend/app/api/v1/integrations.py`
- Modify: `backend/app/api/v1/projects.py`
- Modify: `backend/app/api/v1/search.py`
- Modify: `backend/app/api/v1/ask.py`
- Modify: `backend/app/api/v1/assistant.py`
- Modify: `backend/app/assistant/service.py`
- Modify: `backend/app/agent_runtime/company_memory.py`
- Modify: `backend/app/projects/service.py`
- Modify: `backend/app/agents/rag_orchestrator_agent/agent.py`
- Modify: `backend/app/agents/rag_orchestrator_agent/service.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_auto_review_revocation.py`
- Create: `backend/tests/test_auto_review_source_reconciliation.py`
- Create: `backend/tests/test_auto_review_source_reconciliation_admin.py`
- Create: `backend/tests/test_review_evidence_visibility.py`
- Modify: `backend/tests/test_rag_indexing.py`
- Modify: `backend/tests/test_pgvector_store.py`
- Modify: `backend/tests/test_pgvector_integration.py`
- Modify: `backend/tests/test_knowledge_api.py`
- Modify: `backend/tests/test_dashboard_api.py`
- Modify: `backend/tests/test_review.py`
- Modify: `backend/tests/test_todos_api.py`
- Modify: `backend/tests/test_notifications_api.py`
- Modify: `backend/tests/test_mock_sync.py`
- Modify: `backend/tests/test_integration_runtime_status.py`
- Modify: `backend/tests/test_search_permissions.py`
- Modify: `backend/tests/test_search_retrieval_backend.py`
- Modify: `backend/tests/test_ask_api.py`
- Modify: `backend/tests/test_assistant_api.py`
- Modify: `backend/tests/test_assistant_service.py`
- Modify: `backend/tests/test_assistant_email_agent.py`
- Modify: `backend/tests/test_company_memory_orchestration_service.py`
- Modify: `backend/tests/test_orchestration_api.py`
- Modify: `backend/tests/test_project_memory_api.py`
- Modify: `backend/tests/test_rag_orchestrator_agent.py`
- Modify: `backend/tests/test_rag_orchestrator_service.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`

**Task 6B Mail/Document-owned files (`codex/mail-document-agent`, based on the exact green Task 6A contract commit):**
- Create: `backend/app/ingestion/source_content_signature.py`
- Modify: `backend/app/connectors/google.py`
- Modify: `backend/app/ingestion/service.py`
- Modify: `backend/app/ingestion/source_versions.py`
- Modify: `backend/app/ingestion/sync.py`
- Modify: `backend/app/documents/service.py`
- Create: `backend/tests/test_source_content_signature.py`
- Modify: `backend/tests/test_google_connector.py`
- Modify: `backend/tests/test_connector_ingestion_contract.py`
- Modify: `backend/tests/test_document_ingestion_service.py`

**Interfaces:**
- Adds `delete_many(document_ids)` to production, preview, and deterministic vector-writer/store contracts.
- Adds `narrow_permissions(document_ids, permission_level)` to those contracts. It can only retain or narrow visibility under the same document lock; broadening is a bounded refusal and does not call an embedding provider.
- Limits raw chunk documents to ReviewItems with `status='approved'` and `resolution_source IS NULL OR 'human'`.
- Excludes tombstoned knowledge documents during build, before embedding, and in the same conditional pgvector upsert statement.
- Excludes tombstoned rows in pgvector search before ranking and hidden-match accounting, so even a stale physical vector cannot be served or counted.
- Serializes every production pgvector upsert and revoke for an exact document id with one shared, keyed, per-document PostgreSQL transaction advisory lock. Both paths use the same SQLAlchemy `Session` and transaction for lock, final eligibility/tombstone read, and serving write/delete.
- Holds the shared key-generation barrier plus runtime key row through every production serving mutation; key rotation waits for old-generation writes, and no stale worker can derive a lock or write after the generation advances.
- Revokes only an approved `auto_policy` ReviewItem with complete validation and provenance. It is idempotent and never targets a human approval.
- Keeps shared canonical knowledge serving while any other explicit active link or valid legacy-human base remains.
- Treats canonical source permission as part of effective source state even when the content signature is unchanged. Connector sync may still count a same-content/same-permission event as skipped, but a permission-only event must update `Source` and every current `DocumentChunk`, report a changed source state, and enter synchronous trust reconciliation without rerunning parsing/extraction.
- On an exact unchanged content/permission event, advances only allowlisted connector-operational metadata, refreshed `source_url`, and connector-evidence signature under the Source lock while still reporting fetched/skipped and doing zero parse/extraction/embedding/reconciliation work. A connector can never overwrite a server-owned signature, permission, scope, pointer, or provenance field through that path.
- Applies one fail-closed `TrustedServingEligibility` predicate before Knowledge/Timeline/Dashboard/Project API projection, RAG document construction, pgvector ranking, and hidden-match accounting. An auto-only trusted target with absent, lookup-failed, superseded, quarantined, or unsupported current evidence is invisible even if reconciliation has not run or a stale physical vector remains. Review Queue reads deliberately use the separate permission-aware `ReviewEvidenceVisibilityService`, because quarantined evidence must remain actionable to an authorized reviewer without becoming trusted serving content.
- Reconciles changed source states synchronously after the source-state commit; no CDC/stream is introduced. A crash between source commit and reconciliation is safe because the live serving predicate already excludes or narrows the stale row, and bounded startup/admin recovery finds mismatched active provenance by relational anti-join.
- Requires supported Google adapters to preserve an exact verified upstream semantic timestamp through Task 2's `SourceEvent.semantic_timestamp_raw`; missing/malformed Gmail `internalDate` or Drive `modifiedTime` is a bounded connector-data failure, never `datetime.now()` synthesis. Calendar signature identity comes from its exact start date/date-time, while its operational display timestamp may use valid `updated` or a deterministic parsed start fallback.
- Treats server parser-policy, parser implementation, and chunk-policy identity as source-state authority in addition to content and permission. A policy-only change reparses/rechunks deterministically and invalidates/rebuilds only affected raw chunk index state; it makes zero extraction/validator LLM calls and invokes embedding only for post-hash-skip chunks whose serving content actually changed.
- Makes every Review-derived count/card/mutation actor-aware. Dashboard, Knowledge, Projects, Todo completion, Notifications, and generic Integrations runtime/sync count helpers apply trusted-serving or Review-evidence visibility before aggregation or mutation; missing actor propagation fails closed and no count/title/failed-run identity leaks across workflow/security scope.
- Applies the same live predicate at the last public retrieval boundary for Search, Ask, Assistant, and company-memory orchestration. No stale/quarantined auto-only source id, URL, snippet, citation, hidden count, or answer is serialized. The RAG orchestrator carries a server-only `ServingDependencySnapshot` beside each selected candidate; public `RagAnswer` evidence fields never act as authority. `append_assistant_message` stores the exact dependency rows and answer atomically. A raw-chunk snapshot binds the current chunk/document-version/parser-run/source signature and content hash. A trusted-knowledge snapshot binds the target/content hash plus one deterministically selected active approval effect and all of that effect's exact evidence links, or the narrowly proven pre-C.5 legacy-human base. Any candidate that cannot produce a complete exact snapshot is excluded before answer generation; an answer with zero surviving evidence returns the normal bounded no-evidence result instead of being persisted as evidenced RAG output.
- Keeps persisted answer bytes only as immutable audit storage. `backend.app.assistant.service` performs one shared fail-closed projection used by `serialize_message`, conversation serialization, recent-message context, derived summary, and email-draft context. Every access rechecks all exact dependencies through `TrustedServingEligibilityService`, current version/parser/signature/content hash, actor permission, active non-quarantined effect, and complete evidence-child set. If one required dependency is absent, changed, revoked, quarantined, broader than the actor, or lookup-failed, the entire assistant answer becomes bounded `evidence_unavailable`, with empty citations/source ids/links/snippets, zero hidden-match detail, and a regeneration notice. Raw `AssistantConversation.summary` is audit/cache data only: public serialization and LLM/email context recompute a bounded summary from currently eligible projected messages and never copy the stored summary blindly. A retained pre-C.5 assistant row whose evidence-shaped fields are non-empty but whose exact dependency contract is null/incomplete receives the same fail-closed projection; non-evidence operational messages remain available only when their marker/empty dependency invariant proves they contain no RAG evidence.

- [ ] **Step 1: Write revoke, shared-provenance, and indexing-race tests**

Cover:

- `test_human_or_incomplete_auto_item_cannot_be_revoked`
- `test_revoke_marks_only_the_selected_reaffirmation_link`
- `test_last_provenance_revokes_primary_companion_and_exact_documents`
- `test_history_last_link_and_shared_companion_timeline_revoke_per_effect`
- `test_shared_history_and_last_companion_timeline_revoke_per_effect`
- `test_revoke_replay_returns_the_same_bounded_result`
- `test_replay_uses_first_commit_result_after_other_provenance_later_changes`
- `test_revoke_document_count_is_stable_with_no_physical_vector_and_on_replay`
- `test_selected_audit_allows_direct_revoke_only_after_completed_confirmed`
- `test_pending_remediation_or_critical_audit_blocks_normal_direct_revoke`
- `test_server_owned_critical_recovery_context_is_the_only_gate_bypass`
- `test_human_revoke_requires_strict_reason_enum_and_never_classifies_free_text`
- `test_business_withdrawal_writes_immutable_assessment_and_uses_normal_audit_gate`
- `test_gate_rejected_business_withdrawal_does_not_consume_unique_assessment`
- `test_same_reason_replays_but_concurrent_different_reason_returns_revoke_reason_conflict`
- `test_quality_reason_cannot_use_direct_revoke_without_quality_audit_context`
- `test_source_invalidation_revoke_writes_no_human_revocation_assessment`
- `test_raw_chunks_exclude_auto_policy_approval`
- `test_tombstoned_documents_are_skipped_before_embedding`
- `test_pgvector_delete_many_uses_exact_document_ids`
- `test_pgvector_conditional_upsert_cannot_cross_a_tombstone`
- `test_pgvector_search_excludes_stale_tombstoned_row_before_hidden_count`
- `test_reindex_and_revoke_use_the_same_document_advisory_key_and_session`
- `test_nested_vector_mutation_validates_already_held_context_without_reacquiring_runtime`
- `test_wrong_session_generation_or_document_set_rejects_vector_locked_context`
- `test_serving_mutation_refuses_mixed_fingerprint_key_versions`
- `test_serving_mutation_refuses_same_version_with_different_key_material`
- `test_rotation_waits_for_inflight_old_generation_mutation_then_new_writes_use_new_generation`
- `test_exact_revoke_succeeds_when_projection_not_ready_and_keeps_rebuild_required`
- `test_advisory_lock_closes_guard_snapshot_before_revoke_commit_race`
- `test_in_memory_delete_runs_after_commit_and_is_discarded_on_rollback`
- `test_revoke_failure_rolls_back_item_knowledge_links_tombstones_and_index_state`
- `test_same_content_permission_only_event_is_not_skipped_and_does_not_reparse`
- `test_server_content_signature_detects_changed_body_with_same_external_version_and_signature`
- `test_server_and_connector_signatures_are_separate_and_only_server_signature_is_authoritative`
- `test_legacy_connector_only_signature_cannot_authorize_c5_skip_reuse_or_serving`
- `test_missing_malformed_or_ambiguous_external_signature_never_causes_unchanged_skip`
- `test_server_signature_normalization_boundary_vectors_are_frozen_for_every_source_type`
- `test_gmail_attachment_signature_uses_filename_not_nonexistent_attachment_name`
- `test_calendar_all_day_date_and_offset_equivalent_instant_canonicalization_is_stable`
- `test_calendar_updated_revision_and_event_context_key_are_operational_not_semantic`
- `test_gmail_missing_or_malformed_internal_date_is_rejected_without_now_fallback`
- `test_drive_missing_or_malformed_modified_time_is_rejected_without_now_fallback`
- `test_calendar_missing_updated_uses_deterministic_start_for_display_but_signature_uses_raw_start`
- `test_supported_google_adapter_populates_exact_semantic_timestamp_raw`
- `test_server_parser_registry_ignores_connector_parser_chunk_and_snippet_authority`
- `test_parser_or_chunk_policy_upgrade_is_not_unchanged_and_reparses_without_extraction_or_validation_call`
- `test_policy_rechunk_reindex_embeds_only_changed_content_after_incremental_skip`
- `test_deferred_slack_repeated_event_keeps_legacy_dedupe_but_never_gains_c5_authority`
- `test_semantic_metadata_change_changes_signature_but_cursor_account_or_scope_noise_does_not`
- `test_unchanged_event_advances_safe_cursor_url_and_connector_signature_without_reparse_or_reconciliation`
- `test_connector_metadata_cannot_overwrite_server_signature_permission_scope_or_pointer`
- `test_permission_only_event_updates_source_and_all_current_chunk_permissions`
- `test_permission_only_event_narrows_all_historical_chunks_and_chunk_vectors`
- `test_permission_to_unknown_deletes_raw_chunk_vectors_and_index_states_without_provider_call`
- `test_content_supersession_indexes_only_current_document_version_and_removes_old_chunk_vectors`
- `test_duplicate_v1_labels_with_different_signatures_never_admit_old_chunk_vector`
- `test_post_c5_human_only_and_pre_c5_legacy_human_serving_semantics_are_compatible`
- `test_permission_change_is_visible_to_preflight_before_any_provider_call`
- `test_auto_trusted_vector_is_excluded_before_ranking_while_reconciliation_is_pending`
- `test_knowledge_project_and_rag_serving_fail_closed_before_reconciliation`
- `test_dashboard_and_deterministic_rag_fail_closed_before_reconciliation`
- `test_public_to_internal_reconciliation_narrows_target_and_vector_without_embedding`
- `test_internal_to_public_never_auto_broadens_target_or_vector_permission`
- `test_restricted_unknown_absent_or_superseded_source_revokes_exact_auto_effect`
- `test_source_invalidation_preserves_shared_human_provenance_and_strict_permission`
- `test_review_api_conceals_source_links_after_public_to_internal_narrowing`
- `test_review_api_conceals_revoked_evidence_after_restricted_unknown_invalidation`
- `test_critical_item_is_hidden_from_trusted_serving_but_visible_actionable_in_review_to_authorized_reviewer`
- `test_review_evidence_visibility_returns_404_to_unauthorized_actor_and_never_leaks_source_fields`
- `test_missing_source_keeps_bounded_review_action_for_authorized_scope_but_conceals_evidence`
- `test_dashboard_review_counts_and_titles_use_actor_evidence_visibility_before_aggregation`
- `test_notification_review_and_agent_run_counts_are_actor_scoped_and_actorless_calls_fail_closed`
- `test_integration_runtime_pending_count_is_actor_scoped_without_slack_feature_work`
- `test_stale_or_quarantined_auto_todo_is_concealed_before_complete_mutation`
- `test_valid_human_or_shared_todo_completion_preserves_existing_response_shape`
- `test_search_and_ask_never_serialize_stale_quarantined_source_fields_or_hidden_counts`
- `test_rag_answer_persists_complete_exact_dependencies_with_message_atomically`
- `test_rag_candidate_without_exact_dependency_is_dropped_before_answer_generation`
- `test_assistant_history_redacts_persisted_answer_and_all_source_fields_after_auto_evidence_revoke`
- `test_revoked_after_answer_leaks_nothing_to_message_list_context_summary_or_email_draft`
- `test_assistant_dependency_source_drift_permission_narrowing_or_lookup_failure_fails_closed`
- `test_pre_c5_unbound_evidence_answer_is_audit_only_but_non_evidence_operational_message_survives`
- `test_assistant_dependency_write_failure_rolls_back_message_and_agent_run_linkage`
- `test_source_changes_between_rag_retrieval_and_message_commit_fail_closed_without_raw_answer_persistence`
- `test_company_memory_public_orchestration_rechecks_live_serving_before_answer_projection`
- `test_last_source_invalidated_provenance_tombstones_and_cannot_reindex`
- `test_selected_pending_audit_is_system_invalidated_and_no_longer_blocks_rollout_gate`
- `test_source_reconciliation_recovery_is_bounded_idempotent_and_restart_safe`
- `test_source_reconciliation_admin_has_fixed_actor_aggregate_output_and_exit_codes`
- `test_current_document_version_repair_is_bounded_deterministic_and_never_guesses_ambiguous_rows`
- `test_repair_reports_remaining_ambiguous_rows_and_exit_three_until_resync`
- `test_pointer_repair_requires_existing_verified_server_signature_and_exact_matching_parser_run`
- `test_legacy_transformed_body_or_missing_event_timestamp_cannot_synthesize_signature_and_requires_resync`
- `test_crash_after_source_document_commit_before_reconciliation_is_fail_closed_and_recoverable`
- `test_source_update_and_post_provider_promotion_serialize_on_current_source_state`
- `test_any_critical_or_remediation_audit_quarantines_only_its_auto_effect_before_revoke_cleanup`


Add PostgreSQL interleaving tests for all relevant boundaries: reindex commits its guarded upsert first and revoke subsequently deletes it; revoke commits the tombstone first and the later guarded upsert affects zero rows; and reindex reaches its initial guard/read before revoke starts but cannot write from that stale snapshot because both operations serialize on the same transaction advisory lock and reindex rechecks after acquiring it. Every schedule finishes with a tombstone and no serving vector.

- [ ] **Step 2: Run focused revoke/index tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_source_content_signature.py backend/tests/test_review_evidence_visibility.py backend/tests/test_google_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_document_ingestion_service.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py -q
```

- [ ] **Step 3: Add exact delete and eligibility contracts**

Extend the writer protocol:

```python
class VectorIndexWriter(Protocol):
    def upsert_with_embedding(
        self, document: VectorDocument, embedding: list[float]
    ) -> None:
        raise NotImplementedError

    def delete_many(self, document_ids: Sequence[str]) -> int:
        raise NotImplementedError

    def narrow_permissions(
        self, document_ids: Sequence[str], permission_level: str
    ) -> int:
        raise NotImplementedError
```

Deduplicate and sort document ids before deletion or permission narrowing. Pgvector builds `DELETE FROM {table_name} WHERE document_id = ANY(:document_ids)` and a guarded `UPDATE ... SET permission_level=:strictest` from the already validated `PgVectorConfig.table_name`, returning affected counts. Narrowing verifies the partial order `public < internal < restricted`; unknown is never writable/servable and any attempted broadening fails. Its search CTE adds both `NOT EXISTS` against `vector_serving_tombstones` and the relational live-source eligibility predicate before score/visibility ranking and hidden-match aggregation. The in-memory implementation removes or narrows only exact keys through a transaction-aware mutation queue: apply after commit, discard after rollback, and rely on tombstone/eligibility-aware rebuild after restart. Preview records mutations without changing production state.

Create `serving_locks.py` as the only producer of document serving lock keys. Locking is deliberately two-stage. The outer coordinator first obtains Task 5's `KeyGenerationLockedContext` at the global prefix, then locks any Source/workflow/ReviewItem/knowledge classes in their required order. Only at the document-lock position may it call `VectorServingLockManager.acquire_documents(key_context, document_ids)`. That method verifies the exact same live Session/current generation from the supplied context, derives the domain-separated HMAC (`vector-serving-document-lock:v1`), maps it through the existing signed-64-bit advisory-key helper, deduplicates/sorts document ids, executes only the document `SELECT pg_advisory_xact_lock(:key)` statements, and returns `VectorServingLockedContext(key_context, sorted_document_ids)`. It never reacquires generation/runtime and it holds the already-owned prefix plus document locks through commit. Never use Python `hash()`, raw document ids, or different lock namespaces in reindex and revoke. Same key version with different secret fails. Rotation uses the exclusive barrier/row-update sequence defined in Task 5, rebuilds projections, and only then marks ready, so two live versions/materials never derive different locks.

Change pgvector `_upsert_sql()` from `VALUES` to an `INSERT SELECT` whose guard is `WHERE NOT EXISTS (SELECT 1 FROM vector_serving_tombstones WHERE document_id = :document_id)` in the same SQL statement. Every production `PgVectorStore` upsert/delete/narrow accepts the already-held `VectorServingLockedContext` and validates exact session object, current generation, and document-id membership as defense in depth; it never reacquires the earlier runtime or document lock after Source/workflow/knowledge locks. A standalone store mutation uses an outer coordinator that obtains `KeyGenerationLockedContext`, discovers and locks every applicable later class, then calls `acquire_documents()` at the document position and passes the result inward. Missing, forged, wrong-session, wrong-generation, or incomplete contexts are bounded failures. `reindex_components()` and revoke acquire each stage exactly once in their outer coordinator and inject the same context/settings. Tests force reversed revoke/reindex schedules and reject any nested runtime/document reacquire. A separate Python precheck or conditional statement without the common advisory lock is insufficient at PostgreSQL `READ COMMITTED`.

- [ ] **Step 4: Make source-state changes immediately fail closed and synchronously reconcilable**

Replace both content-only skip checks (`ingestion.sync._changed_content_signature_events` and `ingestion.service._same_content_signature`) for the supported Google/document boundary with one frozen `SourceStateChangeClassification(content_changed, permission_changed, parser_policy_changed, primary_code)` contract. `primary_code` is a bounded deterministic projection of the three booleans; callers branch on the booleans and may report only the code, never infer a parser change from connector metadata. Only three false flags are `unchanged`. `source_content_signature.py` is the single registry/implementation for server-computed `server-source-content:v1`. It freezes semantic metadata keys as `gmail: ()`, `gmail_attachment: ('filename', 'mime_type')`, `drive: ('mime_type',)`, and `calendar: ('attendee_domains', 'end', 'event_status', 'location', 'organizer_email', 'start')`; no connector may extend the tuple dynamically. The payload always contains schema, source type, title, body, author, participants, one registry-selected semantic timestamp, and every registered semantic key. Title/body/author/string metadata are UTF-8 NFC with CRLF/CR normalized to LF, but preserve case plus every other leading/trailing/internal whitespace; body is never trimmed, collapsed, or case-folded. Missing optional values encode JSON null while empty strings remain empty. Required title/body malformed/null is unverifiable. Participants are normalized by the same string rule, exact-deduplicated, and sorted by normalized UTF-8 bytes; case variants stay distinct. A normal event semantic timestamp comes only from Task 2's exact `semantic_timestamp_raw`, parses as an aware datetime, and renders UTC RFC3339 with exactly six fractional digits plus `Z`; missing, naive, or malformed values are unverifiable. Calendar is explicit: its signature timestamp is the exact raw `start`, never connector `updated`/`SourceEvent.timestamp`; `start`/`end` accept either exact ISO `YYYY-MM-DD` encoded as `{'kind':'date','value':'YYYY-MM-DD'}` or an aware datetime encoded as `{'kind':'instant','value':'<UTC RFC3339 microseconds Z>'}`. Offset-equivalent instants canonicalize identically, while date and instant remain distinct. Calendar `event_context_key`, `updated`, attendee response counts, cursor, and external revision are operational and excluded; status/location/organizer/times/participant set remain semantic. `attendee_domains` uses the participant-list rule. Other semantic values accept only their frozen string/null type. Canonical JSON uses `ensure_ascii=False`, sorted keys, compact separators, and `allow_nan=False`; SHA-256 over those exact UTF-8 bytes is the 64-hex signature. Any normalization/key/type/allowlist change requires a new schema version, connector re-sync/full source reconciliation/current-pointer repair, and explicit approval—never an in-place v1 reinterpretation.

The result is stored only in `Source.server_content_signature_schema/server_content_signature`; the connector's `raw_metadata['content_signature']` is copied into `Source.connector_content_signature` as non-authoritative evidence and the ambiguous legacy raw key is no longer read by C.5. Cursor, partition, account, OAuth scope, fetch time, connector version, calendar `updated`/`event_context_key`, external revision, and other unregistered metadata never enter the server signature. `google.py` validates upstream timestamp provenance before constructing a supported event: Gmail stores the exact `internalDate`, Drive the exact `modifiedTime`, Gmail attachments inherit the parent's exact raw value, and Calendar stores its exact start date/date-time. `_timestamp_from_google_millis` and `_timestamp_from_iso` never use `datetime.now(UTC)` as a data fallback. Missing/malformed Gmail or Drive authority produces a bounded connector-data failure and no Source/Document write; Calendar requires a valid start and uses valid `updated` only for its operational timestamp, otherwise a deterministic parsed start (UTC midnight for an all-day date), never wall-clock now. Server parsing semantics are registry-owned as well: C.5 selects exact `parser_policy_version`, parser name, parser implementation version, and chunk-policy version from server code using source type plus allowlisted MIME type, and binds those identities to the immutable parser run; chunks bind that run relationally. Connector `parser_name`, `parser_status`, `chunk_max_chars`, `source_snippet`, or other parser hints are evidence-only/ignored for authority and can never alter parsing, reuse, or chunk identity. A current registry identity mismatch sets `parser_policy_changed=true` even when content and permission match: ingestion creates a new exact parser run/DocumentVersion/chunk set, advances the pointer, removes superseded raw-chunk serving state, and runs incremental reindex after content-hash skips. It makes zero extraction or validator calls; an embedding call is allowed only for a changed post-policy chunk that survived the normal incremental skip. For exact equal signature+permission+parser identity, ingestion may update only bounded allowlisted non-secret operational keys (`sync_cursor`, `sync_partition`, `connector_revision`, `connector_updated_at`), refreshed `source_url`, and `connector_content_signature`; it preserves fetched/skipped counts and makes zero parse/extraction/embedding/reconciliation calls. Account/scope/security ownership remains server-resolved and is not connector-writable even though such noise is excluded from the signature. Parser-run signature/policy fields, relational chunk binding, workflow evidence refs, canonical resolution, and `current_document_version_id` all bind the server signature. The current `source_id:v1` fallback is removed. Missing, malformed, unverifiable, or ambiguous canonical inputs fail closed and can never produce an unchanged classification. Skip requires equal server signatures, equal normalized permission, and the exact current server parser/chunk-policy identity.

Slack remains explicitly deferred. Task 6 does not alter the Slack connector/agent or manufacture unavailable Slack data. `source_type='slack'` stays on a named legacy-dedupe branch that may compare its existing connector-provided content signature solely to avoid repeat parsing in the V2.0 ingestion path; it never writes `server-source-content:v1`, advances a C.5 current-version pointer, or becomes auto-review/trusted-serving eligible. A fake `SourceEvent` regression freezes repeat-event compatibility without constructing a live Slack client. Existing Slack rows without a server signature remain fail-closed for C.5 until a later separately approved Slack recovery/re-sync deliverable.

Canonical source mutation is one ordered transaction `shared key-generation/runtime -> sorted Source FOR UPDATE -> sorted Document/Version/ParserRun/Chunk -> sorted document advisory/vector state`; it never reaches backward into projection, rollout, workflow, or review rows. In a supported permission-only branch it updates the Source and **all** DocumentChunk permission metadata for that Source, including historical versions, and monotonically narrows all existing `chunk:{id}` vectors without reparsing text or creating ReviewItems. `public -> internal -> restricted` uses `narrow_permissions`; transition to unknown updates canonical metadata but exactly deletes all affected raw chunk vectors and `VectorIndexState` rows under the same document locks, with zero embedding/provider call. Content supersession writes the new server signature plus current parser/chunk identities, exact new DocumentVersion pointer, and synchronously deletes older chunk vector/index state. A parser-policy-only change keeps the Source signature and permission, writes a new current-policy parser run/version/chunk set and pointer, tombstones/removes superseded raw-chunk index state, and queues only incremental content-hash-aware reindex; it never creates an extraction AgentRun or ReviewItem. It commits an additive internal `changed_source_states` outbox value while preserving public fetched/created/skipped counts. Only after that commit does reconciliation start a fresh global-order transaction. A crash in the gap is fail-closed through the relational serving predicate and recovered by the bounded scanner. Raw indexing selects only `current_document_version_id`; a live Source+DocumentVersion+server-signature+current parser-policy/run+pointer SQL guard excludes every stale physical row. Null/ambiguous pointers or missing/wrong-policy parser-run bindings remain unavailable. Provider completion/promotion takes Source `FOR SHARE`, so it cannot commit from a stale pre-update snapshot. V2.0 disabled SQLite keeps its process-local guard.

`TrustedServingEligibilityService` is the sole server-side predicate for **trusted knowledge serving**. It relationally resolves active approval/evidence links to the current canonical `Source` and returns only a bounded `{eligible, effective_permission}` value. An auto approval effect with **any** linked audit row—mandatory, sampled, or manual—or later audit correction whose effective outcome is critical or whose status is `remediation_required` is immediately quarantined by that durable relational state and excluded before physical revoke; another independent human/valid auto provenance may still keep the target serving. For an auto-only target, every remaining active non-quarantined auto effect must still have every exact canonical source, matching version/content signature, and supported `public|internal` permission; absence, lookup/database error, unknown/restricted permission, mismatch, or only quarantined effects is `eligible=false`. For shared human/auto targets, human provenance may keep the knowledge row trusted, but effective permission is the strictest of the stored target, ReviewItem, immutable evidence snapshots, and every resolvable current source; it is never broadened. A post-C.5 human-only target uses its complete explicit evidence links and the same monotonic permission calculation; mismatch never imports new source content into the trusted claim. A pre-C.5 legacy-human target with no exact links remains available only at its already-stored permission and can never be auto-reused or auto-broadened; database lookup failure still excludes it for that request. Compatibility tests freeze both cohorts. Knowledge, Dashboard, Timeline, Projects, Todo completion, deterministic RAG-orchestrator, RAG builders, and pgvector reads compute this before user-visible filtering or mutation; implementation uses `rg` to enumerate every direct approved-knowledge query and a regression fails if a serving consumer bypasses the shared predicate. Dashboard/Knowledge/Projects routes receive `CurrentUser` and pass it into the service rather than loading globally and filtering after projection. `complete_todo` returns concealed 404 before title/status mutation when the Todo is ineligible or the actor cannot see its live effective permission. RAG builders drop ineligible documents before embedding. Pgvector's SQL CTE has two exact branches: knowledge documents correlate internal knowledge type/id metadata to approval/evidence links, audit quarantine/correction, and `sources`; raw `chunk:{id}` documents join `document_chunks -> document_versions -> documents.current_document_version_id -> document_parser_runs -> sources` through the exact chunk parser-run FK. Both reject an absent/ambiguous/stale/quarantined/wrong-policy version or effect and a vector whose stored permission is broader than the computed live permission **before** visibility ranking and hidden-count aggregation. Lookup failure is exclusion, never a permissive fallback. No raw source id or mismatch reason enters public results.

`ReviewEvidenceVisibilityService` is the only Review Queue evidence projection. It first applies existing tenant/workflow/RBAC concealment, then re-resolves the current actor's permission against every current Source and fails closed. A reviewer who lacks a known current source permission receives 404 for the item; no count, audit state, link, or snippet leaks. If the actor remains authorized for the stored review scope but evidence is absent/superseded/unknown, the quarantined/revoked item remains visible and actionable with only bounded `action_required`, audit/revoke state, and `evidence_unavailable` metadata; source URL/snippet/id are omitted. Authorized current evidence is narrowed to current effective permission. Dashboard pending groups/counts, Notifications review counts, and the generic Integrations `_pending_review_count` helper all require a `CurrentUser` and aggregate only projections admitted by this service; demo mode cannot bypass it. Notifications also scopes failed AgentRuns by workflow ownership/security scope before exposing a bounded failure notification. Thus critical/remediation/corrected items disappear from Knowledge/Timeline/Dashboard/Project/RAG/pgvector immediately while the Review screen can still complete remediation without creating a trusted-serving bypass. Updating the generic Integrations count helper is a visibility repair only and creates no Slack client, Slack data, or Slack feature behavior.

After committing canonical source state, `AutoReviewSourceReconciliationService.reconcile(changed_states)` runs synchronously and idempotently. It never relies on CDC. Each transaction follows `shared key-generation/runtime -> projection -> sorted rollout rows -> sorted Source rows -> sorted workflow rows -> ReviewItems -> required audits -> approval/evidence links -> targets/fingerprint projection and summary deltas -> sorted document locks -> tombstones/index/vector state`. For unchanged content with a current stricter supported permission, retain the trust decision and atomically narrow the ReviewItem's effective permission, every affected target/companion, `TrustedKnowledgeFingerprint.permission_level`, projection source/projected summary deltas, and every physical vector to the strictest permission; update the non-embedding index-state hash and never broaden later if the source becomes more public. Immutable evidence/link snapshots remain historical, while Review/API visibility always uses the monotonic effective ReviewItem/live-source permission. For restricted/unknown, absent/deleted, or content/version supersession, use a server-owned `source_invalidation` context to revoke only affected auto approval effects; human/legacy provenance remains, and the last active target effect receives the normal tombstone/delete treatment. Immutable evidence snapshots are never rewritten to pretend they observed a new source version.

`AutoReviewAuditTransitionStore` is the single persistence authority for selected-audit status and mandatory counters. Task 6 creates its locked/CAS `complete_source_invalidated(...)` operation; Task 10 extends the same store with authorized human-confirmed and critical/remediation operations instead of issuing separate ORM updates. Every operation requires an already-held runtime/rollout/Source/workflow/ReviewItem/audit context, validates the immutable promotion decision, updates pending/confirmed/invalidated counters exactly once, and returns an immutable replay result. Source reconciliation cannot forge a human outcome, and the later human/critical path cannot use the source-invalidation null-outcome code.

If an invalidated item has a selected required audit still `pending`, the shared store completes it with `outcome=NULL`, internal allowlisted resolution code `source_invalidated_before_audit`, `action_required=false`, and an exactly-once `invalidated_before_audit_count`; this is neither confirmed nor a validator-quality critical outcome and is excluded from every precision numerator/denominator. It removes a dangling action from that item but **does not** satisfy the mandatory-50 trust gate: while `confirmed_mandatory_audit_count < 50`, every next eligible promotion is a mandatory replacement audit and no unsampled/10% cohort begins. Thus progression still requires 50 human-confirmed mandatory audits. Completed/critical/remediation audits remain immutable and follow their existing revoke gate. The unforgeable `SourceInvalidationRevokeContext` is a second narrow gate bypass: only this injected reconciliation service can construct it, and revoke accepts it only while ordered locks prove a current absent/restricted/unknown/version-mismatched Source and a matching affected evidence link, with any pending audit invalidated in the same transaction. It carries no broad actor permission, cannot read or approve content, and authorizes only monotonic permission narrowing or exact revoke; public/human/auto actors cannot request, serialize, or forge it. Startup performs one unlocked bounded stale-id scan (`limit=100`) after key/projection services are ready, then reconciles each id through the same global lock order. Reindex calls the same reconciliation preflight and cannot resurrect a stale target.

`AutoReviewSourceReconciliationService` also exposes `status(limit<=100)`, `recover_stale_sources(limit<=100)`, and `repair_current_document_versions(limit<=100)`. Repair is intentionally pointer-only: it never synthesizes `server-source-content:v1` from a legacy/transformed `DocumentVersion.body`, joined chunks, connector signature, URL, label, timestamp, or `MAX(id)`, because those rows do not preserve the exact original SourceEvent envelope/whitespace/timestamp. It may set `current_document_version_id` only when the Source already has a valid server signature and exactly one same-document parser run/version carries the identical verified server signature and server parser-policy identity. Zero or multiple matches are `requires_resync`/`ambiguous`, make no write, and keep readiness false. Connector re-sync supplies the original supported `SourceEvent`; ingestion then atomically computes/writes the server signature, creates the exact parser run/version/chunks, advances the pointer, and queues reconciliation. Slack without a live source remains deferred and C.5-ineligible rather than guessed.

The local module `python -m backend.app.admin.auto_review_source_reconciliation status --limit 100`, `recover --limit 100`, or `repair-current-document-versions --limit 100` accepts no subject, source id, secret, raw-content, or raw-output option, uses fixed `system:local-auto-review-source-reconciler` attribution, and prints only aggregate stale/reconciled/repaired/ambiguous/remaining/failure counts plus readiness. Exit `0` means no detected stale or ambiguous work remains, `2` is bounded configuration/key refusal, and `3` means retained stale/ambiguous work or a reconciliation failure remains so the operator must resync/remediate and rerun. Lifespan runs exactly one reconciliation recovery batch, not an unbounded repair; the CLI is the continuation/repair path.

- [ ] **Step 5: Implement one stable-lock-order revoke transaction**

Lock in the global total order shared with promotion/rebuild: shared key-generation/runtime, projection, rollout row when present, sorted canonical Source rows, workflow, ReviewItem, immutable promotion decision/its unique audit and correction if any, approval/evidence links, target knowledge/companion rows, active-provenance set, sorted exact document advisory locks, tombstones/index states. Re-read stable key generation, current source state, and the exact target/link/bundle set after the leading locks; never enter revoke with an earlier unlocked snapshot. Human revoke accepts exactly one server-validated `AutoReviewRevokeReasonCode`: non-quality `business_withdrawal`, or quality `incorrect_content|permission_violation|wrong_source_version|policy_violation`. There is no free-text or LLM reason classifier. Normal direct revoke is permitted only for `business_withdrawal` and only if no audit exists or its status is `completed` with outcome `confirmed`; `pending`, `remediation_required`, every critical/corrected outcome, and every quality reason return `audit_required`/`quality_audit_required` with no provenance change **and no revocation assessment row**. After the normal gate passes, the business path inserts/replays the immutable assessment in the same transaction as exact revoke. Quality reasons can proceed only with Task 10's unforgeable breaker-first `QualityRevokeContext`; its coordinator inserts/replays the assessment immediately before the breaker/quarantine commit, and confirmed audits are not a loophole. An existing assessment with a different reason is `revoke_reason_conflict`; it never gets overwritten. Only that critical-audit/recovery context or the independently unforgeable, currently proven `SourceInvalidationRevokeContext` may bypass the normal gate; each is accepted only by its owning coordinator and cannot be supplied through an API. Source invalidation writes no human assessment, replaces the normal actor-visibility check only with locked canonical drift proof, and has no authority to read content or approve. Projection incompleteness or `ready=false` does **not** block an exact revoke: under the lock, update/remove the affected projection row when safe and otherwise keep/set projection state `ready=false, rebuild_required=true`, while canonical provenance/tombstone/vector revocation still commits. Only auto approval requires projection-ready. Then:

1. validate actor capability and exact current permission;
2. return canonical replay if already revoked;
3. mark the selected item/link revoked;
4. recompute active provenance independently for every primary/companion approval-link target;
5. for each target whose own provenance becomes empty, mark only that exact knowledge row revoked; an independently shared companion remains serving even when its primary does not, and vice versa;
6. write tombstones for the distinct serving-document identities whose own target became untrusted and delete every matching `VectorIndexState` plus pgvector document with the same session/transaction;
7. snapshot on the ReviewItem whether any target knowledge remained trusted and the number of distinct serving-document identities tombstoned by this first commit, then write a bounded audit event and commit.

Do not expose document ids or other provenance in the result. Human revoke/assessment attribution uses domain `auto-review-revoke-actor:v1` plus the current key version/material verifier; server recovery/source-invalidation contexts use their fixed typed system attribution and can never accept a caller subject. `revoked_document_count` is the stable number of distinct serving-document identities for which this item made the corresponding target last-provenance and created/owned a tombstone; it is not the physical pgvector affected-row count and is zero for wholly shared provenance. Physical delete counts remain internal metrics. Return only item id, revoked status, replayed flag, the first-commit `knowledge_remains_trusted`, and the persisted first-commit document count. Replay returns those snapshots even if another provenance changes later.

- [ ] **Step 6: Add post-provider locked eligibility checks to reindex**

Filter approved/no-tombstone/live-source-eligible/current-document-version documents before batch estimation, finish the embedding provider call with no open database transaction or advisory lock, and start a fresh write transaction. From the detached batch, discover dependency ids, then acquire the shared generation/runtime guard, sorted current Source rows `FOR SHARE`, and only afterward all changed document locks in sorted order on `db`; while those locks are held, re-read each canonical knowledge status, current Source/DocumentVersion/effective permission, and tombstone, then conditionally upsert and persist `VectorIndexState` on that same `db` transaction. If eligibility changed after embedding, count it as skipped and saved serving write; never persist an indexed state. A permission-only narrowing uses `narrow_permissions` and updates the state hash without a provider call. Preserve the existing indexed/skipped/saved-embedding-call metrics and add bounded stale-source/revoked skip counts only to internal/admin observability if needed.

- [ ] **Step 7: Run and commit the Task 6A revoke/serving/visibility slice GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_review_evidence_visibility.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py -q
uv run --locked ruff check backend/app/review/auto_review_revoke.py backend/app/review/auto_review_source_reconciliation.py backend/app/review/auto_review_audit_transitions.py backend/app/review/evidence_visibility.py backend/app/admin/auto_review_source_reconciliation.py backend/app/knowledge/trusted_serving_eligibility.py backend/app/rag/serving_locks.py backend/app/rag/indexing.py backend/app/rag/reindexing.py backend/app/rag/vector_store.py backend/app/rag/pgvector_store.py backend/app/models/vector_index.py backend/app/api/v1/knowledge.py backend/app/api/v1/dashboard.py backend/app/api/v1/review.py backend/app/api/v1/todos.py backend/app/api/v1/notifications.py backend/app/api/v1/integrations.py backend/app/api/v1/projects.py backend/app/api/v1/search.py backend/app/api/v1/ask.py backend/app/api/v1/assistant.py backend/app/assistant/service.py backend/app/agent_runtime/company_memory.py backend/app/projects/service.py backend/app/agents/rag_orchestrator_agent/agent.py backend/app/agents/rag_orchestrator_agent/service.py backend/app/main.py backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_review_evidence_visibility.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py
git add backend/app/review/auto_review_revoke.py backend/app/review/auto_review_source_reconciliation.py backend/app/review/auto_review_audit_transitions.py backend/app/review/evidence_visibility.py backend/app/admin/auto_review_source_reconciliation.py backend/app/knowledge/trusted_serving_eligibility.py backend/app/rag/serving_locks.py backend/app/rag/indexing.py backend/app/rag/reindexing.py backend/app/rag/vector_store.py backend/app/rag/pgvector_store.py backend/app/models/vector_index.py backend/app/api/v1/knowledge.py backend/app/api/v1/dashboard.py backend/app/api/v1/review.py backend/app/api/v1/todos.py backend/app/api/v1/notifications.py backend/app/api/v1/integrations.py backend/app/api/v1/projects.py backend/app/api/v1/search.py backend/app/api/v1/ask.py backend/app/api/v1/assistant.py backend/app/assistant/service.py backend/app/agent_runtime/company_memory.py backend/app/projects/service.py backend/app/agents/rag_orchestrator_agent/agent.py backend/app/agents/rag_orchestrator_agent/service.py backend/app/main.py backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_review_evidence_visibility.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py
git commit -m "feat: enforce revocable trusted serving"
```

- [ ] **Step 8: Run/commit Task 6B, merge its exact commit, and run the integrated Task 6 gate**

In the Task 6B Mail/Document worktree based on the exact Task 6A commit:

```powershell
uv run --locked pytest backend/tests/test_source_content_signature.py backend/tests/test_google_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_document_ingestion_service.py backend/tests/test_auto_review_source_reconciliation.py -q
uv run --locked ruff check backend/app/ingestion/source_content_signature.py backend/app/connectors/google.py backend/app/ingestion/service.py backend/app/ingestion/source_versions.py backend/app/ingestion/sync.py backend/app/documents/service.py backend/tests/test_source_content_signature.py backend/tests/test_google_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_document_ingestion_service.py backend/tests/test_auto_review_source_reconciliation.py
git add backend/app/ingestion/source_content_signature.py backend/app/connectors/google.py backend/app/ingestion/service.py backend/app/ingestion/source_versions.py backend/app/ingestion/sync.py backend/app/documents/service.py backend/tests/test_source_content_signature.py backend/tests/test_google_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_document_ingestion_service.py
git commit -m "feat: bind ingestion to current source policy"
```

Publish that exact green commit hash. Merge/cherry-pick it into `codex/rag-orchestrator-agent` without contract edits, then run:

```powershell
uv run --locked pytest backend/tests/test_auto_review_revocation.py backend/tests/test_auto_review_source_reconciliation.py backend/tests/test_auto_review_source_reconciliation_admin.py backend/tests/test_source_content_signature.py backend/tests/test_review_evidence_visibility.py backend/tests/test_google_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_document_ingestion_service.py backend/tests/test_rag_indexing.py backend/tests/test_pgvector_store.py backend/tests/test_pgvector_integration.py backend/tests/test_knowledge_api.py backend/tests/test_dashboard_api.py backend/tests/test_review.py backend/tests/test_todos_api.py backend/tests/test_notifications_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py backend/tests/test_search_permissions.py backend/tests/test_search_retrieval_backend.py backend/tests/test_ask_api.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_email_agent.py backend/tests/test_company_memory_orchestration_service.py backend/tests/test_orchestration_api.py backend/tests/test_project_memory_api.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_agent_runtime_lifespan.py -q
git diff --check
```

If integration requires a SourceEvent/parser/permission/Review-visibility contract change, return it to the shared-contract review gate and update both branches/tests; do not hide it in a merge-fix commit. Task 7 starts only after this integrated command is green.

---

### Task 7: Implement Deterministic Eligibility and Policy Authority

**Files:**
- Create: `backend/app/agent_runtime/auto_review_eligibility.py`
- Modify: `backend/app/agent_runtime/auto_review_input_safety.py`
- Create: `backend/app/agent_runtime/auto_review_policy.py`
- Modify: `backend/app/agent_runtime/canonical_sources.py`
- Modify: `backend/app/knowledge/trusted_fingerprint_projection.py`
- Modify: `backend/app/knowledge/trusted_provenance.py`
- Create: `backend/tests/test_auto_review_eligibility.py`
- Create: `backend/tests/test_auto_review_policy.py`

**Interfaces:**
- `AutoReviewEligibilityService.evaluate()` returns only `eligible`, `human_review`, `needs_more_evidence`, or `reuse_trusted` plus an allowlisted reason code and an internal expected target when reuse is safe.
- The pure `AutoReviewPolicyEngine` consumes frozen value objects only; it has no Session, registry, model, network, or clock dependency.
- Only Timeline/History with exact public/internal permission, immutable candidate evidence bindings, complete normalized fields, no uncertainty/high-risk cue, supported registry identity, and available budget can reach the validator.
- Applies a versioned deterministic credential/secret scanner to the permission-filtered plaintext immediately before request assembly. Any high-confidence credential marker is zero-call human-only; C.5 never redacts and then validates a semantically changed claim.
- Exact visible duplicate may become `reuse_trusted`; any non-exact visible collision, indexed hidden/legacy-unknown collision existence, stale/incomplete fingerprint projection, ambiguity, or lookup failure becomes human review without leaking metadata.

- [ ] **Step 1: Write the policy matrix as table-driven tests**

Include positive and hard-negative cases for:

```text
timeline_event/history_event versus decision_record/todo/unknown
public/internal versus restricted/unknown
complete exact ref versus absent/partial/ambiguous ref
current signature versus source drift/deleted evidence/permission change
direct normalized fields versus empty/project-selection/inference/proposal/conditional
exact duplicate versus visible mismatch/hidden EXISTS/legacy unknown scope
one visible exact plus any hidden/legacy-unknown collision versus visible exact with no hidden collision
validator supported/direct_fact/0.9800 versus 0.9799/partial/unknown/extra slot
budget/version/generator identity ready versus unavailable/same model/missing identity
high-confidence API key/password/token/connector-secret marker versus safe prose
```

Assert forced-human cases do not ask for a validator request and source drift maps to needs-more-evidence. Assert policy decisions are identical for repeated frozen inputs and exact `Decimal('0.9800')` passes without rounding.

- [ ] **Step 2: Run eligibility/policy tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_eligibility.py backend/tests/test_auto_review_policy.py -q
```

- [ ] **Step 3: Build canonical preflight without provider access**

Return a frozen result similar to:

```python
@dataclass(frozen=True)
class AutoReviewEligibilityResult:
    decision: AutoReviewEligibilityDecision
    reason_codes: tuple[AutoReviewPolicyReasonCode, ...]
    claim_fingerprint: str | None = None
    duplicate_target: TrustedTargetRef | None = None
    validation_request: CandidateValidationRequest | None = None
```

Build claim text only from promotion preview normalized fields. Load current evidence through candidate refs and canonical resolver, apply exact permission filtering first, then run `credential-scan:v1` over the exact candidate/evidence plaintext before creating ephemeral `Cxx`/`Exx` aliases. The scanner uses reviewed high-confidence provider-token/connector-secret/password-assignment patterns, bounded entropy checks only inside credential-like lexical contexts, and a frozen allowlist of documented fake/example forms. A match yields only allowlisted `sensitive_input_detected`, discards the text, makes no provider/cache/trace call, and never logs the matched bytes. Scanner rule changes require a policy-version bump and golden hard-negative review. Never attach canonical ids, URLs, permissions, or hidden collision details to the returned value.

- [ ] **Step 4: Implement the pure policy engine**

Validate exact candidate slot set, exact two-field set, known unique evidence slots, direct-fact scope, supported verdict, exact decimal threshold, empty uncertainty/conflict lists, and matching version identities. `unknown` is schema-valid but always human. A malformed batch is represented by one bounded failure result and makes every member human-review.

Run the server-side hidden/legacy-unknown `EXISTS` guard for every collision bucket even when one visible exact row exists. `reuse_trusted` is possible only for exactly one visible exact target and no hidden/legacy collision; visible multiplicity, mismatch, or any hidden existence is human-only. It identifies the exact canonical promotion target but does not bypass current evidence validation. The current candidate must still pass the same Terra structured validation, version/permission recheck, and deterministic policy threshold before the locked `reuse_existing` directive can run.

- [ ] **Step 5: Run eligibility/policy tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_eligibility.py backend/tests/test_auto_review_policy.py -q
uv run --locked ruff check backend/app/agent_runtime/auto_review_eligibility.py backend/app/agent_runtime/auto_review_input_safety.py backend/app/agent_runtime/auto_review_policy.py backend/app/agent_runtime/canonical_sources.py backend/app/knowledge/trusted_fingerprint_projection.py backend/app/knowledge/trusted_provenance.py backend/tests/test_auto_review_eligibility.py backend/tests/test_auto_review_policy.py
```

- [ ] **Step 6: Commit the deterministic authority slice**

```powershell
git add backend/app/agent_runtime/auto_review_eligibility.py backend/app/agent_runtime/auto_review_input_safety.py backend/app/agent_runtime/auto_review_policy.py backend/app/agent_runtime/canonical_sources.py backend/app/knowledge/trusted_fingerprint_projection.py backend/app/knowledge/trusted_provenance.py backend/tests/test_auto_review_eligibility.py backend/tests/test_auto_review_policy.py
git commit -m "feat: enforce deterministic auto review policy"
```

---

### Task 8: Add the Real LangChain Terra Validator Boundary

**Files:**
- Create: `backend/app/agent_runtime/auto_review_validator.py`
- Modify: `backend/app/agent_runtime/model_router.py`
- Create: `backend/tests/test_auto_review_validator.py`
- Create: `backend/tests/test_auto_review_model_router.py`
- Modify: `backend/tests/test_langchain_langgraph_dependency_compat.py`

**Interfaces:**
- Adds an OpenAI-only route fixed to `gpt-5.6-terra`, reasoning effort `medium`, prompt `auto-review-validation:v1`, structured output required, 6,000 framed input tokens, 3,072 total output tokens, `max_retries=0`, and no Luna/Sol/Gemini/provider-order fallback. One signed validation ceiling therefore authorizes at most one paid provider attempt per bounded batch.
- Builds a fresh per-call validator with a per-call usage sink and the frozen two-phase `prepare_many(requests) -> PreparedValidationInvocation` / `invoke_prepared(invocation, ProviderAttemptGrant) -> list[CandidateValidationResult]` protocol, without shared mutable `last_usage` state or a second render. The grant is created only by Task 9 after the database attempt marker commits.
- Calls `chat_model.with_structured_output(CandidateValidationBatchResult, method='json_schema', strict=True, include_raw=True)` and invokes it once per bounded batch. The estimator serializes this exact native structured-output schema/response framing; LangChain/OpenAI defaults may not choose the transport implicitly.
- Treats provider errors, timeout, parse error, usage ambiguity, and any batch-integrity violation as sanitized human-review fallback; no provider exception text reaches persistence or API.
- Forces explicit model `verbose=False`, a per-call `langsmith.tracing_context(enabled=False)` boundary, an empty internal callback list, and `cache=False` on the isolated model/runnable. Readiness and the final invoke guard require LangChain debug off, `OPENAI_LOG` not debug, and the effective `openai` SDK logger above DEBUG; they never mutate those process-global controls. Validator construction accepts no external tracer/callback/cache or HTTP debug-hook injection. The only custom HTTP hook is the server-owned body-blind send-fence hook, whose contract exposes deadline/attempt metadata only and never reads/logs request or response bodies.

- [ ] **Step 1: Write router, structured-output, privacy, and bounds tests**

Use a fake chat model that records `with_structured_output` and `invoke` calls. Assert:

- `test_live_route_constructs_chat_openai_terra_medium_without_fallback`
- `test_validator_uses_with_structured_output_and_exact_schema`
- `test_dependency_compat_freezes_json_schema_strict_bound_kwargs_and_framing`
- `test_prompt_treats_evidence_as_data_not_instruction`
- `test_prompt_contains_only_local_slots_claims_and_bounded_text`
- `test_prompt_excludes_ids_urls_permissions_credentials_and_candidate_keys`
- `test_four_candidate_twelve_slot_12000_char_6000_input_3072_total_output_bounds`
- `test_prepared_invocation_is_rendered_once_counted_and_sent_byte_for_byte`
- `test_global_langsmith_tracing_and_callbacks_cannot_capture_raw_validation_messages`
- `test_langchain_global_debug_true_is_zero_call_and_stdout_stderr_receive_zero_prompt_bytes`
- `test_model_is_explicitly_verbose_false_even_when_global_verbose_was_enabled_at_construction`
- `test_openai_log_or_effective_sdk_debug_is_zero_call_and_caplog_has_zero_prompt_bytes`
- `test_unapproved_http_event_hook_is_rejected_and_fenced_hook_is_body_blind`
- `test_process_global_langchain_cache_is_never_read_or_written`
- `test_sensitive_input_scanner_causes_zero_prepare_invoke_trace_callback_and_cache_calls`
- `test_any_missing_duplicate_unknown_or_extra_slot_rejects_whole_batch`
- `test_provider_and_parse_errors_are_sanitized`
- `test_per_call_usage_sink_records_one_bounded_token_cost_record`
- `test_route_disables_sdk_retries_and_one_batch_cannot_exceed_one_provider_attempt`


The fake returns an `AIMessage`/raw wrapper with deterministic usage metadata. No live API key or network is used.

- [ ] **Step 2: Run validator/router tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_validator.py backend/tests/test_auto_review_model_router.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
```

- [ ] **Step 3: Add an isolated Terra route**

Implement a route builder independent of `_available_provider_routes()`:

```python
ChatOpenAI(
    model='gpt-5.6-terra',
    api_key=settings.openai_api_key,
    reasoning_effort='medium',
    use_responses_api=True,
    timeout=provider_attempt_grant.timeout_seconds,
    max_retries=0,
    max_completion_tokens=AUTO_REVIEW_MAX_OUTPUT_TOKENS,
    verbose=False,
    cache=False,
)
```

For the pinned `langchain-openai==1.6.0` contract, `max_completion_tokens=3072` is the constructor field and `use_responses_api=True` deterministically serializes it as Responses API `max_output_tokens=3072`; the compatibility test asserts that exact native body key. Extraction uses the same pinned mapping with `max_completion_tokens=2048`, `reasoning_effort='none'`, and `use_responses_api=True`. Usage parsing treats Responses `output_tokens` as visible plus reasoning tokens and rejects absent/ambiguous usage rather than pricing a smaller visible-only count.

Construct the network-capable model only after attempt admission and inject `FencedOpenAITransport(provider_attempt_grant.permit)`; preparation may bind the frozen schema but cannot own a network client or permit. Do not set a creative temperature if the model/API rejects it. Do not add a retry above or below this boundary unless the cost contract, signed preview multiplier, usage accounting, and approval are all revised together. Readiness requires OpenAI key, configured validation prices, supported registry identities, explicit model `verbose=False`, LangChain global debug off, OpenAI SDK debug logging off, and only the reviewed body-blind transport hook; failure returns `model_unavailable` internally and leaves candidates pending.

- [ ] **Step 4: Implement bounded rendering and result integrity**

Use a fixed system instruction that says evidence blocks are untrusted data and that only the schema may be returned. Bind the runnable once with `method='json_schema', strict=True, include_raw=True`; compatibility fixtures assert the exact bound kwargs and canonical JSON-schema serialization consumed by `openai-o200k-chat:v1`. `prepare_many()` serializes slots exactly once and returns an ephemeral immutable `PreparedValidationInvocation` containing the exact LangChain messages/native response-schema framing to send, character count, frozen estimator token count, completion cap, and keyed content HMAC. Immediately before attempt admission/invoke, reject with zero call if LangChain debug, model/runnable verbose, `OPENAI_LOG=debug`, effective `openai` logger DEBUG, or an unapproved HTTP hook is present; do not temporarily mutate any global. `invoke_prepared(invocation, grant)` enters `langsmith.tracing_context(enabled=False)`, supplies an empty server-owned callback list, forces the isolated runnable/model cache off, uses only the grant's timeout and one-use fenced transport, and passes that same prepared message tuple to the single structured-output runnable; it may not rebuild aliases/messages/schema, accept raw requests, reuse a permit, or accept caller callbacks/tracers/caches. Install a fake process-global cache and capture `caplog`, stdout/stderr, callback, and HTTP-hook events in tests; prove cache lookup/update and raw prompt bytes are all zero. Validate the returned candidate set and every field/slot relationship against the prepared request after Pydantic parsing. Discard the entire parsed batch on any mismatch.

Create a validator factory whose `create(usage_sink)` returns a per-invocation adapter. Capture only input/output tokens and derived cost; never retain raw messages or raw parsed output after the coordinator persists bounded results.

- [ ] **Step 5: Run validator/router tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_validator.py backend/tests/test_auto_review_model_router.py backend/tests/test_langchain_langgraph_dependency_compat.py -q
uv run --locked ruff check backend/app/agent_runtime/auto_review_validator.py backend/app/agent_runtime/model_router.py backend/tests/test_auto_review_validator.py backend/tests/test_auto_review_model_router.py backend/tests/test_langchain_langgraph_dependency_compat.py
```

- [ ] **Step 6: Commit the real LangChain boundary**

```powershell
git add backend/app/agent_runtime/auto_review_validator.py backend/app/agent_runtime/model_router.py backend/tests/test_auto_review_validator.py backend/tests/test_auto_review_model_router.py backend/tests/test_langchain_langgraph_dependency_compat.py
git commit -m "feat: validate review candidates with langchain terra"
```

---

### Task 9: Persist One-Call Validation Leases, Atomic Cost Ledger, Cache, and Revalidation

**Files:**
- Create: `backend/app/agent_runtime/auto_review_validation_store.py`
- Create: `backend/app/agent_runtime/auto_review_orchestrator.py`
- Modify: `backend/app/review/auto_review_resolution.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_auto_review_validation_store.py`
- Create: `backend/tests/test_auto_review_orchestrator.py`
- Create: `backend/tests/test_auto_review_postgres.py`

**Interfaces:**
- Computes the exact workflow-bound validation-key HMAC over workflow execution identity, security-scope HMAC, candidate key, evidence-version hash, normalized claim fingerprint, generation fingerprint, validator provider/model/reasoning, prompt/output-contract version, policy version (which inseparably fixes `credential-scan:v1`), fingerprint-key version/material verifier, token-estimator version/encoding, input/output caps, maximum candidates per batch, batches per workflow, candidates per workflow, attempts, shared provider timeout/send-window/lease/grace seconds, exact validator input/output price snapshots, cost-policy version, validation provider-safety state version, rollout control epoch/percentage/authorization generation, and immutable confirmed extraction/validation/total ceilings. `candidate_key` itself remains frozen as workflow-bound; the explicit workflow/scope inputs and composite uniqueness make cross-workflow replay impossible even if a future candidate-key implementation regresses.
- Claims one short `AutoReviewValidationCall` lease per canonical sorted batch; only an expired lease can be compare-and-swapped. Its child validation keys are claimed atomically, and completed canonical results are replayed after identity verification.
- Batches at most 4 eligible candidates with at most 12 unique evidence slots, 12,000 characters, 6,000 framed input tokens, and 3,072 total output tokens, with at most 2 validation batches/5 candidates per workflow, while preserving candidate-local exact field/slot integrity.
- Commits the lease before calling Terra, calls the validator with no open database transaction, then opens a new transaction to revalidate everything and persist a bounded decision/cost.
- Persists observations only in this slice. Until Task 10 injects the locked rollout/audit authority, even a stored `enforce` request is effectively shadow and cannot call the internal approval service. Provider/model/parse failures leave items pending; canonically proven source drift uses Task 4's internal `mark_evidence_stale` boundary.
- Resolves the workflow owner's current server-side `PermissionContext` before input assembly and again after the provider call. The internal auto actor can narrow that context but can never substitute its own public/internal capability for access the owner has lost.

- [x] **Step 1: Write key, lease, restart, cost, and transaction-boundary tests**

Cover:

- `test_validation_key_changes_for_every_frozen_identity_component`
- `test_validation_key_changes_with_provider_safety_state_version_or_rollout_control_epoch`
- `test_same_content_in_two_workflows_and_scopes_never_collides_or_cross_replays`
- `test_completed_validation_replays_without_a_second_validator_call`
- `test_failed_validation_replay_is_explicit_sanitized_and_keeps_every_member_pending`
- `test_completed_validation_with_zero_projection_is_corrupt_not_failed_sentinel`
- `test_only_expired_claim_can_be_reclaimed_with_cas`
- `test_concurrent_claims_produce_one_canonical_result`
- `test_candidate_permutation_produces_identical_batches_aliases_and_fingerprints`
- `test_single_candidate_over_any_cap_is_zero_call_human_only`
- `test_batch_usage_is_charged_once_and_child_allocations_sum_exactly`
- `test_usd_reservations_and_child_allocations_use_six_place_round_ceiling`
- `test_micro_usd_largest_remainder_child_costs_are_nonnegative_and_sum_to_call`
- `test_workflow_budget_reservation_is_atomic_across_concurrent_batches`
- `test_reservation_cannot_exceed_stored_signed_validation_or_total_ceiling`
- `test_exact_framed_token_count_must_fit_signed_reservation_before_attempt_marker`
- `test_unicode_input_never_uses_character_division_as_token_estimate`
- `test_attempt_start_uses_signed_shared_timing_and_lease_exceeds_send_timeout_grace`
- `test_validation_send_permit_is_one_use_nonserializable_and_terminal_call_never_sends`
- `test_validation_store_owns_lease_token_and_database_timestamps_despite_host_clock_skew`
- `test_validation_attempt_grant_exists_only_after_marker_commit`
- `test_validation_complete_and_fail_reject_wrong_session_or_unlocked_context`
- `test_cancel_between_validation_marker_and_send_latches_until_owner_or_expiry`
- `test_old_workflow_uses_stored_timeout_not_mutable_global_agent_timeout`
- `test_permission_or_source_changes_after_dto_before_attempt_marker_causes_zero_call`
- `test_provider_price_or_cost_policy_drift_before_attempt_marker_causes_zero_call`
- `test_final_prepared_batch_is_rendered_once_and_claim_stores_exact_hmac_and_counts`
- `test_frame_sizer_matches_prepared_json_schema_framing_golden_fixtures`
- `test_unknown_failure_usage_charges_reserved_ceiling_and_never_retries`
- `test_known_usage_over_reservation_records_actual_overrun_opens_readiness_failure_and_never_approves`
- `test_validation_overrun_atomically_appends_one_call_attributed_provider_safety_event`
- `test_overrun_replay_does_not_duplicate_event_or_advance_sequence`
- `test_expired_pre_attempt_lease_may_reclaim_but_started_attempt_never_retries`
- `test_expired_started_attempt_never_releases_workflow_reservation`
- `test_crash_after_provider_send_charges_reserved_ceiling_and_routes_human`
- `test_provider_call_observes_no_open_session_or_row_lock`
- `test_busy_live_lease_does_not_refresh_or_interrupt_and_only_same_node_retries`
- `test_source_permission_or_content_change_discards_provider_result`
- `test_owner_loses_internal_before_input_assembly_causes_zero_call_and_no_leak`
- `test_owner_loses_internal_while_provider_runs_cannot_approve_or_persist_content`
- `test_batch_and_workflow_cost_caps_prevent_provider_call`
- `test_malformed_or_timeout_result_leaves_candidate_pending`
- `test_source_drift_transitions_to_needs_more_evidence_without_approval`
- `test_restart_reuses_completed_validation_and_cost_metadata`
- `test_persisted_validation_contains_no_raw_evidence_url_or_exception`
- `test_missing_rollout_authority_demotes_enforce_to_shadow_and_never_approves`
- `test_claim_and_attempt_marker_lock_rollout_before_source_and_reject_stale_control`
- `test_reverse_route_set_claim_and_rollout_admin_interleaving_has_no_deadlock`
- `test_cancel_before_validation_attempt_marker_is_zero_call_zero_charge_and_releases_reserve`
- `test_cancel_during_validation_call_charges_once_and_persists_no_result_or_approval`
- `test_late_validation_completion_after_cancel_or_restart_cannot_create_decision_or_audit`


The PostgreSQL test uses two sessions and barriers rather than sleeps to prove one batch lease/result, one charged call, one budget winner, and later—after Task 10—one transition winner.

- [x] **Step 2: Run validation-store/orchestrator tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_validation_store.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py -q
```

- [x] **Step 3: Implement canonical claim/replay/CAS operations**

Expose operations with no provider dependency:

```python
@dataclass(frozen=True)
class AutoReviewValidationIdentity:
    workflow_execution_identity_hmac: str
    security_scope_hmac: str
    candidate_key: str
    evidence_version_hash: str
    normalized_claim_fingerprint: str
    candidate_generation_fingerprint: str
    validator_provider: Literal['openai']
    validator_model: Literal['gpt-5.6-terra']
    reasoning_effort: Literal['medium']
    validator_prompt_version: Literal['auto-review-validation:v1']
    validator_output_contract_version: Literal['candidate-validation-batch:v1']
    policy_version: Literal['auto-review-policy:v1']
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    token_estimator_version: Literal['openai-o200k-chat:v1']
    tokenizer_encoding: Literal['o200k_base']
    max_input_tokens: Literal[6000]
    max_output_tokens: Literal[3072]
    max_candidates_per_batch: Literal[4]
    max_batches_per_workflow: Literal[2]
    max_candidates_per_workflow: Literal[5]
    max_provider_attempts: Literal[1]
    provider_timeout_seconds: int
    provider_send_start_window_seconds: int
    provider_attempt_lease_seconds: int
    provider_commit_grace_seconds: int
    input_cost_per_1m_tokens: Decimal
    output_cost_per_1m_tokens: Decimal
    cost_policy_version: Literal['auto-review-cost:v1']
    provider_safety_state_version: int
    rollout_control_epoch: int
    authorized_percentage_at_launch: Literal[0, 10, 100]
    rollout_authorization_generation: int
    confirmed_extraction_cost_ceiling_usd: Decimal
    confirmed_validation_cost_ceiling_usd: Decimal
    confirmed_total_cost_ceiling_usd: Decimal

@dataclass(frozen=True)
class ValidationCandidateClaim:
    review_item_id: int
    validation_key: str
    identity: AutoReviewValidationIdentity

@dataclass(frozen=True)
class ValidationBatchClaimRequest:
    workflow_thread_id: str
    batch_fingerprint: str
    candidates: tuple[ValidationCandidateClaim, ...]
    max_provider_attempts: Literal[1]
    prepared_content_hmac: str
    serialized_char_count: int
    framed_input_tokens: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_cost_usd: Decimal

@dataclass(frozen=True)
class CompletedValidationProjection:
    validation_call_id: int
    validation_id: int
    review_item_id: int
    result: CandidateValidationResult
    policy_decision: AutoReviewPolicyDecision
    policy_reason_codes: tuple[AutoReviewPolicyReasonCode, ...]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal
    cache_hit: bool

@dataclass(frozen=True)
class ValidationClaimResult:
    disposition: Literal['claimed', 'replayed', 'busy']
    validation_call_id: int
    lease_token: str | None
    lease_expires_at: datetime | None
    terminal_status: Literal['completed', 'failed'] | None
    failure_reason_code: AutoReviewPolicyReasonCode | None
    completed: tuple[CompletedValidationProjection, ...]

@dataclass(frozen=True)
class ValidationAttemptAdmission:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str
    prepared_content_hmac: str
    serialized_char_count: int
    framed_input_tokens: int
    max_output_tokens: int
    recomputed_reserved_cost_usd: Decimal
    expected_owner_permission_hmac: str
    expected_source_state_hmac: str
    fingerprint_key_version: str
    fingerprint_key_material_verifier: str
    token_estimator_version: str
    cost_policy_version: str
    expected_effective_mode: Literal['shadow', 'enforce']
    provider_timeout_seconds: int
    provider_send_start_window_seconds: int
    provider_attempt_lease_seconds: int
    provider_commit_grace_seconds: int

@dataclass(frozen=True)
class ProviderAttemptGrant:
    permit: FencedProviderSendPermit
    timeout_seconds: int
    lease_expires_at: datetime

@dataclass(frozen=True)
class ValidationCandidateCompletion:
    review_item_id: int
    validation_key: str
    result: CandidateValidationResult
    policy_decision: AutoReviewPolicyDecision
    policy_reason_codes: tuple[AutoReviewPolicyReasonCode, ...]

@dataclass(frozen=True)
class ValidationBatchCompletion:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str
    candidates: tuple[ValidationCandidateCompletion, ...]
    input_tokens: int
    output_tokens: int

@dataclass(frozen=True)
class ValidationBatchFailure:
    validation_call_id: int
    workflow_thread_id: str
    batch_fingerprint: str
    lease_token: str
    reason_code: AutoReviewPolicyReasonCode
    usage_known: bool
    input_tokens: int | None
    output_tokens: int | None

class AutoReviewValidationStore:
    def claim_or_replay(
        self, request: ValidationBatchClaimRequest
    ) -> ValidationClaimResult:
        raise NotImplementedError

    def mark_attempt_started(
        self, admission: ValidationAttemptAdmission
    ) -> ProviderAttemptGrant:
        raise NotImplementedError

    def complete(
        self,
        context: ValidationLockedContext,
        completion: ValidationBatchCompletion,
    ) -> tuple[CompletedValidationProjection, ...]:
        raise NotImplementedError

    def fail(
        self,
        context: ValidationLockedContext,
        failure: ValidationBatchFailure,
    ) -> None:
        raise NotImplementedError
```

`ValidationClaimResult` has exact state invariants: `claimed` carries the store-owned lease token/expiry and no terminal status/result; `busy` carries no lease token or terminal result (it may expose only the bounded canonical expiry needed for retry scheduling); `replayed+completed` carries a non-empty 1–4 projection tuple and no failure code; `replayed+failed` carries an empty projection tuple plus one allowlisted sanitized failure code. A completed call with zero projections, mixed terminal children, or an implicit empty-tuple failure sentinel is corrupt and fails closed. Failed replay deterministically leaves every batch member pending for human review without a second provider attempt.

Before claiming, sort candidates by canonical `validation_key`. For each candidate, sort/deduplicate its global canonical evidence identities and assign batch-local `E01..E12` aliases from the sorted union; assign `C01..C04` from validation-key order. Apply deterministic first-fit in candidate order with a pure `ValidationFrameSizer` that canonicalizes the candidate/evidence JSON plus the frozen native JSON-schema framing and returns characters/framed tokens/cost without constructing LangChain messages or a runnable invocation. Test existing batches from index 1 upward and place the candidate in the first batch within candidate/evidence/character/framed-token/output/cost caps, otherwise open the next batch. Golden fixtures prove the sizer is byte/token-equivalent to the final `PreparedValidationInvocation`; any mismatch is `registry_unavailable` and disables validation. After partitioning, call `prepare_many()` exactly once for each final batch and keep that immutable object through claim, attempt admission, and invoke. A candidate that cannot fit alone is zero-call human-only. Candidate 6, or any candidate that would require work beyond the second batch, is also human-only and must be handled by another separately previewed workflow; a conforming five-agent extraction set cannot produce candidate 6 because each selected agent is contractually limited to zero or one candidate. No hash-map/database iteration order may influence grouping. Thus permuted workers derive identical partitions, aliases, batch fingerprints, and child claims.

`claim_or_replay` acquires the shared generation barrier and runtime row, locks all required provider-safety rows in lexical order, takes the projection lock when readiness/collision authority is consumed, locks the sorted rollout rows, then sorted current Source rows `FOR SHARE`, and only then the immutable workflow request/cost boundary. It rechecks membership and control identity after those locks. The store creates the lease token and derives claim/expiry timestamps from PostgreSQL `clock_timestamp()` plus the locked immutable timing policy; no request timestamp, host wall clock, or stale worker can extend/backdate a lease. SQLite/unit tests use an injected DB-clock abstraction, while only the post-commit send permit uses process-local monotonic time. Validation cannot begin until every selected extraction call is terminal; any live extraction call returns bounded in-progress and starts no validator. `extraction_obligation` is the exact sum of final Numeric extraction-call charges, never a legacy `AgentRun.estimated_cost_usd` float, and must remain within the signed extraction ceiling. It computes `validation_obligation = SUM(final validation charges, non-final validation reserves) + next_reserved_cost`; every claimed validation call contributes its reservation regardless of lease expiry. It requires at most 2 calls and 5 candidates for the workflow, the next batch reserve <= USD 0.048864, `validation_obligation <= min(USD 0.097728, stored confirmed_validation_cost_ceiling)`, and `extraction_obligation + validation_obligation <= min(stored confirmed_total_cost_ceiling, stored exact V2.1 total budget)`. All price multiplication, reservation, charge, and total comparisons use `Decimal` quantized to six USD places with `ROUND_CEILING`; float conversion is forbidden. It never substitutes a global cap for the smaller amount the user signed. Only then does it derive/verify keyed identities and own the unique batch plus child validation-key rows atomically, persisting the already-prepared content HMAC, exact character/framed-token counts, and reserve. The batch fingerprint covers the deterministic partition/aliases, sorted child validation identities, workflow/scope, key version/material, estimator/encoding/token caps, maximum batches/candidates/attempts, all four shared provider timing values, exact price tuple, provider-safety state version, cost policy, rollout control snapshot, and all signed ceilings; the call row stores the same frozen columns. Replay verifies every frozen identity column, not the HMAC alone.

A lease may be reclaimed only while `provider_attempt_count=0`; reclaim changes ownership/expiry but never releases its existing reservation. A pre-attempt call that is terminally abandoned, globally disabled, or cancelled may release the reservation only while holding the same workflow-request lock and atomically finalizing that canonical call as zero-charge human fallback. Immediately before the network call, a separate short CAS transaction follows the same `shared generation/runtime -> sorted required safety -> projection when consumed -> sorted rollout -> sorted Source -> workflow -> call` order, revalidates current permission/source/key/configured mode/rollout breaker/provider breaker/cancellation, prepared-object identity, exact framed-token bounds, all four stored timing values, and current purpose/provider/model/reasoning/cost-policy/price registry tuple. It sets `provider_attempt_count=1`, `attempt_started_at=database clock_timestamp()`, and `lease_expires_at=attempt_started_at+stored_provider_attempt_lease_seconds`, commits, and only afterward constructs/returns `ProviderAttemptGrant`. Every price change must ship with a new `cost_policy_version`; any version or exact price drift from the immutable workflow/call snapshot is a zero-call human fallback, because a stored V2.1 request cannot obtain a replacement signed ceiling. Once the marker exists, expiry/restart can never invoke the provider again or release the reserve. `complete` requires the same marked call/lease and must arrive within the database-clock lease; a lost/late result after attempt start atomically finalizes bounded failure with the full reserved charge and human fallback.

`mark_attempt_started()` re-resolves the server-owned owner/source/execution guard, reads the current registry purpose/provider/model/reasoning/cost-policy and exact price/timing tuple, recalculates cost inside the store from the admission token counts and frozen call-row prices, and compares every `ValidationAttemptAdmission` identity/count/HMAC/timing value plus current registry values to the immutable request/call row under CAS. The caller's `recomputed_reserved_cost_usd` is an equality assertion, never billing or time authority; any mismatch finalizes zero-call human fallback before the attempt marker. The returned `ProviderAttemptGrant` contains only the shared one-use permit, stored timeout, and database lease expiry; it cannot exist before commit or be reconstructed after restart.

The call row is authoritative for provider usage. `complete`/`fail` accept token counts, never a caller-supplied price or cost; the store computes the authoritative call charge from immutable input/output price snapshots with six-place `ROUND_CEILING`. Allocate integer input/output tokens to children by `divmod(total, candidate_count)` in sorted `validation_key` order. Child cost is observational, not separately rounded billing: convert the authoritative call charge to integer micro-USD, compute each child's unquantized weighted share from its allocated input/output tokens, assign floor micro-USD, then distribute remaining non-negative micro-USD by largest fractional remainder with `validation_key` as tie-breaker. Every child cost is non-negative and their exact sum equals the one call charge; never force an over-rounded remainder into the last child. Workflow accounting sums call rows, never child rows. When a provider failure has ambiguous usage, finalize the call once with its full reserved ceiling as charged cost; `max_retries=0` and replay can never make another attempt. If known provider-reported tokens or the resulting exact charge exceed the stored reserve/signed validation ceiling, persist the actual tokens/charge plus `budget_overrun=true` and exact `budget_overrun_cost_usd`, make every child human-only, and in that same locked/CAS transaction update the exact purpose/provider/model/reasoning `AutoReviewProviderSafetyState`, append exactly one next-sequence `budget_overrun` event attributed by the domain-separated call HMAC and the call's key identities, and set the aggregate's `last_event_id/sequence` before commit. Replay recognizes the already-bound call event and cannot append another. That distinct breaker makes all later shadow/enforce attempts zero-call across scopes; it never mutates key-runtime readiness or silently clears on process restart. The final actual charge remains in the workflow obligation. Never clamp, approve, retry, overwrite an older event, or place an operator actor on the call-attributed event. Persist only allowlisted codes, exact Decimal scores/cost, counts, and tokens.

Define `budget_overrun` with like-for-like comparisons: actual provider input tokens exceed the stored framed-input cap; actual output tokens exceed the stored output cap; recomputed USD charge exceeds this call's USD reserve; the workflow's post-completion sum of final validation charges plus other non-final validation reserves exceeds the signed validation ceiling; or the authoritative final extraction charge sum plus that validation obligation exceeds the signed total ceiling or stored exact V2.1 total budget. Any one condition opens the same purpose-specific validation breaker. Never compare tokens directly with USD, reuse a float mirror, or ignore a workflow-level overrun because the individual call stayed below its local reserve.

- [x] **Step 4: Implement provider-outside-transaction orchestration**

Use this exact phase boundary:

```text
short read phase R:
  load pending candidates and immutable refs
  resolve workflow owner and current allowed permission levels
  canonical preflight + exact collision guard
  close/rollback the read transaction and release every ORM object/Session lock

no database transaction:
  derive canonical validation identities from the frozen read values
  partition with ValidationFrameSizer and its frozen equivalence fixtures
  build each final immutable PreparedValidationInvocation exactly once
  retain only the bounded ephemeral objects needed for claim/invoke

transaction A:
  acquire shared key-generation barrier and runtime key row FOR SHARE
  lock sorted provider-safety rows, projection when readiness is consumed,
    sorted rollout rows, then sorted Source rows FOR SHARE
  lock immutable workflow request/cost row
  re-resolve owner permission and every source/version/signature/content identity
  verify the read snapshot, current purpose/provider/model/reasoning/cost-policy/price tuple,
    provider breaker, effective mode, rollout snapshot, and all signed caps
  atomically reserve the signed batch/workflow ceiling and claim/replay the
    call plus child keys while storing prepared content HMAC and exact counts
  commit; on replay/busy/mismatch discard the prepared object and never call

short transaction A3:
  acquire shared key-generation barrier and runtime key row FOR SHARE
  lock sorted provider-safety rows, projection when consumed, sorted rollout rows,
    sorted Source rows FOR SHARE,
    immutable workflow row, then call/children
  CAS the same live call lease with provider_attempt_count == 0
  re-resolve owner permission and bound source version/signature/content fingerprint
  recheck stored key version/material, purpose/provider/model/reasoning/cost-policy/current price,
    provider breaker, configured mode, rollout-breaker state, and cancellation
  compare the same prepared object's HMAC/counts with stored values and caps
  set provider_attempt_count = 1 and attempt_started_at from database clock_timestamp(),
    then lease_expires_at = attempt_started_at + stored_provider_attempt_lease_seconds
  commit before any network send; only after commit return ProviderAttemptGrant

no database transaction:
  FencedOpenAITransport consumes the grant's one-use permit at actual dispatch,
  applies the stored provider timeout, and invokes one per-call Terra validator
  with the exact prepared object; redirect/retry/second dispatch is rejected
  validate whole structured batch
  discard plaintext DTO, aliases, and rendered messages after parsing

transaction B:
  acquire shared key-generation barrier and runtime key row FOR SHARE
  lock sorted provider-safety rows
  acquire global projection advisory lock
  lock rollout row when Task 10 is installed
  lock sorted Source rows FOR SHARE
  lock immutable workflow cost row, then validation call/children
  CAS batch-call lease owner under that already-held context
  re-resolve workflow cancellation, owner and current allowed permission levels
  re-resolve source/version/signature/content/permission
  re-run projection marker/key-material/readiness/anti-join,
    duplicate/hidden-collision, and policy registry checks
  if cancelled, discard parsed output, persist actual-or-reserve charged failure,
    and create no validation child result, promotion decision, audit, or approval
  otherwise persist bounded validation + decision + usage/cost
  do not approve until Task 10 supplies locked rollout/audit authority
  commit
```

Keeping the bounded plaintext prepared object in process only through claim/admission/provider parsing is the necessary and only exception to the no-raw-content rule. Do not keep ORM objects or a `Session` across preparation or the call; do not serialize the DTO, aliases, prompt, prepared schema/messages, grant, or permit into checkpoint, validation, logs, cache, or AuditLog. Carry immutable HMAC identities into transaction B and reconstruct current canonical state there. A provider failure or crash after the attempt marker finalizes the call ledger once with sanitized fallback and known actual usage or the conservative reserved charge—never a second attempt—without raw exception, and leaves ReviewItems pending. Lease expiry is a database-clock timestamp condition, not a terminal cost state: an expired pre-attempt lease may CAS-reclaim without changing its reservation, while an expired post-attempt lease may only atomically finalize failure with the reserved charge under the workflow lock. V2.1 cancel uses that same workflow lock: attempt zero becomes terminal zero-charge with its reservation released; after attempt one, cancellation only latches output discard and never terminalizes until the owner completes or the stored lease expires. A cancel racing between marker and send is conservatively possibly sent; at most the one live permit can dispatch, and transaction B creates no validation/promotion/audit/approval. Transaction B always re-reads cancellation before persisting parsed results, so a late completion cannot create a validation result, promotion decision, selected audit, or approval. Restart has no permit and recovery applies the identical rule only at/after lease expiry.

`complete()`/`fail()` receive an internal already-held `ValidationLockedContext` proving the workflow row precedes call/children; recovery uses the same order and no store method may lock the workflow after a call row. `ValidationClaimResult(disposition='busy')` is not a human fallback. The V2.1 service returns bounded in-progress/`concurrent_resume`, does not execute `refresh_review_resolution`, does not create a human interrupt, and permits only the same `run_auto_review` node to be retried after the canonical owner completes or the lease is finalized. A replayed completed/failed canonical result may then advance normally; this prevents a late owner from auto-approving behind an already-presented human boundary.

- [x] **Step 5: Run validation-store/orchestrator tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_validation_store.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py -q
uv run --locked ruff check backend/app/agent_runtime/auto_review_validation_store.py backend/app/agent_runtime/auto_review_orchestrator.py backend/app/review/auto_review_resolution.py backend/app/main.py backend/tests/test_auto_review_validation_store.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py
```

- [x] **Step 6: Commit validation coordination**

```powershell
git add backend/app/agent_runtime/auto_review_validation_store.py backend/app/agent_runtime/auto_review_orchestrator.py backend/app/review/auto_review_resolution.py backend/app/main.py backend/tests/test_auto_review_validation_store.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py
git commit -m "feat: coordinate cached auto review validation"
```

---

### Task 10: Add Shadow Evidence, Canary Sampling, Human Audit, and Breaker

**Files:**
- Create: `backend/app/review/auto_review_rollout.py`
- Create: `backend/app/review/auto_review_audit.py`
- Create: `backend/app/review/auto_review_quality_revoke.py`
- Create: `backend/app/admin/auto_review_rollout.py`
- Modify: `backend/app/review/auto_review_audit_transitions.py`
- Modify: `backend/app/agent_runtime/auto_review_orchestrator.py`
- Modify: `backend/app/review/transitions.py`
- Modify: `backend/app/review/auto_review_resolution.py`
- Create: `backend/tests/test_auto_review_rollout.py`
- Create: `backend/tests/test_auto_review_audit.py`
- Create: `backend/tests/test_auto_review_quality_revoke.py`
- Create: `backend/tests/test_auto_review_rollout_admin.py`
- Modify: `backend/tests/test_auto_review_orchestrator.py`
- Modify: `backend/tests/test_auto_review_postgres.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`

**Interfaces:**
- Resolves effective mode from stored mode, current global demotion, persistent breaker, shadow evidence, configured percentage, and a persistent maximum-authorized-percentage latch. It can only demote; metrics alone can never advance or re-enable a stage.
- Treats a missing rollout row through read-only `peek_or_default()` as latch 0, breaker false, control epoch 0, authorized percentage 0, and authorization generation 0; preview never inserts. Start calls `ensure_row()` under lock and rejects any preview/start **control** snapshot difference before thread creation or provider work. Ordinary shadow/audit counters use `state_version` for admin CAS but do not change the separate `control_epoch`, so unrelated activity cannot starve launches.
- Consults the exact purpose/provider/model/reasoning safety row before projection/rollout state. Missing, open, or cost-policy/price/estimator mismatch is zero-call; only the restricted admin control plane can authorize an initial policy or clear an overrun breaker after reviewed regression evidence.
- Uses a stable candidate HMAC for enforce percentage selection and a separate stable promotion/sample HMAC for audit cohorts.
- Creates one immutable `AutoReviewPromotionDecision` for **every** auto approval and creates a required `AutoReviewPostAudit` row in the same transaction only when that decision is selected for audit.
- Records shadow human comparison only after a later authorized human resolution with unchanged evidence version; drifted, unresolved, and skipped cases are excluded from the precision denominator.
- Commits a critical outcome and opens the breaker before attempting revoke. Revoke failure becomes `remediation_required` while the breaker stays open.
- Routes every quality-coded human revoke through the same breaker-first audit authority. A missing audit becomes `manual` critical, a pending audit becomes critical, and an already confirmed audit receives an immutable critical correction; none may use the normal direct-revoke path.
- Allows breaker close only through an admin service with `auto_review_rollout_admin`, bounded reason, resolved remediation, affected-item adjudication, regression-gate reference, and zero audit corrections on the current policy row. Closing never automatically re-enables enforce; any correction requires a reviewed new policy version and fresh rollout gates.

- [x] **Step 1: Write the rollout state machine and sampling tests**

Cover:

- `test_disabled_always_demotes_to_zero_call`
- `test_enforce_without_500_shadow_comparisons_stays_shadow`
- `test_shadow_precision_is_supported_completed_over_completed_only`
- `test_shadow_gate_can_only_authorize_ten_percent_canary`
- `test_canary_cannot_jump_to_hundred_percent`
- `test_stable_ten_percent_canary_selection_is_replay_safe`
- `test_concurrent_first_use_creates_one_default_rollout_row`
- `test_fresh_scope_preview_uses_read_only_default_and_writes_no_row`
- `test_preview_start_race_with_rollout_authorization_is_cost_preview_changed_before_thread`
- `test_unrelated_rollout_counter_change_does_not_invalidate_signed_preview`
- `test_sampling_fixed_vectors_cover_bucket_boundaries_restart_permutation_and_key_change`
- `test_first_fifty_mandatory_slots_all_require_audit`
- `test_full_mandatory_slots_block_unsampled_promotion_until_resolution`
- `test_invalidated_pending_slot_selects_next_eligible_ordinal_as_mandatory_replacement`
- `test_mandatory_replacement_never_counts_as_confirmed_until_human_confirmation`
- `test_concurrent_invalidation_confirmation_and_replacement_use_one_rollout_cas`
- `test_canary_ordinals_51_to_500_use_stable_ten_percent_audit`
- `test_sample_two_starts_only_for_new_authorized_full_enforce_workflows`
- `test_existing_ten_percent_workflow_never_upgrades_or_uses_sample_two`
- `test_latch_zero_workflow_never_strengthens_after_zero_to_ten_authorization`
- `test_only_new_preview_after_ten_to_hundred_authorization_can_store_full_enforce`
- `test_breaker_close_and_reauthorization_generation_never_revives_old_workflow`
- `test_percentage_never_increases_for_an_existing_workflow`
- `test_non_admin_and_auto_policy_actor_cannot_advance_authorization_latch`
- `test_breaker_or_global_demotion_while_validator_runs_prevents_approval`
- `test_auto_promotion_uses_projection_then_rollout_then_review_lock_order`
- `test_auto_promotion_cannot_deadlock_with_human_promotion_or_projection_rebuild`
- `test_critical_audit_commits_breaker_before_revoke`
- `test_critical_audit_commit_immediately_quarantines_auto_effect_across_api_rag_and_pgvector`
- `test_manual_critical_audit_quarantines_before_revoke_and_recovers_identically`
- `test_revoke_failure_or_crash_leaves_quarantine_active_until_recovery`
- `test_revoke_failure_leaves_remediation_required_and_effective_shadow`
- `test_breaker_survives_restart_and_never_closes_automatically`
- `test_admin_close_requires_capability_reason_remediation_and_gate_reference`
- `test_completed_audit_outcome_is_immutable_and_replay_does_not_increment_counters`
- `test_quality_revoke_without_audit_creates_manual_critical_audit_before_revoke`
- `test_quality_revoke_with_pending_audit_finalizes_critical_before_revoke`
- `test_quality_revoke_after_confirmed_audit_appends_correction_without_rewriting_outcome`
- `test_confirmed_audit_correction_increments_counter_opens_breaker_and_quarantines_before_revoke`
- `test_quality_revoke_replay_writes_one_assessment_correction_counter_and_revoke`
- `test_business_withdrawal_does_not_create_quality_correction_or_change_precision`
- `test_corrected_confirmed_audit_is_critical_in_effective_precision_and_gate_queries`
- `test_breaker_close_and_same_policy_reauthorization_refuse_nonzero_corrected_critical_count`
- `test_only_new_reviewed_policy_row_can_start_with_zero_corrections_and_fresh_gates`
- `test_restart_after_breaker_commit_recovers_pending_revoke_idempotently`
- `test_required_audit_then_revoke_interleavings_never_leave_pending_gate_or_bypass_audit`
- `test_every_auto_approval_records_immutable_selection_but_only_selected_creates_audit`
- `test_unsampled_manual_audit_reuses_selection_identity_without_incrementing_counters`
- `test_human_shadow_resolution_and_auto_promotion_share_rollout_then_item_lock_order`
- `test_human_reject_and_needs_more_update_shadow_comparison_exactly_once_under_rollout_then_item_lock`
- `test_recovery_unlocked_scan_and_direct_revoke_interleaving_follow_one_global_order`
- `test_provider_overrun_breaker_is_global_cross_scope_restart_safe_and_zero_call`
- `test_provider_breaker_never_auto_clears_and_requires_new_cost_policy_regression_gate`
- `test_every_rollout_control_cas_appends_one_next_sequence_event_and_updates_backpointer`
- `test_metric_only_counter_updates_append_no_control_event`
- `test_provider_initial_authorize_and_clear_append_operator_events_with_current_key_identity`
- `test_control_event_replay_or_lost_cas_never_duplicates_sequence`
- `test_old_and_new_key_generation_events_remain_append_only_and_verifiable`
- `test_human_audit_reason_is_normalized_persisted_immutable_and_absent_from_public_logs`
- `test_pending_and_source_invalidated_audits_have_no_human_reason`
- `test_rollout_admin_cli_uses_fixed_system_operator_and_never_accepts_subject_override`
- `test_rollout_admin_outputs_aggregate_state_only_and_uses_expected_version_cas`
- `test_rollout_admin_requires_purpose_and_cannot_mutate_the_other_purpose_row`
- `test_reverse_route_set_start_claim_and_admin_locks_follow_one_global_order`


Test replay/concurrency so promotion ordinal, selected/completed/critical counters, and shadow counts increment once.

- [x] **Step 2: Run rollout/audit tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_rollout.py backend/tests/test_auto_review_audit.py backend/tests/test_auto_review_quality_revoke.py backend/tests/test_auto_review_rollout_admin.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py backend/tests/test_agent_runtime_lifespan.py -q
```

- [x] **Step 3: Implement locked, replay-safe rollout projection**

`AutoReviewRolloutPolicyService.peek_or_default()` is strictly read-only. When `(security_scope_id, policy_version)` is missing, it returns the exact sentinel `control_epoch=0`, latch/authorized percentage `0`, authorization generation `0`, breaker false, counters 0; it never inserts or allocates a generation. Zero-call preview signs only the control epoch plus latch/authorization/breaker fields, not metric counters or their optimistic `state_version`. Start first verifies the token, then in its first write transaction uses `INSERT ... ON CONFLICT DO NOTHING` and `SELECT ... FOR UPDATE` to `ensure_row()`. The inserted row has exactly the sentinel values and no raw operator identity. It re-compares every signed control field before thread creation; a concurrent authorization/breaker/control change increments `control_epoch` and becomes `cost_preview_changed`, while shadow/audit counter changes do not invalidate cost/authority. Two untouched first users converge on one default row and the canonical idempotent workflow. Gate calculations use Decimal precision and persisted human outcomes only:

```text
shadow completed comparisons >= 500
shadow precision >= 0.99
hard-negative / permission / version / validator false-approval count == 0
corrected critical count == 0 and no unresolved correction/remediation
no open breaker
requested stored percentage is an authorized configured value
```

No counter or model result automatically changes configured mode, percentage, or the authorization latch. Metric/audit counter-only CAS updates advance aggregate `state_version` but append no control event and do not change `control_epoch`. Every successful authorization, breaker-open/close, or generation-invalidation CAS atomically appends exactly one next-sequence `AutoReviewRolloutControlEvent` whose primary kind and complete prior/new control snapshot cover all fields changed by that operation, then updates `last_event_id/sequence`; a lost CAS or replay appends none. After the shadow gate passes, an authorized operator may latch **at most 10%** for new canary workflows; the service rejects a direct `0 -> 100` jump. Raising the latch from 10 to 100 additionally requires at least 500 canary enforce promotions, exactly 50 human-confirmed mandatory audits, zero pending mandatory audits, every other selected audit completed, zero original or corrected critical outcomes, `corrected_critical_count=0`, no unresolved correction/remediation, and enforce audit precision `effective_confirmed / all_effective_human_completed_quality_audits >= 0.99`; source-invalidated null-outcome rows are excluded from both numerator and denominator, while a confirmed audit with a correction is excluded from the confirmed numerator and counted once as critical. Only then may a separate authorized operator action create a new authorization generation with a bounded regression-gate reference and its event. Existing workflows retain their stored percentage.

Effective percentage is `min(requested_percentage, authorized_percentage_at_launch, current_config_percentage, current_latch)`, and enforce additionally requires the stored `rollout_authorization_generation` to equal the current active authorization generation. Global disabled/shadow, an open breaker, or a generation mismatch demotes it to zero/shadow. Breaker close resets/keeps the latch at zero, invalidates the active generation, and leaves effective shadow. A later separate operator authorization issues a new generation that only a new signed preview/workflow can snapshot; old workflows never silently restore enforce.

Every required purpose-specific provider-safety row is locked in sorted `(purpose, provider, model, reasoning_effort)` order before projection/rollout/Source/workflow whenever they participate: `shared key-generation barrier -> AutoReviewRuntimeKeyState FOR SHARE -> sorted AutoReviewProviderSafetyState FOR UPDATE/SHARE -> projection advisory lock when needed -> sorted rollout rows -> sorted Source rows -> sorted workflow rows -> call/validation -> ReviewItem`. Preview reads a bounded set snapshot without mutation; start, extraction E1/E2, validation claim/A3, and transaction B require every workflow-bound extraction row plus the validation row to exist, be closed, and match the configured provider/model/reasoning, registry-owned estimator, exact registry price tuple, and authorized purpose-specific cost-policy version. An overrun transaction increments only its purpose row's `overrun_count` once, appends the Task 9 call-attributed overrun event, updates the aggregate backpointer, and opens that breaker atomically with the actual charge before any later purpose check can pass. Initial provider authorization and operator clear similarly append exactly one operator-attributed `AutoReviewProviderSafetyEvent` using the current key identity; they never overwrite older events or share a mutable actor slot.

- [x] **Step 4: Create audit selection in the promotion transaction**

Freeze selection as unsigned full-digest arithmetic, not language/runtime hashing. Enforce candidate selection uses HMAC-SHA256 domain `auto-review-enforce-selection:v1`; audit cohort selection uses `auto-review-audit-selection:v1`. Both canonicalize UTF-8/NFC JSON with sorted compact keys over the security-scope HMAC, workflow-execution HMAC, candidate key, policy version, stored rollout state/generation, requested and authorized percentages, fingerprint key version/material verifier; audit selection additionally binds promotion ordinal, the stored enforce-selection fingerprint, and the locked mandatory confirmed/pending counters before this decision. Interpret all 32 digest bytes as one unsigned big-endian integer and select when `integer % 100 < percentage`. While `confirmed_mandatory_audit_count < 50`, a promotion is allowed only when `confirmed + pending < 50`; it deterministically selects `mandatory_50` and increments pending, regardless of ordinal. Full pending slots force human review rather than an unsampled auto approval. Store the resulting fingerprints/counter snapshot/result once; restart, input permutation, source invalidation, or later key/config changes never re-hash an existing decision.

Before committing every auto approval, reserve the next promotion ordinal exactly once and insert its immutable promotion-decision row. The decision stores scope/policy/generation, ordinal, stable sample fingerprint, fingerprint key version/material, workflow percentage, and one selection result:

```text
mandatory confirmed target not met     mandatory_50 (slot-limited 100%)
10% canary after 50 confirmed          sample_10 (stable 10%)
100% workflow authorized after gate   sample_2  (stable 2%)
not selected by the applicable sample  not_selected
```

Insert a pending `AutoReviewPostAudit` referencing the decision only for `mandatory_50|sample_10|sample_2`. A later explicit audit of `not_selected` inserts one `manual` audit row referencing and copying the immutable decision identity; it never allocates another ordinal, recomputes sampling, or increments promotion/selected counters. Mandatory confirmation increments only on a human `confirmed` outcome. A source-invalidated or critical mandatory row frees its pending slot but never increments confirmed; after remediation, the next eligible promotion—even with ordinal greater than 50—fills the replacement slot as `mandatory_50`. Sampling begins only after exactly 50 effective confirmed mandatory audits, zero pending mandatory audits, zero original/corrected critical count, no unresolved correction/remediation, and every other selected gate condition. A corrected mandatory confirmation remains immutable history but is not effective confirmation and immediately opens the breaker; this policy row can never return to enforce. Do not allow a missing selected audit row after an approval commit. `sample_2` is selected only when the workflow was created after the explicit full-enforce authorization and stored percentage 100; merely reaching ordinal 501 in a 10% workflow never selects the 2% cohort.

Integrate the rollout authority into `AutoReviewOrchestrator` at both sides of the provider boundary. Before input assembly, resolve owner permissions and effective mode. In transaction B one coordinator owns the global order: `shared key-generation/runtime -> sorted required provider-safety rows -> projection -> rollout -> sorted Source FOR SHARE -> immutable workflow/cost row -> validation call/children -> ReviewItem -> promotion decision/audit -> approval/evidence links -> target knowledge -> document locks`. Under the leading locks it re-reads every purpose-specific provider safety/cost policy, runtime/projection key version/material/readiness, the active-row anti-join, duplicate/hidden-collision result, breaker, current global demotion, owner permission context, current source state, immutable workflow snapshot, percentage/generation selection, and budget. It then passes an explicit already-held validation/resolution lock context to the store and `AutoReviewResolutionService`; both assert it and never reacquire an earlier lock. If any identity/readiness check fails, or a breaker opens or mode/permissions demote while Terra is in flight, persist only a bounded shadow/fallback validation and leave the ReviewItem pending. The required promotion decision and any selected post-audit row are reserved in the same transaction as the locked approval; there is no intermediate auto-approval path without rollout authority.

Human transitions that contribute shadow comparisons use one coordinator as well. An unlocked locator first discovers the V2.1 scope/policy/source/workflow ids. Reject/needs-more then acquires `shared generation/runtime -> rollout -> sorted Source FOR SHARE -> workflow -> ReviewItem`, rechecks identity/status/evidence, commits the human transition and exactly one unchanged-evidence comparison together. Human approve follows `shared generation/runtime -> projection -> rollout -> sorted Source FOR SHARE -> workflow -> ReviewItem -> provenance/knowledge`, preserving the Task 5 projection rules. V2.0 transitions and V2.1 transitions with no shadow record keep their existing behavior. No path locks a ReviewItem/workflow and then tries to acquire Source, rollout, or projection, so concurrent human, automatic, ingestion, and reconciliation transactions converge without deadlock or double-counting.

- [x] **Step 5: Implement breaker-first critical audit with durable recovery**

Transaction 1 validates the authorized auditor, normalizes the required bounded human reason, and delegates to the shared `AutoReviewAuditTransitionStore`, which persists that immutable access-controlled reason with the outcome/auditor identity, finalizes the critical outcome exactly once, increments counters once, opens the breaker, sets audit status `remediation_required` with system-only code `revoke_pending`, appends the matching control event, and commits. Neither the reason nor source/model content enters public projection, AuditLog, or normal logs. That committed audit state is also the durable serving quarantine: every trusted-serving API/RAG/pgvector predicate immediately excludes only the affected auto approval effect, even if the process crashes before revoke, while `ReviewEvidenceVisibilityService` keeps it actionable to an authorized reviewer. Independent human provenance remains eligible. The same store handles non-critical human confirmation; it is the only writer of selected-audit status/counters and uses actor HMAC domain `auto-review-audit-actor:v1`. Transaction 2 invokes exact idempotent revoke. On success, transaction 3 marks the audit completed and retains immutable critical attribution/reason; on any bounded failure it retains `remediation_required`, quarantine, and a bounded failure code. Restart recovery performs only an unlocked bounded id scan of `revoke_pending` rows; for each id it enters the normal generation/runtime/projection/rollout/Source/workflow/ReviewItem/audit order, rechecks the row, and retries exact revoke idempotently. It never holds an audit-row lock while acquiring an earlier global lock; concurrent workers converge on the already-locked canonical item/audit state. Never roll back the already-open breaker or quarantine, and never leave a critical committed outcome without a durable recovery marker.

`AutoReviewQualityRevokeService` is the only human quality-revoke coordinator. Under the same global order it first validates actor/item/current-evidence ownership and the quality enum without writing an assessment. It then inserts/replays the immutable `AutoReviewRevocationAssessment` immediately before, and in the same transaction as, deterministic dispatch: (a) no audit—create the unique `manual` audit from the existing promotion decision and commit it critical; (b) pending audit—finalize it critical; (c) completed confirmed audit—leave that row byte-immutable, append the unique `AutoReviewAuditCorrection`, increment `corrected_critical_count` once, open the rollout breaker, invalidate the authorization generation, set durable correction quarantine/recovery state, append the exact control event, and commit; (d) already critical/remediation—verify same-reason assessment ownership and reuse the existing critical recovery state. A different already-persisted reason is `revoke_reason_conflict`. Only after that breaker/quarantine transaction commits does it construct the unforgeable `QualityRevokeContext` and invoke exact revoke. Revoke success/failure is finalized/recovered exactly like critical audit. `business_withdrawal` never enters this service, never creates a correction, and never changes quality metrics. A correction/counter cannot be deleted, decremented, reclassified, or reset on the same scope/policy; administrative close/authorize refuses it, so recovery requires a separately reviewed new policy version with fresh shadow/canary/mandatory gates.

`AutoReviewRolloutAdminService` exposes exactly `status(scope, policy)`, `authorize_percentage(scope, policy, percentage, expected_state_version, reason, gate_ref)`, `close_rollout_breaker(scope, policy, expected_state_version, reason, gate_ref)`, `authorize_initial_provider_policy(purpose, provider, model, reasoning_effort, expected_absent=True, new_cost_policy_version, reason, gate_ref)`, `clear_provider_breaker(purpose, provider, model, reasoning_effort, expected_state_version, new_cost_policy_version, reason, gate_ref)`, and `recover_pending_remediation(limit<=100)`. Every mutation uses CAS plus the fixed local system-operator principal `system:local-auto-review-rollout-admin`, HMAC-attributed with current key identities and domain `auto-review-rollout-actor:v1`; it never accepts a subject id. Every successful control mutation writes its immutable event and aggregate backpointer in the same transaction; replay/CAS failure writes neither. `purpose` is required and exactly `extraction|validation`; provider authorization validates that purpose's registry-owned estimator/framing, 24,000-character/10,000-framed-input-token/2,048-total-output-token/one-candidate extraction caps or the frozen 6,000-input-token/3,072-total-output-token/four-candidate/two-batch/five-workflow-candidate validation caps, exact registry prices, exact model snapshot and reasoning effort, cost policy, and regression gate. A purpose/provider/model/reasoning row can never authorize or clear the other purpose or reasoning profile even when provider/model strings match. Initial authorization is insert-only and requires an absent row plus reviewed golden gate. Provider-breaker clear requires an open provider breaker, exact expected state version, resolved overrun evidence, and a **different purpose-specific** cost-policy version. Rollout-breaker close additionally requires `corrected_critical_count=0` and no unresolved audit/correction remediation; a same-policy correction is permanent evidence and cannot be waived by reason/gate-ref. Neither operation can masquerade as the other and configuration change alone never closes anything.

The executable aggregate-only commands are:

```powershell
python -m backend.app.admin.auto_review_rollout status --scope <scope> --policy auto-review-policy:v1
python -m backend.app.admin.auto_review_rollout authorize --scope <scope> --policy auto-review-policy:v1 --percentage 10 --expected-state-version <n> --reason <1..500> --gate-ref <bounded-ref>
python -m backend.app.admin.auto_review_rollout close-breaker --scope <scope> --policy auto-review-policy:v1 --expected-state-version <n> --reason <1..500> --gate-ref <bounded-ref>
python -m backend.app.admin.auto_review_rollout authorize-initial-provider-policy --purpose validation --provider openai --model gpt-5.6-terra --reasoning-effort medium --new-cost-policy auto-review-cost:v1 --reason <1..500> --gate-ref <bounded-ref>
python -m backend.app.admin.auto_review_rollout authorize-initial-provider-policy --purpose extraction --provider openai --model gpt-5.4-mini-2026-03-17 --reasoning-effort none --new-cost-policy auto-review-extraction-cost:v1 --reason <1..500> --gate-ref <bounded-ref>
python -m backend.app.admin.auto_review_rollout clear-provider-breaker --purpose <extraction|validation> --provider <exact-provider> --model <exact-model> --reasoning-effort <none|medium> --expected-state-version <n> --new-cost-policy <purpose-specific-vN> --reason <1..500> --gate-ref <bounded-ref>
python -m backend.app.admin.auto_review_rollout recover-remediation --limit 100
```

The CLI has no subject/secret/raw-output option, prints only aggregate counts, modes, generations, state versions and breaker/readiness booleans, and uses exit `0` success, `2` bounded authorization/configuration/CAS refusal, `3` unresolved remediation/readiness failure. Public Review routes expose none of these mutations. Final lifespan ordering is exact: key bootstrap/projection status, then one bounded source-reconciliation recovery batch, then (after Task 12 installs it) one bounded extraction/validation call-ledger recovery batch, then one bounded `recover_pending_remediation(limit=100)` pass. Repeated source, call-ledger, and rollout admin CLIs are the operator continuation paths.

- [x] **Step 6: Run rollout/audit tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_rollout.py backend/tests/test_auto_review_audit.py backend/tests/test_auto_review_quality_revoke.py backend/tests/test_auto_review_rollout_admin.py backend/tests/test_auto_review_postgres.py backend/tests/test_agent_runtime_lifespan.py -q
uv run --locked ruff check backend/app/review/auto_review_rollout.py backend/app/review/auto_review_audit.py backend/app/review/auto_review_quality_revoke.py backend/app/review/auto_review_audit_transitions.py backend/app/admin/auto_review_rollout.py backend/app/agent_runtime/auto_review_orchestrator.py backend/app/review/transitions.py backend/app/review/auto_review_resolution.py backend/app/main.py backend/tests/test_auto_review_rollout.py backend/tests/test_auto_review_audit.py backend/tests/test_auto_review_quality_revoke.py backend/tests/test_auto_review_rollout_admin.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py backend/tests/test_agent_runtime_lifespan.py
```

- [x] **Step 7: Commit the rollout safety slice**

```powershell
git add backend/app/review/auto_review_rollout.py backend/app/review/auto_review_audit.py backend/app/review/auto_review_quality_revoke.py backend/app/review/auto_review_audit_transitions.py backend/app/admin/auto_review_rollout.py backend/app/agent_runtime/auto_review_orchestrator.py backend/app/review/transitions.py backend/app/review/auto_review_resolution.py backend/app/main.py backend/tests/test_auto_review_rollout.py backend/tests/test_auto_review_audit.py backend/tests/test_auto_review_quality_revoke.py backend/tests/test_auto_review_rollout_admin.py backend/tests/test_auto_review_orchestrator.py backend/tests/test_auto_review_postgres.py backend/tests/test_agent_runtime_lifespan.py
git commit -m "feat: gate auto review rollout with human audit"
```

---

### Task 11: Bind a Zero-Call Cost Preview to One V2.1 Launch

**Files:**
- Create: `backend/app/agent_runtime/launch_confirmation.py`
- Create: `backend/app/agent_runtime/review_workflow_facade.py`
- Modify: `backend/app/agent_runtime/review_v2_drafting.py`
- Modify: `backend/app/agent_runtime/review_v2_preflight.py`
- Create: `backend/tests/test_auto_review_launch_confirmation.py`
- Create: `backend/tests/test_review_workflow_facade.py`
- Modify: `backend/tests/test_review_v2_drafting.py`

**Interfaces:**
- Issues a short-lived HMAC token from a dry-run that calls no provider and performs no write.
- Binds keyed HMAC fingerprints of security scope, actor subject, and the actor's server-resolved allowed-permission set—not their raw ids/claims—plus canonical input/evidence hashes, graph version, stored mode, validator provider/model/reasoning and prompt/output-contract/policy/cost-policy versions, validation provider-safety `state_version`, aggregate extraction-plan-set HMAC, sorted extraction provider-safety snapshot-set HMAC, rollout `control_epoch` (never its noisy counter `state_version`), requested/authorized-at-launch percentages, rollout authorization generation, fingerprint key version/material verifier, validation and extraction estimator/encoding/framing/cap identities, maximum batches/candidates/provider attempts, all four shared provider timeout/send-start-window/attempt-lease/commit-grace values, exact Decimal price snapshots, exact V2.1 total-budget limit, extraction/validation/total ceilings, and expiry.
- Estimates validation from the enforceable selected-agent bound: each of 1–5 selected extraction agents can return zero or one candidate, so dry-run reserves `ceil(selected_agent_count / 4)` batches, each with 6,000 framed input tokens and 3,072 total output tokens. One batch is exactly USD 0.048864 and the five-agent maximum of two batches is USD 0.097728. Extraction reserves `selected_agent_count * USD 0.016716`, up to USD 0.083580; the five-agent full-profile reserve is USD 0.181308, leaving USD 0.018692 under the exact USD 0.20 budget. This is not a predicted-output discount: it uses the schema-enforced one-candidate-per-agent maximum. Missing tokenizer/framing registry, strict extraction contract, exact price-registry match, or provider readiness makes shadow/enforce unavailable; it never substitutes extraction pricing or a character heuristic.
- Verifies the token before thread creation or provider call. Changed/expired/malformed token becomes bounded `cost_preview_changed` and requires a fresh preview.
- Uses existing exact-batch ownership and `client_request_id`; the token is not a new idempotency namespace and cannot create a second thread/call.
- Defines a narrow injected V2.1 lifecycle protocol in the facade so Task 11 tests use a fake service; it does not import the concrete `review_v21_service.py` that is created in Task 12.

- [x] **Step 1: Write token, preview, and replay tests**

Cover:

- `test_disabled_preview_is_v20_and_has_no_token_or_validator_cost`
- `test_shadow_enforce_preview_is_v21_zero_call_and_returns_signed_token`
- `test_token_binds_actor_scope_input_evidence_versions_mode_and_costs`
- `test_expired_changed_or_cross_actor_token_is_cost_preview_changed`
- `test_status_polling_page_load_and_dry_run_never_call_validator`
- `test_same_token_concurrent_start_converges_on_existing_batch_owner`
- `test_token_replay_cannot_create_a_second_thread_or_provider_call`
- `test_missing_terra_price_or_key_is_bounded_unavailable_not_fallback`
- `test_token_contains_scope_and_actor_hmacs_not_raw_ids_and_binds_key_version`
- `test_token_binds_key_material_estimator_token_caps_and_max_batches`
- `test_launch_codec_exact_key_set_rejects_missing_or_unknown_key`
- `test_validator_output_contract_or_any_explicit_extraction_common_field_drift_requires_fresh_preview`
- `test_token_binds_all_shared_provider_timing_values`
- `test_token_binds_exact_input_and_output_price_snapshots`
- `test_token_binds_extraction_plan_and_sorted_safety_sets_and_every_effective_cap`
- `test_extraction_route_prompt_output_contract_canonical_guard_price_or_safety_drift_requires_fresh_preview_before_thread`
- `test_token_binds_authorized_percentage_and_rollout_generation`
- `test_fresh_scope_preview_signs_read_only_rollout_sentinel_and_start_rechecks_ensured_row`
- `test_missing_or_open_provider_safety_state_is_zero_call_unavailable`
- `test_maximal_valid_launch_token_fits_public_2048_character_bound`
- `test_preview_uses_conservative_versioned_token_upper_bound_before_candidates_exist`
- `test_five_selected_agents_reserve_two_validation_batches_and_0097728_without_prediction_discount`
- `test_preview_freezes_openai_mini_snapshot_none_route_and_one_candidate_cap`
- `test_preview_freezes_extraction_10000_2048_and_exact_075_450_prices`
- `test_preview_freezes_validation_6000_3072_four_two_five_and_exact_2_12_prices`
- `test_exact_five_agent_reserves_are_0083580_0097728_and_0181308_total`
- `test_alias_azure_gemini_or_fallback_extraction_route_is_zero_call_before_thread`
- `test_permission_change_after_preview_is_cost_preview_changed_before_thread_or_provider`
- `test_signed_launch_cannot_exceed_confirmed_provider_attempt_or_cost_ceiling`
- `test_total_over_budget_is_zero_call_before_token_or_thread`
- `test_start_locks_all_required_purpose_safety_rows_sorted_before_rollout_and_thread`


Use an injected clock and fixed secret; never sleep.

- [x] **Step 2: Run launch/facade tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_launch_confirmation.py backend/tests/test_review_workflow_facade.py backend/tests/test_review_v2_drafting.py -q
```

- [x] **Step 3: Implement canonical token issue/verify**

Use the existing non-default agent-runtime fingerprint key with explicit `auto-review-launch:v1`, a base64url canonical compact JSON payload plus HMAC-SHA256, and constant-time comparison. One frozen codec owns this exact key set and rejects any missing or unknown key:

| Compact keys | Exact fields |
|---|---|
| `v,g,m` | launch schema, graph version, configured mode |
| `sh,ah,ph,ih,eh` | keyed scope, actor, owner-permission, input, evidence identities |
| `vp,vm,vr,vs,pp,pv,cv,ps` | validator provider, model, reasoning, output-contract, prompt, approval-policy, cost-policy, provider-safety state version |
| `xv,xp,xm,xr,xt,xn` | extraction cost-policy, provider, model snapshot, reasoning, route, selected-agent count |
| `xc,xi,xo,xk` | extraction 24,000 character, 10,000 framed-input-token, 2,048 total-output-token, one-candidate caps |
| `xe,xen,xrp,xfs,xip,xop` | extraction estimator, encoding, reply-priming, framing-safety, exact input/output prices |
| `ep,es` | aggregate selected extraction-plan-set HMAC and complete sorted extraction provider-safety snapshot-set HMAC |
| `rc,rq,ap,rg` | rollout control epoch, requested percentage, authorized-at-launch percentage, authorization generation |
| `kv,km` | fingerprint key version and key-material verifier |
| `te,en,rp,fs,it,ot` | validation estimator, encoding, reply-priming, framing-safety, input/output caps |
| `mb,mc,mw,ma` | maximum validation batches, candidates per batch, candidates per workflow, attempts per call |
| `pt,sw,ls,cg` | provider timeout, send-start window, attempt lease, commit grace |
| `ip,op,bl,ec,vc,tc` | exact validator input/output prices, total budget, extraction/validation/total ceilings |
| `iat,exp` | issuance and expiry |

`ep` additionally commits the canonical sorted selected-agent names and each agent-specific prompt/output schema, prepared-input HMAC/count, lower effective caps, one-candidate contract, and shared timing tuple; `es` commits every selected `(purpose, provider, model, reasoning_effort, state_version)` safety identity. The common extraction tuple is deliberately explicit in `x*` fields rather than hidden only in a digest. A maximal valid token with both digests and every field above must remain <=2,048 characters; exact-key-set, unknown/missing-key, and maximal-size tests are release gates.

Validation checks key version/material verifier, both estimator identities, `o200k_base`, literal 16/512 allowances, validation 6,000-input-token/3,072-total-output-token caps, 4 candidates per batch, 2 batches/5 candidates per workflow, selected-agent count 1–5, all four timing values and their strict inequality, one provider attempt per call, exact Decimal USD 2/USD 12 validator prices, USD 0.048864 per-batch and USD 0.097728 maximum validation reserves, exact extraction snapshot/reasoning/route/prices/candidate cap, USD 0.016716 per-agent and USD 0.083580 maximum extraction reserves, USD 0.181308 maximum full-profile reserve, USD 0.20 budget, and expiry before recomputing current canonical identities and current server-side owner permission fingerprint. Do not place raw actor/scope ids, permission claims, source refs/text, non-public cost details, checkpoint ids, or credentials in the token. Sign and carry canonical HMACs, not raw inputs. Reject launch before thread creation when permissions changed, either extraction digest changed, or any identity/timing/cap/price/ceiling value drifted.

Return `cost_preview_changed` for every token mismatch class so public callers cannot probe which hidden field changed.

- [x] **Step 4: Add the version-selection facade and dry-run aggregation**

The facade depends on explicit V2.0/V2.1 lifecycle protocols and selects:

```text
configured disabled -> existing V2.0 dry-run/start service
configured shadow/enforce -> V2.1 dry-run/start protocol
stored V2.0 thread -> existing V2.0 status/resume/cancel
stored V2.1 thread -> V2.1 status/resume/cancel
unknown stored version -> runtime_version_unavailable
```

For V2.1 dry-run, aggregate extraction and the conservative validation upper bound separately and total them. Let `N` be the signed selected-agent count, where `1 <= N <= 5`. Extraction reserves exactly `N * USD 0.016716`; validation reserves exactly `ceil(N / 4) * USD 0.048864`, because the strict output contract allows each selected agent to produce at most one candidate. At `N=5`, those ceilings are USD 0.083580 and USD 0.097728, totaling USD 0.181308. This does not discount from a predicted candidate count: zero candidates are still charged against the signed maximum, and only the schema-enforced numeric cap permits the batch bound. The V2.1 legacy-named `estimated_*` fields remain the extraction reserve, `auto_review_estimated_*` is the validation reserve, `total_*` is their exact Decimal-priced sum, and `budget_status/budget_limit_usd` evaluate that total against the exact USD 0.20 limit. A route set whose recomputed reserve exceeds its signed ceiling or the overall limit is unavailable until a separately reviewed profile is approved. An over-budget preview returns no launch token and start creates no workflow/thread or provider call. Preview reads every required extraction/validation safety row and `rollout.peek_or_default()` without locks that mutate state; no preview inserts a rollout row. Issue the token only after canonical source/version/permission, strict result-schema, tokenizer/framing, every purpose safety row, and total-cost readiness succeed. Verification occurs before `create_or_reuse_review_thread()`; start recomputes `ep` and `es`, then locks every required purpose-safety row in sorted order before ensuring/locking rollout and rechecks the signed provider/rollout/budget/timing snapshots. Any concurrent selected-agent set, route, prompt, output contract, price, cap, timing, safety, rollout, permission, or source change returns `cost_preview_changed` before a thread or provider call.

- [x] **Step 5: Run launch/facade tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_launch_confirmation.py backend/tests/test_review_workflow_facade.py backend/tests/test_review_v2_drafting.py -q
uv run --locked ruff check backend/app/agent_runtime/launch_confirmation.py backend/app/agent_runtime/review_workflow_facade.py backend/app/agent_runtime/review_v2_drafting.py backend/app/agent_runtime/review_v2_preflight.py backend/tests/test_auto_review_launch_confirmation.py backend/tests/test_review_workflow_facade.py backend/tests/test_review_v2_drafting.py
```

- [x] **Step 6: Commit signed one-click launch preflight**

```powershell
git add backend/app/agent_runtime/launch_confirmation.py backend/app/agent_runtime/review_workflow_facade.py backend/app/agent_runtime/review_v2_drafting.py backend/app/agent_runtime/review_v2_preflight.py backend/tests/test_auto_review_launch_confirmation.py backend/tests/test_review_workflow_facade.py backend/tests/test_review_v2_drafting.py
git commit -m "feat: bind auto review costs to signed launch"
```

---

### Task 12: Compile Immutable V2.1 LangGraph and Dual-Version Lifecycle

**Files:**
- Create: `backend/app/agent_runtime/review_v21_state.py`
- Create: `backend/app/agent_runtime/review_v21_graph.py`
- Create: `backend/app/agent_runtime/review_v21_service.py`
- Create: `backend/app/admin/auto_review_call_recovery.py`
- Modify: `backend/app/agent_runtime/graph_versions.py`
- Modify: `backend/app/agent_runtime/review_workflow_facade.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/api/v1/orchestration_v2.py`
- Create: `backend/tests/test_review_v21_state.py`
- Create: `backend/tests/test_review_v21_graph.py`
- Create: `backend/tests/test_review_v21_service.py`
- Create: `backend/tests/test_review_v21_api.py`
- Create: `backend/tests/test_auto_review_call_recovery.py`
- Modify: `backend/tests/test_agent_runtime_graph_versions.py`
- Modify: `backend/tests/test_agent_runtime_lifespan.py`
- Modify: `backend/tests/test_review_v2_service.py`
- Modify: `backend/tests/test_review_v2_api.py`
- Modify: `backend/tests/test_review_v2_postgres.py`

**Interfaces:**
- Registers V2.0 and `company-memory-review-v2.1-auto-review` under separate immutable keys and builders.
- Gives V2.1 its own exact five-status checkpoint validator and lifecycle projection. The existing V2.0 state, graph builder, service mapper, and exact response keys remain unchanged.
- Compiles actual `StateGraph` nodes `run_auto_review` and `refresh_review_resolution`; pending-first routing controls whether real `interrupt()` occurs.
- Uses a dedicated V2.1 service behind the facade. Status/resume/cancel dispatch from the stored graph version, never current config.
- Returns a `graph_version` discriminated dry-run/status union. V2.0 mapper never emits V2.1 fields.

- [x] **Step 1: Write graph topology, routing, checkpoint privacy, and dual-version tests**

Cover:

- `test_v21_checkpoint_has_exact_eight_safe_keys_and_five_counts`
- `test_v21_graph_uses_real_stategraph_auto_review_refresh_and_interrupt`
- `test_all_auto_resolved_completes_without_interrupt`
- `test_zero_candidates_uses_distinct_finalize_no_candidates_node_and_status`
- `test_pending_takes_priority_over_needs_more_evidence`
- `test_no_pending_with_needs_more_evidence_terminates_needs_more`
- `test_revoked_is_resolved_but_not_approved`
- `test_disabled_effective_mode_is_zero_call_pass_through`
- `test_shadow_never_changes_review_item_status`
- `test_enforce_only_transitions_stable_selected_candidates`
- `test_paused_v20_tuple_resumes_with_unchanged_v20_builder_and_schema`
- `test_registry_resolves_both_exact_versions_and_never_falls_back`
- `test_v20_response_exact_keys_and_v21_discriminated_keys`
- `test_checkpoint_and_status_contain_no_item_validation_source_or_model_output`
- `test_stored_v21_resumes_with_global_disabled_as_zero_call_human_interrupt`
- `test_stored_v21_resumes_with_openai_key_removed_as_bounded_human_fallback`
- `test_resume_re_resolves_owner_permissions_and_cannot_use_auto_actor_as_surrogate`
- `test_completed_v21_status_reconciles_verified_approved_to_revoked_without_checkpoint_rewrite`
- `test_interrupted_v21_resume_accepts_only_verified_monotonic_revoke_delta`
- `test_interrupted_v21_resume_accepts_authorized_human_pending_to_terminal_resolution`
- `test_other_live_checkpoint_count_drift_fails_closed`
- `test_v21_cancel_dispatch_terminalizes_attempt_zero_and_charges_attempt_one_without_late_effects`
- `test_call_recovery_cli_finalizes_old_generation_attempts_without_provider_retry_or_raw_output`


- [x] **Step 2: Run graph/service/API tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_review_v21_state.py backend/tests/test_review_v21_graph.py backend/tests/test_review_v21_service.py backend/tests/test_review_v21_api.py backend/tests/test_auto_review_call_recovery.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review_v2_postgres.py -q
```

- [x] **Step 3: Implement the separate V2.1 state and actual graph**

Copy no V2.0 state by import-and-mutate. Define the separate exact status set and validator, reusing only safe bounded reducers. Compile:

```text
START
 -> validate_input
 -> collect_evidence_refs
 -> plan_agent_runs
 -> draft_review_candidates_transaction
 -> run_auto_review
 -> refresh_review_resolution
 -> route_review_boundary
    -> await_human_review interrupt -> verify resolution -> terminal route
    -> finalize_needs_more_evidence
    -> finalize_auto_resolved
    -> finalize_no_candidates
 -> END
```

The V2.1 `draft_review_candidates_transaction` invokes only the authoritative Task 3 `review_v21_extraction` coordinator and its one-call ledger; it never calls the legacy V2.0 agent adapters directly or permits their retry/fallback/cache policy. The V2.0 builder alone retains those legacy adapters. `run_auto_review` receives only `workflow_thread_id` from state and calls the injected validation coordinator; it never puts result ids/output in state. A LangGraph runtime-context dependency supplies a server-owned current-permission resolver, not a checkpoint field or client-provided permission set. On every node execution/resume the coordinator reloads the stored workflow owner and computes effective visibility as `current_owner_allowed_permission_levels ∩ ('public', 'internal')`; the auto actor is never a broader surrogate. `refresh` opens PostgreSQL and reconstructs exact five-status counts. Routing checks `total == 0` first and records only the bounded `finalize_no_candidates` completion marker; it cannot fall through the vacuously true all-resolved equality.

- [x] **Step 4: Add a dedicated V2.1 lifecycle service and facade dispatch**

Reuse checkpoint execution primitives, not V2.0 private four-status validators. V2.1 service owns its own projection/snapshot validation and exposes counts only. For a **completed terminal checkpoint**, live reconciliation accepts only one post-checkpoint monotonic exception without mutating checkpoint bytes: `approved -= N` and `revoked += N` with total/resolved counts unchanged, after proving each changed workflow-owned item was auto-policy approved with a completed same-item validation and a later `revoked_at`. For an **interrupted human-review checkpoint**, preserve the existing resume contract: workflow-owned `pending_review -> approved|rejected|needs_more_evidence` is accepted only when the current row and transition/AuditLog prove an authorized human/internal-stale transition later than the checkpoint; verified auto `approved -> revoked` is an additional exception. Total changes, unverified terminal changes, terminal-to-pending/reverse transitions, or cross-workflow rows fail closed. Refresh reconstructs live counts and routes without rewriting old checkpoint bytes. On resume, global mode can demote work still ahead of the node, but stored graph/config never upgrades. A V2.1 thread never falls back to V2.0.

Register and compile both graph builders, the coordinator, and the V2.1 lifecycle service unconditionally during lifespan, even when disabled or the Terra key is absent. Inject a zero-call/unavailable validator disposition in that case: an already-stored V2.1 thread can resume, persist a bounded fallback if appropriate, refresh counts, and interrupt for human review instead of returning runtime-version-unavailable/503. Apply readiness only to **new** shadow/enforce preview/start selection. Default disabled boot succeeds without an OpenAI key and keeps new runs on V2.0.

`AutoReviewCallRecoveryService` performs an unlocked bounded id scan, then for each row reacquires the old generation's full ordered workflow/call context. Attempt-zero stale/cancelled calls become zero-charge terminal and release their reservation; attempt-one lost calls become actual-if-known or conservative-reserve charged failure, with no provider retry, candidate, validation child, promotion decision, audit, or approval. It handles extraction and validation independently and is replay-safe. The local aggregate-only CLI is `python -m backend.app.admin.auto_review_call_recovery status --limit 100` or `recover --limit 100`; it accepts no call id, subject, secret, or raw-output option, returns only extraction/validation pending/recovered/remaining/failure counts, and uses exit `0` for no remaining work, `2` for bounded configuration/key refusal, and `3` for retained non-terminal/failure state. Lifespan runs one bounded recovery batch after source reconciliation and before audit-remediation recovery. Rotation remains refused until status reports both remaining counts zero.

- [x] **Step 5: Add version-specific API validation and mapping**

Parse a V2.0 request when no V2.1 token is allowed and a V2.1 request only when the facade selected V2.1 preview identity. Map `cost_preview_changed` to 409, new-preview/start readiness infrastructure failures to bounded 503, hidden permission to 404, and keep existing conflict mappings. A stored V2.1 resume does not return 503 merely because global mode/provider key changed; it follows the zero-call/human-fallback path above. Do not emit raw Pydantic/provider/checkpoint exceptions.

- [x] **Step 6: Run graph/service/API tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_review_v21_state.py backend/tests/test_review_v21_graph.py backend/tests/test_review_v21_service.py backend/tests/test_review_v21_api.py backend/tests/test_auto_review_call_recovery.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review_v2_postgres.py -q
uv run --locked ruff check backend/app/agent_runtime/review_v21_state.py backend/app/agent_runtime/review_v21_graph.py backend/app/agent_runtime/review_v21_service.py backend/app/admin/auto_review_call_recovery.py backend/app/agent_runtime/graph_versions.py backend/app/agent_runtime/review_workflow_facade.py backend/app/main.py backend/app/api/v1/orchestration_v2.py backend/tests/test_review_v21_state.py backend/tests/test_review_v21_graph.py backend/tests/test_review_v21_service.py backend/tests/test_review_v21_api.py backend/tests/test_auto_review_call_recovery.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review_v2_postgres.py
```

- [x] **Step 7: Commit dual-version LangGraph lifecycle**

```powershell
git add backend/app/agent_runtime/review_v21_state.py backend/app/agent_runtime/review_v21_graph.py backend/app/agent_runtime/review_v21_service.py backend/app/admin/auto_review_call_recovery.py backend/app/agent_runtime/graph_versions.py backend/app/agent_runtime/review_workflow_facade.py backend/app/main.py backend/app/api/v1/orchestration_v2.py backend/tests/test_review_v21_state.py backend/tests/test_review_v21_graph.py backend/tests/test_review_v21_service.py backend/tests/test_review_v21_api.py backend/tests/test_auto_review_call_recovery.py backend/tests/test_agent_runtime_graph_versions.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_review_v2_service.py backend/tests/test_review_v2_api.py backend/tests/test_review_v2_postgres.py
git commit -m "feat: add immutable auto review langgraph v21"
```

---

### Task 13: Expose Bounded Auto-Review Metadata, Revoke, and Audit Actions

**Files:**
- Modify: `backend/app/schemas/review.py`
- Modify: `backend/app/api/v1/review.py`
- Modify: `backend/app/services/audit.py`
- Modify: `backend/app/review/auto_review_revoke.py`
- Modify: `backend/app/review/auto_review_audit.py`
- Modify: `backend/app/review/auto_review_quality_revoke.py`
- Create: `backend/tests/test_auto_review_api.py`
- Modify: `backend/tests/test_review.py`
- Modify: `backend/tests/test_review_rbac.py`
- Modify: `backend/tests/test_review_v2_api.py`

**Interfaces:**
- Adds `status=approved&resolution_source=auto_policy` filtering after existing permission/concealment rules and preserves `pending_review` as the default queue view.
- Allows workflow-filtered Review lists for exact registered V2.0 or V2.1 graph versions; it does not broaden to arbitrary version strings.
- Adds only allowlisted resolution, validation summary, and bounded audit projection to visible ReviewItem responses.
- Exposes the exact revoke and audit routes frozen above. Revoke accepts only the strict server-validated reason enum; it never accepts or classifies free text. Foreign/inaccessible items remain concealed 404, invalid transition is bounded 409, and actor capability remains server-owned.
- Maps `business_withdrawal` through the normal audit gate: a selected audit not `completed+confirmed` returns exact HTTP 409 `audit_required`. Every quality reason routes to `AutoReviewQualityRevokeService`, including after a confirmed audit; the audit/correction remains visible/actionable, critical/remediation recovery performs its own revoke, and the gate cannot be orphaned.
- Critical audit invokes breaker-first service semantics before exact revoke; frontend/API cannot close a breaker.
- Returns the revoke success model only after exact revoke. A committed quality quarantine with pending/failed physical revoke maps to exact HTTP 409 `remediation_required`; callers refetch the permission-filtered Review item and no internal failure text is exposed. A same-item replay with a different immutable terminal reason maps to bounded HTTP 409 `revoke_reason_conflict`; it never replaces the first assessment or reveals its reason.

- [x] **Step 1: Write response/privacy/filter/action tests**

Cover:

- `test_default_review_list_still_returns_pending_only`
- `test_auto_filter_returns_only_visible_approved_auto_policy_items`
- `test_v21_workflow_filter_is_supported_without_accepting_unknown_versions`
- `test_auto_summary_and_audit_projection_are_bounded_and_allowlisted`
- `test_internal_collision_lookup_and_failure_codes_can_never_enter_public_summary`
- `test_review_response_never_exposes_validation_link_provenance_or_document_ids`
- `test_revoke_requires_reviewer_permission_and_exact_reason_code_enum`
- `test_revoke_rejects_free_text_unknown_or_extra_reason_fields`
- `test_revoke_response_and_replay_have_exact_public_keys`
- `test_selected_audit_revoke_requires_completed_confirmed_or_returns_bounded_conflict`
- `test_pending_to_critical_direct_and_recovery_interleaving_has_one_server_revoke`
- `test_completed_confirmed_audit_business_withdrawal_direct_revoke_is_allowed`
- `test_completed_confirmed_audit_quality_revoke_creates_correction_and_never_direct_bypasses_breaker`
- `test_gate_rejected_revoke_writes_no_terminal_assessment_and_later_quality_reason_can_proceed`
- `test_different_reason_after_terminal_assessment_returns_bounded_revoke_reason_conflict`
- `test_audit_requires_reviewer_or_admin_allowed_outcome_and_reason`
- `test_critical_audit_returns_breaker_open_even_when_revoke_needs_remediation`
- `test_quality_revoke_failure_returns_bounded_409_then_refetch_shows_action_required`
- `test_quality_revoke_replay_can_finish_same_recovery_and_return_exact_success_schema`
- `test_public_body_cannot_forge_auto_actor_policy_or_target`
- `test_foreign_revoke_and_audit_are_concealed_not_found`


- [x] **Step 2: Run Review API tests and observe RED**

```powershell
uv run --locked pytest backend/tests/test_auto_review_api.py backend/tests/test_review.py backend/tests/test_review_rbac.py backend/tests/test_review_v2_api.py -q
```

- [x] **Step 3: Add strict request/response models and safe projections**

Define the revoke request as only `reason_code: business_withdrawal|incorrect_content|permission_violation|wrong_source_version|policy_violation`; extra/free-text fields are forbidden. Keep the separate audit request's 1–500 character normalized human reason. `auto_review_summary` is built only from the completed canonical validation linked to the same item and includes model, reasoning, versions, supported-field count, minimum score, the separate public allowlist `direct_fact_supported|trusted_exact_reaffirmation`, and completion timestamp. Never pass internal `AutoReviewPolicyReasonCode` through directly: hidden-collision, lookup, permission, registry, budget, drift, and failure codes make the optional summary null/fail closed on corrupt legacy data. `auto_review_audit` includes only effective status/outcome and `action_required`; a confirmed row with an immutable correction projects critical/action-required without exposing correction ids/counters or rewriting the original audit.

Do not return validation id/key, lease, claim fingerprint, cohort/ordinal, raw audit reason, raw output, evidence aliases, hidden collision result, provenance identity/count, vector id, or another ReviewItem id.

- [x] **Step 4: Add filters and actions through existing services**

Apply status/resolution filters in the SQL query together with workflow ownership, then call `ReviewEvidenceVisibilityService` before totals/groups are returned. Revoke and audit routes resolve the item through the same concealment helper, convert the human actor after RBAC, and call services; they do not perform direct updates. Dispatch `business_withdrawal` to normal revoke and every quality enum to the breaker-first quality coordinator. Preserve `audit_required|quality_audit_required|remediation_required` as allowlisted conflicts without exposing cohort/ordinal, correction identity, or why the item was sampled. `remediation_required` is emitted only after durable quarantine exists and exact revoke did not finish; the handler does not attempt to compensate/undo it.

Keep existing approve/reject/evidence/bulk response behavior and paths unchanged.

- [x] **Step 5: Run Review API tests and lint GREEN**

```powershell
uv run --locked pytest backend/tests/test_auto_review_api.py backend/tests/test_review.py backend/tests/test_review_rbac.py backend/tests/test_review_v2_api.py -q
uv run --locked ruff check backend/app/schemas/review.py backend/app/api/v1/review.py backend/app/services/audit.py backend/app/review/auto_review_revoke.py backend/app/review/auto_review_audit.py backend/app/review/auto_review_quality_revoke.py backend/tests/test_auto_review_api.py backend/tests/test_review.py backend/tests/test_review_rbac.py backend/tests/test_review_v2_api.py
```

- [x] **Step 6: Commit the bounded API slice**

```powershell
git add backend/app/schemas/review.py backend/app/api/v1/review.py backend/app/services/audit.py backend/app/review/auto_review_revoke.py backend/app/review/auto_review_audit.py backend/app/review/auto_review_quality_revoke.py backend/tests/test_auto_review_api.py backend/tests/test_review.py backend/tests/test_review_rbac.py backend/tests/test_review_v2_api.py
git commit -m "feat: expose auto review audit and revoke actions"
```

---

### Task 14: Add the Typed V2.1 Client and One-Click Combined Cost Preview

**Files:**
- Modify: `frontend/src/lib/api/types.ts`
- Modify: `frontend/src/lib/api/reviewWorkflow.ts`
- Modify: `frontend/src/app/integrations/ReviewCandidateLaunchPanel.tsx`
- Modify: `frontend/src/app/integrations/page.tsx`
- Modify: `frontend/e2e/review-hitl-v2-api.spec.ts`
- Modify: `frontend/e2e/review-hitl-v2-integrations.spec.ts`

**Interfaces:**
- Defines distinct `ReviewWorkflowDryRunV20|V21` and `ReviewWorkflowStatusV20|V21` unions narrowed by exact `graph_version`.
- Defines distinct `ReviewStatusCountsV20` (the current four keys) and `ReviewStatusCountsV21` (the exact five keys including `revoked`); it never widens a shared record with optional status keys.
- Keeps the V2.0 launch body unchanged. Sends `launch_confirmation_token` only for a V2.1 preview and allowlists `cost_preview_changed` in the bounded client error union.
- Shows extraction estimate, auto-validation maximum, total maximum, mode, and version information in the existing preview panel.
- Uses the same `검토 후보 만들기` button as explicit paid-run confirmation. No route, modal, wizard, selector, or second normal-path click is added.
- On `cost_preview_changed`, discards the stale token, obtains a fresh zero-call preview, and requires the user to press the same launch button again; it never automatically launches a changed paid run.

- [x] **Step 1: Write transport and Integrations behavior tests**

Extend the client test to assert exact bodies:

```typescript
expect(v20Body).toEqual({ source_refs, agent_names, client_request_id });
expect(v21Body).toEqual({
  source_refs,
  agent_names,
  client_request_id,
  launch_confirmation_token: "signed-preview",
});
```

Cover V2.0 exact compatibility, V2.1 combined fields, token forwarding, token omission for V2.0, malformed/untrusted errors hidden, and `cost_preview_changed` allowlisting. In the page tests cover desktop/mobile combined costs, stale sync protection, cached/no-input/over-budget states, refreshed changed preview, and no automatic relaunch.

- [x] **Step 2: Run client/page tests and observe RED**

Start the existing frontend dev server on `127.0.0.1:3000`, then run:

```powershell
Set-Location frontend
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-api.spec.ts review-hitl-v2-integrations.spec.ts --project=chromium-desktop
npm.cmd run test:visual -- review-hitl-v2-integrations.spec.ts --project=chromium-mobile
Set-Location ..
```

Expected: new V2.1 transport/render assertions fail while current V2.0 cases stay green. Stop the server after the run.

- [x] **Step 3: Add exact TypeScript discriminated unions**

Define shared bases only for genuinely common fields; do not make every V2.1 field optional on one interface:

```typescript
export type ReviewWorkflowDryRun =
  | ReviewWorkflowDryRunV20
  | ReviewWorkflowDryRunV21;

export type ReviewWorkflowStatus =
  | ReviewWorkflowStatusV20
  | ReviewWorkflowStatusV21;

export type ReviewStatusCountsV20 = Record<
  'pending_review' | 'approved' | 'rejected' | 'needs_more_evidence',
  number
>;

export type ReviewStatusCountsV21 = Record<
  'pending_review' | 'approved' | 'rejected' | 'needs_more_evidence' | 'revoked',
  number
>;

export type ReviewWorkflowRunRequestV20 = ReviewWorkflowRunBase & {
  launch_confirmation_token?: never;
};

export type ReviewWorkflowRunRequestV21 = ReviewWorkflowRunBase & {
  launch_confirmation_token: string;
};
```

Narrow on `graph_version === 'company-memory-review-v2.1-auto-review'` before reading V2.1 counts/costs or constructing its run body.

- [x] **Step 4: Render combined preview and preserve one explicit launch**

Replace the single cost metric only for V2.1 with labeled extraction, automatic-validation maximum, and total maximum token/cost values. Show selected `shadow|enforce` and bounded policy/model identity without operational internals. Retain V2.0's existing panel exactly.

When the backend returns `cost_preview_changed`, set launch state back to preview loading, erase the token immediately, fetch a new preview, and display Korean copy explaining that the estimate changed. Do not reuse the old `client_request_id` to launch until the new explicit click.

- [x] **Step 5: Run frontend focused tests, lint, and build GREEN**

With no dev server active for lint/build:

```powershell
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

Then start the server and run:

```powershell
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- review-hitl-v2-api.spec.ts review-hitl-v2-integrations.spec.ts --project=chromium-desktop
npm.cmd run test:visual -- review-hitl-v2-integrations.spec.ts --project=chromium-mobile
Set-Location ..
```

Expected: lint/build and all selected tests pass; stop the server and verify port 3000 is closed.

- [x] **Step 6: Commit the typed one-click preview**

```powershell
git add frontend/src/lib/api/types.ts frontend/src/lib/api/reviewWorkflow.ts frontend/src/app/integrations/ReviewCandidateLaunchPanel.tsx frontend/src/app/integrations/page.tsx frontend/e2e/review-hitl-v2-api.spec.ts frontend/e2e/review-hitl-v2-integrations.spec.ts
git commit -m "feat: preview auto review cost in integrations"
```

---

### Task 15: Add Same-Screen Trust Review and Trust-Source Badges

**Files:**
- Create: `frontend/src/lib/api/autoReview.ts`
- Create: `frontend/src/components/review/AutoReviewBadge.tsx`
- Create: `frontend/src/components/review/AutoReviewActions.tsx`
- Create: `frontend/src/app/review/AutoReviewTrustPanel.tsx`
- Modify: `frontend/src/app/review/ReviewWorkflowContextPanel.tsx`
- Modify: `frontend/src/app/review/page.tsx`
- Modify: `frontend/src/lib/api/types.ts`
- Modify: `backend/app/api/v1/knowledge.py`
- Modify: `backend/app/projects/service.py`
- Modify: `frontend/src/components/knowledge/MemoryCollection.tsx`
- Modify: `frontend/src/app/timeline/page.tsx`
- Create: `frontend/e2e/auto-review-trust-promotion.spec.ts`
- Create: `frontend/e2e/auto-review-knowledge-badges.spec.ts`
- Modify: `frontend/e2e/review-hitl-v2-review.spec.ts`
- Modify: `frontend/e2e/timeline-project-date-groups.spec.ts`
- Modify: `backend/tests/test_knowledge_api.py`
- Modify: `backend/tests/test_project_memory_api.py`

**Interfaces:**
- Keeps Review defaulted to pending. Adds an inline two-view control: `검토 대기` and `자동 승인`; the latter requests only visible `approved + auto_policy` rows and preserves an exact workflow filter.
- Keeps the current group-expanded detail and evidence drawer. It adds badges, bounded validator/policy/time/reason metadata, audit outcome action, and revoke reason/action inline; no new route, modal, wizard, or extra navigation click.
- Hides bulk approve/reject, project selection, and human approve/reject/evidence actions in the auto-approved view. It never performs an optimistic trusted-state mutation; it reloads list and workflow status after the server commits.
- Displays V2.1 counts exactly as `자동 승인 N건 | 확인 필요 M건 | 추가 근거 필요 K건`; V2.0 retains `완료 수 / 전체 수`.
- Adds `사람 승인` or `자동 검증` as a separate trust-source badge in Knowledge/History/Decision collections and Timeline while preserving operational status and source evidence behavior. If shared canonical knowledge has any active human or legacy-human provenance, `사람 승인` takes precedence; `자동 검증` is shown only when all active trusted provenance is auto-policy.
- Exposes only permission-filtered `resolution_source` from Knowledge/project Timeline backend projections; it does not expose validation/provenance internals.

- [x] **Step 1: Write backend trust-source projection tests**

Assert that approved Knowledge/History/Timeline records resolve `human` versus `auto_policy` from visible provenance, retain the strictest permission, conceal inaccessible rows, and never expose validation/provenance ids or counts.

Run and observe RED:

```powershell
uv run --locked pytest backend/tests/test_knowledge_api.py backend/tests/test_project_memory_api.py -q
```

- [x] **Step 2: Write Review and trust-badge Playwright tests**

Cover:

```text
pending is the default and makes no approved-auto query
inline automatic-approved filter preserves workflow_thread_id
V2.1 counts and all-auto completion show no human-resume action
automatic-validation, audit-required, and remediation-required badges
confirmed audit refresh
critical audit keeps remediation visible if revoke fails
quality revoke 409 refetches and keeps corrected remediation visible
authorized revoke and replay/shared-knowledge result copy
strict inline revoke reason-code choices with no free-text classifier
confirmed-audit quality revoke reloads as corrected critical/action-required
403/404 and raw backend/provider detail become generic Korean copy
bulk/human actions absent in automatic view
no raw ReviewItem/validation/provenance/document id in automatic detail
Knowledge, History, and Timeline show human versus automatic badge
source evidence still opens through the existing interaction
desktop and mobile add no page/modal/navigation depth
```

Run the new specs once and observe their expected failures against the missing UI.

- [x] **Step 3: Add safe action wrappers and shared badges**

`autoReview.ts` exposes only:

```typescript
type AutoReviewRevokeReasonCode =
  | 'business_withdrawal'
  | 'incorrect_content'
  | 'permission_violation'
  | 'wrong_source_version'
  | 'policy_violation';

revokeAutoApproval(reviewItemId: number, reasonCode: AutoReviewRevokeReasonCode)
submitAutoReviewAudit(
  reviewItemId: number,
  outcome: AutoReviewAuditOutcome,
  reason: string,
)
```

Reuse the generic client error sanitizer. `AutoReviewBadge` maps allowlisted state to Korean labels without printing raw codes. `AutoReviewActions` renders one inline Korean-labeled select for the five frozen revoke reasons (business withdrawal versus four quality reasons), sends only the enum code, keeps actions inline, disables duplicate submission, and waits for canonical server response. On bounded 409 `remediation_required`, it immediately refetches the item and renders `조치 필요`; it never optimistically restores or hides the quarantined effect. Audit submission retains its separate 1–500 character human reason. No modal, second confirmation step, or client-side quality classifier is added.

- [x] **Step 4: Add the two-view Review surface and V2.1 counts**

The current page hard-codes `status=pending_review`; replace that with view-derived query parameters while keeping pending as initial state. In auto view use `status=approved&resolution_source=auto_policy`, retain the exact server-issued `workflow_thread_id`, and render `AutoReviewTrustPanel` in the existing expanded item area. Keep `SourceEvidenceDrawer` unchanged.

For automatic detail, use `상세 내용` rather than the current raw `#{item.id}` label. Do not place ids in visible error/copy. Render `감사 필요` when audit status is pending and `조치 필요` whenever remediation is required; a failed critical remediation cannot be hidden by normal confirmed UI state.

- [x] **Step 5: Add permission-aware trust-source projections and badges**

Join approved knowledge/project Timeline records to their originating ReviewItem/provenance after applying permission filters, return nullable `resolution_source`, and add it to typed item models. Resolve shared provenance deterministically as human when any active explicit/legacy-human source exists, otherwise auto-policy when at least one active auto source exists; never disclose how many links produced the label. `MemoryCollection.tsx` covers Knowledge, History, and Decisions; `timeline/page.tsx` adds the trust badge separately from `승인/완료` operational status and retains its current inline evidence panel.

- [x] **Step 6: Run backend and frontend focused gates GREEN**

```powershell
uv run --locked pytest backend/tests/test_knowledge_api.py backend/tests/test_project_memory_api.py -q
uv run --locked ruff check backend/app/api/v1/knowledge.py backend/app/projects/service.py backend/tests/test_knowledge_api.py backend/tests/test_project_memory_api.py
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

Then start the dev server and run:

```powershell
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
npm.cmd run test:visual -- auto-review-trust-promotion.spec.ts auto-review-knowledge-badges.spec.ts review-hitl-v2-review.spec.ts timeline-project-date-groups.spec.ts --project=chromium-desktop
npm.cmd run test:visual -- auto-review-trust-promotion.spec.ts review-hitl-v2-review.spec.ts --project=chromium-mobile
Set-Location ..
```

Expected: every focused backend/frontend test passes. Stop the server and verify port 3000 is closed.

- [x] **Step 7: Commit the same-screen trust UX**

```powershell
git add frontend/src/lib/api/autoReview.ts frontend/src/components/review/AutoReviewBadge.tsx frontend/src/components/review/AutoReviewActions.tsx frontend/src/app/review/AutoReviewTrustPanel.tsx frontend/src/app/review/ReviewWorkflowContextPanel.tsx frontend/src/app/review/page.tsx frontend/src/lib/api/types.ts backend/app/api/v1/knowledge.py backend/app/projects/service.py frontend/src/components/knowledge/MemoryCollection.tsx frontend/src/app/timeline/page.tsx frontend/e2e/auto-review-trust-promotion.spec.ts frontend/e2e/auto-review-knowledge-badges.spec.ts frontend/e2e/review-hitl-v2-review.spec.ts frontend/e2e/timeline-project-date-groups.spec.ts backend/tests/test_knowledge_api.py backend/tests/test_project_memory_api.py
git commit -m "feat: review and audit automatic trust inline"
```

---

### Task 16: Prove Golden Quality, PostgreSQL Safety, Regression, and Rollback

**Files:**
- Create: `backend/app/review/auto_review_evaluation.py`
- Create: `backend/app/review/auto_review_extraction_compatibility.py`
- Create: `backend/tests/fixtures/auto_review_golden_v1.json`
- Create: `backend/tests/fixtures/auto_review_extraction_compat_v1.json`
- Create: `backend/tests/test_auto_review_golden.py`
- Create: `backend/tests/test_auto_review_evaluation_cli.py`
- Create: `backend/tests/test_auto_review_extraction_compatibility_cli.py`
- Create: `backend/tests/test_auto_review_smoke.py`
- Create: `backend/tests/test_secret_hygiene.py`
- Modify: `backend/tests/test_auto_review_postgres.py`
- Modify: `backend/tests/test_review_v2_postgres.py`
- Modify: `backend/tests/test_pgvector_integration.py`
- Modify: `frontend/playwright.config.ts`
- Modify: `plan.md`
- Modify: `docs/portfolio-log.md`
- Modify: `docs/superpowers/runbooks/session-handoff.md`

**Release gate:** This task adds no new trust behavior. It proves code safety with deterministic/fake provider output, a disposable PostgreSQL+pgvector target, the complete non-Slack/frontend regression set, and a rollback smoke. Separately authorized sanitized paid Terra validation and Mini extraction compatibility gates are additionally required before shadow rollout, never during automated tests. They record aggregate evidence only and leave auto review disabled by default.

**Execution status (2026-08-30):** Steps 1–2 and 4–12 are complete at behavior
commit `4b9132a` plus the following documentation evidence commit. Steps 3 and
3B remain intentionally unchecked because no paid-call authorization was given.
This is a rollout gate, not unfinished product implementation; mode remains
`disabled` until both live aggregate gates pass.

**Task 16 entry prerequisite:** Product Tasks 6–15 must already be implemented and reviewed. Before Step 4, `docs/superpowers/plans/2026-08-29-whole-suite-postgresql-isolation.md` must first be separately user-approved and its infrastructure Tasks 1–11 implemented through their RED/GREEN/reviewed-commit checkpoints. That plan's Task 12 is this task's Steps 4–7 official profile proof. Its controller is the only authoritative backend release path; a focused `--child-id` result is development evidence only and never a release proof.

- [x] **Step 1: Create the Korean-first frozen golden dataset and metric harness**

The fixture must label direct supported Timeline/History, proposal versus decision, planned versus completed, affirmation versus negation, conditional/uncertain wording, date/subject/actor mismatch, conflicting sources, source supersession, permission loss/unknown, partial support, high-confidence hard negative, prompt injection, exact reaffirmation, trusted collision, Decision/Todo, and restricted cases.

The evaluator consumes frozen candidate/evidence plus fake structured validator results and reports:

```text
auto_approval_precision
hard_negative_false_approval_count
permission_or_version_violation_count
duplicate_promotion_count
cross_item_revoke_count
malformed_output_approval_count
validation_replay_mismatch_count
queue_reduction_rate (observation only)
recall (observation only)
```

Release assertions are precision at least 0.99 and every prohibited-count metric exactly zero. Do not tune labels after seeing a failing implementation without documenting a reviewed fixture correction. `auto_review_evaluation.py` has a real `argparse`/`main()` entrypoint with required `--fixture` and `--aggregate-only`, mode `fake` for deterministic tests or `live-openai` for the separately authorized gate, and no raw-output option. Fake mode must accept only an injected/frozen result fixture and construct no provider. Live mode additionally requires both `--allow-paid-provider-call` and `PARAWORKS_ALLOW_PAID_TERRA_EVAL=1`; either alone is insufficient. The CLI emits one aggregate-only JSON object containing the exact safety key `(purpose='validation', provider='openai', model='gpt-5.6-terra', reasoning_effort='medium')` plus prompt/output-contract/policy/cost-policy identities, aggregate metrics, tokens, and cost; it exits `0` only when all identities and gates pass, `2` for invalid configuration/authorization/fixture, and `3` for a metric failure. Tests assert each exit code and that stdout/stderr contain no fixture evidence, prompt, parsed output, id, URL, secret, or HMAC key material.

`auto_review_extraction_compatibility.py` separately owns a five-route compatibility harness. Its frozen fixture contains sanitized allowlisted evidence for `candidate` and `no_candidate` outcomes for every exact registry entry. Fake mode proves exact agent/prompt/output-schema tuples, singular 0/1 cardinality, item-type/payload/field-evidence integrity, the canonical-envelope aggregate 2,048-token validator including accepted-boundary and rejected-over-budget fixtures, exact rendered framed input <=10,000, Responses `max_output_tokens=2048`, one attempt, no fallback, usage parsing, per-route full-cap reserve **equals** USD 0.016716, and five-route full-cap reserve **equals** USD 0.083580 with six-place arithmetic. Live mode requires both `--allow-paid-provider-call` and `PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL=1`, accepts only `(purpose='extraction', provider='openai', model='gpt-5.4-mini-2026-03-17', reasoning_effort='none')`, and emits that exact safety key, all five agent/route/prompt/output-contract identities, extraction cost-policy identity, aggregate pass/fail counts, tokens, and cost—never prompts, evidence, model output, secret, or HMAC key material.

- [x] **Step 2: Run the golden gate**

```powershell
uv run --locked pytest backend/tests/test_auto_review_golden.py backend/tests/test_auto_review_evaluation_cli.py backend/tests/test_auto_review_extraction_compatibility_cli.py -q
```

Expected: all deterministic cases pass and no network/provider client is constructed.

- [ ] **Step 3: Run the separately authorized paid Terra offline gate before shadow rollout**

2026-08-30 execution note: separately authorized and attempted. The first live
request exposed an OpenAI `invalid_json_schema` rejection for the Pydantic
Decimal score representation. That boundary was TDD-fixed and the official
aggregate-only command was rerun. Provider schema validation then succeeded,
but execution was blocked by `credit_balance_exhausted` /
`insufficient_quota` (HTTP 429). This step remains unchecked; no live metric is
recorded and rollout remains disabled.

This is never part of pytest, CI, normal implementation verification, or an automatic continuation. After the code is otherwise green, stop and obtain explicit user authorization for paid provider calls. Use only the reviewed sanitized golden fixture—never production source data—and run the exact production prompt/schema against `gpt-5.6-terra` with reasoning effort `medium`, `max_retries=0`, and the same bounds/cost policy. Write only aggregate confusion-matrix counts, precision, prohibited-case counts, exact `(purpose='validation', provider='openai', model='gpt-5.6-terra', reasoning_effort='medium')` safety key, prompt/output-contract/policy/cost-policy identities, token totals, and cost; discard raw prompts and model outputs.

```powershell
$env:PARAWORKS_ALLOW_PAID_TERRA_EVAL='1'
$evalExit = 0
try {
    uv run --locked python -m backend.app.review.auto_review_evaluation --fixture backend/tests/fixtures/auto_review_golden_v1.json --mode live-openai --provider openai --model gpt-5.6-terra --reasoning-effort medium --aggregate-only --allow-paid-provider-call
    $evalExit = $LASTEXITCODE
} finally {
    Remove-Item Env:PARAWORKS_ALLOW_PAID_TERRA_EVAL -ErrorAction SilentlyContinue
}
if ($evalExit -ne 0) { throw "paid Terra aggregate gate failed with exit code $evalExit" }
```

The paid Terra gate must meet the same >=99% precision and zero prohibited-count criteria before an operator may authorize shadow. A Sol comparison is optional and requires a second explicit paid-call authorization; it is evaluation-only and can never become the C.5 validator or approval authority. If authorization is withheld or the gate fails, the implementation may be code-complete but rollout remains `disabled` and the docs must say the live model gate is pending/failed rather than copying deterministic metrics.

- [ ] **Step 3B: Run the separately authorized paid Mini extraction compatibility gate before shadow rollout**

2026-08-30 execution note: separately authorized and attempted. The first live
request exposed OpenAI `invalid_json_schema` rejections for Decimal score and
discriminated `oneOf` schema artifacts. The gate now reuses the OpenAI SDK
strict Pydantic conversion and TDD-tested provider normalization before the
real LangChain JSON-schema call. The post-fix official command reached provider
execution but was blocked by `credit_balance_exhausted` /
`insufficient_quota` (HTTP 429). This step remains unchecked; no live metric is
recorded and rollout remains disabled.

This is a distinct paid-call authorization and is never implied by Terra approval. Use only the sanitized five-route fixture and the exact production Responses API renderer. The aggregate gate must prove the exact `(purpose='extraction', provider='openai', model='gpt-5.4-mini-2026-03-17', reasoning_effort='none')` safety key and extraction cost-policy, all five agent/route/prompt/output-contract identities, each prepared request stayed within 10,000 framed input tokens, each response satisfied the singular `candidate|no_candidate` contract, its exact item/payload/field-evidence schema, and the complete canonical-envelope 2,048-token guard, output usage stayed within 2,048 total tokens, one attempt and zero fallback/retry occurred, each full-cap route reserve equaled USD 0.016716, the five-route full-cap maximum equaled USD 0.083580, and authoritative usage/cost accounting matched USD 0.75/M input plus USD 4.50/M output. No extracted text is written or printed.

```powershell
$env:PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL='1'
$extractExit = 0
try {
    uv run --locked python -m backend.app.review.auto_review_extraction_compatibility --fixture backend/tests/fixtures/auto_review_extraction_compat_v1.json --mode live-openai --provider openai --model gpt-5.4-mini-2026-03-17 --reasoning-effort none --aggregate-only --allow-paid-provider-call
    $extractExit = $LASTEXITCODE
} finally {
    Remove-Item Env:PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL -ErrorAction SilentlyContinue
}
if ($extractExit -ne 0) { throw "paid Mini extraction compatibility gate failed with exit code $extractExit" }
```

If authorization is withheld or any route/schema fails, rollout remains `disabled`; do not substitute an alias, Azure OpenAI, Gemini, another model, or a fallback route.

- [x] **Step 4: Run the controller-owned PostgreSQL restart, concurrency, audit, revoke, and reindex gate**

Use the approved controller to validate the existing shared service, create only its owned `_test` role/database, allocate serial schema leases, prove exact node coverage and zero skips, and clean exact owned resources:

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile postgres
```

Required zero-skip coverage includes concurrent human/auto transition exactly once; extraction E1/E2/E3 claim/cancel/crash/restart/one-call/atomic-candidate completion and global-disabled zero-call at both pre-attempt boundaries; the durable attempt marker committing before the body-blind transport consumes its one-use dispatch permit; validation lease/restart/cancel terminal-completed and terminal-failed replay, deferred parent/child terminal guards, pre-attempt-only reclaim, global-disabled zero-call, and crash-after-provider-send no-retry conservative charge; rotation refusal during attempt-zero/attempt-one calls followed by old-generation accounting recovery; provider-safety and rollout control transitions appending exactly one immutable old/new-key event plus aggregate backpointer while lost CAS/replay and metric-only updates append none; V2.0/V2.1 immutable evidence refs; exact duplicate/reaffirmation; Gmail/Drive missing/malformed semantic timestamp rejection and Calendar deterministic start; source update or parser-policy-only change versus extraction completion, promotion, reconciliation, and reindex; Source/Document two-phase crash recovery; permission-only vector narrowing and current-version pointer repair; promotion/revoke race; shared-provenance survival; last-provenance knowledge/companion/vector revoke; same-key advisory serialization of stale-snapshot reindex versus tombstone; stale/quarantined vector search exclusion before hidden count; persisted Assistant answer dependency commit versus concurrent source drift followed by zero-leak list/context/summary/email projection; mandatory-50 replacement audit non-skippability; 10%-to-100% authorization latch and full-enforce-only 2% sampling; purpose-specific breaker/global demotion while extraction or validation is in flight; business revoke rejected before assessment insertion; quality revoke versus pending audit, confirmed-audit correction, same-reason replay, and different-reason conflict; critical/manual-audit serving quarantine before revoke and through revoke failure/crash; remediation-required persistence; owner permission loss before/during/resume; reversed provider route-set/admin lock schedules; V2.0 paused resume; stored V2.1 resume without provider readiness; and exact generated-row cleanup. Every barrier-based schedule must have a bounded test timeout and zero deadlocks. Missing `PARAWORKS_TEST_POSTGRES_URL` is a release blocker, not a pass.

- [x] **Step 5: Run the complete C.5 and C/B compatibility suites**

The controller manifest freezes the same six selector groups formerly listed as independent raw pytest commands. It must prove exact canonical collection union, pairwise-disjoint verification children, native-exit/sidecar agreement, zero release-critical skips, and no live provider:

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile compatibility
```

Expected: all selected tests pass with zero skips/xfails/errors, V2.0 exact snapshots remain green, and no live provider is called. An unavailable optional environment is a Task 16 release blocker to configure or investigate, not accepted release evidence.

- [x] **Step 6: Run the approved non-Slack comparison gate**

Use the controller manifest's exact ten user-deferred Slack deselections and add none. The controller must prove selected nodes equal canonical collection minus exactly those ten:

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile non-slack
```

Expected: every selected test passes with zero skips/xfails/errors. An unavailable optional environment is a release blocker, not an accepted skip. Do not add a deselection for C.5.

- [x] **Step 7: Run the full backend suite and compare the deferred Slack baseline**

```powershell
uv run --locked python scripts/backend_release_matrix.py --profile full
```

Expected: the authoritative sidecars prove the same ten user-deferred Slack-related test ids are the only failures, with no errors/skips/xfails, and every C.5/non-Slack test passes. Native pytest exit `1` alone is never success. If the exact failure set or collection differs, stop and investigate rather than editing the manifest.

- [x] **Step 8: Run lock, Ruff, diff, and secret/privacy scans**

`test_secret_hygiene.py` NUL-safely enumerates `git ls-files --cached --others --exclude-standard`, so tracked plus untracked non-ignored release files are scanned even before Task 16's commit, and applies high-confidence OpenAI/Slack/GitHub/private-key/password-assignment patterns plus context-aware entropy rules. It explicitly allowlists documented placeholders such as `xoxb-test`, `<OPENAI_API_KEY>`, `example`, and null/empty configuration. Failure output contains only repository-relative path, line number, and detector kind—never the matched value or surrounding line. The test constructs a real-looking failing sample from split runtime fragments in an untracked temporary repository file, proves the scanner sees it and fails without echoing bytes, then removes it through fixture cleanup; documented examples must pass. This executable test replaces a noisy raw `rg` secret dump.

```powershell
uv lock --check
uv run --locked ruff check backend/app backend/tests
uv run --locked pytest backend/tests/test_secret_hygiene.py -q
git diff --check
rg -n "source_snippet|source_url|provider_exception|validation_key|lease_token" backend/app/agent_runtime/review_v21_state.py backend/app/agent_runtime/review_v21_graph.py backend/app/services/audit.py
```

Expected: lock/Ruff/diff and the executable high-confidence secret scan pass, and the bounded checkpoint/audit symbol scan shows no prohibited raw-content write. Inspect legitimate symbol references rather than treating every source-code name as a leak; never print a suspected secret to diagnose it.

- [x] **Step 9: Run all affected frontend gates on both viewports**

Modify `frontend/playwright.config.ts` so `PLAYWRIGHT_MANAGED_SERVER=1` adds exactly one Playwright `webServer` entry with command `npm run dev -- --hostname 127.0.0.1 --port 3000`, URL `http://127.0.0.1:3000`, `reuseExistingServer: false`, and timeout `120_000`; without the flag the existing external-server behavior remains unchanged. With no dev server active:

```powershell
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

Then let Playwright own start, readiness, and shutdown for each run:

```powershell
$listener = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw 'port 3000 must be free before the managed Playwright gate' }
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3000'
$env:PLAYWRIGHT_MANAGED_SERVER='1'
$testExit = 0
try {
    npm.cmd run test:visual -- review-hitl-v2-api.spec.ts review-hitl-v2-integrations.spec.ts review-hitl-v2-review.spec.ts auto-review-trust-promotion.spec.ts auto-review-knowledge-badges.spec.ts timeline-project-date-groups.spec.ts --project=chromium-desktop
    $testExit = $LASTEXITCODE
    if ($testExit -eq 0) {
        npm.cmd run test:visual -- review-hitl-v2-integrations.spec.ts review-hitl-v2-review.spec.ts auto-review-trust-promotion.spec.ts --project=chromium-mobile
        $testExit = $LASTEXITCODE
    }
    if ($testExit -eq 0) {
        npm.cmd run test:visual -- integration-sync-modal.spec.ts review-bulk-actions.spec.ts review-agent-metadata.spec.ts --project=chromium-desktop
        $testExit = $LASTEXITCODE
    }
} finally {
    Remove-Item Env:PLAYWRIGHT_MANAGED_SERVER -ErrorAction SilentlyContinue
    Remove-Item Env:PLAYWRIGHT_BASE_URL -ErrorAction SilentlyContinue
}
$listener = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
Set-Location ..
if ($listener) { throw 'managed Playwright server did not stop' }
if ($testExit -ne 0) { throw "Playwright gate failed with exit code $testExit" }
```

Expected: lint/build and every selected desktop/mobile test pass; port 3000 is closed afterward.

- [x] **Step 10: Run deterministic disabled, shadow, enforce, and rollback smokes**

`backend/tests/test_auto_review_smoke.py` contains two named tests and no live provider path. `test_disabled_sqlite_smoke` uses only explicit in-memory SQLite plus the process-local non-ready test identity (no durable keyed row survives the process), proves a new disabled run selects V2.0, and keeps both extraction and validation call counters zero. A separate contract test proves file-backed SQLite with a placeholder key refuses every C.5-bound durable write. `test_postgres_control_plane_shadow_enforce_rollback_smoke` uses only a disposable `_test` PostgreSQL+pgvector database, a 32+ byte test-only key/version, fake extraction/validator adapters, and the real migration/services/CLI entrypoints. Its setup/order is executable and exact:

1. deploy/migrate while configured `disabled`, with a strong test-only fingerprint key/version;
2. run key bootstrap, bounded current-document-pointer repair/re-sync fixture, source reconciliation, projection rebuild, and status checks; abort unless runtime/projection are ready and stale/ambiguous counts are zero;
3. through `AutoReviewRolloutAdminService`—never raw row inserts—authorize the deduplicated sorted required safety-key set exactly once per key: one `(purpose='extraction', provider='openai', model='gpt-5.4-mini-2026-03-17', reasoning_effort='none')` policy shared by the five route entries, and one `(purpose='validation', provider='openai', model='gpt-5.6-terra', reasoning_effort='medium')` policy, using explicit deterministic test-gate evidence; this does not stand in for either separately authorized paid release gate;
4. run shadow with fakes and assert the registry has exactly five entries; every selected extraction route is the exact Mini snapshot/none/strict singular schema with 10,000/2,048 caps, one attempt, no fallback, and at most five calls reserving USD 0.083580; validation uses at most two Terra calls reserving USD 0.097728; the five-agent total is USD 0.181308; no cache/trace/debug output occurs; and every item remains pending;
5. record 500 valid fake shadow comparisons through public rollout/audit services, authorize only `0 -> 10`, and run a fixed HMAC-selected 10% workflow; complete all mandatory-50 slots with human-confirmed service calls and prove invalidated replacements do not advance the confirmed counter;
6. record the remaining 500-canary/gate evidence through the same services, complete every selected audit with zero critical outcomes, authorize `10 -> 100`, and prove only a **new** signed workflow receives 100%; no row/counter is seeded or patched directly;
7. prove selected supported Timeline/History can auto-approve, non-selected/high-risk candidates interrupt, and a zero-candidate run uses `finalize_no_candidates` while an all-auto-resolved run uses `finalize_auto_resolved` without interrupt;
8. commit a critical audit, inject revoke failure/crash, and prove the exact auto effect is immediately quarantined from Knowledge/Dashboard/Project/RAG/pgvector before recovery; restart recovery then performs exact revoke without resurrection;
9. change source permission/version and prove live serving excludes/narrows before reconciliation, then run reconciliation/reindex and prove stale vectors remain absent;
10. switch global mode to `disabled` and resume a stored V2.1 workflow at both pre-extraction and pre-validation boundaries; both provider counters remain unchanged, attempt-zero reservations terminalize at zero charge, and the stored graph identity is not rewritten;
11. prove exact direct revoke removes only its serving effect, the human C flow remains available, and no Slack source/connector is constructed.

Run:

```powershell
uv run --locked pytest backend/tests/test_auto_review_smoke.py -q -k disabled_sqlite_smoke
if (-not $env:PARAWORKS_TEST_POSTGRES_URL) {
    throw 'PARAWORKS_TEST_POSTGRES_URL is required for the C.5 release smoke'
}
uv run --locked pytest backend/tests/test_auto_review_smoke.py -q -k postgres_control_plane_shadow_enforce_rollback_smoke
```

The PostgreSQL test itself parses both database and user names and fails unless each ends in `_test`; it runs migrations against a fresh schema and fails on any skip. Record bounded counts, graph/mode/policy/key generations, aggregate call counts/costs, gate states, and statuses only—never ids or raw fixture content.

- [x] **Step 11: Update product truth with observed evidence**

Only after every required non-paid gate passes. If the separately authorized
live gates remain pending, record that fact and keep rollout disabled:

- mark C.5 implemented/verified in `plan.md` and keep D, E, and Slack in their approved order;
- record actual pass/fail/skip counts, dependency path, UX, golden metrics, PostgreSQL target rules, rollback, and known Slack baseline in `docs/portfolio-log.md`;
- record exact commits, migration head, flags, operational gate requirements, breaker state, next unapproved Deliverable D boundary, and no-live-provider test evidence in `docs/superpowers/runbooks/session-handoff.md`.

Do not copy this plan's expected results as if they were observed.

- [x] **Step 12: Commit verified release evidence after reviewed behavior-slice commits**

All test/helper/controller and product behavior must already be present in the reviewed intermediate commits required by their owning implementation plans. This final commit contains only observed product truth and release evidence; do not squash or recommit the behavior files here.

```powershell
git diff --check
git add plan.md docs/portfolio-log.md docs/superpowers/runbooks/session-handoff.md docs/superpowers/runbooks/backend-release-matrix.md
git commit -m "docs: record auto review release verification"
git status --short
```

Expected: the documentation/evidence commit succeeds after all behavior slices and the worktree is clean. Do not push, merge, enable shadow/enforce in a deployed environment, or open a PR unless the user separately requests it.

## Final Implementation Review Checklist

- [ ] Every task's named RED command was observed before production edits, and its GREEN command was rerun after the smallest implementation slice.
- [ ] V2.0 schema dumps, four-status checkpoint, topology, builder key, paused tuple, status/resume/cancel behavior, and frontend transport remain exact.
- [ ] V2.1 is a separately registered real LangGraph with real `interrupt()` and PostgreSQL-authoritative live counts; all-auto completion skips the interrupt only by approved routing.
- [ ] Production validation reaches real LangChain `with_structured_output()` on OpenAI `gpt-5.6-terra` medium; there is no Luna/Sol/Gemini or deterministic production fallback.
- [ ] The extraction registry contains exactly the five frozen agent entries; every V2.1 selected route uses OpenAI `gpt-5.4-mini-2026-03-17`, reasoning `none`, its exact strict singular output schema, 10,000/2,048 token caps, one attempt, and no alias/Azure/Gemini/fallback path.
- [ ] Full-cap arithmetic is exact: at most five extraction calls reserve USD 0.083580, at most two validation calls reserve USD 0.097728, and the maximum combined reserve is USD 0.181308 under USD 0.20.
- [ ] `max_retries=0`, one authoritative call ledger, atomic reservation, conservative unknown-usage charge, and child allocation prove that admission/reservation/replay cannot exceed or multiply the signed ceilings; a provider-reported actual overrun is charged exactly once, never approved, and opens the matching purpose breaker.
- [ ] The deterministic versioned policy—not model confidence—owns approval, and exact `Decimal('0.9800')`, direct-fact, field/slot, permission, identity, version, and cost requirements are all enforced.
- [ ] No raw source/model content, URL, canonical/external id, hidden/other ReviewItem id, alias mapping, prompt, rationale, provider error, or credential entered a checkpoint, validation row, signed token, or public summary. The separately access-controlled AuditLog may store only the exact current action target ReviewItem id required by its existing governed schema, never a hidden/related item id or raw content.
- [ ] Public/internal permission is exact, restricted/unknown always fails closed, hidden collision uses existence only, and strictest source/item/knowledge permission is preserved.
- [ ] Workflow-owner permission is re-resolved before/after provider and on resume, intersected with the auto-policy allowlist, and never replaced by the system actor's broader capability.
- [ ] Every candidate starts pending; disabled/shadow cannot create trusted knowledge; decision/todo/inference/conflict/uncertainty stay human-reviewed.
- [ ] Human and auto approvals/reaffirmations share one locked transition/promotion boundary and every post-migration effect has complete approval plus evidence provenance.
- [ ] Exact duplicate reuse creates no knowledge duplicate; one provenance revoke cannot remove another provenance; last-provenance revoke is replay-safe and all-or-nothing.
- [ ] Raw chunk indexing remains human/legacy-human only; shared per-document transaction locks, tombstone-aware conditional upsert, active search filtering, and commit-aware in-memory deletion make revoked vectors non-resurrectable.
- [ ] Server and connector signatures are separate; only `server-source-content:v1` plus `current_document_version_id` authorizes C.5/current raw-chunk serving, and source drift, pointer ambiguity, critical-audit quarantine, or lookup failure is excluded before API/RAG/pgvector ranking and hidden counts.
- [ ] V2.1 extraction and validation both use the credential scanner, global LangChain debug-off/explicit verbose-false guard, empty callbacks, disabled trace/cache, one immutable render, and provider-outside-transaction boundary; stdout/stderr capture proves zero raw prompt bytes.
- [ ] Dry-run/status/page polling are zero-call; one signed preview and existing button authorize the paid run; changed preview requires a fresh explicit click.
- [ ] The launch token binds the aggregate extraction plan, complete sorted extraction/validation safety snapshots, exact caps/prices/total budget, and rollout control epoch; start recomputes and locks them before creating a thread.
- [ ] Mandatory-50 replacement audits and 10% canary audits are replay-safe; 100% requires exactly 50 human-confirmed mandatory audits plus a separate latched authorization after 500 enforce promotions and all other gates; 2% applies only to new full-enforce workflows; critical audit opens a persistent breaker and serving quarantine before revoke, and neither breaker close nor remediation failure can silently restore enforce or serving.
- [ ] Review defaults to pending, automatic trust/audit/revoke stays on the same screen, and Timeline/History/Knowledge keep evidence while showing human versus automatic trust.
- [ ] Golden precision is at least 99% with zero hard-negative, permission/version, duplicate, cross-item revoke, malformed-output, or replay violations.
- [ ] The deterministic golden gate is never represented as Terra quality; a separately authorized sanitized Terra aggregate gate is recorded before shadow, or rollout remains disabled with the gate explicitly pending.
- [ ] Deterministic extraction fixtures are never represented as live Mini compatibility; a separately authorized sanitized five-route Mini aggregate gate is recorded before shadow, or rollout remains disabled with that gate explicitly pending.
- [ ] PostgreSQL, complete C.5, C/B compatibility, non-Slack, full backend comparison, frontend desktop/mobile, lock, Ruff, diff, secret, and rollback gates have fresh recorded evidence.
- [ ] Config rollback preserves C.5 audit history; populated schema downgrade and local row-reset refuse destructive cleanup.
- [ ] Key rotation refuses every non-terminal extraction/validation call and succeeds only after old-generation zero/one-attempt ledgers are terminally recovered and accounted without a provider retry.
- [ ] Slack, CDC/streaming, Deliverable D retrieval, Deliverable E Neo4j GraphRAG, separate vector stores, and unrelated refactors remain untouched.
- [ ] Default mode remains disabled and no push, merge, PR, or deployed rollout occurs without separate authorization.

## Execution Handoff

This plan is approved. Before product implementation begins, explicitly authorize it and choose one execution mode:

1. **Subagent-Driven Development (recommended):** Stay in this task, invoke `superpowers:subagent-driven-development`, assign each independent task to a fresh implementation worker, and perform spec/code review checkpoints after every task.
2. **Inline Plan Execution:** Stay in this task, invoke `superpowers:executing-plans`, execute the tasks sequentially with the named RED/GREEN/commit checkpoints.

Creating and approving this document is still **planning**. The next unapproved step is actual product-code implementation, gated by an explicit implementation authorization and execution-mode choice. Plan approval alone does not authorize code/migration changes, pushing, merging, opening a PR, enabling a paid mode, or touching Slack.

Tasks 1–5 are complete. The next product slice remains Task 6, then Tasks 7–15. Task 16 has an additional gate: `docs/superpowers/plans/2026-08-29-whole-suite-postgresql-isolation.md` must be separately user-approved and implemented first at Task 16 entry. Completing that isolation plan permits the Task 16 release proof only; it does not automatically authorize Deliverable D.
