# ParaWorks Harness Session Handoff

Updated: 2026-09-01

## 2026-09-01 Deliverable D Task 12 dispatch/accounting closure

- Task 12 now composes `RagCostLedger` with the exact frozen `RagCostPolicy`
  and concrete `RagProviderSafetyService`; do not restore public actual-cost or
  safety-action callbacks. Safety mutation/revalidation completes before the
  component terminal ledger write.
- Provider dispatch uses `RagProviderDispatchAuthority`: callers supply typed
  server inputs, never serialized request bytes, provider clients, or per-call
  send/evidence/safety callbacks. The ledger owns the client capability. The
  authority server-builds canonical bytes and consumes its opaque prepared
  state once inside the continuous provider-safety, projection-owner, and
  evidence/C.5 barriers. Every real acquisition requires its exact order
  capability.
- `_assemble_direct_openai_rag_provider_dispatch_authority` is PostgreSQL-only
  and must load all five committed static advisory capabilities before it may
  create provider or safety artifacts. SQLite uses only the explicit
  provider-free deterministic smoke assembly; never restore the paid
  assembler's SQLite fallback. The PostgreSQL recovery/send gate must remain a
  real two-session lock interleaving rather than a sequential state change.
- Recovery is fail-closed and never redispatches. Dead dispatch becomes
  `abandoned_unknown` admission-only while preserving prior actual/reserve;
  pending projection recovery requires the exact owner fence and ends as
  `persistence_failed` final.
- The additive `a4d5e6f7b8c9` head corrects pending-projection null outcome and
  permits the frozen reviewed inter-component admission-only recovery shape.
  Do not remove it or point head back to `f3c4d5e6a7b8`.
- Fresh affected Task 12/RAG V2 verification: `638 passed, 12 skipped`; Ruff,
  compile/import, diff, and changed-lines secret scan are green. The disposable
  PostgreSQL URL was absent, so its executable gates remain a release blocker,
  not claimed execution. No Docker/network/provider/paid/`.env` access.

## 2026-08-31 Deliverable D Core implementation plan

- The approved D Core design now has a separate 27-task implementation and
  release plan (26 TDD implementation tasks plus one final provider-free
  evidence task) at
  `docs/superpowers/plans/2026-08-31-deliverable-d-core-rag-answer-graph-v2.md`.
  It is a planning artifact awaiting separate approval; no D production code
  has changed. After approval, begin actual implementation at Task 1 using the
  subagent-driven execution mode unless the user chooses inline execution.
- The plan maps all approved design sections 1–23 to executable proof and uses
  five phases: contracts/identity/storage, canonical evidence/retrieval,
  structured generation/cost/LangGraph, V1 API/Assistant UX, and rollout/safety/
  release. It requires real LangChain `Runnable` ports and a real request-local
  compiled LangGraph `StateGraph`, never a custom callable presented as either.
- Persistence is deliberately split into serving revision
  `d1a2b3c4e5f6` (down `9d7f3a1c6e20`) and runtime-safety revision
  `e2b3c4d5f6a7` (down `d1a2b3c4e5f6`). Release exact-six tables stay in
  validation-only metadata outside application metadata and Alembic.
- Implementation and automated validation use fake/deterministic providers;
  paid call count remains zero. Do not initialize or run the live gate from
  this plan. Only after Tasks 1–27, provider-free gates, a clean commit, and an
  exact zero-call preview may the user separately authorize at most 30 cases,
  30 generations, 10 embeddings, 40 dispatches, and USD `0.360000`. The USD
  100 account balance is availability only and does not authorize expansion,
  retry, production traffic, reindexing, D.1, or E spend.

## 2026-08-30 Deliverable D Core written design

- Branch `codex/rag-orchestrator-agent` now has the section-level approved D
  Core direction consolidated at
  `docs/superpowers/specs/2026-08-30-deliverable-d-core-rag-answer-graph-v2-design.md`.
  The consolidated written spec is approved. This is planning/spec
  documentation, not production implementation. The next task is a separate
  TDD implementation plan and remains planning until that plan is approved.
- D Core uses a separate exact-version `RagGraphRegistry` and a request-local
  LangGraph `RagGraphState`; it does not use the durable Review Queue
  `GraphVersionRegistry`, checkpointer, `AgentWorkflowThread`, or checkpoint
  rows. API routes call an application facade rather than LangChain/LangGraph
  directly. RAG registers only in a separate app-wide manifest registry; the
  C.5 exact-five `review_agent_registry` remains sealed and unchanged.
- Keyword and pgvector implement one application-level LangChain
  `Runnable[RetrievalRequest, RetrievalResult]`. Internal
  `serving_document_id` is distinct from V1 public `source_id`; raw is
  `chunk:{id}` internally and exact `Source.source_id` publicly. The model sees
  only `E1..E8`, and fresh canonical projection occurs after revalidating every
  model-visible influence; public citations remain the selected subset and
  output permission remains the strictest influence permission.
- Trusted human/auto-approved knowledge is first. Current canonical
  Gmail/Drive/Calendar raw evidence may support only `source_observation`; the
  exact source-kind allowlist is `gmail | gmail_attachment | drive | calendar`.
  D Core includes the incremental raw-observation indexing lane required for
  keyword/pgvector parity, with Developer B/C shared-contract review. A live
  production reindex or embedding spend requires separate operational
  approval. Pending candidates and Slack are excluded. Preserve the exact ten
  deferred Slack failures and repair Slack last. Actorless raw index jobs use a
  canonical lineage/version/permission resolver; request-time V2 retrieval
  composes that observation with `authorize(SecurityScope, observation)`.
  Legacy trusted predicates and pgvector SQL remain unchanged, so disabled
  mode does not broaden raw visibility and indexing never fabricates an
  all-access actor. V2's trusted branch exact allowlist is the four promoted
  canonical knowledge types; `chunk:*` is always a single
  `source_observation` V2 member even when its legacy ReviewItem is approved.
  The legacy chunk predicate remains compatibility-only and cannot launder a
  raw row into `trusted_fact`. SecurityScope uses server-owned workspace mode,
  typed `project_key:`/`source_pk:` refs, explicit all-current vs constrained
  semantics, and fail-closed missing membership. Trusted multi-provenance is
  globally strictest first, then selects only an actor-authorized link whose
  every evidence child is visible; it never exposes an unauthorized
  higher-priority link. Raw/trusted model text, source ids/URLs, and snippets
  must be strict-scalar, NUL-free, and nonblank. Raw/current explicit snippets
  must match the versioned parser rule `" ".join(text.split())[:240]`; invalid
  evidence is excluded consistently and any D-tracked vector is tombstoned.
- Direct `/ask` and `/search` preserve V1 character-count and lexical-term
  semantics; D adds no 4,000-character or 1,000-term refusal to those surfaces.
  Assistant keeps its existing 4,000-character current-message bound, an
  8,000-character server-built contextual query, and the contextual-query-only
  1,000-term ceiling. Other bounds are candidate window 50, `/search` results
  5, answer slots 8, public hidden count 20, model input 12,000 serialized
  chars, output 512 tokens, and total paid request ceiling USD `0.012` including
  query embedding. Answer model is exact `gpt-5.4-mini-2026-03-17`, reasoning `none`,
  strict structured output, one call, no retry or model fallback. The provider
  schema is one hand-authored OpenAI-subset literal passed through LangChain's
  exact strict wrapper; prompt-renderer/two-message JSON bytes and the
  `"\n\n"` answer joiner are separately HMAC-bound. Server semantic validation
  enforces block/reason XOR, slot/trust/length constraints, and rejects model
  NUL/surrogates without repairing bytes. V1 projection HMACs additionally bind
  score binary64 bits, matched-term order, nullable fields, and source arrays.
  One immutable
  `PreparedAnswerInvocation` binds the final rendered bytes, evidence slots,
  HMAC, usage estimator, and cost reserve. Question/frame credential matches
  block the whole provider call; unsafe evidence is dropped and the invocation
  is completely re-prepared before any call. Pgvector also has one immutable
  request-local query-embedding result with validated usage/cost, shared by
  legacy/V2 shadow paths rather than mutable side-channel accounting. Pgvector
  direct `/ask`/`/search` uses exact caller UTF-8 query bytes in both paths;
  normalized text is validation/scanner-only and a hex+length HMAC preserves
  byte-distinct Unicode identity. Pgvector admission reconciles the full
  exact-four trusted plus raw serving corpus, not
  raw alone. Corpus/index generations are checked before embedding and around
  SQL; concurrent ingestion/promotion/revoke discards partial vectors and does
  one same-scope keyword recomputation without another embedding. Answer paths
  also fresh-recompute bounded hidden matches after generation, reusing the
  same vector or falling back to same-scope keyword without a second embedding.
  Production query embedding has stable family discriminator
  `openai-embeddings-api:v1` and separate `rag-query-embedding-config:v1`
  snapshot with exact 1,536 dimensions; config changes rebind the same family
  instead of creating a fresh-ready bypass. Generation preflight counts compact
  canonical JSON of exact messages plus the strict output schema with
  `o200k_base`, no Unicode normalization, `+16` reply priming and `+512` safety.
  Shared strict chat/embedding usage parsers reject coercion and conflicting
  aliases. A returned response with invalid usage blocks the component family;
  a response-less transport exception remains an ordinary reserved failure.
  For query embedding, it preserves the full reserve but immediately closes
  `retriever_unavailable`; keyword fallback, safe 200, downstream answer
  preparation, and generation calls are all zero.
  Query and corpus vectors must be canonical finite float32, exact-dimension,
  and nonzero after conversion. Writer and DB readiness share this cosine-
  indexability rule (`vector_norm(embedding) > 0`), so pgvector cannot report a
  zero vector ready even though cosine indexes omit it.
  D V2 also rejects pre-provenance `legacy_unbound` trusted rows that have no
  valid explicit approval branch and no linked legacy ReviewItem. Its sole
  legacy branch requires a non-null approved human/non-C5 ReviewItem and
  nonblank, NUL-free, equal-length, byte-equal target/ReviewItem link-snippet arrays;
  effective permission is their strictest value. Disabled legacy output remains
  unchanged, while a shadow mismatch blocks stage advancement until evidence is
  reviewed/migrated.
- New `rag-run:v2` traces retain versioned keyed HMACs, bounded counts,
  latency/usage/cost, and allowlisted errors only. They retain no question,
  evidence text/URL/snippet, prompt, model output, provider error, state, or
  checkpoint. Assistant intentionally retains its owner-bound conversation,
  selected citation bytes, and identities/HMACs for every unselected model
  influence; future dependency-set HMACs bind the full ordered influence set,
  parent content, AgentRun/result, prompt/write-mode, and approval/evidence links.
  V2 uses exact-byte content writes while legacy retains trimming. An additive nullable
  parent marker distinguishes historical `legacy-per-dependency-sha256:v1`
  rows from future `assistant-dependency-set-hmac:v2` writes; no backfill or
  unguarded old-binary downgrade is allowed. Every future evidence-backed
  Assistant write uses V2 HMAC even after mode rollback. The writer is
  integrity-only: it may sign a disabled legacy dependency as
  `legacy_v1_only` without promoting it into D retrieval/readiness; V2 graph
  writes remain `rag_v2`. C.5 keeps one active
  key; rotation intentionally redacts older keyed stored answers instead of
  retaining an old keyring or re-signing history.
  Every paid admission uses `status=running`, `run_record_phase=admission`, and
  explicit sentinels. Its `rag-admission-identity:v1` binds configured surface/
  backend plus query/security/config HMACs and is never an answer-cache lookup
  key. Only a substantive, search, or safe terminal product projection writes
  `run_record_phase=final`, effective source-window/permission/actual-model
  fields, and `rag-final-product-identity:v1` over admission identity, result
  HMAC, and surface; projectionless errors and complete shadow-only cost owners
  use separate explicit D.1-ineligible final-error/final-shadow sentinels.
  None of these identities enables D Core reuse.
  Assistant's provider-free failed/final parent with exact-two terminal-zero
  children is available only when no paid claim exists. Once pgvector query
  embedding has returned a validated successful vector, an inter-component
  refusal must reuse the existing run, preserve the embedding actual cost and
  dispatch, terminalize the answer child at zero, and make no generation call.
  A dead process, including a crash between embedding and generation, closes
  the parent as failed/`abandoned_unknown` and never resumes provider work.
- D Core has no answer reuse. D.1 is a separate default-disabled PostgreSQL
  cache design/plan after D Core is green; Redis is considered only after a
  measured bottleneck. E Neo4j GraphRAG follows D.1; CDC/streaming and Slack
  remain deferred.
- Keep the plan-required SQLite path as a deterministic, provider-free,
  single-process smoke oracle only. A process-local mutex serializes writers
  while a never-replaced process-lifetime OS lock rejects a second file-backed
  smoke process. Neither is live, release, or paid-call authority; second-process,
  pgvector, live, and paid modes refuse before any call. Production vector
  writes remain PostgreSQL+pgvector-only, and D Core has no SQLite answer cache.
- Rollout uses deployment-static `LANGGRAPH_RAG_V2_MODE` and
  `LANGGRAPH_RAG_V2_STAGE=none|ask|search|assistant`; an operator can roll back
  stage or mode. Shadow compares retrieval only and never performs dual answer
  generation. Common-cohort comparison canonicalizes legacy output and allows
  only five intended deltas: V2 raw observations, trusted-tier reorder, raw
  public-id repair, bounded hidden-count semantics, and the user-only Assistant
  context security delta. Any other mismatch
  fails the gate.
  The user authorized a first sanitized 30-case reserve envelope with an exact
  maximum of USD `0.36`; a provider contract overrun is recorded unclamped and
  fails/aborts the gate rather than being represented as a green bounded run.
  The user's later note that up to USD `100` is available does not expand this
  frozen first gate; only a future separately previewed/approved tuple may do so.
  Exact execution remains unauthorized until the final
  clean runner/fixture commit passes provider-free gates and the user confirms
  its zero-call preview. No paid D call has occurred. A partial/crashed
  restart, rerun, expansion, production traffic, raw-observation reindex, D.1,
  or E paid call requires fresh user approval. The final preview must freeze
  the 10/5/10/5 ask/Assistant keyword/pgvector manifest, Git hash, designated
  validation PostgreSQL ledger/marker identity, validation database identity
  HMAC, current generation, and exact current provider authority/envelope plus
  two active family identity/state/version/generation/config-policy snapshot
  HMACs. It does not contain an approval id/HMAC yet; only after the user
  confirms that tuple may `authorization-bootstrap` fresh-match the same whole
  provider snapshot and create its bound approval id/HMAC. Any reset, rebind,
  supersession, breaker transition, or key rotation makes the old authorization
  zero-call stale and requires a new preview/user approval. The DB
  ledger is paired with an HMAC-bound
  monotonic authority outside the repo and DB backup/restore set; any restore,
  identity, generation, or marker mismatch is zero-dispatch and runner repair
  is forbidden. The ledger allows exact 30 case claims, at most 30 generation
  and 10 pgvector embedding component dispatches (40 total). The quality gate
  binds a provider-free baseline definition and three pairwise-distinct
  authenticated human review roles, with no paid LLM judge; green completion
  requires exact 30/10/40. After all 30 frozen cases are terminal, an ordinary
  case failure or a shortfall in the exact component/distribution counts closes
  terminal `finished_failed` with the actual lower counts and no retry/resume.
  Its authorization outcome is respectively `ordinary_execution_failed` or
  `execution_contract_failed`; if adjudication was not reached, the quality
  report is optional. Live calls use one
  composite permit: the release dispatch and runtime AgentRun/cost reserve are
  committed together in the exact same physical validation PostgreSQL
  connection/transaction. One authorization-scoped singleton runner/fence owns
  the whole 30-case execution through scoring, human adjudication, and the final
  transition. Reviewed proof that it died or was drained terminalizes the
  authorization as `aborted_execution_crash` with `abandoned_unknown`, preserves
  committed costs/reserves, and forbids partial resume, provider retry, or
  same-approval reuse. Any live current-corpus snapshot drift immediately
  terminalizes it as `aborted_corpus_drift`, preserves committed and current
  actual-or-reserve cost, permits no remaining call/scoring/report/retry/resume,
  and requires a fresh preview and user approval. Any known overrun records
  unclamped actual,
  moves the whole gate to `aborted_overrun`, and permits no later component.
  Runtime provider readiness uses a stable safety family and an external
  deployment-wide HMAC latch; config/key/model version changes cannot bootstrap
  around a block. It blocks every D-managed admission and later D re-enable
  until reviewed reset, but disabled/non-cutover legacy calls keep exact V1
  behavior after rollback. The latch is one canonical whole-family-set envelope with an
  active-family map, preserved historical blockers, a global generation, and a
  distinct never-replaced ACL-checked sidecar lock. The release marker has its
  own distinct stable sidecar. Only signed data files are atomically replaced;
  every init/read/recovery/admission/finalization locks the stable object first.
  Live-release processes use a global validator requiring provider data/sidecar
  and release data/sidecar to be four pairwise-distinct, non-symlink/non-reparse,
  non-hardlinked leaf files; any cross-alias is zero-call/fail-stop. Ordinary
  disabled/non-cutover startup requires no D authority artifacts and never
  creates a dummy release authority. Provider-free `provider-safety-init` is the
  sole provider-authority bootstrap; every non-bootstrap provider admin mutation,
  actual D paid-component admission, and privileged release init/recovery/
  preview/authorization/runner requires the initialized provider data/sidecar/DB
  peer, and release roles validate all four provider/release leaf paths.
  Its DB peer has a singleton generation/digest,
  exact two active family rows, and append-only transition history. Only the
  provider-free `provider-safety-init` may create generation 0 after proving
  empty DB history, missing latch, and zero D paid attempts; it writes the file
  first and one DB transaction second. It targets the deployment application
  DB; the live gate specifically targets its exact validation DB, never a
  cross-shared authority. A second init is refused, and the sole
  recoverable partial-init shape is a reviewed valid-generation-0-file/empty-DB
  recovery with zero attempts. Runtime pre-call and every post-call finalizer
  use stable provider sidecar lock
  -> DB singleton -> family -> run -> ordered cost-child locking. Live calls
  extend this to provider sidecar -> release sidecar -> advisory ->
  safety/release/run rows, with safety-first overrun finalization. Thus an external
  blocker flush cannot race a stale-ready output commit and simultaneous
  embedding/generation incidents cannot overwrite each other. Known provider
  overrun blocks externally first, then commits the readiness breaker, failed
  run, and exact two cost components before a separate Assistant safe-message
  write. Normal supported/search outcomes use a two-phase contract: immutable
  component costs commit first with the parent in
  `cost_finalized_pending_projection`, then provider safety is reacquired and a
  C.5-compatible corpus lock transaction revalidates every model influence,
  selected citations, and hidden membership and atomically finalizes the
  direct projection or Assistant message/dependencies plus parent outcome.
  A dedicated pre-send evidence fence serializes source/revoke/permission/
  promotion/parser/index mutation through immutable transport-body handoff.
  A writer rollback never erases paid-call accounting or permits a provider
  retry; bounded recovery closes the parent as `persistence_failed`. Every runtime paid component
  first commits a full-reserve `dispatching` claim; an uncertain crash becomes
  `abandoned_unknown` with no retry. Pgvector shadow keeps the legacy public
  response/run but uses a separate internal `rag-run:v2` exact-two cost owner
  for its one shared query embedding, avoiding both unaudited spend and double
  charge.
  Live `component_outcome` terminalizes cost only; `case_outcome` atomically
  closes the case and AgentRun under the retained provider/release/corpus locks.
  The terminal authorization outcome distinguishes ordinary execution failure,
  execution-contract count failure, and the all-executed rubric-red
  `quality_gate_failed`. An append-only quality report is optional for the first
  two when adjudication was not reached, but required for rubric-red and green
  completion.
  Both external files use exact non-recursive signed-payload/HMAC envelopes;
  canonical file and transition digests must also match their DB peers even
  when generations match. Release DB rows are scoped by
  `(ledger_uuid, ledger_epoch)`, transition generation is gapless within an
  epoch, and generation 0 has no transition row. A reviewed restore recovery
  creates a new epoch from the valid predecessor marker external-first, keeps
  old rows read-only, and requires a fresh preview/user approval; a missing or
  corrupt marker requires a new ledger UUID. If provider readiness changes
  after the 30th case outcome, the case-null `authorization_abort_final`
  transition preserves final aggregates instead of misusing a case-bound abort.
- Assistant compatibility also covers pre-message first-turn behavior:
  conversation titles/`initialQuery` use the common credential scanner before
  lookup or commit. Owner-scoped conversation/message misses retain their exact
  existence-hiding 404 responses rather than becoming RAG 403. Generic 500/502
  paths refetch under an active request/conversation guard, reconcile the
  authoritative rows once, and show Korean safe copy; a failed refetch keeps
  one status-unknown frontend-only row whose action retries GET reconciliation,
  never the non-idempotent POST. First-query navigation uses a consume-once
  in-memory handoff rather than raw `?q=` URL/history data.
- Assistant V2 retrieval context contains current/prior user rows only; answer
  generation receives the current turn only. Prior assistant bytes never enter
  V2 retrieval/model input, and shadow records that intended security delta.
  The enforce facade/graph is the sole assistant-row writer and returns a typed
  delivery state, so route/catch paths never append a second success/failure row.
  V2 output is exact-byte plain text; only server-validated HTTP(S) citations
  become links. Direct OpenAI identity is pinned to the standard global endpoint,
  answer `service_tier=default`, and conservative list-rate charge accounting.
- Next: invoke `superpowers:writing-plans` for a separate failing-test-first D
  Core TDD implementation plan. That next task is still planning; do not change
  production code before the implementation plan receives separate approval.
  D.1 planning starts only after D Core is green.

## 2026-08-30 single-root local environment contract

- Local runtime configuration now has one canonical ignored file: root `.env`.
  Backend Settings, Docker Compose, local Python scripts, and Next.js consume
  it; `frontend/.env.local.example` was removed.
- Next.js uses a tested `dotenv` parser boundary that copies only
  `NEXT_PUBLIC_API_BASE_URL` and `NEXT_DIST_DIR`. `OPENAI_API_KEY` and every
  other backend-only value remain outside the frontend environment.
- The existing ignored `.env.local` key was migrated to ignored `.env` without
  outputting its value, verified in the target, and only then removed from the
  worktree. Neither env file is tracked.
- `.env.example` is the documented template. Obsolete API host/port and MinIO
  entries are gone and tracked signer values are blank. The idempotent
  `scripts/bootstrap_local_env.py` creates/fills ignored `.env` with independent
  C.5/session/Google/Slack signing secrets while preserving provider keys. C.5
  remains explicitly disabled; Slack settings are grouped last and deferred.
- Fresh evidence is backend/config/bootstrap/secret `76 passed`, frontend env
  `3 passed`, Ruff/ESLint clean, Next.js production build complete, and an
  independent final review with no unresolved finding (`Ready: Yes`). No live
  provider call occurred.
- Next product activity remains **Deliverable D planning**. No paid provider
  call, rollout, D/E implementation, Slack recovery, push, merge, or PR is
  authorized by this configuration refactor.

## 2026-08-30 C.5 paid release gates complete

- Branch `codex/rag-orchestrator-agent` completed both separately authorized
  aggregate-only paid gates using the existing `.env.local` key only inside the
  child process. No key or raw provider content was logged or committed.
- Terra's final production identity is OpenAI `gpt-5.6-terra`, medium,
  `auto-review-validation:v2`, and `candidate-validation-batch:v1`. Candidate
  and claim order plus candidate-local evidence-slot allowlists are explicit.
  The 18-case live gate passed precision `1.0`, all prohibited counts `0`,
  queue reduction `0.055556`, recall `0.333333`, tokens `3726/2554`, cost USD
  `0.038100`.
- All five extraction entries now use their agent-specific `*:c5-v2` prompt and
  `*:c5-v2` output identity with exact Mini snapshot
  `gpt-5.4-mini-2026-03-17`, reasoning `none`. Canonical prompts list allowed
  item types, candidate-local slots, required/optional field maps, and exact
  binding rules. One known slot may support multiple different fields; field
  keys remain unique and every populated field requires one known slot. The
  live gate passed `5/5` routes, `9/9` checks, tokens `9439/954`, cost USD
  `0.011372`.
- Fresh post-change official compatibility proof is `1,599 collected / 1,595
  passed / 4 exact Slack deselected`, errors/skips/xfails `0`, leases `91/91`,
  `no_live_provider=true`, and DB/role residue `0/0`. Focused regression is
  `119 passed`, `226 passed`, and eligibility `23 passed`; Ruff, lock, and
  secret hygiene pass. A raw all-C.5 pytest attempt without the controller was
  intentionally not accepted as proof because its PostgreSQL URL was absent or
  pointed at a local database with mismatched credentials.
- Deliverable C.5 is complete, but rollout remains `disabled`. No rollout
  authorization, deploy, push, merge, or PR occurred. The next activity is
  **Deliverable D planning**, not implementation; Deliverable E follows D and
  Slack reconstruction remains last.

## 2026-08-30 C.5 paid gate attempt and provider-schema repair

- The user separately authorized the Terra validation and Mini five-route
  extraction gates and chose to reuse the existing `.env.local` key. The key
  was loaded only into each child process and was never printed, persisted, or
  committed.
- Safe model retrieval confirmed access to exact models `gpt-5.6-terra` and
  `gpt-5.4-mini-2026-03-17`. The first live attempts then failed before model
  execution with OpenAI `invalid_json_schema` on `text.format.schema`.
- Root-cause diagnostics proved that Pydantic `Decimal` emitted a
  number-or-regex-string union rejected by Responses strict JSON schema. The
  extraction schemas additionally emitted a discriminated `oneOf`. TDD fixes
  now freeze Terra scores as bounded JSON numbers and build Mini schemas from
  OpenAI SDK `pydantic_function_tool()`, normalizing only Decimal unions and
  discriminated `oneOf` to provider-supported JSON-schema shapes. Domain
  parsing still uses the original Pydantic models and Decimal validation.
- RED/GREEN evidence: the Terra provider-schema regression failed before the
  change and passed afterward; the Mini live CLI fake gate failed `3 != 0`
  before the provider-schema adapter and passed afterward. The combined focused
  regression is `96 passed`; touched-file Ruff is green.
- Post-fix official aggregate-only Terra and Mini CLIs were rerun. Safe bounded
  diagnostics confirmed both schemas now pass provider validation and reach
  execution, where the account returns
  `credit_balance_exhausted` / `insufficient_quota` (HTTP 429). Therefore
  neither live gate is passed and their plan checkboxes remain unchecked.
- Operational mode remains `disabled`. Do not enable rollout or record the
  deterministic fixture metrics as live-model results. After billing/credits
  are restored, obtain fresh explicit paid-call confirmation and rerun both
  exact aggregate-only commands. Deliverable D remains planning-only after the
  C.5 paid release boundary; Slack remains last.

## 2026-08-30 C.5 Task 16 deterministic release proof complete

- Branch `codex/rag-orchestrator-agent` now contains Task 16 behavior commit
  `4b9132a` after Tasks 1–15. Alembic head remains
  `9d7f3a1c6e20`; Task 16 adds no production schema or trust mutation.
- Aggregate-only evaluation CLIs are code-complete. Terra live mode requires
  both `--allow-paid-provider-call` and
  `PARAWORKS_ALLOW_PAID_TERRA_EVAL=1`; Mini requires its own CLI flag plus
  `PARAWORKS_ALLOW_PAID_EXTRACTION_EVAL=1`. Both also require
  `OPENAI_API_KEY`. Automated tests use injected LangChain fake models and
  construct no network client without dual authorization.
- Deterministic golden metrics are precision `1.0`, recall `1.0`, queue
  reduction `0.166667`, and every prohibited count `0`. Do not represent these
  as live Terra/Mini results. The later paid attempts are recorded separately
  above and did not produce a passing live-model report.
- Official controller proof is green:
  - settings: `6 collected / 6 passed`, leases `2/2`;
  - PostgreSQL: `394 / 394`, leases `10/10`;
  - compatibility: `1,595 collected / 1,591 passed / 4 exact Slack
    deselected`, leases `91/91`;
  - non-Slack: `2,036 / 2,026 passed / 10 exact Slack deselected`, leases
    `150/150`;
  - full: `2,036 / 2,026 passed / exactly 10 Slack failures`, leases `150/150`.
  Every profile has errors/skips/xfails `0`, unexpected nodes `0`,
  `release_proof=true`, `no_live_provider=true`, and database/role residue
  `0/0`.
- Frontend lint/build are green. Managed Playwright is desktop `58 passed`,
  mobile `48 passed`, and legacy desktop `9 passed`; port 3000 is closed.
  Lock, whole-tree Ruff, diff, bounded symbol scan, and secret hygiene are green.
- Operational mode remains `disabled`. No rollout authorization, breaker
  transition, production database mutation, deploy, push, merge, or PR occurred.
  The distinct paid Terra and Mini gates subsequently passed as recorded in the
  newer entry above.
- The next product activity is **Deliverable D planning**, not implementation.
  Deliverable E follows D; Slack reconstruction and the visible ten-node
  baseline remain last.

## 2026-08-30 C.5 Task 15 same-screen trust UX complete

- Review retains the pending default and exact workflow filter while adding an
  inline automatic-approved view. Automatic details omit raw ids and human
  bulk/project/edit/approve actions, retain the evidence drawer, and expose only
  bounded validation/audit state plus strict inline audit/revoke controls.
- V2.1 workflow context shows automatic/human-required/more-evidence counts;
  V2.0 keeps the old completed/total display. No modal, wizard, route, or normal
  navigation depth was added.
- Permission-filtered Knowledge and project Timeline projections now include
  nullable trust source only. Human provenance takes precedence over automatic
  provenance, and UI badges remain separate from operational status.
- Fresh evidence: backend focused `30 passed`; frontend lint/build; desktop
  Playwright `31 passed`; mobile `26 passed`. No paid provider or rollout.
- Next is C.5 Task 16, an actual implementation/release-verification task. Its
  deterministic harness and isolation profiles may run automatically, but the
  paid Terra and Mini gates still require separate explicit authorization.

## 2026-08-30 C.5 Task 14 typed one-click preview complete

- Frontend transport now uses exact V2.0/V2.1 discriminated unions and exact
  four-key versus five-key review status counts. V2.0 never accepts or emits a
  launch token; V2.1 requires the server-issued signed preview token.
- Integrations preserves the existing `검토 후보 만들기` button as the only
  normal-path confirmation and displays combined extraction, validation, and
  total maximum costs/tokens inline. Cost drift clears the token, refreshes the
  preview, and waits for another explicit click without automatic relaunch.
- Fresh evidence: lint/build pass, desktop focused Playwright `27 passed`,
  mobile `22 passed`, and the development server/port 3000 was stopped.
- Next is C.5 Task 15, an actual implementation task: same-screen automatic
  trust review plus human/automatic trust-source badges. Slack remains last.

## 2026-08-30 C.5 Task 13 bounded Review API complete

- Added strict `revoke-auto-approval` and `auto-review-audit` routes through
  existing human actor, audit-gate, breaker-first quality, quarantine, and
  exact-revoke services. Request bodies cannot forge actor/policy/target or add
  revoke free text; conflicts expose only allowlisted codes.
- Review list defaults to pending, supports permission-concealed
  `status=approved&resolution_source=auto_policy`, and accepts only exact
  registered V2.0/V2.1 workflow versions. Totals/groups are computed after
  evidence visibility.
- Auto summary/audit projection is allowlist-only and omits ids, raw output,
  reasons, evidence aliases, provenance/document identities, and internal
  failure/collision codes. Corrupt legacy metadata fails closed.
- Fresh evidence: focused API/RBAC/V2 suite `69 passed`; adjacent audit,
  quality-revoke, provenance, and Review suite `184 passed, 58 skipped`; Ruff
  and diff checks pass. No paid/live provider, rollout, push, merge, or PR.
- Next is C.5 Task 14, an actual implementation task: typed V2.1 client and the
  existing one-click combined cost preview. Slack remains last.

## 2026-08-30 C.5 Task 12 immutable V2.1 lifecycle complete

- Added an independent eight-key/five-status V2.1 checkpoint contract and an
  actual LangGraph 1.x topology with authoritative extraction, auto-review,
  database count refresh, pending-first interrupt routing, and distinct
  no-candidate, needs-more-evidence, and all-resolved terminal nodes. V2.0 was
  not mutated.
- Registered V2.0 and `company-memory-review-v2.1-auto-review` under exact
  immutable registry keys. New-run selection uses configured mode; status,
  resume, and cancel use the stored graph version through the facade.
- Added the dedicated V2.1 lifecycle/status mapper, five-count API union,
  checkpoint/live-row reconciliation, verified human transition auditing, and
  verified auto-policy revoke-only terminal drift. Checkpoints and responses
  contain counts only, never item ids, source content, prompts, or model output.
- Added aggregate-only extraction/validation call recovery and lifespan order:
  source reconciliation -> provider-call recovery -> audit remediation. It
  performs no provider retry and charges attempt-zero as zero and attempt-one
  conservatively from its reservation.
- Fresh deterministic Task 12/V2.0 regression evidence is `157 passed`; the
  isolated PostgreSQL V2.0 checkpoint suite is `9 passed` with six Alembic
  deprecation warnings. Ruff and `git diff --check` pass. No live provider
  call, rollout enablement, push, merge, or PR occurred.
- Next is C.5 Task 13, an actual implementation task: bounded review metadata,
  revoke, and audit API actions. Slack remains deferred to the end.

## 2026-08-30 C.5 Task 6 complete and next boundary

- Task 6 is complete on `codex/rag-orchestrator-agent`; its independently
  approved implementation head is `a74cfeb`. Final review verdict: Spec PASS,
  Quality APPROVED, zero open Critical or Important findings.
- Preserve exact server-owned content signature, parser/chunk policy, canonical
  source id, and relational current-version authority. Reconciliation is
  bounded/paginated; trusted serving uses one canonical all-type text builder;
  legacy `decision` dependencies retain their stored key while trust logic uses
  `decision_record`; vector state hashes advance on permission-only updates only
  with proof of the exact prior canonical document.
- Final isolated PostgreSQL evidence: raw Task 6 comparison `432 passed, 4
  failed` (exactly the four approved deferred Slack nodes), exact non-Slack gate
  `432 passed, 4 deselected`, standalone pgvector `28 passed`, Review V2.1
  PostgreSQL `48 passed`, and Review V2 PostgreSQL `9 passed`. Ruff, lock, and
  diff checks pass. The disposable DB and role are deleted; catalog counts are
  `0/0`.
- No live connector, LLM, embedding provider, rollout, release, deploy, push,
  merge, or PR action occurred. The next product work is C.5 Task 7's actual
  TDD implementation slice under the already approved C.5 plan. Do not start
  Task 8 or broaden shared permission/trust contracts without its normal green
  checkpoint and review.
- Slack connector/agent/OAuth/data reconstruction remains last after D and E.
  Preserve the exact approved ten-node Slack deselection list and authorize no
  additional deselections.

## 2026-08-29 Whole-suite PostgreSQL isolation planning checkpoint

- The approved-in-chat design is recorded at
  `docs/superpowers/specs/2026-08-29-whole-suite-postgresql-isolation-design.md`;
  it passed independent Spec/Quality review and is now user-approved. The
  detailed RED/GREEN plan is
  `docs/superpowers/plans/2026-08-29-whole-suite-postgresql-isolation.md`; it
  awaits user review and is not implementation authorization.
- Preserved non-Slack evidence is `1453 passed`, `1 skipped`, ten approved Slack
  deselections, `12 failed`, `144 errors`, with owned database/role cleanup
  `0:0:0:0` and the shared container left running healthy.
- Root causes are separated: shared `public` schema ordering caused the 144
  errors; controller-wide database/fingerprint environment caused six Settings
  failures. A legacy Review V2 fake-draft provenance gap is the leading static
  hypothesis for the remaining six, but a clean long-traceback RED must confirm
  it before test fixture changes.
- Approved architecture: one controller-owned `_test` database/role, unique
  serial schema leases for PostgreSQL tests, a separate temporary SQLite app
  database, allowlisted/hermetic child environments with an early test-only
  dotenv guard, an exact node/lease sidecar, bounded process/schema cleanup, and
  no additional Slack exclusion or production trigger relaxation.
- At this 2026-08-29 checkpoint, this boundary still required separate plan
  approval and did not reorder the then-remaining C.5 Tasks 6–15. Slack
  reconstruction/regression remains last after D and E.
- The C.5 Task 16 plan now points Steps 4–7 to the `postgres`,
  `compatibility`, `non-slack`, and `full` controller profiles and replaces the
  single-final-commit instruction with reviewed behavior-slice commits plus a
  clean documentation/evidence commit. Those amended instructions remain
  blocked until the new isolation plan is approved and implemented at Task 16
  entry.

## 2026-08-29 C.5 product Task 5 database boundary verified

### Current boundary and next work

- Deliverable C.5 product Tasks 1–5 are implemented and independently
  verified. Final Task 5 implementation HEAD is
  `4d31aafd37dc2526ab08d458405113c38a19d9ba` on
  `codex/review-hitl-v2-design`.
- The next product slice is C.5 Task 6, "Make Auto Approval Precisely Revocable
  and Non-Resurrectable." It has not started. After C.5 Tasks 6–16, retain the
  approved order: Deliverable D Retriever/RAG Answer Graph V2, Deliverable E
  Neo4j GraphRAG, then Slack reconstruction/regressions last.
- Do not infer authorization for Task 6, a paid Terra benchmark, rollout
  enablement, frontend C.5 work, release/deploy, push, merge, or PR creation
  from this verification record.

### Public runtime and compatibility contracts

- `backend.app.db.initialization` is a Settings-free leaf boundary exposing
  `DatabaseRuntime`, `initialize_database_runtime(database_url)`,
  `DatabaseConfigurationError(code='database_configuration_invalid')`, and
  `DatabaseInitializationError(code='database_initialization_failed')`.
- It preserves `create_engine(database_url, pool_pre_ping=True)` and
  `sessionmaker(bind=engine, autoflush=False, autocommit=False,
  expire_on_commit=True)`. Construction is connection-lazy: it performs no
  connect, checkout/ping, inspection, or SQL.
- Availability classification is first-match: DBAPI import/load errors;
  `DBAPIError(connection_invalidated=True)`; then operational/interface/pool
  timeout/disconnection errors. Configuration and initialization wrappers carry
  no copied message, URL, driver/module name, cause, or initializer-captured
  context.
- `DatabaseRuntime.dispose()` latches before the engine disposal attempt, so it
  is idempotent even when cleanup fails. A sessionmaker-construction failure
  disposes a partially built engine once; cleanup failure overrides the earlier
  construction failure under the same availability rules.
- `backend.app.db.session` owns one private process-global runtime and preserves
  public `engine`, `SessionLocal`, and `get_db`; `SessionLocal.kw['bind'] is
  engine`, resolved URL precedence, and request-session close semantics remain
  unchanged. The global application runtime remains process-lifetime owned.
- key-admin does not import `SessionLocal` to create storage. It resolves the
  URL, creates its own runtime, runs the command, attempts disposal exactly
  once, and only then emits one bounded stdout JSON line. Configuration is exit
  2; proven storage unavailability and operation failure are exit 3; stderr is
  empty. Cleanup failure overrides an earlier command result without a second
  output.

### Exact implementation commits

- `cedd546f7aaf7fdace40a25d785f0d09df5df108` — add the typed runtime and
  nominal factory tests.
- `cc5faa5bc1cc2959ba047a9eefb9724b2f742a9e` — add availability precedence,
  privacy, partial cleanup, and idempotent disposal.
- `4177f8019401c728786a88c69b5f17571137ad18` — move key-admin to its owned
  runtime and bounded single-emission lifecycle.
- `149ea2323326683767ca3d0d6c9087135277140b` — route the application session
  compatibility adapter through the shared initializer and prove imports/URL
  precedence.
- `4018ddd1917cd441f0d5eb64a3eddacaf3b348bc` — seed the fresh-readiness test
  through the existing human Review transition and real projection path.
- `4d31aafd37dc2526ab08d458405113c38a19d9ba` — commit the seed transaction
  before the independently owned CLI runtime performs its read.

### Fresh Task 5 PostgreSQL evidence

The fail-closed controller reused only the validated shared
`paraworks-postgres` service (`pgvector/pgvector:pg17`, compose service
`postgres`, `127.0.0.1:55432`, server identity `paraworks:postgres`) and created
only these controller-owned identities:

```text
database=paraworks_c5t5_dbinit_20260829_database_test
role=paraworks_c5t5_dbinit_20260829_role_test
run_prefix=paraworks_c5t5_dbinit_20260829
```

The exact ordered test commands and observed results were:

```powershell
uv run --no-cache --locked pytest backend/tests/test_auto_review_migration.py backend/tests/test_auto_review_key_bootstrap.py backend/tests/test_keyed_mutation_guard.py -q
# PASS: 115 passed, 0 skipped

uv run --no-cache --locked pytest backend/tests/test_database_initialization.py -q
# PASS: 58 passed, 0 skipped

uv run --no-cache --locked pytest backend/tests/test_auto_review_provenance.py -q -k "key_admin_module_cli or key_admin_status_exit_code or task5_modules_import or database_import"
# PASS: 23 passed, 137 deselected, 0 skipped

uv run --no-cache --locked pytest backend/tests/test_auto_review_provenance.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_review_transitions.py backend/tests/test_review_transition_postgres.py -q
# PASS: 204 passed, 0 skipped

uv run --no-cache --locked pytest backend/tests/test_review_resolution_actors.py backend/tests/test_auth_api.py backend/tests/test_review_rbac.py backend/tests/test_audit_logs.py -q
# PASS: 70 passed, 0 skipped

uv run --no-cache --locked pytest backend/tests/test_db_init.py backend/tests/test_health.py backend/tests/test_agent_runtime_lifespan.py backend/tests/test_agent_runtime_bootstrap.py backend/tests/test_agent_runtime_retention.py backend/tests/test_rag_indexing_tasks.py backend/tests/test_review_v2_api.py -q
# PASS: 81 passed, 0 skipped
```

Total observed test evidence is `551 passed, 0 skipped` across six invocations.
The exact static gates also passed:

```powershell
uv run --no-cache --locked ruff check backend/app/db/initialization.py backend/app/db/session.py backend/app/admin/auto_review_keys.py backend/tests/test_database_initialization.py backend/tests/test_auto_review_provenance.py
uv run --no-cache --locked python -m compileall -q backend/app/db/initialization.py backend/app/db/session.py backend/app/admin/auto_review_keys.py
uv lock --check
git diff --check
```

The controller terminated exact test-database sessions, dropped only its owned
database and role, and restored controller-managed process environment values.
Its terminal result was
`C.5 Task 5 verification and cleanup PASS (0:0:0:0)`. A separate read-only
catalog query confirmed exact database, exact role, run-prefix database, and
run-prefix role counts `0:0:0:0`. The shared container remains running by
policy; no volume or pre-existing identity was changed.

No live LLM, embedding, connector, OAuth, Slack, Gmail, Drive, Calendar, or
other product-provider API was called. The accepted evidence does not include
frontend, C.5 Task 6, Deliverable D/E, Slack recovery, CDC/streaming, rollout,
release, deploy, push, merge, or PR work.

## 2026-08-28 Deliverable C.5 plan and execution profile finalized

- The user-approved design remains
  `docs/superpowers/specs/2026-08-28-auto-review-trust-promotion-design.md`.
  Its implementation plan is
  `docs/superpowers/plans/2026-08-28-auto-review-trust-promotion.md` on branch
  `codex/review-hitl-v2-design`.
- The plan has sixteen ordered RED/GREEN/commit checkpoints: V2.0 freeze and
  V2.1 contracts, persistence, immutable candidate evidence, resolution actor,
  exact claim/provenance/reaffirmation, revoke/tombstones, eligibility/policy,
  real LangChain Terra validation, lease/cache orchestration, rollout/audit,
  signed preview, separate V2.1 LangGraph/lifecycle, bounded APIs, Integrations
  UX, Review/knowledge badges, and release verification.
- A read-only code map confirmed that existing `state.py`,
  `review_v2_graph.py`, V2.0 Pydantic outputs, and paused tuples must not absorb
  V2.1 fields. The plan uses a separate V2.1 state/graph/service plus a thin
  stored-version facade.
- Persistence is normalized as one `TrustedKnowledgeApprovalLink` per effect
  with many child `TrustedKnowledgeEvidenceLink` rows; this resolves the
  approved one-effect/many-evidence requirement. A named composite FK enforces
  that `ReviewItem.auto_validation_id` belongs to the same ReviewItem.
- The plan freezes a bounded `auto_review_audit` list projection because the
  approved `감사 필요`/`조치 필요` UX cannot survive reload without it. It
  exposes only status, nullable outcome, and action-required—not ids, cohort
  counters, hidden provenance, or raw reason. The normalized one-effect/
  many-evidence relation and this projection are approved/frozen.
- The immutable registry, not deployment configuration, owns exact price and
  model identities. Deployment values may only confirm equality; missing or
  mismatched key/price readiness fails closed before shadow/enforce.
- Independent plan, dependency, and security audits were applied before code:
  new V2.0/V2.1 candidates both bind exact evidence; provider calls use an
  ephemeral DTO outside DB transactions; owner permission is re-resolved before
  and after validation and on resume; exact generator/provider identity is
  persisted; hidden collision uses a complete keyed fingerprint projection.
- The extraction registry contains exactly `mail_document_agent`,
  `timeline_agent`, `history_agent`, `decision_record_agent`, and `todo_agent`.
  Each route uses OpenAI `gpt-5.4-mini-2026-03-17`, reasoning `none`, returns
  zero or one candidate, and reserves USD 0.016716 at full cap; five routes
  reserve USD 0.083580.
- Validation uses OpenAI `gpt-5.6-terra`, reasoning `medium`, four candidates
  per batch and at most two batches/five candidates per workflow. One full
  batch reserves USD 0.048864 and two reserve USD 0.097728. Combined extraction
  plus validation is USD 0.181308 under the immutable USD 0.20 limit.
- Paid calls use authoritative no-retry ledgers with exact child allocation. Revoke and
  reindex use the same document advisory lock/session plus tombstone-filtered
  serving; in-memory deletes occur only after DB commit.
- Persisted Assistant evidence dependencies must be complete and current or all
  answer-derived serving fails closed. Only the server content signature,
  parser policy/run, and relational `current_document_version_id` authorize C.5;
  connector signatures/display labels do not. Provider and rollout control
  history is append-only with aggregate backpointers. Quality revoke uses an
  immutable assessment and audit-or-correction, commits breaker/quarantine
  before physical revoke, and a corrected confirmed audit permanently requires
  a new reviewed policy version.
- Rollout has a persistent operator latch `0 -> 10 -> 100`; 100 cannot be
  skipped to, 2% audit applies only to new full-enforce workflows, a critical
  audit commits breaker plus durable revoke remediation first, and breaker
  close cannot silently re-enable enforce.
- Populated C.5 schema downgrade/local row reset refuses destructive audit
  deletion. The operational rollback is config `disabled`. A paid sanitized
  Terra benchmark still requires explicit authorization before shadow rollout.
- This work is still planning. No product code, migration, provider call,
  feature enablement, push, merge, or PR has occurred. The spec/plan/profile are
  finalized. The next unapproved action is selection of subagent-driven
  (recommended) or inline execution plus explicit product-code authorization;
  that next action begins actual implementation.

## 2026-08-28 Deliverable C.5 design approved

- The approved design is
  `docs/superpowers/specs/2026-08-28-auto-review-trust-promotion-design.md`
  on branch `codex/review-hitl-v2-design`. It is planning/specification only;
  no product code, migration, live model call, rollout enablement, push, merge,
  or PR was performed.
- C.5 is inserted between completed Deliverable C and Deliverable D. The
  remaining order is C.5 Auto-Review Trust Promotion -> D Retriever Port and
  RAG Answer Graph V2 -> E Neo4j GraphRAG -> Slack recovery last. CDC/streaming
  remains deferred until measured need.
- The trust boundary is three-tiered: canonical source evidence, pending AI
  knowledge, and trusted knowledge. Every AI candidate starts pending. Initial
  auto approval is limited to public/internal direct-fact Timeline and narrowly
  extractive History; Decision, Todo, restricted, inferred, conflicting, and
  uncertain items remain human-reviewed.
- The approved/frozen validator is OpenAI `gpt-5.6-terra` with medium reasoning and
  strict LangChain structured output. It has no approval authority. A pure,
  versioned deterministic policy owns the final result, and an exact same
  provider/model as candidate generation cannot validate that candidate.
- Existing durable `company-memory-review-v2.0` graph/state/topology remains
  immutable for every existing and paused thread. C.5 defines
  `company-memory-review-v2.1-auto-review` for new shadow/enforce launches only;
  Registry keeps both versions, and effective runtime mode may be demoted but
  never promoted beyond the mode stored at launch.
- The design adds candidate-to-canonical evidence refs, per-field evidence-slot
  validation, validation leases/cache/cost, an internal-only resolution actor,
  and one locked transition service for both new promotion and exact duplicate
  reaffirmation. No model can call a public auto-approval action.
- Exact post-migration approval provenance prevents cross-item revoke. Revoking
  one reaffirmation removes only its link; shared knowledge remains trusted
  while another active human/auto or legacy human provenance exists. The last
  provenance adds a vector tombstone, revokes the exact knowledge/companion
  Timeline, and deletes exact pgvector/index-state documents transactionally.
- C.5 explicitly prevents auto approval from expanding raw chunk eligibility:
  source chunks remain indexable only from human or legacy-human approvals.
  Auto-policy approvals contribute promoted trusted knowledge documents only;
  the canonical-raw-evidence retrieval lane remains Deliverable D work.
- Shadow is a paid Terra path but can run only after the existing Integrations
  dry-run displays extraction + validation + total maximum cost and the user
  presses the same explicit launch button. A signed preview token binds the
  input, graph, mode, policy/model, cost ceilings, and expiry. Sync/status/page
  polling never calls the provider.
- Approved values include policy/prompt/output-contract versions, `0.9800`
  per-field threshold, five candidates/workflow, four candidates/batch, two
  batches, 12 evidence slots, 6,000 validation input/3,072 total output tokens,
  exact USD 0.083580 extraction + USD 0.097728 validation = USD 0.181308
  profile reserve under the USD 0.20 limit, new schema/status/API fields, 500
  shadow comparisons, 99% precision, 10% canary, and post-audit sampling.
- A scope-wide existence-only collision guard routes hidden restricted
  collisions to human review without exposing content, identity, permission, or
  count. Exact duplicate reuse remains limited to currently visible trusted
  knowledge.
- Sampled auto approvals create durable `AutoReviewPostAudit` rows, while
  `AutoReviewRolloutState` makes first-50/10%/2% audits and the critical
  enforce-to-shadow breaker enforceable rather than a manual checklist. A
  critical outcome opens/commits the breaker before exact revoke is attempted.
- The design/spec and implementation-plan gate is complete. Product-code
  implementation remains unstarted and requires explicit authorization plus an
  execution-mode choice.

## 2026-08-27 Review Queue HITL V2 release verified

- Deliverable C is implemented through the approved eleven-task plan. Its
  release commit uses subject `test: verify review workflow v2 release`; the
  exact SHA is recorded in the Task 11 implementer report and final handoff.
- User flow remains only
  `Integrations -> 검토 후보 만들기 -> Review -> 검토 완료`. V2 still defaults
  off (`langgraph_review_v2_enabled: bool = False`); rollback is disabling the
  flag and keeping existing V2 threads on V2 rather than falling back to V1.
- The PostgreSQL release target must be disposable, include pgvector, and have
  both database and user names ending in `_test`. The fresh sanitized target
  was PostgreSQL 16 with database/user `paraworks_review_test`; no DSN, host,
  port, or credential was committed.
- Exact PostgreSQL recovery/race result: `14 passed, 0 skipped, 10 warnings`.
  It covers independent pool/app restart, root namespace, checkpoint privacy,
  status/checkpoint reconciliation, exact shared-batch ownership, concurrent
  approval/resume/terminal races, canonical promotion ids and Timeline
  provenance, and exact generated-row cleanup.
- A PostgreSQL RED exposed the unused always-empty `review_item_ids`
  checkpoint key. The minimal production privacy fix removes that key from
  checkpoint schema/initialization/probes only; ReviewItem ids remain in the
  application database and API behavior/graph topology is unchanged.
- Final review found and fixed a separate approval/resume defect: comparing the
  immutable paused tuple's original status distribution to mutable live
  ReviewItem rows made a normal approval appear corrupt. Status now remains
  `awaiting_human_review` with `checkpoint_resumable=true`,
  `review_resolution_ready=true`, and `resume_allowed=true`; explicit
  completion preserves `checkpoint_thread_id` and completes through the direct
  `Command(resume=...)` path with exactly two thread state-version transitions.
  It must not enter `checkpoint_failed`, rotate, or repair on ordinary approval.
- Saved tuple validation still fails closed on identity mismatch, missing/extra
  status keys, wrong total review count, invalid values, error codes, or bad
  graph/task shape. Live DB permission and current-version checks are unchanged.
- The restart fixture now gives app A and app B independent application
  SQLAlchemy engines, pools, and sessionmakers plus independent checkpoint
  runtimes/pools/savers. App A is fully closed/disposed before app B is built.
- Fresh gates: touched-module `199 passed`; Deliverable C `333 passed` plus its
  one known deferred Slack orchestration failure; Deliverable B `229 passed`;
  non-Slack `989 passed, 1 skipped, 10 deselected`; full backend `989 passed,
  1 skipped` plus exactly the ten user-deferred Slack failures. The optional
  skip is unrelated to either Task 11 PostgreSQL file.
- Whole-tree Ruff moved from 32 findings to `All checks passed!` through safe
  lint-only cleanup; Slack code received no behavior/data-source change. Lock
  resolved 99 packages and `git diff --check` is green.
- Frontend lint/build passed (18 generated pages); Playwright passed desktop
  V2 `48`, mobile V2 `44`, adjacent UX `9`; deterministic backend smoke passed
  `9`. The frontend server was stopped and port 3000 was verified closed.
- The ten Slack failures remain visible and must stay deferred until the final
  Slack recovery phase, per the user's missing-data-source direction. Do not
  add deselections beyond the approved non-Slack comparison.
- This historical next boundary was superseded on 2026-08-28 by the separate
  Deliverable C.5 design before Deliverable D. C.5 is not implemented yet.

## 2026-08-27 Review Queue HITL V2 implementation plan ready

- The implementation plan is
  `docs/superpowers/plans/2026-08-27-review-queue-hitl-v2.md`; its source design
  remains `docs/superpowers/specs/2026-08-27-review-queue-hitl-v2-design.md`.
- The plan has eleven ordered TDD/commit checkpoints: public/V1 contracts,
  canonical preflight, connector waterline, locked Review promotion, Registry
  and LangChain drafting, real LangGraph interrupt, lifecycle/API/filtering,
  typed frontend client, Integrations launch UX, Review completion UX, and
  PostgreSQL/release gates.
- Approving the plan will freeze its exact diagnostic, dry-run, status, error-code,
  source-reference, and Review replay/promotion shapes. A discovered need for a
  new public/schema field, permission change, token-budget change, Review trust
  change, or duplicate-resolution rule requires another human gate.
- No new Alembic revision, batch/outbox table, CDC/streaming component, Slack
  change, RAG/Neo4j/Knowledge Map change, or Agent Runs navigation is planned.
  Existing `AgentWorkflow*` rows, HMACs, PostgreSQL advisory locks, provenance
  indexes, and `Source.raw_metadata` are the approved persistence path.
- Configured production extraction must use the existing real LangChain
  adapters. The V2 workflow must compile real LangGraph `StateGraph`, pause with
  `interrupt()`, and resume the exact checkpoint mode/thread with
  `Command(resume=...)`; metadata emulation is not acceptable.
- V1 run output as well as status metadata must be made truthful
  (`metadata_only`, non-resumable). Review approval replay for a pre-V2 approved
  row without provenance returns no new effect instead of heuristic backfill.
- Product implementation has not started and is not yet authorized. The next
  user decision is execution mode: subagent-driven development (recommended) or
  inline plan execution. Do not push, merge, enable V2, or modify product code
  before that choice.

## 2026-08-27 Review Queue HITL V2 design approved

- The approved Deliverable C spec is
  `docs/superpowers/specs/2026-08-27-review-queue-hitl-v2-design.md`.
- The UX is fixed at
  `Integrations -> 검토 후보 만들기 -> Review -> 검토 완료`. Do not add a
  primary Agent Runs workflow page or automatic resume.
- The graph is review-only and must use actual `interrupt()` plus the same
  checkpoint thread's `Command(resume=...)`. It must not run RAG before or
  after the review boundary in Deliverable C.
- Resume payload is acknowledgement only. Current PostgreSQL ReviewItem state
  and current actor permission determine resolution.
- All Review action entry points must share the locked state-transition and
  exactly-once promotion service. Every promoted record, including companion
  Timeline rows, must use `source_review_item_id` provenance.
- New V2 routes remain disabled by default. Legacy routes remain for migration
  and rollback but must report `review_boundary=metadata_only`,
  `hitl_checkpointing=false`, and `checkpoint_store=none`.
- In V2 mode, Gmail/Drive/Calendar sync stores canonical sources and returns
  refs but must not run the legacy inline candidate bridge. Async ref recovery
  uses `Source.raw_metadata.last_changed_sync_job_id`; the evidence HMAC is the
  server-enforced V1/V2 batch-ownership key. No new batch/outbox table is
  introduced, and Slack is excluded from this switch.
- The user chose company/workspace scope ownership for an exact source-version
  batch. The batch HMAC excludes `owner_subject_id`; authorized users reuse one
  thread, unauthorized users receive 404, and different security scopes remain
  isolated. Cancelled/failed attempts also remain stable owners; there is no
  replacement exception that could bypass the existing client-request unique
  constraint. They are terminal and expose no user retry action; only
  `checkpoint_failed` uses same-thread repair/retry.
- `client_request_id` is an optional transport-retry key bound only when the
  requesting actor actually creates the thread. Cross-owner shared reuse does
  not persist caller aliases or reserve the key across actors; changing that
  creator-only meaning requires separate persistence and a human gate.
- V2 may create a new thread only for source versions marked by a V2-mode sync.
  Legacy-processed or pre-waterline source versions are rejected with the
  existing `evidence_changed` category, and rollback V1 processing changes the
  marker back to `legacy_inline`.
- Public `agent_names` are exact Registry manifest names only:
  `mail_document_agent`, `timeline_agent`, `history_agent`,
  `decision_record_agent`, and `todo_agent`.
- Slack, RAG cutover, Neo4j, Knowledge Map changes, CDC, outbox, broker, and
  streaming consumers are outside Deliverable C. CDC is reconsidered only
  after measured backlog, freshness, fan-out, or polling/worker bottlenecks.
- No product code is authorized yet. The next worker must use the writing-plans
  workflow to create and obtain review of the separate Deliverable C
  implementation plan before starting TDD implementation.

## 2026-08-26 Runtime Deliverable B complete

- Deliverable B runtime/checkpoint primitives are complete through
  implementation rollback point `4a1867a` (including bounded cyclic/deep
  interrupt normalization in `4a1867a`, exact interrupt and resumability
  error hardening in `5b05bc8`, the narrow Alembic
  logging correction in `e847608`, the original restart test in `403600b`,
  and the no-fix Ruff import correction in `e44dc4f`).
- The locked Review workflow version is `company-memory-review-v2.0`.
  Application graph version selection is separate from LangGraph's root
  `checkpoint_ns == ''`.
- Checkpoint modes are `disabled`, process-local SQLite/demo `memory`, and
  durable production PostgreSQL `postgres`. PostgreSQL unavailability fails
  closed and never falls back to memory.
- Application migration head is `2f6a8b9c0d1e`. LangGraph checkpointer schema
  setup remains an explicit operator bootstrap with backup confirmation; app
  startup performs readiness checks only and never calls `setup()`.
- The dedicated target was an isolated disposable local PostgreSQL database on
  port 55432. With `PARAWORKS_TEST_POSTGRES_URL` already set to that target,
  the exact command was:

  ```powershell
  uv run --locked pytest backend/tests/test_agent_runtime_postgres_checkpoint.py -q
  ```

  Current-HEAD result: `30 passed, 1 warning in 0.75s`, with no
  skip. It proved a real interrupt, pool A shutdown, independent pool/saver B
  restart, same-thread resume, checkpoint-id progression, root namespace,
  fail-before-mutation database/user identity checks, every stored/decoded
  payload shape excluding relationship paths, LLM prompts, and raw connector
  payloads, and exact generated-thread cleanup even when cleanup stages fail.
  The disposable container's host forwarding required one safe restart before
  this final result; the database identity and target did not change.
- Confirmation compares the exact ordered returned/persisted interrupt ids and
  recursively validated JSON-safe values before separately enforcing the
  expected pause state. Resume saver read failures are bounded to
  `checkpoint_unavailable` without provider, DSN, or marker leakage. Cyclic or
  excessively deep values are bounded to `checkpoint interrupt state mismatch`
  rather than leaking a raw `RecursionError`.
- Complete focused result: `224 passed, 9 warnings`. Non-Slack gate:
  `717 passed, 1 skipped, 10 deselected, 9 warnings`; the one skip is the
  existing optional pgvector integration test. Full backend result:
  `10 failed, 717 passed, 1 skipped, 9 warnings`; the failures are exactly the
  ten user-deferred Slack ids and symptoms recorded in the Deliverable B plan.
- The exact lock check resolved 99 packages. The exact planned Ruff gate and
  an independent `ruff check --no-fix` run both report `All checks passed!`
  from a clean worktree, and `git diff --check` is clean.
- No public V2 route, Review Queue state transition, RAG cutover, Slack change,
  live connector/model/provider call, or Deliverable C implementation was
  made.
- Deliverable C still has no implementation authorization. The next worker
  must prepare and obtain review of its separate Review Queue HITL V2 plan
  rather than coding it directly.

## 2026-08-26 Runtime Deliverable B plan and Slack deferral

- The user directed all Slack-related remediation to the final phase because
  the former live Slack data source is no longer available. Do not skip,
  delete, weaken, or mark the affected tests `xfail`; keep them visible until
  a separate Slack recovery design is approved.
- The accepted dependency lock has 11 pre-existing backend failures. Ten are
  Slack-related and deferred. The only non-Slack failure is
  `test_mail_document_agent_preflight_reports_cost_without_running_llm`.
- Root cause is confirmed: the test seeds `restricted` Drive evidence but uses
  the `PermissionContext` default `('public', 'internal')`. Permission hardening
  commit `b6c5d04` updated the production filter and similar test callers but
  missed this preflight fixture. A diagnostic run returned `default_count=0`
  and `explicit_count=1`. Production API/company-memory paths already pass the
  authenticated user's exact permission levels, so do not weaken the runtime
  filter or infer access from the `admin` role string.
- The implementation-ready Deliverable B plan is
  `docs/superpowers/plans/2026-08-26-langgraph-runtime-checkpoint-primitives.md`.
  It begins with the bounded test-only permission fixture repair, then adds
  source-agnostic state/fingerprint, workflow schema, checkpointer lifecycle,
  explicit bootstrap, graph version, and saver-confirmation primitives.
- Deliverable B must not add a public V2 route, Review Queue state transition,
  RAG cutover, Neo4j component, or Slack change. Gmail, Drive, Calendar,
  approved knowledge, deterministic fixtures, and fake models remain the
  verification path.
- Next work classification: actual implementation. Obtain explicit approval
  before executing Task 1 or changing production/test code.

## 2026-08-26 LangChain·LangGraph dependency compatibility

- Deliverable A is complete with LangChain 1.3.17, LangGraph 1.2.11,
  langchain-openai 1.6.0, langchain-google-genai 4.3.5, and
  langgraph-checkpoint-postgres 3.1.2 locked.
- The complete backend comparison is old lock `506 passed, 11 failed,
  1 skipped` versus accepted lock `518 passed, 11 failed, 1 skipped`; all 11
  failures are identical pre-existing failures, so Deliverable A introduced
  zero new backend failures. Do not describe the full repository suite as
  all-green.
- `backend/tests/test_langchain_langgraph_dependency_compat.py` proves the
  supported version ranges and no-network API surfaces.
- `PostgresSaver` is import-ready only. No saver lifecycle, `.setup()`, schema,
  feature flag, runtime context contract, or public V2 route was implemented in
  Deliverable A.
- The next worker must write the Deliverable B runtime/checkpoint primitives
  plan from the approved foundation spec before changing production code.

## 2026-08-26 LangChain·LangGraph runtime foundation decision

- The user approved a focused runtime refactor before Neo4j GraphRAG work.
- A second architecture review split the work into four independent
  deliverables: dependency compatibility, runtime/checkpoint primitives,
  Review Queue HITL V2, and RAG retriever/graph V2.
- Current code imports and invokes real LangChain/LangGraph libraries, but the
  graph is a linear callable wrapper without conditional routing, durable
  checkpointing, or actual `interrupt()` / `Command(resume=...)` behavior.
- The existing `hitl_checkpoint` response is descriptive Review Queue metadata,
  not a LangGraph checkpoint.
- Follow
  `docs/superpowers/specs/2026-08-26-langchain-langgraph-runtime-foundation-design.md`
  before starting Neo4j foundation work.
- Preserve `EvidencePacket`, `PermissionContext`, Review Queue promotion,
  permission filtering, citations, cache, cost, and SQLite smoke contracts.
- Keep legacy routes stable during migration; new actual-HITL behavior belongs
  to disabled-by-default V2 routes.
- `needs_more_evidence` never promotes knowledge. In V2 it resumes only to
  close the current candidate attempt without downstream RAG or promotion; a
  later evidence run creates a new linked candidate version.
- Checkpoint state must contain opaque ids/hashes only. Runtime DB sessions,
  questions, source URLs/snippets, model output, and provider errors stay out.
- Use thread-bound database uniqueness and reconciliation for ReviewItem and
  AgentRun idempotency; the existing cache key is not sufficient.
- Do not use LangGraph `checkpoint_ns` as the application graph version. Keep
  the root namespace, use a server-issued checkpoint thread id, select the
  immutable builder from `AgentWorkflowThread.graph_version`, invoke with sync
  durability, and confirm the saver tuple before exposing a durable pause.
- Review V2 accepts canonical source/version references rather than raw
  questions. Model work runs outside row locks under a short claim lease, then
  persistence revalidates lease, evidence signature, permission, and state.
- Add unique `source_review_item_id` provenance to DecisionRecord,
  HistoryEvent, TimelineEvent, and Todo. Approval locks the ReviewItem and
  insert-or-returns existing promoted rows, including generated Timeline rows.
- A reused client idempotency key is valid only for the exact same canonical
  input/evidence/graph parameters; mismatch returns
  `idempotency_key_reused`.
- RAG V2 must validate structured answer blocks against server-issued evidence
  slots and project only selected canonical citations. Keep it in shadow mode
  until faithfulness does not regress.
- The user approved the revised Review Queue state machine and exactly-once
  promotion contract on 2026-08-26.
- The first execution plan is
  `docs/superpowers/plans/2026-08-26-langchain-langgraph-dependency-compatibility.md`.
  It covers Deliverable A only: target dependency ranges, targeted `uv.lock`
  refresh, and no-network compatibility tests. Runtime/checkpoint production
  code remains Deliverable B.
- No production code has been changed for this decision yet.

## 2026-05-16 Dashboard calendar sync and Review bulk actions

- Scope:
  - Dashboard Calendar API/UI, connector duplicate filtering, and Review Queue
    bulk action UX.
- Changes:
  - `backend/app/api/v1/dashboard.py` now returns both `today_events` and
    `calendar_events`. The former remains today's KPI/list source; the latter
    feeds the interactive calendar so synced events on other dates are visible.
  - `frontend/src/app/dashboard/page.tsx` consumes `calendar_events` with a
    `today_events` fallback and refreshes Dashboard state when
    `REVIEW_QUEUE_UPDATED_EVENT` fires, keeping review counts aligned with the
    sidebar.
  - `backend/app/ingestion/sync.py` filters unchanged duplicate source events
    before calling ingestion, while still reporting skipped counts from the
    fetched connector payload.
  - `frontend/src/app/review/page.tsx` adds Gmail-style top selection, selected
    bulk approve/reject, project assignment before bulk processing,
    duplicate/similar bulk approve/reject, right-click approve/reject, and an
    in-app confirmation modal instead of the old browser confirm path.
- Verification:
  - RED backend dashboard test failed on missing `calendar_events` before the
    API change.
  - RED Review Playwright test failed on missing `review-select-all` before the
    bulk UI change.
  - GREEN:
    `uv run pytest backend/tests/test_dashboard_api.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_review.py`
    -> 33 passed.
  - GREEN:
    `uv run ruff check backend/app/api/v1/dashboard.py backend/app/ingestion/sync.py backend/tests/test_dashboard_api.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_review.py`
    -> passed.
  - GREEN:
    `npm.cmd run test:visual -- review-bulk-actions.spec.ts dashboard-workflow.spec.ts --project=chromium-desktop`
    -> 3 passed.
  - GREEN: `npm.cmd run lint` -> passed with pre-existing timeline unused-import
    warnings only.
  - GREEN: `npm.cmd run build` -> passed.

## 2026-05-16 Review bulk action UX follow-up

- Scope:
  - Follow-up UX corrections for Review Queue bulk actions.
- Changes:
  - Review group headers no longer show the expand chevron; the left-most
    control is now a checkbox-style group selection button.
  - Duplicate/similar approve and reject buttons are shown only on duplicate
    group headers, positioned before the average confidence block.
  - Bulk confirm dialogs and right-click context menus are rendered through
    `createPortal(..., document.body)` so fixed backdrops cover the full
    viewport instead of the page content column.
  - Bulk failure copy now uses readable Korean:
    `승인 처리 중 N개 항목은 건너뛰었습니다. 필수 정보와 근거를 확인해 주세요.`
- Verification:
  - RED Review Playwright test first failed because group-level checkbox and
    group-level duplicate/similar buttons were missing, and because the modal
    backdrop started below the viewport top.
  - GREEN:
    `npm.cmd run test:visual -- review-bulk-actions.spec.ts --project=chromium-desktop`
    -> 2 passed.
  - GREEN: `npm.cmd run lint` -> passed with pre-existing timeline unused-import
    warnings only.
  - GREEN: `npm.cmd run build` -> passed.

## 2026-05-16 Timeline calendar status and filter cleanup

- Scope:
  - Timeline page only.
- Changes:
  - `frontend/src/app/timeline/page.tsx` now marks Calendar-sourced timeline
    items as `완료` when their `occurred_at` or `created_at` timestamp is before
    the current time.
  - Timeline status filters now expose only `상태 전체`, `approved`, and `완료`.
  - Timeline source filters now expose only `소스 전체`, `Slack`, `Gmail`,
    `Drive`, and `Calendar`; the fallback `Source` label is still used on rows
    when a source cannot be classified, but it is not a filter option.
  - Removed unused Timeline icon imports that had been producing lint warnings.
- Verification:
  - RED Timeline Playwright test first failed because `reviewing` was still in
    the status dropdown.
  - GREEN:
    `npm.cmd run test:visual -- timeline-project-date-groups.spec.ts --project=chromium-desktop`
    -> 3 passed.
  - GREEN: `npm.cmd run lint` -> passed with no warnings.
  - GREEN: `npm.cmd run build` -> passed.

## 2026-05-16 Docker Postgres port fallback

- Symptom:
  - `.\scripts\paraworks-docker.ps1` detected that `127.0.0.1:5432` was in use
    but still printed `Using Postgres host port 5432 for ParaWorks`, then Docker
    failed to bind Postgres on the same occupied or forbidden socket.
- Root cause:
  - `scripts/paraworks-docker.ps1` and the older `scripts/start-pgvector-dev.ps1`
    set `$fallbackPort = 5432`, so the auto-fallback path selected the original
    failing port.
- Fix:
  - Both PowerShell helpers now use `Get-AvailableHostPort -PreferredPort 5433`
    and pick the next free host port when the default 5432 listener is not the
    ParaWorks Postgres container.
  - Helpers now detect and reuse an already-running ParaWorks Postgres host port
    so repeated starts do not keep moving from `5433` to higher ports.
  - `docs/superpowers/runbooks/pgvector-dev.md` now documents 5433 examples and
    matching `DATABASE_URL` values for alternate host ports.
- Verification:
  - RED regression checks for the missing fallback failed before implementation.
  - A second RED check captured the repeated-start port drift before the reuse
    fix.
  - GREEN direct static test execution passed for
    `backend/tests/test_paraworks_docker_script.py` and
    `backend/tests/test_pgvector_dev_runbook.py`.
  - PowerShell parser checks passed for both helper scripts.
  - Escalated local verification started Docker services, ran migrations/schema
    checks, then started backend/frontend successfully; health and login smoke
    both returned HTTP 200.
- Note:
  - `uv run pytest ...` could not run in this shell because the global uv cache
    path and project uv trampoline were denied by the local Windows environment;
    the static test functions were executed directly with the available Python.

## 2026-05-16 Dashboard responsive UI polish

- Scope:
  - Dashboard-only frontend polish in `frontend/src/app/dashboard/page.tsx` and
    `frontend/src/app/globals.css`; no backend API or data contract changes.
- Changes:
  - Hero card now separates `dashboard-hero-copy` from
    `dashboard-hero-illustration`, with lucide/CSS mock workspace cards,
    avatars, chat, document, AI, and review visual elements.
  - Dashboard content grid is one column by default and becomes
    `main + 340px utility column` only at `min-width: 1280px`.
  - Hero illustration is hidden at tablet widths; Korean hero text uses
    wrapping-friendly sizing and no truncation.
  - Calendar card is kept in normal document flow with `overflow: visible`, and
    popovers clamp left/right edge alignment.
  - Initial selected calendar date now moves to the first event-bearing date if
    today's selected date has no events and the user has not clicked a date yet.
- Verification:
  - RED Playwright regression first failed because `.dashboard-hero-copy` did
    not exist.
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop`
    -> 2 passed.
  - `npm.cmd run lint` -> passed with pre-existing timeline unused-import
    warnings only.
  - `npm.cmd run build` -> passed.
  - Playwright viewport measurements passed for 2560, 1920, 1440, 1366, 1024,
    and 768 widths: no horizontal overflow, compact hero height, no hero
    copy/illustration overlap, right column flows below wide desktop.
- Note:
  - The Codex in-app browser plugin could not start in this Windows sandbox
    because Node hit an `EPERM` reading `C:\Users\hanvv\AppData`; browser
    verification used escalated Playwright instead.

## 2026-05-16 Dashboard calendar week order polish

- Scope:
  - Dashboard calendar UI only; no backend/API changes.
- Changes:
  - `WEEKDAY_LABELS` now renders `일, 월, 화, 수, 목, 금, 토`.
  - `buildCalendarDays()` now starts each 42-day grid from the Sunday before or
    on the first of the visible month.
  - `.dashboard-calendar-day.today` and `.selected` styles are separated so
    today's date remains softly highlighted when another date is selected.
- Verification:
  - RED Playwright regression first failed on the old Monday-start weekday
    order.
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop`
    -> 2 passed.
  - `npm.cmd run lint` -> passed with pre-existing timeline unused-import
    warnings only.
  - `npm.cmd run build` -> passed.

## 2026-05-16 Review Mail Docs Calendar source labels

- Scope:
  - Only `/review` UX and Review API source-evidence fallback were changed.
  - Slack Agent, RAG orchestration, connector ingestion, and project routing
    were intentionally left untouched.
- Backend:
  - `backend/app/api/v1/review.py` no longer falls back missing
    `source_type` to `slack`.
  - Source evidence now resolves metadata by source id as well as URL, then
    falls back to indexed `ReviewItem.payload.source_types`, scalar
    `payload.source_type`, and source URL heuristics.
- Frontend:
  - `frontend/src/lib/sourceLabels.ts` centralizes source-family labels.
  - `/review` card agent badges show `Mail`, `Docs`, `Calendar`, or combined
    labels for `mail_document_agent` items.
  - Gmail attachments map to `Mail`.
- Tests added:
  - Backend regression in `backend/tests/test_review.py`.
  - Playwright regression in
    `frontend/e2e/review-mail-docs-source-labels.spec.ts`.

## 2026-05-15 Google Calendar updatedMin fallback

- Symptom:
  - Google Calendar sync jobs failed with
    `failed: The requested minimum modification time lies too far in the past.`
- Root cause:
  - Calendar sync uses per-calendar `updatedMin` cursors from existing
    `Source.raw_metadata.sync_cursor` values.
  - The local Postgres database had an older holiday-calendar cursor
    (`calendar:ko.south_korea#holiday@group.v.calendar.google.com`,
    latest cursor `2026-04-10T17:52:54.663Z`), which Google Calendar rejected
    as `updatedMinTooLongAgo`.
- Fix:
  - `backend/app/connectors/google.py` now catches only this Calendar
    `updatedMin` expiry error and refetches that calendar through the existing
    initial Calendar window (`now-30d` to `now+180d`) instead of failing the
    whole sync.
  - Other Google API errors still fail normally.
- Verification:
  - RED regression:
    `uv run pytest backend/tests/test_google_connector.py::test_google_connector_refetches_calendar_window_when_updated_min_is_too_old -q`
    failed before implementation with the original `GoogleApiError`.
  - GREEN verification:
    `uv run pytest backend/tests/test_google_connector.py -q` -> `30 passed`.
  - `uv run pytest backend/tests/test_connector_ingestion_contract.py backend/tests/test_google_connector.py -q`
    -> `41 passed`.
  - `uv run ruff check backend/app/connectors/google.py backend/tests/test_google_connector.py`
    -> passed.

## 2026-05-15 Dashboard Calendar today events visibility

- Scope:
  - Connected Calendar source visibility to the existing Dashboard schedule
    panel. This did not add a Calendar Agent or promote raw Calendar events into
    trusted knowledge.
- Backend changes:
  - `backend/app/api/v1/dashboard.py` now returns `today_events` from Calendar
    `Source` rows whose `raw_metadata.event_start` or `raw_metadata.start` falls
    within today's Asia/Seoul date.
  - Invalid Calendar datetime strings and events outside today are ignored.
  - Returned fields are `id`, `title`, `start`, `end`, `location`, `organizer`,
    `attendee_summary`, `source_url`, and `permission_level`.
- Frontend changes:
  - `frontend/src/lib/api/types.ts` includes `DashboardResponse.today_events`.
  - `frontend/src/app/dashboard/page.tsx` maps `today_events` into the top
    schedule count and right-side "today schedule" list.
- Verification:
  - `uv run pytest backend/tests/test_dashboard_api.py -q` -> 4 passed.
  - `uv run ruff check backend/app/api/v1/dashboard.py backend/tests/test_dashboard_api.py` -> passed.
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop` -> 1 passed.
  - `npm.cmd run lint` -> passed with pre-existing timeline unused-import warnings.
  - `npm.cmd run build` -> passed.

## 2026-05-15 Google Calendar all-calendars MVP

- Scope:
  - Calendar stayed under Developer B's Mail/Document ownership and is now
    treated as Mail/Docs/Calendar evidence. No separate Calendar Agent or new
    endpoint was added.
- Backend changes:
  - `backend/app/connectors/google.py` now calls Google `calendarList` and then
    fetches events for each accessible calendar.
  - Initial Calendar collection uses `now-30d` to `now+180d`; delta collection
    uses per-calendar `updatedMin` from `sync_partition=calendar:{calendar_id}`.
  - Calendar source ids are `calendar:{calendar_id}:{event_id}`.
  - Calendar metadata is preserved through Source/DocumentChunk, Mail/Docs
    evidence packets, AgentRun evidence summary, ReviewItem payload/source
    evidence, and Projects/Timeline occurrence time.
  - Mail/Docs deterministic extraction now emits `timeline_event` for confirmed
    meetings/milestones, `todo` for preparation/deadline/follow-up, and skips
    low-signal personal calendar events.
  - `backend/app/projects/service.py` now prefers `event_start`/`start` from a
    Calendar source when computing `occurred_at`.
- Frontend changes:
  - Review source evidence can display Calendar name/start/end/location/
    organizer/attendee summary.
- Verification to rerun if continuing:
  - `uv run pytest backend/tests/test_google_connector.py backend/tests/test_connector_golden_dataset.py backend/tests/test_mail_document_agent.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_mail_document_agent_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_project_memory_api.py backend/tests/test_rag_indexing.py -q`
  - `uv run ruff check backend/app/connectors/google.py backend/app/agents/mail_document_agent/agent.py backend/app/agents/mail_document_agent/llm.py backend/app/agents/mail_document_agent/service.py backend/app/agent_runtime/evidence_summary.py backend/app/api/v1/review.py backend/app/projects/service.py`
  - `npm.cmd run lint`
  - `npm.cmd run build`
## 2026-05-15 대시보드 오늘 할 일 및 담당 프로젝트 개선

- 변경 배경:
  - `frontend/src/app/dashboard/page.tsx`에서 `visibleAssignedProjects`가 빈 배열로 하드코딩되어 `내 담당 프로젝트`가 항상 비어 있었다.
  - `backend/app/api/v1/dashboard.py`의 `today_todos`는 `pending_review` todo ReviewItem을 날짜 필터 없이 내려주고 있었다.
- 변경:
  - `backend/app/api/v1/dashboard.py`
    - 승인된 `ReviewItem(item_type="todo", status="approved")` 중 `payload.due_date`가 오늘(Asia/Seoul 기준) 이후인 항목을 가까운 마감일 순으로 `today_todos`에 반환한다.
    - `today_todos`에 `priority`를 포함한다.
    - `assigned_projects`를 추가해 `build_project_memory()` 결과의 프로젝트명, 요약, 근거 수, 활동 수, 검토 대기 수를 반환한다.
  - `frontend/src/app/dashboard/page.tsx`
    - 클라이언트 컴포넌트로 전환해 `/api/v1/dashboard`를 로드한다.
    - 오늘 할 일 카드에 완료 버튼을 추가하고, 클릭 시 현재 대시보드 state에서만 숨긴다.
    - `내 담당 프로젝트`에 `assigned_projects` 목록을 표시한다.
- 검증:
  - `uv run pytest backend/tests/test_dashboard_api.py -q` -> `3 passed`
  - `uv run ruff check backend/app/api/v1/dashboard.py backend/tests/test_dashboard_api.py` -> `All checks passed!`
  - `npm.cmd run lint` -> passed
  - `npm.cmd run build` -> passed
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop` -> `1 passed`
  - 실제 Docker DB 기준 `2026-05-18`, `2026-05-22` 마감 승인 todo 2건이 수정된 코드에서 반환됨을 확인했다.
- 주의:
  - 현재는 별도 “프로젝트 담당자” 모델이 없어서 `내 담당 프로젝트`는 사용자가 볼 수 있는 등록 프로젝트/프로젝트 메모리를 표시한다.
  - 완료 버튼은 의도대로 서버 상태를 변경하지 않는다. 새로고침하면 API 기준 오늘 이후 할 일이 다시 표시될 수 있다.
  - 이미 실행 중인 backend 서버는 재시작해야 변경된 `today_todos` 기준이 반영된다.

## 2026-05-15 프로젝트 근거 기본 선택 및 Slack 원문 시각 보강

- 배경:
  - 실제 Docker DB에는 `project-paraworks-mvp`에 원본 근거 12건과 타임라인 6건이 있었지만, `project-k`가 최신 생성 프로젝트라 프로젝트/타임라인 탭에서 먼저 선택되어 빈 화면처럼 보였다.
  - `backend/app/projects/service.py`의 `occurred_at` 계산은 Source URL이 매칭된 뒤 `raw_metadata.ts`가 없으면 Slack permalink timestamp보다 Source 생성 시각을 먼저 fallback할 수 있었다.
- 변경:
  - `frontend/src/app/projects/page.tsx`는 초기 진입 시 승인된 근거/활동/타임라인이 있는 첫 프로젝트를 기본 선택한다.
  - `frontend/src/app/timeline/page.tsx`는 초기 진입 시 승인된 타임라인 항목이 있는 첫 프로젝트를 기본 선택한다.
  - `backend/app/projects/service.py`는 Source가 매칭되더라도 `raw_metadata.ts` 파싱 실패 후 Slack permalink의 `p...` timestamp를 먼저 확인하고, 마지막에만 `Source.created_at`으로 fallback한다.
- 검증:
  - `uv run pytest backend/tests/test_project_memory_api.py::test_project_timeline_prefers_slack_permalink_timestamp_when_source_metadata_is_missing backend/tests/test_project_memory_api.py::test_project_timeline_items_use_slack_source_timestamp_for_occurred_at -q` -> `2 passed`
  - `uv run pytest backend/tests/test_project_memory_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py -q` -> `47 passed`
  - `uv run ruff check backend/app/projects/service.py backend/tests/test_project_memory_api.py` -> `All checks passed!`
  - `npm.cmd run lint` -> passed
  - `npm.cmd run build` -> passed
  - `npm.cmd run test:visual -- timeline-project-date-groups.spec.ts projects-source-links.spec.ts slack-project-routing-flow.spec.ts gmail-drive-project-routing-flow.spec.ts --project=chromium-desktop` -> `6 passed`
- 주의:
  - 이미 실행 중인 backend/frontend dev server는 코드 변경을 반영하려면 재시작이 필요할 수 있다.

## 2026-05-15 Gmail/Drive 프로젝트 라우팅 승인 연결

- 역할 경계:
  - 개발자 C는 Gmail/Drive 고도화 중 공용 프로젝트 라우팅 계약, Review 승인 정책, Review UX, Timeline/Projects/RAG 연결만 담당했다.
  - `backend/app/connectors/google.py`, `backend/app/agents/mail_document_agent/`, `agent_slack/`, `backend/app/agents/slack_agent/`는 수정하지 않았다.
- 주요 변경:
  - `backend/app/agent_runtime/project_routing.py`에 `ProjectOption`, `ProjectRoutingCandidate`, `ProjectRoutingDecision`, `ProjectRoutingResult`, `ProjectRouterModel`, `route_projects_for_candidates`, `apply_project_routing_to_payload` 공용 계약을 추가했다.
  - Gmail/Drive Mail/Document Agent ReviewItem 중 `project_assignment_method="llm_tool"`인 지식 후보는 `project_key`가 확정되지 않았거나 `project_needs_user_selection=true`이면 승인할 수 없도록 했다.
  - Review 화면은 프로젝트 선택이 필요한 후보의 승인 버튼을 비활성화하고, 등록 프로젝트를 선택하면 같은 ReviewItem을 PATCH로 보정한 뒤 승인 가능하게 만든다.
  - 승인된 Gmail/Drive 후보의 `project_key`는 Timeline/Projects 활동과 RAG indexing metadata까지 보존된다.
- 검증:
  - `uv run pytest backend/tests/test_agent_runtime_project_routing.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_project_memory_api.py backend/tests/test_rag_indexing.py -q` -> `74 passed`
  - `uv run ruff check backend/app/agent_runtime/project_routing.py backend/app/knowledge/promotion.py backend/app/api/v1/review.py backend/app/projects/service.py backend/app/rag/indexing.py backend/tests/test_agent_runtime_project_routing.py backend/tests/test_review.py backend/tests/test_project_memory_api.py backend/tests/test_rag_indexing.py` -> `All checks passed!`
  - `npm.cmd run lint` -> passed
  - `npm.cmd run build` -> passed
  - `npm.cmd run test:visual -- gmail-drive-project-routing-flow.spec.ts` -> desktop/mobile `2 passed`
  - `npm.cmd run test:visual -- review-project-routing.spec.ts` -> desktop/mobile `2 passed`

## 2026-05-15 Slack 프로젝트 Tool Routing 통합 완료

- 기준 계획서:
  - `docs/superpowers/plans/2026-05-15-unified-slack-project-tool-routing.md`
- 주요 변경:
  - `agent_slack/project_routing.py`
    - router rules에 `등록 프로젝트에 해당한다고 판단한 경우에만 project_key를 채우세요`, `모든 candidate_items에 대해 decisions 항목을 하나씩 반환하세요`를 추가했다.
  - `backend/app/agents/slack_agent/sync_service.py`
    - Slack Agent ReviewItem 저장 시 `_determine_project_from_tag()` fallback과 `back_propagate_slack_tags()` 호출을 제거했다.
    - payload의 `project_key`, `project_name`은 tool routing 결과가 있을 때만 저장한다.
  - `backend/app/api/v1/integrations.py`
    - Slack sync에서는 규칙 기반 `project_assignment`를 만들지 않도록 했다.
  - `backend/app/projects/classifier.py`
    - deterministic project classifier 대상에서 Slack source를 제외했다. Gmail/Drive/Calendar deterministic backfill은 유지된다.
  - `backend/app/knowledge/promotion.py`
    - Slack Agent `llm_tool` 후보는 `project_key`가 없으면 promotion preview와 approve API에서 승인 불가다.
  - `backend/app/projects/service.py`
    - project pending count가 project_key를 가진 pending ReviewItem 전체를 반영한다.
  - `frontend/src/app/review/page.tsx`
    - 프로젝트 미선택 후보에 `프로젝트 선택 후 승인 가능` 안내와 `새 프로젝트 만들기` 링크를 표시한다.
  - `frontend/src/app/timeline/page.tsx`
    - 프로젝트 타임라인을 날짜 단위로 그룹 표시한다.
  - `frontend/src/app/projects/page.tsx`
    - metric 영역을 모바일 1열, 넓은 화면 3열로 안정화했다.
- 신규/수정 테스트:
  - `backend/tests/test_agent_slack_pipeline_quality.py`
  - `backend/tests/test_slack_agent_api.py`
  - `backend/tests/test_mock_sync.py`
  - `backend/tests/test_project_memory_api.py`
  - `backend/tests/test_review.py`
  - `frontend/e2e/review-project-routing-required.spec.ts`
  - `frontend/e2e/timeline-project-date-groups.spec.ts`
  - `frontend/e2e/projects-responsive-metrics.spec.ts`
  - `frontend/e2e/slack-project-routing-flow.spec.ts`
- 검증:
  - `python -m pytest backend/tests/test_agent_slack_project_routing.py backend/tests/test_agent_slack_pipeline_quality.py backend/tests/test_slack_agent_api.py backend/tests/test_mock_sync.py backend/tests/test_project_memory_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py -q` -> `65 passed`
  - `python -m ruff check ...` -> `All checks passed!`
  - `npm.cmd exec tsc -- --noEmit` -> passed
  - `npm.cmd run lint` -> passed
  - `npm.cmd run build` -> passed
  - `npm.cmd run test:visual -- review-project-routing-required.spec.ts timeline-project-date-groups.spec.ts projects-responsive-metrics.spec.ts slack-project-routing-flow.spec.ts` -> `8 passed`
- 주의:
  - 기존 DB에 이미 남은 `project_assignment` 항목은 이번 변경으로 자동 삭제하지 않는다.
  - 신규 Slack sync부터는 Slack source가 deterministic project classifier로 다시 들어가지 않는다.
  - Gmail/Drive/Calendar deterministic classifier 경로는 개발자 B/C 분업 전까지 유지된다.

## 2026-05-15 Slack 프로젝트 Router Tool Agent 인수인계

- 목적:
  - 사용자가 등록한 프로젝트 목록을 기준으로 Slack Agent가 추출한 `decision_record`, `todo`, `history_event` 후보를 LLM tool-calling 방식으로 프로젝트에 연결한다.
  - 프로젝트 연결 요약과 근거를 Review Queue에서 확인한 뒤 사용자가 프로젝트를 바꾸거나 승인할 수 있게 한다.
- 주요 변경 파일:
  - `agent_slack/project_routing.py`
  - `agent_slack/agent_slack.py`
  - `agent_slack/slack_agent_langgraph.md`
  - `backend/app/agents/slack_agent/sync_service.py`
  - `backend/app/api/v1/integrations.py`
  - `frontend/src/app/review/page.tsx`
  - `frontend/e2e/review-project-routing.spec.ts`
  - `backend/tests/test_agent_slack_project_routing.py`
- 현재 LangGraph:
  - `START -> preprocess -> classify -> summarize -> extract -> project_route -> END`
  - `classify`에서 업무 신호가 없으면 바로 `END`로 종료한다.
  - `project_route`는 등록 프로젝트와 추출 후보가 있을 때만 LangChain tool-calling router를 실행한다.
- 데이터 계약:
  - `ProjectOption(project_key, name, summary)`
  - `ProjectRoutingDecision(source_id, item_index, project_key, project_name, confidence_score, assignment_summary, assignment_reason, alternatives, needs_user_selection)`
  - `ReviewItem.payload` 추가 필드:
    - `project_assignment_method=llm_tool`
    - `project_assignment_summary`
    - `project_assignment_reason`
    - `project_assignment_confidence`
    - `project_alternatives`
    - `project_needs_user_selection`
- 안전 경계:
  - 테스트에서 live LLM을 호출하지 않는다. fake model 또는 monkeypatch를 사용한다.
  - Review 승인 전까지 LLM project routing 결과는 trusted knowledge가 아니다.
  - Slack LLM routing으로 Agent 후보가 생성된 경우에만 deterministic `project_assignment` 중복 생성을 건너뛴다. provider key가 있더라도 Agent 후보가 0개인 no-op sync에서는 fallback 분류가 막히지 않는다.
- 검증:
  - `uv run pytest backend/tests/test_agent_slack_project_routing.py backend/tests/test_agent_slack_pipeline_quality.py backend/tests/test_slack_agent_api.py backend/tests/test_mock_sync.py backend/tests/test_project_memory_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py -q` -> `60 passed`
  - `uv run ruff check ...` -> `All checks passed!`
  - `npm.cmd exec tsc -- --noEmit` -> passed
  - `npm.cmd run lint` -> passed
  - `npm.cmd run build` -> passed
  - `npm.cmd run test:visual -- review-project-routing.spec.ts` -> desktop/mobile `2 passed`

## 2026-05-15 Slack 장시간 동기화 실패 오인 인수인계

- 증상:
  - 사용자가 Slack 동기화 진행 중 `동기화 실패`가 표시된다고 보고했다.
- 실제 확인:
  - Playwright로 로그인 후 `/integrations`에서 Slack 동기화를 실행했다.
  - 최신 job `slack-34e086b550e64dcb94b75072f87577b6`은 `complete`, `last_error=null`이었다.
  - message는 `fetched=0 created_review_items=5 skipped_events=0 pending_review_items=13`.
  - 이번 실제 검증으로 pending review가 8개에서 13개로 증가했다.
- 원인:
  - 대량 sync job은 약 153초 걸린 기록이 있었다.
  - 프론트 polling 한도는 135초라, 백엔드가 정상 running 중이어도 프론트가 timeout을 error로 처리했다.
  - 그 결과 모달 제목이 `Slack 동기화 실패`로 표시될 수 있었다.
- 수정:
  - `frontend/src/app/integrations/page.tsx`에 `SYNC_BACKGROUND_NOTICE_DELAY_MS=120_000`과 background-running 안내 문구를 추가했다.
  - polling timeout 메시지는 실제 실패가 아니라 running/backgrounded 상태로 유지한다.
  - 모달은 `백그라운드에서 계속 진행 중입니다. 완료되면 작업 스트림의 최근 sync 상태에 반영됩니다.`를 표시한다.
- 검증:
  - `npm.cmd run test:visual -- integration-sync-modal.spec.ts --project=chromium-desktop -g "polling timeout"` -> `1 passed`
  - `npm.cmd run test:visual -- integration-sync-modal.spec.ts` -> desktop/mobile `6 passed`
  - `npm.cmd exec tsc -- --noEmit` -> passed
  - `npm.cmd run lint` -> passed
  - `npm.cmd run build` -> passed

## 2026-05-14 Slack sync ReviewItem 복구 및 검토사항 정렬

- 증상:
  - 사용자가 DB에서 `review_items` 데이터를 직접 삭제한 뒤 Slack 동기화를 다시 눌러도 검토사항 화면에 업무 후보가 보이지 않았다.
  - 실제 API에는 `project_assignment` 후보가 많이 생성되어 첫 화면을 차지했고, Slack Agent가 만든 `decision_record`, `todo`, `history_event` 후보는 뒤로 밀려 사용자가 “검토사항이 없다”고 판단하기 쉬웠다.
- 원인:
  - connector sync는 `Source` 중복을 비용 절감 신호로 보고 `changed_source_ids`가 없으면 Slack Agent ReviewItem 생성을 건너뛰었다.
  - 따라서 `Source`는 남아 있고 `ReviewItem`만 삭제된 복구 상황에서는 기존 Slack 원본을 다시 검토 후보로 승격하지 못했다.
  - Review 목록 기본 정렬이 최신 id 기준이라 대량의 `project_assignment` 후보가 업무 지식 후보보다 먼저 표시됐다.
- 수정:
  - `backend/app/api/v1/integrations.py`에서 중복 sync라도 해당 connector의 Agent ReviewItem이 하나도 없고 기존 `Source`가 있으면 기존 source ids로 Slack/Mail-Document Agent review 생성을 복구하도록 했다.
  - `backend/app/api/v1/review.py`에서 검토사항 정렬 우선순위를 `decision_record`, `todo`, `history_event`, `timeline_event`, 기타, `project_assignment` 순으로 조정했다.
  - 서버를 재시작해 최신 코드가 실제 API 프로세스에 반영되도록 했다.
- 검증:

```powershell
uv run pytest backend/tests/test_mock_sync.py::test_duplicate_slack_sync_recreates_agent_reviews_when_review_items_were_deleted backend/tests/test_review.py::test_review_list_prioritizes_knowledge_candidates_before_project_assignments -q
uv run pytest backend/tests/test_mock_sync.py backend/tests/test_review.py backend/tests/test_project_memory_api.py backend/tests/test_slack_agent_api.py -q
uv run pytest backend/tests/test_agent_slack_pipeline_quality.py backend/tests/test_slack_agent_quality.py backend/tests/test_slack_agent.py backend/tests/test_slack_agent_review_bridge.py backend/tests/test_slack_agent_api.py backend/tests/test_project_memory_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_mock_sync.py -q
uv run ruff check backend/app/api/v1/integrations.py backend/app/api/v1/review.py backend/app/projects/classifier.py backend/tests/test_mock_sync.py backend/tests/test_review.py backend/tests/test_project_memory_api.py
```

결과:

- 신규 복구/정렬 회귀 테스트 2 passed.
- 관련 sync/review/project/slack API 묶음 43 passed.
- Slack/Review/Project 전체 타깃 회귀 묶음 69 passed.
- ruff passed.
- 실행 중인 API에서 `/api/v1/review?status=pending_review&limit=10`가 `total_count=217`을 반환하고 첫 항목들이 `decision_record`, `todo`, `history_event` 순으로 노출되는 것을 확인했다.

## 2026-05-14 사용자 정의 프로젝트 동기화 검토사항 수정

- 증상:
  - 동기화 버튼을 눌러도 프로젝트 관련 항목이 Review Queue로 들어오지 않았다.
  - 새로 만든 프로젝트 요약 뒤에 `?꾩쭅 ?뱀씤???꾨줈?앺듃 evidence...`
    같은 깨진 한글 문자열이 붙었다.
- 원인:
  - `/api/v1/projects/define`은 `Project` 행만 저장하고, 이미 동기화된 source를
    새 프로젝트 기준으로 다시 분류하지 않았다.
  - `/api/v1/integrations/{connector_type}/sync`는 connector별 review agent는
    실행했지만, 사용자 정의 프로젝트에 대한 `project_assignment` 후보는 만들지
    않았다.
  - `backend/app/projects/service.py`와
    `backend/app/projects/classifier.py`에 깨진 한글 fallback 문자열이 남아 있었다.
- 수정:
  - 프로젝트 생성 시 새 `Project` 행을 flush한 뒤
    `create_project_assignment_review_items()`를 호출하고
    `created_review_items`를 응답에 포함했다.
  - connector sync가 source 저장 뒤 같은 deterministic 프로젝트 분류기를 실행하고,
    응답/audit metadata에 `project_assignment_items`를 포함하게 했다. 변경 source가
    없고 skipped 된 기존 source만 있어도 프로젝트 분류는 다시 시도한다.
  - 프로젝트 요약, 근거 사유, 승인 타임라인 사유, fallback 프로젝트 라벨, source
    라벨을 읽을 수 있는 한국어로 다시 작성했다.
  - 한국어 프로젝트 키워드 추출 범위를 `가-힣`로 정리했다.
- 검증:

```powershell
uv run pytest backend/tests/test_project_memory_api.py::test_define_project_returns_readable_empty_summary backend/tests/test_project_memory_api.py::test_define_project_creates_pending_assignment_candidates_from_existing_sources -q
uv run pytest backend/tests/test_mock_sync.py::test_sync_creates_project_assignment_review_items_for_defined_projects -q
uv run pytest backend/tests/test_mock_sync.py::test_duplicate_sync_still_classifies_existing_sources_for_new_project backend/tests/test_slack_agent_api.py::test_slack_sync_uses_agent_slack_llm_pipeline_when_provider_key_exists -q
uv run pytest backend/tests/test_project_memory_api.py backend/tests/test_mock_sync.py backend/tests/test_slack_agent_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py -q
uv run ruff check backend/app/api/v1/projects.py backend/app/api/v1/integrations.py backend/app/projects/service.py backend/app/projects/classifier.py backend/tests/test_project_memory_api.py backend/tests/test_mock_sync.py
cd frontend
npm.cmd exec tsc -- --noEmit
npm.cmd run build
```

결과:

- 신규 프로젝트 회귀 테스트: 2 passed;
- sync 프로젝트 연결 회귀 테스트: 1 passed;
- 중복 sync + Slack LLM sync 회귀 테스트: 2 passed;
- 관련 백엔드 테스트 묶음: 37 passed;
- ruff: passed;
- 프론트엔드 TypeScript/build: passed.

## 2026-05-14 Mail/Document Agent Review Quality and Promotion Flow

- Mail/Document live LLM review generation now uses source-grouped windows
  instead of one all-corpus candidate. Gmail attachments remain grouped with
  their parent email, and Drive/Calendar evidence stays source-local.
- The shared agent LLM default model is now `gpt-5.4-mini`; `.env.example`,
  backend settings, and Mail/Docs/Slack LLM defaults are aligned. Local `.env`
  values can still override this.
- Mail/Docs LLM parsing now treats string `"false"` as false and filters
  reserved `structured_data` fields so LLM output cannot overwrite ReviewItem
  source ids, AgentRun ids, title/summary, or cost metadata.
- Mail/Docs ReviewItems can carry action-oriented fields such as
  `business_context`, `task_summary`, `recommended_next_step`, `assignee`,
  `due_date`, `counterparty`, and `source_subject`; Review Queue shows these as
  an 업무 판단 block before source evidence.
- Review approval now returns `promotion_result` with created knowledge ids,
  created timeline ids, project key, and next routes; the frontend displays a
  post-approval navigation CTA.
- Todo promotion copy was repaired to clean Korean timeline text, and todo
  approval can use `recommended_next_step` or `task_summary` as the priority
  reason fallback.
- Document Agent portfolio notes for this work are in
  `docs/portfolio-log-docs-agent.md`; do not duplicate this entry in
  `docs/portfolio-log.md`.

Verification:

```powershell
uv run pytest backend/tests/test_mail_document_agent.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_mail_document_agent_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_project_memory_api.py -q
uv run ruff check backend/app/agents/mail_document_agent backend/app/agents/slack_agent/llm.py backend/app/api/v1/integrations.py backend/app/api/v1/review.py backend/app/knowledge/promotion.py backend/app/core/config.py
cd frontend
npm.cmd exec tsc -- --noEmit
npm.cmd run build
```

Result: 51 backend tests passed, ruff passed, TypeScript check passed, and
frontend production build passed.

## 2026-05-14 Project/Timeline/RAG Approval Visibility Fix

- Approved knowledge records now preserve `project_key` into project timeline
  API items, so `/api/v1/projects` can attach promoted Timeline, History,
  Decision, and Todo records directly to the matching project.
- `/projects` now shows approved workflow items from `timeline_items` in
  addition to connector assignment evidence.
- Mail/Document ReviewItems now preserve `source_ids`, `source_types`,
  `source_urls`, and `source_authors`, allowing approved source chunks to enter
  RAG indexing through the approval-based policy.
- `backend/app/rag/indexing.py` already had `ReviewItem` imported in the
  current checkout; the old `NameError` was not present during this session.
- RAG tests were updated to the current policy: original source chunks are
  indexed only when their external `Source.source_id` appears in an approved
  ReviewItem payload; approved knowledge records are still indexed separately.
- Verification:

```powershell
uv run pytest backend/tests/test_project_memory_api.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_rag_indexing.py -q
cd frontend
npm.cmd exec tsc -- --noEmit
npm.cmd run build
```

Result:

- targeted backend tests: 44 passed;
- frontend TypeScript check: passed;
- frontend build: passed.

- Broader backend suite status should still distinguish unrelated existing
  Slack OAuth PKCE and fake client contract failures from this targeted fix.
- If old local rows still have `timeline_events.project_key = NULL`, rerun
  project classification and approve fresh Review Queue candidates, or use a
  deliberate local-only migration after inspecting source links.

## 2026-05-14 AI Assistant Service NameError Hotfix

- Symptom: `/api/v1/assistant/conversations/{conversation_id}/messages`
  returned 500 after the LLM call, so the AI assistant appeared not to answer.
- Error log:
  `backend/app/assistant/service.py` raised
  `NameError: name 'MAX_CONTEXT_MESSAGE_CHARS' is not defined` inside
  `_compact_context_text()`.
- Root cause:
  - the context-deduplication service expected `MAX_CONTEXT_MESSAGE_CHARS` and
    `MAX_SUMMARY_LINES`;
  - the constants were missing from the current branch;
  - the same context block also still appended raw recent messages after the
    compacted/deduped messages, which defeated the dedupe path.
- Fix:
  - restored `MAX_CONTEXT_MESSAGE_CHARS = 500`;
  - restored `MAX_SUMMARY_LINES = 4`;
  - removed the duplicate raw recent-message append;
  - updated stale assistant service tests from `employee-jun` to the current
    `hanvv-employee` demo user key.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_service.py -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py -q
git diff --check
```

Result:

- assistant service tests: 11 passed;
- assistant API + service tests: 24 passed;
- whitespace check: passed.

- Runtime check:
  - restarted `scripts/paraworks-docker.ps1` serious mode;
  - backend health returned `{"status":"ok","service":"paraworks","demo_mode":false}`;
  - a short authenticated assistant API smoke request returned HTTP 200;
  - backend error log after smoke showed no repeated `NameError`.

## 2026-05-14 Project Recognition Handoff

- `/projects` now uses canonical company projects instead of loose source
  grouping:
  - `k-tech-pilot` / `K테크 파일럿`
  - `seed-ir` / `시드 투자 IR`
- Deterministic project classification lives behind the Review Queue as
  `project_assignment` candidates. It scans Slack, Gmail, Drive, and Calendar
  sources for project aliases and intentionally uses no live LLM or token
  budget.
- `POST /api/v1/projects/reclassify?dry_run=true` previews candidate counts and
  cost policy. `dry_run=false` creates pending Review Queue items for approved
  reviewer handling.
- `/projects` returns both canonical projects with approved evidence,
  pending-review counts, and project-scoped `timeline_items`. Legacy labels
  like `미분류 프로젝트`, `Project Newbiegenie`, and `프로젝트 결과` should not be
  displayed as projects.
- `/timeline` now reads `/api/v1/projects` so the top menu is project-scoped
  and timeline evidence explains why each item is connected.
- Deterministic RAG fallback no longer has a hard-coded Redis/PostgreSQL answer.
  It now formats retrieved evidence snippets, and AgentRun metadata records
  `retrieval_backend`, `rag_model_mode`, and any fallback reason.
- Assistant conversation context deduplicates repeated assistant answers so an
  old bad answer does not keep contaminating later RAG questions.
- For a clean local/dev rerun, use `uv run python scripts/reset_connector_data.py`
  for dry-run counts, then `uv run python scripts/reset_connector_data.py
--execute --confirm` only in local env. This preserves auth users and
  integration connections but clears connector-derived source/review/knowledge,
  vector, AgentRun, and assistant data.
- After reset, rerun connector sync, call project reclassify, and approve the
  resulting `project_assignment` Review Queue candidates.
- Existing DB rows are not deleted or migrated. Run deterministic reclassify
  and approve the resulting Review Queue candidates to attach current source
  data to projects.

## 2026-05-13 Work Data and Assignment Extraction Handoff

- Dashboard recent timeline output now uses real `TimelineEvent` fields:
  `summary`, `created_at`, `confidence_score`, and `source_links`.
  Frontend code should not reintroduce `event_time` or `importance`.
- `/projects` is connected to `GET /api/v1/projects`; the page no longer uses
  local ORION/Nova/Atlas seed data.
- Future todo promotion creates clean Korean timeline entries such as
  `[할 일] ...` and `담당자: ..., 기한: ...`. Existing broken DB rows are not
  migrated by this slice.
- Mail/Docs and Memory Extraction deterministic models now detect generic
  Korean/English work assignment cues from Gmail, Drive, and Calendar evidence.
  Live LLM execution remains closed; only preflight endpoints were added.
- Verification completed:
  `uv run pytest backend/tests/test_dashboard_api.py backend/tests/test_knowledge_api.py backend/tests/test_review.py backend/tests/test_mail_document_agent.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_memory_extraction_agent.py backend/tests/test_memory_extraction_review_bridge.py backend/tests/test_agent_preflight.py -q`,
  `uv run ruff check ...`, `npm run lint`, `npm run build`, and
  `git diff --check`.

## 2026-05-13 RAG Orchestrator Assistant Handoff

- Active branch: `codex/rag-orchestrator-assistant-memory`.
- Latest pushed commit before this handoff update:
  `506b257 fix: surface Gmail send failures`.
- Current local serious mode status:
  - backend: `http://127.0.0.1:8000`
  - frontend: `http://127.0.0.1:3000`
  - backend health returned `demo_mode=false`.
- Gmail runtime status checked during the session:
  - Gmail integration was connected;
  - credential status was available for `hanvv3@koreacu.ac.kr`.
- Email-send approval flow investigation:
  - `/search` already calls
    `POST /api/v1/assistant/messages/{messageId}/email/send` when a user
    approves a pending email draft.
  - The backend send path goes through
    `backend/app/assistant/gmail_sender.py`.
  - The sender requires a connected Gmail integration, `gmail.send` scope, a
    stored token in `.tokens.json`, and refresh-token credentials when the
    access token is expired.
  - The local backend had been running without reload, so changed backend code
    required a server restart before the send endpoint could reflect updates.
- Implemented in commit `506b257`:
  - Added focused tests in `backend/tests/test_gmail_sender.py`.
  - Gmail API send failures now surface as explicit `GmailSendError` codes such
    as `gmail_api_send_failed:403` instead of becoming opaque runtime errors.
  - Gmail refresh failures now surface as explicit error codes such as
    `gmail_refresh_failed:{status}` and `gmail_refresh_unreachable`.
  - `/search` maps backend email-send error codes to Korean user-facing
    messages so the user can tell whether the problem is missing connection,
    missing `gmail.send` scope, missing token, refresh failure, or Gmail API
    rejection.
- Verification completed for that commit:

```powershell
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_gmail_sender.py -q
cd frontend
npm.cmd run lint
npm.cmd run build
git diff --check
```

Result:

- backend targeted tests: 14 passed;
- frontend lint: passed with an existing warning in
  `frontend/src/app/projects/page.tsx` about unused `projectSeedData`;
- frontend build: passed;
- whitespace check: passed.

Next recommended steps:

1. Reproduce the approve-send flow in the browser after logging in with the
   Gmail-connected account.
2. If sending still fails, capture the backend response body from the
   `/api/v1/assistant/messages/{messageId}/email/send` request. The new error
   code should now point to the exact missing OAuth/token/Gmail API condition.
3. If the error is `gmail_send_scope_required`, reconnect Gmail after the
   expanded `gmail.send` scope change so Google issues a token with send
   permission.
4. If the error is `gmail_api_send_failed:403`, check Google Cloud OAuth app
   verification/test-user status and Gmail API enablement.

## 2026-05-12 Demo Data Boundary Update

- Default settings now use `PARAWORKS_DEMO_MODE=false` and
  `PARAWORKS_SEED_DEMO_DATA=false`.
- Smoke mode is the only intended path for seeded dummy content:
  `scripts/start-smoke.ps1` sets both demo mode and seed demo data to true.
- Docker/pgvector dev mode (`scripts/start-pgvector-dev.ps1`) starts the app
  with demo mode and seed demo data disabled. With no Slack or Google connection
  installed, the product should show empty states rather than mock business
  content.
- Production-like connector sync must not fall back to mock connectors. It now
  returns a clear not-connected error until OAuth/credentials are available.
- The Review page no longer displays hard-coded fallback review items when the
  API fails or returns no items.
- Dashboard, Projects, and Timeline no longer render sample ORION/Nova/Atlas
  items as visible product data when no connector-backed data exists.

Verification from this session:

```powershell
uv run pytest backend/tests -v
cd frontend
npm run build
```

Result: backend 297 passed, 1 skipped; frontend build passed.

## Active Project

- Repository: `C:\Users\hanvv\Study\potenup3\pj04_ParaWorks`
- Plan draft: `C:\Users\hanvv\Downloads\plan-merged.md`
- Primary spec: `docs/superpowers/specs/2026-04-30-paraworks-harness-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-30-paraworks-harness.md`
- Assistant guide: `AGENTS.md`
- Portfolio log: `docs/portfolio-log.md`
- Current browser URL during handoff: `http://127.0.0.1:3000/dashboard`

## Product Alignment

ParaWorks is currently aligned as an Adapter-First Demo Harness for a company-wide knowledge and decision-history platform. It is not a team task manager and not a Streamlit app.

The MVP harness keeps real SaaS integrations behind connector contracts and validates the core workflow with mock Drive, Gmail, Slack, and Calendar data:

1. Start mock sync from the frontend.
2. FastAPI creates a sync job.
3. SSE streams job status.
4. Ingestion normalizes source events.
5. Deterministic extraction creates pending review items.
6. Review UI exposes evidence, approve/reject/edit/request-more-evidence actions.
7. Search returns permission-filtered source evidence.

## 2026-05-12 AI 비서 ChatGPT-style Polish and RAG LLM Handoff

- Active branch for this work: `codex/rag-orchestrator-assistant-memory`.
- `/search` is now the primary AI 비서 surface and should feel closer to a
  natural ChatGPT-style conversation:
  - the left history shows compact conversation titles only;
  - `+` reuses an existing empty `새 대화` instead of creating duplicates;
  - evidence and source details live inside each assistant message behind a
    fold/unfold control;
  - the input composer remains at the bottom of the chat surface while evidence
    scrolls inside its own bounded panel.
- Assistant conversations remain database-backed per logged-in user through
  `assistant_conversations` and `assistant_messages`.
- In demo mode, RAG answering stays deterministic for smoke tests and cheap
  demos.
- In non-demo 진심모드, RAG answering builds a real LangChain model chain:
  - primary OpenAI model: `gpt-5.4-mini`;
  - fallback OpenAI model: value from `AGENT_LLM_OPENAI_MODEL` in `.env`;
  - provider fallback continues through `AGENT_LLM_PROVIDER_ORDER`, including
    Gemini when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is configured.
- For another local machine to continue this branch, pull the branch, run
  `uv sync`, `cd frontend && npm.cmd ci`, then set `.env` for 진심모드 with
  `PARAWORKS_DEMO_MODE=false`, `OPENAI_API_KEY`, and optional
  `AGENT_LLM_OPENAI_MODEL` fallback before starting Docker.
- Additional 2026-05-13 UI refinements:
  - conversation history order is based on `updated_at`, not click selection;
  - only the chat transcript pane scrolls when the viewport is short;
  - user messages render as rounded full pills without a `나` label;
  - assistant role/permission badges were removed from message bodies;
  - assistant answers render basic markdown and both user/assistant messages
    expose a small copy action;
  - recommended rounded-full prompt chips above the composer send immediately
    when clicked.

## Latest Session Changes

- Fixed frontend dependency drift by reinstalling `frontend/node_modules` from `package-lock.json` with `npm.cmd ci`.
- Added `outputFileTracingRoot` in `frontend/next.config.ts` so Next does not infer `C:\Users\hanvv` as the workspace root because of an upper-level `package-lock.json`.
- Fixed `frontend/src/hooks/useJobStatus.ts` so a normal SSE `done` event closes the stream without being overwritten by `job stream unavailable`.
- Updated `.gitignore` for local generated files:
  - `.tmp/`
  - `frontend/tsconfig.tsbuildinfo`
  - existing local env ignores

## Verification Completed

Backend:

```powershell
uv run pytest backend/tests -v
```

Result: 18 passed.

Frontend:

```powershell
cd frontend
npm.cmd run build
```

Result: build passed with Next.js 15.5.15.

Browser smoke test used the in-app browser and an SQLite smoke DB because Docker is not available on PATH in this environment.

Verified pages and flows:

- `/integrations`: Slack mock sync runs.
- SSE job stream displays completion JSON.
- Sync creates 3 pending review items on a fresh smoke DB.
- `/review`: review items render.
- Evidence drawer opens and shows source snippets/links.
- `/search`: viewer Redis search returns accessible Slack evidence.
- `/dashboard`: source, pending review, and recent job counts render.

## Runtime State Left Running

At the end of the latest session, local servers were started for manual inspection:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`

The backend was started against a temporary SQLite DB:

```powershell
DATABASE_URL=sqlite:///./.tmp/paraworks-smoke-fresh.db
```

If a later session needs a clean smoke run, create a new `.tmp/*.db` file or delete the old one.

## Important Environment Notes

- `docker` is not currently recognized in PATH, so `docker compose config` and Postgres/Redis/MinIO runtime verification could not be completed.
- Backend tests pass in the real environment with `uv run`; sandboxed runs may fail with local uv cache or Python spawn permission errors.
- `next dev` can enter a stale `.next` state if `npm.cmd run build` is run while the dev server is still active. Restart `next dev` after production builds.
- The frontend depends on Next 15 according to `package.json` and `package-lock.json`. If `npm ls next` shows Next 16, run `npm.cmd ci` from `frontend`.

## Current Git Status To Expect

Expected modified files from the latest session:

- `.gitignore`
- `frontend/next.config.ts`
- `frontend/src/hooks/useJobStatus.ts`

Expected untracked file:

- `frontend/.env.local.example`

Generated files under `.tmp/`, `.next/`, and `frontend/tsconfig.tsbuildinfo` should be ignored.

## Suggested Next Steps

1. Install or expose Docker Desktop/CLI if full Postgres + Redis + MinIO verification is required.
2. Add a frontend regression test for the SSE hook behavior, especially that `done` does not become `job stream unavailable`.
3. Move Messenger messages from in-memory mock state to database-backed persistence.
4. Connect Messenger actions to Review/Knowledge workflows.

## 2026-05-01 Korean I18n and Messenger Update

Added a Korean-first UX pass and Slack-like mock Messenger MVP.

- Spec: `docs/superpowers/specs/2026-05-01-korean-i18n-messenger-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-korean-i18n-messenger.md`
- Backend API:
  - `GET /api/v1/messages/channels`
  - `GET /api/v1/messages/channels/{channel_id}/messages`
  - `POST /api/v1/messages/channels/{channel_id}/messages`
- Frontend:
  - Korean default shell labels.
  - Korean/English language switch in desktop sidebar and mobile header.
  - New `/messages` screen with channels, message timeline, and composer.
  - Existing dashboard, integrations, review, search chrome converted to Korean-first copy.

Verification:

```powershell
uv run pytest backend/tests -v
cd frontend
npm.cmd run build
```

Result: backend 22 tests passed; frontend build passed.

Browser smoke covered:

- Open `/messages`.
- Verify Korean default labels and Korean business channel seed data.
- Switch to English with the mobile `EN` control.
- Post a message and see it appended to the current channel.

## 2026-05-01 SQLite Smoke Mode Update

Added a Docker-free smoke mode for quick product review and browser testing.

- Runbook: `docs/superpowers/runbooks/sqlite-smoke.md`
- Script: `scripts/start-smoke.ps1`
- Updated:
  - `docs/superpowers/runbooks/local-dev.md`
  - `docs/superpowers/runbooks/verification.md`

Use:

```powershell
.\scripts\start-smoke.ps1
```

This initializes `.tmp/paraworks-smoke.db`, starts FastAPI on
`http://127.0.0.1:8000`, and starts Next.js on `http://127.0.0.1:3000`.

## 2026-05-01 Messenger Persistence Update

Moved Messenger from process memory to SQLAlchemy-backed persistence.

- Model: `backend/app/models/messages.py`
- Service: `backend/app/messages/service.py`
- Test: `backend/tests/test_messages.py`

Tables:

- `message_channels`
- `messages`

The message service seeds the three demo channels and their initial messages
when the first message endpoint is called against an empty database. Posted
messages are inserted into `messages`, so they survive page reloads and remain
available while the same SQLite/Postgres database is used.

Verification:

```powershell
uv run pytest backend/tests -v
cd frontend
npm.cmd run build
```

Result after this update: backend 23 tests passed; frontend build passed.

## 2026-05-01 Messenger to Review Queue Update

Connected Messenger to the Review workflow.

- API: `POST /api/v1/messages/messages/{message_id}/send-to-review`
- UI: `/messages` now shows `검토 큐로 보내기` on each message.
- Created review items use:
  - `item_type="message_review"`
  - `payload.title="메신저 검토 요청"`
  - `source_links=["paraworks://messages/{message_id}"]`
  - `source_snippets=[message.body]`

Verification:

```powershell
uv run pytest backend/tests -v
cd frontend
npm.cmd run build
```

Result after this update: backend 25 tests passed; frontend build passed.

Browser smoke covered:

- Open `/messages`.
- Click `검토 큐로 보내기`.
- See `검토 큐에 추가했습니다.`
- Open `/review`.
- Confirm `메신저 검토 요청` appears in the review queue.

## 2026-05-01 Slack Connector Preparation Update

Added a testable real Slack connector boundary without making live Slack API
calls.

- Connector: `backend/app/connectors/slack.py`
- Test: `backend/tests/test_slack_connector.py`
- Runbook: `docs/superpowers/runbooks/slack-integration.md`
- Environment placeholders added to `.env.example`:
  - `SLACK_BOT_TOKEN`
  - `SLACK_CHANNEL_IDS`
  - `SLACK_WORKSPACE_URL`

The connector maps Slack `conversations.history` message payloads into
ParaWorks `SourceEvent` records and records required history scopes in
`raw_metadata`.

The next Slack step is to implement a real Web API client behind the
`SlackApiClient` protocol with cursor pagination and rate-limit handling.

## 2026-05-02 Source Evidence Review Drawer Update

Aligned with the current root `plan.md` Milestone 3.

- Review API responses now include `source_evidence` rows for each ReviewItem.
- Evidence rows expose source URL, snippet, permission level, confidence,
  rank, importance score, source id, author/timestamp when available, and
  originating AgentRun id.
- `/review` now passes structured evidence into the shared
  `SourceEvidenceDrawer`.
- The Drawer shows reviewer-ready evidence cards and links to the originating
  AgentRun when available.
- The "request more evidence" action now opens a reviewer note field and sends
  the note to the backend before moving the item to `needs_more_evidence`.

Next recommended step from `plan.md`:

1. Add quality and permission regression coverage for Review Queue, RAG, and
   connector evidence.
2. Expand Track A and Track B evidence metadata so more Drawer rows have rank,
   author, timestamp, and source ids.

## 2026-05-02 LangGraph HITL Checkpoint Strategy Update

Aligned with the current root `plan.md` Milestone 4.

- Company Memory orchestration now emits `hitl_checkpoint` from
  `draft_review_candidates`.
- The checkpoint records `checkpoint_type=review_queue`, target ReviewItem ids,
  required statuses, `resume_from_node=retrieve_company_memory`, and
  `resume_policy=resume_after_review_queue_resolution`.
- Orchestration status APIs now expose `hitl_checkpointing`,
  `checkpoint_store=review_queue`, and
  `trusted_knowledge_requires_approval` in the cost/trust policy.

Next recommended step from `plan.md`:

1. Expand Track A and Track B evidence metadata so more Drawer rows have rank,
   author, timestamp, and source ids.
2. Continue connector quality hardening for Slack/Gmail/Drive/Calendar.

## 2026-05-02 Quality And Permission Regression Suite Update

Aligned with the current root `plan.md` Milestone 6.

- Added `backend/tests/test_quality_permission_regression_suite.py`.
- The suite covers source-less review approval rejection, restricted RAG hidden
  match reporting without snippet/citation leakage, HITL checkpoint metadata,
  and cache-hit dedupe for AgentRun/ReviewItem records.
- The suite uses deterministic local fixtures and does not call live Slack,
  Google, LLM, embedding, or external APIs.

Next recommended step from `plan.md`:

1. Continue connector quality hardening for Slack/Gmail/Drive/Calendar.
2. Add connector-specific golden dataset fixtures.

## 2026-05-02 Cross-Agent Evidence Summary Update

Aligned with the current root `plan.md` Track C next-priority cleanup.

- Added `backend/app/agent_runtime/evidence_summary.py`.
- Mail/Document Agent bridge now stores `evidence_summary` in AgentRun metadata.
- Track C Timeline/History/Decision/Todo extraction runs now store
  `evidence_summary` in AgentRun metadata.
- The metadata includes rank, source id, source URL, source type, timestamp,
  author, permission level, importance score, and snippet.

Next recommended step from `plan.md`:

1. Continue connector quality hardening for Slack/Gmail/Drive/Calendar.
2. Add connector-specific golden dataset fixtures.
3. Prepare structured LangChain output adapters behind the deterministic agent
   contracts.

## 2026-05-02 Search Retrieval Backend Alignment Update

Checked `/search` retrieval behavior before continuing connector hardening.

- `/search` page calls both `/api/v1/ask` and `/api/v1/search`.
- `/api/v1/ask` could already use pgvector behind
  `RAG_USE_PGVECTOR_SEARCH=true`, PostgreSQL, and OpenAI embedding key.
- `/api/v1/search` previously always used deterministic lexical ranking.
- Added `backend/app/rag/search_store.py` so both Ask and Search can share the
  pgvector search adapter builder.
- `/api/v1/search` now returns `retrieval_backend` and `cost_policy`.
- The Search UI now shows whether the evidence list used pgvector or the
  default deterministic zero-cost path.

Cost note:

- Default local/demo search still has `embedding_query_call=false`.
- pgvector search requires the feature flag and will make one query embedding
  call when enabled in a PostgreSQL environment with `OPENAI_API_KEY`.

Next recommended step from `plan.md`:

1. Continue connector quality hardening for Slack/Gmail/Drive/Calendar.
2. Add connector-specific golden dataset fixtures.
3. Prepare structured LangChain output adapters behind the deterministic agent
   contracts.

## 2026-05-02 Slack Thread Context Chunking Update

Aligned with the current root `plan.md` Milestone 5.

- Slack connector reply SourceEvents now set body to:
  `Thread parent: <parent text>\nThread reply: <reply text>`.
- Reply metadata now includes `thread_parent_text`, `thread_reply_index`, and
  `thread_context_window=parent_plus_reply`.
- Parent messages remain single-message chunks.
- This improves downstream Review/RAG quality without extra LLM or embedding
  calls.

Next recommended step from `plan.md`:

1. Harden Drive parser/version metadata.
2. Add connector-specific golden dataset fixtures.

## 2026-05-02 Gmail Thread Domain Metadata Update

Aligned with the current root `plan.md` Milestone 5.

- Gmail SourceEvents now parse From, To, and Cc header addresses into
  participants.
- Gmail raw metadata now includes `thread_context_key`, `from_domain`,
  `participant_domains`, `external_domains`, and
  `has_external_participants`.
- This is zero-cost local preprocessing over already fetched Gmail payloads.

Next recommended step from `plan.md`:

1. Harden Drive parser/version metadata.
2. Add connector-specific golden dataset fixtures.
3. Prepare structured LangChain output adapters behind deterministic contracts.

## 2026-05-02 Drive Parser Version Metadata Update

Aligned with the current root `plan.md` Milestone 5.

- Drive files list collection now requests `version` and `headRevisionId`.
- Drive SourceEvents now record `parser_name=google_drive_metadata`,
  `parser_status=metadata_only`,
  `parser_status_reason=content_export_not_enabled`, `document_version`,
  `revision_id`, and `content_signature`.
- This is a low-cost metadata hardening step before full Drive file export and
  parser-specific chunking are added.

Next recommended step from `plan.md`:

1. Harden Calendar connector quality metadata.
2. Add connector-specific golden dataset fixtures.
3. Prepare structured LangChain output adapters behind deterministic contracts.

## 2026-05-02 Calendar Event Quality Metadata Update

Aligned with the current root `plan.md` Milestone 5.

- Calendar SourceEvents now include `event_context_key`, `event_status`,
  `organizer_email`, `creator_email`, `recurring_event_id`,
  `attendee_response_statuses`, `attendee_domains`, `external_domains`,
  `has_external_attendees`, and `duration_minutes`.
- The implementation derives these values locally from already fetched
  Calendar event payloads.
- Milestone 5 connector quality hardening is now complete for Slack, Gmail,
  Drive, and Calendar.

Next recommended step from `plan.md`:

1. Add connector-specific golden dataset fixtures.
2. Add RAG precision/recall smoke metrics.
3. Prepare structured LangChain output adapters behind deterministic contracts.

## 2026-05-02 Connector Golden Dataset Update

Aligned with the current root `plan.md` Milestone 6.

- Added `backend/tests/fixtures/connector_golden_payloads.json` with static
  Slack, Gmail, Drive, and Calendar payloads.
- Added `backend/tests/test_connector_golden_dataset.py`.
- The test verifies agent-ready evidence metadata across connectors:
  Slack thread context, Gmail external-domain flags, Drive parser/version
  signatures, and Calendar RSVP/duration/external-domain metadata.
- The suite is deterministic and makes no live SaaS, LLM, or embedding calls.

Next recommended step from `plan.md`:

1. Add RAG precision/recall smoke metrics.
2. Prepare structured LangChain output adapters behind deterministic contracts.
3. Continue product completion pages after evaluation hooks are stable.

## 2026-05-02 RAG Retrieval Smoke Metrics Update

Aligned with the current root `plan.md` Milestone 6.

- Added `backend/app/rag/evaluation.py`.
- Added `backend/tests/fixtures/rag_smoke_eval_cases.json`.
- Added `backend/tests/test_rag_evaluation_metrics.py`.
- Metrics include precision@k, recall@k, hit rate, expected/retrieved counts,
  and matched expected source ids.
- The smoke fixture runs deterministic retrieval over local seeded chunks and
  confirms expected source ids are recovered.

Cost note:

- This evaluation path uses local fixtures only. It does not call paid LLMs,
  embedding APIs, Slack, or Google.

Next recommended step from `plan.md`:

1. Prepare structured LangChain output adapters behind deterministic contracts.
2. Add product completion pages for decisions/timeline/history.
3. Add production auth plan after product surfaces stabilize.

## 2026-05-02 Structured Memory Extraction Adapter Update

Aligned with the current root `plan.md` Milestone 6.

- Added `backend/app/agents/memory_extraction_agent/langchain_adapter.py`.
- The adapter implements the existing `MemoryExtractionModel` contract and
  returns `MemoryExtractionModelResponse`.
- It uses `chat_model.with_structured_output(StructuredMemoryExtractionOutput)`
  so real LangChain providers can be injected later without changing Track C
  agent contracts.
- Prompt rendering includes bounded evidence rows with source id, source URL,
  timestamp, author, permission level, and text.

Cost note:

- No provider builder or live model call is enabled by default.
- Tests use fake chat models only.
- Evidence rendering is bounded by `max_input_chars`.

Next recommended step from `plan.md`:

1. Add product completion pages for decisions/timeline/history.
2. Add production auth plan after product surfaces stabilize.
3. Keep expanding golden fixtures as new real-data failures appear.

## 2026-05-02 Product Memory Pages Update

Aligned with the current root `plan.md` Milestone 7.

- `/api/v1/knowledge` now includes `timeline_events` and a
  `counts.timeline_events` value.
- `/knowledge` is now an approved company-memory overview.
- Added `/decisions`, `/timeline`, and `/history` pages.
- Added `frontend/src/components/knowledge/MemoryCollection.tsx` for shared
  glass-card memory rendering.
- Extended frontend route inventory and clean-render Playwright coverage for
  the new pages.

Cost note:

- These pages are read-only and do not trigger paid LLM calls, embedding calls,
  provider sync, or reindex jobs.

Next recommended step from `plan.md`:

1. Add production auth plan: httpOnly cookie + refresh token.
2. Add deployment runbook.
3. Consider Notifications/Knowledge Map only after auth/deploy boundaries are
   documented.

## 2026-05-02 Production Auth Plan Update

Aligned with the current root `plan.md` Milestone 7.

- Added `docs/superpowers/runbooks/production-auth.md`.
- The plan moves ParaWorks from demo `X-Demo-User` headers to httpOnly cookie
  sessions with rotating refresh tokens.
- It covers backend auth tables/endpoints, frontend `credentials: "include"`,
  RBAC, source permissions, CSRF, rate limits, audit logs, demo-mode fallback,
  and migration order.

Cost note:

- Auth must not trigger LLM calls, embedding calls, connector sync, or RAG
  reindexing.

Next recommended step from `plan.md`:

1. Add deployment runbook.
2. Then revisit whether Notifications or Knowledge Map are worth building for
   the portfolio demo.

## 2026-05-02 Deployment Runbook Update

Aligned with the current root `plan.md` Milestone 7.

- Added `docs/superpowers/runbooks/deployment.md`.
- The runbook covers Next.js, FastAPI, PostgreSQL + pgvector, Redis, Celery,
  Slack/Google OAuth, environment variables, deployment order, verification,
  cost gates, rollback, monitoring, and production readiness.

Cost note:

- Production verification keeps paid LLM and embedding actions behind explicit
  dry-run or confirmation gates.

Next recommended step from `plan.md`:

1. Add Notifications only if they directly support Review Queue or agent-run
   workflow visibility.
2. Add Knowledge Map only if there is enough time after core product polish.

## 2026-05-02 Notifications Update

Aligned with the current root `plan.md` Milestone 7.

- Added `/api/v1/notifications`.
- The endpoint derives alerts from pending Review Queue items,
  `needs_more_evidence` items, and recent non-complete AgentRuns.
- Added `/notifications` frontend page and sidebar navigation.
- Added Playwright route inventory and render coverage for the page.

Cost note:

- Notifications are read-only database summaries and do not trigger paid LLMs,
  embeddings, provider sync, or RAG reindexing.

Next recommended step from `plan.md`:

1. Add Knowledge Map only if it can be useful without distracting from the core
   Review/RAG story.
2. Otherwise spend the next pass on frontend consistency and final portfolio
   polish.

## 2026-05-03 Knowledge Map Update

Aligned with the current root `plan.md` Milestone 7.

- Added `/api/v1/knowledge/map`.
- The endpoint derives memory nodes from approved Decision, Timeline, History,
  and Todo records, then connects them to source-evidence nodes through stored
  source links.
- Evidence source nodes inherit the strictest connected permission level so the
  map does not make restricted evidence look broadly shareable.
- Added `/knowledge-map`, sidebar navigation, Knowledge Library cross-link, and
  Playwright route inventory coverage.

Cost note:

- Knowledge Map is read-only database aggregation. It does not call LLMs,
  embeddings, connector sync, or reindex jobs.

Next recommended step from `plan.md`:

1. Frontend global consistency and final Liquid Glass polish across all pages.
2. Production auth implementation from `docs/superpowers/runbooks/production-auth.md`.
3. Final demo script and portfolio evidence capture.

## 2026-05-03 Production Auth Cookie Slice

Aligned with the current root `plan.md` Milestone 8.

- Added `AuthUser` and `RefreshToken` models.
- Login now upserts the selected demo account into `auth_users`, stores only a
  hashed refresh token, and sets httpOnly `paraworks_session` and
  `paraworks_refresh` cookies.
- `/api/v1/auth/me` now prefers the signed session cookie over `X-Demo-User`.
- Demo mode still falls back to `X-Demo-User`; production mode rejects requests
  without a valid session cookie.
- Added `/api/v1/auth/refresh` for refresh-token rotation and
  `/api/v1/auth/logout` for refresh-family revocation and cookie clearing.
- Frontend `apiGet`, `apiPost`, and `apiPatch` now send
  `credentials: "include"`.

Cost note:

- Auth remains isolated from paid model, embedding, sync, and reindex paths.

Next recommended step from `plan.md`:

1. Continue frontend global consistency polish where pages still use legacy
   fixed-color alert/card classes.
2. Run final screenshot capture for the portfolio case study.
3. Add Alembic migrations, CSRF, and rate limiting if moving auth closer to
   production deployment.

## 2026-05-03 Portfolio Demo Script Update

Aligned with the current root `plan.md` Milestone 8.

- Added `docs/superpowers/runbooks/portfolio-demo-script.md`.
- The script covers login, integrations, AgentRun observability, Review Queue,
  approved knowledge pages, Knowledge Map, permission-aware RAG, and final
  portfolio close.
- It includes cost and security language for recording or presenting the
  project.

Next recommended step from `plan.md`:

1. Capture final portfolio screenshots or short clips.
2. Add production hardening details that remain outside the current harness:
   Alembic migrations, CSRF, rate limiting, and real identity verification.
3. Keep whole-app Playwright regression green after any frontend polish.

## 2026-05-03 Azure OpenAI-Compatible Alias Update

Aligned with the current root `plan.md` Milestone 8 Azure staging preparation.

- Added `docs/superpowers/specs/2026-05-03-azure-integration-design.md`.
- Added `azure_openai` as a valid Slack LLM provider alias.
- The current alias intentionally uses `OPENAI_API_KEY`,
  `AGENT_LLM_OPENAI_MODEL`, and the existing OpenAI-compatible ChatOpenAI path.
- Added `openai_compatible_embedding_config`, which accepts `azure_openai` but
  still defaults to `https://api.openai.com/v1` for this first key-swap slice.
- Updated `docs/superpowers/runbooks/deployment.md` with Azure Container Apps,
  PostgreSQL pgvector, Redis, Key Vault, Managed Identity, and the alias
  boundary.

Usage:

```text
AGENT_LLM_PROVIDER_ORDER=azure_openai,openai,gemini
OPENAI_API_KEY=<openai-compatible-key>
```

Important:

- This is not yet true Azure OpenAI endpoint/deployment mode. Future work should
  add endpoint, API version, and deployment-name settings behind the same
  `azure_openai` provider name.
- Do not create Azure resources or commit keys without user confirmation on
  budget, region, resource group, and staging domain.

## 2026-05-03 Google Identity and RBAC Update

Aligned with `docs/superpowers/specs/2026-05-03-google-identity-rbac-design.md`.

- Added `docs/superpowers/plans/2026-05-03-google-identity-rbac.md`.
- Google identity login now has a separate login URL and callback path:
  - `GET /api/v1/auth/google/login-url`
  - `GET /api/v1/auth/google/callback`
- Google identity login uses `openid email profile` and `prompt=select_account`.
- Gmail, Drive, and Calendar OAuth remain separate data integration flows.
- Added seeded accounts:
  - `hanvv3@gmail.com`: admin, `public/internal/restricted`
  - `hanvv3@koreacu.ac.kr`: employee, `public/internal`
  - `mina@paraworks.com`: reviewer, `public/internal`
- Added admin user management:
  - `GET /api/v1/admin/users`
  - `PATCH /api/v1/admin/users/{external_id}`
- Admin UI can change role, status, and permission levels, and changes create
  audit logs.
- Review Queue approval is now role-aware:
  - reviewer: `public/internal`
  - manager/admin: `public/internal/restricted`
- Frontend navigation now hides admin/integrations/agent-runs from non-admin
  users and hides Review Queue from users below reviewer.
- Cost and operations APIs are now backend admin-only:
  - `/api/v1/agent-runs`
  - `/api/v1/agent-runs/summary`
  - `/api/v1/agent-runs/{run_id}`
  - `/api/v1/rag/reindex`
  - `/api/v1/rag/reindex/jobs`
  - `/api/v1/rag/reindex/jobs/{job_id}`
  - `/api/v1/rag/indexing/summary`
- Direct AgentRun pages render an admin-required state instead of a 500 when
  the active context cannot access admin observability data.
- Google identity readiness is visible from `/api/v1/auth/google/login-url`
  and `/login`:
  - `redirect_uri`
  - `missing_config`
  - `configured`
- `configured=true` now requires client id, client secret, identity redirect
  URI, and identity state secret.

Environment:

```text
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_IDENTITY_REDIRECT_URI=http://localhost:3000/login/google/callback
GOOGLE_IDENTITY_STATE_SECRET=replace-with-local-google-login-state-secret
```

Cost note:

- Identity login, RBAC checks, admin user management, and Review Queue role
  checks do not call paid LLMs, embeddings, sync jobs, or reindex jobs.

## Portfolio Recording Rule

When future ParaWorks work changes the product story, architecture, UX, testing
evidence, or demo flow, update `docs/portfolio-log.md` in the same session.
Write entries so they can later be reused for a portfolio case study: problem,
implementation, verification evidence, and portfolio angle.

## 2026-05-11 Sidebar Navigation Update

- Sidebar now foregrounds `대시보드`, `프로젝트`, `검토사항`, and `타임라인`.
- Removed Decision, History, and Knowledge Map from the sidebar navigation.
  Their routes still exist for now, but they are no longer primary menu items.
- Added `/projects` as a frontend project workspace with a top project switcher,
  Gantt-style planning, calendar scheduling, board status, and task list views.
- Reworked `/timeline` as a project-scoped timeline. Each timeline item has a
  history summary and a history/Slack icon path that opens the related source
  conversation panel.
- The global top search submits through the left search icon and routes to
  `/search?q=...`, which drives `AI 비서`.
- Navigation and `/search` now label the assistant surface as `AI 비서`.
- `/dashboard` is now a personalized work-home view: today's assigned tasks,
  personal review count, meetings, mentions/updates, assigned projects, and an
  AI 비서 suggestion. Workspace-wide source counts were removed from Dashboard.
- `/dashboard` includes a visible `검토사항` section for assigned review work.
- `/review` is titled `검토사항` and keeps demo fallback review items visible if
  the backend cannot return pending Review Queue data.
- `/timeline` starts full-width and opens the history/source panel only after a
  history icon click; closing the panel returns the timeline to full width.
- `/integrations` now includes the source-by-connector collection status panel,
  since source health and sync volume are connector operations context.
- Verification: `cd frontend && npm run build` passed.

## 2026-05-12 Mail/Document/Calendar Project Grouping Update

- Added `docs/mail-doc-calendar-agent-status.md` as the Developer B status and
  remaining-work document for Google Drive, Gmail, and Calendar project memory.
- Mail/Document Agent evidence packet now includes Calendar chunks in addition
  to Gmail, Gmail attachment, and Drive chunks.
- Calendar event metadata is preserved into the agent evidence packet:
  `event_context_key`, `event_status`, organizer/creator, attendee metadata,
  external domains, and duration.
- Added `GET /api/v1/projects`.
  - Groups Gmail/Drive/Calendar evidence by explicit `project_key`, then
    `scenario`, then URL/title/source-id fallback.
  - Returns project summary, source types, evidence count, strictest permission,
    latest timestamp, and source evidence rows.
  - Hides projects whose strictest permission is outside the current user's
    permission levels and returns `hidden_project_count`.
- Backend test fixtures now attach matching CSRF cookie/header values to unsafe
  requests and clear in-memory auth rate-limit state between tests.
- Verification:

```bash
uv run pytest backend/tests -v
```

Result: 287 passed, 1 skipped.

## 2026-05-12 Local Docker Auth and CSRF Update

- Local production-like Docker mode now seeds auth users and pending Review
  Queue evidence through `backend.app.db.init_db` when `PARAWORKS_ENV=local`.
- `PARAWORKS_DEMO_MODE=false` no longer leaves local email login unusable in
  local development: seeded emails can issue real httpOnly session, refresh,
  and CSRF cookies.
- The login page no longer redirects to `/dashboard` after a failed backend
  login by storing only a local demo account id. AppShell also no longer treats
  localStorage as authenticated state when `/api/v1/auth/me` fails.
- Root cause fixed for the observed symptoms:
  - fake localStorage login made the UI enter the app without backend cookies;
  - unsafe POST routes such as `/api/v1/ask` then failed CSRF validation;
  - admin-only pages saw the user as unauthenticated/non-admin;
  - fresh Docker DBs had no seeded Review Queue items.
- Verification:

```powershell
uv run pytest backend/tests -q
cd frontend
npm.cmd run build
```

Result: backend 289 passed, 1 skipped; frontend build passed.

Direct Docker-backed API check on a secondary backend port confirmed:

- `admin@paraworks.com` and `hanvv3@gmail.com` login return role `admin`;
- `/api/v1/agent-runs` and `/api/v1/admin/users` return 200 for those sessions;
- `/api/v1/review?status=pending_review` returns seeded review items;
- `/api/v1/ask` returns 200 when the `paraworks_csrf` cookie is echoed in
  `X-CSRF-Token`.

## 2026-05-14 Scoped Sync-Driven Agent Review Update

- `/integrations` no longer exposes separate generic Agent execution buttons
  for Slack/Gmail/Drive. The user-facing sync button is now the single path:
  sync fetches changed Source/DocumentChunk rows, then runs only the matching
  connector review agent for changed source ids.
- Duplicate sync is handled at the ingestion contract boundary by returning
  `changed_source_ids=[]` when the source content signature is unchanged. This
  prevents repeat Agent Review cost without splitting sync and Agent execution
  into two user actions.
- Slack review extraction and Mail/Document review extraction can now scope
  evidence packets by explicit `source_ids`, so Gmail sync does not process
  Drive data and Drive sync does not process Gmail data.
- Connector factories now fail loudly when an installed Slack/Google OAuth
  connection exists but its local token is missing, instead of silently falling
  back to demo/config behavior.
- AI 비서 now uses the low-cost email action sub-agent as a routing layer with
  configurable confidence gating (`assistant_email_agent_min_confidence=0.72`).
  High-confidence email drafts and lightweight general replies skip expensive
  RAG; low-confidence decisions fall back to the existing RAG orchestrator.
- Verification:

```powershell
uv run pytest backend/tests/test_mail_document_agent_api.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_assistant_api.py backend/tests/test_connector_factory.py backend/tests/test_connector_ingestion_contract.py -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_models.py backend/tests/test_assistant_service.py -q
uv run pytest backend/tests/test_integration_runtime_status.py backend/tests/test_review.py backend/tests/test_dashboard_api.py -q
uv run ruff check backend/app/ingestion/service.py backend/app/ingestion/sync.py backend/app/agents/mail_document_agent/service.py backend/app/agents/slack_agent/service.py backend/app/api/v1/integrations.py backend/app/connectors/factory.py backend/app/core/config.py backend/app/assistant/email_agent.py backend/app/api/v1/assistant.py backend/tests/test_mail_document_agent_api.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_assistant_api.py backend/tests/test_connector_factory.py backend/tests/test_connector_ingestion_contract.py
cd frontend
npm.cmd run lint
npm.cmd run build
```

Result: targeted backend tests passed (`46`, `29`, and `17` tests); ruff,
frontend lint, and frontend production build passed.

Residual note:

- A full `uv run pytest backend/tests -q` run still has unrelated pre-existing
  failures around Slack OAuth PKCE expectations, Slack connector fake-client
  contracts, and RAG indexing tests that still expect all chunks to index
  without approved ReviewItem source ids. The sync/assistant tests listed above
  are green after this change.

## 2026-05-14 Developer B Drive/Gmail Review Fix

- Google Drive sync now runs the Mail/Document Agent per changed Drive source
  instead of sending every changed Drive file in one evidence packet. This
  prevents multiple synced documents from being collapsed into a single Review
  Queue candidate.
- Gmail sync still groups a message and its changed attachments together, so
  attachment evidence keeps the parent email context without mixing unrelated
  emails.
- Gmail live fetch now sends a business-focused Gmail search query by default:
  `newer_than:90d` plus spam/trash/social/promotions/forums exclusions. Delta
  sync keeps the `after:<cursor>` constraint and applies the same exclusions.
- Gmail message SourceEvents now include a `content_signature` based on the
  message id and `internalDate`, so the ingestion boundary has an explicit
  dedupe/update signal instead of treating every existing Gmail message as
  same-content by fallback.
- Verification:

```powershell
uv run pytest backend/tests/test_mail_document_agent_api.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_google_connector.py backend/tests/test_connector_ingestion_contract.py backend/tests/test_connector_factory.py backend/tests/test_integration_runtime_status.py -q
uv run ruff check backend/app/agents/mail_document_agent/service.py backend/app/agents/mail_document_agent/__init__.py backend/app/api/v1/integrations.py backend/app/connectors/google.py backend/tests/test_mail_document_agent_api.py backend/tests/test_google_connector.py
```

Result: 63 targeted backend tests passed; ruff passed.

## 2026-05-14 Mail/Document Operating MVP Hardening

- Mail/Document evidence now filters `DocumentChunk.permission_level` through
  `PermissionContext.allowed_permission_levels`, and integrations/orchestration
  pass the current user's permission levels explicitly.
- Manual `/mail-docs/agent-review` and company-memory orchestration now create
  grouped ReviewItems instead of one all-corpus item. Gmail attachments stay
  grouped with their parent email; Drive/Calendar sources stay source-local.
- Mail/Docs has Slack-style live LLM boundaries:
  `GET /api/v1/integrations/mail-docs/agent-review/llm/preflight` and
  `POST /api/v1/integrations/mail-docs/agent-review/llm` with
  `confirm_paid_run=true`. Connector sync still uses deterministic review
  generation and does not auto-trigger paid LLM calls.
- Review rejection preserves linked `Source` and `DocumentChunk` rows. Audit
  metadata records `source_ids_preserved` and `rejected_review_item_id`.
- RAG indexing now ignores malformed approved `payload.source_ids` unless it is
  a `list[str]`, and approved `TimelineEvent` rows are indexed as trusted
  knowledge documents.
- Observability follows the Slack pattern: no new `*_LOG_PATH`/`*_LOG_FILE`
  settings. Mail/Docs live runs store `source_window`, evidence counts,
  included source types, parser status counts, selection strategy, and
  preflight data in `AgentRun.metadata_`, `AuditLog.metadata_`, and API
  responses. Legacy Slack sync now uses a module logger instead of `print()`.
- Verification:

```powershell
uv run pytest backend/tests/test_mail_document_agent.py backend/tests/test_mail_document_agent_review_bridge.py backend/tests/test_mail_document_agent_api.py backend/tests/test_review.py backend/tests/test_rag_indexing.py backend/tests/test_company_memory_orchestration_service.py -q
uv run ruff check backend/app/agents/mail_document_agent backend/app/agents/slack_agent/sync_service.py backend/app/api/v1/integrations.py backend/app/api/v1/review.py backend/app/rag/indexing.py
cd frontend
npm.cmd exec tsc -- --noEmit
npm.cmd run build
```

Result: `63 passed`, ruff passed, frontend TypeScript check and production
build passed.

## 2026-05-14 Slack sync와 agent_slack LLM 파이프라인 연결

- 사용자가 Slack sync 후 `review_items`에 `Redis 큐 관련 결정사항 추출됨` 1건만
  생성된다고 보고했다.
- 확인 결과 해당 항목은 실제 LLM 결과가 아니라
  `DeterministicSlackAgentModel`의 결정론/fake 결과였다.
- `backend/app/agents/slack_agent/sync_service.py`에는 이미
  `agent_slack.process_daily_slack_sync()` 결과를 `slack_agent_v2` AgentRun과
  ReviewItem으로 저장하는 `trigger_slack_agent_analysis()`가 있었다.
- 이번 변경으로 `/api/v1/integrations/slack/sync`가 운영형 local/prod 모드와
  provider key가 있는 경우 위 `agent_slack` LLM 파이프라인을 호출한다.
- `trigger_slack_agent_analysis()`는 이제 `source_ids`를 받을 수 있다.
  Slack sync에서 방금 변경된 source만 넘기므로 최근 7일 전체 재분석과 중복 비용을
  피한다.
- demo/test 모드 또는 provider key가 없는 환경에서는 기존 결정론 스모크 경로를
  유지한다. 자동 테스트가 live LLM API를 호출하지 않게 하기 위한 경계다.
- 관련 검증:

```powershell
uv run pytest backend/tests/test_slack_agent_api.py::test_slack_sync_uses_agent_slack_llm_pipeline_when_provider_key_exists -q
uv run pytest backend/tests/test_slack_agent_api.py backend/tests/test_mock_sync.py backend/tests/test_integration_runtime_status.py::test_sync_returns_configuration_error_when_connector_is_not_configured -q
uv run ruff check backend/app/api/v1/integrations.py backend/app/agents/slack_agent/sync_service.py backend/tests/test_slack_agent_api.py
uv run pytest backend/tests/test_slack_agent_review_bridge.py backend/tests/test_slack_agent.py backend/tests/test_slack_agent_api.py backend/tests/test_mock_sync.py -q
```

Result: `1 passed`, `8 passed`, ruff passed, `23 passed`.

## 2026-05-14 AI Assistant Model and Tool Logging

- AI Assistant RAG answer generation now uses `AGENT_LLM_OPENAI_PRIMARY_MODEL`
  as the primary OpenAI model. The default primary model is `gpt-5.4`, while
  `AGENT_LLM_OPENAI_MODEL` remains the OpenAI fallback and defaults to
  `gpt-5.4-mini`.
- Assistant tool logs now follow the Slack Agent pattern and use the Python
  `AssistantTool` logger instead of opening a path from `.env`; local/docker
  runs still expose the lines through the backend stderr log redirection.
- Assistant message creation now logs email action routing, RAG retrieval, and
  RAG answer generation in English with the format
  `[Tool: tool_name] ...description...`.
- The log intentionally sanitizes non-ASCII characters before writing so Korean
  user input or model output does not become mojibake inside the tool trace.
- Verification:

```powershell
uv run pytest backend/tests/test_rag_orchestrator_service.py::test_rag_service_uses_configured_stronger_primary_model backend/tests/test_assistant_api.py::test_assistant_tool_middleware_logs_email_and_rag_tools_in_english -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_rag_orchestrator_agent.py -q
uv run ruff check backend/app/api/v1/assistant.py backend/app/assistant/tool_logging.py backend/app/core/config.py backend/app/agents/rag_orchestrator_agent/service.py backend/app/agents/rag_orchestrator_agent/llm.py backend/tests/test_assistant_api.py backend/tests/test_rag_orchestrator_service.py
```

Result: targeted RED tests failed before implementation, then passed after the
change; wider assistant/RAG tests passed with 40 tests; ruff passed.

## 2026-05-14 Email Intent Gate and Draft Composer Split

- The AI Assistant email path is now split into single-purpose sub-agents:
  `email_intent_gate` only decides whether the latest user message is an email
  action, and `email_draft_composer` only writes the approval-only draft or a
  clarification question after intent is accepted.
- The old combined email prompt that also classified general replies and RAG was
  removed from the active path. Non-email messages fall through to the normal
  RAG answer path.
- `EmailIntentDecision.requires_rag_result` allows a flow such as "find this in
  company memory and email it": assistant orchestration runs RAG first, renders
  the RAG answer/source context, then passes that context to the draft composer.
- Tool logs now show the split route with `[Tool: email_intent_gate]`,
  `[Tool: rag_retrieval]`, `[Tool: rag_answer]`, and
  `[Tool: email_draft_composer]`.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_api.py::test_assistant_tool_middleware_logs_email_and_rag_tools_in_english backend/tests/test_assistant_api.py::test_assistant_can_draft_email_from_rag_answer -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_rag_orchestrator_agent.py -q
uv run ruff check backend/app/api/v1/assistant.py backend/app/assistant/email_agent.py backend/app/assistant/email_actions.py backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py
```

Result: targeted RED tests failed before implementation, then passed after the
change; wider assistant/RAG tests passed with 43 tests; ruff passed.

## 2026-05-14 Email Continuation Context and Docker Startup Order

- Fixed the AI Assistant email draft path where recipient-only follow-ups such
  as `kjw4work@gmail.com` caused the draft composer to ask for content again.
- `render_email_action_context()` now preserves complete JSON message rows
  instead of slicing the serialized JSON mid-string. This keeps recent user and
  assistant messages readable for the low-cost email sub-agents.
- Added `render_recent_assistant_context_for_email()` so the draft composer
  receives recent assistant answers as explicit body-source context for phrases
  like "이 내용으로" or "최근 결정된 사항만 요약해서 보내줘".
- `scripts/paraworks-docker.ps1` now waits for backend `/health` before starting
  the frontend, avoiding transient frontend `ECONNREFUSED 127.0.0.1:8000`
  startup noise.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_rag_orchestrator_service.py backend/tests/test_rag_orchestrator_agent.py backend/tests/test_paraworks_docker_script.py -q
uv run ruff check backend/app/api/v1/assistant.py backend/app/assistant/email_agent.py backend/app/assistant/email_actions.py backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_paraworks_docker_script.py
```

Result: 47 tests passed; ruff passed; PowerShell parser check for
`scripts/paraworks-docker.ps1` passed.

## 2026-05-14 Docker Startup Guardrails

- Hardened `scripts/paraworks-docker.ps1` so native command failures from
  Docker, Alembic, and schema checks stop the script immediately instead of
  continuing to a misleading final `Ready` state.
- Added a Postgres readiness wait between `docker compose up -d` and the
  pgvector schema check. This removes the transient first-run connection error
  where Postgres had started as a container but was not yet accepting database
  connections.
- Made the `project_key` Alembic migration idempotent against the current-schema
  baseline migration. Fresh databases created by `0001_create_current_schema`
  already include these columns, so the follow-up migration now skips columns
  and indexes that are already present.
- Suppressed the expected SQLAlchemy reflection warning for pgvector's
  `vector` type inside the schema checker while preserving the explicit
  PostgreSQL type/dimension validation.
- Verification:

```powershell
uv run pytest backend/tests/test_paraworks_docker_script.py backend/tests/test_db_schema_operations.py backend/tests/test_pgvector_dev_runbook.py -q
uv run ruff check scripts/check_db_schema.py backend/migrations/versions/5f8d874023d7_add_project_key_to_knowledge_models.py backend/tests/test_paraworks_docker_script.py backend/tests/test_db_schema_operations.py
.\scripts\paraworks-docker.ps1
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8000/health -TimeoutSec 5
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:3000/login -TimeoutSec 10
```

Result: 14 targeted tests passed; ruff passed; PowerShell parser check passed;
Docker services, backend health, and frontend login smoke passed.

## 2026-05-14 Assistant Recipient Resolver

- Added the design note
  `docs/superpowers/specs/2026-05-14-assistant-recipient-resolver-design.md`
  for the AI Assistant email recipient resolution layer.
- Added `backend/app/assistant/recipient_resolver.py`, a deterministic,
  cost-free resolver that collects contact candidates from recent assistant
  context, `AuthUser`, `demo_auth.USERS`, and Google `Source` metadata
  (`gmail`, `gmail_attachment`, `drive`, `calendar`).
- The email orchestration now runs `[Tool: recipient_resolver]` after
  `email_intent_gate` and before `email_draft_composer`, passing
  `resolved_recipients` into the draft prompt.
- This supports flows such as "김용희님한테 오늘 회의 3시에 있다고 메일 보내줘" when the
  recent conversation or synced contact metadata contains
  `김용희 (yonghee199702@gmail.com)`.
- Duplicate names are surfaced as `ambiguous`; department/group messages such
  as `Product팀 전체` resolve to all matching known users.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_recipient_resolver.py backend/tests/test_assistant_api.py::test_assistant_passes_resolved_recipient_to_email_draft_composer -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py -q
uv run ruff check backend/app/assistant/recipient_resolver.py backend/app/assistant/email_agent.py backend/app/api/v1/assistant.py backend/tests/test_assistant_recipient_resolver.py backend/tests/test_assistant_api.py
```

Result: targeted resolver/API tests passed; 37 assistant tests passed; ruff
passed.

## 2026-05-14 Email Draft Composer Model Upgrade

- Split the email sub-agent model settings by responsibility:
  `ASSISTANT_EMAIL_AGENT_MODEL` remains the low-cost intent gate model and
  defaults to `gpt-4.1-nano`.
- Added `ASSISTANT_EMAIL_DRAFT_AGENT_MODEL`, defaulting to `gpt-5.4-mini`, so
  the draft composer can produce better Korean business email drafts without
  making every email-intent classification more expensive.
- `build_email_draft_composer()` now instantiates `ChatOpenAI` with
  `settings.assistant_email_draft_agent_model`; `build_email_intent_gate()`
  still uses `settings.assistant_email_agent_model`.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_email_agent.py::test_email_draft_composer_defaults_to_stronger_model_than_intent_gate backend/tests/test_assistant_email_agent.py::test_email_draft_composer_builder_uses_dedicated_draft_model -q
uv run pytest backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_api.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py -q
uv run ruff check backend/app/core/config.py backend/app/assistant/email_agent.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_api.py
```

Result: targeted model tests passed; 39 assistant tests passed; ruff passed.

## 2026-05-14 Assistant Contact Lookup Routing

- Split direct contact lookup away from the email intent/draft path with
  `backend/app/assistant/contact_lookup.py`.
- Requests such as `김종우님 이메일 알려줘.` now resolve known contacts directly
  through the deterministic `recipient_resolver` and return the address instead
  of asking the user to provide it.
- Follow-up replies such as `너가 알려줘야지.` reuse the latest contact lookup
  request from the conversation context, so the assistant does not accidentally
  enter the email draft or RAG path.
- Added Korean aliases for key `demo_auth.USERS` contacts and lowered demo
  contact confidence so active `AuthUser` records still win when real DB users
  are present.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_recipient_resolver.py::test_recipient_resolver_uses_demo_user_korean_alias backend/tests/test_assistant_api.py::test_assistant_contact_lookup_returns_known_email_without_email_draft backend/tests/test_assistant_api.py::test_assistant_contact_lookup_followup_uses_recent_lookup_request -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_assistant_recipient_resolver.py -q
uv run ruff check backend/app/assistant/contact_lookup.py backend/app/assistant/recipient_resolver.py backend/app/api/v1/assistant.py backend/app/core/demo_auth.py backend/tests/test_assistant_api.py backend/tests/test_assistant_recipient_resolver.py
```

Result: targeted contact lookup tests passed; wider assistant backend tests
passed with 47 tests; ruff passed.

## 2026-05-14 Assistant Referenced Email Drafts

- Added `backend/app/assistant/email_draft_context.py` so the assistant can
  treat recent AI answers and pending email drafts as explicit email body
  sources.
- Requests such as `이 내용을 용희님한테 메일로 보내줘.` now route before the
  low-cost `email_intent_gate`, select the latest sendable assistant answer,
  resolve recipients, and pass that selected source to `email_draft_composer`.
- Draft revision complaints such as `내용이 하나도 안 들어가 있잖아.` now reuse
  the pending draft's recipient/subject plus the earlier assistant answer,
  creating a new approval-required draft instead of falling into RAG chat.
- Added a source-content guardrail: if the draft composer compresses `이 내용`
  into a generic note and omits the actual selected body, the selected source is
  appended to the draft before storing the pending approval metadata.
- Isolated `test_email_draft_composer_defaults_to_stronger_model_than_intent_gate`
  from local `.env` overrides with `Settings(_env_file=None)`.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_api.py::test_assistant_referenced_answer_email_keeps_selected_content backend/tests/test_assistant_api.py::test_assistant_revises_pending_draft_when_user_says_body_is_missing -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_assistant_recipient_resolver.py -q
uv run ruff check backend/app/api/v1/assistant.py backend/app/assistant/email_draft_context.py backend/app/assistant/email_agent.py backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py
```

Result: targeted referenced-email tests passed; wider assistant backend tests
passed with 49 tests; ruff passed.

## 2026-05-14 Assistant Generate-Then-Email Drafts

- Extended `backend/app/assistant/email_draft_context.py` with a generated
  source request detector for messages such as
  `ParaWorks 회사 소개서 작성해서 용희님한테 메일 보내줘.`.
- This route now runs before `email_intent_gate`: it extracts the generation
  question (`ParaWorks 회사 소개서 작성해줘`), retrieves/generates the answer through
  the RAG orchestrator, resolves the recipient, and then passes the generated
  answer to `email_draft_composer`.
- The existing source-content guardrail also applies here, so if the draft
  composer returns a generic body, the generated RAG answer is appended before
  the pending approval draft is stored.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_api.py::test_assistant_generates_requested_content_before_email_draft -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_assistant_recipient_resolver.py -q
uv run ruff check backend/app/api/v1/assistant.py backend/app/assistant/email_draft_context.py backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py
```

Result: targeted generate-then-email test passed; wider assistant backend tests
passed with 50 tests; ruff passed.

## 2026-05-14 Assistant Recipient Correction Safety

- Fixed recipient resolution so known contacts are only returned when an alias,
  email local-part, display name, or title actually matches the latest message.
  This prevents unrelated demo users from appearing as ambiguous candidates for
  unknown names such as `한승혁`.
- Added a pre-RAG recipient gate for generate-then-email requests. If the
  recipient is not resolved, the assistant asks for the exact recipient instead
  of spending RAG/LLM draft cost or reusing a previous pending draft recipient.
- Added a pending draft recipient-update path. Messages such as
  `SeungHun Han님한테 보내줘.` reuse the pending draft body/source while replacing
  only the recipient.
- Added an explicit correction response for `메일 주소가 잘못됐어.` so it asks for
  the corrected recipient rather than falling into contact lookup or RAG.
- Added `SeungHun Han` / `한승헌` aliases to the demo admin contact.
- Verification:

```powershell
uv run pytest backend/tests/test_assistant_api.py::test_assistant_generate_email_unknown_recipient_does_not_reuse_pending_draft backend/tests/test_assistant_api.py::test_assistant_recipient_only_revision_preserves_pending_draft_body backend/tests/test_assistant_api.py::test_assistant_wrong_email_address_asks_for_correct_recipient backend/tests/test_assistant_recipient_resolver.py -q
uv run pytest backend/tests/test_assistant_api.py backend/tests/test_assistant_email_agent.py backend/tests/test_assistant_service.py backend/tests/test_assistant_models.py backend/tests/test_assistant_recipient_resolver.py -q
uv run ruff check backend/app/api/v1/assistant.py backend/app/assistant/email_draft_context.py backend/app/assistant/recipient_resolver.py backend/app/core/demo_auth.py backend/tests/test_assistant_api.py backend/tests/test_assistant_recipient_resolver.py
```

Result: targeted recipient-correction tests passed; wider assistant backend
tests passed with 53 tests; ruff passed.

## 2026-05-15 Gmail/Google Drive 프로젝트 Tool Routing 분업 가이드

- 새 문서:
  - `docs/superpowers/runbooks/2026-05-15-gmail-drive-project-routing-collaboration-guide.md`
- 목적:
  - Gmail/Drive 데이터도 Slack Agent와 같은 LangChain tool 기반 프로젝트 라우팅으로 바꿀 때, Slack 담당자와 Mail/Document 담당자의 작업 영역이 겹치지 않게 한다.
- 핵심 경계:
  - 공용 프로젝트 Router 계약은 `backend/app/agent_runtime/project_routing.py`에 둔다.
  - Mail/Document 담당자는 `backend/app/agents/mail_document_agent/`와 관련 테스트만 수정한다.
  - Slack 담당자는 `agent_slack/`, `backend/app/agents/slack_agent/`만 수정한다.
  - Review 승인 정책, 통합 API, 프론트 UI, Playwright 통합 테스트는 통합 담당자가 별도 브랜치에서 맡는다.
  - Mail/Document 담당자는 `agent_slack/project_routing.py`를 직접 import하지 않는다.
- Mail/Document 전환 방향:
  - 기존 `EvidencePacket -> MailDocumentAgent.run() -> ReviewCandidate -> ReviewItem` 흐름은 유지한다.
  - `MailDocumentAgent.run()` 이후 후보에 project routing을 적용한다.
  - Gmail 본문+첨부 grouping, Drive 파일 단위 grouping은 유지한다.
  - ReviewItem payload 필드는 Slack과 동일한 `project_assignment_method='llm_tool'`, `project_key`, `project_name`, `project_assignment_summary`, `project_assignment_reason`, `project_assignment_confidence`, `project_alternatives`, `project_needs_user_selection`을 사용한다.

## 2026-05-15 타임라인 실제 source 시각 및 프로젝트 근거 UX

- 계획서:
  - `docs/superpowers/plans/2026-05-15-timeline-project-evidence-ux.md`
- 변경 요약:
  - `backend/app/projects/service.py`의 `ProjectTimelineItem`에 `occurred_at`을 추가했다.
  - 프로젝트 타임라인 날짜/시간은 Slack permalink 또는 `Source.raw_metadata.ts`에서 계산한 실제 source 발생 시각을 우선 사용한다.
  - source timestamp가 없으면 기존 knowledge row `created_at`을 fallback으로 사용한다.
  - 프로젝트 탭 `연결된 원본 근거`는 더 이상 approved `project_assignment`에만 의존하지 않는다. 승인된 `decision_record`, `history_event`, `timeline_event`, `todo`의 source link/snippet에서도 `ProjectEvidence`를 만든다.
  - 타임라인 화면은 기본 title-only 리스트로 바뀌었고, 날짜 그룹은 최신 날짜만 기본 표시한다.
  - `날짜 전체 보기`, 날짜별 `자세히 보기/간단히 보기`로 compact/detail 전환이 가능하다.
- 주의:
  - 기존 테스트 중 “approved knowledge를 connector evidence로 바꾸지 않는다”는 기대는 사용자 요구와 충돌해 새 정책으로 수정했다.
  - Gmail/Drive도 같은 프로젝트 API를 쓰므로 approved activity에 source link/snippet이 있으면 프로젝트 근거에 표시된다.
- 검증:
  - `uv run pytest backend/tests/test_project_memory_api.py backend/tests/test_review.py backend/tests/test_review_knowledge_promotion.py -q` → `44 passed`
  - `uv run ruff check backend/app/projects/service.py backend/tests/test_project_memory_api.py` → 통과
  - `npm run lint` → 통과
  - `npm run build` → 통과
  - `npm run test:visual -- timeline-project-date-groups.spec.ts projects-source-links.spec.ts slack-project-routing-flow.spec.ts --project=chromium-desktop` → `3 passed`

## 2026-05-15 대시보드 todo 완료 상태 영구 저장

- 변경 요약:
  - `Todo` 모델과 Postgres schema에 `assignee`, `due_date`, `completed_at`, `completed_by`를 추가했다.
  - 새 migration은 기존 approved todo ReviewItem payload에서 담당자와 마감일을 backfill한다.
  - Review 승인 시 todo의 담당자/마감일을 trusted `Todo` row에 저장한다.
  - `POST /api/v1/todos/{todo_id}/complete`가 완료 시각과 완료자 ID를 저장한다.
  - 완료 API는 사용자가 접근할 수 없는 permission level의 todo를 403으로 거부한다.
  - 대시보드 `today_todos`는 approved ReviewItem이 아니라 승인된 미완료 `Todo`를 읽는다.
  - 완료된 todo는 대시보드에서 빠지고, 프로젝트 activity/timeline item에는 `completed_at`, `completed_by`가 내려간다.
  - 타임라인에서는 완료된 todo가 `완료` 상태로 보이고, 프로젝트 활동 카드에도 `완료` 배지가 표시된다.
- 로컬 DB:
  - `alembic upgrade head`로 Docker Postgres에 `b4b6d9f4d3e1` migration을 적용했다.
  - 기존 todo 2건의 `assignee`, `due_date` backfill 확인 완료.
- 검증:
  - `uv run ... pytest backend/tests/test_project_memory_api.py backend/tests/test_dashboard_api.py backend/tests/test_review_knowledge_promotion.py backend/tests/test_todos_api.py -q` → `36 passed`
  - `uv run ... ruff check ...` → 통과
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts timeline-project-date-groups.spec.ts projects-source-links.spec.ts --project=chromium-desktop` → `5 passed`
- 주의:
  - 실행 중인 backend 서버가 이전 코드로 떠 있으면 새 `/api/v1/todos/{id}/complete` endpoint가 없으므로 서버 재시작이 필요하다.
  - 현재 임시 Python 테스트 환경은 `.tmp/uv-test-venv`를 사용했다. 기본 `.venv`는 기존 uv Python 경로 문제로 바로 실행되지 않았다.

## 2026-05-16 타임라인 날짜 UX, Review 우클릭 메뉴, 대시보드 검토 동기화

- 변경 요약:
  - 타임라인 status UI에서 `approved`는 `승인됨`으로 표시한다.
  - 타임라인 날짜 그룹은 최근 7일의 활동 날짜만 기본 펼침 상태로 두고, 더
    오래된 날짜는 접힌 상태로 시작한다.
  - 월 단위 sticky header와 좌측 미니 날짜 인덱스를 추가했다. 인덱스 날짜를
    누르면 해당 날짜 그룹으로 스크롤하고 자동으로 펼친다.
  - `전체 날짜 보기` 토글을 켜면 활동이 없는 날짜도 인덱스와 그룹에 표시한다.
  - Review item 우클릭 메뉴는 항목 제목, 승인, 반려 설명을 포함하는 작은
    드롭다운 UI로 바뀌었고 viewport 바깥으로 넘치지 않게 좌표를 보정한다.
  - Dashboard 검토사항 카드는 `/api/v1/dashboard`의 `pending_review_count`
    전체 값을 배지로 쓰고, 목록은 Review Queue 우선순위 정렬 기준 상위 3개만
    표시한다.
- 검증:
  - `.\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_dashboard_api.py` → `6 passed`
  - `.\\.venv\\Scripts\\python.exe -m ruff check backend/app/api/v1/dashboard.py backend/tests/test_dashboard_api.py` → 통과
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
  - `npm.cmd run test:visual -- timeline-project-date-groups.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop` → `2 passed`
  - `npm.cmd run test:visual -- review-bulk-actions.spec.ts --project=chromium-desktop` → `2 passed`
- 주의:
  - 기본 `python -m pytest`는 로컬 환경에서 `pydantic_settings`가 없어 실패했다.
    repo `.venv` 실행은 sandbox 권한 문제로 escalated 실행이 필요했다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 대시보드 검토사항 카드 deep link와 표시 제목 정합성

- 변경 요약:
  - `backend/app/services/review_display.py`를 추가해 ReviewItem 표시 제목을
    공용으로 계산한다.
  - `ParaWorks source 연결`, `source 연결`, `untitled`, `unknown` 같은 낮은
    정보량 제목은 실제 검토자가 볼 수 있는 `summary`, `decision_summary`,
    `reason`, `task_summary`, `source_title` 등으로 대체한다.
  - Dashboard API의 `pending_items`는 공용 display title과
    `review_url=/review?itemId={id}`를 내려준다.
  - Review API 그룹 제목도 같은 display title을 사용한다.
  - Dashboard 검토사항 카드 항목은 `review_url`로 이동한다.
  - Review 페이지는 `itemId`/`item_id` query를 읽고 해당 item이 포함된 그룹을
    자동으로 펼친 뒤 항목 위치로 스크롤한다.
- 검증:
  - `.\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_review.py::test_review_list_uses_display_title_when_payload_title_is_low_signal backend/tests/test_dashboard_api.py` → `8 passed`
  - `.\\.venv\\Scripts\\python.exe -m ruff check backend/app/api/v1/dashboard.py backend/app/api/v1/review.py backend/app/services/review_display.py backend/tests/test_dashboard_api.py backend/tests/test_review.py` → 통과
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop` → `2 passed`
  - `npm.cmd run test:visual -- review-bulk-actions.spec.ts --project=chromium-desktop` → `2 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 대시보드 검토사항 카드 중복 그룹 접기

- 변경 요약:
  - Dashboard API의 `pending_items`는 Review Queue와 같은
    `item_type + display title` 그룹 기준으로 중복을 제거한 뒤 상위 3개를
    내려준다.
  - `pending_review_count`는 dedupe하지 않고 실제 pending review 총수를 유지한다.
  - 이로써 `ParaWorks source 연결`에서 파생된 같은 display title 후보가 여러 개
    있어도 대시보드 카드에는 하나만 표시된다.
- 검증:
  - `.\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_dashboard_api.py` → `8 passed`
  - `.\\.venv\\Scripts\\python.exe -m ruff check backend/app/api/v1/dashboard.py backend/tests/test_dashboard_api.py` → 통과
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop` → `2 passed`

## 2026-05-16 프로젝트 워크스페이스 UI 리디자인

- 변경 요약:
  - `frontend/src/app/projects/page.tsx`를 dashboard 계열 디자인 톤의 workspace
    구조로 재구성했다.
  - 상단 header 아래 선택 프로젝트 overview hero를 추가해 프로젝트명, 설명,
    근거 수, 활동 수, 검토 대기 수를 먼저 보여준다.
  - 본문은 project list, evidence panel, activity timeline panel로 나뉜다.
  - 프로젝트 목록은 선택 상태를 indigo soft background로 강조하고, 많은 프로젝트가
    있을 때 내부 스크롤을 사용한다.
  - 원본 근거 패널은 전체/Drive/Gmail/Slack/Calendar filter tab과 hover card
    스타일을 제공한다.
  - 승인 활동 패널은 subtle vertical timeline 구조와 activity type badge를 사용한다.
  - 반응형은 `2xl` 3영역, `xl` 2영역+활동 하단, 그 이하는 세로 stack이다.
- 검증:
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
  - `npm.cmd run test:visual -- projects-responsive-metrics.spec.ts projects-source-links.spec.ts --project=chromium-desktop` → `4 passed`
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 프로젝트 워크스페이스 board UI 정교화

- 변경 요약:
  - 사용자가 제공한 target UI를 참고해 프로젝트 페이지 본문을 calm kanban/workspace
    스타일로 한 단계 더 다듬었다.
  - 중요한 제약대로 `AppShell`과 ParaWorks 사이드바는 수정하지 않았다.
  - 페이지 헤더는 soft glass card와 rounded action button으로 정리했다.
  - 선택 프로젝트 overview는 subtle pastel gradient summary board로 바꾸고 metric
    blocks를 white mini-stat 카드로 유지했다.
  - 프로젝트 목록, 연결된 원본 근거, 승인된 프로젝트 활동은 각각 board lane처럼
    보이도록 rounded container, diffuse shadow, pill chip, soft card rhythm을
    적용했다.
  - 원본 근거와 승인 활동 카드는 source/type별 very light pastel tint를 사용한다.
  - 프로젝트 선택, 프로젝트 검색, source filter, 원본 링크, 활동 렌더링,
    responsive 3/2/1 column 흐름은 기존 그대로 유지했다.
- 검증:
  - `npm.cmd run lint` → 통과
  - `npm.cmd run test:visual -- projects-responsive-metrics.spec.ts projects-source-links.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run build` → 통과
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 프로젝트 목록 sticky follow

- 변경 요약:
  - `frontend/src/app/projects/page.tsx`의 `ProjectListPanel`을 `xl` 이상에서
    `position: sticky`로 동작하게 했다.
  - top offset은 전역 sticky top bar 아래에 걸리도록 `xl:top-28`을 사용했다.
  - 프로젝트 목록 내부 스크롤은 `xl:max-h-[calc(100vh-18rem)]`로 제한해 낮은
    노트북 화면에서도 패널이 viewport 밖으로 길게 밀리지 않게 했다.
  - `frontend/e2e/projects-responsive-metrics.spec.ts`에 desktop project list가
    sticky position과 top offset을 갖는지 확인하는 회귀 테스트를 추가했다.
- 검증:
  - `npm.cmd run test:visual -- projects-responsive-metrics.spec.ts --project=chromium-desktop` → `2 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `fixed` position은 사용하지 않았다. sticky는 grid document flow 안에서만 동작한다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 대시보드 업무/프로젝트 카드 링크 교체

- 변경 요약:
  - `frontend/src/app/dashboard/page.tsx`에서 `오늘 해야 할 업무` 카드 우측 링크를
    `타임라인 보기` / `/timeline`으로 바꿨다.
  - `담당 프로젝트` 카드 우측 링크를 `프로젝트 보기` / `/projects`로 바꿨다.
  - `frontend/e2e/dashboard-workflow.spec.ts`에 두 카드의 링크 라벨과 href를
    고정하는 회귀 테스트를 추가했다.
- 검증:
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop` → `2 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - UI 라벨/링크만 교체했으며 dashboard data/API 흐름은 수정하지 않았다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 검토사항 Agent 배지 source별 분리와 sticky action bar

- 변경 요약:
  - `frontend/src/app/review/page.tsx`에 `primarySourceType`, `agentBadgeLabel`,
    `agentBadgeClass` helper를 추가했다.
  - Agent 배지는 `payload.agent_name`만 보지 않고 `payload.source_type`을 우선,
    없으면 `source_evidence[].source_type`을 fallback으로 사용한다.
  - `mail_document_agent`라도 source가 `gmail`/`gmail_attachment`이면 `Mail Agent`,
    `drive`면 `Google Drive Agent`, `calendar`면 `Calendar Agent`로 표시한다.
  - `slack_agent` 또는 `source_type=slack`은 `Slack Agent`로 표시한다.
  - 색상은 프로젝트 페이지 source badge와 맞춰 Slack violet, Mail rose,
    Google Drive blue, Calendar emerald 계열을 사용한다.
  - Review bulk action bar는 `fixed`가 아니라 `sticky top-24`로 바꿔 전역 topbar
    아래에서 문서 흐름 안에 머물며 따라오게 했다.
  - `frontend/e2e/review-bulk-actions.spec.ts`에 source별 Agent label/color와
    sticky action bar 회귀 테스트를 추가했다.
- 검증:
  - `npm.cmd run test:visual -- review-bulk-actions.spec.ts --project=chromium-desktop` → `3 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - Review API payload shape은 변경하지 않았다. UI에서 기존 payload/source evidence를
    해석하는 방식만 보강했다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 타임라인 Explorer UI 압축 리디자인

- 변경 요약:
  - `frontend/src/app/timeline/page.tsx`만 수정해 타임라인 본문을 soft SaaS
    workspace 스타일로 재구성했다. ParaWorks sidebar/AppShell은 변경하지 않았다.
  - 상단에 `timeline-summary-strip`을 추가해 전체 히스토리, 승인됨, 주요 소스,
    최근 날짜를 compact KPI로 보여준다.
  - 프로젝트 선택은 rounded pill tab으로 바꿨고 각 프로젝트 history count badge를
    표시한다.
  - 필터 영역은 white/rounded toolbar 톤으로 정리하되 기존 기간/소스/상태 필터,
    전체 날짜 보기, 필터 초기화 기능은 유지했다.
  - 좌측 날짜 인덱스는 `xl` 이상에서만 보이는 sticky compact month navigator로
    바꿨다. 기본은 최근 월만 펼치고 오래된 월은 접힘 상태다.
  - 중앙 목록도 월별 header + 날짜 group card 구조로 바꿨다. 오래된 월은 collapsed
    summary card로 보이고, 펼치면 날짜 그룹이 나타난다.
  - 날짜 그룹은 기본 3개 항목만 노출하고 `N건 더 보기`로 확장한다.
  - timeline item은 source icon, 1줄 title/preview, time, source badge,
    status badge, detail icon 중심의 compact card로 정리했다.
  - 상세 패널은 오른쪽 side panel로 유지하되 rounded/glass card 톤으로 조정했다.
  - `frontend/e2e/timeline-project-date-groups.spec.ts`에 summary strip, 오래된 월
    default collapse, month navigator expand/jump 회귀 검증을 추가했다.
- 검증:
  - `npm.cmd run test:visual -- timeline-project-date-groups.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - API/data shape은 변경하지 않았다.
  - 추가로 `npm.cmd run test:visual -- page-regression.spec.ts gmail-drive-project-routing-flow.spec.ts slack-project-routing-flow.spec.ts --project=chromium-desktop`를 시도했다. Gmail/Drive flow는 통과했지만, `page-regression`은 기존 route inventory의 `/documents` 불일치와 auth 401 콘솔 에러로 실패했고, Slack flow는 `/integrations`의 `slack-card-actions`를 찾지 못해 타임라인 진입 전 실패했다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 타임라인 날짜 인덱스 sticky 동작 보정

- 변경 요약:
  - `aria-label="타임라인 날짜 인덱스"` aside는 이미 `xl:sticky xl:top-28`였지만,
    상위 timeline list panel의 `overflow-hidden` 때문에 실제 페이지 스크롤에서는
    sticky 기준이 깨져 위로 밀려났다.
  - `frontend/src/app/timeline/page.tsx`의 timeline list panel root를
    `overflow-visible`로 바꿔 날짜 인덱스가 fixed가 아닌 sticky로 자연스럽게
    따라오도록 했다.
  - `frontend/e2e/timeline-project-date-groups.spec.ts`에 computed position/top과
    스크롤 후 y 좌표가 sticky top 근처에 유지되는 회귀 검증을 추가했다.
- 검증:
  - `npm.cmd run test:visual -- timeline-project-date-groups.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `position: fixed`는 사용하지 않았다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후
    해당 생성 변경은 되돌렸다.

## 2026-05-16 유틸리티 워크스페이스 페이지 SaaS 톤 정리

- 변경 요약:
  - `frontend/src/app/search/page.tsx`, `agent-runs/page.tsx`,
    `integrations/page.tsx`, `notifications/page.tsx`, `admin/page.tsx`의 page root에
    `utility-workspace` 스코프를 추가했다.
  - AI 비서 페이지는 추가로 `utility-workspace-chat` 스코프를 사용해 conversation
    rail과 chat surface만 대시보드급 rounded/glass 스타일로 보정한다.
  - `frontend/src/app/globals.css`에 scoped styles를 추가해 page heading, utility
    badge, panel/reference card, integration card, admin metric/table, chat shell,
    action button을 soft SaaS workspace 톤으로 통일했다.
  - 기능/API/data shape은 변경하지 않았다. 사이드바/AppShell 구조도 그대로다.
  - `frontend/e2e/utility-workspace-style.spec.ts`를 추가해 다섯 페이지가 공통 UI
    스코프를 유지하는지 source-level 회귀 테스트를 둔다.
- 검증:
  - `npm.cmd run lint` → 통과
  - `npm.cmd run test:visual -- utility-workspace-style.spec.ts integration-sync-modal.spec.ts` → `12 passed`
  - `npm.cmd run build` → 통과
- 주의:
  - 추가 확인 중 `assistant-memory.spec.ts`와 `orchestration.spec.ts`를 함께 실행해
    보았으나, 전자는 AppShell의 `/api/v1/dashboard`, `/api/v1/notifications` 조회를
    테스트 allowlist가 막는 기존 목킹 범위 문제로 중단됐고, 후자는 `/agent-runs`
    서버 데이터/권한 의존성 때문에 `app-shell`을 찾지 못했다. 이번 커밋 범위의
    UI 스코프 테스트와 integrations 동기화 흐름은 통과했다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후 해당
    생성 변경은 되돌렸다.

## 2026-05-16 AI 비서 채팅 히스토리 접기 컨트롤 복구

- 변경 요약:
  - `frontend/src/app/search/page.tsx`에서 채팅 히스토리 open/close 컨트롤을
    명시적으로 다시 분리했다.
  - 기존 패널 안 `대화 목록 접기` 버튼은 toggle 대신 `setSidebarCollapsed(true)`를
    호출한다.
  - 히스토리가 펼쳐진 상태에서도 채팅 본문 좌상단에 `히스토리 접기` 버튼을 보여
    사용자가 본문에서 바로 다시 접을 수 있게 했다.
  - AI 비서 root에 `data-assistant-hydrated`를 추가해 상호작용 테스트가 hydration
    이후 실행되도록 했다.
  - `frontend/e2e/assistant-memory.spec.ts`에 AppShell dashboard/notifications
    배지 조회 mock을 보강하고, 펼침 -> 접힘 -> 재펼침 회귀를 추가했다.
  - `frontend/e2e/utility-workspace-style.spec.ts`는 open/collapse 컨트롤 계약을
    확인한다.
- 검증:
  - `npm.cmd run test:visual -- utility-workspace-style.spec.ts` → `4 passed`
  - `npm.cmd run test:visual -- assistant-memory.spec.ts utility-workspace-style.spec.ts --project=chromium-desktop` → `3 passed`
  - `npm.cmd run test:visual -- assistant-memory.spec.ts utility-workspace-style.spec.ts --project=chromium-mobile` → `3 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꿔 빌드 후 해당
    생성 변경은 되돌렸다.

## 2026-05-17 대시보드 캘린더 refresh 오늘 날짜 유지

- 변경 요약:
  - `frontend/src/app/dashboard/page.tsx`에서 캘린더 선택 날짜에 일정이 없으면
    연동 일정 중 가장 빠른 날짜로 자동 이동하던 effect를 제거했다.
  - 원인은 2026-05-17 오늘 일정이 없고 연동 데이터의 가장 빠른 일정이
    2026-04-17일 때, 새로고침 후 `selectedDate`와 `visibleMonth`가 함께
    4월로 바뀌는 로직이었다.
  - 사용자가 날짜를 클릭해 선택하는 동작은 그대로 유지하고, 연동된 과거/미래
    일정은 해당 월로 이동했을 때 dot과 일정 목록으로 확인하도록 했다.
  - `frontend/e2e/dashboard-calendar-state.spec.ts`를 추가해 이 자동 이동 로직이
    다시 들어오지 않도록 source-level 회귀 테스트를 둔다.
  - `frontend/e2e/dashboard-workflow.spec.ts`에는 이전 월에만 일정이 있어도
    2026년 5월과 5월 17일 선택 상태를 유지하는 브라우저 회귀 케이스를 추가했다.
- 검증:
  - `npm.cmd run test:visual -- dashboard-calendar-state.spec.ts --project=chromium-desktop` → `1 passed`
  - `npm.cmd run test:visual -- dashboard-workflow.spec.ts --project=chromium-desktop -g "previous month"` → `1 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸면 빌드 후 해당
    생성 변경은 되돌린다.

## 2026-05-17 검토사항 프로젝트 연결 캘린더 raw metadata 표시 정리

- 변경 요약:
  - 증상: 검토사항 페이지의 `<프로젝트 연결>` 그룹 제목에
    `Description: <p>...`, `Marker`, `Location`, `Start`, `End`가 그대로 붙어
    긴 raw 캘린더 본문처럼 보였다.
  - 원인: Google Calendar sync가 `Source.body`를 `제목 + Description + Location +
    Start + End` 형태로 보존하고, 프로젝트 분류기가 이 chunk snippet 전체를
    `summary/task_summary`로 저장했다. Review API는 `프로젝트 source 연결`을 낮은
    신호 제목으로 보고 summary를 그룹 제목으로 선택했다.
  - `backend/app/projects/classifier.py`에서 프로젝트 연결 후보의 task summary를
    HTML 태그와 캘린더 metadata label 이전의 실제 이벤트 제목 중심으로 정리한다.
  - `backend/app/services/review_display.py`에서 기존 DB에 이미 raw summary가 들어간
    ReviewItem도 API group title에서 깨끗하게 보이도록 display text sanitizer를
    적용했다.
  - `frontend/src/app/review/page.tsx`에서도 item title, 상세 summary, 프로젝트 연결
    후보 필드를 같은 방식으로 정리해 mock/API payload가 raw여도 UI가 무너지지 않게 했다.
  - `frontend/e2e/review-agent-metadata.spec.ts`는 source별 Agent 배지 정책에 맞춰
    `project_classifier` + Slack source를 `Slack Agent`로 기대하도록 갱신했다.
- 검증:
  - RED 확인: 신규 backend 회귀 테스트 2개가 기존 코드에서 실패함을 확인.
  - `uv run pytest backend/tests/test_review.py::test_project_assignment_group_title_sanitizes_calendar_metadata backend/tests/test_project_memory_api.py::test_calendar_project_assignment_summary_uses_event_title_not_raw_metadata -q` → `2 passed`
  - `uv run pytest backend/tests/test_review.py backend/tests/test_project_memory_api.py -q` → `45 passed`
  - `uv run ruff check --no-fix backend/app/projects/classifier.py backend/app/services/review_display.py backend/tests/test_project_memory_api.py backend/tests/test_review.py` → 통과
  - `npm.cmd run test:visual -- review-agent-metadata.spec.ts review-bulk-actions.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `uv run ruff check backend`는 저장소 기존 B008/N806/F841 이슈 때문에 실패하며,
    현재 설정상 관련 없는 파일 자동 수정도 발생할 수 있다. 이번 작업에서는 자동 수정된
    무관 파일을 되돌리고 수정 파일만 `--no-fix`로 검사했다.
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후 해당
    생성 변경은 되돌렸다.

## 2026-05-17 프로젝트 목록 설명 자동 연결 통계 문구 제거

- 변경 요약:
  - `frontend/src/app/projects/page.tsx`의 좌측 프로젝트 목록 카드에서
    `project.summary` 뒤에 붙어 내려오는 `승인된 원본 근거 N건과 승인된 프로젝트 활동 N건이
    연결되어 있습니다.` 문구를 표시하지 않도록 했다.
  - 이 처리는 프로젝트 목록 카드 전용 표시 함수로 제한했다. 프로젝트 상세 summary, evidence/activity
    metric, 목록 하단 `근거 · 활동 · 검토 대기` 수치는 유지한다.
  - `frontend/e2e/projects-responsive-metrics.spec.ts`에 프로젝트 목록 패널에서 자동 연결
    통계 문구가 노출되지 않는 회귀 검증을 추가했다.
- 검증:
  - RED 확인: 새 기대값이 기존 코드에서 실패함을 확인.
  - `npm.cmd run test:visual -- projects-responsive-metrics.spec.ts --project=chromium-desktop -g "responsive workspace"` → `1 passed`
  - `npm.cmd run test:visual -- projects-responsive-metrics.spec.ts projects-source-links.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후 해당
    생성 변경은 되돌렸다.

## 2026-05-17 검토사항 프로젝트 연결 메일/Drive metadata 표시 정리

- 변경 요약:
  - 증상: 검토사항 페이지의 `<프로젝트 연결>` 항목 제목/연결 내용에 Gmail `From`,
    `Date` 헤더와 깨진 발신자 문자열이 붙어 보였다.
  - 원인: Gmail connector가 `Subject + From + Date + body`를 Source body로 보존하고,
    프로젝트 분류기가 첫 chunk snippet 전체를 `summary/task_summary`로 저장했다.
    Review display title은 낮은 신호 제목인 `프로젝트 source 연결` 대신 summary를
    사용하므로 raw header가 화면에 올라왔다.
  - `backend/app/services/review_display.py`의 display sanitizer를 확장해
    `From`, `Date`, `Mime type`, `Owner`, `Last modifier`, `Modified`,
    `Parent subject`, `Attachment size` label 이후 metadata를 제거한다.
  - `Google Drive file changed:`와 `Gmail attachment:` prefix도 표시용 summary에서
    제거해 Drive/첨부도 실제 파일명/제목 중심으로 보이게 했다.
  - `backend/app/projects/classifier.py`는 이미 해당 sanitizer를 공유하므로 새로 생성되는
    규칙 기반 프로젝트 연결 후보의 `task_summary`도 메일/Drive/첨부 metadata 없이 저장된다.
  - `frontend/src/app/review/page.tsx`도 같은 표시 방어 로직을 사용해 raw mock/API payload가
    와도 상세 `연결 내용`, `원본`, item summary가 UI를 밀지 않도록 했다.
- 확인한 source별 상태:
  - Gmail: `From`/`Date` 제거, subject만 표시.
  - Google Calendar: 기존 `Description`/`Location`/`Start`/`End` 제거 유지.
  - Google Drive: `Google Drive file changed:` prefix와 `Mime type`/`Owner`/`Modified` 제거.
  - Gmail attachment: `Gmail attachment:` prefix와 `Parent subject`/`Mime type`/`Attachment size` 제거.
  - Slack: connector body가 메시지/스레드 본문이라 같은 header metadata 증상은 없고 기존 Slack 업무 신호 필터/표시 흐름 유지.
- 검증:
  - RED 확인: 신규 Gmail/Drive metadata 회귀 테스트 2개가 기존 코드에서 실패함을 확인.
  - `uv run pytest backend/tests/test_review.py::test_project_assignment_group_title_sanitizes_mail_and_drive_metadata backend/tests/test_project_memory_api.py::test_project_assignment_summary_strips_mail_and_drive_metadata -q` → `2 passed`
  - `uv run pytest backend/tests/test_review.py::test_project_assignment_group_title_sanitizes_mail_and_drive_metadata backend/tests/test_project_memory_api.py::test_project_assignment_summary_strips_mail_and_drive_metadata backend/tests/test_review.py::test_project_assignment_group_title_sanitizes_calendar_metadata backend/tests/test_project_memory_api.py::test_calendar_project_assignment_summary_uses_event_title_not_raw_metadata -q` → `4 passed`
  - `uv run pytest backend/tests/test_review.py backend/tests/test_project_memory_api.py -q` → `47 passed`
  - `uv run ruff check --no-fix backend/app/projects/classifier.py backend/app/services/review_display.py backend/tests/test_project_memory_api.py backend/tests/test_review.py` → 통과
  - `npm.cmd run test:visual -- review-agent-metadata.spec.ts review-bulk-actions.spec.ts --project=chromium-desktop` → `4 passed`
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
- 주의:
  - `next build`가 `frontend/next-env.d.ts`를 `.next/types`로 바꾸므로 빌드 후 해당
    생성 변경은 되돌렸다.

## 2026-05-15 타임라인 상태 한글화 및 완료 todo 병합

- 변경 요약:
  - 타임라인 상태 필터 옵션과 row chip을 `승인됨`, `완료`로 한글화했다.
  - `/projects`의 `timeline_items`는 더 이상 완료된 `todo` record를 별도 타임라인 row로 포함하지 않는다.
  - 완료된 `Todo`는 같은 프로젝트, 같은 source link, `[할 일] {todo.title}` 제목을 가진 기존 `TimelineEvent`와 매칭된다.
  - 매칭된 기존 타임라인 이벤트에 `completed_at`, `completed_by`를 병합해 프론트에서 `완료` 상태로 표시한다.
- 주의:
  - 기존 `TimelineEvent`가 없는 legacy todo는 타임라인에 새로 추가하지 않는다. 프로젝트 활동 목록에는 계속 todo로 표시된다.
- 검증:
  - `uv run ... pytest backend/tests/test_todos_api.py backend/tests/test_project_memory_api.py backend/tests/test_dashboard_api.py -q` → `32 passed`
  - `uv run ... ruff check backend/app/projects/service.py backend/tests/test_todos_api.py` → 통과
  - `npm.cmd run lint` → 통과
  - `npm.cmd run build` → 통과
  - `npm.cmd run test:visual -- timeline-project-date-groups.spec.ts dashboard-workflow.spec.ts --project=chromium-desktop` → `3 passed`

## 2026-08-28 C.5 Task 2 persistence boundary

- Head migration is `7c5a2e9f4b10` over `2f6a8b9c0d1e`. It is additive for
  retained V2.0 rows; new V2.1 rows bind exact generation and evidence identity.
- PostgreSQL guards enforce ReviewItem cutover provenance, validation terminal
  consistency, Assistant lineage, provider/rollout event-backed state, and
  source/parser/chunk authority. Do not replace these with SQLite-only checks.
- Provider aggregate mutations cover the full authorization/overrun snapshot
  and require exact state-version/event/backpointer alignment. `budget_overrun`
  preserves the authorized estimator/framing/price tuple; `breaker_cleared`
  requires an open breaker, operator attribution, a different reviewed cost
  policy, and the exact replacement authority. Provider event attribution is
  explicitly non-null and mutually exclusive: operator events cannot carry a
  call HMAC, and overrun events cannot carry an operator HMAC. Rollout control
  kinds are transition-specific: percentage authorization cannot change a breaker,
  breaker open/close cannot masquerade as authorization or invalidation, and
  generation invalidation only lowers the latch and advances its generation.
  Rollout metrics use `state_version + 1` without changing `control_epoch` or
  the last control event; control mutations increment both, and
  `corrected_critical_count` is monotonic.
- Parser-run identity and chunk lineage are frozen on every UPDATE, including
  legacy null-to-value attempts. C.5 repair must insert replacement rows.
- `backend/app/admin/auto_review_retained_state.py` is the shared exhaustive
  schema/retained-state authority for bootstrap, reset, and downgrade.
  Inspection/query errors must propagate; lifespan bootstrap database errors
  close the checkpoint runtime and abort startup before service construction.
  Only empty/test schemas downgrade.
- PostgreSQL verification requires `PARAWORKS_TEST_POSTGRES_URL`; the isolated
  Docker fixture is on port `55432` with pgvector. The migration test module
  creates a unique schema and drops it in `finally`, so generated rows do not
  leak between runs. Missing URLs fail, not skip.
- Historical-upgrade coverage is pinned to a manual `2f6a8b9c0d1e` schema and
  must not call current `Base.metadata`. Current round-4 evidence is `176`
  focused tests and `87` PostgreSQL-only tests (zero skips), plus the unchanged
  `118`-test Task 1 compatibility suite.
- Cutover stays drain/migrate/deploy/bootstrap/reconcile. Slack remains
  C.5-ineligible and Task 2 never calls live providers or connectors.

## 2026-08-28 C.5 Task 3 V2.1 request/extraction boundary

- Alembic head is now `9d7f3a1c6e20` over Task 2 revision
  `7c5a2e9f4b10`. The new revision replaces the named extraction-call
  lifecycle constraint for already-migrated databases. Do not collapse it
  into Task 2 metadata-driven table creation.
- V2.1 request creation must go through the explicit V2.1 preflight dispatch;
  stored graph/checkpoint identity and every extraction snapshot field are
  immutable replay inputs. V2.0 keeps its legacy graph/HMAC/checkpoint path.
- `ReviewWorkflowService` requires a supplied V2.1 launch authority in
  shadow/enforce, stores the exact V2.1 request, and returns the durable
  `created` thread without entering the V2.0 graph. Task 12 owns V2.1 graph
  registration/execution. Missing or invalid authority fails zero-call with
  bounded `cost_preview_changed`; disabled mode remains exact V2.0.
- `ExtractionCallStore` is the PostgreSQL E1/E2/E3 authority. Callers must
  supply the current owner subject and allowed permission set. A callback
  return value is never terminal evidence: successful candidate completion
  requires a fresh query of exactly one same-run ReviewItem and its exact
  contiguous evidence children before the call and AgentRun become complete.
- `ProviderAttemptGrant` cannot be constructed or issued by application code:
  there is no importable issuer/factory and no grant-visible transport or
  public transport dispatch method. The committed store creates a private
  closure capability, authenticates its exact identity, rechecks the locked
  live attempt in a short transaction, closes that transaction, consumes the
  permit under the store lock, releases the lock, and only then performs the
  one provider I/O through `FencedOpenAITransport` and the server-owned Task 1
  hook. Every terminal or true authority-loss path invalidates retained
  grants. Cancellation before E2 terminalizes zero-charge and revokes; after
  E2 it latches output discard while preserving one dispatch until permit or
  lease expiry. E3/failure then charges actual usage or reserve exactly once
  and persists no candidate/cache marker.
- Evidence replay recomputes the selected aggregate message set from prepared
  slot identities and locked current canonical refs, and binds source
  kind/id/version or signature, strictest permission, fingerprint key identities,
  workflow/scope/candidate identity, child ordinals, and the terminal result
  set. ReviewItem permission must equal the recomputed strictest selected
  evidence permission; both broadened and inconsistently narrowed values fail.
  Completion/replay ambiguity persists bounded `evidence_binding_mismatch`
  only and leaves no ReviewItem or fake empty marker.
- Required PostgreSQL verification uses a freshly empty disposable database
  at `127.0.0.1:55432`, fails rather than skips without its URL, creates a
  unique schema per test, and drops it in `finally`. Round-three evidence is
  `47` real store/authority/concurrency cases, `147` core Task 3 unit/adapter
  tests, `220` Task 1/2 regressions, `118` service/API/integration tests, and
  `5` standalone migration tests, zero skips. Provider-blocking barriers cover
  success, failure, and timeout while concurrent cancellation still commits.

## 2026-08-29 C.5 Task 6 handoff

- Task 6 is integrated on `codex/rag-orchestrator-agent`. Task 6A's reviewed
  serving/revoke head was `853f476`; the `primary_code` consumer amendment is
  `d2085bc..2691691`; exact Task 6B green head `f356422` was merged by
  `242c071`. Fixture/integration hardening is `445aaf1`, `7e1cc7f`, and
  `59c7549`; final authority/reconciliation hardening continues through
  implementation head `a74cfeb`.
- `CommittedSourceStateChange` booleans are the only reconciliation control
  authority. `primary_code` is validated bounded observability and must never
  be used to infer or suppress a change.
- Supported Google/document ingestion requires exact canonical ids and
  semantic authority: `gmail:`, `gmail_attachment:`, `drive:`, `calendar:`;
  Gmail millisecond `internalDate`, Drive aware `modifiedTime`, and Calendar
  exact start date/date-time. Server signature/parser/current-pointer fields
  are authoritative; connector raw signatures and parser hints are not.
- Retrieval and projection paths must continue using
  `TrustedServingEligibilityService` and exact dependency snapshots. Review
  Queue reads use `ReviewEvidenceVisibilityService`; do not merge these trust
  boundaries.
- Final Task 6 PostgreSQL evidence used one freshly empty pgvector DB with all
  four database URLs pinned to it. Raw gate: `432 passed, 4 failed`, exactly
  the approved Slack nodes. Non-Slack gate: `432 passed, 4 deselected`;
  standalone pgvector, Review V2.1 PostgreSQL, and Review V2 PostgreSQL gates
  are `28 passed`, `48 passed`, and `9 passed`. Ruff, `uv lock --check`, and
  diff checks passed; cleanup returned DB/role `0/0`.
- The canonical Mail/Document resolver already consumes the exact server
  signature/current-version contract. All 12 focused endpoint tests and the
  expanded 328-test Review V2/V2.1 suite are green; do not restore raw connector
  signature authority. Task 7 is the next separate product slice.
- Slack connector/agent/OAuth/data work remains last by explicit user choice.
  Preserve the exact approved ten-node deselection list; Task 6 reaches four
  of those nodes. No additional deselection is authorized.

## 2026-08-30 C.5 Task 7 handoff

- Task 7 is implementation-complete on `codex/rag-orchestrator-agent`.
  `backend/app/agent_runtime/auto_review_eligibility.py` owns deterministic DB
  preflight and bounded validator DTO assembly;
  `backend/app/agent_runtime/auto_review_policy.py` is the pure frozen policy
  authority. API routes/connectors must not bypass these boundaries.
- DB callers supply the current actor-visible permission levels, evidence
  messages, registry, and budget result, but cannot supply workflow scope or
  fingerprint projection readiness. Scope comes from the stored V2.1 workflow;
  PostgreSQL readiness comes from exact runtime/projection/key/count/checksum
  and missing/extra-row state. SQLite returns trusted lookup unavailable.
- Before validation, require exact candidate-key recomputation, immutable
  ReviewItem evidence bindings, message-set HMACs, current canonical
  source/version/permission, strictest permission, exact C.5 generation route,
  and `AgentRunResult` manifest output contract. Missing or drifting authority
  is bounded human/needs-more-evidence and creates no validator request.
- Keep `ValidationPolicyInput.validation_identity` and
  `post_validation_state_matches` explicit. Do not restore approving defaults.
  Reuse of one exact trusted target still requires the same successful Terra
  validation and post-call recheck as a new promotion.
- `credential-scan:v1` uses exact provider patterns plus bounded entropy only
  for exact/delimiter-aware credential assignment labels. Documented fakes are
  exact allowlist entries; do not reintroduce prefix exemptions or substring
  label matching.
- Final evidence: focused `54 passed`; existing contracts `41 passed`;
  adjacent SQLite `230 passed, 58 skipped`; isolated PostgreSQL readiness,
  drift, visible/hidden collision `12 passed`; Ruff/compile/lock/diff PASS;
  disposable DB/role catalog `0/0`; independent Spec PASS / Quality APPROVED,
  no open Critical or Important findings. No connector, embedding, or LLM
  provider was called.
- Next is C.5 Task 8, an actual implementation task: add the isolated real
  LangChain `gpt-5.6-terra` structured validator and model-router boundary with
  fake-model tests, zero fallback/retry/cache/tracing, and a server-owned
  one-dispatch fence. Task 8 must not enable rollout or make a live paid call.
  Slack recovery remains last.

## 2026-08-30 C.5 Task 8 handoff

- Task 8 is implementation-complete on `codex/rag-orchestrator-agent`.
  `backend/app/agent_runtime/auto_review_validator.py` owns the immutable
  two-phase validator protocol; `build_auto_review_validator_model_route()` is
  the only Task 8 Terra construction boundary. Do not call it from routes or
  connectors and do not reintroduce provider-order fallback.
- Preparation serializes local-only claims/evidence once, freezes the exact
  Responses JSON-schema framing and native estimator body, applies the 12,000
  serialized-character and 6,000 framed-token caps, and signs the body with a
  keyed HMAC. Invoke must bind the frozen dict rather than the Pydantic class;
  otherwise schema rendering can drift after a Task 9 claim.
- Invoke requires a dispatcher-held committed grant, rechecks provider logging
  inside the dispatch boundary, permits one provider start, disables LangSmith
  tracing/callbacks/cache, and records only bounded usage/cost after complete
  batch integrity. `token_usage` and `usage`, when present together, must both
  agree exactly with `usage_metadata`; ambiguity is a sanitized whole-batch
  failure.
- Task 8 intentionally does not own a database lease, attempt marker, replay,
  permission/source revalidation, or `FencedOpenAITransport`. Task 9 must issue
  the grant only after marker commit, validate the frozen invocation identity,
  consume the one-use transport, call with no open DB transaction, then reopen
  and revalidate before persisting bounded observations.
- Final evidence: exact Task 8/dependency compatibility `46 passed`; adjacent
  Task 3/7 regression `117 passed`; shared contracts `41 passed`; Ruff,
  compile, `uv lock --check`, and `git diff --check` PASS. Independent rereview
  is Spec PASS / Quality APPROVED with no Critical or Important findings. Tests
  used fake models only; no live paid provider call or rollout enablement
  occurred.
- A usable `OPENAI_API_KEY` is stored only in ignored local `.env.local`; its
  plaintext was never logged or committed. Current Settings load `.env` by
  default, so Task 9 must not assume `.env.local` is implicitly loaded. Do not
  make a paid/live validation call without separate authorization.
- Next is C.5 Task 9, an actual implementation task, not planning: persistent
  validation identities, one-call leases, atomic reservations/cost ledger,
  replay, cancellation/crash handling, and post-provider permission/source
  revalidation. Slack data reconstruction and Slack regressions remain last.

## 2026-08-30 C.5 Task 9 handoff

- Task 9 is implementation-complete on `codex/rag-orchestrator-agent`.
  `auto_review_validation_store.py` owns PostgreSQL-authoritative validation
  leases, committed one-attempt markers, exact Decimal reservations/charges,
  restart replay, cancellation/recovery, and bounded child observations.
- Every claim/attempt/dispatch/completion path acquires the shared key-generation
  barrier and runtime row. Provider safety, projection/rollout, current Source,
  workflow, call, and child state are rechecked in fixed order. Runtime-key,
  price/cost-policy, owner-permission, source-content, cancellation, or rollout
  drift fails closed without promotion; pre-attempt refusals charge zero.
- `ValidationFrameSizer` now shares a pure renderer/tokenizer frame with the
  final immutable invocation without constructing that invocation. Final
  batches are deterministic under candidate permutation and are prepared once.
- Provider dispatch occurs only after the attempt marker commits and after a
  second one-use fence check with no database transaction held during the fake
  or real provider callback. Unknown usage charges the reserve once; known
  overrun records actual usage and atomically opens the validation safety
  breaker with one call-attributed event.
- Validation tables persist only allowlisted decisions, scores, counters,
  tokens, costs, and keyed identities. Regression coverage proves raw evidence
  URLs and provider exception text are absent. This slice stores observations
  only and cannot approve ReviewItems.
- Fresh automated evidence used fake providers only: Task 9 unit/lifespan
  regression `63 passed`; isolated PostgreSQL integration `13 passed`; existing
  provenance regression retained its established SQLite pass/skip boundary.
  No paid provider call or rollout enablement occurred.
- Next is C.5 Task 10, an actual implementation task: locked rollout/canary
  authority, deterministic audit selection, breaker handling, promotion
  coordination, and quality revoke. Slack remains last.

## 2026-08-30 C.5 Task 10 complete

- Task 10 is implementation-complete on `codex/rag-orchestrator-agent`.
  Read-only default rollout state, monotonic authorization latches, stable
  enforce/audit HMAC cohorts, mandatory audit slots, immutable promotion
  decisions, and same-transaction selected audits now gate automatic approval.
- Human audit and quality-coded revoke use a breaker-first durable quarantine.
  Missing audits become manual critical audits, confirmed audits receive an
  immutable correction, failed exact revoke stays remediation-required, and
  lifespan recovery retries only persisted pending/failed remediation.
- Human shadow outcomes update the rollout denominator exactly once only for
  unchanged evidence. Breaker close keeps the authorization latch at zero and
  refuses unresolved remediation or any corrected critical evidence.
- Fresh fake-provider verification: focused Task 10 + transition/validation
  suite `103 passed`; disposable PostgreSQL Task 10 suite `54 passed`; Ruff,
  compile, and diff checks pass. No paid provider call, rollout enablement,
  push, merge, or PR action occurred. Task 11 is the next actual implementation
  slice: signed zero-call launch confirmation and version-selection facade.

## 2026-08-30 C.5 Task 11 complete

- Added the exact-key-set compact HMAC launch token. It binds current source,
  owner permission, provider safety, extraction plan/safety digests, rollout
  control epoch/generation, caps, timings, six-place prices, and combined
  extraction/validation budget; malformed, changed, cross-actor, or expired
  tokens collapse to `cost_preview_changed` before thread creation.
- V2.1 dry-run is read-only and provider-free. It reserves extraction at
  `N * 0.016716` and validation at `ceil(N/4) * 0.048864`; the five-agent bound
  remains `0.181308 <= 0.20` without a predicted-output discount.
- Added a narrow facade that chooses V2.0/V2.1 for new runs from current mode,
  but always routes existing status/resume/cancel by the stored immutable graph
  version. The database launch authority requires current runtime key,
  purpose-specific safety rows, registry prices/caps, source permissions, and
  read-only rollout control.
- Fresh Task 11 + adjacent lifecycle evidence is `124 passed`; Ruff and diff
  checks pass. No provider call, rollout enablement, push, or PR occurred. Task
  12 is the next actual implementation slice: immutable V2.1 LangGraph and
  dedicated lifecycle service.
- Preserve the persisted schema spelling `first_50`. The Task 10 plan prose
  uses `mandatory_50` in a few paragraphs, but the approved design and existing
  database check constraint use `first_50`; changing that output schema requires
  an explicit human decision.

## 2026-09-01 Deliverable D Core Task 13 implementation complete

- Task 13 adds the sole atomic final-product boundary for direct and Assistant
  RAG V2 results. The boundary fresh-locks serving generations and the pending
  parent, verifies the exact-two terminal cost snapshot and projection-owner
  fence, reruns the same retriever without provider dispatch, fresh-resolves
  every evidence dependency, and returns a DTO only after commit succeeds.
- Assistant V2 persistence is finalizer-owned and exact-byte. It locks and
  verifies the string owner/conversation/user-message target, writes the answer
  plus full selected/unselected model-influence dependency set, and never lets
  route or legacy helpers request the server-only `rag_v2_exact` mode.
- Dead projection-owner recovery accepts only the concrete sealed cost-ledger
  authority. It retains paid sidecar/safety when paid work occurred, performs a
  nonblocking exact session-lock reacquisition, rechecks the owner fence and
  runtime-cost snapshot under row locks, and closes `persistence_failed` without
  provider/output retry. The real two-session parent-mutation check is URL-gated.
- SQLite smoke is deterministic and provider-free only. The coordinator itself
  owns fresh keyword retrieval, canonical projection, exact-two terminal-zero
  parent/product and optional Assistant dependencies in one `BEGIN IMMEDIATE`.
  A process-lifetime file lock plus stable RLock rejects DB/sidecar hardlinks,
  cached-handle replacement, case/path aliases, and a second process before
  mutation; in-memory mode is server-settings-gated to tests.
- Independent review follow-up evidence: focused `75 passed, 5 skipped`;
  affected RAG V2/lock/safety/Assistant regression `665 passed, 10 skipped,
  2172 deselected`. All skips require absent `PARAWORKS_TEST_POSTGRES_URL`; the
  real PostgreSQL parent-mutation recovery gate is executable but unrun. No
  external/provider/paid call, Docker action, or `.env` read occurred.
- Before Task 14 begins, run the independent Task 13 implementation review. The
  remaining release gate is the URL-gated real PostgreSQL concurrency suite.
