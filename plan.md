# ParaWorks Product Plan

## 1. Direction

ParaWorks is a Korean-first, multi-agent company memory product.

The original merged plan remains a reference architecture, but this file is the
current execution plan. The product is being built by three developers, so the
implementation must optimize for clear ownership, stable contracts, frequent
integration, and portfolio-quality evidence of collaboration.

The product goal is not a generic project-management tool. ParaWorks should help
business users ask:

- What happened?
- Why did we decide that?
- Which evidence supports this timeline, history, decision, or todo?
- What changed across Slack, email, documents, and company knowledge?
- Can this answer be trusted under the current user's permissions?

Every AI result must preserve source links, snippets, confidence, permission
level, cost metadata, and human-review status.

## 2. Product Principles

1. Evidence first
   - No source evidence means no Review Queue item.
   - LLM output starts as `pending_review`, never as trusted knowledge.

2. Korean business user first
   - UI copy, review flows, and summaries should be comfortable for Korean
     business users.
   - English can remain for developer/debug labels when it improves clarity.

3. Cost-aware by default
   - Do not send full synced corpora to paid LLMs.
   - Use delta sync, content hashes, dedupe, ranked evidence windows, cache
     reuse, token caps, and explicit paid-run confirmation.
   - Cost savings should be visible in API responses, UI, logs, and portfolio
     notes.

4. Permission-safe by default
   - Source permissions and ParaWorks RBAC both matter.
   - Restricted input cannot produce broader output.
   - Retrieval must report hidden matches without leaking content.

5. Team-scale architecture
   - Each developer owns a product agent/domain, not just a technical layer.
   - Shared contracts live in `backend/app/agent_runtime/`.
   - Agent implementations communicate through contracts, Review Queue, and
     LangGraph, not direct cross-agent imports.

## 3. Three Developer Tracks

### Track A. Communication Intelligence

Primary owner: Developer A

Scope:

- Slack integration and Slack Agent
- Gmail integration handoff with Track B where needed
- Slack/Gmail message and thread evidence extraction
- conversation summarization
- ranked evidence selection
- timeline/history/todo candidate creation from communication sources
- source URL and thread context preservation

Current state:

- Slack OAuth and live sync boundary exist.
- Slack selected-channel sync and incremental cursor logic exist.
- Real Slack LLM adapter exists with OpenAI primary and Gemini fallback.
- Paid LLM runs use ranked, deduped, budget-capped evidence windows.
- Slack AgentRun detail exposes ranked evidence summary.

Next priorities:

1. Expand communication-specific extraction prompts/schemas.
2. Add communication golden dataset cases.

### Track B. Document and Knowledge Pipeline

Primary owner: Developer B

Scope:

- Google Drive integration
- internal document ingestion
- document parsing and chunking
- Gmail document/attachment boundary where relevant
- pgvector indexing
- approved knowledge indexing
- Knowledge Asset pipeline
- source/version metadata quality

Current state:

- Google OAuth and installed sync boundary exist.
- Gmail/Drive source collection path exists in harness form.
- pgvector adapter and incremental vector indexing exist.
- RAG indexing observability and reindex approval UX exist.
- Embedding calls are guarded by delta/hash skip logic.
- Drive events now preserve metadata-only parser status, document version,
  revision id, and content signature for future parser/index decisions.

Next priorities:

1. Harden Google Drive file parsing by type.
2. Add parser run records and parse status.
3. Improve document metadata: page/paragraph-level parser provenance.
4. Prepare HWP/HWPX parser adapter decision.
5. Add document golden dataset cases.

### Track C. Orchestration and Review Product

Primary owner: Developer C

Scope:

- LangGraph orchestration
- Review Queue and human-in-the-loop
- RAG answer agent
- DecisionRecord, Timeline, History, Todo, Validation boundaries
- permission-aware answer generation
- token-cost routing, caching, model policy
- frontend review/search/agent observability

Current state:

- LangGraph company-memory workflow exists.
- Slack, Mail/Docs, and RAG agent runs are orchestrated.
- Review Queue approval promotes reviewed candidates into knowledge tables.
- AgentRun cost summary and detail views exist.
- Ranked Slack evidence is visible in orchestration and AgentRun detail.
- Track C extraction boundaries now exist for Timeline, History, Decision
  Record, Todo, and Validation in deterministic harness mode.
- Review Queue items now expose structured source evidence for reviewer
  inspection, including source URL, snippet, permission, confidence, rank, and
  originating AgentRun where available.
- Company Memory orchestration now emits a Review Queue HITL checkpoint
  strategy with target ReviewItem ids, resume policy, required statuses, and
  trusted-knowledge approval boundary.
- Deliverable C Review Queue HITL V2 is implemented behind its
  disabled-by-default flag. Gmail/Drive/Calendar can launch the exact completed
  source batch from Integrations, pause through real LangGraph `interrupt()`,
  resolve authoritative ReviewItem rows in Review, and explicitly resume the
  same thread with `Command(resume=...)`.
- PostgreSQL restart, reconciliation, exact-batch launch, concurrent Review
  transition, concurrent resume, terminal-race, checkpoint privacy, and exact
  cleanup coverage is release-verified with zero PostgreSQL skips.
- Deliverable C.5 Auto-Review Trust Promotion product Tasks 1–5 are implemented
  and independently verified through final HEAD `4d31aaf`. Task 5 now includes
  a DB-owned typed SQLAlchemy runtime initializer, an application compatibility
  adapter, and a separately owned key-admin CLI runtime with bounded,
  privacy-safe failure outcomes. Product Tasks 6–16 remain unstarted; no paid
  provider call or rollout enablement occurred.
- A later Task 16 release-gate dry run exposed a test-infrastructure boundary,
  not a Task 5 product regression: one shared PostgreSQL `public` schema caused
  144 fresh-empty fixture errors, controller environment overrides caused six
  Settings-contract failures, and six Review V2 PostgreSQL failures require a
  clean targeted RED before their leading fixture-provenance hypothesis can be
  accepted. The approved architecture is documented in
  `docs/superpowers/specs/2026-08-29-whole-suite-postgresql-isolation-design.md`;
  its written spec passed independent Spec/Quality review, is awaiting user
  review, and no isolation implementation has started.

Next priorities:

1. Plan and implement C.5 product Task 6, precise revoke and non-resurrection,
   under its separate approval/review boundary.
2. Continue C.5 product Tasks 7–16 in the approved order; keep rollout disabled
   and live paid-provider benchmarks separately authorized. At Task 16 entry,
   first implement the separately approved whole-suite PostgreSQL isolation
   boundary; planning it now does not reorder Tasks 6–15.
3. After all C.5 product tasks are complete, plan Deliverable D Retriever Port
   and RAG Answer Graph V2, followed by Deliverable E Neo4j GraphRAG.
4. Continue frontend consistency only in its planned C.5 tasks, and keep Slack
   data reconstruction plus its visible regression baseline last.

## 4. Shared Runtime Contracts

Shared contracts are the team integration layer.

Current shared concepts:

- `EvidencePacket`
- `EvidenceMessage`
- `ReviewCandidate`
- `PermissionContext`
- `AgentRunResult`
- `AgentRunCost`
- `AgentManifest`
- `AgentRegistry`

Rules:

- Contract changes require tests before implementation.
- Contract changes must update all affected tracks.
- Agents must preserve source links, snippets, confidence, permission level, and
  uncertainty reason.
- Review Queue is the trust boundary.

Near-term contract improvements:

1. Add structured candidate payload fields for:
   - `timeline_event`
   - `history_event`
   - `decision_record`
   - `todo`
2. Add validation result metadata:
   - `validation_status`
   - `missing_evidence`
   - `faithfulness_status`
   - `permission_status`
3. Add source evidence summaries for UI display.

## 5. Architecture

```text
Next.js app
  -> FastAPI API
  -> connector ingestion
  -> Source / Document / DocumentChunk
  -> pgvector / deterministic smoke retrieval
  -> LangGraph orchestration
  -> source-specific agents
  -> Review Queue
  -> approved knowledge tables
  -> RAG answer/search surfaces
```

Default production direction:

- Backend: FastAPI, SQLAlchemy, Alembic, Pydantic
- DB: PostgreSQL + pgvector
- Queue: Celery + Redis
- Frontend: Next.js App Router, TypeScript, Tailwind
- Agent: LangChain >= 1.2 and current LangGraph
- Human review: Review Queue first, then real LangGraph interrupt/resume with
  PostgreSQL checkpoint persistence before GraphRAG retrieval expansion

SQLite smoke mode must continue working for local demo and tests.

## 6. MVP Definition

MVP is complete when ParaWorks can:

1. Connect or simulate Slack, Gmail, Google Drive, and Google Calendar sources.
2. Store source evidence with permission metadata.
3. Parse/chunk evidence into searchable records.
4. Index approved and source evidence into pgvector when enabled.
5. Run source agents through LangGraph.
6. Create pending Review Queue candidates for:
   - Timeline
   - History
   - Decision Record
   - Todo
7. Show source evidence and confidence to reviewers.
8. Approve/reject/request-more-evidence.
9. Promote approved items into trusted knowledge tables.
10. Answer user questions with permission-aware RAG.
11. Show AgentRun cost, token usage, cache, ranked evidence, and source window.
12. Pass whole-app Playwright smoke checks on desktop and mobile.

## 7. Current Completed Work

Completed harness slices include:

- Next.js frontend shell with Korean/English language option
- Slack-like workspace UI and messenger page
- Liquid Glass frontend theme iteration
- demo login and admin console
- Slack OAuth and runtime status
- Google OAuth and runtime status
- Slack live sync and incremental cursor
- Gmail/Drive installed sync boundary
- Review Queue approval and knowledge promotion
- pgvector adapter and incremental vector indexing
- RAG ask/search path with citation ranking
- `/search` retrieval backend disclosure and pgvector feature-flag path
- LangGraph company-memory foundation
- agent cost budget observability
- OpenAI primary and Gemini fallback for Slack LLM runs
- ranked Slack LLM evidence selection with dedupe and cost caps
- AgentRun ranked evidence detail UI
- Track C deterministic extraction boundaries for Timeline, History, Decision
  Record, Todo, and Validation
- Source Evidence Drawer and reviewer "request more evidence" note workflow
- LangGraph HITL checkpoint strategy surfaced through the Company Memory
  orchestration API
- Focused quality and permission regression suite covering source-less review
  rejection, restricted RAG hiding, HITL checkpoints, and cache dedupe
- Mail/Docs and Track C memory extraction AgentRuns now store evidence summary
  metadata for richer Review Drawer rows
- Slack connector thread replies now preserve parent context in chunk text and
  metadata for better agent/RAG evidence quality
- Gmail connector now preserves thread context keys, participants, participant
  domains, and external-domain flags for better review/RAG filtering
- Drive connector now preserves metadata-only parser status, document version,
  revision id, and content signature for safer parser/index decisions
- Calendar connector now preserves event context keys, status, organizer,
  attendee response counts, duration, and external attendee domains
- Connector golden dataset fixture now locks Slack, Gmail, Drive, and Calendar
  agent-ready metadata expectations
- RAG smoke evaluation fixture now reports precision, recall, hit rate, and
  matched expected source ids for deterministic retrieval quality
- Track C memory extraction now has a LangChain `with_structured_output`
  adapter boundary behind the same deterministic `MemoryExtractionModel`
  contract
- Product memory pages now expose approved Decisions, Timeline, and History
  records from the shared Knowledge API
- Production auth migration plan now defines the httpOnly cookie, refresh token,
  RBAC, audit, and demo-mode migration boundary
- Deployment runbook now defines the production FastAPI, Next.js, Postgres
  pgvector, Redis, Celery, OAuth, verification, cost, and rollback boundaries
- Notifications now surface Review Queue and AgentRun alerts without creating
  new paid model or embedding paths
- Knowledge Map now visualizes approved company memory records and their source
  evidence links through a read-only, zero-paid-call graph endpoint and page
- Production auth implementation has started with persistent auth users,
  hashed refresh-token records, httpOnly session/refresh cookies, refresh
  rotation, logout revocation, and demo-mode header fallback
- Portfolio demo script now covers login, integrations, agent runs, Review
  Queue, approved knowledge, Knowledge Map, permission-aware RAG, cost controls,
  and security talking points
- Azure integration design now defines Container Apps, PostgreSQL pgvector,
  Redis, Key Vault, and a first-stage `azure_openai` provider alias that reuses
  the existing `OPENAI_API_KEY` path for key-swap compatibility
- Google identity login and RBAC now separate ParaWorks login from Google data
  integrations, seed admin/employee/reviewer accounts, gate admin/review pages,
  and enforce role-aware Review Queue approval
- portfolio log and case-study documentation

## 8. Recommended Roadmap From Here

### Milestone 1. Track Alignment Documentation

Goal: Make the 3-track plan the source of truth.

Tasks:

- Keep this `plan.md` updated.
- Keep `AGENTS.md` aligned with this plan.
- Keep `docs/portfolio-log.md` updated after each milestone.
- Record integration rules for coding assistants.

### Milestone 2. Track C Extraction Boundaries

Goal: Make the original Phase 4 agents real, but owned under the orchestration
track.

Status: deterministic harness slice implemented.

Tasks:

- Add deterministic Timeline extraction boundary. Done.
- Add deterministic History extraction boundary. Done.
- Add deterministic Decision Record extraction boundary. Done.
- Add deterministic Todo extraction boundary. Done.
- Add Validation gate before Review Queue persistence. Done.
- Wire them into LangGraph after source-specific agents. Done.
- Next: replace deterministic extractors with structured LangChain outputs
  behind the same contracts when quality tests are ready.

Why next:

- It directly supports the product goal.
- It is portfolio-visible.
- It creates stable integration targets for Tracks A and B.

### Milestone 3. Source Evidence Drawer and Review UX

Goal: Make human review genuinely usable.

Tasks:

- Add Source Evidence Drawer. Done.
- Show source URL, snippet, permission, rank, confidence, and agent run. Done.
- Improve `needs_more_evidence` workflow. Done.
- Add reviewer notes. Done.

### Milestone 4. LangGraph HITL Checkpoint Strategy

Goal: Replace the descriptive checkpoint harness with real LangGraph
interrupt/resume and durable production checkpoint persistence before adding
Neo4j GraphRAG retrieval.

Tasks:

- Emit Review Queue checkpoint metadata from Company Memory orchestration. Done.
- Include target ReviewItem ids, required statuses, resume node, and resume
  policy. Done.
- Expose checkpoint policy through orchestration status APIs. Done.
- Deliverable A dependency compatibility: Done.
  - Locked LangChain 1.3.17, LangGraph 1.2.11, langchain-openai 1.6.0,
    langchain-google-genai 4.3.5, and langgraph-checkpoint-postgres 3.1.2.
  - Added no-network version, provider constructor, structured-output,
    `create_agent`, runtime-context, interrupt/resume, and checkpointer import
    compatibility tests.
  - Complete backend baseline comparison: old lock `506 passed, 11 failed,
    1 skipped`; accepted lock `518 passed, 11 failed, 1 skipped`. The same 11
    failures are pre-existing, so this deliverable introduced zero new backend
    failures.
- Deliverable B runtime/checkpoint primitives: Done.
  - Added JSON-safe, HMAC-keyed runtime contracts, application-owned workflow
    schema, explicit checkpointer bootstrap/readiness, retention, immutable
    graph registry, and synchronous saved-tuple confirmation.
  - Locked the Review workflow graph version at
    `company-memory-review-v2.0` with root `checkpoint_ns == ''`.
  - Checkpoint modes are `disabled`, process-local SQLite/demo `memory`, and
    durable production PostgreSQL `postgres`; production never falls back to
    memory when PostgreSQL is unavailable.
  - Verified real PostgreSQL pause, pool restart, and same-thread resume on an
    isolated disposable database at migration revision `2f6a8b9c0d1e`.
  - Checkpoint confirmation now matches the exact ordered returned/saved
    interrupt ids and JSON-safe values, and resume availability sanitizes
    saver connection/deserialization failures as `checkpoint_unavailable`.
    Cyclic or excessively deep interrupt values are also bounded to the opaque
    `checkpoint interrupt state mismatch` error instead of leaking recursion
    failures.
  - The non-Slack backend gate is green and the same ten deferred Slack
    failures remain visible in the complete backend suite. Latest gates:
    dedicated PostgreSQL `30 passed`, focused Deliverable B `224 passed`,
    non-Slack backend `717 passed, 1 skipped, 10 deselected`, and full backend
    `10 failed, 717 passed, 1 skipped` with only the deferred Slack ids.
  - At the Deliverable B completion checkpoint, Deliverable C was still
    unimplemented and awaiting its separate authorization. That historical
    boundary is superseded by the completed Deliverable C record below.
- Deliverable C Review Queue HITL V2: Implemented and release-verified; rollout
  remains disabled by default.
  - The approved spec is
    `docs/superpowers/specs/2026-08-27-review-queue-hitl-v2-design.md`.
  - The implementation plan is
    `docs/superpowers/plans/2026-08-27-review-queue-hitl-v2.md` and divides work
    into eleven independently reviewable TDD commits from public contracts and
    canonical sync refs through Review transitions, actual LangGraph HITL,
    two-screen frontend UX, and PostgreSQL release evidence.
  - The primary UX remains two screens:
    `Integrations -> 검토 후보 만들기 -> Review -> 검토 완료`; no Agent Runs
    navigation or automatic resume is added.
  - V2 uses an actual LangGraph `interrupt()` and same-thread
    `Command(resume=...)`; current PostgreSQL ReviewItem state remains the
    approval authority.
  - Single, bulk, and agent-candidate Review actions share one locked state
    transition and exactly-once promotion boundary.
  - Exact source-version batches are owned at the company/workspace security
    scope, so authorized users reuse one workflow and cannot create duplicate
    ReviewItem, knowledge, or Timeline effects for the same batch and policy.
  - V2 processes only source versions marked by a post-cutover V2-mode sync;
    legacy-processed historical versions are not regenerated through V2.
  - CDC, transactional outbox, brokers, and streaming projections are deferred
    until measured ingestion backlog, freshness, fan-out, or polling/worker
    bottlenecks justify a separate design.
  - Release verification on a disposable PostgreSQL 16 + pgvector target:
    PostgreSQL recovery/race `14 passed, 0 skipped`; Deliverable C selection
    `333 passed` plus the one visible deferred Slack failure; Deliverable B
    `229 passed`; non-Slack backend `989 passed, 1 skipped, 10 deselected`; full
    backend `989 passed, 1 skipped` plus exactly the ten deferred Slack failures.
    Whole-tree Ruff, lock, and diff gates are green.
  - Frontend release gates are green: lint/build with 18 generated pages,
    desktop V2 `48 passed`, mobile V2 `44 passed`, and adjacent UX `9 passed`.
    The deterministic backend two-screen smoke is `9 passed`.
  - The release privacy correction removes an unused, always-empty
    `review_item_ids` field from checkpoint state only; PostgreSQL ReviewItem
    rows remain authoritative and V2 remains off by default.
  - Final review verified that normal Review approval changes only the live,
    authoritative PostgreSQL status distribution. The immutable paused tuple
    remains valid by exact identity/schema/total count, status stays resumable,
    and explicit completion uses the original checkpoint thread with the
    normal `awaiting -> resuming -> completed` state-version delta of `+2`.
    Missing/extra count keys, wrong total count, permission mismatches, and
    tuple identity corruption still fail closed.
  - PostgreSQL restart evidence now uses fully independent application A/B
    SQLAlchemy engines, pools, and sessionmakers as well as independent
    checkpoint runtimes/pools/savers; A is closed and disposed before B exists.
  - This historical boundary is superseded: the C.5 design/spec, implementation
    plan, and exact execution profile are finalized.
- Deliverable C.5 Auto-Review Trust Promotion: product Tasks 1–5 implemented
  and independently verified; product Task 6 is the next unstarted slice.
  - Approved spec:
    `docs/superpowers/specs/2026-08-28-auto-review-trust-promotion-design.md`.
  - Implementation plan:
    `docs/superpowers/plans/2026-08-28-auto-review-trust-promotion.md`, with
    sixteen ordered RED/GREEN/commit checkpoints covering frozen V2.0/V2.1
    contracts, persistence, evidence bindings, policy/validator, provenance,
    revoke/tombstones, rollout/audit, dual LangGraph lifecycle, same-screen UX,
    and release verification.
  - Actual product-task status:
    1. V2.1 contracts and immutable V2.0 compatibility: complete.
    2. C.5 persistence and provenance ownership: complete.
    3. Immutable V2.1 request data and version-neutral evidence bindings:
       complete.
    4. Server-issued resolution actors with unchanged human behavior: complete.
    5. Exact claim fingerprints, provenance, reaffirmation, and the database
       initialization acceptance boundary: complete and PostgreSQL-verified.
    6. Precise revoke and non-resurrection: not started; next planned slice.
    7. Tasks 7–16, Deliverable D/E, and Slack recovery: not started.
  - Separates canonical source evidence, pending AI knowledge, and trusted
    knowledge. Raw evidence is not official knowledge and C.5 does not broaden
    current RAG indexing.
  - Uses OpenAI `gpt-5.6-terra` at medium reasoning only as an independent
    structured validator. A deterministic versioned policy owns the final
    decision; generation models cannot self-approve.
  - Initial allowlist is public/internal direct-fact Timeline and narrowly
    extractive History. Decision, Todo, restricted, inferred, conflicting, or
    uncertain candidates remain human-reviewed.
  - Keeps the same locked exactly-once Review transition/promotion boundary,
    adds disabled/shadow/enforce modes with default disabled, and requires exact
    provenance-aware revoke without affecting unrelated reaffirmations.
  - Keeps immutable V2.0 threads on V2.0 and defines a separate V2.1 graph for
    new shadow/enforce launches, with signed paid previews and a persistent
    post-audit rollout breaker.
  - Independent plan/dependency/security audits froze version-neutral evidence
    provenance for new V2.0/V2.1 candidates, exact generator identity, current
    workflow-owner permission checks, a no-retry single-call cost ledger,
    indexed hidden-collision fingerprints, and monotonic revoke reconciliation.
  - Vector revoke/reindex uses one shared per-document PostgreSQL transaction
    lock plus tombstone-aware writes and reads. Rollout uses an explicit
    operator latch from 0 to 10 to 100 percent; 2 percent audit begins only on
    newly authorized full-enforce workflows and breaker close never re-enables
    enforce automatically.
  - Frozen gates include zero hard-negative, permission/version, duplicate,
    and cross-item-revoke failures; at least 500 shadow comparisons and at least
    99% precision before operational enforce.
  - The exact five extraction routes are `mail_document_agent`,
    `timeline_agent`, `history_agent`, `decision_record_agent`, and `todo_agent`;
    each returns zero or one candidate using OpenAI
    `gpt-5.4-mini-2026-03-17`, reasoning `none`, 10,000 input/2,048 total output
    tokens, and USD 0.75/M input plus USD 4.50/M output.
  - Validation uses OpenAI `gpt-5.6-terra`, reasoning `medium`, 6,000 input/
    3,072 total output tokens, four candidates per batch, and at most two
    batches/five candidates per workflow at USD 2/M input plus USD 12/M output.
    Worst-case extraction USD 0.083580 plus validation USD 0.097728 equals
    USD 0.181308, leaving USD 0.018692 below the immutable USD 0.20 limit.
  - Persisted Assistant evidence dependencies fail closed on incomplete, stale,
    revoked, quarantined, permission-drifted, or otherwise mismatched bindings.
    C.5 authority comes only from the server-owned content signature, exact
    parser policy/run, and relational `current_document_version_id`; connector
    signatures and display labels are not authority. Provider and rollout
    controls use append-only events with aggregate backpointers. Quality revoke
    uses immutable assessment/audit-or-correction, then breaker/quarantine, then
    exact revoke; a corrected confirmed audit is permanent and requires a new
    reviewed policy version for recovery.
  - Automated tests remain fake/deterministic. A separately authorized,
    sanitized paid Terra benchmark is required before shadow rollout, otherwise
    mode stays disabled. Populated C.5 data is not destroyed by downgrade/reset;
    config disablement is the operational rollback.
  - Tasks 1–5 changed product code and schema under their approved TDD plan.
    Task 5's final database-boundary commits are `cedd546`, `cc5faa5`,
    `4177f80`, `149ea23`, `4018ddd`, and `4d31aaf`; their final PostgreSQL
    acceptance is `551 passed, 0 skipped`, with Ruff/compile/lock/diff PASS and
    controller-owned cleanup `0:0:0:0`. Product Task 6 and later work remain
    separately gated.
- User-directed execution order for the remaining program:
  1. Deliverable C.5 Auto-Review Trust Promotion Tasks 6–16.
  2. Deliverable D Retriever Port and RAG Answer Graph V2 using Gmail, Drive,
     Calendar, trusted knowledge, and deterministic fixtures.
  3. Deliverable E Neo4j GraphRAG after D establishes the safe retriever and
     answer contracts.
  4. Slack data recovery and Slack-related regressions last, after choosing
     between deterministic local reconstruction, a newly seeded Slack
     workspace, or an alternate chat connector.
- Do not skip or hide Slack regressions while they are deferred. Keep the ten
  known Slack-related backend failures visible in the full-suite report and
  require every non-Slack gate to remain green.
- Preserve independent green checkpoints for dependency compatibility,
  runtime/checkpoint primitives, Review Queue HITL V2, Auto-Review Trust
  Promotion, Retriever/RAG Answer Graph V2, and Neo4j GraphRAG.
- Keep legacy orchestration routes stable while disabled-by-default V2 routes
  prove actual `interrupt()` / `Command(resume=...)` and PostgreSQL
  checkpointing.
- Enforce thread-bound DB idempotency and ReviewItem state transitions before
  enabling durable HITL.
- Keep LangGraph `checkpoint_ns` at the root namespace; select immutable graph
  versions through the application thread registry and require synchronous
  saver confirmation before reporting a durable pause.
- Add canonical workflow evidence references, short claim/lease transactions,
  and per-knowledge-table `source_review_item_id` uniqueness so retries and
  concurrent single/bulk approvals cannot duplicate promotion or generated
  Timeline rows.
- Put keyword and pgvector behind a common retriever port before adding the
  Neo4j retriever branch, while keeping the RAG graph independent of
  checkpointer availability.
- Require structured claim-to-evidence slot mapping and faithfulness parity in
  shadow mode before cutting RAG V2 over to `/ask`, `/search`, or assistant.

### Milestone 5. Connector Quality Hardening

Goal: Improve real data quality before adding more shiny features.

Tasks:

- Slack thread context-aware chunking. Done.
- Gmail thread metadata and domain filtering. Done.
- Drive file parser status and version metadata. Done.
- Calendar ingestion boundary. Done.

### Milestone 6. Evaluation and Regression Suite

Goal: Make quality measurable.

Tasks:

- Focused quality and permission regression suite. Done.
- Permission leakage tests. Initial suite done.
- Source validation tests. Initial suite done.
- Cost/cache regression tests. Initial suite done.
- Golden dataset for Slack, Gmail, Drive, Calendar. Initial fixture done.
- RAG precision/recall smoke metrics. Initial fixture done.
- Structured LangChain output adapters behind deterministic contracts. Initial
  Track C adapter done.

### Milestone 7. Product Completion Layer

Goal: Move from harness to product-like MVP.

Tasks:

- Decisions page. Done.
- Timeline and History views. Done.
- Notifications. Done for Review Queue and AgentRun visibility.
- Knowledge Map. Done.
- Production auth plan: httpOnly cookie + refresh token. Done.
- Deployment runbook. Done.

### Milestone 8. Final Product Hardening

Goal: Move the harness toward a portfolio-ready, product-like demo without
weakening the 3-track ownership model.

Recommended order:

1. Frontend global consistency and final Liquid Glass polish across all pages.
2. Production auth implementation based on the documented httpOnly cookie and
   refresh-token plan. Initial cookie/session slice done.
3. End-to-end demo script covering Slack/Gmail/Drive evidence, Review Queue,
   approved knowledge, Knowledge Map, and permission-aware RAG. Done.
4. Final whole-app Playwright, backend suite, frontend build, and portfolio
   case-study evidence capture. Current regression pass done; final screenshot
   capture remains.
5. Azure staging preparation:
   - Azure design spec. Done.
   - `azure_openai` OpenAI-compatible provider alias. Done.
   - IaC/resource creation. Not started; requires budget, region, resource
     group, and staging domain confirmation.
6. Production identity and RBAC:
   - Google identity login boundary. Done.
   - Google identity readiness diagnostics. Done.
   - Seeded admin/employee/reviewer users. Done.
   - Admin user management API and UI. Initial slice done.
   - Review Queue role checks. Initial slice done.
   - AgentRun/RAG cost observability API admin guards. Done.
   - Future: CSRF, rate limiting, Alembic migration, and multi-step approval
     states.

## 9. Cost Policy

Every new agent or connector must answer:

- Does it avoid duplicate sync?
- Does it avoid duplicate embeddings?
- Does it bound paid LLM input?
- Does it reuse cache when evidence is unchanged?
- Does it expose estimated and actual token usage?
- Does it fail closed when over budget?

Default cost rules:

- Status APIs must not trigger paid LLM calls.
- Live paid LLM calls require explicit user action.
- Preflight and actual run must use the same evidence packet.
- Ranked/deduped evidence windows are preferred over recent-only windows.
- Full source sync is allowed; full source LLM input is not.

## 10. Quality Gates

Before a milestone is considered done:

- Backend tests pass.
- Frontend lint/build pass when frontend changed.
- Playwright checks run for affected pages.
- API smoke check runs for affected backend flows.
- `docs/portfolio-log.md` is updated.
- Changes are committed.

For live provider calls:

- Never print secrets.
- Run preflight first.
- Keep paid runs minimal and intentional.
- Record estimated vs actual cost when available.

## 11. How Coding Assistants Should Work

Coding assistants must:

1. Read this `plan.md` before major work.
2. Read `AGENTS.md`.
3. Check `git status --short`.
4. Identify which track owns the change.
5. Avoid cross-track rewrites unless the task is explicitly integration work.
6. Use tests before behavior changes.
7. Preserve cost and permission guardrails.
8. Update portfolio or handoff docs after meaningful milestones.

If the old merged plan and this file conflict, follow this file unless the user
explicitly says to return to the old plan.
