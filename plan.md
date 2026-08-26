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

Next priorities:

1. Add Knowledge Map only if time allows after core product polish.
2. Continue frontend consistency pass before final portfolio recording.

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
- Deliverable B runtime/checkpoint primitives plan: Ready for review.
  - Execute `docs/superpowers/plans/2026-08-26-langgraph-runtime-checkpoint-primitives.md`
    only after explicit implementation approval.
  - First repair the one stale non-Slack preflight permission fixture, then
    build JSON-safe state/fingerprint contracts, workflow persistence schema,
    checkpointer lifecycle/bootstrap, and graph-version confirmation
    primitives without adding public V2 routes.
- User-directed execution order for the remaining program:
  1. Deliverable B runtime/checkpoint primitives.
  2. Deliverable C Review Queue HITL V2.
  3. Deliverable D and GraphRAG using Gmail, Drive, Calendar, approved
     knowledge, and deterministic fixtures as the primary evidence path.
  4. Slack data recovery and Slack-related regressions last, after choosing
     between deterministic local reconstruction, a newly seeded Slack
     workspace, or an alternate chat connector.
- Do not skip or hide Slack regressions while they are deferred. Keep the ten
  known Slack-related backend failures visible in the full-suite report and
  require every non-Slack gate to remain green.
- Execute the reviewed foundation as four independent green deliverables:
  dependency compatibility, runtime/checkpoint primitives, Review Queue HITL
  V2, and the RAG retriever/graph V2 migration.
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
